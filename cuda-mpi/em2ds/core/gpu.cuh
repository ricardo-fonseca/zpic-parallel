#pragma once

/**
 * @file gpu.hpp
 * @brief CUDA support routines: error handling, device query, and
 *        warp / block / device level primitives.
 * 
 */

#include <cuda_runtime.h>

#include <iostream>
#include <string>
#include <string_view>
#include <source_location>
#include <cstdlib>

#include "../parallel/mpi.hpp"

#include "device_types.h"

/**
 * @brief If not using nvcc define dummy kernel variables to allow kernel code linting
 * 
 */
#ifndef __NVCC__
inline const dim3 blockIdx{0};
inline const dim3 blockDim{0};
inline const dim3 threadIdx{0};
inline const dim3 gridDim{0};
#endif

namespace gpu {

/**
 * @brief CUDA Number of threads per warp
 * 
 */
constexpr int warp_size = 32;

/**
 * @brief CUDA Maximum number of warps per block
 * 
 */
constexpr int max_warps = 32;

/**
 * @brief Full participation mask for warp shuffle intrinsics
 */
constexpr unsigned int full_mask = 0xffffffffu;

static_assert( warp_size == 32,
    "The warp level routines use 32-bit shuffle masks and assume a 32 lane warp. "
    "Porting to a 64 lane wavefront requires 64-bit masks throughout." );

static_assert( max_warps <= warp_size,
    "The block level scan requires max_warps <= warp_size so that a single "
    "warp can scan all per-warp partial sums." );

/**
 * @brief Checks if the operation was successuful, otherwise aborts code
 *
 * @note This will reset the CUDA device
 * 
 * @param err       Error code
 * @param msg       Error message to print in case of error
 * @param location  call location (set automatically)
 */
inline void check_err(
    const cudaError_t err, const std::string& msg,
    const std::source_location location =
          std::source_location::current() )
{
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) " << msg << '\n'
                  << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << '\n'
                  << "(*error*) error state in " 
                  << location.file_name() << ':' << location.line()
                  << " " << location.function_name() << '\n'
                  << "(*error*) aborting..." << std::endl;;
        cudaDeviceReset();
        mpi::abort(1);
    } 
}

/**
 * @brief Aborts code
 *
 * @note This will reset the CUDA device
 * 
 * @param msg       Error message to print
 * @param location  call location (set automatically)
 */
[[noreturn]] inline void abort(
    const std::string& msg,
    const std::source_location location =
          std::source_location::current() )
{
    std::cerr << "(*error*) " << msg << '\n'
                << "(*error*) abort issued in " 
                << location.file_name() << ':' << location.line()
                << " " << location.function_name() << '\n'
                << "(*error*) aborting..." << std::endl;;
    cudaDeviceReset();
    mpi::abort(1);

    // This will silence the return warning
    std::exit(1);
}

/**
 * @brief Set active device for a parallel run
 *
 * @note Devices are assigned according to MPI rank on supplied communicator
 * 
 * @param comm 
 */
inline void parallel_init( MPI_Comm comm ) {
    MPI_Comm local_comm;
    int global_rank, local_rank;

    // Create a communicator with processes sharing the same hardware node
    MPI_Comm_rank( comm, &global_rank );
    MPI_Comm_split_type( comm, MPI_COMM_TYPE_SHARED, global_rank,  MPI_INFO_NULL, &local_comm );
    
    // Get rank in local communicator
    MPI_Comm_rank(local_comm, &local_rank);

    // Free the communicator
    MPI_Comm_free(&local_comm);

    // Get number of GPU devices on node
    int num_devices; 
    gpu::check_err( cudaGetDeviceCount(&num_devices), "unable to get number of available GPU devices" );

    // Use local_rank to select GPU device on node usina a round-robin algorithm
    int device = local_rank % num_devices;
    gpu::check_err( cudaSetDevice( device ), "unable to select GPU device" );
}


/**
 * @brief Prints GPU device info
 * 
 */
inline void print_info( bool verbose = false ) {

    int device, nDevices;
    cudaDeviceProp prop;

    gpu::check_err( cudaGetDevice( & device ), 
               "unable to get current device" );
    gpu::check_err( cudaGetDeviceCount( & nDevices ),
               "unable to get number of devices" );
    gpu::check_err( cudaGetDeviceProperties(&prop, device),
               "unable to get device properties" );

    std::cout << "Device Number           : " << device << " (of " << nDevices << ")\n";
    std::cout << "  Device name           : " << prop.name << '\n';

    if ( verbose ) {
        std::cout << "  Memory Bus Width      : " << prop.memoryBusWidth << " (bits) \n";
        std::cout << "  Multiprocessors       : " << prop.multiProcessorCount << '\n';
        std::cout << "  Global memory         : " << prop.totalGlobalMem / (1024*1024) << " (MB) \n";
    }

    std::cout << "  Max. block size       : " << prop.maxThreadsPerBlock << "\n";
    std::cout << "  Warp size             : " << prop.warpSize << "\n";
    std::cout << "  Max. warps/block      : " << prop.maxThreadsPerBlock / prop.warpSize << "\n";
    std::cout << "  Shared memory (def.)  : " << prop.sharedMemPerBlock / 1024 << " (kB) \n";
    std::cout << "  Shared memory (optin) : " << prop.sharedMemPerBlockOptin / 1024 << " (kB) \n";

    if ( prop.warpSize != gpu::warp_size ) {
        std::cerr << "(*warning*) device warp size (" << prop.warpSize
                  << ") differs from compile time gpu::warp_size ("
                  << gpu::warp_size << ")\n";
    }
}

/**
 * @brief Warp level routines
 * 
 * @warning These assume that the block is 1D, i.e., that only threadIdx.x is used
 */
namespace warp {

    /**
     * @brief Number of threads in warp
     * 
     * @return int 
     */
    __device__ __forceinline__ 
    constexpr int num_threads() {
        return gpu::warp_size;
    }
    
    /**
     * @brief Thread rank inside warp (lane id)
     */
    __device__ __forceinline__ int thread_rank() {
        return threadIdx.x & ( gpu::warp_size - 1 );
    }

    /**
     * @brief Warp ID inside block
     */
    __device__ __forceinline__ int group_rank() {
        return threadIdx.x / gpu::warp_size;
    }

    /**
     * @brief Warp level reduce add
     *
     * @warning All threads in the warp must participate
     */
    template<typename T>
    __device__ __inline__ T reduce_add( T const input ) {
        T value = input;
        #pragma unroll
        for( int i = 1; i < gpu::warp_size; i <<= 1 )
            value += __shfl_xor_sync( gpu::full_mask, value, i );
        return value;
    }
    
    /**
     * @brief Warp level reduce max
     *
     * @warning All threads in the warp must participate
     */
    template<typename T>
    __device__ __inline__ T reduce_max( T const input ) {
        T value = input;
        #pragma unroll
        for( int i = 1; i < gpu::warp_size; i <<= 1 ) {
            T tmp = __shfl_xor_sync( gpu::full_mask, value, i );
            if ( tmp > value ) value = tmp;
        }
        return value;
    }

    /**
     * @brief Warp level reduce min
     *
     * @warning All threads in the warp must participate
     */
    template<typename T>
    __device__ __inline__ T reduce_min( T const input ) {
        T value = input;
        #pragma unroll
        for( int i = 1; i < gpu::warp_size; i <<= 1 ) {
            T tmp = __shfl_xor_sync( gpu::full_mask, value, i );
            if ( tmp < value ) value = tmp;
        }
        return value;
    }
    
    /**
     * @brief Warp level exclusive scan (add)
     *
     * @warning All threads in the warp must participate
     */
    template<class T>
    __device__ __inline__ T exscan_add( T const input ) {
        T value = input;
        const int laneId = thread_rank();
        #pragma unroll
        for( int i = 1; i < gpu::warp_size; i <<= 1 ) {
            T tmp = __shfl_up_sync( gpu::full_mask, value, i );
            if ( laneId >= i ) value += tmp;
        }

        value = __shfl_up_sync( gpu::full_mask, value, 1 );
        return ( laneId > 0 ) ? value : T{0};
    }
    
    /**
     * @brief Warp level inclusive scan (add)
     *
     * @warning All threads in the warp must participate
     */
    template<class T>
    __device__ __inline__ T inscan_add( T const input ) {
        T value = input;
        const int laneId = thread_rank();
        #pragma unroll
        for( int i = 1; i < gpu::warp_size; i <<= 1 ) {
            T tmp = __shfl_up_sync( gpu::full_mask, value, i );
            if ( laneId >= i ) value += tmp;
        }
        return value;
    }
    
    /**
     * @brief Warp level reverse inclusive scan (add)
     *
     * @note Same as an inclusive scan but in the opposite order (right to left)
     *
     * @warning All threads in the warp must participate
     */
    template<class T>
    __device__ __inline__ T rev_inscan_add( T const input ) {
        T value = input;
        const int laneId = thread_rank();
        #pragma unroll
        for( int i = 1; i < gpu::warp_size; i <<= 1 ) {
            T tmp = __shfl_down_sync( gpu::full_mask, value, i );
            if ( laneId < gpu::warp_size - i ) value += tmp;
        }
        return value;
    }

} // namespace warp

/**
 * @brief Block level routines
 * 
 */
namespace block {

    /**
     * @brief Thread Id inside block
     *
     * Same as block.thread_rank() in CUDA cooperative groups.
     */
    __device__ __forceinline__
    int thread_rank() { return threadIdx.x; }

    /**
     * @brief Number of threads inside block
     *
     * Same as block.num_threads() in CUDA cooperative groups.
     */
    __device__ __forceinline__
    int num_threads() { return blockDim.x; }

    /**
     * @brief Number of warps inside block (rounded up)
     */
    __device__ __forceinline__
    int num_warps() {
        return ( blockDim.x + gpu::warp_size - 1 ) / gpu::warp_size;
    }

    /**
     * @brief Synchronize threads inside block
     *
     * Same as block.sync() in CUDA cooperative groups.
     */
    __device__ __forceinline__
    void sync() { __syncthreads(); }

    /**
     * @brief Returns pointer to (block) shared memory region
     * 
     * @tparam T    Datatype (defaults to char)
     * @return T *  Datatype pointer to shared memory
     */
    template < typename T = char >
    __device__ __inline__
    T * shared_mem() { 
        extern __shared__ char block_shm[];
        return reinterpret_cast<T *>(block_shm);
    }
    
    /**
     * @brief Returns max. shared memory per block (opt. in)
     *
     * @param id            Device id, defaults to active device
     * @return size_t       Maximum shared memory per block (opt. in) in bytes
     */
    __host__ inline
    size_t shared_mem_size( int id = -1 ) {
        if ( id < 0 )
            gpu::check_err( cudaGetDevice( & id ),
               "unable to get current device" );
        
        cudaDeviceProp prop;
        gpu::check_err( cudaGetDeviceProperties( &prop, id ),
                        "unable to query device properties" );
        return prop.sharedMemPerBlockOptin;
    }

    /**
     * @brief Default (non opt-in) shared memory limit per block
     *
     * @param id            Device id, defaults to 0
     * @return size_t       Default shared memory per block in bytes
     */
    __host__ inline
    size_t default_shared_mem_size( int id = -1 ) {
        if ( id < 0 )
            gpu::check_err( cudaGetDevice( & id ),
               "unable to get current device" );

        int v = 0;
        gpu::check_err(
            cudaDeviceGetAttribute( &v, cudaDevAttrMaxSharedMemoryPerBlock, id ),
            "unable to query default shared memory limit" );
        return static_cast<size_t>( v );
    }

    /**
     * @brief Set the shared memory size for the specified kernel
     * @note If requested memory is below 48kb (the default size) this call is silently ignored
     * 
     * @tparam T            Kernel type (const void)
     * @param entry         Kernel function
     * @param shm_size      Requested size in bytes
     * @return cudaError_t  Error code for operation
     */
    template<class T>
    __host__ inline
    cudaError_t set_shmem_size( T *entry, size_t shm_size ) {
        if ( shm_size > 49152 ) {
            return cudaFuncSetAttribute( 
                entry, cudaFuncAttributeMaxDynamicSharedMemorySize, 
                static_cast<int>( shm_size ));
        }
        return cudaSuccess;
    }
    
    /**
     * @brief Atomic fetch-add operation (block level)
     */
    template< typename T >
    __device__ __forceinline__
    auto atomic_fetch_add( T * address, T val ) {
        return atomicAdd_block( address, val );
    }

    /**
     * @brief Atomic fetch-max operation (block level)
     */
    template< typename T >
    __device__ __forceinline__
    auto atomic_fetch_max( T * address, T val ) {
        return atomicMax_block( address, val );
    }
    
    /**
     * @brief Atomic fetch-min operation (block level)
     */
    template< typename T >
    __device__ __forceinline__
    auto atomic_fetch_min( T * address, T val ) {
        return atomicMin_block( address, val );
    }

    /**
     * @brief Atomic fetch-max for float (block level)
     *
     * @note CUDA has no floating point atomicMax. This uses the standard
     *       ordering trick on the IEEE-754 bit pattern: for non-negative
     *       values the signed integer order matches the float order, and for
     *       negative values the unsigned order is reversed.
     *
     * @warning Undefined for NaN operands.
     */
    __device__ __forceinline__
    float atomic_fetch_max( float * address, float val ) {
        return ( val >= 0.0f )
            ? __int_as_float(  atomicMax_block( reinterpret_cast<int *>(address),
                                                __float_as_int(val) ) )
            : __uint_as_float( atomicMin_block( reinterpret_cast<unsigned int *>(address),
                                                __float_as_uint(val) ) );
    }

    /**
     * @brief Atomic fetch-min for float (block level)
     *
     * @warning Undefined for NaN operands.
     */
    __device__ __forceinline__
    float atomic_fetch_min( float * address, float val ) {
        return ( val >= 0.0f )
            ? __int_as_float(  atomicMin_block( reinterpret_cast<int *>(address),
                                                __float_as_int(val) ) )
            : __uint_as_float( atomicMax_block( reinterpret_cast<unsigned int *>(address),
                                                __float_as_uint(val) ) );
    }
    
    /**
     * @brief Block level exclusive scan (add)
     *
     * @warning Must be called by all threads in the block. The block must
     *          contain a whole number of warps.
     *
     * @note The routine synchronizes on entry and on exit, so consecutive
     *       calls are safe.
     *
     * @tparam T        Type
     * @param input     Input value
     * @return T        Exclusive prefix sum for this thread
     */
    template<typename T>
    __device__ __inline__ T exscan_add( T const input ) {

        __shared__ T tmp[ gpu::max_warps ];

        const int nwarps = num_warps();

        // Protects tmp[] against a previous call still reading it
        sync();

        T v = warp::exscan_add( input );

        if ( warp::thread_rank() == gpu::warp_size - 1 )
            tmp[ warp::group_rank() ] = v + input;
        sync();

        // Only warp 0 does this. max_warps <= warp_size, so one pass suffices.
        if ( warp::group_rank() == 0 ) {
            const int lane = warp::thread_rank();
            // Lanes beyond the number of populated warps contribute zero
            T t = ( lane < nwarps ) ? tmp[ lane ] : T{0};
            t = warp::exscan_add( t );
            if ( lane < nwarps ) tmp[ lane ] = t;
        }
        sync();

        // Add in contribution from previous warps
        v += tmp[ warp::group_rank() ];
        sync();

        return v;
    }
    
    /**
     * @brief Block level memcpy of float3 values
     * 
     * @warning Must be called by all threads in block
     * 
     * @param dst   Destination address
     * @param src   Source address
     * @param n     Number of elements to copy
     */
    __device__ __inline__
    void memcpy( float3 * __restrict__ dst, float3 const * __restrict__ src, const size_t n ) {
        float *       __restrict__ _dst = reinterpret_cast<float *>( dst );
        float const * __restrict__ _src = reinterpret_cast<float const *>( src );

        for( size_t i = thread_rank(); i < 3*n; i += num_threads() )
            _dst[i] = _src[i];
    }
    
    /**
     * @brief Simultaneous block level memcpy of 2 buffers of float3 values
     * 
     * @note  The 2 buffers must have the same size and don't overlap
     * 
     * @param dst1  Destination address 1
     * @param src1  Source address 1
     * @param dst2  Destination address 2
     * @param src2  Source address 2
     * @param n     Number of elments to copy
     */
    __device__ __inline__
    void memcpy2( float3 * __restrict__ dst1, float3 const * __restrict__ src1,
                  float3 * __restrict__ dst2, float3 const * __restrict__ src2,
                  const size_t n ) {
        float *       __restrict__ _dst1 = reinterpret_cast<float *>( dst1 );
        float const * __restrict__ _src1 = reinterpret_cast<float const *>( src1 );
        float *       __restrict__ _dst2 = reinterpret_cast<float *>( dst2 );
        float const * __restrict__ _src2 = reinterpret_cast<float const *>( src2 );

        for( size_t i = thread_rank(); i < 3*n; i += num_threads() ) {
            _dst1[i] = _src1[i];
            _dst2[i] = _src2[i];
        }
    }
} // namespace block

namespace grid {
    /**
    * @brief Thread Id inside grid
    * 
    * same as grid.thread_rank() in CUDA cooperative groups, assuming that only 
    * threadIdx.x and blockIdx.x are used.
    * 
    */
    __device__ __forceinline__
    int thread_rank() {
        return threadIdx.x + blockIdx.x * blockDim.x;
    }

    /**
     * @brief Total number of threads in grid
     */
    __device__ __forceinline__
    int num_threads() {
        return blockDim.x * gridDim.x;
    }
}

/**
 * @brief Device routines
 * 
 */
namespace device {
    
    /**
     * @brief Class representing a scalar variable in device memory
     * 
     * @note This class simplifies the creation of scalar variables in unified
     *       memory. Note that getting the variable in the host (`get()`) will
     *       always cause a memcpy from device to host.
     * 
     * @tparam T    variable datatype
     */
    template< typename T> class var {
        private:
    
        T * data = nullptr;
    
        public:
    
        /**
         * @brief Construct a new var<T> object
         * 
         */
        var() {
            gpu::check_err( cudaMalloc( &data, sizeof(T) ), 
                            "Unable to allocate memory for device::var" );
        }
    
        /**
         * @brief Construct a new var<T> object and set value to val
         * 
         * @param val 
         */
        explicit var( const T val ) : var() { set( val ); }
    
        /**
         * @brief Destroy the var<T> object
         * 
         */
        ~var() {
            if ( data ) {
                gpu::check_err( cudaFree( data ), 
                "Unable to free memory for device::var" );
            }
        }

        /**
         * @brief Delete copy constructor
         * 
         */
        var( var const & ) = delete;

        /**
         * @brief Delete copy assignement
         * 
         * @return var& 
         */
        var & operator=( var const & ) = delete;
        
        /**
         * @brief Move constructor
         * 
         * @param other 
         */
        var( var && other ) noexcept : data( other.data ) { 
            other.data = nullptr; 
        }

        /**
         * @brief Move assigment
         * 
         * @param other 
         * @return var& 
         */
        var & operator=( var && other ) noexcept {
            if ( this != &other ) {
                if ( data ) cudaFree( data );
                data = other.data;
                other.data = nullptr;
            }
            return *this;
        }

        /**
         * @brief Pointer to variable data
         * 
         * @return T* 
         */
        inline T * ptr() const { return data; }

        /**
         * @brief Sets the value of the var<T> object
         * 
         * @note Data is copied to device using cudaMemcpyAsync()
         * 
         * @param val       value to set
         * @return T const  returns same value
         */
        inline T const set( const T val ) {
            auto err = cudaMemcpyAsync( data, &val, sizeof(T), cudaMemcpyHostToDevice );
            gpu::check_err( err, "Failed to copy value to device on device::var.set()" );

            return val;
        }
    
    
        /**
         * @brief Returns value of variable
         * 
         * @warning This will always perform a cudaMemcpy()
         * 
         * @return T const 
         */
        inline T const get() const { 
            
            T val;

            auto err = cudaMemcpy( &val, data, sizeof(T), cudaMemcpyDeviceToHost );
            gpu::check_err( err, "Failed to copy data from device on device::var.get()" );

            return val;
        }

        /**
         * @brief Stream << operator - outputs value of variable.
         * 
         * @warning Device operations will be synchcronized first.
         * 
         * @tparam U 
         * @param os 
         * @param d 
         * @return std::ostream& 
         */
        friend std::ostream& operator<< (std::ostream& os, device::var<T> const & d) { 
            return os << d.get();
        }
    };

    /**
     * @brief Resets the GPU device, stopping any active kernels and clearing error states.
     */
    inline cudaError_t reset() {
        return cudaDeviceReset();
    }


    /**
     * @brief Wait for compute device to finish.
     * 
     * @return      If the GPU is in an error state, the function will return an error
     */
    __host__ inline
    cudaError_t sync() {
        return cudaDeviceSynchronize();
    }

    /**
     * @brief Checks for synchronous (launch) errors only.
     *
     * @note Cheap: does not synchronize. Safe to call after every kernel
     *       launch in production builds.
     */
    inline void check_launch(
        const std::source_location location =
            std::source_location::current() )
    {
        auto err = cudaPeekAtLastError();
        if ( err != cudaSuccess ) {
            std::cerr << "(*error*) CUDA kernel launch failed at "
                      << location.file_name() << ':' << location.line()
                      << " " << location.function_name() << '\n'
                      << "(*error*) " << cudaGetErrorString(err)
                      << " (" << static_cast<int>(err) << ")\n";
            cudaDeviceReset();
            mpi::abort( 1 );
        }
    }

    /**
    * @brief Checks if there are any synchronous or asynchronous errors from CUDA calls
    * 
    * @note If any errors are found the routine will print out the error messages and exit
    *       the program
    */
    inline void check(
        const std::source_location location =
            std::source_location::current() )
    {
        auto err_sync = cudaPeekAtLastError();
        auto err_async = cudaDeviceSynchronize();
        if (( err_sync != cudaSuccess ) || ( err_async != cudaSuccess )) {
            std::cerr << "(*error*) CUDA device is on error state at " 
                    << location.file_name() << ':' << location.line()
                    << " " << location.function_name() << '\n';
            if ( err_sync != cudaSuccess )
                std::cerr << "(*error*) Sync. error message: " 
                        << cudaGetErrorString(err_sync) 
                        << " (" << err_sync << ") \n";
            if ( err_async != cudaSuccess )
                std::cerr << "(*error*) Async. error message: " 
                        << cudaGetErrorString(err_async) 
                        << " (" << err_async << ") \n";
            cudaDeviceReset();
            mpi::abort(1);
        }
    }


    /**
     * @brief Atomic fetch-add operation. Returns the value before the operation.
     */
    template< typename T >
    __device__ __forceinline__
    auto atomic_fetch_add( T * address, T val ) {
        return atomicAdd( address, val );
    }

    /**
     * @brief Atomic fetch-max operation. Returns the value before the operation.
     */
    template< typename T >
    __device__ __forceinline__
    auto atomic_fetch_max( T * address, T val ) {
        return atomicMax( address, val );
    }

    /**
     * @brief Atomic fetch-min operation. Returns the value before the operation.
     */
    template< typename T >
    __device__ __forceinline__
    auto atomic_fetch_min( T * address, T val ) {
        return atomicMin( address, val );
    }

    /**
     * @brief Atomic fetch-max for float
     *
     * @note CUDA has no floating point atomicMax; see block::atomic_fetch_max.
     * @warning Undefined for NaN operands.
     */
    __device__ __forceinline__
    float atomic_fetch_max( float * address, float val ) {
        return ( val >= 0.0f )
            ? __int_as_float(  atomicMax( reinterpret_cast<int *>(address),
                                          __float_as_int(val) ) )
            : __uint_as_float( atomicMin( reinterpret_cast<unsigned int *>(address),
                                          __float_as_uint(val) ) );
    }

    /**
     * @brief Atomic fetch-min for float
     *
     * @warning Undefined for NaN operands.
     */
    __device__ __forceinline__
    float atomic_fetch_min( float * address, float val ) {
        return ( val >= 0.0f )
            ? __int_as_float(  atomicMin( reinterpret_cast<int *>(address),
                                          __float_as_int(val) ) )
            : __uint_as_float( atomicMax( reinterpret_cast<unsigned int *>(address),
                                          __float_as_uint(val) ) );
    }
    
    namespace detail {
    
    /**
     * @brief Single block exclusive scan (add) kernel
     *
     * Every thread in the block executes every loop iteration, with
     * out-of-range elements contributing zero. This keeps __syncthreads() and
     * the full-mask warp shuffles convergent, which the original
     * `for( i = tid; i < size; i += nthreads )` form did not: threads with
     * i >= size dropped out of the loop while the rest were still
     * synchronizing.
     *
     * Because the inactive lanes contribute zero, the running total carried in
     * `prev` is still correct in the final partial chunk.
     *
     * @note `out` and `in` may alias (in place operation), so neither is
     *       marked __restrict__.
     *
     * @tparam T            Template datatype
     * @param out           Output data buffer
     * @param in            Input data buffer
     * @param size          Data buffer size (number of elements)
     * @param reduction     Output reduction (optional). Set to a non-null
     *                      pointer to store the global sum on this address.
     */
    template < typename T >
    __global__
    void exscan_add_kernel( T * out, T const * in, size_t const size, T * reduction )
    {
        static_assert( gpu::max_warps <= gpu::warp_size,
            "This implementation requires gpu::max_warps to be <= gpu::warp_size" );

        __shared__ T tmp[ gpu::max_warps ];
        __shared__ T prev;

        const int nthreads = block::num_threads();
        const int nwarps   = block::num_warps();
        const int lane     = warp::thread_rank();
        const int wid      = warp::group_rank();

        if ( block::thread_rank() == 0 ) prev = T{0};
        block::sync();

        for( size_t base = 0; base < size; base += nthreads ) {

            const size_t i = base + block::thread_rank();
            const T s = ( i < size ) ? in[i] : T{0};

            T v = warp::exscan_add( s );

            // Last lane of each warp publishes that warp's total
            if ( lane == gpu::warp_size - 1 ) tmp[ wid ] = v + s;
            block::sync();

            // Only warp 0 does this. The number of warps is always <= the warp
            // size, so a single pass suffices.
            if ( wid == 0 ) {
                // Lanes beyond the populated warps contribute zero rather than
                // whatever was left in tmp[] by the previous iteration
                T t = ( lane < nwarps ) ? tmp[ lane ] : T{0};
                t = warp::exscan_add( t );
                if ( lane < nwarps ) tmp[ lane ] = t + prev;
            }
            block::sync();

            // Add in contribution from previous warps and chunks
            v += tmp[ wid ];
            if ( i < size ) out[i] = v;

            // Running total for the next chunk. The last thread of the block
            // always holds it, since out-of-range elements contribute zero.
            if ( block::thread_rank() == nthreads - 1 ) prev = v + s;
            block::sync();
        }

        // The reduction (sum) value is also available, store it if requested
        if ( reduction != nullptr && block::thread_rank() == 0 ) *reduction = prev;
    }

    /**
     * @brief Block size for the single block scan.
     *
     * Rounded to a whole number of warps, as required by the full-mask warp
     * primitives, and capped at max_warps * warp_size.
     */
    __host__ inline
    unsigned int scan_block_size( size_t const size ) {
        constexpr unsigned int max_block = gpu::max_warps * gpu::warp_size;
        if ( size == 0 ) return gpu::warp_size;
        size_t rounded = ( ( size + gpu::warp_size - 1 ) / gpu::warp_size ) * gpu::warp_size;
        return ( rounded < max_block ) ? static_cast<unsigned int>( rounded ) : max_block;
    }

    } // namespace detail
    
    /**
     * @brief Perform exclusive scan (add) operation on device (in place)
     *
     * @warning This runs on a single block, i.e. a single SM, so it does not
     *          scale with array size. For large buffers use
     *          cub::DeviceScan::ExclusiveSum instead.
     *
     * @tparam T        Template data type
     * @param data      Data buffer (input/output)
     * @param size      Data buffer size (number of elements)
     */
    template< typename T >
    __host__ inline
    void exscan_add( T * const data, size_t const size )
    {
        if ( size == 0 ) return;
        const unsigned int block = detail::scan_block_size( size );
        detail::exscan_add_kernel<<< 1, block >>>( data, data, size, static_cast<T*>(nullptr) );
        check_launch();
    }

    /**
     * @brief Perform exclusive scan (add) operation on device
     *
     * @warning Single block; see the in place overload.
     *
     * @tparam T        Template data type
     * @param out       Output buffer
     * @param in        Input buffer
     * @param size      Data buffer size (number of elements)
     */
    template< typename T >
    __host__ inline
    void exscan_add( T * const out, T const * const in, size_t const size )
    {
        if ( size == 0 ) return;
        const unsigned int block = detail::scan_block_size( size );
        detail::exscan_add_kernel<<< 1, block >>>( out, in, size, static_cast<T*>(nullptr) );
        check_launch();
    }
    

    /**
     * @brief Perform exclusive scan (add) on device, return the total on host
     *
     * @note This synchronizes with the device.
     * @warning Single block; see the in place overload.
     *
     * @tparam T        Template data type
     * @param data      Data buffer (input/output)
     * @param size      Data buffer size (number of elements)
     * @return T        Sum of all elements
     */
    template< typename T >
    __host__ inline
    T exscan_reduce_add( T * const data, size_t const size )
    {
        if ( size == 0 ) return T{0};
        var<T> sum;
        const unsigned int block = detail::scan_block_size( size );
        detail::exscan_add_kernel<<< 1, block >>>( data, data, size, sum.ptr() );
        check_launch();
        return sum.get();
    }
    
} // namespace device

} // namespace gpu

