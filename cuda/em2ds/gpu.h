#ifndef GPU_H_
#define GPU_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <cstring>

/**
 * @brief CUDA Number of threads per warp
 * 
 */
#define WARP_SIZE 32

/**
 * @brief CUDA Maximum number of warps per block
 * 
 */
#define MAX_WARPS 32

/**
 * @brief Checks if the operation was successuful, otherwise aborts code
 * @note This will reset the CUDA device
 * 
 * @param   err_    Error code
 * @param   msg_    Error message to print in case of error
 */
#define CHECK_ERR( err_, msg_ ) { \
    auto __local_err = (err_); \
    if ( __local_err != cudaSuccess ) { \
        std::cerr << "(*error*) " << (msg_) << std::endl; \
        std::cerr << "(*error*) code: " << __local_err << ", reason: " << cudaGetErrorString(__local_err) << std::endl; \
        std::cerr << "(*error*) error state in " << __func__ << "()"; \
        std::cerr << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        cudaDeviceReset(); \
        exit(1); \
    } \
}

/**
 * @brief Aborts code
 * @note This will reset the CUDA device
 * 
 * @param   msg_    Error message to print in case of error
 */
#define ABORT( msg_ ) { \
    std::cerr << "(*error*) " << (msg_) << "\n"; \
    std::cerr << "(*error*) abort issued in " << __func__ << "()"; \
    std::cerr << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
    cudaDeviceReset(); \
    exit(1); \
}

/**
 * @brief Resets the GPU device, stopping any active kernels and clearing error states.
*/
#define deviceReset() { \
    cudaDeviceReset(); \
}

/**
 * @brief Checks if there are any synchronous or asynchronous errors from CUDA calls
 * 
 * @note If any errors are found the routine will print out the error messages and exit
 *       the program
 */
#define deviceCheck() { \
    auto err_sync = cudaPeekAtLastError(); \
    auto err_async = cudaDeviceSynchronize(); \
    if (( err_sync != cudaSuccess ) || ( err_async != cudaSuccess )) { \
        std::cerr << "(*error*) CUDA device is on error state at " << __func__ << "()"; \
        std::cerr << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        if ( err_sync != cudaSuccess ) \
            std::cerr << "(*error*) Sync. error message: " << cudaGetErrorString(err_sync) << " (" << err_sync << ") \n"; \
        if ( err_async != cudaSuccess ) \
            std::cerr << "(*error*) Async. error message: " << cudaGetErrorString(err_async) << " (" << err_async << ") \n"; \
        cudaDeviceReset(); \
        exit(1); \
    } \
}

/**
 * @brief Thread Id inside block
 * 
 * same as block.thread_rank() in CUDA cooperative groups, assuming that only 
 * threadIdx.x is used.
 * 
 */
#define block_thread_rank() (threadIdx.x)

/**
 * @brief Number of threads inside block
 * 
 * same as block.num_threads() in CUDA cooperative groups, assuming that only 
 * threadIdx.x is used.
 * 
 */
#define block_num_threads() (blockDim.x)

/**
 * @brief Thread Id inside grid
 * 
 * same as grid.thread_rank() in CUDA cooperative groups, assuming that only 
 * threadIdx.x and blockIdx.x are used.
 * 
 */

#define grid_thread_rank()  (threadIdx.x + blockIdx.x * blockDim.x)

/**
 * @brief Synchronize threads inside block
 * 
 * same as block.sync() in CUDA cooperative groups
 * 
 */
#define block_sync() __syncthreads()


/**
 * @brief Prints GPU device info
 * 
 */
static inline void print_gpu_info( ) {

    int device, nDevices;
    cudaDeviceProp prop;

    CHECK_ERR( cudaGetDevice( & device ), 
               "unable to get current device" );
    CHECK_ERR( cudaGetDeviceCount( & nDevices ),
               "unable to get number of devices" );
    CHECK_ERR( cudaGetDeviceProperties(&prop, device),
               "unable to get device properties" );

    std::cout << "Device Number           : " << device << " (of " << nDevices << ")\n";
    std::cout << "  Device name           : " << prop.name << '\n';

#if 0
    std::cout << "  Memory Clock Rate     : " << prop.memoryClockRate << " (KHz) \n";
    std::cout << "  Memory Bus Width      : " << prop.memoryBusWidth << " (bits) \n";
    std::cout << "  Peak Memory Bandwidth : " <<
        2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 << " (GB/s) \n";
#endif

    std::cout << "  Max. block size       : " << prop.maxThreadsPerBlock << "\n";
    std::cout << "  Warp size             : " << prop.warpSize << "\n";
    std::cout << "  Max. warps/block      : " << prop.maxThreadsPerBlock / prop.warpSize << "\n";
    std::cout << "  Shared memory (def.)  : " << prop.sharedMemPerBlock / 1024 << " (kB) \n";
    std::cout << "  Shared memory (optin) : " << prop.sharedMemPerBlockOptin / 1024 << " (kB) \n";
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
    __device__ __forceinline__ int num_threads() {
        return WARP_SIZE;
    }
    
    /**
     * @brief Thread rank inside warp
     * 
     * @return int 
     */
    __device__ __forceinline__ int thread_rank() {
        return threadIdx.x & (WARP_SIZE-1);
    }
    
    /**
     * @brief Warp ID in group
     * 
     * @return int 
     */
    __device__ __forceinline__ int group_rank() {
    #if WARP_SIZE == 64
        // Warp size 64 (2^6 = 64)
        return threadIdx.x >> 6;
    #elif WARP_SIZE == 32
        // Warp size 32 (2^5 = 32)
        return threadIdx.x >> 5;
    #else
        // General
        return threadIdx.x / WARP_SIZE;
    #endif
    }
    
    /**
     * @brief Warp level reduce add
     * 
     * @tparam T 
     * @param input 
     * @return __device__ 
     */
    template<typename T>
    __device__ __inline__ T reduce_add( T const input ) {
        T value = input;
        #pragma unroll
        for( int i = 1; i < WARP_SIZE; i <<= 1 )
            value += __shfl_xor_sync( 0xffffffff, value, i);
        return value;
    }
    
    /**
     * @brief Warp level reduce max
     * 
     * @tparam T 
     * @param input 
     * @return __device__ 
     */
    template<typename T>
    __device__ __inline__  T reduce_max( T const input ) {
        T value = input;
        #pragma unroll
        for( int i = 1; i < WARP_SIZE; i <<= 1 ) {
            T tmp = __shfl_xor_sync( 0xffffffff, value, i);
            if (tmp > value) value = tmp;
        }
        return value;
    }
    
    /**
     * @brief Warp level reduce min
     * 
     * @tparam T 
     * @param input 
     * @return __device__ 
     */
    template<typename T>
    __device__ __inline__  T reduce_min( T const input ) {
        T value = input;
        #pragma unroll
        for( int i = 1; i < WARP_SIZE; i <<= 1 ) {
            T tmp = __shfl_xor_sync( 0xffffffff,  value, i);
            if (tmp < value) value = tmp;
        }
        return value;
    }
    
    /**
     * @brief Warp level exclusive scan (add)
     * 
     * @tparam T 
     * @param input 
     * @return __device__ 
     */
    template<class T>
    __device__ __inline__ T exscan_add( T const input ) {
        T value = input;
        const int laneId = threadIdx.x & ( WARP_SIZE - 1 );
        #pragma unroll
        for( int i = 1; i < WARP_SIZE; i <<= 1 ) {
            T tmp = __shfl_up_sync( 0xffffffff, value, i );
            if ( laneId >= i ) value += tmp;
        }
    
        value = __shfl_up_sync( 0xffffffff, value, 1 );
        return (laneId > 0) ? value : 0;
    }
    
    /**
     * @brief Warp level inclusive scan (add)
     * 
     * @tparam T 
     * @param input 
     * @return __device__ 
     */
    template<class T>
    __device__ __inline__ T inscan_add( T const input ) {
        T value = input;
        const int laneId = threadIdx.x & ( WARP_SIZE - 1 );
        #pragma unroll
        for( int i = 1; i < WARP_SIZE; i <<= 1 ) {
            T tmp = __shfl_up_sync( 0xffffffff, value, i );
            if ( laneId >= i ) value += tmp;
        }
        return value;
    }
    
    /**
     * @brief Warp level reverse inclusive scan (add)
     * 
     * @note + Same as an inclusive scan but in the opposite order (right to left)
     * @note + All threads in warp must participate
     * 
     * @tparam T            Function type
     * @param input         Input value (independent for each lane)
     * @return              reverse scan result (different for each lane)
     */
    template<class T>
    __device__ __inline__ T rev_inscan_add( T const input ) {
        T value = input;
        const int laneId = threadIdx.x & ( WARP_SIZE - 1 );
        #pragma unroll
        for( int i = 1; i < WARP_SIZE; i <<= 1 ) {
            T tmp = __shfl_down_sync( 0xffffffff, value, i );
            if ( laneId < WARP_SIZE - i ) value += tmp;
        }
        return value;
    }

} // end of namespace warp

/**
 * @brief Block level routines
 * 
 */
namespace block {

    /**
     * @brief Returns pointer to (block) shared memory region
     * 
     * @tparam T    Datatype (defaults to char)
     * @return T *  Datatype pointer to shared memory
     */
    template< typename T = char >
    __device__ __inline__
    T * shared_mem() { 
        extern __shared__ char block_shm[];
        return reinterpret_cast<T *>(block_shm);
    };
    
    /**
     * @brief Returns max. shared memory per block (opt. in)
     * 
     * @param id            Device id, defaults to 0
     * @return size_t       Maximum shared memory per block (opt. in) in bytes
     */
    __host__ __inline__
    static size_t shared_mem_size( int id = 0 ) {
        cudaDeviceProp prop;
        auto err = cudaGetDeviceProperties(&prop, id);
        if ( err != cudaSuccess ) {
            std::cerr << "(*error*) Unable to query device(" << id << ") properties.\n";
            std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << "\n";
            cudaDeviceReset();
            exit(1);
        }
        return prop.sharedMemPerBlockOptin;
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
    static __inline__ __host__ cudaError_t set_shmem_size(
      T *entry, size_t shm_size
    ) {
        if ( shm_size > 49152 ) {
            return cudaFuncSetAttribute( entry, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size );
        } else {
            return cudaSuccess;
        }
    }
    
    /**
     * @brief Atomic fetch-add operation (block level)
     * 
     * @tparam T            Data type
     * @param address       Target address for operation
     * @param val           Value to add
     * @return T            Target value before add
     */
    template< typename T >
    __device__ __forceinline__
    auto atomic_fetch_add( T * address, T val ) {
        return atomicAdd_block( address, val );
    }

    /**
     * @brief Atomic fetch-max operation (block level)
     * 
     * @tparam T            Data type
     * @param address       Target address for operation
     * @param val           Value to compare
     * @return T            Target value before comparison
     */
    template< typename T >
    __device__ __forceinline__
    auto atomic_fetch_max( T * address, T val ) {
        return atomicMax_block( address, val );
    }
    
    /**
     * @brief Atomic fetch-min operation (block level)
     * 
     * @tparam T            Data type
     * @param address       Target address for operation
     * @param val           Value to compare
     * @return T            Target value before comparison
     */
    template< typename T >
    __device__ __forceinline__
    auto atomic_fetch_min( T * address, T val ) {
        return atomicMin_block( address, val );
    }
    
    /**
     * @brief Synchronize threads (block level)
     * 
     */
    __device__ __forceinline__
    static void sync() { __syncthreads(); }
    
    /**
     * @brief Block level exclusive scan (add)
     * 
     * @tparam T        Type
     * @param input     Input value
     * @return T        Result
     */
    
    template<typename T>
    __device__ T exscan_add( T const input ) {
    
        __shared__ int tmp[ MAX_WARPS ];
    
        T v = warp::exscan_add( input );
    
        if ( warp::thread_rank() == warp::num_threads() - 1 )
            tmp[ warp::group_rank() ] = v + input;
        block::sync();
    
        // Only 1 warp does this
        if ( warp::group_rank() == 0 ) {
            auto t = tmp[ warp::thread_rank() ];
            t = warp::exscan_add( t );
            tmp[ warp::thread_rank() ] = t;
        }
        block::sync();
    
        // Add in contribution from previous warps
        v += tmp[ warp::group_rank() ];
    
        return v;
    }
    
    __device__ __inline__
    /**
     * @brief Block level memcpy of float3 values
     * 
     * @warning Must be called by all threads in block
     * 
     * @param dst   Destination address
     * @param src   Source address
     * @param n     Number of elements to copy
     */
    void memcpy( float3 * __restrict__ dst, float3 const * __restrict__ src, const size_t n ) {
        float *       __restrict__ _dst = reinterpret_cast<float *>(dst);
        float const * __restrict__ _src = reinterpret_cast<float const *>(src);
    
        for( auto i = block_thread_rank(); i < 3*n; i += block_num_threads() )
            _dst[i] = _src[i];
    }
    
    __device__ __inline__
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
    void memcpy2( float3 * __restrict__ dst1, float3 const * __restrict__ src1, 
                  float3 * __restrict__ dst2, float3 const * __restrict__ src2,
                  const size_t n ) {
        float *       __restrict__ _dst1 = reinterpret_cast<float *>(dst1);
        float const * __restrict__ _src1 = reinterpret_cast<float const *>(src1);
        float *       __restrict__ _dst2 = reinterpret_cast<float *>(dst2);
        float const * __restrict__ _src2 = reinterpret_cast<float const *>(src2);
    
        for( auto i = block_thread_rank(); i < 3*n; i += block_num_threads() ) {
            _dst1[i] = _src1[i];
            _dst2[i] = _src2[i];
        }
    }
}

/**
 * @brief Device routines
 * 
 */
namespace device {
    
    /**
     * @brief Wait for compute device to finish.
     * 
     * @return      If the GPU is in an error state, the function will return an error
     */
    __host__
    inline int sync() {
        return cudaDeviceSynchronize();
    }


    __host__
    [[noreturn]] inline void exit(int status) {
        cudaDeviceReset();
        exit( status );
    }

    /**
     * @brief   Allocate memory on device
     * 
     * @tparam T    Data type
     * @param size  Size (number of elements) to allocate
     */
    template< typename T >
    __host__
    T * malloc( std::size_t const size ) {
        T * buffer;
        auto err = cudaMalloc( &buffer, size * sizeof(T) );
        if ( err != cudaSuccess ) {
            std::cerr << "(*error*) Unable to allocate " << size << " elements of type " << typeid(T).name() << " on device.\n";
            std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << "\n";
            cudaDeviceReset();
            exit(1);
        }
        return buffer;
    }
    
    /**
     * @brief   Free device allocated memory
     * 
     * @tparam T    Data type
     * @param ptr   Pointer to allocated memory
     */
    template< typename T >
    __host__
    void free( T * ptr ) {
        if ( ptr != nullptr ) {
            auto err = cudaFree( ptr );
            if ( err != cudaSuccess ) {
                std::cerr << "(*error*) Unable to deallocate " << typeid(T).name() << " buffer at " << ptr << " from device.\n";
                std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << "\n";
                cudaDeviceReset();
                exit(1);
            }
        }
    }
    
    /**
     * @brief Zeroes data buffer
     * 
     * @tparam T    Data type
     * @param ptr   Pointer to data buffer
     * @param size  Buffer size (number of elements)
     */
    template< typename T >
    __host__
    void zero( T * const __restrict__ ptr, std::size_t const size ) {
        auto err = cudaMemsetAsync( ptr, 0, size * sizeof(T) );
        if ( err != cudaSuccess ) {
            std::cerr << "(*error*) Unable to zero device memory." << std::endl;
            std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
            cudaDeviceReset();
            exit(1);
        }
    }
    
    namespace {
    
    /**
     * @brief CUDA kernel for setval routine
     * 
     * @tparam T 
     * @param d_data 
     * @param size 
     * @param val 
     * @return __global__ 
     */
    template < typename T >
    __global__
    void setval_kernel( T * __restrict__ d_data, std::size_t const size, const T val ) {
        int i = grid_thread_rank();
        if ( i < size ) d_data[i] = val;
    }
    
    }
    
    /**
     * @brief Sets buffer to scalar value
     * 
     * @tparam T    Data type
     * @param ptr   Pointer to data buffer
     * @param size  Buffer size (number of elements)
     * @param val   Scalar value to set (passed by copy)
     * @param q     SYCL queue
     * @return      SYCL event object
     */
    template< typename T >
    __host__
    auto setval( T * const __restrict__ d_data, std::size_t const size, const T val ) {
        
        const auto block = ( size < 1024 ) ? size : 1024 ;
        const auto grid  = ( size - 1 ) / block + 1;
    
        setval_kernel <<< grid, block >>> ( d_data, size, val );
    }
    
    
    /**
     * @brief Copies data from device to host
     * 
     * @warning The code will wait for the queue to finish before submitting the memcpy action
     * 
     * @tparam T        Data type
     * @param h_out     Output host buffer
     * @param d_in      Input device buffer
     * @param size      Buffer size (number of elements)
     */
    template< typename T >
    void memcpy_tohost( T * const __restrict__ h_out, T const * const __restrict__ d_in, size_t const size) {
        
        auto err = cudaMemcpy( h_out, d_in, size * sizeof(T), cudaMemcpyDeviceToHost );
        if ( err != cudaSuccess ) {
            std::cerr << "(*error*) Unable to copy " << size << " elements of type " << typeid(T).name() << " from device to host.\n";
            std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << "\n";
            cudaDeviceReset();
            exit(1);
        }
    }

    /**
     * @brief Atomic fetch-add operation. Returns the value before the operation.
     * 
     * @tparam T        Template type
     * @param addr      Address of the value to be modified
     * @param val       Value to be added
     * @return T        Value before the operation
     */
    template< typename T >
    __device__ __forceinline__
    auto atomic_fetch_add( T * address, T val ) {
        return atomicAdd( address, val );
    }
    
    /**
     * @brief Atomic fetch-max operation. Returns the value before the operation.
     * 
     * @tparam T        Template type
     * @param addr      Address of the value to be modified
     * @param val       Value to be compared with the target value
     * @return T        Value before the operation
     */
    template< typename T >
    __device__ __forceinline__
    auto atomic_fetch_max( T * address, T val ) {
        return atomicMax( address, val );
    }
    
    /**
     * @brief Atomic fetch-min operation. Returns the value before the operation.
     * 
     * @tparam T        Template type
     * @param addr      Address of the value to be modified
     * @param val       Value to be compared with the target value
     * @return T        Value before the operation
     */
    template< typename T >
    __device__ __forceinline__
    auto atomic_fetch_min( T * address, T val ) {
        return atomicMin( address, val );
    }
    
    namespace {
    
    /**
     * @brief Kernel for `exscan_add()` inplace function
     * 
     * @tparam T            Template datatype
     * @param data          Data buffer (in/out)
     * @param size          Data buffer size (number of elements)
     * @param reduction     Output reduction (optional). Set to a non-null pointer to
     *                      store global reduction on this address.
     */
    template < typename T >
    __global__ 
    void exscan_add_kernel(
        T * __restrict__ data, unsigned int const size, T * __restrict__ reduction = nullptr
    ) {
    
        static_assert( MAX_WARPS <= WARP_SIZE, "This implementation requires MAX_WARPS to be <= WARP_SIZE");
        
        __shared__ T tmp[ MAX_WARPS ];
        __shared__ T prev;
    
        // Contribution from previous warp
        prev = 0;
    
        for( unsigned int i = block_thread_rank(); i < size; i += block_num_threads() ) {
            auto s = data[i];
    
            auto v = warp::exscan_add(s);
            if ( warp::thread_rank() == WARP_SIZE - 1 ) tmp[ warp::group_rank() ] = v + s;
            block_sync();
    
            // Only 1 warp does this
            if (warp::group_rank() == 0 ) {
                // The maximum number of warps will always be less or equal than
                // the warp size, so we only need to do this once
                auto t = tmp[ warp::thread_rank() ];
                t = warp::exscan_add(t);
                tmp[ warp::thread_rank() ] = t + prev;
            }
            block_sync();
    
            // Add in contribution from previous threads
            v += tmp[ warp::group_rank() ];
            data[i] = v;
    
            if ((block_thread_rank() == block_num_threads() - 1) || ( i + 1 == size ) )
                prev = v + s;
    
            block_sync();
        }
    
        // The reduction (sum) value is also available, store it if requested
        if ( reduction != nullptr )
            if ( block_thread_rank() == 0 ) *reduction = prev;
    }

    /**
     * @brief Kernel for `exscan_add()` function
     * 
     * @tparam T            Template datatype
     * @param out           Output data buffer
     * @param in            Input data buffer
     * @param size          Data buffer size (number of elements)
     * @param reduction     Output reduction (optional). Set to a non-null pointer to
     *                      store global reduction on this address.
     */
    template < typename T >
    __global__ 
    void exscan_add_kernel(
        T * __restrict__ out, T * __restrict__ in, 
        unsigned int const size, T * __restrict__ reduction = nullptr
    ) {
    
        static_assert( MAX_WARPS <= WARP_SIZE, "This implementation requires MAX_WARPS to be <= WARP_SIZE");
        
        __shared__ T tmp[ MAX_WARPS ];
        __shared__ T prev;
    
        // Contribution from previous warp
        prev = 0;
    
        for( unsigned int i = block_thread_rank(); i < size; i += block_num_threads() ) {
            auto s = in[i];
    
            auto v = warp::exscan_add(s);
            if ( warp::thread_rank() == WARP_SIZE - 1 ) tmp[ warp::group_rank() ] = v + s;
            block_sync();
    
            // Only 1 warp does this
            if (warp::group_rank() == 0 ) {
                // The maximum number of warps will always be less or equal than
                // the warp size, so we only need to do this once
                auto t = tmp[ warp::thread_rank() ];
                t = warp::exscan_add(t);
                tmp[ warp::thread_rank() ] = t + prev;
            }
            block_sync();
    
            // Add in contribution from previous threads
            v += tmp[ warp::group_rank() ];
            out[i] = v;
    
            if ((block_thread_rank() == block_num_threads() - 1) || ( i + 1 == size ) )
                prev = v + s;
    
            block_sync();
        }
    
        // The reduction (sum) value is also available, store it if requested
        if ( reduction != nullptr )
            if ( block_thread_rank() == 0 ) *reduction = prev;
    }

    }
    
    /**
     * @brief Perform exclusive scan (add) operation on device (inplace)
     * 
     * @tparam T        Template data type
     * @param data      Data buffer (input/output)
     * @param size      Data buffer size (number of elements)
     */
    template< typename T >
    __host__
    void exscan_add( T * const __restrict__ data, size_t const size )
    {
        unsigned int block = ( size < 1024 ) ? size : 1024 ;
        exscan_add_kernel <<< 1, block >>> ( data, size );
    }

    /**
     * @brief Perform exclusive scan (add) operation on device
     * 
     * @tparam T 
     * @param data 
     * @param size 
     * @return T 
     */
    template< typename T >
    __host__
    void exscan_add( T * const __restrict__ out, T * const __restrict__ in, size_t const size )
    {
        unsigned int block = ( size < 1024 ) ? size : 1024 ;
        exscan_add_kernel <<< 1, block >>> ( out, in, size );
    }
    
    /**
     * @brief Class representing a scalar variable in device memory
     * 
     * @note This class simplifies the creation of scalar variables in unified
     *       memory. Note that getting the variable in the host (`get()`) will
     *       always cause a memcpy from device to host.
     * 
     * @tparam T    Variable datatype
     */
    template< typename T> class Var {
        private:
    
        T * data;
    
        public:
    
        __host__
        /**
         * @brief Construct a new Var<T> object
         * 
         */
        Var() {
            auto err = cudaMalloc( &data, sizeof(T) );
            CHECK_ERR( err, "Unable to allocate managed memory for device::Var" );
        }
    
        __host__
        /**
         * @brief Construct a new Var<T> object and set value to val
         * 
         * @param val 
         */
        Var( const T val ) : Var() {
            set( val );
        }
    
        __host__
        /**
         * @brief Destroy the Var<T> object
         * 
         */
        ~Var() {
            auto err = cudaFree( data );
            CHECK_ERR( err, "Unable to free memory for device::Var" );
        }
    
        __host__
        /**
         * @brief Sets the value of the Var<T> object
         * 
         * @note Data is copied to device using cudaMemcpyAsync()
         * 
         * @param val       value to set
         * @return T const  returns same value
         */
        T const set( const T val ) {
            auto err = cudaMemcpyAsync( data, &val, sizeof(T), cudaMemcpyHostToDevice );
            CHECK_ERR( err, "Failed to copy value to device on device::Var.set()" );

            return val;
        }
    
        __host__
        /**
         * @brief Overloaded assignment operation for setting the object value
         * 
         * @param val 
         * @return T 
         */
        T operator= (const T val) {
            return set(val);
        }
    
        __host__
        /**
         * @brief Returns value of variable
         * 
         * @warning This will always perform a cudaMemcpy()
         * 
         * @return T const 
         */
        T const get() const { 
            
            T val;

            auto err = cudaMemcpy( &val, data, sizeof(T), cudaMemcpyDeviceToHost );
            CHECK_ERR( err, "Failed to copy data from device on device::Var.get()" );

            return val;
        }
    
        __host__ 
        /**
         * @brief Pointer to variable data
         * 
         * @return T* 
         */
        T * ptr() const { return data; }
    
        __host__
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
        friend std::ostream& operator<< (std::ostream& os, device::Var<T> const & d) { 
            return os << d.get();
        }
    
    };
    
    /**
     * @brief Perform exclusive scan (add) operation on device, return reduction on host
     * 
     * @tparam T 
     * @param data 
     * @param size 
     * @return T 
     */
    template< typename T >
    __host__
    T exscan_reduce_add( T * const __restrict__ data, size_t const size )
    {
        device::Var<T> sum;
        unsigned int block = ( size < 1024 ) ? size : 1024 ;
        exscan_add_kernel <<< 1, block >>> ( data, size, sum.ptr());
        return sum.get();
    }
    
} // end namespace device


/**
 * @brief Device routines
 * 
 */
namespace managed {

    /**
     * @brief   Allocate managed memory
     * 
     * @tparam T    Data type
     * @param size  Size (number of elements) to allocate
     */
    template< typename T >
    T * malloc( std::size_t const size ) {
        T * buffer;
        auto err = cudaMallocManaged( &buffer, size * sizeof(T) );
        if ( err != cudaSuccess ) {
            std::cerr << "(*error*) Unable to allocate " << size << " elements of type " << typeid(T).name() << " on managed memory.\n";
            std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << "\n";
            cudaDeviceReset();
            exit(1);
        }
        return buffer;
    }

    /**
     * @brief   Free allocated managed memory
     * 
     * @tparam T    Data type
     * @param ptr   Pointer to allocated memory
     */
    template< typename T >
    void free( T * ptr ) {
        if ( ptr != nullptr ) {
            auto err = cudaFree( ptr );
            if ( err != cudaSuccess ) {
                std::cerr << "(*error*) Unable to deallocate " << typeid(T).name() << " buffer at " << ptr << " from managed memory.\n";
                std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << "\n";
                cudaDeviceReset();
                exit(1);
            }
        }
    }

} // end namespace managed

/**
 * @brief Host memory routines
 * 
 */
namespace host {
    
    /**
     * @brief   Allocate memory on host
     * 
     * @tparam T    Data type
     * @param size  Size (number of elements) to allocate
     */
    template< typename T >
    T * malloc( std::size_t const size ) {
    
        T * buffer;
        auto err = cudaMallocHost( &buffer, size * sizeof(T) );
        if ( err != cudaSuccess ) {
            std::cerr << "(*error*) Unable to allocate " << size << " elements of type " << typeid(T).name() << " on host.\n";
            std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << "\n";
            cudaDeviceReset();
            exit(1);
        }
        return buffer;
    }
    
    /**
     * @brief   Free host allocated memory
     * 
     * @tparam T    Data type
     * @param ptr   Pointer to allocated memory
     */
    template< typename T >
    void free( T * ptr ) {
        if ( ptr != nullptr ) {
            auto err = cudaFreeHost( ptr );
            if ( err != cudaSuccess ) {
                std::cerr << "(*error*) Unable to deallocate " << typeid(T).name() << " buffer at " << ptr << " from host.\n";
                std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << "\n";
                cudaDeviceReset();
                exit(1);
            }
        }
    }

    /**
     * @brief Sets a memory region to 0
     * 
     * @tparam T        Data type
     * @param data      Pointer to buffer
     * @param size      Data size (# of elements)
     * @return T* 
     */
    template< typename T >
    T * zero( T * const __restrict__ data, unsigned int const size ) {
        return (T *) std::memset( (void *) data, 0, size * sizeof(T) );
    }

    /**
     * @brief Copies data from host to device
     * 
     * @tparam T        Data type
     * @param d_out     Output device buffer
     * @param h_in      Input host buffer
     * @param size      Buffer size (number of elements)
     */
    template< typename T >
    void memcpy_todevice( T * const __restrict__ d_out, T const * const __restrict__ h_in, size_t const size) {
        
        auto err = cudaMemcpy( d_out, h_in, size * sizeof(T), cudaMemcpyHostToDevice );
        if ( err != cudaSuccess ) {
            std::cerr << "(*error*) Unable to copy " << size << " elements of type " << typeid(T).name() << " from host to device.\n";
            std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << "\n";
            cudaDeviceReset();
            exit(1);
        }
    }

} // end of namespace host

#endif