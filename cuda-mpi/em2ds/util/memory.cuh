#pragma once

#include <cstring>
#include <string>
#include "../core/gpu.cuh"
#include "device_types.h"

namespace gpu {

/**
 * @brief Device memory routines
 * 
 */
namespace device {

    /**
     * @brief   Allocate memory on device
     * 
     * @tparam T    Data type
     * @param size  Size (number of elements) to allocate
     */
    template< typename T >
    __host__
    inline T * malloc( std::size_t const size ) {
        T * buffer;
        gpu::check_err( 
            cudaMalloc( &buffer, size * sizeof(T) ),
            "Unable to allocate " + std::to_string( size ) + " elements of type " 
            + typeid(T).name() + " on device."
        );

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
    inline void free( T * ptr ) {
        if ( ptr != nullptr ) {
            gpu::check_err(
                cudaFree( ptr ),
                std::string( "Unable to deallocate " ) + typeid(T).name() + 
                " buffer from device."
            );
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
    inline void zero( T * const __restrict__ ptr, std::size_t const size ) {
        gpu::check_err(
            cudaMemsetAsync( ptr, 0, size * sizeof(T) ),
            "Unable to zero device memory."
        );
    }
    
    namespace kernel {
    
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
    void set( T * __restrict__ d_data, std::size_t const size, const T val ) {
        int i = grid::thread_rank();
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
     */
    template< typename T >
    __host__
    inline void set( T * const __restrict__ d_data, std::size_t const size, const T val ) {
        
        const auto block = ( size < 1024 ) ? size : 1024 ;
        const auto grid  = ( size - 1 ) / block + 1;
    
        kernel::set <<< grid, block >>> ( d_data, size, val );
    }


    namespace kernel {
    
    /**
     * @brief CUDA kernel for add function (lhs += rhs)
     * 
     * @tparam T            Datatype, must support += operation
     * @param lhs           Target buffer
     * @param rhs           Source buffer
     * @param size          Buffer size
     * @return __global__ 
     */
    template < typename T >
    __global__
    void add( T * const __restrict__ lhs, T const * const __restrict__ rhs, std::size_t const size ) {
        int i = grid::thread_rank();
        if ( i < size ) lhs[i] += rhs[i];
    }
    
    }

    /**
     * @brief In-place addition of 2 vectors
     * 
     * @tparam T            Datatype, must support += operation
     * @param lhs           Target buffer
     * @param rhs           Source buffer
     * @param size          Buffer size
     */
    template< typename T >
    __host__
    inline void add( T * const __restrict__ lhs, T const * const __restrict__ rhs, std::size_t const size ) {
        const auto block = ( size < 1024 ) ? size : 1024 ;
        const auto grid  = ( size - 1 ) / block + 1;
    
        kernel::add <<< grid, block >>> ( lhs, rhs, size );
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
    __host__
    inline void memcpy_tohost( T * const __restrict__ h_out, T const * const __restrict__ d_in, size_t const size) {
        gpu::check_err(
            cudaMemcpy( h_out, d_in, size * sizeof(T), cudaMemcpyDeviceToHost ),
            "Unable to copy " + std::to_string(size ) + " elements of type " + typeid(T).name() + " from device to host."
        );
    }

    /**
     * @brief Copies data from device to device
     *
     * @warning The code will wait for the queue to finish before submitting the memcpy action
     * 
     * @tparam T        Data type
     * @param d_out     Ouput device buffer
     * @param d_in      Input device buffer
     * @param size      Buffer size (number of elements)
     */
    template< typename T >
    __host__
    inline void memcpy_todevice( T * const __restrict__ d_out, T const * const __restrict__ d_in, size_t const size) {
        gpu::check_err(
            cudaMemcpy( d_out, d_in, size * sizeof(T), cudaMemcpyDeviceToDevice ),
            "Unable to copy " + std::to_string(size ) + " elements of type " + typeid(T).name() + " from device to device."
        );
    }
}


/**
 * @brief Managed memory routines
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
        gpu::check_err(
            cudaMallocManaged( &buffer, size * sizeof(T) ),
            "Unable to allocate " + std::to_string(size) + " elements of type "
            + typeid(T).name() + " on managed memory."
        );
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
            gpu::check_err(
                cudaFree( ptr ),
                std::string("Unable to deallocate ") + typeid(T).name() + 
                " buffer from managed memory."
            );
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
        gpu::check_err(
            cudaMallocHost( &buffer, size * sizeof(T) ),
            "Unable to allocate " + std::to_string( size ) + " elements of type " 
                      + typeid(T).name() + " on host."
        );
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
            gpu::check_err(
                cudaFreeHost( ptr ),
                std::string("Unable to deallocate ") + typeid(T).name() 
                          + " buffer from host."
            );
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
        
        gpu::check_err(
            cudaMemcpy( d_out, h_in, size * sizeof(T), cudaMemcpyHostToDevice ),
            "Unable to copy " + std::to_string( size ) + " elements of type " 
                      + typeid(T).name() + " from host to device."
        );
    }

} // end of namespace host

} // namespace gpu