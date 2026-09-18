#pragma once

#include <cstddef>
#include <iostream>

#include "math.hpp" // for roundup<>()

namespace memory {

/**
 * @brief Allocates aligned block of memory
 * 
 * @tparam T        Data type
 * @tparam align    Memory alignment, defaults to 64 bit. Must be a power of 2.
 * @param size      Number of elements (not bytes)
 * @return T*       Pointer to allocated memory
 */
template< typename T, int align = 64 >
T * malloc( std::size_t const size ) {

    static_assert( align > 0, "align must be > 0");
    static_assert( !(align & (align-1)), "align must be a power of 2" );

    std::size_t size_align = roundup<align>( size * sizeof(T) );
    T * buffer = (T *) std::aligned_alloc( align, size_align );

    if ( buffer == nullptr ) {
        std::cerr << "(*error*) Unable to allocate " << size << " elements of type " << typeid(T).name();
        std::cerr << " (" << (size_align) << " bytes)\n";
        std::exit(1);
    }

    return buffer;
}

/**
 * @brief Copy n elements of type T from src to dst
 * 
 * @warning dst and src must not overlap
 * 
 * @tparam T    Datatype
 * @param dst   Destination
 * @param src   Source
 * @param n     Number of elements (not bytes)
 * @return T* 
 */
template< typename T >
T * memcpy( T * __restrict__ dst, const T * __restrict__ src, size_t n ) {
    return reinterpret_cast<T *> ( std::memcpy( dst, src, n * sizeof(T) ) );
}

/**
 * @brief Deallocates block of memory. To be used with the memory::malloc() routines
 * 
 * @tparam T        Data type
 * @param buffer    Pointer to allocated block of memory
 */
template< typename T >
void free( T * buffer ) {
    if ( buffer != nullptr ) {
        std::free( buffer );
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
T * zero( T * const __restrict__ data, std::size_t size ) {
    return (T *) std::memset( (void *) data, 0, size * sizeof(T) );
}


}