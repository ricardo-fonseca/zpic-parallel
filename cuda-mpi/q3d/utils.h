#ifndef UTILS_H_
#define UTILS_H_

#include "gpu.h"

#include <cstddef>
#include <cstring>
#include <typeinfo>
#include <iostream>


/**
 * @brief Rounds up to a multiple of 4
 * 
 * @tparam T    Value type (must be integer like)
 * @param a     Value to round up
 * @return T    a rounded up to the nearest multiple of 4
 */
template < typename T >
__host__ __device__
T roundup4( T a ) { return (a + 3) & static_cast<T>(-4);}

/**
 * @brief Rounds up to a multiple of N (where N is a power of 2)
 * 
 * @tparam N    Value will be rounded to a multiple of N. Must be a power of 2.
 * @tparam T    Value type. Must be an integer type (int, long, unsigned, int64_t, etc.)
 * @param a     Value to round up
 * @return T    Value rounded up to a multiple of N
 */
template < int N, typename T >
__host__ __device__
T roundup( T a ) {
    static_assert( N > 0, "N must be > 0");
    static_assert( !(N & (N-1)), "N must b a power of 2" );
    return ( a + (N-1) ) & static_cast<T>(-N);
};

/**
 * @brief Swaps 2 pointer values
 * 
 * @tparam T    Value type
 * @param a     Value a
 * @param b     Value b
 */
template < typename T >
void swap( T* &a, T* &b ) {
    T * tmp = a; a = b; b = tmp;
}

/**
 * @brief Returns maximum of 2 values
 * 
 * @tparam T    Value type
 * @param a     Value a
 * @param b     Value b
 * @return T    Maximum of a, b
 */
template < typename T >
T max( T a, T b ) {
    return ( b > a ) ? b : a ;
}

/**
 * @brief Prints a 2D array
 * 
 * @tparam T        Data type
 * @tparam T2       Array dimensions data type, must have .x and .y fields
 * @param buffer    Data buffer
 * @param dims      Array dimensions (.x and .y)
 */
template < typename T, typename T2 >
inline void print_array( T * __restrict__ buffer, T2 dims ) {
    for( auto i1 = 0; i1 < dims.y; i1 ++ ) {
        std::cout << buffer[i1 * dims.x];
        for( auto i0 = 1; i0 < dims.x; i0 ++ ) {
            std::cout << " " << buffer[i1 * dims.x + i0];
        }
        std::cout << '\n';
    }
}

/**
 * @brief ANSI escape codes for console output
 * 
 */
namespace ansi {
    static const std::string bold(  "\033[1m" );
    static const std::string reset( "\033[0m" );

    static const std::string black   ( "\033[30m" );
    static const std::string red     ( "\033[31m" );
    static const std::string green   ( "\033[32m" );
    static const std::string yellow  ( "\033[33m" );
    static const std::string blue    ( "\033[34m" );
    static const std::string magenta ( "\033[35m" );
    static const std::string cyan    ( "\033[36m" );
    static const std::string white   ( "\033[37m" );

}

namespace memspace {
    enum space { host = 0, device };
}

namespace memory {

    /**
     * @brief Allocate memory
     * 
     * @tparam T    Data type
     * @tparam s    Memory space (host / device)
     * @param size  Size (number of elements) to allocate
     * @return T*   Pointer to allocated region
     */
    template< typename T, memspace::space s >
    T * malloc( std::size_t const size ) { 
        if constexpr( s == memspace::host ) return host::malloc<T>( size );
        if constexpr( s == memspace::device ) return device::malloc<T>( size );
        // unreacheable
        return nullptr;
    }

    /**
     * @brief Free allocated memory
     * 
     * @tparam T    Data type
     * @tparam s    Memory space (host / device)
     * @param ptr   Pointer to allocated memory
     */
    template< typename T, memspace::space s >
    void free( T * ptr ) { 
        if constexpr( s == memspace::host ) host::free( ptr );
        if constexpr( s == memspace::device ) device::free( ptr );
    }

    /**
     * @brief Zeroes data buffer
     * 
     * @tparam T    Data type
     * @tparam s    Memory space (host / device)
     * @param ptr   Pointer to allocated memory
     * @param size  Buffer size (number of elements)
     */
    template< typename T, memspace::space s >
    void zero( T * const __restrict__ ptr, std::size_t const size ) {
        if constexpr( s == memspace::host ) host::zero( ptr, size );
        if constexpr( s == memspace::device ) device::zero( ptr, size );
    }
}


#endif