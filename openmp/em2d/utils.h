#ifndef UTILS_H_
#define UTILS_H_

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cstddef>
#include <cstring>
#include <typeinfo>
#include <iostream>

#include <limits>

#include <cmath>

/**
 * @brief Rounds up to a multiple of 4
 * 
 * @tparam T    Value type (must be integer like)
 * @param a     Value to round up
 * @return T    a rounded up to the nearest multiple of 4
 */
template < typename T >
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
 * @brief Dummy atomicAdd function
 * 
 * @warning This function does not insure atomicity, it is only used as a placeholder
 * to ensure that when porting to other architectures we use a proper atomic operation.
 * 
 * @note The syntax is similar to that of the CUDA atomicAdd() operation, but we use a reference
 * instead of a memory address.
 * 
 * @tparam T    Data type
 * @param a     Reference to data value
 * @param b     Value to be added (atomically)
 * @return T    Data value before operation
 */
template < typename T >
inline T atomicAdd( T & a, T b ) {
    T tmp = a; a += b; return tmp;
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

namespace ops {

#define FP_FAST_FMA 1

/**
 * @brief
 * Multiply-add operation: f = (x * y) + z
 * 
 * @note
 * If the FP_FAST_FMA macro is defined then the routine will call std::fma()
 * which is supposed to implement a (faster) fused multply-add operation.
 * Otherwise, we just do the normal operation to avoid calling the much slower
 * fma operation in libm.
 * 
 * @tparam T 
 * @param x 
 * @param y 
 * @param z 
 * @return auto 
 */
template<typename T>
auto fma( T const x, T const y, T const z ) {

#ifdef FP_FAST_FMA
    return std::fma( x, y, z );
#else
    return (x*y)+z;
#endif

}

}

namespace memory {

/**
 * @brief Allocates aligned block of memory
 * 
 * @tparam T        Data type
 * @tparam align    Memory aligment, defaults to 64 bit. Must be a power of 2.
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
        exit(1);
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

namespace omp {

/**
 * @brief Atomic fetch/add operation
 * 
 * @note If OpenMP support is not enabled this just performs a standard 
 *       fetch/add operation
 * 
 * @tparam T    Template data type
 * @param addr  Target value address
 * @param val   Value to be added
 * @return T    Value at address before adding val
 */
template <class T>
inline T atomic_fetch_add( T * addr, T val ) {
    T t;
    #pragma omp atomic capture
    { t = *addr; *addr += val; }
    return t;
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



#endif