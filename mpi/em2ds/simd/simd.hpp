#pragma once

#include <cstddef>
#include <string>
#include <iostream>

/**
 * @brief Enable use of x86 AVX2 optimized code
 * 
 */
#ifdef USE_AVX2

#ifdef SIMD
#error SIMD has already been defined, only 1 type of SIMD code should be enabled.
#endif

#include "avx2.h"
#define SIMD AVX2

#endif

/**
 * @brief Enable use of x86 AVX512 optimized code
 * 
 */
#ifdef USE_AVX512

#ifdef SIMD
#error SIMD has already been defined, only 1 type of SIMD code should be enabled.
#endif

#include "avx512.h"
#define SIMD AVX512

#endif

/**
 * @brief Enable use of ARM NEON optimized code
 * 
 */
#ifdef USE_NEON

#ifdef SIMD
#error SIMD has already been defined, only 1 type of SIMD code should be enabled.
#endif

#include "neon.hpp"
#define SIMD NEON

#endif

/**
 * @brief Enable use of ARM SVE optimized code
 * 
 */
#ifdef USE_SVE

#ifdef SIMD
#error SIMD has already been defined, only 1 type of SIMD code should be enabled.
#endif

#include "sve.h"
#define SIMD SVE

#endif


/**
 * @brief Disable use of SIMD optimized code
 * 
 */
#ifndef SIMD

constexpr char vecname[] = "none";
constexpr int vecwidth = 1;
inline int simd_init() {return 0;}

#endif

/**
 * Memory alignment routines
 */

/**
 * @brief Checks if supplied address is aligned to the n-bit boundary
 * 
 * @tparam n        Number of bits (must be a power of 2)
 * @tparam T        Address type (determined from the caller variable)
 * @param addr      Address
 * @return true     The address is n-bit aligned
 * @return false    The address is not n-bit aligned
 */
template< unsigned int n, typename T >
constexpr bool is_aligned( T * addr ) {
    static_assert( (n & (n-1)) == 0, "n must be a power of 2" );
    return (((uintptr_t)addr & (n-1)) == 0);
}

/**
 * @brief Assert the address is n bit aligned
 * 
 * If the address is not aligned the routine will call `abort()` stopping the
 * program
 * 
 * @tparam n        Number of bits (must be a power of 2)
 * @tparam T        Address type (determined from the caller variable)
 * @param addr      Address
 * @param msg       (optional) Message to print in case the address is not
 *                  aligned
 */
template< unsigned int n, typename T >
void assert_aligned( T * addr, std::string msg = "" ) {
    if ( ! is_aligned<n>(addr) ) {
        if ( ! msg.empty() ) std::cerr << msg << '\n';
        // We cast the address to int* to avoid the address being interpreted
        // as a string in case we call the function with char*
        std::cerr << "Address " << (int*) addr << " is not ";
        std::cerr << n << " bit aligned, aborting." << std::endl;
        abort();
    }
}

#ifdef SIMD

/**
 * @brief Accumulates float values from src into dst ( dst[i] += src[i] )
 *
 * @warning dst and tgt must not overlap
 * @warning Both addresses must be aligned to the SIMD vector size
 *
 * @param dst   Target buffer (read-modify-write)
 * @param src   Source buffer
 * @param n     Number of float values
 */
inline void vec_memadd( float * __restrict__ dst, const float * __restrict__ src, std::size_t n ) {

    constexpr std::size_t blk = 4 * vecwidth;

    std::size_t i = 0;

    // Main loop, unrolled 4x to keep several loads in flight
    for( ; i + blk <= n; i+= blk ) {
        vfloat a0 = vec_add( vec_load(&dst[i             ]), vec_load(&src[i             ]) );
        vfloat a1 = vec_add( vec_load(&dst[i +   vecwidth]), vec_load(&src[i +   vecwidth]) );
        vfloat a2 = vec_add( vec_load(&dst[i + 2*vecwidth]), vec_load(&src[i + 2*vecwidth]) );
        vfloat a3 = vec_add( vec_load(&dst[i + 3*vecwidth]), vec_load(&src[i + 3*vecwidth]) );

        vec_store( &dst[i             ], a0 );
        vec_store( &dst[i +   vecwidth], a1 );
        vec_store( &dst[i + 2*vecwidth], a2 );
        vec_store( &dst[i + 3*vecwidth], a3 );
    }

    // remaining full vectors
    for( ; i + vecwidth <=n; i += vecwidth )
        vec_store( &dst[i], vec_add( vec_load( &dst[i] ), vec_load( &src[i] ) ) );

    // remaining scalars
    for( ; i < n; i++ )
        dst[i] += src[i];
}

#endif