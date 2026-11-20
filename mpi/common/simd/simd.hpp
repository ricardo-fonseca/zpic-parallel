#pragma once

/**
 * @file simd.hpp
 * @brief Explicit SIMD vectorization support
 * 
 * @details Select (at most) one backend at compile time with USE_AVX2,
 *          USE_AVX512, USE_NEON or USE_SVE. When a backend is selected the
 *          SIMD macro is defined (e.g. `#define SIMD AVX2`).
 * 
 *          Everything is defined in the `simd` namespace:
 *          - `simd::name`, `simd::width` and `simd::init()` are always
 *            available (with no backend: "none", 1, and a no-op)
 *          - `simd::is_aligned()` and `simd::assert_aligned()` are always
 *            available
 *          - The vector types (`vfloat`, `vint`, `vmask`, `vfloat2`, ...) and
 *            the `vec_*()` functions are only available when a backend is
 *            selected (`#ifdef SIMD`)
 * 
 *          The `vec_*()` functions take raw intrinsic types (e.g. `__m256`),
 *          which have no namespace, so argument-dependent lookup will not
 *          find them. Source files using them should add
 *          `using namespace simd;` inside their `#ifdef SIMD` section (never
 *          in a header file).
 */

#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

#include "avx2.hpp"
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

#include "avx512.hpp"
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

#include "sve.hpp"
#define SIMD SVE

#endif


namespace simd {

/**
 * @brief Disable use of SIMD optimized code if SIMD has not been defined
 * 
 */
#ifndef SIMD

/// @brief SIMD vector unit name
constexpr char name[] = "none";
/// @brief Main SIMD vector width
constexpr int width = 1;
/// @brief SIMD initialization (no initialization required)
inline int init() {return 0;}

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
    return (((std::uintptr_t)addr & (n-1)) == 0);
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
        std::abort();
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

    constexpr std::size_t blk = 4 * width;

    std::size_t i = 0;

    // Main loop, unrolled 4x to keep several loads in flight
    for( ; i + blk <= n; i+= blk ) {
        vfloat a0 = vec_add( vec_load(&dst[i             ]), vec_load(&src[i             ]) );
        vfloat a1 = vec_add( vec_load(&dst[i +   width]), vec_load(&src[i +   width]) );
        vfloat a2 = vec_add( vec_load(&dst[i + 2*width]), vec_load(&src[i + 2*width]) );
        vfloat a3 = vec_add( vec_load(&dst[i + 3*width]), vec_load(&src[i + 3*width]) );

        vec_store( &dst[i             ], a0 );
        vec_store( &dst[i +   width], a1 );
        vec_store( &dst[i + 2*width], a2 );
        vec_store( &dst[i + 3*width], a3 );
    }

    // remaining full vectors
    for( ; i + width <=n; i += width )
        vec_store( &dst[i], vec_add( vec_load( &dst[i] ), vec_load( &src[i] ) ) );

    // remaining scalars
    for( ; i < n; i++ )
        dst[i] += src[i];
}

#endif

} // namespace simd
