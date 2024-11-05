#ifndef SIMD_H_
#define SIMD_H_

#include <string>

/**
 * @brief Enable use of x86 AVX2 optimized code
 * 
 */
#ifdef USE_AVX2

#ifdef SIMD
#error SIMD has already been defined, only 1 type of SIMD code should be enabled.
#endif

#include "avx2.h"

constexpr char vecname[] = "x86_64 AVX2";
constexpr int vecwidth = 8;
typedef __m256 vfloat;
typedef __m256i vint;
typedef __m256i vmask;
typedef Vec8Float VecFloat_s;
typedef Vec8Int VecInt_s;
typedef Vec8Mask VecMask_s;

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

constexpr char vecname[] = "x86_64 AVX512f";
constexpr int vecwidth = 16;
typedef __m512 vfloat;
typedef __m512i vint;
typedef __mmask16 vmask;
typedef Vec16Float VecFloat_s;
typedef Vec16Int VecInt_s;
typedef Vec16Mask VecMask_s;

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

#include "neon.h"

constexpr char vecname[] = "ARM NEON";
constexpr int vecwidth = 4;
typedef vec_f32 vfloat;
typedef vec_i32 vint;
typedef vec_mask32 vmask;

typedef Vec4Float  VecFloat_s;
typedef Vec4Int    VecInt_s;
typedef Vec4Mask   VecMask_s;

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

constexpr char vecname[] = "ARM SVE";

typedef vec_f32 vfloat;
typedef vec_i32 vint;
typedef vec_mask vmask;

typedef VecFloat  VecFloat_s;
typedef VecInt    VecInt_s;
typedef VecMask   VecMask_s;

#define SIMD SVE

#endif


/**
 * @brief Disable use of SIMD optimized code
 * 
 */
#ifndef SIMD

constexpr char vecname[] = "none";
constexpr int vecwidth = 1;

#endif

/**
 * Memory alingment routines
 */


/**
 * @brief Check if address is 64 bit aligned
 * 
 * @tparam T 
 * @param addr  Address to check 
 * @return int 
 */
template< typename T >
int is_aligned_64( T * addr ) {
    return (((uintptr_t)addr & 0x3F) == 0);
}

/**
 * @brief Assert that address is 64 bit aligned
 * 
 * @tparam T 
 * @param addr  Address to check 
 * @param msg 
 */
template< typename T >
void assert_aligned_64( T * addr, std::string msg ) { 
    if ( ! is_aligned_64(addr) ) {
        std::cerr << msg << '\n';
        std::cerr << "Address " << addr << " is not 64 bit aligned, aborting." << std::endl;
        abort();
    }
}

/**
 * @brief Check if address is 32 bit aligned
 * 
 * @tparam T 
 * @param addr  Address to check
 * @return constexpr int 
 */
template< typename T >
constexpr int is_aligned_32( T * addr ) { 
    return (((uintptr_t)addr & 0x1F) == 0);
}

/**
 * @brief Assert that address is 32 bit aligned
 * 
 * @tparam T 
 * @param addr  Address to check
 * @param msg 
 */
template< typename T >
void assert_aligned_32( T * addr, std::string msg ) { 
    if ( ! is_aligned_32(addr) ) {
        std::cerr << msg << '\n';
        std::cerr << "Address " << addr << " is not 32 bit aligned, aborting." << std::endl;
        abort();
    }
}

/**
 * @brief Assert that address is 16 bit aligned
 * 
 * @tparam T 
 * @param addr  Address to check
 * @return constexpr int 
 */
template< typename T >
constexpr int is_aligned_16( T * addr ) {
    return (((uintptr_t)addr & 0x0F) == 0);
}

/**
 * @brief Assert that address is 16 bit aligned
 * 
 * @tparam T 
 * @param addr  Address to check
 * @param msg   Message to print if the assert fails
 */
template< typename T >
void assert_aligned_16( T * addr, std::string msg ) { 
    if ( ! is_aligned_16(addr) ) {
        std::cerr << msg << '\n';
        std::cerr << "Address " << addr << " is not 16 bit aligned, aborting." << std::endl;
        abort();
    }
}


#endif
