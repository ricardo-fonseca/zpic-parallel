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

// No initialization requrired
#define simd_init()

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

// No initialization requrired
#define simd_init()

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

// No initialization requrired
#define simd_init()

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
constexpr int vecwidth = sve_vec_width;

typedef vec_f32 vfloat;
typedef vec_i32 vint;
typedef vec_mask vmask;

typedef VecFloat  VecFloat_s;
typedef VecInt    VecInt_s;
typedef VecMask   VecMask_s;

// Initialize the SVE vector length
#define simd_init()    prctl(PR_SVE_SET_VL, __ARM_FEATURE_SVE_BITS / 8);

#define SIMD SVE

#endif


/**
 * @brief Disable use of SIMD optimized code
 * 
 */
#ifndef SIMD

constexpr char vecname[] = "none";
constexpr int vecwidth = 1;
#define simd_init()

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

#endif
