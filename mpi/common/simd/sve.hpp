#pragma once

#ifdef __linux__
#include <sys/prctl.h>
#else
#error "ARM SVE support only available on Linux"
#endif

#ifndef __ARM_FEATURE_SVE_BITS
#error "SVE vector length not defined, please use -msve-vector-bits=512 (or other value)"
#endif

#include <arm_sve.h>

#include <iostream>
#include <iomanip>

constexpr int sve_vec_width = __ARM_FEATURE_SVE_BITS / 32;

typedef svfloat32_t vec_f32  __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));
typedef svint32_t   vec_i32  __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));
typedef svuint32_t  vec_u32  __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));
typedef svbool_t    vec_mask __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));

/**
 * @brief SIMD vector unit name
 * 
 */
constexpr char vecname[] = "ARM SVE";
/**
 * @brief Main SIMD vector width
 * 
 */
constexpr int vecwidth = sve_vec_width;

/**
 * @brief float32 vector
 * 
 */
using vfloat = vec_f32;
/**
 * @brief int32 vector
 * 
 */
using vint   = vec_i32;
/**
 * @brief vector mask
 * 
 */
using vmask  = vec_mask;

/**
 * @brief SIMD unit initialization
 * 
 * @warning Currently only supported in Linux
 *
 * @return int 
 */
inline int simd_init() { 
    return prctl(PR_SVE_SET_VL, __ARM_FEATURE_SVE_BITS / 8);
};


/**
 * @brief Extract a single float from a vec_f32 vector
 * 
 * @tparam imm      Which value to extract
 * @param v         Input vector
 * @return float    Selected value
 */
template< int imm > 
inline float vec_extract( const vec_f32 v ) {
    static_assert( imm >= 0 && imm < sve_vec_width, "imm must be in the range [0..vec_width[" );
    return  v[imm];
}

/**
 * @brief Extract a single float from a vec_f32 vector
 * 
 * @param v         Input vector
 * @param i         Element index
 * @return float    Selected value
 */
inline float vec_extract( const vec_f32 v, int i ) {
    return svlastb_f32( svwhilele_b32( 0, i ), v );
}

/**
 * @brief Writes the textual representation of vector v to os
 * 
 * @param os    Output stream
 * @param v     Float vector value
 * @return std::ostream& 
 */
inline std::ostream& operator<<(std::ostream& os, const vec_f32 v) {
    os << "[";
       <<         vec_extract< 0>( v )
       << ", " << vec_extract< 1>( v )
       << ", " << vec_extract< 2>( v )
       << ", " << vec_extract< 3>( v )

#if __ARM_FEATURE_SVE_BITS > 128
       << ", " << vec_extract< 4>( v )
       << ", " << vec_extract< 5>( v )
       << ", " << vec_extract< 6>( v )
       << ", " << vec_extract< 7>( v )
#endif

#if __ARM_FEATURE_SVE_BITS > 256
       << ", " << vec_extract< 8>( v )
       << ", " << vec_extract< 9>( v )
       << ", " << vec_extract<10>( v )
       << ", " << vec_extract<11>( v )
       << ", " << vec_extract<12>( v )
       << ", " << vec_extract<13>( v )
       << ", " << vec_extract<14>( v )
       << ", " << vec_extract<15>( v )
#endif

       << "]";

    return os;
}

/**
 * @brief Returns a zero valued vector
 * 
 * @return vec_f32 
 */
inline vec_f32 vec_zero_float() {
    return svdup_n_f32(0);
}

/**
 * @brief Create a vector with all elements equal to scalar value
 * 
 * @param s 
 * @return vec_f32 
 */
inline vec_f32 vec_float( float s ) {
    return svdup_n_f32(s);
}

/**
 * @brief Create a float vector from an integer vector
 * 
 * @param vi 
 * @return vec_f32 
 */
inline vec_f32 vec_float( vec_i32 vi ) {
    return svcvt_f32_s32_x( svptrue_b32(), vi );
}

/**
 * @brief Create a vector from the scalar elements
 * 
 * @param a
 * @param b 
 * @param c 
 * @param d 
 * @return vec_f32 
 */

#if (__ARM_FEATURE_SVE_BITS==128)
inline vec_f32 vec_float( float f0, float f1, float f2, float f3) {
    return vec_f32{ f0, f1, f2, f3 };
}
#elif (__ARM_FEATURE_SVE_BITS==256)
inline vec_f32 vec_float( 
    float f0, float f1, float f2, float f3,
    float f4, float f5, float f6, float f7)
{
    return vec_f32{ f0, f1, f2, f3, f4, f5, f6, f7 };
}
#elif (__ARM_FEATURE_SVE_BITS==512)
inline vec_f32 vec_float( 
    float  f0, float  f1, float  f2, float  f3,
    float  f4, float  f5, float  f6, float  f7,
    float  f8, float  f9, float f10, float f11,
    float f12, float f13, float f14, float f15)
{
    return vec_f32{ 
        f0,  f1,  f2,  f3,  f4,  f5,  f6,  f7,
        f8,  f9, f10, f11, f12, f13, f14, f15
    };
}
#endif

/**
 * @brief Loads vector from memory
 * 
 * @warning The address must be aligned to a 32 byte boundary
 * 
 * @param mem_addr 
 * @return vec_f32 
 */
inline vec_f32 vec_load( const float * mem_addr) { 
    return svld1_f32( svptrue_b32(), (float32_t *) mem_addr );
}

/**
 * @brief Stores vector to memory
 * 
 * @warning The address must be aligned to a 32 byte boundary
 * 
 * @param mem_addr 
 * @param a 
 */
inline vec_f32 vec_store( float * mem_addr, vec_f32 a ) {
    svst1( svptrue_b32(), mem_addr, a ); return a;
}

/**
 * @brief Negate vector value
 * 
 * @param a         Input value
 * @return vec_f32  Output value (-a)
 */
inline vec_f32 vec_neg( vec_f32 a ) {
    // return svneg_f32_x( svptrue_b32(), a );
    return -a;
}

/**
 * @brief Adds 2 vector values (a+b)
 * 
 * @param a 
 * @param b 
 * @return vec_f32 
 */
inline vec_f32 vec_add( vec_f32 a, vec_f32 b ) {
    // return svadd_f32_x( svptrue_b32(), a, b );
    return a + b;
}

/**
 * @brief Adds scalar value (s) to all components of vector value (a). 
 * 
 * @param a 
 * @param b 
 * @return vec_f32 
 */
inline vec_f32 vec_add( vec_f32 a, float s ) { 
    return svadd_n_f32_x( svptrue_b32(), a, s );
}

/**
 * @brief Subtracts 2 vector values (a-b)
 * 
 * @param a 
 * @param b 
 * @return vec_f32 
 */
inline vec_f32 vec_sub( vec_f32 a, vec_f32 b ) {
    // return svsub_f32_x( svptrue_b32(), a, b );
    return a - b;
}

/**
 * @brief Multiplies 2 vector values (a*b)
 * 
 * @param a 
 * @param b 
 * @return vec_f32 
 */
inline vec_f32 vec_mul( vec_f32 a, vec_f32 b ) { 
    return svmul_f32_x( svptrue_b32(), a, b );
}

/**
 * @brief Multiplies vector (a) by scalar value (s)
 * 
 * @param a 
 * @param s 
 * @return vec_f32 
 */
inline vec_f32 vec_mul( vec_f32 a, float s ) {
    return svmul_n_f32_x( svptrue_b32(), a, s );
}

/**
 * @brief Divides 2 vector values (a/b)
 * 
 * @param a 
 * @param b 
 * @return vec_f32 
 */
inline vec_f32 vec_div( vec_f32 a, vec_f32 b ) {
    return svdiv_f32_x( svptrue_b32(), a, b );
}

/**
 * @brief Compares 2 vector values (by element) for equality (==)
 * 
 * @param a 
 * @param b 
 * @return vec_i32
 */
inline vec_mask vec_eq( vec_f32 a, vec_f32 b ) { 
    return svcmpeq_f32( svptrue_b32(), a, b );
}

/**
 * @brief Compares 2 vector values (by element) for inequality (!=) and return mask
 * 
 * @param a 
 * @param b 
 * @return vec_i32
 */
inline vec_mask vec_ne( vec_f32 a, vec_f32 b ) { 
    return svcmpne_f32( svptrue_b32(), a, b );
}

/**
 * @brief Compares 2 vector values (by element) for "greater-than" (>) and return mask
 * 
 * @param a 
 * @param b 
 * @return vec_mask
 */
inline vec_mask vec_gt( vec_f32 a, vec_f32 b ) { 
    return svcmpgt_f32( svptrue_b32(), a, b );
}

/**
 * @brief Compares 2 vector values (by element) for "greater of equal" (>=) and return mask
 * 
 * @param a 
 * @param b 
 * @return      Resulting mask, for each element i 0 if false, -1 if true
 */
inline vec_mask vec_ge( vec_f32 a, vec_f32 b ) { 
    return svcmpge_f32( svptrue_b32(), a, b );
}

/**
 * @brief Compares 2 vector values (by element) for "greater of equal" (>=) and return v (float)
 * 
 * @param a 
 * @param b 
 * @param v
 * @return      Result, for each element i, 0 if false and v[i] if true
 */
inline vec_f32 vec_ge( vec_f32 a, vec_f32 b, vec_f32 v ) { 

    vec_mask m = svcmpge_f32( svptrue_b32(), a, b );
    return svsel_f32( m, v, svdup_n_f32(0) );
}

/**
 * @brief Compares 2 vector values (by element) for "greater of equal" (>=) and return v (integer)
 * 
 * @param a 
 * @param b 
 * @param vi 
 * @return      Result, for each element i, 0 if false and v[i] if true
 */
inline vec_i32 vec_ge( vec_f32 a, vec_f32 b, vec_i32 vi ) { 
    vec_mask m = svcmpge_f32( svptrue_b32(), a, b );
    return svsel_s32( m, vi, svdup_n_s32(0) );
}


/**
 * @brief Compares 2 vector values (by element) for "less than" (<) and return mask
 * 
 * @param a 
 * @param b 
 * @return      Resulting mask, for each element i 0 if false, -1 if true
 */
inline vec_mask vec_lt( vec_f32 a, vec_f32 b ) { 
    return svcmplt_f32( svptrue_b32(), a, b );
}

/**
 * @brief Compares 2 vector values (by element) for "less than" (<) and return value
 * 
 * @param a     Value a
 * @param b     Value b
 * @param v     Result, for each element i, 0 if false and v[i] if true
 * @return vec_f32 
 */
inline vec_f32 vec_lt( vec_f32 a, vec_f32 b, vec_f32 v ) { 
    vec_mask m = svcmplt_f32( svptrue_b32(), a, b );
    return svsel_f32( m, v, svdup_n_f32(0) );
}

/**
 * @brief Compares 2 vector values (by element) for "less than" (<) and return value (integer)
 * 
 * @param a     Value a
 * @param b     Value b
 * @param v     Result, for each element i, 0 if false and v[i] if true
 * @return vec_f32 
 */

inline vec_i32 vec_lt( vec_f32 a, vec_f32 b, vec_i32 vi ) { 
    vec_mask m = svcmplt_f32( svptrue_b32(), a, b );
    return svsel_s32( m, vi, svdup_n_s32(0) );
}

/**
 * @brief Compares 2 vector values (by element) for "less or equal" (<=)
 * 
 * @param a 
 * @param b 
 * @return vec_f32 
 */
inline vec_mask vec_le( vec_f32 a, vec_f32 b ) { 
    return svcmple_f32( svptrue_b32(), a, b );
}

/**
 * @brief Fused multiply add: (a*b)+c
 * 
 * @param a 
 * @param b 
 * @param c 
 * @return vec_f32 
 */
inline vec_f32 vec_fmadd( vec_f32 a, vec_f32 b, vec_f32 c ) { 
    return svmad_f32_x( svptrue_b32(), a, b, c );
}

/**
 * @brief Fused multiply subtract: (a*b)-c
 * 
 * @param a 
 * @param b 
 * @param c 
 * @return vec_f32 
 */
inline vec_f32 vec_fmsub( vec_f32 a, vec_f32 b, vec_f32 c ) { 
    return svmad_f32_x( svptrue_b32(), a, b, svneg_f32_x( svptrue_b32(), c ));
}

/**
 * @brief Fused negate multiply add: -(a*b)+c
 * 
 * @param a 
 * @param b 
 * @param c 
 * @return vec_f32 
 */
inline vec_f32 vec_fnmadd( vec_f32 a, vec_f32 b, vec_f32 c ) {
    return svmsb_f32_x( svptrue_b32(), a, b, c );
}

/**
 * @brief Reciprocal (1/a)
 * 
 * @param a
 * @return vec_f32 
 */
inline vec_f32 vec_recp( const vec_f32 a )
{
    // Full calculation
    auto recp = svdiv_f32_x( svptrue_b32(), svdup_n_f32(1.0f), a );

/*
    // Fast estimate + 1 Newton-Raphson iteration
    auto recp =  svrecpe_f32( a );
    recp = svmul_f32_x( svptrue_b32(),svrecps_f32( a, recp ), recp );
*/

    return recp;
}

/**
 * @brief Reciprocal square root 1/sqrt(a)
 * 
 * @param a 
 * @return vec_f32 
 */
inline vec_f32 vec_rsqrt( const vec_f32 a ) {

    // Full calculation
    auto rsqrt = svdiv_f32_x( svptrue_b32(), svdup_n_f32(1.0f) , svsqrt_f32_x( svptrue_b32(), a) );

/*
    // Fast estimate + 2 Newton-Raphson iterations
    auto rsqrt = svrsqrte_f32( a );

    rsqrt = svmul_f32_x( svptrue_b32(),svrsqrts_f32( svmul_f32_x( svptrue_b32(), a, rsqrt ), rsqrt ), rsqrt );
    rsqrt = svmul_f32_x( svptrue_b32(),svrsqrts_f32( svmul_f32_x( svptrue_b32(), a, rsqrt ), rsqrt ), rsqrt );
*/

    return rsqrt;
}

/**
 * @brief Square root
 * 
 * @param a 
 * @return vec_f32 
 */
inline vec_f32 vec_sqrt( const vec_f32 a ) {
    return svsqrt_f32_x( svptrue_b32(), a );
}

/**
 * @brief Absolute value
 * 
 * @param a 
 * @return vec_f32 
 */
inline vec_f32 vec_fabs( const vec_f32 a ) { 
    return svabs_f32_x( svptrue_b32(), a );
}

/**
 * @brief Selects between vector elements of vectors a and b according to the mask
 * 
 * @param a     a vector
 * @param b     b vector
 * @param mask  selection mask
 * @return vec_i32 
 */
inline vec_f32 vec_select( const vec_f32 a, const vec_f32 b, const vec_mask mask ) {
    return svsel_f32( mask, b, a );
}


/**
 * @brief Add all vector elements
 * 
 * @param a 
 * @return float 
 */
inline float vec_reduce_add( const vec_f32 a ) {
    return svaddv_f32( svptrue_b32(), a );
}

/**
 * @brief Gather values from base address + vector index
 * 
 * @param base_addr 
 * @param vindex 
 * @return vec_f32 
 */
inline vec_f32 vec_gather( float const * base_addr, vec_i32 vindex ) {
    return svld1_gather_s32index_f32( svptrue_b32(), base_addr, vindex );
}


/**
 * @brief Cody-Waite reduction and core polynomial evaluation (internal)
 *
 * The argument is written as $ x = r + q \pi/2 $, with $ |r| \le \pi/4 $
 * and q integer, using a 4 term split of $ \pi/2 $. $ \sin r $ and
 * $ \cos r $ are then evaluated with Taylor series, both accurate to ~1 ulp
 * on this interval, and the results swapped / sign flipped according to the
 * quadrant.
 *
 * Note that for $ q = 0 $ the reduction is exact (r == x bitwise), so in
 * that case `sinc_r` is $ \sin x / x $ and no division is required.
 *
 * @warning Valid for $ |x| < 2^{24} \pi / 2 \approx 2.6 \times 10^7 $, which
 *          is the point where q can no longer be held exactly in a float. Above
 *          this the reduction fails and the result is meaningless (not NaN, just
 *          wrong); handling it would require a Payne-Hanek reduction
 *
 * @param x         (simd vector) Argument
 * @param s         (simd vector, out) $ \sin x $
 * @param c         (simd vector, out) $ \cos x $
 * @param sinc_r    (simd vector, out) $ \sin r / r $, r being the reduced argument
 * @param q         (simd vector, out) Quadrant index
 */
inline void __sin_cos_kernel( const vec_f32 x, vec_f32 & s, vec_f32 & c,
                            vec_f32 & sinc_r, vec_i32 & q )
{
const svbool_t pg = svptrue_b32();
 
    const vec_f32 PIO2_A = svdup_n_f32( 1.5703125f                 );
    const vec_f32 PIO2_B = svdup_n_f32( 4.8351287841796875e-04f    );
    const vec_f32 PIO2_C = svdup_n_f32( 3.1385570764541626e-07f    );
    const vec_f32 PIO2_D = svdup_n_f32( 6.0771006282767103811e-11f );
 
    const vec_f32 qf = svrintn_f32_x( pg, svmul_n_f32_x( pg, x, 0.636619772367581343f ) );
    q = svcvt_s32_f32_x( pg, qf );
 
    vec_f32 r;
    r = svmsb_f32_x( pg, qf, PIO2_A, x );
    r = svmsb_f32_x( pg, qf, PIO2_B, r );
    r = svmsb_f32_x( pg, qf, PIO2_C, r );
    r = svmsb_f32_x( pg, qf, PIO2_D, r );
 
    const vec_f32 r2 = svmul_f32_x( pg, r, r );
 
    vec_f32 sp = svdup_n_f32( 1.0f/362880 );
    sp = svmad_f32_x( pg, sp, r2, svdup_n_f32( -1.0f/5040 ) );
    sp = svmad_f32_x( pg, sp, r2, svdup_n_f32(  1.0f/120  ) );
    sp = svmad_f32_x( pg, sp, r2, svdup_n_f32( -1.0f/6    ) );
    sp = svmad_f32_x( pg, sp, r2, svdup_n_f32(  1.0f      ) );
 
    vec_f32 cp = svdup_n_f32( -1.0f/3628800 );
    cp = svmad_f32_x( pg, cp, r2, svdup_n_f32(  1.0f/40320 ) );
    cp = svmad_f32_x( pg, cp, r2, svdup_n_f32( -1.0f/720   ) );
    cp = svmad_f32_x( pg, cp, r2, svdup_n_f32(  1.0f/24    ) );
    cp = svmad_f32_x( pg, cp, r2, svdup_n_f32( -1.0f/2     ) );
    cp = svmad_f32_x( pg, cp, r2, svdup_n_f32(  1.0f       ) );
 
    const vec_f32 sin_r = svmul_f32_x( pg, r, sp );
    const vec_f32 cos_r = cp;
 
    const vec_mask swap = svcmpne_n_s32( pg, svand_n_s32_x( pg, q, 1 ), 0 );
    const vec_f32 sin_x = svsel_f32( swap, cos_r, sin_r );
    const vec_f32 cos_x = svsel_f32( swap, sin_r, cos_r );
 
    const vec_i32 sgn_s = svlsl_n_s32_x( pg, svand_n_s32_x( pg, q, 2 ), 30 );
    const vec_i32 sgn_c = svlsl_n_s32_x( pg,
        svand_n_s32_x( pg, svadd_n_s32_x( pg, q, 1 ), 2 ), 30 );
 
    s = svreinterpret_f32_s32( sveor_s32_x( pg, svreinterpret_s32_f32( sin_x ), sgn_s ) );
    c = svreinterpret_f32_s32( sveor_s32_x( pg, svreinterpret_s32_f32( cos_x ), sgn_c ) );
 
    sinc_r = sp;
}

/**
 * @brief Simultaneous evaluation of $ \sin x $ and $ \cos x $
 *
 * Absolute error below $ 10^{-7} $ for both outputs, x of either sign. See
 * `__sin_cos_kernel()` for the algorithm and the range limit.
 *
 * @param x     (simd vector) Argument
 * @param s     (simd vector, out) $ \sin x $
 * @param c     (simd vector, out) $ \cos x $
 */
inline void vec_sin_cos( const vec_f32 x, vec_f32 & s, vec_f32 & c )
{
    vec_f32 sinc_r;
    vec_i32 q;
    __sin_cos_kernel( x, s, c, sinc_r, q );
}

/**
 * @brief Simultaneous evaluation of sinc(x) = sin(x)/x and cos(x)
 *
 * For $ |x| \le \pi/4 $ the argument needs no reduction (q == 0) and the
 * core polynomial already is $ \sin x / x $, so it is used directly. This
 * avoids two roundings (the multiply by r and the division), and since x == 0
 * implies q == 0 it also removes the need to special case the origin, where the
 * polynomial evaluates to exactly 1.
 *
 * @param x     (simd vector) Argument
 * @param s     (simd vector, out) sin(x) / x
 * @param c     (simd vector, out) cos(x)
 */
inline void vec_sinc_cos( const vec_f32 x, vec_f32 & s, vec_f32 & c )
{
    const svbool_t pg = svptrue_b32();
 
    vec_f32 sin_x, sinc_r;
    vec_i32 q;
    __sin_cos_kernel( x, sin_x, c, sinc_r, q );
 
    s = svsel_f32( svcmpeq_n_s32( pg, q, 0 ), sinc_r, svdiv_f32_x( pg, sin_x, x ) );
}


/**
 * @brief Integer (32 bit) SIMD types
 */

/**
 * @brief Extract a single integer from a vec_i32 vector
 * 
 * @tparam imm      Which value to extract
 * @param v         Input vector
 * @return int      Selected value
 */
template< int imm > 
inline int vec_extract( const vec_i32 v ) {
    static_assert( imm >= 0 && imm < sve_vec_width, "imm must be in the range [0..vec_width[" );
    return  v[imm];
}

/**
 * @brief Extract a single integer from a vec_i32 vector
 * 
 * @param v         Input vector
 * @param i         Which value to extract
 * @return int      Selected value
 */
inline int vec_extract( const vec_i32 v, int i ) {
    return svlastb_s32( svwhilele_b32( 0, i ), v );
}

/**
 * @brief Writes the textual representation of vector v to os
 * 
 * @param os    Output stream
 * @param v     int vector value
 * @return std::ostream& 
 */
inline std::ostream& operator<<(std::ostream& os, const vec_i32 v) {
    os << "[";
    os <<         vec_extract<0>( v );
    os << ", " << vec_extract<1>( v );
    os << ", " << vec_extract<2>( v );
    os << ", " << vec_extract<3>( v );


#if __ARM_FEATURE_SVE_BITS > 128
    os << ", " << vec_extract<4>( v );
    os << ", " << vec_extract<5>( v );
    os << ", " << vec_extract<6>( v );
    os << ", " << vec_extract<7>( v );
#endif

#if __ARM_FEATURE_SVE_BITS > 256
    os << ", " << vec_extract<8>( v );
    os << ", " << vec_extract<9>( v );
    os << ", " << vec_extract<10>( v );
    os << ", " << vec_extract<11>( v );
    os << ", " << vec_extract<12>( v );
    os << ", " << vec_extract<13>( v );
    os << ", " << vec_extract<14>( v );
    os << ", " << vec_extract<15>( v );
#endif

    os << "]";

    return os;
}

/**
 * @brief Returns a zero valued vector
 * 
 * @return vec_i32 
 */
inline vec_i32 vec_zero_int() {
    return svdup_n_s32(0);
}

/**
 * @brief Create a vector with all elements equal to scalar value
 * 
 * @param a 
 * @return vec_i32 
 */
inline vec_i32 vec_int( int s ) {
    return svdup_n_s32(s);
}

/**
 * @brief Create a vector from the scalar elements
 * 
 * @param a 
 * @param b 
 * @param c 
 * @param d 
 * @return vec_i32 
 */
#if (__ARM_FEATURE_SVE_BITS==128)
inline vec_i32 vec_int( int i0, int i1, int i2, int i3) {
    return vec_i32{ i0, i1, i2, i3 };
}
#elif (__ARM_FEATURE_SVE_BITS==256)
inline vec_i32 vec_int( 
    int i0, int i1, int i2, int i3,
    int i4, int i5, int i6, int i7)
{
    return vec_i32{ i0, i1, i2, i3, i4, i5, i6, i7 };
}
#elif (__ARM_FEATURE_SVE_BITS==512)
inline vec_i32 vec_int( 
    int i0, int i1, int i2, int i3,
    int i4, int i5, int i6, int i7,
    int i8, int i9, int i10, int i11,
    int i12, int i13, int i14, int i15)
{
    return vec_i32{ i0, i1, i2, i3, i4, i5, i6, i7,
                    i8, i9, i10, i11, i12, i13, i14, i15 };
}
#endif

/**
 * @brief Loads vector from memory
 * 
 * @warning The address must be aligned to a 32 byte boundary
 * 
 * @param mem_addr 
 * @return vec_i32 
 */
inline vec_i32 vec_load( const int * mem_addr) { 
    return svld1_s32( svptrue_b32(), mem_addr );
}

/**
 * @brief Stores vector to memory
 * 
 * @warning The address must be aligned to a 32 byte boundary
 * 
 * @param mem_addr 
 * @param a 
 */
inline vec_i32 vec_store( int * mem_addr, vec_i32 a ) { 
    svst1_s32( svptrue_b32(), mem_addr, a ); return a;
}

/**
 * @brief Adds 2 vector values (a+b)
 * 
 * @param a 
 * @param b 
 * @return vec_i32 
 */
inline vec_i32 vec_add( vec_i32 a, vec_i32 b ) { 
    return a+b;
}

/**
 * @brief Subtracts 2 vector values (a-b)
 * 
 * @param a 
 * @param b 
 * @return vec_i32 
 */
inline vec_i32 vec_sub( vec_i32 a, vec_i32 b ) {
    return a-b;
}

/**
 * @brief Adds scalar value (s) to all components of vector value (a). 
 * 
 * @param a 
 * @param b 
 * @return vec_i32 
 */
inline vec_i32 vec_add( vec_i32 a, int s ) { 
    return svadd_n_s32_x( svptrue_b32(), a, s );
}

/**
 * @brief Multiplies 2 vector values (a*b)
 * 
 * @param a 
 * @param b 
 * @return vec_i32 
 */
inline vec_i32 vec_mul( vec_i32 a, vec_i32 b ) {
    return svmul_s32_x( svptrue_b32(), a, b );
}

/**
 * @brief Multiplies vector (a) by scalar value (s)
 * 
 * @param a 
 * @param b 
 * @return vec_i32 
 */
inline vec_i32 vec_mul( vec_i32 a, int s ) { 
    return svmul_n_s32_x( svptrue_b32(), a, s );
}

/**
 * @brief Multiplies vector by 3
 * 
 * @param a 
 * @return vec_i32 
 */
inline vec_i32 vec_mul3( vec_i32 a ) {
    // return a*3;
    return a + a + a;
    // return svmul_n_s32_x( svptrue_b32(), a, 3 );
}

/**
 * @brief Compares 2 vector values (by element) for equality (==)
 * 
 * @param a 
 * @param b 
 * @return vec_i32 
 */
inline vec_mask vec_eq( vec_i32 a, vec_i32 b ) { 
    return svcmpeq_s32( svptrue_b32(), a, b );
}

/**
 * @brief Compares 2 vector values (by element) for inequality (!=)
 * 
 * @param a 
 * @param b 
 * @return vec_i32 
 */
inline vec_mask vec_ne( vec_i32 a, vec_i32 b ) { 
    return svcmpne_s32( svptrue_b32(), a, b );
}

/**
 * @brief Compares 2 vector values (by element) for "greater-than" (>)
 * 
 * @param a 
 * @param b 
 * @return vec_i32 
 */
inline vec_mask vec_gt( vec_i32 a, vec_i32 b ) { 
    return svcmpgt_s32( svptrue_b32(), a, b );
}

/**
 * @brief Compares 2 vector values (by element) for "less-than" (<)
 * 
 * @param a 
 * @param b 
 * @return vec_i32 
 */
inline vec_mask vec_lt( vec_i32 a, vec_i32 b ) { 
    return svcmplt_s32( svptrue_b32(), a, b );
}
/**
 * @brief Absolute value
 * 
 * @param a 
 * @return vec_i32 
 */
inline vec_i32 vec_abs( vec_i32 a ) {
    return svabs_s32_x( svptrue_b32(), a );
}

/**
 * @brief Selects between vector elements of vectors a and b according to the mask
 * 
 * @param a     a vector
 * @param b     b vector
 * @param mask  selection mask, 0 selects a vector element, -1 selects b vector element
 * @return vec_i32 
 */
inline vec_i32 vec_select( const vec_i32 a, const vec_i32 b, const vec_mask mask ) {
    return svsel_s32( mask, b, a );
}

/**
 * @brief Mask functions
 * 
 */


/**
 * @brief Bitwise complement (not)
 * 
 * @param a 
 * @return vec_mask 
 */
inline vec_mask vec_not( vec_mask a ) {
    return svnot_b_z(svptrue_b32(), a);
}

/**
 * @brief Bitwise or 
 * 
 * @param a 
 * @param b 
 * @return vec_mask 
 */
inline vec_mask vec_or( vec_mask a, vec_mask b ) {
    return svorr_b_z( svptrue_b32(), a, b );
}

/**
 * @brief Bitwise and 
 * 
 * @param a 
 * @param b 
 * @return vec_mask 
 */
inline vec_mask vec_and( vec_mask a, vec_mask b ) {
    return svand_b_z( svptrue_b32(), a, b );
}


/**
 * @brief Returns true (1) if all of the mask values are true
 * 
 * @param mask 
 * @return int 
 */
inline int vec_all( const vec_mask mask ) {

    return ! svptest_any( svptrue_b32(), svnot_b_z(svptrue_b32(), mask) );

}

/**
 * @brief Returns true (1) if any of the mask values is true
 * 
 * @param mask 
 * @return int 
 */
inline int vec_any( const vec_mask mask ) {
    return svptest_any( svptrue_b32(), mask );
}

/**
 * @brief Returns a mask of all true values
 * 
 * @return vec_mask 
 */
inline vec_mask vec_true() { 
    return svptrue_b32();
}

/**
 * @brief Returns a mask of all false values
 * 
 * @return vec_mask 
 */
inline vec_mask vec_false() { 
    return svpfalse_b();
}

/**
 * @brief Extract a single mask bit from a vec_mask
 * 
 * @param mask      Input vec_mask
 * @param i         Which value to extract
 * @return int      Selected value 
 */
inline int vec_extract( const vec_mask mask, int i ) {
    vec_i32 v = svsel_s32( mask, svdup_n_s32(1), svdup_n_s32(0) );
    return svlastb_s32( svwhilele_b32( 0, i ), v );
}

/**
 * @brief Extract a single mask bit from a vec_mask
 * 
 * @tparam imm      Which value to extract
 * @param mask      Input vec_mask
 * @return int      Selected value 
 */
template< int imm > 
inline int vec_extract( const vec_mask mask ) {
    static_assert( imm >= 0 && imm < sve_vec_width, "imm must be in the range [0..vec_width[" );
    vec_i32 v = svsel_s32( mask, svdup_n_s32(1), svdup_n_s32(0) );
    return  v[imm];
}

/**
 * @brief Writes the textual representation of vector v to os
 * 
 * @param os    Output stream
 * @param v     int vector value
 * @return std::ostream& 
 */
inline std::ostream& operator<<(std::ostream& os, const vec_mask mask) {
    vec_i32 v = svsel_s32( mask, svdup_n_s32(1), svdup_n_s32(0) );

    os << "[";
    os << v[0];
    os << v[1];
    os << v[2];
    os << v[3];

#if __ARM_FEATURE_SVE_BITS > 128
    os << v[4];
    os << v[5];
    os << v[6];
    os << v[7];
#endif

#if __ARM_FEATURE_SVE_BITS > 256
    os << v[8];
    os << v[9];
    os << v[10];
    os << v[11];
    os << v[12];
    os << v[13];
    os << v[14];
    os << v[15];
#endif

    os << "]";

    return os;
}

/**
 * @brief Vector version of the float2 type holding 2 (.x, .y) vectors
 * 
 */
struct alignas(vec_f32) vfloat2 {
    vec_f32 x, y;
};

/**
 * @brief Returs a zero valued vfloat2
 * 
 * @return vfloat2 
 */
inline vfloat2 vfloat2_zero( ) {
    vfloat2 v{ vec_zero_float(), vec_zero_float() };
    return v;
}

/**
 * @brief Loads 2-element structure from memory
 * 
 * @note Data is loaded sequentially and de-interleaved into 2 vectors
 * 
 * @param addr 
 * @return vfloat2 
 */
inline vfloat2 vec_load_s2( const float * addr ) {
    
    svfloat32x2_t tmp = svld2_f32(svptrue_b32(),(float32_t const *) addr );
    return vfloat2{ svget2_f32(tmp,0), svget2_f32(tmp,1) };
}

/**
 * @brief Stores 2-element structure to memory
 * 
 * @note Data is interleaved from 2 vectors and stored sequentially
 * 
 * @param addr 
 * @param v 
 */
inline void vec_store_s2( float * addr, const vfloat2 v ) {
    svfloat32x2_t tmp = svcreate2_f32( v.x, v.y );
    svst2( svptrue_b32(),(float32_t *) addr, tmp );

}

/**
 * @brief Vector version of the float3 type holding 3 (.x, .y, .z) vectors
 * 
 */
struct alignas(vec_f32) vfloat3 {
    vec_f32 x, y, z;
};

/**
 * @brief Returs a zero valued vfloat3
 * 
 * @return vfloat3 
 */
inline vfloat3 vfloat3_zero( ) {
    vfloat3 v{ vec_zero_float(), vec_zero_float(), vec_zero_float() };
    return v;
}

/**
 * @brief Loads 3-element structure from memory
 * 
 * @note Data is loaded sequentially and de-interleaved into 3 vectors
 * 
 * @param addr 
 * @return vfloat3 
 */
inline vfloat3 vec_load_s3( const float * addr ) {
    svfloat32x3_t tmp = svld3_f32(svptrue_b32(),(float32_t const *) addr );
    return vfloat3{ svget3_f32(tmp,0), svget3_f32(tmp,1), svget3_f32(tmp,2) };
}

/**
 * @brief Stores 3-element structure to memory
 * 
 * @note Data is interleaved from 3 vectors and stored sequentially
 * 
 * @param addr 
 * @param v 
 */
inline void vec_store_s3( float * addr, const vfloat3 v ) {
    svfloat32x3_t tmp = svcreate3_f32( v.x, v.y, v.z );
    svst3( svptrue_b32(),(float32_t *) addr, tmp );
}

/**
 * @brief Vector version of the int2 type holding 2 (.x, .y) vectors
 * 
 */
struct alignas(vec_i32) vint2 {
    vec_i32 x, y;
};

/**
 * @brief Loads 2-element structure from memory
 * 
 * @note Data is loaded sequentially and de-interleaved into 2 vectors
 * 
 * @param addr 
 * @return vint2 
 */
inline vint2 vec_load_s2( int const * addr ) {

    svint32x2_t tmp = svld2_s32(svptrue_b32(),(int32_t const *) addr );
    return vint2{ svget2_s32(tmp,0), svget2_s32(tmp,1) };
}

/**
 * @brief Stores 2-element structure to memory
 * 
 * @note Data is interleaved from 2 vectors and stored sequentially
 * 
 * @param addr 
 * @param v 
 */
inline void vec_store_s2( int * addr, const vint2 v ) {

    svint32x2_t tmp = svcreate2_s32( v.x, v.y );
    svst2( svptrue_b32(),(int32_t *) addr, tmp );
}

/**
 * @brief Returns a vint2 with 0 components
 * 
 * @return vint2 
 */
inline vint2 vint2_zero( ) {
    vint2 v{ vec_zero_int(), vec_zero_int() };
    return v;
}

/**
 * @brief Vector version of the vec_mask type holding 2 (.x, .y) masks
 * 
 */
struct alignas(vec_mask) vmask2 {
    vec_mask x, y;
};

