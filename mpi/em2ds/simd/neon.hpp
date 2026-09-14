#pragma once

#include <arm_neon.h>
#include <cstdint>
#include <iostream>

/**
 * @brief ARM NEON SIMD vectors
 * 
 */
using vec_f32    = float32x4_t;
using vec_i32    = int32x4_t;
using vec_mask32 = uint32x4_t;

/**
 * @brief Extract a single float from a vec_f32 vector
 * 
 * @tparam imm      Which value to extract
 * @param v         Input vector
 * @return float    Selected value
 */
template< int imm > 
inline float vec_extract( const vec_f32 v ) {
    static_assert( imm >= 0 && imm < 4, "imm must be in the range [0..4]" );
    return vgetq_lane_f32( v, imm );
}

/**
 * @brief Extract a single float from a vec_f32 vector
 * 
 * @param v         Input vector
 * @param i         Element index
 * @return float    Selected value
 */
inline float vec_extract( const vec_f32 v, int i ) {
    return v[i];
}

/**
 * @brief Stream extraction
 * 
 * @param os    Output stream
 * @param v     Float vector value
 * @return std::ostream& 
 */
inline std::ostream& operator<<(std::ostream& os, const vec_f32 v) {
    os << "["
       <<         vec_extract<0>( v )
       << ", " << vec_extract<1>( v )
       << ", " << vec_extract<2>( v )
       << ", " << vec_extract<3>( v )
       << "]";

    return os;
}

/**
 * @brief Returns a zero valued vector
 * 
 * @return vec_f32 
 */
inline vec_f32 vec_zero_float() {
    return vdupq_n_f32(0);
}

/**
 * @brief Create a vector with all elements equal to scalar value
 * 
 * @param s 
 * @return vec_f32 
 */
inline vec_f32 vec_float( float s ) {
    return vdupq_n_f32(s);
}

/**
 * @brief Create a float vector from an integer vector
 * 
 * @param vi 
 * @return vec_f32 
 */
inline vec_f32 vec_float( vec_i32 vi ) {
    return vcvtq_f32_s32( vi );
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
inline vec_f32 vec_float( float a, float b, float c, float d ) {
    return vec_f32{ a, b, c, d };
}

/**
 * @brief Loads vector from memory
 * 
 * @warning The address must be aligned to a 32 byte boundary
 * 
 * @param mem_addr 
 * @return vec_f32 
 */
inline vec_f32 vec_load( const float * mem_addr) { 
    return vld1q_f32( (float32_t *) mem_addr );
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
    vst1q_f32( mem_addr, a ); return a;
}

inline vec_f32 vec_neg( vec_f32 a ) {
    return vnegq_f32( a );
}

/**
 * @brief Adds 2 vector values (a+b)
 * 
 * @param a 
 * @param b 
 * @return vec_f32 
 */
inline vec_f32 vec_add( vec_f32 a, vec_f32 b ) {
    return a+b;
}

/**
 * @brief Adds scalar value (s) to all components of vector value (a). 
 * 
 * @param a 
 * @param b 
 * @return vec_f32 
 */
inline vec_f32 vec_add( vec_f32 a, float s ) { 
    return a + vdupq_n_f32(s);
}

/**
 * @brief Subtracts 2 vector values (a-b)
 * 
 * @param a 
 * @param b 
 * @return vec_f32 
 */
inline vec_f32 vec_sub( vec_f32 a, vec_f32 b ) {
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
    return a * b;
}

/**
 * @brief Multiplies vector (a) by scalar value (s)
 * 
 * @param a 
 * @param s 
 * @return vec_f32 
 */
inline vec_f32 vec_mul( vec_f32 a, float s ) {
    return vmulq_n_f32(a, s);
}

/**
 * @brief Divides 2 vector values (a/b)
 * 
 * @param a 
 * @param b 
 * @return vec_f32 
 */
inline vec_f32 vec_div( vec_f32 a, vec_f32 b ) {
    return a / b;
}

/**
 * @brief Compares 2 vector values (by element) for equality (==)
 * 
 * @param a 
 * @param b 
 * @return vec_i32
 */
inline vec_mask32 vec_eq( vec_f32 a, vec_f32 b ) { 
    return vceqq_f32( a, b );
}

/**
 * @brief Compares 2 vector values (by element) for inequality (!=)
 * 
 * @param a 
 * @param b 
 * @return vec_i32
 */
inline vec_mask32 vec_ne( vec_f32 a, vec_f32 b ) { 
    return vmvnq_u32( vceqq_f32( a, b ) );
}

/**
 * @brief Compares 2 vector values (by element) for "greater-than" (>)
 * 
 * @param a 
 * @param b 
 * @return vec_f32 
 */
inline vec_mask32 vec_gt( vec_f32 a, vec_f32 b ) { 
    return vcgtq_f32( a, b );
}

/**
 * @brief Compares 2 vector values (by element) for "greater of equal" (>=) and return mask
 * 
 * @param a 
 * @param b 
 * @return      Resulting mask, for each element i 0 if false, -1 if true
 */
inline vec_mask32 vec_ge( vec_f32 a, vec_f32 b ) { 
    return vcgeq_f32( a, b );
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

    return vreinterpretq_f32_u32( 
        vandq_u32( vcgeq_f32( a, b ), vreinterpretq_u32_f32(v)) );
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
    return  vreinterpretq_s32_u32( vandq_u32( vcgeq_f32( a, b ), vreinterpretq_u32_s32(vi) ) );
}


/**
 * @brief Compares 2 vector values (by element) for "less than" (<) and return mask
 * 
 * @param a 
 * @param b 
 * @return      Resulting mask, for each element i 0 if false, -1 if true
 */
inline vec_mask32 vec_lt( vec_f32 a, vec_f32 b ) { 
    return vcltq_f32( a, b );
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
    return vreinterpretq_f32_u32( 
        vandq_u32( vcltq_f32( a, b ), vreinterpretq_u32_f32(v)) );
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
    return  vreinterpretq_s32_u32( 
        vandq_u32( vcltq_f32( a, b ), vreinterpretq_u32_s32(vi) ) );
}

/**
 * @brief Compares 2 vector values (by element) for "less or equal" (<=)
 * 
 * @param a 
 * @param b 
 * @return vec_f32 
 */
inline vec_mask32 vec_le( vec_f32 a, vec_f32 b ) { 
    return vcleq_f32( a, b );
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
    return vmlaq_f32( c, b, a );
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
    return vmlaq_f32( c, b, vnegq_f32(a) );
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
    return vmlsq_f32( c, b, a );
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
    auto recp = vdupq_n_f32(1.0f) / a;

/*
    // Fast estimate + 2 Newton-Raphson iterations
    auto recp =  vrecpeq_f32( a );
   // 2 iterations are required for full precision
   recp = vrecpsq_f32( a, recp ) * recp;
   recp = vrecpsq_f32( a, recp ) * recp;
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
    auto rsqrt = vdupq_n_f32(1.0f) / vsqrtq_f32(a);

/*
   // Fast estimate + 2 Newton-Raphson iterations
   auto rsqrt = vrsqrteq_f32( a );

   // 2 iterations are required for full precision
   rsqrt = vrsqrtsq_f32( a * rsqrt, rsqrt ) * rsqrt;
   rsqrt = vrsqrtsq_f32( a * rsqrt, rsqrt ) * rsqrt;

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
    return vsqrtq_f32(a);
}

/**
 * @brief Absolute value
 * 
 * @param a 
 * @return vec_f32 
 */
inline vec_f32 vec_fabs( const vec_f32 a ) { 
    return vabsq_f32( a );
}

/**
 * @brief Selects between vector elements of vectors a and b according to the mask
 * 
 * @param a     a vector
 * @param b     b vector
 * @param mask  selection mask, 0 selects a vector element, -1 selects b vector element
 * @return vec_i32 
 */
inline vec_f32 vec_select( const vec_f32 a, const vec_f32 b, const vec_mask32 mask ) {
    return vbslq_f32( mask, b, a );
}


/**
 * @brief Add all vector elements
 * 
 * @param a 
 * @return float 
 */
inline float vec_reduce_add( const vec_f32 a ) {
    auto tmp = vpaddq_f32(a,a);
    return vgetq_lane_f32(tmp,0) + vgetq_lane_f32(tmp,1);
}

/**
 * @brief Gather values from base address + vector index
 * 
 * @param base_addr 
 * @param vindex 
 * @return vec_f32 
 */
inline vec_f32 vec_gather( float const * base_addr, vec_i32 vindex ) {

    vec_f32 v;

    v = vld1q_dup_f32(  base_addr + vgetq_lane_s32( vindex, 0 ) );
    v = vld1q_lane_f32( base_addr + vgetq_lane_s32( vindex, 1 ), v, 1 );
    v = vld1q_lane_f32( base_addr + vgetq_lane_s32( vindex, 2 ), v, 2 );
    v = vld1q_lane_f32( base_addr + vgetq_lane_s32( vindex, 3 ), v, 3 );

    return v;
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
    // Cody-Waite split of π/2. Every constant is exactly representable in
    // single precision (trailing mantissa bits are 0) so that no significance
    // is lost in the reduction below
    const vec_f32 PIO2_A = vdupq_n_f32( 1.5703125f                 );
    const vec_f32 PIO2_B = vdupq_n_f32( 4.8351287841796875e-04f    );
    const vec_f32 PIO2_C = vdupq_n_f32( 3.1385570764541626e-07f    );
    const vec_f32 PIO2_D = vdupq_n_f32( 6.0771006282767103811e-11f );
 
    // Argument reduction, x = r + q π/2 with |r| ≤ π/4
    const vec_f32 qf = vrndnq_f32( vmulq_n_f32( x, 0.636619772367581343f ) );
    q = vcvtq_s32_f32( qf );
 
    vec_f32 r;
    r = vfmsq_f32( x, qf, PIO2_A );
    r = vfmsq_f32( r, qf, PIO2_B );
    r = vfmsq_f32( r, qf, PIO2_C );
    r = vfmsq_f32( r, qf, PIO2_D );
 
    const vec_f32 r2 = vmulq_f32( r, r );
 
    // sin(r)/r , |r| ≤ π/4
    vec_f32 sp = vdupq_n_f32( 1.0f/362880 );
    sp = vfmaq_f32( vdupq_n_f32( -1.0f/5040 ), sp, r2 );
    sp = vfmaq_f32( vdupq_n_f32(  1.0f/120  ), sp, r2 );
    sp = vfmaq_f32( vdupq_n_f32( -1.0f/6    ), sp, r2 );
    sp = vfmaq_f32( vdupq_n_f32(  1.0f      ), sp, r2 );
 
    // cos(r) , |r| ≤ π/4
    vec_f32 cp = vdupq_n_f32( -1.0f/3628800 );
    cp = vfmaq_f32( vdupq_n_f32(  1.0f/40320 ), cp, r2 );
    cp = vfmaq_f32( vdupq_n_f32( -1.0f/720   ), cp, r2 );
    cp = vfmaq_f32( vdupq_n_f32(  1.0f/24    ), cp, r2 );
    cp = vfmaq_f32( vdupq_n_f32( -1.0f/2     ), cp, r2 );
    cp = vfmaq_f32( vdupq_n_f32(  1.0f       ), cp, r2 );
 
    const vec_f32 sin_r = vmulq_f32( r, sp );
    const vec_f32 cos_r = cp;
 
    // Odd quadrants swap sin and cos
    const uint32x4_t swap = vtstq_s32( q, vdupq_n_s32(1) );
    const vec_f32 sin_x = vbslq_f32( swap, cos_r, sin_r );
    const vec_f32 cos_x = vbslq_f32( swap, sin_r, cos_r );
 
    // Quadrant sign: bit 1 of q (resp. q+1) moved onto the sign bit
    const uint32x4_t sgn_s = vshlq_n_u32(
        vandq_u32( vreinterpretq_u32_s32( q ), vdupq_n_u32(2) ), 30 );
    const uint32x4_t sgn_c = vshlq_n_u32(
        vandq_u32( vreinterpretq_u32_s32( vaddq_s32( q, vdupq_n_s32(1) ) ), vdupq_n_u32(2) ), 30 );
 
    s = vreinterpretq_f32_u32( veorq_u32( vreinterpretq_u32_f32( sin_x ), sgn_s ) );
    c = vreinterpretq_f32_u32( veorq_u32( vreinterpretq_u32_f32( cos_x ), sgn_c ) );
 
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
    vec_f32 sin_x, sinc_r;
    vec_i32 q;
    __sin_cos_kernel( x, sin_x, c, sinc_r, q );
 
    s = vbslq_f32( vceqzq_s32( q ), sinc_r, vdivq_f32( sin_x, x ) );
}

/**
 * @brief Integer (32 bit) SIMD types
 * 
 */

/**
 * @brief Extract a single integer from a vec_i32 vector
 * 
 * @tparam imm      Element index
 * @param v         Input vector
 * @return float    Selected value
 */

template< int imm > 
inline int vec_extract( const vec_i32 v ) {
    static_assert( imm >= 0 && imm < 4, "imm must be in the range [0..3]" );
    return vgetq_lane_s32( v, imm );
}

/**
 * @brief Extract a single integer from a vec_i32 vector
 * 
 * @param v         Input vector
 * @param i         Element index
 * @return float    Selected value
 */
inline int vec_extract( const vec_i32 v, int i ) {
   return v[i];
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
    os << "]";

    return os;
}

/**
 * @brief Returns a zero valued vector
 * 
 * @return vec_i32 
 */
inline vec_i32 vec_zero_int() {
    return vdupq_n_s32(0);
}

/**
 * @brief Create a vector with all elements equal to scalar value
 * 
 * @param a 
 * @return vec_i32 
 */
inline vec_i32 vec_int( int s ) {
    return vdupq_n_s32(s);
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
inline vec_i32 vec_int( int a, int b, int c, int d ){ 
    return vec_i32{ a, b, c, d };
}

/**
 * @brief Loads vector from memory
 * 
 * @warning The address must be aligned to a 32 byte boundary
 * 
 * @param mem_addr 
 * @return vec_i32 
 */
inline vec_i32 vec_load( const int * mem_addr) { 
    return vld1q_s32( mem_addr );
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
    vst1q_s32( mem_addr, a ); return a;
}

/**
 * @brief Adds 2 vector values (a+b)
 * 
 * @param a 
 * @param b 
 * @return vec_i32 
 */
inline vec_i32 vec_add( vec_i32 a, vec_i32 b ) { 
    return a + b;
}

/**
 * @brief Subtracts 2 vector values (a-b)
 * 
 * @param a 
 * @param b 
 * @return vec_i32 
 */
inline vec_i32 vec_sub( vec_i32 a, vec_i32 b ) {
    return a - b;
}

/**
 * @brief Adds scalar value (s) to all components of vector value (a). 
 * 
 * @param a 
 * @param b 
 * @return vec_i32 
 */
inline vec_i32 vec_add( vec_i32 a, int s ) { 
    return a + vdupq_n_s32(s);
}

/**
 * @brief Multiplies 2 vector values (a*b)
 * 
 * @param a 
 * @param b 
 * @return vec_i32 
 */
inline vec_i32 vec_mul( vec_i32 a, vec_i32 b ) {
    return a * b;
}

/**
 * @brief Multiplies vector (a) by scalar value (s)
 * 
 * @param a 
 * @param b 
 * @return vec_i32 
 */
inline vec_i32 vec_mul( vec_i32 a, int s ) { 
    return vmulq_n_s32( a, s );
}

/**
 * @brief Multiplies vector by 3
 * 
 * @param a 
 * @return vec_i32 
 */
inline vec_i32 vec_mul3( vec_i32 a ) {
    return a + a + a;
}

/**
 * @brief Compares 2 vector values (by element) for equality (==)
 * 
 * @param a 
 * @param b 
 * @return vec_i32 
 */
inline vec_mask32 vec_eq( vec_i32 a, vec_i32 b ) { 
    return vceqq_s32( a, b );
}

/**
 * @brief Compares 2 vector values (by element) for inequality (!=)
 * 
 * @param a 
 * @param b 
 * @return vec_i32 
 */
inline vec_mask32 vec_ne( vec_i32 a, vec_i32 b ) { 

    return vmvnq_u32( vceqq_s32( a, b ) );
}

/**
 * @brief Compares 2 vector values (by element) for "greater-than" (>)
 * 
 * @param a 
 * @param b 
 * @return vec_i32 
 */
inline vec_mask32 vec_gt( vec_i32 a, vec_i32 b ) { 
    return vcgtq_s32( a, b );
}

/**
 * @brief Compares 2 vector values (by element) for "less-than" (<)
 * 
 * @param a 
 * @param b 
 * @return vec_i32 
 */
inline vec_mask32 vec_lt( vec_i32 a, vec_i32 b ) { 
    return vcltq_s32( a, b );
}
/**
 * @brief Absolute value
 * 
 * @param a 
 * @return vec_i32 
 */
inline vec_i32 vec_abs( vec_i32 a ) {
    return vabsq_s32( a );
}

/**
 * @brief Selects between vector elements of vectors a and b according to the mask
 * 
 * @param a     a vector
 * @param b     b vector
 * @param mask  selection mask, 0 selects a vector element, -1 selects b vector element
 * @return vec_i32 
 */
inline vec_i32 vec_select( const vec_i32 a, const vec_i32 b, const vec_mask32 mask ) {
    return vbslq_s32( mask, b, a );
}

/**
 * @brief Mask functions
 * 
 * @note In ARM Neon the mask is simply an integer vector vec_i32
 * 
 */


/**
 * @brief Bitwise complement (not)
 * 
 * @param a 
 * @return vec_mask32 
 */
inline vec_mask32 vec_not( vec_mask32 a ) {
    return vmvnq_u32(a);
}

/**
 * @brief Bitwise or 
 * 
 * @param a 
 * @param b 
 * @return vec_mask32 
 */
inline vec_mask32 vec_or( vec_mask32 a, vec_mask32 b ) {
    return vorrq_u32( a, b );
}

/**
 * @brief Bitwise and 
 * 
 * @param a 
 * @param b 
 * @return vec_mask32 
 */
inline vec_mask32 vec_and( vec_mask32 a, vec_mask32 b ) {
    return vandq_u32( a, b );
}


/**
 * @brief Returns true (1) if all of the mask values are true
 * 
 * @param mask 
 * @return int 
 */
inline int vec_all( const vec_mask32 mask ) {
    // Check this
    // https://stackoverflow.com/questions/41005281/testing-neon-simd-registers-for-equality-over-all-lanes

    // Convert 4 x 32 bit integer to 4 x 16 bit integers
    // Cast to 1 x 64 bit vector
    // Get 1st (and only) lane
    // Compare to int

    uint16x4_t t = vqmovn_u32( mask );
    return vget_lane_u64(vreinterpret_u64_u16(t), 0) == (uint64_t)(-1);

    // Another option is to use vminvq_u32() (minimum accross lanes)
    // return vminvq_u32( mask ) == -1;
}

/**
 * @brief Returns true (1) if any of the mask values is true
 * 
 * @param mask 
 * @return int 
 */
inline int vec_any( const vec_mask32 mask ) {
    // Check this
    // https://stackoverflow.com/questions/41005281/testing-neon-simd-registers-for-equality-over-all-lanes

    // Convert 4 x 32 bit integer to 4 x 16 bit integers
    // Cast to 1 x 64 bit vector
    // Get 1st (and only) lane
    // Compare to int

    uint16x4_t t = vqmovn_u32( mask );
    return vget_lane_u64(vreinterpret_u64_u16(t), 0) != 0;

    // Another option is to use vmaxvq_u32() (maximum accross lanes)
    // return vmaxvq_u32( mask ) == -1;

}

/**
 * @brief Returns a mask with all elements set to true
 * 
 * @return vec_mask32 
 */
inline vec_mask32 vec_true() { 
    return vdupq_n_u32(-1);
}

/**
 * @brief Returns a mask with all elements set to false
 * 
 * @return vec_mask32 
 */
inline vec_mask32 vec_false() { 
    return vdupq_n_u32(0);
}

/**
 * @brief Extract a single logical value from a mask
 * 
 * @param v         Input vector mask
 * @param i         Element index
 * @return float    Selected value
 */
inline int vec_extract( const vec_mask32 v, int i ) {
   return v[i] & 1;
}

/**
 * @brief Extract a single logical value from a mask
 * 
 * @tparam imm      Element index
 * @param v         Input vector mask
 * @return int      Selected value
 */
template< int imm > 
inline int vec_extract( const vec_mask32 v ) {
    static_assert( imm >= 0 && imm < 4, "imm must be in the range [0..4]" );
    return vgetq_lane_u32( v, imm ) & 1;
}

/**
 * @brief Stream extraction
 * 
 * @param os    Output stream
 * @param v     int vector value
 * @return std::ostream& 
 */
inline std::ostream& operator<<(std::ostream& os, const vec_mask32 v) {
    os << "[";
    os << vec_extract<0>( v );
    os << vec_extract<1>( v );
    os << vec_extract<2>( v );
    os << vec_extract<3>( v );
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
    float32x4x2_t tmp = vld2q_f32((float32_t const *) addr );
    return vfloat2{ tmp.val[0], tmp.val[1] };
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
    float32x4x2_t tmp{ v.x, v.y };
    vst2q_f32( addr, tmp );
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
    float32x4x3_t tmp = vld3q_f32((float32_t const *) addr );
    return vfloat3{ tmp.val[0], tmp.val[1], tmp.val[2] };
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
    float32x4x3_t tmp{ v.x, v.y, v.z };
    vst3q_f32( addr, tmp );
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
inline vint2 vec_load_s2( int * addr ) {
    int32x4x2_t tmp = vld2q_s32((int32_t const *) addr );
    return vint2{ tmp.val[0], tmp.val[1] };
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
    int32x4x2_t tmp{ v.x, v.y };
    vst2q_s32( addr, tmp );
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
struct alignas(vec_mask32) vmask2 {
    vec_mask32 x, y;
};

