#include <arm_neon.h>

#include <iostream>
#include <iomanip>

typedef float32x4_t vec_f32;
typedef int32x4_t   vec_i32 ;
typedef uint32x4_t  vec_mask32 ;

/**
 * @brief Floating point (32 bit) SIMD types
 * 
 * @note For ARM Neon this corresponds to the vec_f32 vector
 */

/**
 * @brief Extract a single float from a _mm256 vector
 * 
 * @tparam imm      Which value to extract
 * @param v         Input vector
 * @return float    Selected value
 */
template< int imm > 
static inline float vec_extract( const vec_f32 v ) {
    static_assert( imm >= 0 && imm < 4, "imm must be in the range [0..4]" );
    return vgetq_lane_f32( v, imm );
}

static inline float vec_extract( const vec_f32 v, int i ) {
    return vgetq_lane_f32( v, i & 3 );
}

/**
 * @brief Writes the textual representation of vector v to os
 * 
 * @param os    Output stream
 * @param v     Float vector value
 * @return std::ostream& 
 */
std::ostream& operator<<(std::ostream& os, const vec_f32 v) {
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
 * @return vec_f32 
 */
static inline vec_f32 vec_zero_float() {
    return vreinterpretq_f32_u32( veorq_u32(v,v) );
}

/**
 * @brief Create a vector with all elements equal to scalar value
 * 
 * @param s 
 * @return vec_f32 
 */
static inline vec_f32 vec_float( float s ) {
    return vdupq_n_f32(s);
}

/**
 * @brief Create a float vector from an integer vector
 * 
 * @param vi 
 * @return vec_f32 
 */
static inline vec_f32 vec_float( vec_i32 vi ) {
    return vdupq_n_s32( vi );
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
static inline vec_f32 vec_float( float a, float b, float c, float d ) {
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
static inline vec_f32 vec_load( const float * mem_addr) { 
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
static inline vec_f32 vec_store( float * mem_addr, vec_f32 a ) {
    vec_store_f32( mem_addr, a ); return a;
}

static inline vec_f32 vec_neg( vec_f32 a ) {
    return vnegq_f32( a );
}

/**
 * @brief Adds 2 vector values (a+b)
 * 
 * @param a 
 * @param b 
 * @return vec_f32 
 */
static inline vec_f32 vec_add( vec_f32 a, vec_f32 b ) {
    return a+b;
}

/**
 * @brief Adds scalar value (s) to all components of vector value (a). 
 * 
 * @param a 
 * @param b 
 * @return vec_f32 
 */
static inline vec_f32 vec_add( vec_f32 a, float s ) { 
    return a + vdupq_n_f32(s);
}

/**
 * @brief Subtracts 2 vector values (a-b)
 * 
 * @param a 
 * @param b 
 * @return vec_f32 
 */
static inline vec_f32 vec_sub( vec_f32 a, vec_f32 b ) {
    return a - b;
}

/**
 * @brief Multiplies 2 vector values (a*b)
 * 
 * @param a 
 * @param b 
 * @return vec_f32 
 */
static inline vec_f32 vec_mul( vec_f32 a, vec_f32 b ) { 
    return a * b;
}

/**
 * @brief Multiplies vector (a) by scalar value (s)
 * 
 * @param a 
 * @param s 
 * @return vec_f32 
 */
static inline vec_f32 vec_mul( vec_f32 a, float s ) {
    return vmulq_n_f32(a, s);
}

/**
 * @brief Divides 2 vector values (a/b)
 * 
 * @param a 
 * @param b 
 * @return vec_f32 
 */
static inline vec_f32 vec_div( vec_f32 a, vec_f32 b ) {
    return a / b;
}

/**
 * @brief Compares 2 vector values (by element) for equality (==)
 * 
 * @param a 
 * @param b 
 * @return vec_i32
 */
static inline vec_mask32 vec_eq( vec_f32 a, vec_f32 b ) { 
    return vceqq_f32( a, b );
}

/**
 * @brief Compares 2 vector values (by element) for inequality (!=)
 * 
 * @param a 
 * @param b 
 * @return vec_i32
 */
static inline vec_mask32 vec_ne( vec_f32 a, vec_f32 b ) { 
    return vcneq_f32( a, b );
}

/**
 * @brief Compares 2 vector values (by element) for "greater-than" (>)
 * 
 * @param a 
 * @param b 
 * @return vec_f32 
 */
static inline vec_mask32 vec_gt( vec_f32 a, vec_f32 b ) { 
    return vcgtq_f32( a, b );
}

/**
 * @brief Compares 2 vector values (by element) for "greater of equal" (>=) and return mask
 * 
 * @param a 
 * @param b 
 * @return      Resulting mask, for each element i 0 if false, -1 if true
 */
static inline vec_mask32 vec_ge( vec_f32 a, vec_f32 b ) { 
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
static inline vec_f32 vec_ge( vec_f32 a, vec_f32 b, vec_f32 v ) { 

    return vreinterpretq_f32_u32( 
        vandrq_u32( vcgeq_f32( a, b ), vreinterpretq_u32_f32(v)) );
}

/**
 * @brief Compares 2 vector values (by element) for "greater of equal" (>=) and return v (integer)
 * 
 * @param a 
 * @param b 
 * @param vi 
 * @return      Result, for each element i, 0 if false and v[i] if true
 */
static inline vec_i32 vec_ge( vec_f32 a, vec_f32 b, vec_i32 vi ) { 
    return  vandrq_u32( vcgeq_f32( a, b ), vi );
}


/**
 * @brief Compares 2 vector values (by element) for "less than" (<) and return mask
 * 
 * @param a 
 * @param b 
 * @return      Resulting mask, for each element i 0 if false, -1 if true
 */
static inline vec_mask32 vec_lt( vec_f32 a, vec_f32 b ) { 
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
static inline vec_f32 vec_lt( vec_f32 a, vec_f32 b, vec_f32 v ) { 
    return vreinterpretq_f32_u32( 
        vandrq_u32( vcltq_f32( a, b ), vreinterpretq_u32_f32(v)) );
}

/**
 * @brief Compares 2 vector values (by element) for "less than" (<) and return value (integer)
 * 
 * @param a     Value a
 * @param b     Value b
 * @param v     Result, for each element i, 0 if false and v[i] if true
 * @return vec_f32 
 */

static inline vec_mask32 vec_lt( vec_f32 a, vec_f32 b, vec_i32 vi ) { 
    return  vandrq_u32( vcltq_f32( a, b ), vi );
}

/**
 * @brief Compares 2 vector values (by element) for "less or equal" (<=)
 * 
 * @param a 
 * @param b 
 * @return vec_f32 
 */
static inline vec_mask32 vec_le( vec_f32 a, vec_f32 b ) { 
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
static inline vec_f32 vec_fmadd( vec_f32 a, vec_f32 b, vec_f32 c ) { 
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
static inline vec_f32 vec_fmsub( vec_f32 a, vec_f32 b, vec_f32 c ) { 
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
static inline vec_f32 vec_fnmadd( vec_f32 a, vec_f32 b, vec_f32 c ) {
    return vmlsq_f32( c, b, a );
}

/**
 * @brief Reciprocal (1/a)
 * 
 * @param a
 * @return vec_f32 
 */
static inline vec_f32 vec_recp( const vec_f32 a )
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
static inline vec_f32 vec_rsqrt( const vec_f32 a ) {

    // Full calculation
    auto rsqrt = vdupq_n_f32(1.0f) / vsqrt_f32(a);

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
static inline vec_f32 vec_sqrt( const vec_f32 a ) {
    return vsqrt_f32(a);
}

/**
 * @brief Absolute value
 * 
 * @param a 
 * @return vec_f32 
 */
static inline vec_f32 vec_fabs( const vec_f32 a ) { 
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
static inline vec_f32 vec_select( const vec_f32 a, const vec_f32 b, const vec_mask32 mask ) {
    #warning need to check the order of the arguments
    return vbslq_f32( a, b, mask );
}


/**
 * @brief Add all vector elements
 * 
 * @param a 
 * @return float 
 */
static inline float vec_reduce_add( const vec_f32 a ) {
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
static inline vec_f32 vec_gather( float const * base_addr, vec_i32 vindex ) {

    vec_f32 v;

    v = vld1q_dup_f32(  base_addr + vgetq_lane_s32( vindex, 0 ) );
    v = vld1q_lane_f32( base_addr + vgetq_lane_s32( vindex, 1 ), v, 1 );
    v = vld1q_lane_f32( base_addr + vgetq_lane_s32( vindex, 2 ), v, 2 );
    v = vld1q_lane_f32( base_addr + vgetq_lane_s32( vindex, 3 ), v, 3 );

    return v;
}

/**
 * @brief Integer (32 bit) SIMD types
 * 
 * @note For AVX this corresponds to the vec_i32 vector
 */

/**
 * @brief Extract a single integer from a _mm256i vector
 * 
 * @tparam imm      Which value to extract
 * @param v         Input vector
 * @return float    Selected value
 */

template< int imm > 
static inline int vec_extract( const vec_i32 v ) {
    static_assert( imm >= 0 && imm < 4, "imm must be in the range [0..3]" );
    return vgetq_lane_s32( v, imm );
}

static inline int vec_extract( const vec_i32 v, int i ) {
   return vgetq_lane_s32( v, i );
}

/**
 * @brief Writes the textual representation of vector v to os
 * 
 * @param os    Output stream
 * @param v     int vector value
 * @return std::ostream& 
 */
std::ostream& operator<<(std::ostream& os, const vec_i32 v) {
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
static inline vec_i32 vec_zero_int() {
    vec_i32 v;
    return veorq_s32(v,v);

/**
 * @brief Create a vector with all elements equal to scalar value
 * 
 * @param a 
 * @return vec_i32 
 */
static inline vec_i32 vec_int( int s ) {
    return vdupq_n_f32(s);
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
static inline vec_i32 vec_int( int a, int b, int c, int d ){ 
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
static inline vec_i32 vec_load( const int * mem_addr) { 
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
static inline vec_i32 vec_store( int * mem_addr, vec_i32 a ) { 
    vst1q_s32( mem_addr, a ); return a;
}

/**
 * @brief Adds 2 vector values (a+b)
 * 
 * @param a 
 * @param b 
 * @return vec_i32 
 */
static inline vec_i32 vec_add( vec_i32 a, vec_i32 b ) { 
    return a + b;
}

/**
 * @brief Subtracts 2 vector values (a-b)
 * 
 * @param a 
 * @param b 
 * @return vec_i32 
 */
static inline vec_i32 vec_sub( vec_i32 a, vec_i32 b ) {
    return a - b;
}

/**
 * @brief Adds scalar value (s) to all components of vector value (a). 
 * 
 * @param a 
 * @param b 
 * @return vec_i32 
 */
static inline vec_i32 vec_add( vec_i32 a, int s ) { 
    return v + vdupq_n_s32(s);
}

/**
 * @brief Multiplies 2 vector values (a*b)
 * 
 * @param a 
 * @param b 
 * @return vec_i32 
 */
static inline vec_i32 vec_mul( vec_i32 a, vec_i32 b ) {
    return a * b;
}

/**
 * @brief Multiplies vector (a) by scalar value (s)
 * 
 * @param a 
 * @param b 
 * @return vec_i32 
 */
static inline vec_i32 vec_mul( vec_i32 a, int s ) { 
    return vmulq_n_s32( a, s );
}

/**
 * @brief Multiplies vector by 3
 * 
 * @param a 
 * @return vec_i32 
 */
static inline vec_i32 vec_mul3( vec_i32 a ) {
    return a + a + a;
}

/**
 * @brief Compares 2 vector values (by element) for equality (==)
 * 
 * @param a 
 * @param b 
 * @return vec_i32 
 */
static inline vec_mask32 vec_eq( vec_i32 a, vec_i32 b ) { 
    return vceqq_s32( a, b );
}

/**
 * @brief Compares 2 vector values (by element) for inequality (!=)
 * 
 * @param a 
 * @param b 
 * @return vec_i32 
 */
static inline vec_mask32 vec_ne( vec_i32 a, vec_i32 b ) { 
    return vcneq_s32( a, b );
}

/**
 * @brief Compares 2 vector values (by element) for "greater-than" (>)
 * 
 * @param a 
 * @param b 
 * @return vec_i32 
 */
static inline vec_mask32 vec_gt( vec_i32 a, vec_i32 b ) { 
    return vcgtq_s32( a, b );
}

/**
 * @brief Compares 2 vector values (by element) for "less-than" (<)
 * 
 * @param a 
 * @param b 
 * @return vec_i32 
 */
static inline vec_mask32 vec_lt( vec_i32 a, vec_i32 b ) { 
    return vcltq_s32( a, b );
}
/**
 * @brief Absolute value
 * 
 * @param a 
 * @return vec_i32 
 */
static inline vec_i32 vec_abs( vec_i32 a ) {
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
static inline vec_i32 vec_select( const vec_i32 a, const vec_i32 b, const vec_i32 mask ) {
    #warning need to check the order of the arguments
    return vbslq_s32( mask, a, b );
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
static inline vec_mask32 vec_not( vec_mask32 a ) {
    return vmvnq_u32(a);
}

/**
 * @brief Bitwise or 
 * 
 * @param a 
 * @param b 
 * @return vec_mask32 
 */
static inline vec_mask32 vec_or( vec_mask32 a, vec_mask32 b ) {
    return vorrq_u32( a, b );
}

/**
 * @brief Bitwise and 
 * 
 * @param a 
 * @param b 
 * @return vec_mask32 
 */
static inline vec_mask32 vec_and( vec_mask32 a, vec_mask32 b ) {
    return vandq_u32( a, b );
}


/**
 * @brief Returns true (1) if all of the mask values are true
 * 
 * @param mask 
 * @return int 
 */
static inline int vec_all( const vec_mask32 mask ) {
    // Check this
    // https://stackoverflow.com/questions/41005281/testing-neon-simd-registers-for-equality-over-all-lanes

    // Convert 4 x 32 bit integer to 4 x 16 bit integers
    // Cast to 1 x 64 bit vector
    // Get 1st (and only) lane
    // Compare to int

    uint16x4_t t = vqmovn_u32( mask );
    return vget_lane_u64(vreinterpret_u64_u16(t), 0) == -1;

    // Another option is to use vminvq_u32() (minimum accross lanes)
    // return vminvq_u32( mask ) == -1;
}

/**
 * @brief Returns true (1) if any of the mask values is true
 * 
 * @param mask 
 * @return int 
 */
static inline int vec_any( const vec_mask32 mask ) {
    // Check this
    // https://stackoverflow.com/questions/41005281/testing-neon-simd-registers-for-equality-over-all-lanes

    // Convert 4 x 32 bit integer to 4 x 16 bit integers
    // Cast to 1 x 64 bit vector
    // Get 1st (and only) lane
    // Compare to int

    uint16x4_t t = vqmovn_u32( mask );
    return vget_lane_u64(vreinterpret_u64_u16(t), 0) != 0

    // Another option is to use vmaxvq_u32() (maximum accross lanes)
    // return vmaxvq_u32( mask ) == -1;

}


static inline vec_mask32 vec_true() { 
    
    // vec_i32 a; return vceqq_s32( a, a );
    return vdupq_n_f32(-1);
}

static inline vec_mask32 vec_false() { 
    vec_i32 a; return veorq_s32( a, a );
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
static inline vfloat2 vfloat2_zero( ) {
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
static inline vfloat2 vec_load_s2( const float * addr ) {
    
    // float32x4x2_t tmp = vld2q_f32((float32_t const *) addr );
    // return vfloat2{ tmp.val[0], tmp.val[1] };
    
    vfloat2 v = vld2q_f32((float32_t const *) addr );
    return v;
}

/**
 * @brief Stores 2-element structure to memory
 * 
 * @note Data is interleaved from 2 vectors and stored sequentially
 * 
 * @param addr 
 * @param v 
 */
static inline void vec_store_s2( float * addr, const vfloat2 v ) {
    // float32x4x2_t tmp{ v.x, v.y };
    // vst2q_f32( addr, tmp );

    vst2q_f32( addr, v );
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
static inline vfloat3 vfloat3_zero( ) {
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
static inline vfloat3 vec_load_s3( const float * addr ) {
    // float32x4x3_t tmp = vld3q_f32((float32_t const *) addr );
    // return vfloat3{ tmp.val[0], tmp.val[1], tmp.val[2] };
    
    vfloat3 v = vld3q_f32((float32_t const *) addr );
    return v;
}

/**
 * @brief Stores 3-element structure to memory
 * 
 * @note Data is interleaved from 3 vectors and stored sequentially
 * 
 * @param addr 
 * @param v 
 */
static inline void vec_store_s3( float * addr, const vfloat3 v ) {
    // float32x4x3_t tmp{ v.x, v.y, v.z };
    // vst3q_f32( addr, tmp );

    vst3q_f32( addr, v );
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
static inline vint2 vec_load_s2( int * addr ) {

    // int32x4x2_t tmp = vld2q_s32((int32_t const *) addr );
    // return vint2{ tmp.val[0], tmp.val[1] };
    
    vint2 v = vld2q_s32((int32_t const *) addr );
    return v;

}

/**
 * @brief Stores 2-element structure to memory
 * 
 * @note Data is interleaved from 2 vectors and stored sequentially
 * 
 * @param addr 
 * @param v 
 */
static inline void vec_store_s2( int * addr, const vint2 v ) {
    // int32x4x2_t tmp{ v.x, v.y, v.z };
    // vst2q_s32( addr, tmp );

    vst2q_s32( addr, v );
}

static inline vint2 vint2_zero( ) {
    vint2 v{ vec_zero_int(), vec_zero_int() };
    return v;
}


class Vec4Float {
    vec_f32 v;
    public:
    Vec8Float( const vec_f32 v ) v(v) {};
    Vec8Float( const float s ) v( vdupq_n_f32(s) ) {};
    float extract( const int i ) { return vgetq_lane_f32(s); }
    friend std::ostream& operator<<(std::ostream& os, const Vec4Float& obj) { 
        os << obj.v;
        return os;
    }
};

class Vec4Int {
    vec_i32 v;
    public:
    Vec8Int( const vec_i32 v ) v(v) {};
    Vec8Int( const int s ) v( vdupq_n_s32(s) ) {};
    int extract( const int i ) { return vgetq_lane_s32(s); }
    friend std::ostream& operator<<(std::ostream& os, const Vec4Int& obj) { 
        os << obj.v;
        return os;
    }
};

constexpr int vecwidth = 4;

typedef vec_f32    vfloat;
typedef vec_i32    vint;
typedef vec_mask32 vmask;

typedef Vec4Float  VecFloat_s;
typedef Vec4Int    VecInt_s;


template< typename T >
int is_aligned_32( T * addr ) { return (((uintptr_t)addr & 0x1F) == 0); }

template< typename T >
void assert_aligned_32( T * addr, std::string msg ) { 
    if ( ! is_aligned_32(addr) ) {
        std::cerr << msg << '\n';
        std::cerr << "Address " << addr << " is not 32 bit aligned, aborting." << std::endl;
        abort();
    }
}

template< typename T >
int is_aligned_16( T * addr ) { return (((uintptr_t)addr & 0x0F) == 0); }

template< typename T >
void assert_aligned_16( T * addr, std::string msg ) { 
    if ( ! is_aligned_16(addr) ) {
        std::cerr << msg << '\n';
        std::cerr << "Address " << addr << " is not 16 bit aligned, aborting." << std::endl;
        abort();
    }
}
