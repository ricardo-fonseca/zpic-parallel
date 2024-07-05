#include <immintrin.h>

#include <iostream>
#include <iomanip>

/**
 * @brief Floating point (32 bit) SIMD types
 * 
 * @note For AVX this corresponds to the __m256 vector
 */

/**
 * @brief Extract a single float from a _mm256 vector
 * 
 * @tparam imm      Which value to extract
 * @param v         Input vector
 * @return float    Selected value
 */
template< int imm > 
static inline float vec_extract( const __m256 v ) {
    static_assert( imm >= 0 && imm < 8, "imm must be in the range [0..7]" );
    
    // The compiler will (usually) optimize this and avoid the memory copy
    alignas( __m256 ) float buf[8];
    _mm256_store_ps( buf, v );
    return buf[imm];
}

static inline float vec_extract( const __m256 v, int i ) {
    union {
        __m256 v;
        float s[8];
    } b;
    b.v = v;
    return b.s[i & 7];
}

/**
 * @brief Writes the textual representation of vector v to os
 * 
 * @param os    Output stream
 * @param v     Float vector value
 * @return std::ostream& 
 */
std::ostream& operator<<(std::ostream& os, const __m256 v) {
    os << "[";
    os <<         vec_extract<0>( v );
    os << ", " << vec_extract<1>( v );
    os << ", " << vec_extract<2>( v );
    os << ", " << vec_extract<3>( v );
    os << ", " << vec_extract<4>( v );
    os << ", " << vec_extract<5>( v );
    os << ", " << vec_extract<6>( v );
    os << ", " << vec_extract<7>( v );
    os << "]";

    return os;
}

/**
 * @brief Returns a zero valued vector
 * 
 * @return __m256 
 */
static inline __m256 vec_zero_float() {
    return _mm256_setzero_ps();
}

/**
 * @brief Create a vector with all elements equal to scalar value
 * 
 * @param s 
 * @return __m256 
 */
static inline __m256 vec_float( float s ) {
    return _mm256_set1_ps(s);
}

/**
 * @brief Create a float vector from an integer vector
 * 
 * @param vi 
 * @return __m256 
 */
static inline __m256 vec_float( __m256i vi ) {
    return _mm256_cvtepi32_ps( vi );
}

/**
 * @brief Create a vector from the scalar elements
 * 
 * @param a
 * @param b 
 * @param c 
 * @param d 
 * @param e 
 * @param f 
 * @param g 
 * @param h 
 * @return __m256 
 */
static inline __m256 vec_float( float a, float b, float c, float d, float e, float f, float g, float h ) {
    return _mm256_setr_ps(a, b, c, d, e, f, g, h);
}

/**
 * @brief Loads vector from memory
 * 
 * @warning The address must be aligned to a 32 byte boundary
 * 
 * @param mem_addr 
 * @return __m256 
 */
static inline __m256 vec_load( const float * mem_addr) { 
    return _mm256_load_ps( mem_addr );
}

/**
 * @brief Stores vector to memory
 * 
 * @warning The address must be aligned to a 32 byte boundary
 * 
 * @param mem_addr 
 * @param a 
 */
static inline __m256 vec_store( float * mem_addr, __m256 a ) {
    _mm256_store_ps( mem_addr, a ); return a;
}

static inline __m256 vec_neg( __m256 a ) {
    return _mm256_sub_ps( _mm256_setzero_ps(), a );
}

/**
 * @brief Adds 2 vector values (a+b)
 * 
 * @param a 
 * @param b 
 * @return __m256 
 */
static inline __m256 vec_add( __m256 a, __m256 b ) {
    return _mm256_add_ps(a,b);
}

/**
 * @brief Adds scalar value (s) to all components of vector value (a). 
 * 
 * @param a 
 * @param b 
 * @return __m256 
 */
static inline __m256 vec_add( __m256 a, float s ) { 
    return _mm256_add_ps(a,_mm256_set1_ps(s));
}

/**
 * @brief Subtracts 2 vector values (a-b)
 * 
 * @param a 
 * @param b 
 * @return __m256 
 */
static inline __m256 vec_sub( __m256 a, __m256 b ) {
    return _mm256_sub_ps(a,b);
}

/**
 * @brief Multiplies 2 vector values (a*b)
 * 
 * @param a 
 * @param b 
 * @return __m256 
 */
static inline __m256 vec_mul( __m256 a, __m256 b ) { 
    return _mm256_mul_ps(a,b);
}

/**
 * @brief Multiplies vector (a) by scalar value (s)
 * 
 * @param a 
 * @param s 
 * @return __m256 
 */
static inline __m256 vec_mul( __m256 a, float s ) {
    return _mm256_mul_ps(a,_mm256_set1_ps(s));
}

/**
 * @brief Divides 2 vector values (a/b)
 * 
 * @param a 
 * @param b 
 * @return __m256 
 */
static inline __m256 vec_div( __m256 a, __m256 b ) {
    return _mm256_div_ps(a,b);
}

/**
 * @brief Compares 2 vector values (by element) for equality (==)
 * 
 * @param a 
 * @param b 
 * @return __m256i
 */
static inline __m256i vec_eq( __m256 a, __m256 b ) { 
    return _mm256_castps_si256( _mm256_cmp_ps(a,b,_CMP_EQ_OQ) );
}

/**
 * @brief Compares 2 vector values (by element) for inequality (!=)
 * 
 * @param a 
 * @param b 
 * @return __m256i
 */
static inline __m256i vec_ne( __m256 a, __m256 b ) { 
    return _mm256_castps_si256( _mm256_cmp_ps(a,b,_CMP_NEQ_OQ) );
}

/**
 * @brief Compares 2 vector values (by element) for "greater-than" (>)
 * 
 * @param a 
 * @param b 
 * @return __m256i
 */
static inline __m256i vec_gt( __m256 a, __m256 b ) { 
    return _mm256_castps_si256( _mm256_cmp_ps(a,b,_CMP_GT_OQ) );
}

/**
 * @brief Compares 2 vector values (by element) for "greater of equal" (>=) and return mask
 * 
 * @param a 
 * @param b 
 * @return      Resulting mask, for each element i 0 if false, -1 if true
 */
static inline __m256i vec_ge( __m256 a, __m256 b ) { 
    return _mm256_castps_si256( _mm256_cmp_ps(a,b,_CMP_GE_OQ) );
}

/**
 * @brief Compares 2 vector values (by element) for "greater of equal" (>=) and return v (float)
 * 
 * @param a 
 * @param b 
 * @param v
 * @return      Result, for each element i, 0 if false and v[i] if true
 */
static inline __m256 vec_ge( __m256 a, __m256 b, __m256 v ) { 
    return _mm256_and_ps( _mm256_cmp_ps(a,b,_CMP_GE_OQ), v );
}

/**
 * @brief Compares 2 vector values (by element) for "greater of equal" (>=) and return v (integer)
 * 
 * @param a 
 * @param b 
 * @param vi 
 * @return      Result, for each element i, 0 if false and v[i] if true
 */
static inline __m256i vec_ge( __m256 a, __m256 b, __m256i vi ) { 
    return _mm256_and_si256( _mm256_castps_si256( _mm256_cmp_ps(a,b,_CMP_GE_OQ) ), vi );
}


/**
 * @brief Compares 2 vector values (by element) for "less than" (<) and return mask
 * 
 * @param a 
 * @param b 
 * @return      Resulting mask, for each element i 0 if false, -1 if true
 */
static inline __m256i vec_lt( __m256 a, __m256 b ) { 
    return _mm256_castps_si256( _mm256_cmp_ps(a,b,_CMP_LT_OQ) );
}

/**
 * @brief Compares 2 vector values (by element) for "less than" (<) and return value
 * 
 * @param a     Value a
 * @param b     Value b
 * @param v     Result, for each element i, 0 if false and v[i] if true
 * @return __m256 
 */
static inline __m256 vec_lt( __m256 a, __m256 b, __m256 v ) { 
    return _mm256_and_ps( _mm256_cmp_ps(a,b,_CMP_LT_OQ), v );
}

/**
 * @brief Compares 2 vector values (by element) for "less than" (<) and return value (integer)
 * 
 * @param a     Value a
 * @param b     Value b
 * @param v     Result, for each element i, 0 if false and v[i] if true
 * @return __m256 
 */

static inline __m256i vec_lt( __m256 a, __m256 b, __m256i vi ) { 
    return _mm256_and_si256( _mm256_castps_si256( _mm256_cmp_ps(a,b,_CMP_LT_OQ) ), vi );
}

/**
 * @brief Compares 2 vector values (by element) for "less or equal" (<=)
 * 
 * @param a 
 * @param b 
 * @return __m256 
 */
static inline __m256i vec_le( __m256 a, __m256 b ) { 
    return _mm256_castps_si256( _mm256_cmp_ps(a,b,_CMP_LE_OQ) );
}

/**
 * @brief Fused multiply add: (a*b)+c
 * 
 * @param a 
 * @param b 
 * @param c 
 * @return __m256 
 */
static inline __m256 vec_fmadd( __m256 a, __m256 b, __m256 c ) { 
    return _mm256_fmadd_ps( a, b, c );
}

/**
 * @brief Fused multiply subtract: (a*b)-c
 * 
 * @param a 
 * @param b 
 * @param c 
 * @return __m256 
 */
static inline __m256 vec_fmsub( __m256 a, __m256 b, __m256 c ) { 
    return _mm256_fmsub_ps( a, b, c );
}

/**
 * @brief Fused negate multiply add: -(a*b)+c
 * 
 * @param a 
 * @param b 
 * @param c 
 * @return __m256 
 */
static inline __m256 vec_fnmadd( __m256 a, __m256 b, __m256 c ) {
    return _mm256_fnmadd_ps( a, b, c );
}

/**
 * @brief Reciprocal (1/a)
 * 
 * @param a
 * @return __m256 
 */
static inline __m256 vec_recp( const __m256 a )
{
    // Full calculation
    auto recp = _mm256_div_ps( _mm256_set1_ps( 1 ), a );

/*
    // Fast estimate + 1 Newton-Raphson iteration
    auto recp = _mm256_rcp_ps( a );
    recp = _mm256_mul_ps(recp, _mm256_fnmadd_ps(recp, a, _mm256_set1_ps( 2 )));
*/

    return recp;
}

/**
 * @brief Reciprocal square root 1/sqrt(a)
 * 
 * @param a 
 * @return __m256 
 */
static inline __m256 vec_rsqrt( const __m256 a ) {

    auto rsqrt = _mm256_div_ps( _mm256_set1_ps(1), _mm256_sqrt_ps(a) );

/*
    // Fast estimate + 1 Newton-Raphson iteration
    auto rsqrt = _mm256_rsqrt_ps( a );
    auto const c1_2 = _mm256_set1_ps( 0.5 );
    auto const c3   = _mm256_set1_ps( 3 );
    rsqrt = _mm256_mul_ps( c1_2, 
                _mm256_mul_ps( rsqrt, 
                    _mm256_fnmadd_ps( _mm256_mul_ps( rsqrt, rsqrt ), a, c3 )
                )
            );
*/

    return rsqrt;
}

/**
 * @brief Square root
 * 
 * @param a 
 * @return __m256 
 */
static inline __m256 vec_sqrt( const __m256 a ) {
    return _mm256_sqrt_ps(a);
}

/**
 * @brief Absolute value
 * 
 * @param a 
 * @return __m256 
 */
static inline __m256 vec_fabs( const __m256 a ) { 
    const __m256i mask = _mm256_set1_epi32 (~(1<<31) );
    return _mm256_and_ps(a, _mm256_castsi256_ps(mask));
}

/**
 * @brief Selects between vector elements of vectors a and b according to the mask
 * 
 * @param a     a vector
 * @param b     b vector
 * @param mask  selection mask, 0 selects a vector element, -1 selects b vector element
 * @return __m256i 
 */
static inline __m256 vec_select( const __m256 a, const __m256 b, const __m256i mask ) {
    return _mm256_blendv_ps( a, b, _mm256_castsi256_ps(mask) );
}


/**
 * @brief Add all vector elements
 * 
 * @param a 
 * @return float 
 */
static inline float vec_reduce_add( const __m256 a ) {
   __m128 r = _mm_add_ps( _mm256_extractf128_ps(a, 1), _mm256_castps256_ps128(a));
   r = _mm_hadd_ps( r, r );
   r = _mm_hadd_ps( r, r );
   return _mm_cvtss_f32( r );
}

/**
 * @brief Gather values from base address + vector index
 * 
 * @param base_addr 
 * @param vindex 
 * @return __m256 
 */
static inline __m256 vec_gather( float const * base_addr, __m256i vindex ) {

    // This has terrible performance
    //__m256 v = _mm256_i32gather_ps( base_addr, vindex, 4 );

    union { __m256i v; int s[8]; } index;
    index.v = vindex;

    __m128 lo = _mm_setr_ps (
        base_addr[index.s[0]],
        base_addr[index.s[1]],
        base_addr[index.s[2]],
        base_addr[index.s[3]]
    );

    __m128 hi = _mm_setr_ps (
        base_addr[index.s[4]],
        base_addr[index.s[5]],
        base_addr[index.s[6]],
        base_addr[index.s[7]]
    );

    __m256 v = _mm256_set_m128(hi,lo);

    return v;
}

/**
 * @brief Integer (32 bit) SIMD types
 * 
 * @note For AVX this corresponds to the __m256i vector
 */

/**
 * @brief Extract a single integer from a _mm256i vector
 * 
 * @tparam imm      Which value to extract
 * @param v         Input vector
 * @return float    Selected value
 */

template< int imm > 
static inline int vec_extract( const __m256i v ) {
    static_assert( imm >= 0 && imm < 8, "imm must be in the range [0..7]" );
    
    // The compiler will (usually) optimize this and avoid the memory copy
    alignas( __m256i ) int buf[8];
    _mm256_store_si256( ( __m256i * ) buf, v );
    return buf[imm];
}

static inline int vec_extract( const __m256i v, int i ) {
    union {
        __m256i v;
        int32_t s[8];
    } b;
    b.v = v;
    return b.s[i & 7];
}

/**
 * @brief Writes the textual representation of vector v to os
 * 
 * @param os    Output stream
 * @param v     int vector value
 * @return std::ostream& 
 */
std::ostream& operator<<(std::ostream& os, const __m256i v) {
    os << "[";
    os <<         vec_extract<0>( v );
    os << ", " << vec_extract<1>( v );
    os << ", " << vec_extract<2>( v );
    os << ", " << vec_extract<3>( v );
    os << ", " << vec_extract<4>( v );
    os << ", " << vec_extract<5>( v );
    os << ", " << vec_extract<6>( v );
    os << ", " << vec_extract<7>( v );
    os << "]";

    return os;
}

/**
 * @brief Returns a zero valued vector
 * 
 * @return __m256i 
 */
static inline __m256i vec_zero_int() {
    return _mm256_setzero_si256();
}

/**
 * @brief Create a vector with all elements equal to scalar value
 * 
 * @param a 
 * @return __m256i 
 */
static inline __m256i vec_int( int s ) {
    return _mm256_set1_epi32(s);
}

/**
 * @brief Create a vector from the scalar elements
 * 
 * @param a 
 * @param b 
 * @param c 
 * @param d 
 * @param e 
 * @param f 
 * @param g 
 * @param h 
 * @return __m256i 
 */
static inline __m256i vec_int( int a, int b, int c, int d, int e, int f, int g, int h ){ 
    return _mm256_setr_epi32( a, b, c, d, e, f, g, h );
}

/**
 * @brief Loads vector from memory
 * 
 * @warning The address must be aligned to a 32 byte boundary
 * 
 * @param mem_addr 
 * @return __m256i 
 */
static inline __m256i vec_load( const int * mem_addr) { 
    return _mm256_load_epi32( mem_addr );
}

/**
 * @brief Stores vector to memory
 * 
 * @warning The address must be aligned to a 32 byte boundary
 * 
 * @param mem_addr 
 * @param a 
 */
static inline __m256i vec_store( int * mem_addr, __m256i a ) { 
    _mm256_store_epi32( mem_addr, a ); return a;
}

/**
 * @brief Adds 2 vector values (a+b)
 * 
 * @param a 
 * @param b 
 * @return __m256i 
 */
static inline __m256i vec_add( __m256i a, __m256i b ) { 
    return _mm256_add_epi32(a,b);
}

/**
 * @brief Subtracts 2 vector values (a-b)
 * 
 * @param a 
 * @param b 
 * @return __m256i 
 */
static inline __m256i vec_sub( __m256i a, __m256i b ) {
    return _mm256_sub_epi32(a,b);
}

/**
 * @brief Adds scalar value (s) to all components of vector value (a). 
 * 
 * @param a 
 * @param b 
 * @return __m256i 
 */
static inline __m256i vec_add( __m256i a, int s ) { 
    return _mm256_add_epi32(a, _mm256_set1_epi32(s) );
}

/**
 * @brief Multiplies 2 vector values (a*b)
 * 
 * @param a 
 * @param b 
 * @return __m256i 
 */
static inline __m256i vec_mul( __m256i a, __m256i b ) {
    return _mm256_mullo_epi32(a,b);
}

/**
 * @brief Multiplies vector (a) by scalar value (s)
 * 
 * @param a 
 * @param b 
 * @return __m256i 
 */
static inline __m256i vec_mul( __m256i a, int s ) { 
    return _mm256_mullo_epi32(a, _mm256_set1_epi32(s) );
}

/**
 * @brief Multiplies vector by 3
 * 
 * @param a 
 * @return __m256i 
 */
static inline __m256i vec_mul3( __m256i a ) {
    return _mm256_add_epi32( _mm256_add_epi32( a, a ), a );
}

/**
 * @brief Compares 2 vector values (by element) for equality (==)
 * 
 * @param a 
 * @param b 
 * @return __m256i 
 */
static inline __m256i vec_eq( __m256i a, __m256i b ) { 
    return _mm256_cmpeq_epi32( a, b );
}

/**
 * @brief Compares 2 vector values (by element) for inequality (!=)
 * 
 * @param a 
 * @param b 
 * @return __m256i 
 */
static inline __m256i vec_ne( __m256i a, __m256i b ) { 
    return ~ _mm256_cmpeq_epi32( a, b );
}

/**
 * @brief Compares 2 vector values (by element) for "greater-than" (>)
 * 
 * @param a 
 * @param b 
 * @return __m256i 
 */
static inline __m256i vec_gt( __m256i a, __m256i b ) { 
    return _mm256_cmpgt_epi32( a, b );
}

/**
 * @brief Compares 2 vector values (by element) for "less-than" (<)
 * 
 * @param a 
 * @param b 
 * @return __m256i 
 */
static inline __m256i vec_lt( __m256i a, __m256i b ) { 
    return _mm256_cmpgt_epi32( b, a );
}

/**
 * @brief Bitwise complement (not)
 * 
 * @param a 
 * @return __m256i 
 */
static inline __m256i vec_not( __m256i a ) {
    return ~ a;
}

/**
 * @brief Bitwise or 
 * 
 * @param a 
 * @param b 
 * @return __m256i 
 */
static inline __m256i vec_or( __m256i a, __m256i b ) {
    return _mm256_or_si256( a, b );
}

/**
 * @brief Bitwise and 
 * 
 * @param a 
 * @param b 
 * @return __m256i 
 */
static inline __m256i vec_and( __m256i a, __m256i b ) {
    return _mm256_and_si256( a, b );
}

/**
 * @brief Absolute value
 * 
 * @param a 
 * @return __m256i 
 */
static inline __m256i vec_abs( __m256i a ) {
    return _mm256_abs_epi32( a );
}

/**
 * @brief Selects between vector elements of vectors a and b according to the mask
 * 
 * @param a     a vector
 * @param b     b vector
 * @param mask  selection mask, 0 selects a vector element, -1 selects b vector element
 * @return __m256i 
 */
static inline __m256i vec_select( const __m256i a, const __m256i b, const __m256i mask ) {
    return _mm256_blendv_epi8( a, b, mask );
}

/**
 * @brief Returns true (1) if all of the mask values are 1
 * 
 * @param mask 
 * @return int 
 */
static inline int vec_all( const __m256i mask ) {
    return _mm256_testc_si256( mask, _mm256_cmpeq_epi32( mask, mask ) );
}

/**
 * @brief Returns true (1) if any of the mask values is 1
 * 
 * @param mask 
 * @return int 
 */
static inline int vec_any( const __m256i mask ) {
    return ! _mm256_testz_si256( mask, mask );
}


static inline __m256i vec_true() { 
    
    // __m256i a; return _mm256_cmpeq_epi32( a, a );
    return _mm256_set1_epi32(-1);
}

static inline __m256i vec_false() { 
    return _mm256_setzero_si256();
}


/**
 * @brief Vector version of the float2 type holding 2 (.x, .y) vectors
 * 
 */
struct alignas(__m256) vfloat2 {
    __m256 x, y;
};

/**
 * @brief Returs a zero valued vfloat2
 * 
 * @return vfloat2 
 */
static inline vfloat2 vfloat2_zero(  ) {
    vfloat2 v{ _mm256_setzero_ps(), _mm256_setzero_ps() };
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

    __m256 m02, m13;
    m02 = _mm256_castps128_ps256(_mm_loadu_ps( & addr[0] ) );
    m13 = _mm256_castps128_ps256(_mm_loadu_ps( & addr[4] ) );
    m02 = _mm256_insertf128_ps(m02 ,_mm_loadu_ps(&addr[8]),1);
    m13 = _mm256_insertf128_ps(m13 ,_mm_loadu_ps(&addr[12]),1);
    
    vfloat2 v { 
        _mm256_shuffle_ps( m02, m13, _MM_SHUFFLE( 2, 0, 2, 0 ) ),
        _mm256_shuffle_ps( m02, m13, _MM_SHUFFLE( 3, 1, 3, 1 ) )
    };

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
    __m256 r02, r13;
    r02 = _mm256_unpacklo_ps( v.x, v.y );
    r13 = _mm256_unpackhi_ps( v.x, v.y );
    _mm_storeu_ps( &addr[ 0], _mm256_castps256_ps128( r02 ) );
    _mm_storeu_ps( &addr[ 4], _mm256_castps256_ps128( r13 ) );
    _mm_storeu_ps( &addr[ 8], _mm256_extractf128_ps( r02 ,1 ) );
    _mm_storeu_ps( &addr[12], _mm256_extractf128_ps( r13 ,1 ) ); 
}

/**
 * @brief Vector version of the float3 type holding 3 (.x, .y, .z) vectors
 * 
 */
struct alignas(__m256) vfloat3 {
    __m256 x, y, z;
};

/**
 * @brief Returs a zero valued vfloat3
 * 
 * @return vfloat3 
 */
static inline vfloat3 vfloat3_zero( ) {
    vfloat3 v{ _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
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
    __m256 m03, m14, m25;
    m03 = _mm256_castps128_ps256(   _mm_loadu_ps( & addr[ 0] ) );
    m14 = _mm256_castps128_ps256(   _mm_loadu_ps( & addr[ 4] ) );
    m25 = _mm256_castps128_ps256(   _mm_loadu_ps( & addr[ 8] ) );
    m03 = _mm256_insertf128_ps(m03 ,_mm_loadu_ps( & addr[12] ) , 1 );
    m14 = _mm256_insertf128_ps(m14 ,_mm_loadu_ps( & addr[16] ) , 1 );
    m25 = _mm256_insertf128_ps(m25 ,_mm_loadu_ps( & addr[20] ) , 1 );

    __m256 xy = _mm256_shuffle_ps(m14, m25, _MM_SHUFFLE(2,1,3,2));
    __m256 yz = _mm256_shuffle_ps(m03, m14, _MM_SHUFFLE(1,0,2,1));

    vfloat3 v {
        _mm256_shuffle_ps(m03, xy , _MM_SHUFFLE(2,0,3,0)),
        _mm256_shuffle_ps(yz , xy , _MM_SHUFFLE(3,1,2,0)),
        _mm256_shuffle_ps(yz , m25, _MM_SHUFFLE(3,0,3,1))
    };
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
    __m256 rxy = _mm256_shuffle_ps(v.x,v.y, _MM_SHUFFLE(2,0,2,0));
    __m256 ryz = _mm256_shuffle_ps(v.y,v.z, _MM_SHUFFLE(3,1,3,1));
    __m256 rzx = _mm256_shuffle_ps(v.z,v.x, _MM_SHUFFLE(3,1,2,0));

    __m256 r03 = _mm256_shuffle_ps(rxy, rzx, _MM_SHUFFLE(2,0,2,0));
    __m256 r14 = _mm256_shuffle_ps(ryz, rxy, _MM_SHUFFLE(3,1,2,0));
    __m256 r25 = _mm256_shuffle_ps(rzx, ryz, _MM_SHUFFLE(3,1,3,1));

    _mm_storeu_ps( & addr [ 0], _mm256_castps256_ps128( r03 ) );
    _mm_storeu_ps( & addr [ 4], _mm256_castps256_ps128( r14 ) );
    _mm_storeu_ps( & addr [ 8], _mm256_castps256_ps128( r25 ) );
    _mm_storeu_ps( & addr [12], _mm256_extractf128_ps( r03, 1 ) );
    _mm_storeu_ps( & addr [16], _mm256_extractf128_ps( r14, 1 ) );
    _mm_storeu_ps( & addr [20], _mm256_extractf128_ps( r25, 1 ) );
}

/**
 * @brief Vector version of the int2 type holding 2 (.x, .y) vectors
 * 
 */
struct alignas(__m256i) vint2 {
    __m256i x, y;
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

    // AVX has no shuffle operation with 2 integer vectors so we treat data as floats
    // 
    __m256 m02, m13;
    m02 = _mm256_castps128_ps256(_mm_loadu_ps(   (float *) & addr[ 0] )   );
    m13 = _mm256_castps128_ps256(_mm_loadu_ps(   (float *) & addr[ 4] )   );

    m02 = _mm256_insertf128_ps(m02 ,_mm_loadu_ps((float *) & addr[ 8] ),1 );
    m13 = _mm256_insertf128_ps(m13 ,_mm_loadu_ps((float *) & addr[12] ),1 );

    return vint2 {
        _mm256_castps_si256( _mm256_shuffle_ps( m02, m13, _MM_SHUFFLE( 2, 0, 2, 0 ) ) ),
        _mm256_castps_si256( _mm256_shuffle_ps( m02, m13, _MM_SHUFFLE( 3, 1, 3, 1 ) ) )
    };

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
    __m256i r02, r13;
    r02 = _mm256_unpacklo_epi32( v.x, v.y );
    r13 = _mm256_unpackhi_epi32( v.x, v.y );
    _mm_storeu_si128((__m128i *) & addr[ 0], _mm256_castsi256_si128(   r02 ) );
    _mm_storeu_si128((__m128i *) & addr[ 4], _mm256_castsi256_si128(   r13 ) );
    _mm_storeu_si128((__m128i *) & addr[ 8], _mm256_extracti128_si256( r02 ,1 ) );
    _mm_storeu_si128((__m128i *) & addr[12], _mm256_extracti128_si256( r13 ,1 ) ); 
}

static inline vint2 vint2_zero( ) {
    vint2 v{ _mm256_setzero_si256(), _mm256_setzero_si256() };
    return v;
}

/**
 * @brief Vector version of the __mmask16 type holding 2 (.x, .y) masks
 * 
 */
struct alignas(__m256i) vmask2 {
    __m256i x, y;
};

class Vec8Float {
    union {
        __m256 v;
        float s[8];
    } data;
    public:
    Vec8Float( const __m256 v ) { data.v = v; }
    Vec8Float( const float s ) { data.v = _mm256_set1_ps(s); }
    float extract( const int i ) { return data.s[ i ]; }
    friend std::ostream& operator<<(std::ostream& os, const Vec8Float& obj) { 
        os << obj.data.v;
        return os;
    }
};

class Vec8Int {
    union {
        __m256i v;
        int s[8];
    } data;
    public:
    Vec8Int( const __m256i v ) { data.v = v; }
    Vec8Int( const int s ) { data.v = _mm256_set1_epi32(s); }
    int extract( const int i ) const { return data.s[ i ]; }
    friend std::ostream& operator<<(std::ostream& os, const Vec8Int& obj) { 
        os << obj.data.v;
        return os;
    }
};

class Vec8Mask {
    union {
        __m256i v;
        int s[8];
    } data;
    public:
    Vec8Mask( const __m256i v ) { data.v = v; }
    int extract( const unsigned i ) const { 
        return data.s[ i ] ;
    }
    friend std::ostream& operator<<(std::ostream& os, const Vec8Mask& obj) { 
        for( unsigned i = 0; i < 8; i ++) 
            os << (( obj.extract(i) == 0 ) ? 0 : 1 );
        return os;
    }
};

constexpr int vecwidth = 8;
typedef __m256 vfloat;
typedef __m256i vint;
typedef __m256i vmask;
typedef Vec8Float VecFloat_s;
typedef Vec8Int VecInt_s;
typedef Vec8Mask VecMask_s;

template< typename T >
constexpr int is_aligned_32( T * addr ) { 
    return (((uintptr_t)addr & 0x1F) == 0);
}

template< typename T >
void assert_aligned_32( T * addr, std::string msg ) { 
    if ( ! is_aligned_32(addr) ) {
        std::cerr << msg << '\n';
        std::cerr << "Address " << addr << " is not 32 bit aligned, aborting." << std::endl;
        abort();
    }
}


template< typename T >
constexpr int is_aligned_16( T * addr ) { return (((uintptr_t)addr & 0x0F) == 0); }

template< typename T >
void assert_aligned_16( T * addr, std::string msg ) { 
    if ( ! is_aligned_16(addr) ) {
        std::cerr << msg << '\n';
        std::cerr << "Address " << addr << " is not 16 bit aligned, aborting." << std::endl;
        abort();
    }
}
