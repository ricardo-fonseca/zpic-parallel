#include <immintrin.h>

#include <iostream>
#include <iomanip>




static inline __m512 _mm512_set_m256( __m256 hi, __m256 lo ) {
    // Using AVX512F _mm512_insertf64x4
    return _mm512_castpd_ps(_mm512_insertf64x4(_mm512_castps_pd(_mm512_castps256_ps512(lo)), _mm256_castps_pd(hi), 1));

    // Using AVX512DQ _mm512_insertf32x8
    // return _mm512_insertf32x8( _mm512_castps256_ps512(lo), hi, 1 );
}

/**
 * @brief Floating point (32 bit) SIMD types
 * 
 * @note For AVX this corresponds to the __m512 vector
 */

template< int imm > 
static inline float vec_extract( const __m256 v ) {
    static_assert( imm >= 0 && imm < 8, "imm must be in the range [0..7]" );
    
    // The compiler will (usually) optimize this and avoid the memory copy
    alignas( __m256 ) float buf[8];
    _mm256_store_ps( buf, v );
    return buf[imm];
}

static inline
std::ostream& operator<<(std::ostream& os, const __m256 v) {
    os << "[";
    os <<         vec_extract< 0>( v );
    os << ", " << vec_extract< 1>( v );
    os << ", " << vec_extract< 2>( v );
    os << ", " << vec_extract< 3>( v );
    os << ", " << vec_extract< 4>( v );
    os << ", " << vec_extract< 5>( v );
    os << ", " << vec_extract< 6>( v );
    os << ", " << vec_extract< 7>( v );
    os << "]";

    return os;
}

/**
 * @brief Extract a single float from a _mm512 vector
 * 
 * @tparam imm      Which value to extract
 * @param v         Input vector
 * @return float    Selected value
 */
template< int imm > 
static inline float vec_extract( const __m512 v ) {
    static_assert( imm >= 0 && imm < 16, "imm must be in the range [0..15]" );
    
    // The compiler will (usually) optimize this and avoid the memory copy
    alignas( __m512 ) float buf[16];
    _mm512_store_ps( buf, v );
    return buf[imm];
}

static inline float vec_extract( const __m512 v, int i ) {
    union {
        __m512 v;
        float s[16];
    } b;
    b.v = v;
    return b.s[i & 15];

    // __mmask16 mask = (1 << i);
    // return _mm512_cvtss_f32( _mm512_maskz_compress_ps( mask, v ) );

}

/**
 * @brief Writes the textual representation of vector v to os
 * 
 * @param os    Output stream
 * @param v     Float vector value
 * @return std::ostream& 
 */
static inline
std::ostream& operator<<(std::ostream& os, const __m512 v) {
    os << "[";
    os <<         vec_extract< 0>( v );
    os << ", " << vec_extract< 1>( v );
    os << ", " << vec_extract< 2>( v );
    os << ", " << vec_extract< 3>( v );
    os << ", " << vec_extract< 4>( v );
    os << ", " << vec_extract< 5>( v );
    os << ", " << vec_extract< 6>( v );
    os << ", " << vec_extract< 7>( v );
    os << ", " << vec_extract< 8>( v );
    os << ", " << vec_extract< 9>( v );
    os << ", " << vec_extract<10>( v );
    os << ", " << vec_extract<11>( v );
    os << ", " << vec_extract<12>( v );
    os << ", " << vec_extract<13>( v );
    os << ", " << vec_extract<14>( v );
    os << ", " << vec_extract<15>( v );
    os << "]";

    return os;
}

/**
 * @brief Returns a zero valued vector
 * 
 * @return __m512 
 */
static inline __m512 vec_zero_float() {
    return _mm512_setzero_ps();
}

/**
 * @brief Create a vector with all elements equal to scalar value
 * 
 * @param s 
 * @return __m512 
 */
static inline __m512 vec_float( float s ) {
    return _mm512_set1_ps(s);
}

/**
 * @brief Create a float vector from an integer vector
 * 
 * @param vi 
 * @return __m512 
 */
static inline __m512 vec_float( __m512i vi ) {
    return _mm512_cvtepi32_ps( vi );
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
 * @return __m512 
 */
static inline __m512 vec_float( 
    float e0, float e1, float e2, float e3, float e4, float e5, float e6, float e7,
    float e8, float e9, float e10, float e11, float e12, float e13, float e14, float e15 ) 
{
    return _mm512_setr_ps( e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15 );
}

/**
 * @brief Loads vector from memory
 * 
 * @warning The address must be aligned to a 32 byte boundary
 * 
 * @param mem_addr 
 * @return __m512 
 */
static inline __m512 vec_load( const float * mem_addr) { 
    return _mm512_load_ps( mem_addr );
}

/**
 * @brief Stores vector to memory
 * 
 * @warning The address must be aligned to a 32 byte boundary
 * 
 * @param mem_addr 
 * @param a 
 */
static inline __m512 vec_store( float * mem_addr, __m512 a ) {
    _mm512_store_ps( mem_addr, a ); return a;
}

static inline __m512 vec_neg( __m512 a ) {
    return _mm512_sub_ps( _mm512_setzero_ps(), a );
}

/**
 * @brief Adds 2 vector values (a+b)
 * 
 * @param a 
 * @param b 
 * @return __m512 
 */
static inline __m512 vec_add( __m512 a, __m512 b ) {
    return _mm512_add_ps(a,b);
}

/**
 * @brief Adds scalar value (s) to all components of vector value (a). 
 * 
 * @param a 
 * @param b 
 * @return __m512 
 */
static inline __m512 vec_add( __m512 a, float s ) { 
    return _mm512_add_ps(a,_mm512_set1_ps(s));
}

/**
 * @brief Subtracts 2 vector values (a-b)
 * 
 * @param a 
 * @param b 
 * @return __m512 
 */
static inline __m512 vec_sub( __m512 a, __m512 b ) {
    return _mm512_sub_ps(a,b);
}

/**
 * @brief Multiplies 2 vector values (a*b)
 * 
 * @param a 
 * @param b 
 * @return __m512 
 */
static inline __m512 vec_mul( __m512 a, __m512 b ) { 
    return _mm512_mul_ps(a,b);
}

/**
 * @brief Multiplies vector (a) by scalar value (s)
 * 
 * @param a 
 * @param s 
 * @return __m512 
 */
static inline __m512 vec_mul( __m512 a, float s ) {
    return _mm512_mul_ps(a,_mm512_set1_ps(s));
}

/**
 * @brief Divides 2 vector values (a/b)
 * 
 * @param a 
 * @param b 
 * @return __m512 
 */
static inline __m512 vec_div( __m512 a, __m512 b ) {
    return _mm512_div_ps(a,b);
}

/**
 * @brief Compares 2 vector values (by element) for equality (==)
 * 
 * @param a 
 * @param b 
 * @return __mmask16
 */
static inline __mmask16 vec_eq( __m512 a, __m512 b ) { 
    return _mm512_cmp_ps_mask(a,b,_CMP_EQ_OQ);
}

/**
 * @brief Compares 2 vector values (by element) for inequality (!=)
 * 
 * @param a 
 * @param b 
 * @return __mmask16
 */
static inline __mmask16 vec_ne( __m512 a, __m512 b ) { 
    return _mm512_cmp_ps_mask(a,b,_CMP_NEQ_OQ);
}

/**
 * @brief Compares 2 vector values (by element) for "greater-than" (>)
 * 
 * @param a 
 * @param b 
 * @return __mmask16 
 */
static inline __mmask16 vec_gt( __m512 a, __m512 b ) { 
    return _mm512_cmp_ps_mask(a,b,_CMP_GT_OQ);
}

/**
 * @brief Compares 2 vector values (by element) for "greater of equal" (>=) and return mask
 * 
 * @param a 
 * @param b 
 * @return      Resulting mask, for each element i 0 if false, -1 if true
 */
static inline __mmask16 vec_ge( __m512 a, __m512 b ) { 
    return _mm512_cmp_ps_mask(a,b,_CMP_GE_OQ);
}

/**
 * @brief Compares 2 vector values (by element) for "greater of equal" (>=) and return v (float)
 * 
 * @param a 
 * @param b 
 * @param v
 * @return      Result, for each element i, 0 if false and v[i] if true
 */
static inline __m512 vec_ge( __m512 a, __m512 b, __m512 v ) { 
    const __mmask16 mask = _mm512_cmp_ps_mask(a,b,_CMP_GE_OQ);
    __m512 res = _mm512_setzero_ps();
    return _mm512_mask_mov_ps( res, mask, v );
}

/**
 * @brief Compares 2 vector values (by element) for "greater of equal" (>=) and return v (integer)
 * 
 * @param a 
 * @param b 
 * @param vi 
 * @return      Result, for each element i, 0 if false and v[i] if true
 */
static inline __m512i vec_ge( __m512 a, __m512 b, __m512i vi ) {
    const __mmask16 mask = _mm512_cmp_ps_mask(a,b,_CMP_GE_OQ);
    __m512i res = _mm512_setzero_epi32();
    return _mm512_mask_mov_epi32( res, mask, vi );
}


/**
 * @brief Compares 2 vector values (by element) for "less than" (<) and return mask
 * 
 * @param a 
 * @param b 
 * @return      Resulting mask, for each element i 0 if false, -1 if true
 */
static inline __mmask16 vec_lt( __m512 a, __m512 b ) { 
    return _mm512_cmp_ps_mask(a,b,_CMP_LT_OQ);
}

/**
 * @brief Compares 2 vector values (by element) for "less than" (<) and return value
 * 
 * @param a     Value a
 * @param b     Value b
 * @param v     Result, for each element i, 0 if false and v[i] if true
 * @return __m512 
 */
static inline __m512 vec_lt( __m512 a, __m512 b, __m512 v ) { 
    const __mmask16 mask = _mm512_cmp_ps_mask(a,b,_CMP_LT_OQ);
    __m512 res = _mm512_setzero_ps();
    return _mm512_mask_mov_ps( res, mask, v );
}

/**
 * @brief Compares 2 vector values (by element) for "less than" (<) and return value (integer)
 * 
 * @param a     Value a
 * @param b     Value b
 * @param v     Result, for each element i, 0 if false and v[i] if true
 * @return __m512 
 */

static inline __m512i vec_lt( __m512 a, __m512 b, __m512i vi ) { 
    const __mmask16 mask = _mm512_cmp_ps_mask(a,b,_CMP_LT_OQ);
    __m512i res = _mm512_setzero_epi32();
    return _mm512_mask_mov_epi32( res, mask, vi );
}

/**
 * @brief Compares 2 vector values (by element) for "less or equal" (<=)
 * 
 * @param a 
 * @param b 
 * @return __m512 
 */
static inline __mmask16 vec_le( __m512 a, __m512 b ) { 
    return _mm512_cmp_ps_mask(a,b,_CMP_LE_OQ);
}

/**
 * @brief Fused multiply add: (a*b)+c
 * 
 * @param a 
 * @param b 
 * @param c 
 * @return __m512 
 */
static inline __m512 vec_fmadd( __m512 a, __m512 b, __m512 c ) { 
    return _mm512_fmadd_ps( a, b, c );
}

/**
 * @brief Fused multiply subtract: (a*b)-c
 * 
 * @param a 
 * @param b 
 * @param c 
 * @return __m512 
 */
static inline __m512 vec_fmsub( __m512 a, __m512 b, __m512 c ) { 
    return _mm512_fmsub_ps( a, b, c );
}

/**
 * @brief Fused negate multiply add: -(a*b)+c
 * 
 * @param a 
 * @param b 
 * @param c 
 * @return __m512 
 */
static inline __m512 vec_fnmadd( __m512 a, __m512 b, __m512 c ) {
    return _mm512_fnmadd_ps( a, b, c );
}

/**
 * @brief Reciprocal (1/a)
 * 
 * @param a
 * @return __m512 
 */
static inline __m512 vec_recp( const __m512 a )
{
    // Full calculation
    auto recp = _mm512_div_ps( _mm512_set1_ps( 1 ), a );

/*
    // Fast estimate + 1 Newton-Raphson iteration
    auto recp = _mm512_rcp_ps( a );
    recp = _mm512_mul_ps(recp, _mm512_fnmadd_ps(recp, a, _mm512_set1_ps( 2 )));
*/

    return recp;
}

/**
 * @brief Reciprocal square root 1/sqrt(a)
 * 
 * @param a 
 * @return __m512 
 */
static inline __m512 vec_rsqrt( const __m512 a ) {

    auto rsqrt = _mm512_div_ps( _mm512_set1_ps(1), _mm512_sqrt_ps(a) );

/*
    // Fast estimate + 1 Newton-Raphson iteration
    auto rsqrt = _mm512_rsqrt_ps( a );
    auto const c1_2 = _mm512_set1_ps( 0.5 );
    auto const c3   = _mm512_set1_ps( 3 );
    rsqrt = _mm512_mul_ps( c1_2, 
                _mm512_mul_ps( rsqrt, 
                    _mm512_fnmadd_ps( _mm512_mul_ps( rsqrt, rsqrt ), a, c3 )
                )
            );
*/

    return rsqrt;
}

/**
 * @brief Square root
 * 
 * @param a 
 * @return __m512 
 */
static inline __m512 vec_sqrt( const __m512 a ) {
    return _mm512_sqrt_ps(a);
}

/**
 * @brief Absolute value
 * 
 * @param a 
 * @return __m512 
 */
static inline __m512 vec_fabs( const __m512 a ) { 
    return _mm512_abs_ps(a);
}

/**
 * @brief Selects between vector elements of vectors a and b according to the mask
 * 
 * @param a     a vector
 * @param b     b vector
 * @param mask  selection mask, bit value 0 selects a vector element, 1 selects b vector element
 * @return __m512i 
 */
static inline __m512 vec_select( const __m512 a, const __m512 b, const __mmask16 mask ) {
    return _mm512_mask_blend_ps( mask, a, b );
}


/**
 * @brief Add all vector elements
 * 
 * @param a 
 * @return float 
 */
static inline float vec_reduce_add( const __m512 a ) {
   
   return _mm512_reduce_add_ps( a );
}

/**
 * @brief Gather values from base address + vector index
 * 
 * @param base_addr 
 * @param vindex 
 * @return __m512 
 */
static inline __m512 vec_gather( float const * __restrict__ base_addr, __m512i vindex ) {

/*
    // This has terrible performance
    __m512 v = _mm512_i32gather_ps( vindex, base_addr, 4 );
*/

    union { __m512i v; int s[16]; } index;
    index.v = vindex;

    __m128 a = _mm_setr_ps (
        base_addr[index.s[ 0]],
        base_addr[index.s[ 1]],
        base_addr[index.s[ 2]],
        base_addr[index.s[ 3]]
    );

    __m128 b = _mm_setr_ps (
        base_addr[index.s[ 4]],
        base_addr[index.s[ 5]],
        base_addr[index.s[ 6]],
        base_addr[index.s[ 7]]
    );

    __m128 c = _mm_setr_ps (
        base_addr[index.s[ 8]],
        base_addr[index.s[ 9]],
        base_addr[index.s[10]],
        base_addr[index.s[11]]
    );

    __m128 d = _mm_setr_ps (
        base_addr[index.s[12]],
        base_addr[index.s[13]],
        base_addr[index.s[14]],
        base_addr[index.s[15]]
    );

    __m256 lo = _mm256_set_m128( b, a );
    __m256 hi = _mm256_set_m128( d, c );
    __m512 v  = _mm512_set_m256( hi, lo );

#if 0
    union { __m512i v; int s[16]; } index;
    index.v = vindex;

    alignas( __m512 ) float buffer[16];

    buffer[0]  = base_addr[ index.s[0] ];
    buffer[1]  = base_addr[ index.s[1] ];
    buffer[2]  = base_addr[ index.s[2] ];
    buffer[3]  = base_addr[ index.s[3] ];
    buffer[4]  = base_addr[ index.s[4] ];
    buffer[5]  = base_addr[ index.s[5] ];
    buffer[6]  = base_addr[ index.s[6] ];
    buffer[7]  = base_addr[ index.s[7] ];
    buffer[8]  = base_addr[ index.s[8] ];
    buffer[9]  = base_addr[ index.s[9] ];
    buffer[10] = base_addr[ index.s[10] ];
    buffer[11] = base_addr[ index.s[11] ];
    buffer[12] = base_addr[ index.s[12] ];
    buffer[13] = base_addr[ index.s[13] ];
    buffer[14] = base_addr[ index.s[14] ];
    buffer[15] = base_addr[ index.s[15] ];

/*    // 6.39 s
     __m512 v  = _mm512_load_ps( buffer ); */

/*    // 6.23 s
    __m512 v = _mm512_setr_ps( 
        buffer[ 0], buffer[ 1], buffer[ 2], buffer[ 3],
        buffer[ 4], buffer[ 5], buffer[ 6], buffer[ 7],
        buffer[ 8], buffer[ 9], buffer[10], buffer[11],
        buffer[12], buffer[13], buffer[14], buffer[15]
    ); */

/*  // 9.33 s
    __m256 lo = _mm256_load_ps( &buffer[0] );
    __m256 hi = _mm256_load_ps( &buffer[8] );
    __m512 v  = _mm512_set_m256( hi, lo ); */

/*
    // 5.26 s
    __m128 a = _mm_load_ps( & buffer[0] );
    __m128 b = _mm_load_ps( & buffer[4] );
    __m128 c = _mm_load_ps( & buffer[8] );
    __m128 d = _mm_load_ps( & buffer[12] );

    __m256 lo = _mm256_set_m128( b, a );
    __m256 hi = _mm256_set_m128( d, c );
    __m512 v  = _mm512_set_m256( hi, lo ); */

    __m128 a = _mm_setr_ps (
        buffer[ 0],
        buffer[ 1],
        buffer[ 2],
        buffer[ 3]
    );

    __m128 b = _mm_setr_ps (
        buffer[ 4],
        buffer[ 5],
        buffer[ 6],
        buffer[ 7]
    );

    __m128 c = _mm_setr_ps (
        buffer[ 8],
        buffer[ 9],
        buffer[10],
        buffer[11]
    );

    __m128 d = _mm_setr_ps (
        buffer[12],
        buffer[13],
        buffer[14],
        buffer[15]
    );

    __m256 lo = _mm256_set_m128( b, a );
    __m256 hi = _mm256_set_m128( d, c );
    __m512 v  = _mm512_set_m256( hi, lo );
#endif

#if 0
    // 5.25 s
    union { __m512i v; int s[16]; } index;
    index.v = vindex;

    float buffer0  = base_addr[ index.s[0] ];
    float buffer1  = base_addr[ index.s[1] ];
    float buffer2  = base_addr[ index.s[2] ];
    float buffer3  = base_addr[ index.s[3] ];
    float buffer4  = base_addr[ index.s[4] ];
    float buffer5  = base_addr[ index.s[5] ];
    float buffer6  = base_addr[ index.s[6] ];
    float buffer7  = base_addr[ index.s[7] ];
    float buffer8  = base_addr[ index.s[8] ];
    float buffer9  = base_addr[ index.s[9] ];
    float buffer10 = base_addr[ index.s[10] ];
    float buffer11 = base_addr[ index.s[11] ];
    float buffer12 = base_addr[ index.s[12] ];
    float buffer13 = base_addr[ index.s[13] ];
    float buffer14 = base_addr[ index.s[14] ];
    float buffer15 = base_addr[ index.s[15] ];

    __m128 a = _mm_setr_ps (
        buffer0,
        buffer1,
        buffer2,
        buffer3
    );

    __m128 b = _mm_setr_ps (
        buffer4,
        buffer5,
        buffer6,
        buffer7
    );

    __m128 c = _mm_setr_ps (
        buffer8,
        buffer9,
        buffer10,
        buffer11
    );

    __m128 d = _mm_setr_ps (
        buffer12,
        buffer13,
        buffer14,
        buffer15
    );

    __m256 lo = _mm256_set_m128( b, a );
    __m256 hi = _mm256_set_m128( d, c );
    __m512 v  = _mm512_set_m256( hi, lo );
#endif

    return v;
}

/**
 * @brief Integer (32 bit) SIMD types
 * 
 * @note For AVX this corresponds to the __m512i vector
 */

/**
 * @brief Extract a single integer from a _mm512i vector
 * 
 * @tparam imm      Which value to extract
 * @param v         Input vector
 * @return float    Selected value
 */

template< int imm > 
static inline int vec_extract( const __m512i v ) {
    static_assert( imm >= 0 && imm < 16, "imm must be in the range [0..15]" );
    
    // The compiler will (usually) optimize this and avoid the memory copy
    alignas( __m512i ) int buf[16];
    _mm512_store_si512( ( __m512i * ) buf, v );
    return buf[imm];
}

static inline int vec_extract( const __m512i v, int i ) {
    union {
        __m512i v;
        int32_t s[16];
    } b;
    b.v = v;
    return b.s[i & 15];

    // __mmask16 mask = (1 << i);
    // return _mm512_cvtsi512_si32( _mm512_maskz_compress_epi32( mask, v ) );
}

/**
 * @brief Writes the textual representation of vector v to os
 * 
 * @param os    Output stream
 * @param v     int vector value
 * @return std::ostream& 
 */
static inline
std::ostream& operator<<(std::ostream& os, const __m512i v) {
    os << "[";
    os <<         vec_extract< 0>( v );
    os << ", " << vec_extract< 1>( v );
    os << ", " << vec_extract< 2>( v );
    os << ", " << vec_extract< 3>( v );
    os << ", " << vec_extract< 4>( v );
    os << ", " << vec_extract< 5>( v );
    os << ", " << vec_extract< 6>( v );
    os << ", " << vec_extract< 7>( v );
    os << ", " << vec_extract< 8>( v );
    os << ", " << vec_extract< 9>( v );
    os << ", " << vec_extract<10>( v );
    os << ", " << vec_extract<11>( v );
    os << ", " << vec_extract<12>( v );
    os << ", " << vec_extract<13>( v );
    os << ", " << vec_extract<14>( v );
    os << ", " << vec_extract<15>( v );
    os << "]";

    return os;
}

/**
 * @brief Returns a zero valued vector
 * 
 * @return __m512i 
 */
static inline __m512i vec_zero_int() {
    return _mm512_setzero_si512();
}

/**
 * @brief Create a vector with all elements equal to scalar value
 * 
 * @param a 
 * @return __m512i 
 */
static inline __m512i vec_int( int s ) {
    return _mm512_set1_epi32(s);
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
 * @return __m512i 
 */
static inline __m512i vec_int( 
    int e0, int e1, int e2, int e3, int e4, int e5, int e6, int e7,
    int e8, int e9, int e10, int e11, int e12, int e13, int e14, int e15 ) 
{
    return _mm512_setr_epi32( e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15 );
}

/**
 * @brief Loads vector from memory
 * 
 * @warning The address must be aligned to a 32 byte boundary
 * 
 * @param mem_addr 
 * @return __m512i 
 */
static inline __m512i vec_load( const int * mem_addr) { 
    return _mm512_load_epi32( mem_addr );
}

/**
 * @brief Stores vector to memory
 * 
 * @warning The address must be aligned to a 32 byte boundary
 * 
 * @param mem_addr 
 * @param a 
 */
static inline __m512i vec_store( int * mem_addr, __m512i a ) { 
    _mm512_store_epi32( mem_addr, a ); return a;
}

/**
 * @brief Adds 2 vector values (a+b)
 * 
 * @param a 
 * @param b 
 * @return __m512i 
 */
static inline __m512i vec_add( __m512i a, __m512i b ) { 
    return _mm512_add_epi32(a,b);
}

/**
 * @brief Subtracts 2 vector values (a-b)
 * 
 * @param a 
 * @param b 
 * @return __m512i 
 */
static inline __m512i vec_sub( __m512i a, __m512i b ) {
    return _mm512_sub_epi32(a,b);
}

/**
 * @brief Adds scalar value (s) to all components of vector value (a). 
 * 
 * @param a 
 * @param b 
 * @return __m512i 
 */
static inline __m512i vec_add( __m512i a, int s ) { 
    return _mm512_add_epi32(a, _mm512_set1_epi32(s) );
}

/**
 * @brief Multiplies 2 vector values (a*b)
 * 
 * @param a 
 * @param b 
 * @return __m512i 
 */
static inline __m512i vec_mul( __m512i a, __m512i b ) {
    return _mm512_mullo_epi32(a,b);
}

/**
 * @brief Multiplies vector (a) by scalar value (s)
 * 
 * @param a 
 * @param b 
 * @return __m512i 
 */
static inline __m512i vec_mul( __m512i a, int s ) { 
    return _mm512_mullo_epi32(a, _mm512_set1_epi32(s) );
}

/**
 * @brief Multiplies vector by 3
 * 
 * @param a 
 * @return __m512i 
 */
static inline __m512i vec_mul3( __m512i a ) {
    return _mm512_add_epi32( _mm512_add_epi32( a, a ), a );
}

/**
 * @brief Compares 2 vector values (by element) for equality (==)
 * 
 * @param a 
 * @param b 
 * @return __m512i 
 */
static inline __mmask16 vec_eq( __m512i a, __m512i b ) { 
    return _mm512_cmpeq_epi32_mask( a, b );
}

/**
 * @brief Compares 2 vector values (by element) for inequality (!=)
 * 
 * @param a 
 * @param b 
 * @return __m512i 
 */
static inline __mmask16 vec_ne( __m512i a, __m512i b ) { 
    return _mm512_cmpneq_epi32_mask( a, b );
}

/**
 * @brief Compares 2 vector values (by element) for "greater-than" (>)
 * 
 * @param a 
 * @param b 
 * @return __m512i 
 */
static inline __mmask16 vec_gt( __m512i a, __m512i b ) { 
    return _mm512_cmpgt_epi32_mask( a, b );
}

/**
 * @brief Compares 2 vector values (by element) for "less-than" (<)
 * 
 * @param a 
 * @param b 
 * @return __m512i 
 */
static inline __mmask16 vec_lt( __m512i a, __m512i b ) { 
    return _mm512_cmplt_epi32_mask( b, a );
}

/**
 * @brief Absolute value
 * 
 * @param a 
 * @return __m512i 
 */
static inline __m512i vec_abs( __m512i a ) {
    return _mm512_abs_epi32( a );
}

/**
 * @brief Selects between vector elements of vectors a and b according to the mask
 * 
 * @param a     a vector
 * @param b     b vector
 * @param mask  selection mask, 0 selects a vector element, -1 selects b vector element
 * @return __m512i 
 */
static inline __m512i vec_select( const __m512i a, const __m512i b, const __mmask16 mask ) {
    return _mm512_mask_blend_epi32( mask, a, b );
}

/**
 * @brief Bitwise complement (not)
 * 
 * @param a 
 * @return __mmask16 
 */
static inline __mmask16 vec_not( __mmask16 a ) {
    return _mm512_knot(a);
}

/**
 * @brief Bitwise or 
 * 
 * @param a 
 * @param b 
 * @return __m512i 
 */
static inline __mmask16 vec_or( __mmask16 a, __mmask16 b ) {
    return  _mm512_kor( a, b );
}

/**
 * @brief Bitwise and 
 * 
 * @param a 
 * @param b 
 * @return __m512i 
 */
static inline __mmask16 vec_and( __mmask16 a, __mmask16 b ) {
    return _mm512_kand( a, b );
}

/**
 * @brief Returns true (1) if all of the mask values are 1
 * 
 * @param mask 
 * @return int 
 */
static inline int vec_all( __mmask16 mask ) {
    return mask == 0xFFFF;
}

/**
 * @brief Returns true (1) if any of the mask values is 1
 * 
 * @param mask 
 * @return int 
 */
static inline int vec_any( const __mmask16 mask ) {
    return mask != 0;
}


static inline __mmask16 vec_true() { 
    return -1;
}

static inline __mmask16 vec_false() { 
    return 0;
}


/**
 * @brief Vector version of the float2 type holding 2 (.x, .y) vectors
 * 
 */
struct alignas(__m512) vfloat2 {
    __m512 x, y;
};

/**
 * @brief Returs a zero valued vfloat2
 * 
 * @return vfloat2 
 */
static inline vfloat2 vfloat2_zero(  ) {
    vfloat2 v{ _mm512_setzero_ps(), _mm512_setzero_ps() };
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

    const __m512i perm = _mm512_set_epi32(15,13,11, 9, 7, 5, 3, 1,14,12,10, 8, 6, 4, 2, 0);

    __m512i t0 = _mm512_permutexvar_epi32( perm, _mm512_loadu_epi32(&addr[ 0]) );
    __m512i t1 = _mm512_permutexvar_epi32( perm, _mm512_loadu_epi32(&addr[16]) );
    
    vfloat2 v { 
        _mm512_castsi512_ps( _mm512_mask_alignr_epi32( t0, 0xFF00, t1, t1, 8 ) ),
        _mm512_castsi512_ps( _mm512_mask_alignr_epi32( t1, 0x00FF, t0, t0, 8 ) )
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

    const __m512i perm0 =  _mm512_set_epi32(15, 7,14, 6,13, 5,12, 4,11, 3,10, 2, 9, 1, 8, 0);
    __m512i t0 = _mm512_permutexvar_epi32( perm0, _mm512_castps_si512(v.x) );
    __m512i t1 = _mm512_permutexvar_epi32( perm0, _mm512_castps_si512(v.y) );
    _mm512_store_epi32( &addr[ 0], _mm512_mask_alignr_epi32( t0, 0xAAAA, t1, t1, 15 ) );
    _mm512_store_epi32( &addr[16], _mm512_mask_alignr_epi32( t1, 0x5555, t0, t0, 1 ) ); 

}

/**
 * @brief Vector version of the float3 type holding 3 (.x, .y, .z) vectors
 * 
 */
struct alignas(__m512) vfloat3 {
    __m512 x, y, z;
};

/**
 * @brief Returs a zero valued vfloat3
 * 
 * @return vfloat3 
 */
static inline vfloat3 vfloat3_zero( ) {
    vfloat3 v{ _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
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

    const __m512i perm0 = _mm512_set_epi32(14,11,8,5,2,13,10,7,4,1,15,12,9,6,3,0);
    __m512i t0 = _mm512_permutexvar_epi32( perm0, _mm512_loadu_epi32(&addr[ 0]) );
    __m512i t1 = _mm512_permutexvar_epi32( perm0, _mm512_loadu_epi32(&addr[16]) );
    __m512i t2 = _mm512_permutexvar_epi32( perm0, _mm512_loadu_epi32(&addr[32]) );
    __m512i vx = _mm512_mask_alignr_epi32( t0, 0x07C0, t1, t1, 5 );
    __m512i vy = _mm512_mask_alignr_epi32( t2, 0x001F, t0, t0, 6 );
    __m512i vz = _mm512_mask_alignr_epi32( _mm512_alignr_epi32( t0, t0, 11 ), 0x03E0,t1, t1, 1 );

    vfloat3 v {
        _mm512_castsi512_ps( _mm512_mask_alignr_epi32( vx, 0xF800, t2, t2, 11 ) ),
        _mm512_castsi512_ps( _mm512_mask_alignr_epi32( vy, 0x07E0, t1, t1, 11 ) ),
        _mm512_castsi512_ps( _mm512_mask_alignr_epi32( vz, 0xFC00, t2, t2,  6 ) )
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
    const __m512i perm0 =  _mm512_set_epi32( 5,10,15, 4, 9,14, 3, 8,13, 2, 7,12, 1, 6,11, 0);
    __m512i vx = _mm512_permutexvar_epi32( perm0, _mm512_castps_si512( v.x ) );
    __m512i vy = _mm512_permutexvar_epi32( perm0, _mm512_castps_si512( v.y ) );
    __m512i vz = _mm512_permutexvar_epi32( perm0, _mm512_castps_si512( v.z ) );
    vy = _mm512_alignr_epi32( vy, vy, 15 );
    vz = _mm512_alignr_epi32( vz, vz, 14 );
    __m512i t0 = _mm512_mask_mov_epi32( _mm512_mask_mov_epi32( vx, 0x2492, vy ), 0x4924, vz );
    __m512i t1 = _mm512_mask_mov_epi32( _mm512_mask_mov_epi32( vx, 0x9249, vy ), 0x2492, vz );
    __m512i t2 = _mm512_mask_mov_epi32( _mm512_mask_mov_epi32( vx, 0x4924, vy ), 0x9249, vz );
    _mm512_storeu_epi32( &addr[ 0], t0 );
    _mm512_storeu_epi32( &addr[16], t1 );
    _mm512_storeu_epi32( &addr[32], t2 );
}

/**
 * @brief Vector version of the int2 type holding 2 (.x, .y) vectors
 * 
 */
struct alignas(__m512i) vint2 {
    __m512i x, y;
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

    const __m512i perm =  _mm512_set_epi32(15,13,11, 9, 7, 5, 3, 1,14,12,10, 8, 6, 4, 2, 0);
    __m512i t0 = _mm512_permutexvar_epi32( perm, _mm512_loadu_epi32(&addr[ 0]) );
    __m512i t1 = _mm512_permutexvar_epi32( perm, _mm512_loadu_epi32(&addr[16]) );

    return vint2 {
        _mm512_mask_alignr_epi32( t0, 0xFF00, t1, t1, 8 ),
        _mm512_mask_alignr_epi32( t1, 0x00FF, t0, t0, 8 )
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
const __m512i perm0 =  _mm512_set_epi32(15, 7,14, 6,13, 5,12, 4,11, 3,10, 2, 9, 1, 8, 0);
  __m512i t0 = _mm512_permutexvar_epi32( perm0, v.x );
  __m512i t1 = _mm512_permutexvar_epi32( perm0, v.y );
  _mm512_storeu_epi32( &addr[ 0], _mm512_mask_alignr_epi32( t0, 0xAAAA, t1, t1, 15 ) );
  _mm512_storeu_epi32( &addr[16], _mm512_mask_alignr_epi32( t1, 0x5555, t0, t0, 1 ) );
}

static inline vint2 vint2_zero( ) {
    vint2 v{ _mm512_setzero_si512(), _mm512_setzero_si512() };
    return v;
}

/**
 * @brief Vector version of the __mmask16 type holding 2 (.x, .y) masks
 * 
 */
struct alignas(__mmask16) vmask2 {
    __mmask16 x, y;
};

class Vec16Float {
    union {
        __m512 v;
        float s[16];
    } data;
    public:
    Vec16Float( const __m512 v ) { data.v = v; }
    Vec16Float( const float s ) { data.v = _mm512_set1_ps(s); }
    float extract( const int i ) { return data.s[ i ]; }
    friend std::ostream& operator<<(std::ostream& os, const Vec16Float& obj) { 
        os << obj.data.v;
        return os;
    }
};

class Vec16Int {
    union {
        __m512i v;
        int s[16];
    } data;
    public:
    Vec16Int( const __m512i v ) { data.v = v; }
    Vec16Int( const int s ) { data.v = _mm512_set1_epi32(s); }
    int extract( const int i ) { return data.s[ i ]; }
    friend std::ostream& operator<<(std::ostream& os, const Vec16Int& obj) { 
        os << obj.data.v;
        return os;
    }
};


/*

// This is much slower than the above version

class Vec16Float {
    __m512 v;
    public:
    Vec16Float( const __m512 v ): v(v) {}
    Vec16Float( const float s ): v(_mm512_set1_ps(s) ) {}
    float extract( const int i ) { 
        __mmask16 mask = (1 << i);
        return _mm512_cvtss_f32( _mm512_maskz_compress_ps( mask, v ) );
    }
    friend std::ostream& operator<<(std::ostream& os, const Vec16Float& obj) { 
        os << obj.v;
        return os;
    }
};

class Vec16Int {
    __m512i v;
    public:
    Vec16Int( const __m512i v ): v(v) {}
    Vec16Int( const int s ): v( _mm512_set1_epi32(s) ) {}
    int extract( const int i ) { 
        __mmask16 mask = (1 << i);
        return _mm512_cvtsi512_si32( _mm512_maskz_compress_epi32( mask, v ) );
    }
    friend std::ostream& operator<<(std::ostream& os, const Vec16Int& obj) { 
        os << obj.v;
        return os;
    }
};
*/

class Vec16Mask {
    __mmask16 mask;
    public:
    Vec16Mask( const __mmask16 v ) { mask = v; }
    int extract( const unsigned i ) const { 
        return mask & (1 << i);
    }
    friend std::ostream& operator<<(std::ostream& os, const Vec16Mask& obj) { 
        for( unsigned i = 0; i < 16; i ++) 
            os << (( obj.extract(i) == 0 ) ? 0 : 1);
        return os;
    }
};



