#pragma once

/**
 * Counter-based PRNG (Philox4x32-10) for per-thread use in CUDA kernels.
 *
 * Salmon, Moraes, Dror & Shaw, "Parallel random numbers: as easy as 1, 2, 3",
 * SC'11 (Random123). Same generator as cuRAND's Philox4_32_10, but header-only
 * and usable from both host and device code.
 */

#include <cstdint>
#include <cmath>
#include "../core/vec_types.cuh"

namespace zrandom {

/**
 * @brief Compute the most and least significant 32 bits of the product of the
 *        two 32-bit unsigned integers.
 * 
 * @param a     Operand a
 * @param b     Operand b
 * @param hi    Most significant 32 bits of a*b
 * @param lo    Least significant 32 bits of a*b
 */
__host__ __device__ 
inline void mulhilo32( uint32_t a, uint32_t b, uint32_t & hi, uint32_t & lo ) {
#ifdef __CUDA_ARCH__
    lo = a * b;
    hi = __umulhi( a, b );
#else
    uint64_t p = static_cast<uint64_t>(a) * b;
    lo = static_cast<uint32_t>( p );
    hi = static_cast<uint32_t>( p >> 32 );
#endif
}

/**
 * @brief Philox4x32-10 bijection: returns 4 random 32-bit words
 * 
 * @param ctr   Counter (must be unique for each call with the same key)
 * @param key   PRNG key
 * @return 
 */
__host__ __device__ 
inline uint4 philox4x32( uint4 ctr, uint2 key ) {
    #pragma unroll
    for( int r = 0; r < 10; ++r ) {
        uint32_t hi0, lo0, hi1, lo1;
        mulhilo32( 0xD2511F53u, ctr.x, hi0, lo0 );
        mulhilo32( 0xCD9E8D57u, ctr.z, hi1, lo1 );
        ctr = make_uint4( hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0 );
        key.x += 0x9E3779B9u;
        key.y += 0xBB67AE85u;
    }
    return ctr;
}

/**
 * @brief Convert uint32 to float in [0,1), 24 bit resolution 
 * 
 * @param u 
 * @return float 
 */
__host__ __device__ 
inline float u01_co_f( uint32_t u ) { return ( u >> 8 ) * 0x1p-24f; }

/**
 * @brief Convert uint32 to float in (0,1), 23 bit resolution
 * 
 * @param u 
 * @return float 
 */
__host__ __device__ 
inline float u01_oo_f( uint32_t u ) { return ( ( u >> 9 ) + 0.5f ) * 0x1p-23f; }

/**
 * @brief Convert uint32 to double in [0,1), 53 bit resolution (uses 2 words)
 * 
 * @param a         Word A
 * @param b         Word B
 * @return double 
 */
__host__ __device__ 
inline double u01_co_d( uint32_t a, uint32_t b ) {
    return ( ( a >> 5 ) * 67108864.0 + ( b >> 6 ) ) * 0x1p-53;
}

/**
 * @brief Convert uint32 to double in (0,1), 52 bit resolution (uses 2 words)
 * 
 * @param a         Word A
 * @param b         Word B
 * @return double 
 */
__host__ __device__ 
inline double u01_oo_d( uint32_t a, uint32_t b ) {
    return ( ( a >> 6 ) * 67108864.0 + ( b >> 6 ) + 0.5 ) * 0x1p-52;
}

/**
 * @brief Convert uint32 to float in (0,1]
 *
 * @note For use with Box-Muller; 32 bit resolution near 0 (where it matters 
 *       for the tails, max |n| ~ 6.8 sigma. Rounding to 1.0f near the top is
 *       harmless (gives radius 0).
 * 
 * @param u 
 * @return float 
 */
__host__ __device__ 
inline float u01_rad_f( uint32_t u ) { return ( u + 0.5f ) * 0x1p-32f; }

/**
 * @brief Box-Muller method for generating random deviates with a normal
 *        (Gaussian) distribution
 *
 * @note  Takes 2 uint32 random numbers
 *
 * @param a         Random value A
 * @param b         Random value B
 * @return float2   2 independent N(0,1) floats
 */
__host__ __device__ 
inline float2 box_muller_f( uint32_t a, uint32_t b ) {
    float rad = sqrtf( -2.0f * logf( u01_rad_f( a ) ) );
    float u   = u01_co_f( b );
    float s, c;
#ifdef __CUDA_ARCH__
    sincospif( 2.0f * u, &s, &c );
#else
    s = sinf( 6.283185307179586f * u ); c = cosf( 6.283185307179586f * u );
#endif
    return make_float2( rad * c, rad * s );
}

/**
 * @brief Generate two independent N(0,1) floats
 * 
 * @param ctr       PRNG counter
 * @param key       PRNG key
 * @return float2   2 independent N(0,1) floats
 */
__host__ __device__ 
inline float2 rand_norm2_f( uint4 ctr, uint2 key ) {
    uint4 r = philox4x32( ctr, key );
    return box_muller_f( r.x, r.y );
}

/**
 * @brief Generate four independent N(0,1) floats
 * 
 * @param ctr       PRNG counter
 * @param key       PRNG key
 * @return float4   4 independent N(0,1) floats
 */
__host__ __device__ 
inline float4 rand_norm4_f( uint4 ctr, uint2 key ) {
    uint4 r = philox4x32( ctr, key );
    float2 n0 = box_muller_f( r.x, r.y );
    float2 n1 = box_muller_f( r.z, r.w );
    return make_float4( n0.x, n0.y, n1.x, n1.y );
}

/**
 * @brief Generate two independent N(0,1) doubles
 * 
 * @note Also uses the Box-Muller algorithm which is reimplemented here in
 *       double precision
 *
 * @param ctr       PRNG counter
 * @param key       PRNG key
 * @return float2   2 independent N(0,1) doubles
 */
__host__ __device__ 
inline double2 rand_norm2_d( uint4 ctr, uint2 key ) {
    uint4 r = philox4x32( ctr, key );
    double rad = sqrt( -2.0 * log( u01_oo_d( r.x, r.y ) ) );
    double u   = u01_co_d( r.z, r.w );
    double s, c;
#ifdef __CUDA_ARCH__
    sincospi( 2.0 * u, &s, &c );
#else
    s = sin( 6.283185307179586 * u ); c = cos( 6.283185307179586 * u );
#endif
    return make_double2( rad * c, rad * s );
}

// Compatibility interface

/**
 * @brief PRNG state
 * 
 */
struct rand_state {
    uint4 ctr;      ///< next counter
    uint2 key;      ///< seed
    uint4 buf;      ///< unused words from last Philox call
    int   nbuf;     ///< number of words left in buf
};

/**
 * @brief Initialize random number generator
 * 
 * @param stream_id     Unique stream id (e.g. global thread or particle id)
 * @param seed          Seed value
 * @param state         PRNG state
 * @param tag           (optional) Call-site tag, so different uses of the same
 *                      seed and stream_id get independent numbers
 */
__host__ __device__ 
inline void rand_init( const uint64_t stream_id, const uint2 seed,
    rand_state & state, const uint32_t tag = 0 ) {
    state.ctr  = make_uint4( 0, uint32_t( stream_id ), uint32_t( stream_id >> 32 ), tag );
    state.key  = seed;
    state.buf  = make_uint4( 0, 0, 0, 0 );
    state.nbuf = 0;
}

/**
 * @brief 32 bit pseudo random number using Philox algorithm
 * 
 * @note Given that each Philox call generates 4 new random numbers, we will
 *       try to use numbers from previous call
 *
 * @param state         PRNG state, will be modified by this call
 * @return uint32_t     Random value
 */
__host__ __device__ 
inline uint32_t rand_uint32( rand_state & state ) {
    if ( state.nbuf == 0 ) {
        state.buf  = philox4x32( state.ctr, state.key );
        state.ctr.x++;
        state.nbuf = 4;
    }
    const uint32_t r = state.buf.x;
    state.buf = make_uint4( state.buf.y, state.buf.z, state.buf.w, 0 );
    state.nbuf--;
    return r;
}

/**
 * @brief double on [0,1] (32 bit resolution)
 * 
 * @param state 
 * @return double 
 */
__host__ __device__
inline double rand_real1( rand_state & state ) {
    return rand_uint32( state ) * 0x1.00000001p-32;
}

/**
 * @brief double on [0,1) (32 bit resolution); exact k * 2^-32 values
 * 
 * @param state 
 * @return double 
 */
__host__ __device__
inline double rand_real2( rand_state & state ) {
    return rand_uint32( state ) * 0x1p-32;
}

/**
 * @brief double on (0,1) (32 bit resolution)
 * 
 * @param state     PRNG state, will be modified by this call
 * @return double 
 */
__host__ __device__
inline double rand_real3( rand_state & state ) {
    return ( rand_uint32( state ) + 0.5 ) * 0x1p-32;
}


} // namespace zrandom
