#pragma once

/**
 * @file vec_complex.hpp
 * @brief Complex SIMD vectors, built on the vec_*() functions of the selected
 *        SIMD backend (see simd.hpp)
 * 
 * A vcomplex holds simd::width complex numbers in split (structure of arrays)
 * form, one vfloat with the real parts and one with the imaginary parts.
 * 
 * The complex product uses the same operation order as util::complex (and the
 * scalar code in the q3d species), i.e.
 * 
 *      re = fma( a.re, b.re, -a.im * b.im )
 *      im = fma( a.re, b.im,  a.im * b.re )
 * 
 * so that scalar and vector code give the same results.
 */

#include "simd.hpp"

#include "../util/complex.hpp"

#ifdef SIMD

namespace simd {

/**
 * @brief SIMD vector of complex (single precision) values
 */
struct vcomplex {
    vfloat re;
    vfloat im;
};

/**
 * @brief Complex vector from real and imaginary part vectors
 */
inline vcomplex vec_complex( const vfloat re, const vfloat im ) {
    return vcomplex{ re, im };
}

/**
 * @brief Complex vector with all elements set to re + i im
 */
inline vcomplex vec_complex( const float re, const float im ) {
    return vcomplex{ vec_float( re ), vec_float( im ) };
}

inline vfloat vec_real( const vcomplex a ) { return a.re; }
inline vfloat vec_imag( const vcomplex a ) { return a.im; }

/**
 * @brief Complex conjugate
 */
inline vcomplex vec_conj( const vcomplex a ) {
    return vcomplex{ a.re, vec_neg( a.im ) };
}

inline vcomplex vec_add( const vcomplex a, const vcomplex b ) {
    return vcomplex{ vec_add( a.re, b.re ), vec_add( a.im, b.im ) };
}

inline vcomplex vec_sub( const vcomplex a, const vcomplex b ) {
    return vcomplex{ vec_sub( a.re, b.re ), vec_sub( a.im, b.im ) };
}

/**
 * @brief Product of a complex vector by a real vector
 */
inline vcomplex vec_mul( const vcomplex a, const vfloat s ) {
    return vcomplex{ vec_mul( a.re, s ), vec_mul( a.im, s ) };
}

inline vcomplex vec_mul( const vfloat s, const vcomplex a ) {
    return vcomplex{ vec_mul( s, a.re ), vec_mul( s, a.im ) };
}

inline vcomplex vec_mul( const vcomplex a, const float s ) {
    return vcomplex{ vec_mul( a.re, s ), vec_mul( a.im, s ) };
}

/**
 * @brief Complex product (same operation order as util::complex)
 */
inline vcomplex vec_mul( const vcomplex a, const vcomplex b ) {
    return vcomplex{
        vec_fmsub( a.re, b.re, vec_mul( a.im, b.im ) ),
        vec_fmadd( a.re, b.im, vec_mul( a.im, b.re ) )
    };
}

/**
 * @brief a * s + c, with s real
 */
inline vcomplex vec_fmadd( const vcomplex a, const vfloat s, const vcomplex c ) {
    return vcomplex{ vec_fmadd( a.re, s, c.re ), vec_fmadd( a.im, s, c.im ) };
}

/**
 * @brief Real part of the complex product, Re( a * b )
 */
inline vfloat vec_real_mul( const vcomplex a, const vcomplex b ) {
    return vec_fmsub( a.re, b.re, vec_mul( a.im, b.im ) );
}

/**
 * @brief Gathers complex values from an (interleaved) complex array
 * 
 * Uses vec_gather_s2() (architecture specific) to gather the (real, imaginary)
 * pairs with 64 bit gathers / loads
 * 
 * @param base      Base address
 * @param vindex    Element indices (in complex elements)
 * @return vcomplex 
 */
inline vcomplex vec_gather_c( util::complex<float> const * base, const vint vindex ) {
    const vfloat2 v = vec_gather_s2( reinterpret_cast< const float * >( base ), vec_add( vindex, vindex ) );
    return vcomplex{ v.x, v.y };
}

/**
 * @brief Extracts a single complex value from a complex vector
 */
template< int imm > 
inline util::complex<float> vec_extract( const vcomplex a ) {
    return util::complex<float>( vec_extract<imm>( a.re ), vec_extract<imm>( a.im ) );
}

/**
 * @brief Extracts a single complex value from a complex vector
 */
inline util::complex<float> vec_extract( const vcomplex a, const int i ) {
    return util::complex<float>( vec_extract( a.re, i ), vec_extract( a.im, i ) );
}




} // namespace simd

#endif
