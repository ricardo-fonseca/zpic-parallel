#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <complex>

namespace util {

namespace detail {

/// sin and cos of the same angle, using sincosf() on device
__host__ __device__
inline void sincos( const float a, float & s, float & c ) {
#ifdef __CUDA_ARCH__
    ::sincosf( a, &s, &c );
#else
    s = ::sinf( a );
    c = ::cosf( a );
#endif
}

}



class complex64 : public float2 {
    public:

    using value_type = float;

    /**
     * @brief Default constructor, leaves the value uninitialized
     */
    complex64() = default;

    /**
     * @brief Construct from real and (optional) imaginary parts
     *
     * @param re    Real part
     * @param im    Imaginary part, defaults to 0
     */
    __host__ __device__
    constexpr complex64( const float re, const float im = 0.0f ) : float2{ re, im } {}

    /**
     * @brief Construct from float2 / cufftComplex / cuComplex
     *
     * @param z     Value to convert
     */
    __host__ __device__
    constexpr complex64( const float2 z ) : float2{ z.x, z.y } {}

    /**
     * @brief Construct from std::complex<float> (host only)
     *
     * @param z     Value to convert
     */
    __host__
    constexpr complex64( const std::complex<float> & z ) : float2{ z.real(), z.imag() } {}

    /**
     * @brief Convert to std::complex<float> (host only)
     */
    __host__
    constexpr operator std::complex<float>() const { return { x, y }; }

    // ---------------------------------------------------------------------
    // Element access
    // ---------------------------------------------------------------------

    /// Real part (read / write reference)
    __host__ __device__
    constexpr float & real() { return x; }

    /// Imaginary part (read / write reference)
    __host__ __device__
    constexpr float & imag() { return y; }

    /// Real part
    __host__ __device__
    constexpr float real() const { return x; }

    /// Imaginary part
    __host__ __device__
    constexpr float imag() const { return y; }

    /// Set real part (std::complex style)
    __host__ __device__
    constexpr void real( const float re ) { x = re; }

    /// Set imaginary part (std::complex style)
    __host__ __device__
    constexpr void imag( const float im ) { y = im; }

    // ---------------------------------------------------------------------
    // Compound assignment
    // ---------------------------------------------------------------------

    __host__ __device__
    constexpr complex64 & operator+=( const complex64 rhs ) {
        x += rhs.x; y += rhs.y;
        return *this;
    }

    __host__ __device__
    constexpr complex64 & operator+=( const float rhs ) {
        x += rhs;
        return *this;
    }

    __host__ __device__
    constexpr complex64 & operator-=( const complex64 rhs ) {
        x -= rhs.x; y -= rhs.y;
        return *this;
    }

    __host__ __device__
    constexpr complex64 & operator-=( const float rhs ) {
        x -= rhs;
        return *this;
    }

    /**
     * @brief Complex multiplication
     *
     * Uses explicit FMAs so that host and device give the same result
     * (nvcc contracts to FMA by default, host compilers may not).
     *
     * @note With this formulation z * conj(z) generally has a small nonzero
     *       imaginary part (the rounding error of one product). Use norm(z)
     *       for |z|^2.
     */
    __host__ __device__
    complex64 & operator*=( const complex64 rhs ) {
        const float re = ::fmaf( x, rhs.x, -y * rhs.y );
        const float im = ::fmaf( x, rhs.y,  y * rhs.x );
        x = re; y = im;
        return *this;
    }

    __host__ __device__
    constexpr complex64 & operator*=( const float rhs ) {
        x *= rhs; y *= rhs;
        return *this;
    }

    /**
     * @brief Complex division
     *
     * @note Unscaled algorithm: |rhs|^2 overflows (and the result is lost)
     *       for |rhs| > ~1.8e19, and underflows for |rhs| < ~1e-19. This is
     *       not range-safe the way std::complex division is.
     */
    __host__ __device__
    complex64 & operator/=( const complex64 rhs ) {
        const float inv = 1.0f / ::fmaf( rhs.x, rhs.x, rhs.y * rhs.y );
        const float re  = ::fmaf( x, rhs.x,  y * rhs.y ) * inv;
        const float im  = ::fmaf( y, rhs.x, -x * rhs.y ) * inv;
        x = re; y = im;
        return *this;
    }

    __host__ __device__
    constexpr complex64 & operator/=( const float rhs ) {
        x /= rhs; y /= rhs;
        return *this;
    }

    // ---------------------------------------------------------------------
    // Unary operators
    // ---------------------------------------------------------------------

    __host__ __device__
    friend constexpr complex64 operator+( const complex64 z ) { return z; }

    __host__ __device__
    friend constexpr complex64 operator-( const complex64 z ) { return { -z.x, -z.y }; }

    // ---------------------------------------------------------------------
    // Binary operators
    //
    // Mixed float overloads are given explicitly so that the imaginary part
    // is left untouched (as in std::complex) instead of having 0 added to it.
    // ---------------------------------------------------------------------

    __host__ __device__
    friend constexpr complex64 operator+( complex64 lhs, const complex64 rhs ) { return lhs += rhs; }

    __host__ __device__
    friend constexpr complex64 operator+( complex64 lhs, const float rhs ) { return lhs += rhs; }

    __host__ __device__
    friend constexpr complex64 operator+( const float lhs, complex64 rhs ) { return rhs += lhs; }

    __host__ __device__
    friend constexpr complex64 operator-( complex64 lhs, const complex64 rhs ) { return lhs -= rhs; }

    __host__ __device__
    friend constexpr complex64 operator-( complex64 lhs, const float rhs ) { return lhs -= rhs; }

    __host__ __device__
    friend constexpr complex64 operator-( const float lhs, const complex64 rhs ) { return { lhs - rhs.x, -rhs.y }; }

    __host__ __device__
    friend complex64 operator*( complex64 lhs, const complex64 rhs ) { return lhs *= rhs; }

    __host__ __device__
    friend constexpr complex64 operator*( complex64 lhs, const float rhs ) { return lhs *= rhs; }

    __host__ __device__
    friend constexpr complex64 operator*( const float lhs, complex64 rhs ) { return rhs *= lhs; }

    __host__ __device__
    friend complex64 operator/( complex64 lhs, const complex64 rhs ) { return lhs /= rhs; }

    __host__ __device__
    friend constexpr complex64 operator/( complex64 lhs, const float rhs ) { return lhs /= rhs; }

    // ---------------------------------------------------------------------
    // Comparison
    // ---------------------------------------------------------------------

    __host__ __device__
    friend constexpr bool operator==( const complex64 lhs, const complex64 rhs ) {
        return lhs.x == rhs.x && lhs.y == rhs.y;
    }

    __host__ __device__
    friend constexpr bool operator!=( const complex64 lhs, const complex64 rhs ) {
        return !( lhs == rhs );
    }

    // ---------------------------------------------------------------------
    // Functions
    // ---------------------------------------------------------------------

    /// Real part
    __host__ __device__
    friend constexpr float real( const complex64 z ) { return z.x; }

    /// Imaginary part
    __host__ __device__
    friend constexpr float imag( const complex64 z ) { return z.y; }

    /// Complex conjugate
    __host__ __device__
    friend constexpr complex64 conj( const complex64 z ) { return { z.x, -z.y }; }

    /// Squared magnitude |z|^2 (exact for z * conj(z), real by construction)
    __host__ __device__
    friend float norm( const complex64 z ) { return ::fmaf( z.x, z.x, z.y * z.y ); }

    /// Magnitude |z|, computed without intermediate overflow / underflow
    __host__ __device__
    friend float abs( const complex64 z ) { return ::hypotf( z.x, z.y ); }

    /// Phase angle, in the interval [-pi, pi]
    __host__ __device__
    friend float arg( const complex64 z ) { return ::atan2f( z.y, z.x ); }

    /// Complex exponential, e^z
    __host__ __device__
    friend complex64 exp( const complex64 z ) {
        const float r = ::expf( z.x );
        float s, c;
        detail::sincos( z.y, s, c );
        return { r * c, r * s };
    }

    /// Principal value of the natural logarithm
    __host__ __device__
    friend complex64 log( const complex64 z ) {
        return { ::logf( ::hypotf( z.x, z.y ) ), ::atan2f( z.y, z.x ) };
    }

    /// Principal value of the base 10 logarithm
    __host__ __device__
    friend complex64 log10( const complex64 z ) {
        constexpr float inv_ln10 = 0.434294481903251827651f;   // 1 / ln(10)
        return log( z ) * inv_ln10;
    }

    /**
     * @brief Complex power, z^w = exp( w log(z) )
     *
     * @note No special case for z = 0: pow(0, w) returns NaN components
     *       instead of 0.
     */
    __host__ __device__
    friend complex64 pow( const complex64 z, const complex64 w ) {
        return exp( w * log( z ) );
    }

    // ---------------------------------------------------------------------
    // I/O
    // ---------------------------------------------------------------------

    /*
    __host__
    friend std::ostream & operator<<( std::ostream & os, const complex64 z ) {
        return os << '(' << z.x << ',' << z.y << ')';
    }
    */

    __host__
    friend std::ostream & operator<<( std::ostream & os, const complex64 z ) {
        if ( z.y < 0 ) {
            os << '(' << z.x << " - " << ::fabs(z.y) << "𝑖)";
        } else {
            os << '(' << z.x << " + " << z.y << "𝑖)";
        }
        return os;
    }

};

/**
 * @brief Complex number from polar coordinates
 *
 * Namespace-scope (not a hidden friend) because it takes no complex64
 * argument, so ADL could never find it.
 *
 * @param r         Magnitude
 * @param theta     Phase angle, defaults to 0
 */
__host__ __device__
inline complex64 polar( const float r, const float theta = 0.0f ) {
    float s, c;
    detail::sincos( theta, s, c );
    return { r * c, r * s };
}

/**
 * @brief Imaginary unit
 *
 * A function rather than a namespace-scope constexpr variable: nvcc does not
 * allow device code to reference non-scalar host constexpr variables.
 */
__host__ __device__
constexpr complex64 imag_unit() { return { 0.0f, 1.0f }; }

namespace literals {

/// Imaginary literal, e.g. 2.5_i, 3_i (requires `using namespace util::literals`)
__host__ __device__
constexpr complex64 operator""_i( const long double im ) { return { 0.0f, static_cast<float>( im ) }; }

__host__ __device__
constexpr complex64 operator""_i( const unsigned long long im ) { return { 0.0f, static_cast<float>( im ) }; }

}

// -------------------------------------------------------------------------
// Layout checks (cufftComplex == cuComplex == float2)
// -------------------------------------------------------------------------

static_assert( sizeof( complex64 ) == sizeof( float2 ),
    "complex64 must have the same size as float2 / cufftComplex" );
static_assert( alignof( complex64 ) == alignof( float2 ),
    "complex64 must have the same alignment as float2 / cufftComplex" );
static_assert( sizeof( complex64 ) == sizeof( std::complex<float> ),
    "complex64 must have the same size as std::complex<float>" );
static_assert( std::is_standard_layout<complex64>::value,
    "complex64 must be standard layout" );
static_assert( std::is_trivially_copyable<complex64>::value,
    "complex64 must be trivially copyable" );
static_assert( std::is_trivially_default_constructible<complex64>::value,
    "complex64 must be trivially default constructible" );

}

