#pragma once

#include "core/vec_types.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <complex>

namespace util {

namespace detail {

// -------------------------------------------------------------------------
// Precision-explicit math wrappers
//
// One overload per precision, calling the suffixed C function explicitly.
// These exist on both host and device, so the precision used never depends
// on which overloads <cmath> / the CUDA headers place in the global
// namespace, and host and device use the same correctly rounded fma.
// -------------------------------------------------------------------------

__host__ __device__ inline float  fma( const float a,  const float b,  const float c )  { return ::fmaf( a, b, c ); }
__host__ __device__ inline double fma( const double a, const double b, const double c ) { return ::fma ( a, b, c ); }

__host__ __device__ inline float  exp( const float a )  { return ::expf( a ); }
__host__ __device__ inline double exp( const double a ) { return ::exp ( a ); }

__host__ __device__ inline float  log( const float a )  { return ::logf( a ); }
__host__ __device__ inline double log( const double a ) { return ::log ( a ); }

__host__ __device__ inline float  fabs( const float a )  { return ::fabsf( a ); }
__host__ __device__ inline double fabs( const double a ) { return ::fabs ( a ); }

__host__ __device__ inline float  hypot( const float a,  const float b )  { return ::hypotf( a, b ); }
__host__ __device__ inline double hypot( const double a, const double b ) { return ::hypot ( a, b ); }

__host__ __device__ inline float  atan2( const float y,  const float x )  { return ::atan2f( y, x ); }
__host__ __device__ inline double atan2( const double y, const double x ) { return ::atan2 ( y, x ); }

/// sin and cos of the same angle, using sincos() on device
__host__ __device__ inline void sincos( const float a, float & s, float & c ) {
#ifdef __CUDA_ARCH__
    ::sincosf( a, &s, &c );
#else
    s = std::sin( a );
    c = std::cos( a );
#endif
}

/// sin and cos of the same angle, using sincos() on device
__host__ __device__ inline void sincos( const double a, double & s, double & c ) {
#ifdef __CUDA_ARCH__
    ::sincos( a, &s, &c );
#else
    s = std::sin( a );
    c = std::cos( a );
#endif
}

/// Widening (or same precision) conversion from S2 to S
template< typename S2, typename S >
struct is_widening : std::integral_constant< bool, ( sizeof( S2 ) <= sizeof( S ) ) > {};

} // namespace detail

/**
 * @brief Complex number class for host and device code
 *
 * Supports S = float (complex64) and S = double (complex128).
 *
 * Conversion rules follow std::complex: converting from a lower (or equal)
 * precision complex is implicit, converting from a higher precision complex
 * is explicit. Mixed complex<float> / complex<double> arithmetic is therefore
 * carried out in double precision and returns complex<double>, while mixing
 * a complex with a real scalar keeps the precision of the complex operand.
 *
 * @tparam S    Floating point type of the real and imaginary parts
 */
template< class S >
class complex : public vec2<S> {

    static_assert( std::is_same<S, float>::value || std::is_same<S, double>::value,
        "util::complex<S> requires S = float or S = double" );

    protected:

    using V = vec2<S>;

    using V :: x;
    using V :: y;

    public:

    using value_type = S;

    /**
     * @brief Default constructor, leaves the value uninitialized
     */
    complex() = default;

    /**
     * @brief Construct from real and (optional) imaginary parts
     *
     * @param re    Real part
     * @param im    Imaginary part, defaults to 0
     */
    template< typename S2, typename S3 = S2,
        typename std::enable_if< std::is_arithmetic<S2>::value && std::is_arithmetic<S3>::value, int >::type = 0 >
    __host__ __device__
    constexpr complex( const S2 re, const S3 im = S3(0) ) : V{ static_cast<S>(re), static_cast<S>(im) } {}

    /**
     * @brief Construct from a lower (or equal) precision complex value (implicit)
     *
     * @param z     Value to convert
     */
    template< typename S2,
        typename std::enable_if< detail::is_widening<S2, S>::value, int >::type = 0 >
    __host__ __device__
    constexpr complex( const complex<S2> z ) : V{ static_cast<S>(z.real()), static_cast<S>(z.imag()) } {}

    /**
     * @brief Construct from a higher precision complex value (explicit)
     *
     * @param z     Value to convert
     */
    template< typename S2,
        typename std::enable_if< ! detail::is_widening<S2, S>::value, int >::type = 0 >
    __host__ __device__
    explicit constexpr complex( const complex<S2> z ) : V{ static_cast<S>(z.real()), static_cast<S>(z.imag()) } {}

    /**
     * @brief Construct from float2 / cufftComplex / cuComplex (complex64)
     *        or double2 / cufftDoubleComplex / cuDoubleComplex (complex128)
     *
     * Only enabled for the vector type of the same precision.
     *
     * @param v     Value to convert
     */
    template< typename T,
        typename std::enable_if<
            ( std::is_same<S, float>::value  && std::is_same<T, float2>::value ) ||
            ( std::is_same<S, double>::value && std::is_same<T, double2>::value ), int >::type = 0 >
    __host__ __device__
    constexpr complex( const T v ) : V{ v.x, v.y } {}

    /**
     * @brief Construct from a lower (or equal) precision std::complex (host only, implicit)
     *
     * @param z     Value to convert
     */
    template< typename S2,
        typename std::enable_if< detail::is_widening<S2, S>::value, int >::type = 0 >
    __host__
    constexpr complex( const std::complex<S2> & z ) : V{ static_cast<S>(z.real()), static_cast<S>(z.imag()) } {}

    /**
     * @brief Construct from a higher precision std::complex (host only, explicit)
     *
     * @param z     Value to convert
     */
    template< typename S2,
        typename std::enable_if< ! detail::is_widening<S2, S>::value, int >::type = 0 >
    __host__
    explicit constexpr complex( const std::complex<S2> & z ) : V{ static_cast<S>(z.real()), static_cast<S>(z.imag()) } {}

    /**
     * @brief Convert to std::complex<S2> (host only)
     */
    template< typename S2 >
    __host__
    constexpr operator std::complex<S2>() const { return { static_cast<S2>(x), static_cast<S2>(y) }; }

    // ---------------------------------------------------------------------
    // Element access
    // ---------------------------------------------------------------------

    /// Real part (read / write reference)
    __host__ __device__
    constexpr S & real() { return x; }

    /// Imaginary part (read / write reference)
    __host__ __device__
    constexpr S & imag() { return y; }

    /// Real part
    __host__ __device__
    constexpr S real() const { return x; }

    /// Imaginary part
    __host__ __device__
    constexpr S imag() const { return y; }

    /// Set real part (std::complex style)
    __host__ __device__
    constexpr void real( const S re ) { x = re; }

    /// Set imaginary part (std::complex style)
    __host__ __device__
    constexpr void imag( const S im ) { y = im; }

    // ---------------------------------------------------------------------
    // Compound assignment
    //
    // The complex operand may have any precision (as in std::complex), it is
    // converted to S before the operation. Real operands are taken as S.
    // ---------------------------------------------------------------------

    template< typename S2 >
    __host__ __device__
    constexpr complex & operator+=( const complex<S2> rhs ) {
        x += static_cast<S>( rhs.real() );
        y += static_cast<S>( rhs.imag() );
        return *this;
    }

    __host__ __device__
    constexpr complex & operator+=( const S rhs ) {
        x += rhs;
        return *this;
    }

    template< typename S2 >
    __host__ __device__
    constexpr complex & operator-=( const complex<S2> rhs ) {
        x -= static_cast<S>( rhs.real() );
        y -= static_cast<S>( rhs.imag() );
        return *this;
    }

    __host__ __device__
    constexpr complex & operator-=( const S rhs ) {
        x -= rhs;
        return *this;
    }

    /**
     * @brief Complex multiplication
     *
     * Uses explicit FMAs (fmaf / fma on both host and device) so that host
     * and device give the same result (nvcc contracts to FMA by default,
     * host compilers may not).
     *
     * @note With this formulation z * conj(z) generally has a small nonzero
     *       imaginary part (the rounding error of one product). Use norm(z)
     *       for |z|^2.
     */
    template< typename S2 >
    __host__ __device__
    complex & operator*=( const complex<S2> rhs ) {
        const S a  = static_cast<S>( rhs.real() );
        const S b  = static_cast<S>( rhs.imag() );
        const S re = detail::fma( x, a, -y * b );
        const S im = detail::fma( x, b,  y * a );
        x = re; y = im;
        return *this;
    }

    __host__ __device__
    constexpr complex & operator*=( const S rhs ) {
        x *= rhs; y *= rhs;
        return *this;
    }

    /**
     * @brief Complex division
     *
     * @note Unscaled algorithm: |rhs|^2 overflows (and the result is lost)
     *       for |rhs| > ~1.8e19 (float) / ~1.3e154 (double), and underflows
     *       for |rhs| < ~1e-19 (float) / ~1e-154 (double). This is not
     *       range-safe the way std::complex division is.
     */
    template< typename S2 >
    __host__ __device__
    complex & operator/=( const complex<S2> rhs ) {
        const S a   = static_cast<S>( rhs.real() );
        const S b   = static_cast<S>( rhs.imag() );
        const S inv = static_cast<S>(1) / detail::fma( a, a, b * b );
        const S re  = detail::fma( x, a,  y * b ) * inv;
        const S im  = detail::fma( y, a, -x * b ) * inv;
        x = re; y = im;
        return *this;
    }

    __host__ __device__
    constexpr complex & operator/=( const S rhs ) {
        x /= rhs; y /= rhs;
        return *this;
    }

    // ---------------------------------------------------------------------
    // Unary operators
    // ---------------------------------------------------------------------

    __host__ __device__
    friend constexpr complex operator+( const complex z ) { return z; }

    __host__ __device__
    friend constexpr complex operator-( const complex z ) { return { -z.x, -z.y }; }

    // ---------------------------------------------------------------------
    // Binary operators
    //
    // Mixed real overloads are given explicitly so that the imaginary part
    // is left untouched (as in std::complex) instead of having 0 added to it.
    //
    // Real operands are non-template S parameters: int / float / double
    // scalars reach these through a standard conversion, which always beats
    // the user-defined conversion to complex, so a scalar never changes the
    // precision of the result.
    // ---------------------------------------------------------------------

    __host__ __device__
    friend constexpr complex operator+( complex lhs, const complex rhs ) { return lhs += rhs; }

    __host__ __device__
    friend constexpr complex operator+( complex lhs, const S rhs ) { return lhs += rhs; }

    __host__ __device__
    friend constexpr complex operator+( const S lhs, complex rhs ) { return rhs += lhs; }

    __host__ __device__
    friend constexpr complex operator-( complex lhs, const complex rhs ) { return lhs -= rhs; }

    __host__ __device__
    friend constexpr complex operator-( complex lhs, const S rhs ) { return lhs -= rhs; }

    __host__ __device__
    friend constexpr complex operator-( const S lhs, const complex rhs ) { return { lhs - rhs.x, -rhs.y }; }

    __host__ __device__
    friend complex operator*( complex lhs, const complex rhs ) { return lhs *= rhs; }

    __host__ __device__
    friend constexpr complex operator*( complex lhs, const S rhs ) { return lhs *= rhs; }

    __host__ __device__
    friend constexpr complex operator*( const S lhs, complex rhs ) { return rhs *= lhs; }

    __host__ __device__
    friend complex operator/( complex lhs, const complex rhs ) { return lhs /= rhs; }

    __host__ __device__
    friend constexpr complex operator/( complex lhs, const S rhs ) { return lhs /= rhs; }

    /// Real / complex, same range limitations as complex division
    __host__ __device__
    friend complex operator/( const S lhs, const complex rhs ) {
        const S inv = lhs / detail::fma( rhs.x, rhs.x, rhs.y * rhs.y );
        return { rhs.x * inv, -rhs.y * inv };
    }

    // ---------------------------------------------------------------------
    // Comparison
    // ---------------------------------------------------------------------

    __host__ __device__
    friend constexpr bool operator==( const complex lhs, const complex rhs ) {
        return lhs.x == rhs.x && lhs.y == rhs.y;
    }

    __host__ __device__
    friend constexpr bool operator!=( const complex lhs, const complex rhs ) {
        return !( lhs == rhs );
    }

    // ---------------------------------------------------------------------
    // Functions
    // ---------------------------------------------------------------------

    /// Real part
    __host__ __device__
    friend constexpr S real( const complex z ) { return z.x; }

    /// Imaginary part
    __host__ __device__
    friend constexpr S imag( const complex z ) { return z.y; }

    /// Complex conjugate
    __host__ __device__
    friend constexpr complex conj( const complex z ) { return { z.x, -z.y }; }

    /// Squared magnitude |z|^2 (exact for z * conj(z), real by construction)
    __host__ __device__
    friend S norm( const complex z ) { return detail::fma( z.x, z.x, z.y * z.y ); }

    /// Magnitude |z|, computed without intermediate overflow / underflow
    __host__ __device__
    friend S abs( const complex z ) { return detail::hypot( z.x, z.y ); }

    /// Phase angle, in the interval [-pi, pi]
    __host__ __device__
    friend S arg( const complex z ) { return detail::atan2( z.y, z.x ); }

    /// Complex exponential, e^z
    __host__ __device__
    friend complex exp( const complex z ) {
        const S r = detail::exp( z.x );
        S s, c;
        detail::sincos( z.y, s, c );
        return { r * c, r * s };
    }

    /// Principal value of the natural logarithm
    __host__ __device__
    friend complex log( const complex z ) {
        return { detail::log( detail::hypot( z.x, z.y ) ), detail::atan2( z.y, z.x ) };
    }

    /// Principal value of the base 10 logarithm
    __host__ __device__
    friend complex log10( const complex z ) {
        constexpr S inv_ln10 = static_cast<S>( 0.434294481903251827651 );   // 1 / ln(10)
        return log( z ) * inv_ln10;
    }

    /**
     * @brief Complex power, z^w = exp( w log(z) )
     *
     * @note No special case for z = 0: pow(0, w) returns NaN components
     *       instead of 0.
     */
    __host__ __device__
    friend complex pow( const complex z, const complex w ) {
        return exp( w * log( z ) );
    }

    // ---------------------------------------------------------------------
    // I/O
    // ---------------------------------------------------------------------

    __host__
    friend std::ostream & operator<<( std::ostream & os, const complex z ) {
        if ( std::signbit( z.y ) ) {
            os << '(' << z.x << " - " << -z.y << "𝑖)";
        } else {
            os << '(' << z.x << " + " << z.y << "𝑖)";
        }
        return os;
    }

};

/**
 * @brief Complex number from polar coordinates
 *
 * Namespace-scope (not a hidden friend) because it takes no complex
 * argument, so ADL could never find it.
 *
 * @param r         Magnitude
 * @param theta     Phase angle, defaults to 0
 */
template< typename T,
    typename std::enable_if< std::is_same<T, float>::value || std::is_same<T, double>::value, int >::type = 0 >
__host__ __device__
inline complex<T> polar( const T r, const T theta = T(0) ) {
    T s, c;
    detail::sincos( theta, s, c );
    return { r * c, r * s };
}

/**
 * @brief Imaginary unit, e.g. imag_unit() (float) or imag_unit<double>()
 *
 * A function rather than a namespace-scope constexpr variable: nvcc does not
 * allow device code to reference non-scalar host constexpr variables.
 */
template< typename T = float >
__host__ __device__
constexpr complex<T> imag_unit() { return { T(0), T(1) }; }

namespace literals {

/**
 * Imaginary literals (require `using namespace util::literals`)
 *
 * _i  -> complex<float>,  e.g. 2.5_i, 3_i
 * _id -> complex<double>, e.g. 2.5_id, 3_id
 *
 * _i stays single precision so that existing float code (and float kernels)
 * do not get silently promoted to double.
 */
__host__ __device__
constexpr complex<float> operator""_i( const long double im ) { return { 0.0f, static_cast<float>( im ) }; }

__host__ __device__
constexpr complex<float> operator""_i( const unsigned long long im ) { return { 0.0f, static_cast<float>( im ) }; }

__host__ __device__
constexpr complex<double> operator""_id( const long double im ) { return { 0.0, static_cast<double>( im ) }; }

__host__ __device__
constexpr complex<double> operator""_id( const unsigned long long im ) { return { 0.0, static_cast<double>( im ) }; }

} // namespace literals


/**
 * @brief Real scalar versions of the complex functions
 *
 * The util::complex functions (real, imag, conj, norm, abs, arg) are hidden
 * friends, reachable only through ADL, and there is no ADL for float /
 * double. These overloads let generic code call them for both real and
 * complex types through a using-declaration followed by an unqualified call:
 *
 *      using util::scalar::real;
 *      auto re = real( x );    // float / double -> util::scalar, complex -> ADL
 *
 * Kept in a nested namespace (not in util itself) so that they never hide
 * ::abs etc. from unqualified calls made inside namespace util.
 */
namespace scalar {

/// Real part (identity)
__host__ __device__ constexpr float  real( const float  x ) { return x; }
__host__ __device__ constexpr double real( const double x ) { return x; }

/// Imaginary part (always 0)
__host__ __device__ constexpr float  imag( const float  )   { return 0; }
__host__ __device__ constexpr double imag( const double )   { return 0; }

/// Complex conjugate (identity)
__host__ __device__ constexpr float  conj( const float  x ) { return x; }
__host__ __device__ constexpr double conj( const double x ) { return x; }

/// Squared magnitude x^2, consistent with norm( complex )
__host__ __device__ constexpr float  norm( const float  x ) { return x * x; }
__host__ __device__ constexpr double norm( const double x ) { return x * x; }

/// Magnitude |x|
__host__ __device__ inline float  abs( const float  x ) { return detail::fabs( x ); }
__host__ __device__ inline double abs( const double x ) { return detail::fabs( x ); }

/// Phase angle: 0 for x >= +0, pi for x <= -0 (as std::arg), NaN for NaN
__host__ __device__ inline float  arg( const float  x ) { return detail::atan2( 0.0f, x ); }
__host__ __device__ inline double arg( const double x ) { return detail::atan2( 0.0,  x ); }

} // namespace scalar


// -------------------------------------------------------------------------
// Convenience types
// -------------------------------------------------------------------------
using complex64  = complex<float>;
using complex128 = complex<double>;


// -------------------------------------------------------------------------
// Layout checks
//
// cufftComplex       == cuComplex       == float2
// cufftDoubleComplex == cuDoubleComplex == double2
//
// double2 is 16-byte aligned while std::complex<double> is only 8-byte
// aligned, so alignment is checked against the CUDA vector types only.
// complex128 matching double2 requires vec2<double> to be alignas(16).
// -------------------------------------------------------------------------

static_assert( sizeof( complex64 ) == sizeof( float2 ),
    "complex64 must have the same size as float2 / cufftComplex" );
static_assert( alignof( complex64 ) == alignof( float2 ),
    "complex64 must have the same alignment as float2 / cufftComplex" );
static_assert( sizeof( complex64 ) == sizeof( std::complex<float> ),
    "complex64 must have the same size as std::complex<float>" );

static_assert( sizeof( complex128 ) == sizeof( double2 ),
    "complex128 must have the same size as double2 / cufftDoubleComplex" );
static_assert( alignof( complex128 ) == alignof( double2 ),
    "complex128 must have the same alignment as double2 / cufftDoubleComplex (vec2<double> must be alignas(16))" );
static_assert( sizeof( complex128 ) == sizeof( std::complex<double> ),
    "complex128 must have the same size as std::complex<double>" );

static_assert( std::is_standard_layout<complex64>::value &&
               std::is_standard_layout<complex128>::value,
    "complex64 / complex128 must be standard layout" );
static_assert( std::is_trivially_copyable<complex64>::value &&
               std::is_trivially_copyable<complex128>::value,
    "complex64 / complex128 must be trivially copyable" );
static_assert( std::is_trivially_default_constructible<complex64>::value &&
               std::is_trivially_default_constructible<complex128>::value,
    "complex64 / complex128 must be trivially default constructible" );

} // namespace util

// -------------------------------------------------------------------------
// Atomic operations
//
// Guarded with __CUDACC__ (not __CUDA_ARCH__) so the declarations are
// visible in both nvcc compilation passes.
// -------------------------------------------------------------------------
#ifdef __CUDACC__
namespace gpu {
namespace device {
/**
 * @brief Atomic add operation - device level
 *
 * @note Implemented using 2 separate atomicAdd operations for the real and
 *       imaginary parts, so the update as a whole is not atomic (fine for
 *       accumulation, other threads may observe a half-updated value).
 *       complex<double> requires sm_60 or later.
 *
 * @param address   Address of value to update
 * @param val       Value to add
 */
template< typename T >
__device__
inline void atomic_add( util::complex<T> * address, const util::complex<T> val ) {
    atomicAdd( &( address -> real() ), val.real() );
    atomicAdd( &( address -> imag() ), val.imag() );
}
} // namespace device

namespace block {
/**
 * @brief Atomic add operation - block level
 *
 * @note Implemented using 2 separate atomicAdd_block operations for the real
 *       and imaginary parts, so the update as a whole is not atomic.
 *       Requires sm_60 or later.
 *
 * @param address   Address of value to update
 * @param val       Value to add
 */
template< typename T >
__device__
inline void atomic_add( util::complex<T> * address, const util::complex<T> val ) {
    atomicAdd_block( &( address -> real() ), val.real() );
    atomicAdd_block( &( address -> imag() ), val.imag() );
}
} // namespace block

} // namespace gpu

#endif // __CUDACC__
