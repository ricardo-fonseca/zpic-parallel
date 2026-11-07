#pragma once

#include <iostream>
#include <string>
#include <type_traits>

#include "util/complex.hpp"
#include "core/gpu.cuh"

/**
 * @brief 2D cylindrical coordinates (z,r)
 *
 */
namespace coord {

    enum class cyl : int { z = 0, r };

    inline constexpr cyl cyl_all[] = { cyl::z, cyl::r };

    inline std::string name( const cyl c ) {
        switch (c) {
            case cyl::z : return "z";
            case cyl::r : return "r";
        }
        return "-";
    }
}

/**
 * @brief Cylindrical field components (z,r,θ)
 *
 */
namespace fcomp {

    enum class cyl : int { z = 0, r, th };

    inline constexpr cyl cyl_all[] = { cyl::z, cyl::r, cyl::th };

    inline std::string name( const cyl fc ) {
        switch (fc) {
            case cyl::z  : return "z";
            case cyl::r  : return "r";
            case cyl::th : return "θ";
        }
        return "-";
    }
}

template < class T > struct cyl3;

/// Type trait, true for cyl3<T> specializations
template < class X > struct is_cyl3 : std::false_type {};
template < class T > struct is_cyl3< cyl3<T> > : std::true_type {};

namespace detail_cyl3 {
/// true if T *= S (mul) / T /= S (div) is valid and S is not a cyl3
template < class T, class S, class = void > struct can_mul : std::false_type {};
template < class T, class S >
struct can_mul< T, S, std::void_t< decltype( std::declval<T&>() *= std::declval<const S&>() ) > > :
    std::bool_constant< ! is_cyl3<S>::value > {};

template < class T, class S, class = void > struct can_div : std::false_type {};
template < class T, class S >
struct can_div< T, S, std::void_t< decltype( std::declval<T&>() /= std::declval<const S&>() ) > > :
    std::bool_constant< ! is_cyl3<S>::value > {};
}

/**
 * @brief Cylindrical vector field (z,r,θ)
 *
 * @note This is represented in memory the same way as a vec3<T> type with:
 * @note +  vec3.x -> cyl3.z
 * @note +  vec3.y -> cyl3.r
 * @note +  vec3.z -> cyl3.th
 * @note Size and alignment are checked against float3 / double3 below.
 *
 * @note The default constructor is trivial (leaves the value uninitialized,
 *       like float3 and util::complex), so cyl3 can be used in __shared__
 *       memory. Use cyl3<T>{} to get a zero value.
 *
 * @note Arithmetic with scalars of a different precision follows the
 *       component type T (e.g. cyl3<float> * double is computed in float).
 *
 * @tparam T    Base datatype: float, double, util::complex64 or util::complex128
 */
template < class T >
struct cyl3 {
    T z, r, th;

    using value_type = T;

    /**
     * @brief Default constructor, leaves the value uninitialized
     *
     */
    cyl3() = default;

    /**
     * @brief Constructor from components
     *
     */
    __host__ __device__
    constexpr cyl3( const T & z_, const T & r_, const T & th_ ) :
        z(z_), r(r_), th(th_) {}

    /**
     * @brief Explicit conversion from a cyl3 of a different component type
     *        (e.g. float -> double, or real -> complex)
     *
     */
    template < class U,
        std::enable_if_t< ! std::is_same_v< U, T > && std::is_constructible_v< T, U >, int > = 0 >
    __host__ __device__
    explicit constexpr cyl3( const cyl3<U> & s ) :
        z( T( s.z ) ), r( T( s.r ) ), th( T( s.th ) ) {}

    /**
     * @brief Stream extraction
     *
     * @param os        stream
     * @return std::ostream&
     */
    __host__
    friend std::ostream & operator<<( std::ostream& os , const cyl3 & obj ) {
        os << "(" << obj.z << " [z], " << obj.r << " [r], " << obj.th << " [θ])";
        return os;
    }

    // ---------------------------------------------------------------------
    // Vector arithmetic
    // ---------------------------------------------------------------------

    __host__ __device__
    cyl3 & operator+=( const cyl3 & rhs ) {
        z += rhs.z;
        r += rhs.r;
        th += rhs.th;
        return *this;
    }

    __host__ __device__
    friend cyl3 operator+ ( cyl3 lhs, const cyl3 & rhs ) { return lhs += rhs; }

    __host__ __device__
    cyl3 & operator-=( const cyl3 & rhs ) {
        z -= rhs.z;
        r -= rhs.r;
        th -= rhs.th;
        return *this;
    }

    __host__ __device__
    friend cyl3 operator- ( cyl3 lhs, const cyl3 & rhs ) { return lhs -= rhs; }

    __host__ __device__
    friend cyl3 operator- ( const cyl3 & v ) { return { -v.z, -v.r, -v.th }; }

    __host__ __device__
    friend bool operator==( const cyl3 & lhs, const cyl3 & rhs ) {
        return lhs.z == rhs.z && lhs.r == rhs.r && lhs.th == rhs.th;
    }

    __host__ __device__
    friend bool operator!=( const cyl3 & lhs, const cyl3 & rhs ) { return !( lhs == rhs ); }

    // ---------------------------------------------------------------------
    // Scalar arithmetic
    //
    // S may be any real or complex scalar for which T *= S (T /= S) is
    // valid, so e.g. cyl3<float> * complex64 is rejected at overload
    // resolution instead of failing inside the function body.
    // ---------------------------------------------------------------------

    template < class S, std::enable_if_t< detail_cyl3::can_mul<T,S>::value, int > = 0 >
    __host__ __device__
    cyl3 & operator*=( const S & rhs ) {
        z *= rhs;
        r *= rhs;
        th *= rhs;
        return *this;
    }

    template < class S, std::enable_if_t< detail_cyl3::can_mul<T,S>::value, int > = 0 >
    __host__ __device__
    friend cyl3 operator* ( cyl3 lhs, const S & rhs ) { return lhs *= rhs; }

    template < class S, std::enable_if_t< detail_cyl3::can_mul<T,S>::value, int > = 0 >
    __host__ __device__
    friend cyl3 operator* ( const S & lhs, cyl3 rhs ) { return rhs *= lhs; }

    template < class S, std::enable_if_t< detail_cyl3::can_div<T,S>::value, int > = 0 >
    __host__ __device__
    cyl3 & operator/=( const S & rhs ) {
        z /= rhs;
        r /= rhs;
        th /= rhs;
        return *this;
    }

    template < class S, std::enable_if_t< detail_cyl3::can_div<T,S>::value, int > = 0 >
    __host__ __device__
    friend cyl3 operator/ ( cyl3 lhs, const S & rhs ) { return lhs /= rhs; }

    // ---------------------------------------------------------------------
    // Component-wise functions, valid for all 4 component types
    //
    // The using-declarations supply the float / double overloads from
    // util::scalar (complex.hpp); for util::complex the hidden friends are
    // found through ADL.
    // ---------------------------------------------------------------------

    /// Component-wise magnitude
    __host__ __device__
    friend auto abs( const cyl3 & v ) {
        using util::scalar::abs;
        using R = decltype( abs( v.r ) );
        return cyl3< R >{ abs( v.z ), abs( v.r ), abs( v.th ) };
    }

    /// Component-wise real part
    __host__ __device__
    friend auto real( const cyl3 & v ) {
        using util::scalar::real;
        using R = decltype( real( v.r ) );
        return cyl3< R >{ real( v.z ), real( v.r ), real( v.th ) };
    }

    /// Component-wise imaginary part
    __host__ __device__
    friend auto imag( const cyl3 & v ) {
        using util::scalar::imag;
        using R = decltype( imag( v.r ) );
        return cyl3< R >{ imag( v.z ), imag( v.r ), imag( v.th ) };
    }

    /// Component-wise squared magnitude |.|^2 (as std::norm), not the vector norm
    __host__ __device__
    friend auto norm( const cyl3 & v ) {
        using util::scalar::norm;
        using R = decltype( norm( v.r ) );
        return cyl3< R >{ norm( v.z ), norm( v.r ), norm( v.th ) };
    }

    /// Component-wise complex conjugate
    __host__ __device__
    friend auto conj( const cyl3 & v ) {
        using util::scalar::conj;
        return cyl3{ conj( v.z ), conj( v.r ), conj( v.th ) };
    }

};

using cyl_float3   = cyl3< float >;
using cyl_double3  = cyl3< double >;
using cyl_cfloat3  = cyl3< util::complex64 >;
using cyl_cdouble3 = cyl3< util::complex128 >;

// -------------------------------------------------------------------------
// Layout checks
//
// Trivial default construction is required for __shared__ variables, trivial
// copy for memcpy / CUB / reinterpret_cast use. Size and alignment must match
// the corresponding CUDA vector types for the vec3 <-> cyl3 cast.
// -------------------------------------------------------------------------

namespace detail_cyl3 {
template < class T >
constexpr bool layout_ok =
    std::is_standard_layout_v< cyl3<T> > &&
    std::is_trivially_copyable_v< cyl3<T> > &&
    std::is_trivially_default_constructible_v< cyl3<T> > &&
    sizeof( cyl3<T> ) == 3 * sizeof( T ) &&
    alignof( cyl3<T> ) == alignof( T );
}

static_assert( detail_cyl3::layout_ok< float > );
static_assert( detail_cyl3::layout_ok< double > );
static_assert( detail_cyl3::layout_ok< util::complex64 > );
static_assert( detail_cyl3::layout_ok< util::complex128 > );

static_assert( sizeof( cyl_float3 )  == sizeof( float3 )  && alignof( cyl_float3 )  == alignof( float3 ) );
static_assert( sizeof( cyl_double3 ) == sizeof( double3 ) && alignof( cyl_double3 ) == alignof( double3 ) );

namespace gpu {
namespace block {

/**
 * @brief Block level memcpy of cyl3<T> values
 *
 * Copies component by component; for complex64 / complex128 each copy is a
 * single 8 / 16 byte element (alignment guaranteed by complex.hpp).
 *
 * @warning Must be called by all threads in block
 *
 * @param dst   Destination address
 * @param src   Source address
 * @param n     Number of elements to copy
 */
template < class T >
__device__
inline void memcpy( cyl3<T> * __restrict__ dst, cyl3<T> const * __restrict__ src, const size_t n ) {
    T *       __restrict__ _dst = reinterpret_cast<T *>(dst);
    T const * __restrict__ _src = reinterpret_cast<T const *>(src);

    for( size_t i = gpu::block::thread_rank(); i < 3*n; i += gpu::block::num_threads() )
        _dst[i] = _src[i];
}

/**
 * @brief Block level zero of cyl3<T> values
 *
 * @warning Must be called by all threads in block
 *
 * @param dst   Destination address
 * @param n     Number of elements to zero
 */
template < class T >
__device__
inline void zero( cyl3<T> * __restrict__ dst, const size_t n ) {
    T * __restrict__ _dst = reinterpret_cast<T *>(dst);

    for( size_t i = gpu::block::thread_rank(); i < 3*n; i += gpu::block::num_threads() )
        _dst[i] = T{};
}

} // namespace block
} // namespace gpu
