#ifndef CYL3_H_
#define CYL3_H_

#include <iostream>

/**
 * @brief Field components (z,r,θ)
 * 
 */
namespace fcomp {
    enum cyl  { z = 0, r, θ };
}

/**
 * @brief Cylindrical vector field (z,r,θ)
 * 
 * @note This is represented in memory the same way as a vec3<T> type with:
 * @note +  vec3.x -> cyl3.z
 * @note +  vec3.y -> cyl3.r
 * @note +  vec3.z -> cyl3.θ
 * @note It is safe to cast between these two types
 * 
 * @tparam T    Base datatype
 */
template < class T >
struct cyl3 {
    T z, r, θ;
    
    /**
     * @brief Constructor from components 
     * 
     */
    constexpr cyl3( const T & z = T(), const T & r = T() , const T & θ = T() ) : 
        z(z), r(r), θ(θ) {}

    /**
     * @brief copy constructor
     * 
     */
    constexpr cyl3( const cyl3 & s) : z(s.z), r(s.r), θ(s.θ) {}

    /**
     * @brief Stream extraction
     * 
     * @param os        stream
     * @return std::ostream& 
     */
    friend auto & operator<<( std::ostream& os , const cyl3 & obj ) { 
        os << "(" << obj.z << " [z], " << obj.r << " [r], " << obj.θ << " [θ])"; 
        return os;
    }

    auto & operator+=( const cyl3 & rhs ) {
        r += rhs.r;
        θ += rhs.θ;
        z += rhs.z;
        return *this;
    }

    auto friend operator+ ( cyl3 lhs, const cyl3 & rhs ) { return lhs += rhs; }

    auto & operator-=( const cyl3 & rhs ) {
        r -= rhs.r;
        θ -= rhs.θ;
        z -= rhs.z;
        return *this;
    }

    auto friend operator- ( cyl3 lhs, const cyl3 & rhs ) { return lhs -= rhs; }

    template < class S >
    auto & operator*=( const S & rhs ) {
        r *= rhs;
        θ *= rhs;
        z *= rhs;
        return *this;
    }

    template < class S >
    auto friend operator* ( cyl3 lhs, const S & rhs ) { return lhs *= rhs; }

    template < class S >
    auto friend operator* ( const S & lhs, cyl3 rhs ) { return rhs *= lhs; }

    auto friend abs( cyl3 v ) {
        auto r = abs( v.r );
        auto θ = abs( v.θ );
        auto z = abs( v.z );
        return cyl3< decltype(r) > { z, r, θ };
    }

    // The following will only work if T is a complex type
    auto friend real( cyl3 v ) {
        auto r = real( v.r );
        auto θ = real( v.θ );
        auto z = real( v.z );
        return cyl3< decltype(r) > { z, r, θ };
    }

    auto friend imag( cyl3 v ) {
        auto r = imag( v.r );
        auto θ = imag( v.θ );
        auto z = imag( v.z );
        return cyl3< decltype(r) > { z, r, θ };
    }

    auto friend norm( cyl3 v ) {
        auto r = norm( v.r );
        auto θ = norm( v.θ );
        auto z = norm( v.z );
        return cyl3< decltype(r) > { z, r, θ };
    }

    auto friend conj( cyl3 v ) {
        auto r = conj( v.r );
        auto θ = conj( v.θ );
        auto z = conj( v.z );
        return cyl3< decltype(r) > { z, r, θ };
    }

};

using cyl_float3  = cyl3<float>;
using cyl_double3 = cyl3<double>;

using cyl_cfloat3  = cyl3<std::complex<float>>;
using cyl_cdouble3 = cyl3<std::complex<double>>;

#endif