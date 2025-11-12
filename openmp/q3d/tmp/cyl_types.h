#ifndef CYL_TYPES_H_
#define CYL_TYPES_H_


#include <iostream>
#include "vec_types.h"

/**
 * @brief Cylindrical vector field (z,r)
 * 
 * @note This is represented in memory as a vec2<T> type with:
 * @note +  vec2.x -> cyl2.z
 * @note +  vec2.y -> cyl2.r
 * @note It is safe to cast between these two types
 * 
 * @tparam T    Base datatype
 */
template < class T > 
struct cyl2 {
    T z, r;
    constexpr cyl2( const T & z = T(), const T & r = T() ) : z(z), r(r) {}
    constexpr cyl2( const cyl2 & s ) : z(s.z), r(s.r) {}

    template < class V >
    constexpr cyl2( const V & v2 ) : z(v2.x), r(v2.y) {}

    friend float2 make_float2( const cyl2 & obj ) {
        return make_float2( obj.z, obj.r );
    }

    friend double2 make_double2( const cyl2 & obj ) {
        return make_double2( obj.z, obj.r );
    }

    friend auto & operator<<(std::ostream& os, const cyl2 & obj) { 
        os << "(" << obj.z << " [z], " << obj.r << " [r])"; 
        return os;
    }
};

using cyl_float2  = cyl2<float>;
using cyl_double2 = cyl2<double>;

/**
 * @brief Cylindrical vector field (z,r,θ)
 * 
 * @note This is represented in memory as a vec3<T> type with:
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
     * @brief Constructor from vec3<> types
     * 
     * @tparam V3 
     */
    template < class V3 >
    constexpr cyl3( const V3 & v3 ) : z(v3.x), r(v3.y), θ(v3.z) {}

    /**
     * @brief Stream extraction
     * 
     * @param os        stream
     * @return std::ostream& 
     */
    friend auto & operator<<( std::ostream& os ) { 
        os << "(" << z << " [z], " << r << " [r], " << θ << " [θ])"; 
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
        r *= s;
        θ *= s;
        z *= s;
        return *this;
    }
};

using cyl_float3  = cyl3<float>;
using cyl_double3 = cyl3<double>;

using cyl_cfloat3  = cyl3<std::complex<float>>;
using cyl_cdouble3 = cyl3<std::complex<double>>;

#endif