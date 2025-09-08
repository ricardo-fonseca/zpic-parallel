#ifndef VEC3_H_
#define VEC3_H_

/**
 * @brief Field components (x,y,z)
 * 
 */
namespace fcomp {
    enum cart  { x = 0, y, z };
}

/**
 * @brief Cartesian vector field (x,y,z)
 * 
 * @tparam T    Base datatype
 */
template < class T >
struct vec3 {
    T x, y, z;
    
    /**
     * @brief Constructor from components 
     * 
     */
    constexpr vec3( const T & x = T(), const T & y = T() , const T & z = T() ) : 
        x(x), y(y), z(z) {}

    /**
     * @brief copy constructor
     * 
     */
    constexpr vec3( const vec3 & src) : x(src.x), y(src.y), z(src.z) {}

    /**
     * @brief Constructor from other vec3<> types
     * 
     * @tparam V3 
     */
    template < class V3 >
    constexpr vec3( const V3 & v3 ) : z(v3.x), y(v3.y), z(v3.z) {}

    /**
     * @brief Stream extraction
     * 
     * @param os        stream
     * @return std::ostream& 
     */
    friend auto & operator<<( std::ostream& os ) { 
        os << "(" << x << " [x], " << y << " [y], " << z << " [z])"; 
        return os;
    }

    auto & operator+=( const vec3 & rhs ) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }

    auto friend operator+ ( vec3 lhs, const vec3 & rhs ) { return lhs += rhs; }

    auto & operator-=( const vec3 & rhs ) {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        return *this;
    }

    auto friend operator- ( vec3 lhs, const vec3 & rhs ) { return lhs -= rhs; }

    template < class S >
    auto & operator*=( const S & rhs ) {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }
};

using int3    = vec3<int>;
using uint3   = vec3<unsigned int>;
using float3  = vec3<float>;
using double3 = vec3<double>; 

#endif