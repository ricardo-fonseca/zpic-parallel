#ifndef VEC_TYPES_H_
#define VEC_TYPES_H_

#include <iostream>

/**
 * @brief Define basic vec2 types compatible with CUDA types and a set of
 *        utility functions
 * 
 * @note
 * 
 * It implements the following functions:
 * 
 * + make_vec2() - where vec2 is the vector type name
 * + operator+=
 * + operator-=
 * + operator*=
 * + operator+
 * + operator-
 * + operator* ( v2, v2 ) - multiplies each component
 * + operator* ( v2, scalar )
 * + operator* ( scalar, v2 )
 * + operator==
 * + operator!=
 * + operator<<
 * + dot ( v2, v2 ) - dot product
 * + vec2<T> - template for returning vector type from scalar type
 * 
 * @param V     Name of vector type
 * @param S     Corresponding scalar type
 * @param A     Alignment value
 */


template < class T > struct vec2_class{ 
    static_assert(0,"Unsupported datatype");
    using type = void;
};

template < class T >
using vec2 = typename vec2_class<T>::type;

#define __GEN_VEC2( V, S, A ) \
struct alignas(A) V { S x, y; }; \
\
static inline \
V make_##V ( S x, S y ) { V vec2{x,y}; return vec2; } \
\
static inline \
V& operator+= ( V& lhs, const V& rhs ) { lhs.x += rhs.x; lhs.y += rhs.y; return lhs; } \
\
static inline \
V& operator-= ( V& lhs, const V& rhs ) { lhs.x -= rhs.x; lhs.y -= rhs.y; return lhs; } \
\
static inline \
V& operator*= ( V& lhs, const V& rhs ) { lhs.x *= rhs.x; lhs.y *= rhs.y; return lhs; } \
\
static inline \
V operator+ ( V lhs, const V& rhs ) { lhs += rhs; return lhs; } \
\
static inline \
V operator- ( V lhs, const V& rhs ) { lhs -= rhs; return lhs; } \
\
static inline \
V operator* ( V lhs, const V& rhs ) { lhs *= rhs; return lhs;  }\
\
static inline std::ostream& operator<<(std::ostream& os, const V& obj) { \
    os << "(" << obj.x << ", " << obj.y << ")"; return os; } \
\
static inline \
bool operator==( const V &lhs, const V &rhs ) { \
    return lhs.x == rhs.x && lhs.y == rhs.y; } \
\
static inline \
bool operator!=( const V &lhs, const V &rhs ) { \
    return lhs.x != rhs.x || lhs.y != rhs.y; } \
\
static inline \
auto dot( const V lhs, const V rhs  ) { return lhs.x*rhs.x + lhs.y*rhs.y; } \
\
static inline \
V operator*( V v2, const S& s ) { v2.x *= s; v2.y *= s; return v2; } \
\
static inline \
V operator*( const S& s, V v2 ) { v2.x *= s; v2.y *= s; return v2; } \
\
template <> struct vec2_class<S> { using type = V; };

__GEN_VEC2( int2, int, 8 )
__GEN_VEC2( uint2, unsigned int, 8 )
__GEN_VEC2( float2, float, 8 )
__GEN_VEC2( double2, double, 16 )

#undef __GEN_VEC2


/**
 * @brief Define basic vec3 types compatible with CUDA types and a set of
 *        utility functions
 * 
 * @warning The vec3 types are not enforcing memory alignment
 * 
 * @note
 * 
 * It implements the following functions:
 * 
 * + operator+=
 * + operator-=
 * + operator*=
 * + operator+
 * + operator-
 * + operator* ( v3, v3 ) - multiplies each component
 * + operator* ( v3, scalar )
 * + operator* ( scalar, v3 )
 * + operator==
 * + operator!=
 * + operator<<
 * + dot ( v3, v3 ) - dot product 
 * + cross ( v3, v3 ) - dot product 
 * + vec2<T> - template for returning vector type from scalar type
 */


template < class T > struct vec3_class{ 
    static_assert(0,"Unsupported datatype");
    using type = void;
};

template < class T >
using vec3 = typename vec3_class<T>::type;

#define __GEN_VEC3( V, S ) \
struct V { S x, y, z; }; \
\
static inline \
V make_##V ( S x, S y, S z ) { V vec3{x,y,z}; return vec3; } \
\
static inline \
V& operator+= ( V& lhs, const V& rhs ) { lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z; return lhs; }\
\
static inline \
V& operator-= ( V& lhs, const V& rhs ) { lhs.x -= rhs.x; lhs.y -= rhs.y; lhs.z -= rhs.z; return lhs; }\
\
static inline \
V& operator*= ( V& lhs, const V& rhs ) { lhs.x *= rhs.x; lhs.y *= rhs.y; lhs.z *= rhs.z; return lhs; }\
\
static inline \
V operator+ ( V lhs, const V& rhs ) { lhs += rhs; return lhs; }\
\
static inline \
V operator- ( V lhs, const V& rhs ) { lhs -= rhs; return lhs; }\
\
static inline \
V operator* ( V lhs, const V& rhs ) { lhs *= rhs; return lhs; }\
\
static inline std::ostream& operator<<(std::ostream& os, const V& obj) { \
    os << "(" << obj.x << ", " << obj.y << ", " << obj.z << ")"; return os; } \
\
static inline \
bool operator==( const V &lhs, const V &rhs ) { \
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z; } \
\
static inline \
bool operator!=( const V &lhs, const V &rhs ) { \
    return lhs.x != rhs.x || lhs.y != rhs.y || lhs.z != rhs.z; } \
\
static inline \
auto dot( const V lhs, const V rhs  ) { return lhs.x*rhs.x + lhs.y*rhs.y + lhs.z*rhs.z; } \
\
static inline \
auto cross( const V u, const V v  ) { return V{ u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x }; } \
\
static inline \
V operator*( V v3, const S& s ) { v3.x *= s; v3.y *= s; v3.z *= s; return v3; } \
\
static inline \
V operator*( const S& s, V v3 ) { v3.x *= s; v3.y *= s; v3.z *= s; return v3; } \
\
template <> struct vec3_class<S> { using type = V; };

__GEN_VEC3( int3, int )
__GEN_VEC3( uint3, unsigned int )
__GEN_VEC3( float3, float )
__GEN_VEC3( double3, double )

#undef __GEN_VEC3


#endif
