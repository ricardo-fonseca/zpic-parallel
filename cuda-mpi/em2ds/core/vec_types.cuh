#pragma once

#include <iostream>
#include <sstream>

/**
 * CUDA already defines vector types, so we just need to define the utility functions
 * 
 */
#include <cuda_runtime.h>

/**
 * @brief Generates the utility functions for the vec2 types
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
    static_assert( sizeof(T) == 0,"Unsupported datatype");
    using type = void;
};

template < class T >
using vec2 = typename vec2_class<T>::type;

#define __GEN_VEC2_UTIL( V, S ) \
__inline__ __host__ __device__ \
V& operator+= ( V& lhs, const V& rhs ) { lhs.x += rhs.x; lhs.y += rhs.y; return lhs; } \
\
__inline__ __host__ __device__ \
V& operator-= ( V& lhs, const V& rhs ) { lhs.x -= rhs.x; lhs.y -= rhs.y; return lhs; } \
\
__inline__ __host__ __device__ \
V& operator*= ( V& lhs, const V& rhs ) { lhs.x *= rhs.x; lhs.y *= rhs.y; return lhs; } \
\
__inline__ __host__ __device__ \
V& operator/= ( V& lhs, const V& rhs ) { lhs.x /= rhs.x; lhs.y /= rhs.y; return lhs; } \
\
__inline__ __host__ __device__ \
V operator+ ( V lhs, const V& rhs ) { lhs += rhs; return lhs; } \
\
__inline__ __host__ __device__ \
V operator- ( V lhs, const V& rhs ) { lhs -= rhs; return lhs; } \
\
__inline__ __host__ __device__ \
V operator* ( V lhs, const V& rhs ) { lhs *= rhs; return lhs;  }\
\
__inline__ __host__ __device__ \
V operator/ ( V lhs, const V& rhs ) { lhs /= rhs; return lhs;  }\
\
__inline__ __host__ \
std::ostream& operator<<(std::ostream& os, const V& obj) { \
    return os << "(" << obj.x << ", " << obj.y << ")"; } \
\
__inline__ __host__ \
std::string to_string( const V& obj ) { \
    std::ostringstream oss; oss << obj; return oss.str(); \
} \
__inline__ __host__ __device__ \
bool operator==( const V &lhs, const V &rhs ) { \
    return lhs.x == rhs.x && lhs.y == rhs.y; } \
\
__inline__ __host__ __device__ \
bool operator!=( const V &lhs, const V &rhs ) { \
    return lhs.x != rhs.x || lhs.y != rhs.y; } \
\
__inline__ __host__ __device__ \
auto dot( const V lhs, const V rhs  ) { return lhs.x*rhs.x + lhs.y*rhs.y; } \
\
__inline__ __host__ __device__ \
V operator*( V v2, const S& s ) { v2.x *= s; v2.y *= s; return v2; } \
\
__inline__ __host__ __device__ \
V operator*( const S& s, V v2 ) { v2.x *= s; v2.y *= s; return v2; } \
\
template <> struct vec2_class<S> { using type = V; };

__GEN_VEC2_UTIL( int2, int )
__GEN_VEC2_UTIL( uint2, unsigned int )
__GEN_VEC2_UTIL( float2, float )
__GEN_VEC2_UTIL( double2, double )

#undef __GEN_VEC2_UTIL


/**
 * @brief Generates the utility functions for the vec3 types
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
    static_assert( sizeof(T) == 0, "Unsupported datatype");
    using type = void;
};

template < class T >
using vec3 = typename vec3_class<T>::type;

#define __GEN_VEC3_UTIL( V, S ) \
__inline__ __host__ __device__ \
V& operator+= ( V& lhs, const V& rhs ) { lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z; return lhs; }\
\
__inline__ __host__ __device__ \
V& operator-= ( V& lhs, const V& rhs ) { lhs.x -= rhs.x; lhs.y -= rhs.y; lhs.z -= rhs.z; return lhs; }\
\
__inline__ __host__ __device__ \
V& operator*= ( V& lhs, const V& rhs ) { lhs.x *= rhs.x; lhs.y *= rhs.y; lhs.z *= rhs.z; return lhs; }\
\
__inline__ __host__ __device__ \
V& operator/= ( V& lhs, const V& rhs ) { lhs.x /= rhs.x; lhs.y /= rhs.y; lhs.z /= rhs.z; return lhs; }\
\
__inline__ __host__ __device__ \
V operator+ ( V lhs, const V& rhs ) { lhs += rhs; return lhs; }\
\
__inline__ __host__ __device__ \
V operator- ( V lhs, const V& rhs ) { lhs -= rhs; return lhs; }\
\
__inline__ __host__ __device__ \
V operator* ( V lhs, const V& rhs ) { lhs *= rhs; return lhs; }\
\
__inline__ __host__ __device__ \
V operator/ ( V lhs, const V& rhs ) { lhs /= rhs; return lhs; }\
\
__inline__ __host__ std::ostream& operator<<(std::ostream& os, const V& obj) { \
    return os << "(" << obj.x << ", " << obj.y << ", " << obj.z << ")"; } \
\
__inline__ __host__ std::string to_string( const V& obj ) { \
    std::ostringstream oss; oss << obj; return oss.str(); \
} \
__inline__ __host__ __device__ \
bool operator==( const V &lhs, const V &rhs ) { \
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z; } \
\
__inline__ __host__ __device__ \
bool operator!=( const V &lhs, const V &rhs ) { \
    return lhs.x != rhs.x || lhs.y != rhs.y || lhs.z != rhs.z; } \
\
__inline__ __host__ __device__ \
auto dot( const V lhs, const V rhs  ) { return lhs.x*rhs.x + lhs.y*rhs.y + lhs.z*rhs.z; } \
\
__inline__ __host__ __device__ \
auto cross( const V u, const V v  ) { return V{ u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x }; } \
\
__inline__ __host__ __device__ \
V operator*( V v3, const S& s ) { v3.x *= s; v3.y *= s; v3.z *= s; return v3; } \
\
__inline__ __host__ __device__ \
V operator*( const S& s, V v3 ) { v3.x *= s; v3.y *= s; v3.z *= s; return v3; } \
\
template <> struct vec3_class<S> { using type = V; };

__GEN_VEC3_UTIL( int3, int )
__GEN_VEC3_UTIL( uint3, unsigned int )
__GEN_VEC3_UTIL( float3, float )
__GEN_VEC3_UTIL( double3, double )

#undef __GEN_VEC3_UTIL

