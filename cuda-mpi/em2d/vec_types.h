#ifndef VEC_TYPES_H_
#define VEC_TYPES_H_

#include "gpu.h"
#include <iostream>

/**
 * CUDA already defines vector types, so we just need to define the utility functions
 * 
 */

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
 */

#define __GEN_VEC2_UTIL( T ) \
static __inline__ __host__ __device__ \
T& operator+= ( T& lhs, const T& rhs ) { lhs.x += rhs.x; lhs.y += rhs.y; return lhs; }\
\
static __inline__ __host__ __device__ \
T& operator-= ( T& lhs, const T& rhs ) { lhs.x -= rhs.x; lhs.y -= rhs.y; return lhs; }\
\
static __inline__ __host__ __device__ \
T& operator*= ( T& lhs, const T& rhs ) { lhs.x *= rhs.x; lhs.y *= rhs.y; return lhs; }\
\
static __inline__ __host__ __device__ \
T operator+ ( T lhs, const T& rhs ) { lhs += rhs; return lhs; }\
\
static __inline__ __host__ __device__ \
T operator- ( T lhs, const T& rhs ) { lhs -= rhs; return lhs; }\
\
static __inline__ __host__ __device__ \
T operator* ( T lhs, const T& rhs ) { lhs *= rhs; return lhs; }\
\
static inline std::ostream& operator<<(std::ostream& os, const T& obj) { \
    os << "(" << obj.x << ", " << obj.y << ")"; return os; } \
\
static __inline__ __host__ __device__ \
bool operator==( const T &lhs, const T &rhs ) { \
    return lhs.x == rhs.x && lhs.y == rhs.y; } \
\
static __inline__ __host__ __device__ \
bool operator!=( const T &lhs, const T &rhs ) { \
    return lhs.x != rhs.x || lhs.y != rhs.y; } \
\
static __inline__ __host__ __device__ \
auto dot( const T lhs, const T rhs  ) { return lhs.x*rhs.x + lhs.y*rhs.y; } \
\
template< typename S > \
static __inline__ __host__ __device__ \
T operator*( T v2, const S& s ) { v2.x *= s; v2.y *= s; return v2; } \
\
template< typename S > \
static __inline__ __host__ __device__ \
T operator*( const S& s, T v2 ) { v2.x *= s; v2.y *= s; return v2; } \
\

__GEN_VEC2_UTIL( int2 )
__GEN_VEC2_UTIL( uint2 )
__GEN_VEC2_UTIL( float2 )
__GEN_VEC2_UTIL( double2 )

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
 */

#define __GEN_VEC3_UTIL( T ) \
static __inline__ __host__ __device__ \
T& operator+= ( T& lhs, const T& rhs ) { lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z; return lhs; }\
\
static __inline__ __host__ __device__ \
T& operator-= ( T& lhs, const T& rhs ) { lhs.x -= rhs.x; lhs.y -= rhs.y; lhs.z -= rhs.z; return lhs; }\
\
static __inline__ __host__ __device__ \
T& operator*= ( T& lhs, const T& rhs ) { lhs.x *= rhs.x; lhs.y *= rhs.y; lhs.z *= rhs.z; return lhs; }\
\
static __inline__ __host__ __device__ \
T operator+ ( T lhs, const T& rhs ) { lhs += rhs; return lhs; }\
\
static __inline__ __host__ __device__ \
T operator- ( T lhs, const T& rhs ) { lhs -= rhs; return lhs; }\
\
static __inline__ __host__ __device__ \
T operator* ( T lhs, const T& rhs ) { lhs *= rhs; return lhs; }\
\
static inline std::ostream& operator<<(std::ostream& os, const T& obj) { \
    os << "(" << obj.x << ", " << obj.y << ", " << obj.z << ")"; return os; } \
\
static __inline__ __host__ __device__ \
bool operator==( const T &lhs, const T &rhs ) { \
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z; } \
\
static __inline__ __host__ __device__ \
bool operator!=( const T &lhs, const T &rhs ) { \
    return lhs.x != rhs.x || lhs.y != rhs.y || lhs.z != rhs.z; } \
\
static __inline__ __host__ __device__ \
auto dot( const T lhs, const T rhs  ) { return lhs.x*rhs.x + lhs.y*rhs.y + lhs.z*rhs.z; } \
\
static __inline__ __host__ __device__ \
auto cross( const T u, const T v  ) { return T{ u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x }; } \
\
template< typename S > \
static __inline__ __host__ __device__ \
T operator*( T v3, const S& s ) { v3.x *= s; v3.y *= s; v3.z *= s; return v3; } \
\
template< typename S > \
static __inline__ __host__ __device__ \
T operator*( const S& s, T v3 ) { v3.x *= s; v3.y *= s; v3.z *= s; return v3; } \
\

__GEN_VEC3_UTIL( int3 )
__GEN_VEC3_UTIL( uint3 )
__GEN_VEC3_UTIL( float3 )
__GEN_VEC3_UTIL( double3 )

#undef __GEN_VEC3_UTIL

#endif
