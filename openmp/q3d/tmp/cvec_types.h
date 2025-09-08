#ifndef CVEC_TYPES_H_
#define CVEC_TYPES_H_

#include <iostream>
#include "vec_types.h"

// Use std::complex
#include <complex>
namespace cvec {
    using complex64   = std::complex<float>;
    using complex128  = std::complex<double>;
}

/**
 * @brief Define basic complex vector2 types
 * 
 * @note
 * 
 * It implements the following functions:
 * 
 * + make_cvec2() - where cvec2 is the vector type name
 * + operator+=
 * + operator-=
 * + operator*=
 * + operator+
 * + operator-
 * + operator* ( v2, v2 ) - multiplies each component
 * + operator* ( v2, complex scalar )
 * + operator* ( complex scalar, v2 )
 * + operator* ( v2, real scalar )
 * + operator* ( real scalar, v2 )
 * + operator==
 * + operator!=
 * + operator<<
 * + dot ( v2, v2 ) - dot product
 * + real ( v2 ) - gets real vector2 from real parts
 * + imag ( v2 ) - gets real vector2 from imaginary parts
 * + abs  ( v2 ) - gets real vector2 from magnitudes
 * + cvec2<C> - template for returning vector type from complex scalar type
 * 
 * @param V     Name of vector type
 * @param R     Corresponding real type
 * @param C     Corresponding complex type
 * @param A     Alignment value
 */

#define __GEN_CVEC2( V, R, C, A ) \
struct alignas(A) V { C x, y; }; \
\
static inline \
auto make_##V ( C x, C y ) { V v{x,y}; return v; } \
\
static inline \
auto & operator+= ( V& lhs, const V& rhs ) { lhs.x += rhs.x; lhs.y += rhs.y; return lhs; } \
\
static inline \
auto & operator-= ( V& lhs, const V& rhs ) { lhs.x -= rhs.x; lhs.y -= rhs.y; return lhs; } \
\
static inline \
auto & operator*= ( V& lhs, const V& rhs ) { lhs.x *= rhs.x; lhs.y *= rhs.y; return lhs; } \
\
static inline \
auto operator+ ( V lhs, const V& rhs ) { lhs += rhs; return lhs; } \
\
static inline \
auto operator- ( V lhs, const V& rhs ) { lhs -= rhs; return lhs; } \
\
static inline \
auto operator* ( V lhs, const V& rhs ) { lhs *= rhs; return lhs;  }\
\
static inline \
auto & operator<<(std::ostream& os, const V& obj) { \
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
auto operator*( V v2, const C& s ) { v2.x *= s; v2.y *= s; return v2; } \
\
static inline \
auto operator*( const C& s, V v2 ) { v2.x *= s; v2.y *= s; return v2; } \
\
static inline \
auto operator*( V v2, const R& s ) { v2.x *= s; v2.y *= s; return v2; } \
\
static inline \
auto operator*( const R& s, V v2 ) { v2.x *= s; v2.y *= s; return v2; } \
\
static inline \
auto real( const V v2 ) { return vec2<R>{ real(v2.x), real(v2.y) }; } \
\
static inline \
auto imag( const V v2 ) { return vec2<R>{ imag(v2.x), imag(v2.y) }; } \
\
static inline \
auto abs( const V v2 ) { return vec2<R>{ abs(v2.x), abs(v2.y) }; } \
\
template <> struct vec2_class<C> { using type = V; };

__GEN_CVEC2( cfloat2,  float,  cvec::complex64,  16 )
__GEN_CVEC2( cdouble2, double, cvec::complex128, 32 )

#undef __GEN_CVEC2

/**
 * @brief Define basic complex vector3 types
 * 
 * @note
 * 
 * It implements the following functions:
 * 
 * + make_cvec3() - where cvec3 is the vector type name
 * + operator+=
 * + operator-=
 * + operator*=
 * + operator+
 * + operator-
 * + operator* ( v3, v3 ) - multiplies each component
 * + operator* ( v3, complex scalar )
 * + operator* ( complex scalar, v3 )
 * + operator* ( v3, real scalar )
 * + operator* ( real scalar, v3 )
 * + operator==
 * + operator!=
 * + operator<<
 * + dot ( v3, v3 ) - dot product
 * + real ( v3 ) - gets real vector3 from real parts
 * + imag ( v3 ) - gets real vector3 from imaginary parts
 * + abs  ( v3 ) - gets real vector3 from magnitudes
 * + cvec3<C> - template for returning vector type from complex scalar type
 * 
 * @param V     Name of vector type
 * @param R     Corresponding real type
 * @param R     Corresponding complex type
 */

#define __GEN_CVEC3( V, R, C ) \
struct V { C x, y, z; }; \
\
static inline \
auto make_##V ( C x, C y, C z ) { V v{x,y,z}; return v; } \
\
static inline \
auto & operator+= ( V& lhs, const V& rhs ) { lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z; return lhs; } \
\
static inline \
auto & operator-= ( V& lhs, const V& rhs ) { lhs.x -= rhs.x; lhs.y -= rhs.y; lhs.z -= rhs.z; return lhs; } \
\
static inline \
auto & operator*= ( V& lhs, const V& rhs ) { lhs.x *= rhs.x; lhs.y *= rhs.y; lhs.z *= rhs.z; return lhs; } \
\
static inline \
auto operator+ ( V lhs, const V& rhs ) { lhs += rhs; return lhs; } \
\
static inline \
auto operator- ( V lhs, const V& rhs ) { lhs -= rhs; return lhs; } \
\
static inline \
auto operator* ( V lhs, const V& rhs ) { lhs *= rhs; return lhs;  }\
\
static inline \
auto & operator<<(std::ostream& os, const V& obj) { \
    os << '(' << obj.x << ", " << obj.y << ", " << obj.z << ')'; return os; } \
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
auto operator*( V v3, const C& s ) { v3.x *= s; v3.y *= s; v3.z *= s; return v3; } \
\
static inline \
auto operator*( const C& s, V v3 ) { v3.x *= s; v3.y *= s; v3.z *= s; return v3; } \
\
static inline \
auto operator*( V v3, const R& s ) { v3.x *= s; v3.y *= s; v3.z *= s; return v3; } \
\
static inline \
auto operator*( const R& s, V v3 ) { v3.x *= s; v3.y *= s; v3.z *= s; return v3; } \
\
static inline \
auto real( const V v3 ) { return vec3<R>{ real(v3.x), real(v3.y), real(v3.z) }; } \
\
static inline \
auto imag( const V v3 ) { return vec3<R>{ imag(v3.x), imag(v3.y), imag(v3.z) }; } \
\
static inline \
auto abs( const V v3 ) { return vec3<R>{ abs(v3.x), abs(v3.y), abs(v3.z) }; } \
\
template <> struct vec3_class<C> { using type = V; };

__GEN_CVEC3( cfloat3,  float,  cvec::complex64 )
__GEN_CVEC3( cdouble3, double, cvec::complex128 )

#undef __GEN_CVEC3


#endif
