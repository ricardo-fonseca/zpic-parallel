#ifndef CYL_TYPES_H_
#define CYL_TYPES_H_


#include <iostream>

template < class T > struct cyl2_class{ 
    static_assert(0,"Unsupported datatype");
    using type = void;
};

template < class T >
using cyl2 = typename cyl2_class<T>::type;

#define __GEN_CYL2( V, S, A ) \
struct alignas(A) V { S z, r; }; \
\
static inline \
auto make_##V ( S z, S r ) { V cyl2{z,r}; return cyl2; } \
\
static inline \
auto& operator+= ( V& lhs, const V& rhs ) { lhs.z += rhs.z; lhs.r += rhs.r; return lhs; } \
\
static inline \
auto& operator-= ( V& lhs, const V& rhs ) { lhs.z -= rhs.z; lhs.r -= rhs.r; return lhs; } \
\
static inline \
auto& operator*= ( V& lhs, const V& rhs ) { lhs.z *= rhs.z; lhs.r *= rhs.r; return lhs; } \
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
static inline std::ostream& operator<<(std::ostream& os, const V& obj) { \
    os << "(" << obj.z << " [z], " << obj.r << " [r])"; return os; } \
\
static inline \
bool operator==( const V &lhs, const V &rhs ) { \
    return lhs.z == rhs.z && lhs.r == rhs.r; } \
\
static inline \
bool operator!=( const V &lhs, const V &rhs ) { \
    return lhs.z != rhs.z || lhs.r != rhs.r; } \
\
static inline \
auto dot( const V lhs, const V rhs  ) { return lhs.z*rhs.z + lhs.r*rhs.r; } \
\
static inline \
auto operator*( V v2, const S& s ) { v2.z *= s; v2.r *= s; return v2; } \
\
static inline \
auto operator*( const S& s, V v2 ) { v2.z *= s; v2.r *= s; return v2; } \
\
template <> struct cyl2_class<S> { using type = V; };

__GEN_CYL2( icyl2, int, 8 )
__GEN_CYL2( ucyl2, unsigned int, 8 )
__GEN_CYL2( fcyl2, float, 8 )
__GEN_CYL2( dcyl2, double, 16 )

#undef __GEN_CYL2

template < class T > struct cyl3_class{ 
    static_assert(0,"Unsupported datatype");
    using type = void;
};

template < class T >
using cyl3 = typename cyl3_class<T>::type;

#define __GEN_CYL3( V, S ) \
struct V { S z, r, θ; }; \
\
static inline \
auto make_##V ( S z, S r, S θ ) { V cyl3{z,r,θ}; return cyl3; } \
\
static inline \
auto& operator+= ( V& lhs, const V& rhs ) { lhs.z += rhs.z; lhs.r += rhs.r; lhs.θ += rhs.θ; return lhs; }\
\
static inline \
auto& operator-= ( V& lhs, const V& rhs ) { lhs.z -= rhs.z; lhs.r -= rhs.r; lhs.θ -= rhs.θ; return lhs; }\
\
static inline \
auto& operator*= ( V& lhs, const V& rhs ) { lhs.z *= rhs.z; lhs.r *= rhs.r; lhs.θ *= rhs.θ; return lhs; }\
\
static inline \
auto operator+ ( V lhs, const V& rhs ) { lhs += rhs; return lhs; }\
\
static inline \
auto operator- ( V lhs, const V& rhs ) { lhs -= rhs; return lhs; }\
\
static inline \
auto operator* ( V lhs, const V& rhs ) { lhs *= rhs; return lhs; }\
\
static inline std::ostream& operator<<(std::ostream& os, const V& obj) { \
    os << "(" << obj.z << " [z], " << obj.r << " [r], " << obj.θ << " [θ])"; return os; } \
\
static inline \
bool operator==( const V &lhs, const V &rhs ) { \
    return lhs.z == rhs.z && lhs.r == rhs.r && lhs.θ == rhs.θ; } \
\
static inline \
bool operator!=( const V &lhs, const V &rhs ) { \
    return lhs.z != rhs.z || lhs.r != rhs.r || lhs.θ != rhs.θ; } \
\
static inline \
auto operator*( V v3, const S& s ) { v3.z *= s; v3.r *= s; v3.θ *= s; return v3; } \
\
static inline \
auto operator*( const S& s, V v3 ) { v3.z *= s; v3.r *= s; v3.θ *= s; return v3; } \
\
template <> struct cyl3_class<S> { using type = V; };

__GEN_CYL3( icyl3, int )
__GEN_CYL3( ucyl3, unsigned int )
__GEN_CYL3( fcyl3, float )
__GEN_CYL3( dcyl3, double )

#undef __GEN_CYL3


#endif
