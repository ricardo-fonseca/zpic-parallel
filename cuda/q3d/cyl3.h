#ifndef CYL3_H_
#define CYL3_H_

#include <iostream>

/**
 * @brief Field components (z,r,θ)
 * 
 */
namespace fcomp {
    enum cyl  { z = 0, r, th };
}

/**
 * @brief Cylindrical vector field (z,r,θ)
 * 
 * @note This is represented in memory the same way as a vec3<T> type with:
 * @note +  vec3.x -> cyl3.z
 * @note +  vec3.y -> cyl3.r
 * @note +  vec3.z -> cyl3.th
 * @note It is safe to cast between these two types
 * 
 * @tparam T    Base datatype
 */
template < class T >
struct cyl3 {
    T z, r, th;
    
    __host__ __device__
    /**
     * @brief Constructor from components 
     * 
     */
    cyl3( const T & z = T(), const T & r = T() , const T & th = T() ) : 
        z(z), r(r), th(th) {}

    __host__ __device__
    /**
     * @brief copy constructor
     * 
     */
    cyl3( const cyl3 & s) : z(s.z), r(s.r), th(s.th) {}

    __host__
    /**
     * @brief Stream extraction
     * 
     * @param os        stream
     * @return std::ostream& 
     */
    friend auto & operator<<( std::ostream& os , const cyl3 & obj ) { 
        os << "(" << obj.z << " [z], " << obj.r << " [r], " << obj.th << " [θ])"; 
        return os;
    }

    __host__ __device__
    auto & operator+=( const cyl3 & rhs ) {
        z += rhs.z;
        r += rhs.r;
        th += rhs.th;
        return *this;
    }

    __host__ __device__
    auto friend operator+ ( cyl3 lhs, const cyl3 & rhs ) { return lhs += rhs; }

    __host__ __device__
    auto & operator-=( const cyl3 & rhs ) {
        z -= rhs.z;
        r -= rhs.r;
        th -= rhs.th;
        return *this;
    }

    __host__ __device__
    auto friend operator- ( cyl3 lhs, const cyl3 & rhs ) { return lhs -= rhs; }

    template < class S >
    __host__ __device__
    auto & operator*=( const S & rhs ) {
        z *= rhs;
        r *= rhs;
        th *= rhs;
        return *this;
    }

    template < class S >
    __host__ __device__
    auto friend operator* ( cyl3 lhs, const S & rhs ) { return lhs *= rhs; }

    template < class S >
    __host__ __device__
    auto friend operator* ( const S & lhs, cyl3 rhs ) { return rhs *= lhs; }

    __host__ __device__
    auto friend abs( cyl3 v ) {
        auto z = abs( v.z );
        auto r = abs( v.r );
        auto th = abs( v.th );
        return cyl3< decltype(r) > { z, r, th };
    }

    // The following will only work if T is a complex type
    __host__ __device__
    auto friend real( cyl3 v ) {
        auto th = real( v.th );
        auto r  = real( v.r );
        auto z  = real( v.z );
        return cyl3< decltype(r) > { z, r, th };
    }

    __host__ __device__
    auto friend imag( cyl3 v ) {
        auto z = imag( v.z );
        auto r = imag( v.r );
        auto th = imag( v.th );
        return cyl3< decltype(r) > { z, r, th };
    }

    __host__ __device__
    auto friend norm( cyl3 v ) {
        auto z = norm( v.z );
        auto r = norm( v.r );
        auto th = norm( v.t );
        return cyl3< decltype(r) > { z, r, th };
    }

    __host__ __device__
    auto friend conj( cyl3 v ) {
        auto z = conj( v.z );
        auto r = conj( v.r );
        auto th = conj( v.th );
        return cyl3< decltype(r) > { z, r, th };
    }

};

using cyl_float3  = cyl3<float>;
using cyl_double3 = cyl3<double>;

using cyl_cfloat3  = cyl3< ops::complex<float> >;
using cyl_cdouble3 = cyl3< ops::complex<double> >;

namespace block {

__device__ __inline__
/**
 * @brief Block level memcpy of float3 values
 * 
 * @warning Must be called by all threads in block
 * 
 * @param dst   Destination address
 * @param src   Source address
 * @param n     Number of elements to copy
 */
void memcpy( cyl_float3 * __restrict__ dst, cyl_float3 const * __restrict__ src, const size_t n ) {
    float *       __restrict__ _dst = reinterpret_cast<float *>(dst);
    float const * __restrict__ _src = reinterpret_cast<float const *>(src);

    for( size_t i = block_thread_rank(); i < 3*n; i += block_num_threads() )
        _dst[i] = _src[i];
}

__device__ __inline__
/**
 * @brief Block level memcpy of float3 values
 * 
 * @warning Must be called by all threads in block
 * 
 * @param dst   Destination address
 * @param src   Source address
 * @param n     Number of elements to copy
 */
void memcpy( cyl_cfloat3 * __restrict__ dst, cyl_cfloat3 const * __restrict__ src, const size_t n ) {
    float2 *       __restrict__ _dst = reinterpret_cast<float2 *>(dst);
    float2 const * __restrict__ _src = reinterpret_cast<float2 const *>(src);

    for( size_t i = block_thread_rank(); i < 3*n; i += block_num_threads() )
        _dst[i] = _src[i];
}


}

#endif