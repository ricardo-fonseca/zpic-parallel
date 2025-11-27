#ifndef COMPLEX_H_
#define COMPLEX_H_

#include "vec_types.h"
#include "gpu.h"

namespace ops {

template< class S >
class complex : public vec2<S> {
    protected:

    using V = vec2<S>;
    
    public:

    using V :: x;
    using V :: y;

    __host__ __device__
    /**
     * @brief Construct a new complex object
     * 
     * @param re    Real part
     * @param im    Imaginary part
     */
    constexpr complex( S re, S im ) : V{re, im} {}

    template < class S2 >
    __host__ __device__
    complex ( const complex<S2>& z ) : V{ static_cast<S>( z.x ), static_cast<S>( z.y ) } {}

    __host__ __device__
    /**
     * @brief Construct a new complex object, no imaginary part
     * 
     * @param re    Real part
     */
    constexpr complex( S re ) : V{re, 0} {} 

    __host__ __device__
    /**
     * @brief Construct a new complex object, no imaginary part
     * 
     * @param re    Real part
     */
    constexpr complex( ) : V{0, 0} {} 

    __host__ __device__
    /**
     * @brief Construct a new complex object from V
     * 
     * @param z 
     */
    complex( V z ) : V{ z.x, z.y } {}

    __host__ __device__
    /**
     * @brief Access real part
     * 
     * @return S& 
     */
    constexpr S & real() {return x;};

    __host__ __device__
    /**
     * @brief Access imaginary part
     * 
     * @return S& 
     */
    constexpr auto & imag() {return y;};

    __host__ __device__
    inline  complex& operator+= ( const complex& rhs ) { 
        x += rhs.x;
        y += rhs.y;
        return *this;
    }

    __host__ __device__
    inline  complex& operator+= ( const S& rhs ) { 
        x += rhs;
        return *this;
    }

    __host__ __device__
    inline  complex& operator-= ( const complex& rhs ) { 
        x -= rhs.x;
        y -= rhs.y;
        return *this;
    }

    __host__ __device__
    inline  complex& operator-= ( const S& rhs ) { 
        x -= rhs;
        return *this;
    }

    __host__ __device__
    inline  complex& operator*= ( const complex& rhs ) { 
//        S re = x * rhs.x - y * rhs.y;
//        S im = x * rhs.y + y * rhs.x;
        S re = fma( x , rhs.x , - y * rhs.y );
        S im = fma( x , rhs.y , + y * rhs.x );

        x = re;
        y = im;

        return *this;
    }

    __host__ __device__
    inline  complex& operator*= ( const S& rhs ) { 
        x *= rhs;
        y *= rhs;
        return *this;
    }


    __host__ __device__
    inline  complex& operator/= ( const complex& rhs ) { 
        
        S n = rhs.x*rhs.x + rhs.y*rhs.y;

        S re = ( x * rhs.x + y * rhs.y ) / n;
        S im = ( y * rhs.y - x * rhs.y ) / n;

        x = re;
        y = im;

        return *this;
    }

    __host__ __device__
    inline  complex& operator/= ( const S& rhs ) { 
        x /= rhs;
        y /= rhs;

        return *this;
    }

    __host__ __device__
    inline  friend complex operator+ ( const complex z ) {
        return z;
    }

    __host__ __device__
    inline  friend complex operator- ( complex z ) {
        z.x = -z.x; z.y = -z.y;
        return z;
    }

    __host__ __device__
    /**
     * @brief Complex addition
     * 
     * @param lhs           Left-hand side value
     * @param rhs           Right-hand side value
     * @return complex
     */
    inline  friend complex operator+ ( complex lhs, const complex rhs ) {
        lhs += rhs;
        return lhs;
    }

    __host__ __device__
    /**
     * @brief Complex subtraction
     * 
     * @param lhs           Left-hand side value
     * @param rhs           Right-hand side value
     * @return complex
     */
    inline  friend complex operator- ( complex lhs, const complex rhs ) {
        lhs -= rhs;
        return lhs;
    }

    __host__ __device__
    /**
     * @brief Complex multiplication
     * 
     * @param lhs           Left-hand side value
     * @param rhs           Right-hand side value
     * @return complex
     */
    inline  friend complex operator* ( complex lhs, const complex rhs ) {
        lhs *= rhs;
        return lhs;
    }

    __host__ __device__
    inline  friend complex operator* ( complex lhs, const S rhs ) {
        lhs.x *= rhs;
        lhs.y *= rhs;
        return lhs;
    }

    __host__ __device__
    inline  friend complex operator* ( const S lhs, complex rhs ) {
        rhs.x *= lhs;
        rhs.y *= lhs;
        return rhs;
    }

    __host__ __device__
    /**
     * @brief Complex division
     * 
     * @param lhs           Left-hand side value
     * @param rhs           Right-hand side value
     * @return complex
     */
    inline  friend complex operator/ ( complex lhs, const complex rhs ) {
        lhs /= rhs;
        return lhs;
    }

    __host__ __device__
    inline  friend complex operator/ ( complex lhs, const S rhs ) {
        lhs.x /= rhs;
        lhs.y /= rhs;
        return lhs;
    }

    __host__ __device__
    /**
     * @brief Real part
     * 
     * @param z         Complex number
     * @return S 
     */
    inline  friend S real( const complex z ) {
        return z.x;
    };

    __host__ __device__
    /**
     * @brief Imaginary part
     * 
     * @param z         Complex number
     * @return S 
     */
    inline  friend S imag( const complex z ) {
        return z.y;
    };

    __host__ __device__
    /**
     * @brief 
     * 
     * @param z 
     * @return S 
     */
    inline  friend S norm( const complex z ) {
        return z.x*z.x + z.y*z.y;
    } 

    __host__ __device__
    /**
     * @brief 
     * 
     * @param z 
     * @return S 
     */
    inline  friend S abs( const complex z ) {
        return std::sqrt( z.x*z.x + z.y*z.y );
    } 

    __host__ __device__
    inline  friend S arg( const complex z ) {
        return std::atan2( z.y, z.x );
    } 

    __host__
    inline friend std::ostream& operator<<(std::ostream& os, const complex z) {
        if ( z.y < 0 ) {
            os << '(' << z.x << " - " << std::fabs(z.y) << "𝑖)";
        } else {
            os << '(' << z.x << " + " << z.y << "𝑖)";
        }
        return os;
    }

    __host__ __device__
    inline  friend complex conj( const complex& z ) {
        return complex( z.x, -z.y );
    }

    __host__ __device__
    inline  friend complex exp( const complex& z ) {
        S r = std::exp( z.x );
        return complex( r * std::cos(z.y), r * std::sin(z.y));
    }

    __host__ __device__
    /**
     * @brief Principal value of the logarithm (natural base)
     * 
     * @param z           Complex number
     * @return complex 
     */
    inline  friend complex log( const complex& z ) {
        return complex( std::log( std::sqrt(z.x*z.x + z.y*z.y) ), std::atan2( z.y, z.x ) );
    }

    __host__ __device__
    /**
     * @brief Principal value of the logarithm (natural base)
     * 
     * @param z           Complex number
     * @return complex 
     */
    inline  friend complex log10( const complex& z ) {
         S m_ln10 = M_LN10;
        return complex( std::log( std::sqrt(z.x*z.x + z.y*z.y) ) / m_ln10, std::atan2( z.y, z.x )  / m_ln10 );
    }

    __host__ __device__
    /**
     * @brief Complex exponentiation, z^w
     * 
     * @param z             Base
     * @param w             Exponent
     * @return complex 
     */
    inline  friend complex pow( const complex& z, const complex& w ) {
        return exp( w * log(z) );
    }

    __host__ __device__
    /**
     * @brief Build a complex number from polar values (r,θ)
     * 
     * @param r 
     * @param t 
     * @return complex 
     */
    inline friend complex polar( const S& r, const S& t ) {
        return complex( r * std::cos(t), r * std::sin(t));
    }
};

//  complex I{0,1};


using complex64  = complex< float >;
using complex128 = complex< double >;

namespace device {
/**
 * @brief Atomic add operation - device level
 * 
 * @note This is implemented using 2 separate atomic_fetch_add operations
 *       for the real and complex parts
 * 
 * @param address 
 * @param val 
 */
template< typename T >
 __device__
inline void atomic_add( complex<T> * address, const complex<T> val ) {
    atomicAdd( &( address -> x ), val.x );
    atomicAdd( &( address -> y ), val.y );
}

}

namespace block {
/**
 * @brief Atomic add operation - block level
 * 
 * @note This is implemented using 2 separate atomic_fetch_add operations
 *       for the real and complex parts
 * 
 * @param address 
 * @param val 
 */
template< typename T >
 __device__
inline void atomic_add( complex<T> * address, const complex<T> val ) {
    atomicAdd_block( &( address -> x ), val.x );
    atomicAdd_block( &( address -> y ), val.y );
}
}

}

#endif
