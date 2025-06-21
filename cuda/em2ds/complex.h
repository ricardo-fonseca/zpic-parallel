#ifndef COMPLEX_H_
#define COMPLEX_H_

#include "cuda.h"
#include <cmath>

namespace fft {

class complex64 : public float2 {
    public:

    __host__ __device__
    /**
     * @brief Construct a new complex64 object
     * 
     * @param re    Real part
     * @param im    Imaginary part
     */
    constexpr complex64( float re, float im ) : float2{re, im} {}

    __host__ __device__
    /**
     * @brief Construct a new complex64 object, no imaginary part
     * 
     * @param re    Real part
     */
    constexpr complex64( float re ) : float2{re, 0} {} 

    __host__ __device__
    /**
     * @brief Construct a new complex64 object from cufftComplex
     * 
     * @param z 
     */
    constexpr complex64( float2 z ) : float2{ z.x, z.y } {}

    __host__ __device__
    /**
     * @brief Access real part
     * 
     * @return float& 
     */
    inline constexpr float& real() {return x;};

    __host__ __device__
    /**
     * @brief Access imaginary part
     * 
     * @return float& 
     */
    inline constexpr float& imag() {return y;};

    __host__ __device__
    inline constexpr complex64& operator+= ( const complex64& rhs ) { 
        x += rhs.x;
        y += rhs.y;
        return *this;
    }

    __host__ __device__
    inline constexpr complex64& operator+= ( const float& rhs ) { 
        x += rhs;
        return *this;
    }

    __host__ __device__
    inline constexpr complex64& operator-= ( const complex64& rhs ) { 
        x -= rhs.x;
        y -= rhs.y;
        return *this;
    }

    __host__ __device__
    inline constexpr complex64& operator-= ( const float& rhs ) { 
        x -= rhs;
        return *this;
    }

    __host__ __device__
    inline constexpr complex64& operator*= ( const complex64& rhs ) { 
        float re = x * rhs.x - y * rhs.y;
        float im = x * rhs.y + y * rhs.x;

        x = re;
        y = im;

        return *this;
    }

    __host__ __device__
    inline constexpr complex64& operator*= ( const float& rhs ) { 
        x *= rhs;
        y *= rhs;
        return *this;
    }


    __host__ __device__
    inline constexpr complex64& operator/= ( const complex64& rhs ) { 
        
        float n = rhs.x*rhs.x + rhs.y*rhs.y;

        float re = ( x * rhs.x + y * rhs.y ) / n;
        float im = ( y * rhs.y - x * rhs.y ) / n;

        x = re;
        y = im;

        return *this;
    }

    __host__ __device__
    inline constexpr complex64& operator/= ( const float& rhs ) { 
        x /= rhs;
        y /= rhs;

        return *this;
    }

    __host__ __device__
    inline constexpr friend complex64 operator+ ( const complex64 z ) {
        return z;
    }

    __host__ __device__
    inline constexpr friend complex64 operator- ( complex64 z ) {
        z.x = -z.x; z.y = -z.y;
        return z;
    }

    __host__ __device__
    /**
     * @brief Complex addition
     * 
     * @param lhs           Left-hand side value
     * @param rhs           Right-hand side value
     * @return complex64
     */
    inline constexpr friend complex64 operator+ ( complex64 lhs, const complex64 rhs ) {
        lhs += rhs;
        return lhs;
    }

    __host__ __device__
    /**
     * @brief Complex subtraction
     * 
     * @param lhs           Left-hand side value
     * @param rhs           Right-hand side value
     * @return complex64
     */
    inline constexpr friend complex64 operator- ( complex64 lhs, const complex64 rhs ) {
        lhs -= rhs;
        return lhs;
    }

    __host__ __device__
    /**
     * @brief Complex multiplication
     * 
     * @param lhs           Left-hand side value
     * @param rhs           Right-hand side value
     * @return complex64
     */
    inline constexpr friend complex64 operator* ( complex64 lhs, const complex64 rhs ) {
        lhs *= rhs;
        return lhs;
    }

    __host__ __device__
    inline constexpr friend complex64 operator* ( complex64 lhs, const float rhs ) {
        lhs.x *= rhs;
        lhs.y *= rhs;
        return lhs;
    }

    __host__ __device__
    inline constexpr friend complex64 operator* ( const float lhs, complex64 rhs ) {
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
     * @return complex64
     */
    inline constexpr friend complex64 operator/ ( complex64 lhs, const complex64 rhs ) {
        lhs /= rhs;
        return lhs;
    }

    __host__ __device__
    inline constexpr friend complex64 operator/ ( complex64 lhs, const float rhs ) {
        lhs.x /= rhs;
        lhs.y /= rhs;
        return lhs;
    }

    __host__ __device__
    /**
     * @brief Real part
     * 
     * @param z         Complex number
     * @return float 
     */
    inline constexpr friend float real( const complex64 z ) {
        return z.x;
    };

    __host__ __device__
    /**
     * @brief Imaginary part
     * 
     * @param z         Complex number
     * @return float 
     */
    inline constexpr friend float imag( const complex64 z ) {
        return z.y;
    };

    __host__ __device__
    /**
     * @brief 
     * 
     * @param z 
     * @return float 
     */
    inline constexpr friend float abs( const complex64 z ) {
        return std::sqrt( z.x*z.x + z.y*z.y );
    } 

    __host__ __device__
    inline constexpr friend float arg( const complex64 z ) {
        return std::atan2( z.y, z.x );
    } 

    __host__
    inline friend std::ostream& operator<<(std::ostream& os, const complex64 z) {
        if ( z.y < 0 ) {
            os << '(' << z.x << " - " << std::fabs(z.y) << "𝑖)";
        } else {
            os << '(' << z.x << " + " << z.y << "𝑖)";
        }
        return os;
    }

    __host__ __device__
    inline constexpr friend complex64 conj( const complex64& z ) {
        return complex64( z.x, -z.y );
    }

    __host__ __device__
    inline constexpr friend complex64 exp( const complex64& z ) {
        float r = std::exp( z.x );
        return complex64( r * std::cos(z.y), r * std::sin(z.y));
    }

    __host__ __device__
    /**
     * @brief Principal value of the logarithm (natural base)
     * 
     * @param z           Complex number
     * @return complex64 
     */
    inline constexpr friend complex64 log( const complex64& z ) {
        return complex64( std::log( std::sqrt(z.x*z.x + z.y*z.y) ), std::atan2( z.y, z.x ) );
    }

    __host__ __device__
    /**
     * @brief Principal value of the logarithm (natural base)
     * 
     * @param z           Complex number
     * @return complex64 
     */
    inline constexpr friend complex64 log10( const complex64& z ) {
        constexpr float m_ln10 = M_LN10;
        return complex64( std::log( std::sqrt(z.x*z.x + z.y*z.y) ) / m_ln10, std::atan2( z.y, z.x )  / m_ln10 );
    }

    __host__ __device__
    /**
     * @brief Complex exponentiation, z^w
     * 
     * @param z             Base
     * @param w             Exponent
     * @return complex64 
     */
    inline constexpr friend complex64 pow( const complex64& z, const complex64& w ) {
        return exp( w * log(z) );
    }

    __host__ __device__
    inline friend complex64 polar( const float& r, const float& θ ) {
        return complex64( r * std::cos(θ), r * std::sin(θ));
    }

};

constexpr complex64 I{0,1};

}

#endif
