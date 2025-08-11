#ifndef ZPIC_H_
#define ZPIC_H_

#include "utils.h"
#include "simd/simd.h"
#include "vec_types.h"

/**
 * @brief Coordinates (x,y)
 * 
 */
namespace coord {
    enum cart  { x = 0, y };
    enum cyl { z = 0, r };
}


/**
 * C++ 20 mathematical constants
 * 
 * std::numbers::e
 * std::numbers::pi
 * etc.
 */

// #include <numbers>
#include <cmath>

/**
 * Complex number support
 * 
 */
#include <complex>

/**
 * Add support for (some) mixed precision complex<float> and double / complex<double> operations
 * 
 */

constexpr auto operator*( const std::complex< float > & c, const double & s ) {
    return std::complex< double > { s * c.real(), s * c.imag() }; 
}

constexpr auto operator*( const double & s, const std::complex< float > & c ) {
    return std::complex< double > { s * c.real(), s * c.imag() }; 
}

constexpr auto operator*( const std::complex< double > & c1, const std::complex< float > & c2 ) {
    std::complex< double > t { c2.real(), c2.imag() };
    return c1 * t; 
}

constexpr auto operator*( const std::complex< float > & c1, const std::complex< double > & c2 ) {
    std::complex< double > t { c1.real(), c1.imag() };
    return t * c2; 
}

constexpr auto operator+( const std::complex< double > & c1, const std::complex< float > & c2 ) {
    std::complex< double > t { c2.real(), c2.imag() };
    return c1 + t; 
}

constexpr auto operator+( const std::complex< float > & c1, const std::complex< double > & c2 ) {
    std::complex< double > t { c1.real(), c1.imag() };
    return t + c2; 
}

constexpr auto operator-( const std::complex< double > & c1, const std::complex< float > & c2 ) {
    std::complex< double > t { c2.real(), c2.imag() };
    return c1 - t; 
}

constexpr auto operator-( const std::complex< float > & c1, const std::complex< double > & c2 ) {
    std::complex< double > t { c1.real(), c1.imag() };
    return t - c2; 
}

namespace zpic {

/**
 * @brief Print information about SIMD and OpenMP support
 * 
 */
inline void sys_info() {
#ifdef SIMD
    std::cout << "SIMD support enabled\n";
    std::cout << "  vector unit : " << vecname << '\n';
    std::cout << "  vector width: " << vecwidth <<'\n';
#else
    std::cout << "SIMD support not enabled\n";
#endif

#ifdef _OPENMP

    std::cout << "OpenMP enabled\n";
    std::cout << "  # procs           : " << omp_get_num_procs() << '\n';
    std::cout << "  max_threads       : " << omp_get_max_threads() << '\n';
    #pragma omp parallel
    {
        if ( omp_get_thread_num() == 0 )
            std::cout << "  default # threads : " << omp_get_num_threads() << '\n';
    }
#else
    std::cout << "OpenMP support not enabled\n";
#endif
}

/**
 * @brief CFL time limit from cell size
 * 
 * @param dx        Cell size
 * @return float    CFL time limit
 */
inline float courant( const float2 dx ) {
    return std::sqrt( 1.0f/( 1.0f/(dx.x*dx.x) + 1.0f/(dx.y*dx.y) ) );
}

/**
 * @brief CFL time limit number cells and box size 
 * 
 * @param gnx       Global number of cells
 * @param box       Simulation box size (global)
 * @return float    CFL time limit
 */
inline float courant( const uint2 gnx, const float2 box  ) {
    float2 dx = make_float2( box.x/gnx.x, box.y/gnx.y);
    return courant(dx);
}

/**
 * @brief CFL time limit from number of tiles, tile size and simulation box size
 * 
 * @param ntiles    Number of tiles
 * @param nx        Number of cells per tile
 * @param box       Simulation box size (global)
 * @return float    CFL time limit
 */
inline float courant( const uint2 ntiles, const uint2 nx, const float2 box ) {
    float2 dx = make_float2( box.x / ( nx.x * ntiles.x ), box.y / ( nx.y * ntiles.y ) );
    return courant(dx);
}

}

#endif
