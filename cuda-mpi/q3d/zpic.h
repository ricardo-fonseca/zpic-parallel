#ifndef ZPIC_H_
#define ZPIC_H_

#include "utils.h"
#include "vec_types.h"

// Not implemented yet
// #include "simd/simd.h"



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
#include "complex.h"

namespace zpic {

/**
 * @brief Print information about GPUs
 * 
 */
static inline void sys_info() {
    std::cout << ansi::bold;
    std::cout << "System info\n";
    std::cout << ansi::reset;

    print_gpu_info();
}

/**
 * @brief CFL time limit from cell size
 * 
 * @param dx        Cell size
 * @return float    CFL time limit
 */
static inline float courant( const unsigned m, const float2 dx ) {
    auto dz = dx.x;
    auto dr = dx.y;

    auto cour = 1.0f/(dz*dz) + 1.0f/(dr*dr);

    cour += ( m > 1 ) ?
            ( 4.0f * (m-1) * (m-1) )/ (dr * M_PI * dr * M_PI):
            1.0f / (dr * M_PI * dr * M_PI);

    return std::sqrt( 1.0f/cour );
}

/**
 * @brief CFL time limit number cells and box size 
 * 
 * @param gnx       Global number of cells
 * @param box       Simulation box size (global)
 * @return float    CFL time limit
 */
static inline float courant( const unsigned m, const uint2 gnx, const float2 box  ) {
    auto dx = make_float2( box.x/gnx.x, box.y/gnx.y);
    return courant(m,dx);
}

/**
 * @brief CFL time limit from number of tiles, tile size and simulation box size
 * 
 * @param ntiles    Number of tiles
 * @param nx        Number of cells per tile
 * @param box       Simulation box size (global)
 * @return float    CFL time limit
 */
static inline float courant( const unsigned m, const uint2 ntiles, const uint2 nx, const float2 box ) {
    auto dx = make_float2( box.x / ( nx.x * ntiles.x ), box.y / ( nx.y * ntiles.y ) );
    return courant(m,dx);
}

}

#endif
