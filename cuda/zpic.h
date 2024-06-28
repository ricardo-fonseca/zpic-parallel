#ifndef ZPIC_H_
#define ZPIC_H_

// This includes the SYCL headers and utility functions
#include "utils.h"

/**
 * @brief Coordinates (x,y)
 * 
 */
namespace coord {
    enum cart  { x = 0, y };
}

/**
 * C++ 20 mathematical constants
 * 
 * std::numbers::e
 * std::numbers::pi
 * etc.
 */

#include <numbers>
#include <cmath>

namespace zpic {

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
    float2 dx = float2{ box.x/gnx.x, box.y/gnx.y};
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
    float2 dx = float2{ box.x / ( nx.x * ntiles.x ), box.y / ( nx.y * ntiles.y ) };
    return courant(dx);
}

}

#endif
