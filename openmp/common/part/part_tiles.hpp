#pragma once

#include "../core/vec_types.hpp"

namespace part {

/**
 * @brief Gets tile id from coordinates
 * 
 * @details Coordinates may be at most 1 tile outside the tile grid (i.e. in
 *          the range [-1, ntiles]). Along periodic directions coordinates
 *          outside the grid are wrapped around; along non-periodic
 *          directions there is no neighbor tile and -1 is returned.
 * 
 * @param coords        Tile coordinates
 * @param ntiles        Tile grid dimensions
 * @param periodic      Periodic boundaries (x,y)
 * @return int          Tile id on success, -1 on out of bounds
 */
inline int tid_coords( int2 coords, int2 const ntiles, int2 const periodic ) {

    // x periodic wrap
    if ( periodic.x ) {
        if      ( coords.x < 0 )         coords.x += ntiles.x; 
        else if ( coords.x >= ntiles.x ) coords.x -= ntiles.x;
    }

    // y periodic wrap
    if ( periodic.y ) {
        if      ( coords.y < 0 )         coords.y += ntiles.y;
        else if ( coords.y >= ntiles.y ) coords.y -= ntiles.y;
    }

    // No neighbor in this direction
    if ( coords.x < 0 || coords.x >= ntiles.x ||
         coords.y < 0 || coords.y >= ntiles.y ) return -1;

    return coords.y * ntiles.x + coords.x;
}

/**
 * @brief Total number of tiles
 * 
 * @param ntiles    Number of tiles (x,y)
 * @return int      Total number of tiles
 */
inline constexpr int grid_tiles( const uint2 ntiles ) {
    return ntiles.x * ntiles.y;
}

} // namespace part
