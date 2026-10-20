#pragma once

#include "../core/vec_types.cuh"
#include "../parallel/partition.hpp"

#include "bnd_type.hpp"

namespace part {

/**
 * @brief edge tile direction from shift (dx, dy)
 * 
 * Returns:
 * 
 * | Δy | Δx | dir |
 * | -- | -- | --- |
 * | -1 | -1 |  0  |
 * | -1 |  0 |  1  |
 * | -1 | +1 |  2  |
 * |  0 | -1 |  3  |
 * |  0 |  0 |  4  |
 * |  0 | +1 |  5  |
 * | +1 | -1 |  6  |
 * | +1 |  0 |  7  |
 * | +1 | +1 |  8  |
 * 
 * @param dx    x edge tile shift (-1, 0 or 1)
 * @param dy    y edge tile shift (-1, 0 or 1)
 * @return int  Direction (0-8)
 */
__host__ __device__
inline constexpr int edge_dir_shift( const int dx, const int dy ) {
    return (dy + 1)*3 + (dx + 1);
}

/**
 * @brief edge tile shift (dx, dy) from direction
 * 
 * @details
 * Returns:
 * 
 * | dir | Δy | Δx |
 * | --- | -- | -- |
 * |  0  | -1 | -1 |
 * |  1  | -1 |  0 |
 * |  2  | -1 | +1 |
 * |  3  |  0 | -1 |
 * |  4  |  0 |  0 |
 * |  5  |  0 | +1 |
 * |  6  | +1 | -1 |
 * |  7  | +1 |  0 |
 * |  8  | +1 | +1 |
 * 
 * @param dir       Direction index (0-8)
 * @param dx        x edge tile direction (-1, 0 or 1)
 * @param dy        y edge tile direction (-1, 0 or 1)
 */
__host__ __device__
inline constexpr void edge_shift_dir( const int dir, int & dx, int & dy ) {
    dx = dir % 3 - 1;
    dy = dir / 3 - 1;
}

/**
 * @brief Number of edge tiles per direction
 * 
 * Returns:
 * 
 * | dir | ntiles   |
 * | --- | -------- |
 * |  0  | 1        |
 * |  1  | ntiles.x |
 * |  2  | 1        |
 * |  3  | ntiles.y |
 * |  4  | 0        |
 * |  5  | ntiles.y |
 * |  6  | 1        |
 * |  7  | ntiles.x |
 * |  8  | 1        |
 * 
 * @note Direction complies to `edge_shift_dir()`
 * 
 * @param dir               Direction (0-8)
 * @param ntiles            Number of local tiles (x,y)
 * @return unsigned int     Number of edge tiles in the specified direction
 */
__host__ __device__
inline constexpr int edge_ntiles( const int dir, const int2 ntiles ) {
    int size = 1;                                 // corners
    if ( dir == 1 || dir == 7 ) size = ntiles.x;  // y boundary
    if ( dir == 3 || dir == 5 ) size = ntiles.y;  // x boundary
    if ( dir == 4 ) size = 0;                     // local

    return size;
}

/**
 * @brief Offset (from first edge tile) per direction
 * 
 * Returns:
 * 
 * | idx |            start            |
 * | --- | --------------------------- |
 * |  0  | 0                           |
 * |  1  | 1                           |
 * |  2  | 1 +   ntiles.x              |
 * |  3  | 2 +   ntiles.x              |
 * |  4  | 2 +   ntiles.x +   ntiles.y |
 * |  5  | 2 +   ntiles.x +   ntiles.y |
 * |  6  | 2 +   ntiles.x + 2*ntiles.y |
 * |  7  | 3 +   ntiles.x + 2*ntiles.y |
 * |  8  | 3 + 2*ntiles.x + 2*ntiles.y |
 * 
 * @note This assumes the edge tile (number of particle) information is stored
 *       in a contiguous buffer, following the same order as set by
 *      `edge_shift_dir()` and sizes according to `edge_ntiles()`
 * 
 * @param dir       Direction (0-8)
 * @param ntiles    Number of local tiles (x,y)
 * @return int      Start of edge tiles in the specified direction
 */
__host__ __device__
inline constexpr int edge_tile_start( const int dir, const int2 ntiles ) {
    int a, b, c;
    a = b = c = 0;

    if (dir > 0) a = 1;
    if (dir > 2) a = 2;
    if (dir > 6) a = 3;

    if (dir > 1) b =     ntiles.x;
    if (dir > 7) b = 2 * ntiles.x;

    if (dir > 3) c =     ntiles.y;
    if (dir > 5) c = 2 * ntiles.y;

    return a + b + c ;
}

/**
 * @brief Direction from edge tile id.
 *
 * @note Edge tile id is counted from the first edge tile, i.e., this function
 *       is the inverse from edge_tile_start()
 * 
 * @param edge_idx      Edge tile index from first edge tile
 * @param ntiles        Number of local tiles
 * @return  int         Edge tile direction (0-8) (never 4)
 */
__host__ __device__
inline constexpr int edge_tile_dir( const int edge_idx, const int2 ntiles ) {

#if 0
    // Simple loop
    for( int i = 0; i < 8; i++ ) {
        if ( i != 4 && edge_idx < part::edge_tile_start(i+1, ntiles) ) { 
            return i;
        }
    }
#else
    // manually unrolled loop
    if ( edge_idx < part::edge_tile_start(1, ntiles) ) return 0;
    if ( edge_idx < part::edge_tile_start(2, ntiles) ) return 1;
    if ( edge_idx < part::edge_tile_start(3, ntiles) ) return 2;
    if ( edge_idx < part::edge_tile_start(4, ntiles) ) return 3;
    if ( edge_idx < part::edge_tile_start(6, ntiles) ) return 5;
    if ( edge_idx < part::edge_tile_start(7, ntiles) ) return 6;
    if ( edge_idx < part::edge_tile_start(8, ntiles) ) return 7;
#endif

    return 8;
}

/**
 * @brief Checks whether selected edge is connectec to another node
 *
 * @details Returns true if communication along the selected edge will be
 *          required. Direction 4 (local) always returns true; callers are
 *          expected to handle it separately.
 *
 * @param dir           Direction (0-8)
 * @param local_bnd     Local boundary type (none, local periodic, comm)
 * @return bool         True if the direction corresponds to a comm boundary
 */
__host__ __device__
inline bool edge_dir_comm( const int dir, part::bnd_type const local_bnd ) {
    int dx, dy;
    part::edge_shift_dir( dir, dx, dy );

    if ( dx < 0 && local_bnd.x.lower != part::bnd_t::comm ) return false;
    if ( dx > 0 && local_bnd.x.upper != part::bnd_t::comm ) return false;
    if ( dy < 0 && local_bnd.y.lower != part::bnd_t::comm ) return false;
    if ( dy > 0 && local_bnd.y.upper != part::bnd_t::comm ) return false;

    return true;
}

/**
 * @brief Gets tile id from coordinates, including edge tiles
 * 
 * @note Assumes edge tile information is in the same tile buffer 
 *       beggining at the end of the local tile information (position 
 *       `ntiles.y * ntiles.x`) and following the order set by
 *       `edge_shift_dir()`
 * 
 * @param coords        Tile coordinates
 * @param ntiles        Tile grid dimensions
 * @param local_bnd     local boundary type (none, local periodic, comm)
 * @return int          Tile id on success, -1 on out of bounds
 */
__host__ __device__
inline int tid_coords( int2 coords, int2 const ntiles, part::bnd_type const local_bnd ) {

    // assert( coords.x >= -1 && coords.x <= ntiles.x );
    // assert( coords.y >= -1 && coords.y <= ntiles.y );

    // Local (non-parallel) periodic wrap
    if ( local_bnd.x.lower == part::bnd_t::periodic ) {
        if      ( coords.x < 0 )         coords.x += ntiles.x; 
        else if ( coords.x >= ntiles.x ) coords.x -= ntiles.x;
    }

    // Local (non-parallel) y periodic
    if ( local_bnd.y.lower == part::bnd_t::periodic ) {
        if      ( coords.y < 0 )         coords.y += ntiles.y;
        else if ( coords.y >= ntiles.y ) coords.y -= ntiles.y;
    }

    // Edge (communication with other nodes) shift
    const int dx  = ( coords.x >= ntiles.x ) - ( coords.x < 0 );
    const int dy  = ( coords.y >= ntiles.y ) - ( coords.y < 0 );
    const int dir = part::edge_dir_shift( dx, dy );

    // Local tile
    if ( dir == 4 ) return coords.y * ntiles.x + coords.x;

    // No neighbor in this direction
    if ( ! part::edge_dir_comm( dir, local_bnd ) ) return -1;

    int tid = ntiles.y * ntiles.x +                 // base
              part::edge_tile_start( dir, ntiles );

    if      ( dy == 0 ) tid += coords.y;    // x boundary (dir 3, 5)
    else if ( dx == 0 ) tid += coords.x;    // y boundary (dir 1, 7)
                                            // corners take no offset

    return tid;
}

/**
 * @brief Local tile id receiving the k-th edge tile of a given direction
 *
 * @details Edge tile data arrives ordered along the shared boundary. This maps
 *          position `k` within direction `dir` onto the local tile grid: the
 *          bottom row for dir 1, the left column for dir 3, the single corner
 *          tile for dir 0, and so on.
 *
 * @note Direction complies to `edge_shift_dir()`. Valid for
 *       `0 <= k < edge_ntiles( dir, ntiles )`.
 *
 * @param dir       Direction (0-8)
 * @param k         Position within the direction's edge tiles
 * @param ntiles    Number of local tiles (x,y)
 * @return int      Local tile id
 */
__host__ __device__
inline constexpr int local_edge_tid( const int dir, const int k, const int2 ntiles ) {
    const int stride = ( dir == 3 || dir == 5 ) ? ntiles.x : 1;

    const int xoff = ( dir % 3 == 2 ) ? ntiles.x - 1 : 0;
    const int yoff = ( dir / 3 == 2 ) ? ( ntiles.y - 1 ) * ntiles.x : 0;

    return k * stride + yoff + xoff;
}

/**
 * @brief Total number of edge tiles
 * 
 * @param ntiles    Local number of tiles (x,y)
 * @return int      Total number of tiles
 */
__host__ __device__
inline constexpr int msg_tiles( const uint2 ntiles ) {
    return  2 * ntiles.y +          // x boundary
            2 * ntiles.x +          // y boundary
            4;                      // corners
}

/**
 * @brief Total number of local tiles
 * 
 * @param ntiles    Local number of tiles (x,y)
 * @return int      Total number of tiles
 */
__host__ __device__
inline constexpr int local_tiles( const uint2 ntiles ) {
    return ntiles.x * ntiles.y;
}

/**
 * @brief Total number of tiles, including edge tiles
 * 
 * @param ntiles    Local number of tiles (x,y)
 * @return int      Total number of tiles
 */
__host__ __device__
inline constexpr int all_tiles( const uint2 ntiles ) {
    return local_tiles( ntiles ) + msg_tiles( ntiles );
}

} // namespace part