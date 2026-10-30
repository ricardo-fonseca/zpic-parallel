#pragma once

#include "part_tiles.cuh"

namespace part {

/**
 * @brief Returns rank of MPI neighbor according to shift
 * 
 * @param shiftx        Direction x-shift (-1,0,+1)
 * @param shifty        Direction y-shift (-1,0,+1)
 * @param partition     Parallel partition
 * @param periodic      Global (particle) periodic boundaries
 * @return int          Neighbor rank, -1 if no neighbor
 */
inline int shift_neighbor( const int shiftx, const int shifty, 
    const mpi::cart2d & partition, const int2 periodic ) {

    const int2 dims   = make_int2( partition.dims.x, partition.dims.y );

    // Single node along this axis: never requires messages
    if ( shiftx != 0 && dims.x == 1 ) return -1;
    if ( shifty != 0 && dims.y == 1 ) return -1;

    const auto coords = partition.get_coords();

    // Shift crosses a non-periodic global boundary
    // This accounts for non-periodic particle boundaries on periodic parallel partitions
    if ( ! periodic.x && ( ( shiftx < 0 && coords.x == 0 ) ||
                           ( shiftx > 0 && coords.x == dims.x - 1 ) ) ) return -1;
    if ( ! periodic.y && ( ( shifty < 0 && coords.y == 0 ) ||
                           ( shifty > 0 && coords.y == dims.y - 1 ) ) ) return -1;

    // Get communication neighbor
    const int neighbor = partition.get_neighbor( shiftx, shifty );
    return ( neighbor != partition.get_rank() ) ? neighbor : -1;
}


/**
 * @brief Returns rank of MPI neighbor according to direction
 * 
 * @note Messages to self are not allowed; if a node is its own neighbor,
 *       e.g. parallel partition with only 1 node, return -1.
 *
 * @param dir           Direction (0-8)
 * @param partition     Parallel partition
 * @param periodic      Global (particle) periodic boundaries
 * @return int          Neighbor rank, -1 if no neighbor
 */
inline int dir_neighbor( const int dir, const mpi::cart2d & partition, const int2 periodic ) {
    int shiftx, shifty;
    part::edge_shift_dir( dir, shiftx, shifty );
    return shift_neighbor( shiftx, shifty, partition, periodic );
}

}