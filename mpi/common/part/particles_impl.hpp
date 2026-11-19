/**
 * @file particles_impl.hpp
 * @brief Generic tiled particle buffer (common implementation)
 * 
 * @details This file is not meant to be compiled directly. Each particle type
 *          creates a local source file (e.g. `cyl_part/particles.cpp`) that
 *          includes the local wrapper header, defines the namespace and then
 *          includes this file:
 * 
 *              #include "particles.hpp"
 *              #define PARTICLES_NAMESPACE cyl_part
 *              #include "part/particles_impl.hpp"
 * 
 *          The particle quantities are not needed here; all per-quantity
 *          operations go through `particles_view` methods (`load()`,
 *          `store()`, `pack()`, `unpack()`, `check_finite()`) generated in
 *          `common/part/particles_common.hpp`.
 * 
 * @note    This file is only meant to be included (it uses the .hpp extension
 *          so that build system `*.cpp` globs do not pick it up), it will fail
 *          with a #error if compiled on its own.
 */

#ifndef PARTICLES_NAMESPACE
#error "PARTICLES_NAMESPACE must be defined before including common/part/particles_impl.hpp"
#endif

#include <iostream>
#include <sstream>
#include <string>
#include <cmath>
#include <vector>
#include <limits>

#include "../parallel/omp.hpp"

// Everything (including helper functions) is defined inside the particle
// namespace, so that several particle types can be linked together
namespace PARTICLES_NAMESPACE {

/**
 * @brief   Check which particles have left the tile and determine new number
 *          of particles per tile.
 * 
 * @warning This function expects that sort.new_np has been zeroed before being
 *          called.
 * 
 * @param data          (in) Particle data
 * @param sort          (out) Sort data (new number of particles per tile, indices
 *                      particles leaving the tile, etc.)
 * @param local_bnd     (in) Information on local node boundaries
 */
void bnd_check( 
    particles_view data, 
    part::particle_sort_view sort, 
    const part::bnd_type local_bnd)
{
    // ntiles needs to be set to signed because of the comparisons below
    const int2 ntiles = make_int2( data.local_ntiles.x, data.local_ntiles.y );
    const int2 tile_dims = make_int2( data.tile_dims.x, data.tile_dims.y );

    #pragma omp parallel for schedule(dynamic)
    for( int tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

        const int tx = tid % ntiles.x;
        const int ty = tid / ntiles.x;

        const int np     = data.tile_np[ tid ];
        const int offset = data.tile_offset[ tid ];

        int2 const * __restrict__ ix = &data.ix[ offset ];

        /// @brief Indices of particles leaving tile (relative to tile start)
        int * __restrict__ idx = &sort.idx[ offset ];

        /// @brief Number of particles moving in each direction
        int _npt[9];
        for( int i = 0; i < 9; ++i ) _npt[i] = 0;
        
        /// @brief Number of particle leaving tile
        int _nout = 0;

        // Count particles according to their motion
        // Store indices of particles leaving tile
        for( int i = 0; i < np; ++i ) {
            const int2 ipos = ix[i];
            const int xcross = ( ipos.x >= tile_dims.x ) - ( ipos.x < 0 );
            const int ycross = ( ipos.y >= tile_dims.y ) - ( ipos.y < 0 );
            
            if ( xcross || ycross ) {
                _npt[ (ycross+1) * 3 + (xcross+1) ] += 1;
                idx[ _nout ] = i; _nout += 1;
            }
        }

        // particles remaining on the tile
        _npt[4] = np - _nout;

        for( int i = 0; i < 9; ++i ) {
            
            // Find target tile
            const int2 target = make_int2( tx + i % 3 - 1, ty + i / 3 - 1 );
            const int target_tid = part::tid_coords( target, ntiles, local_bnd );
            
            if ( target_tid >= 0 && _npt[i] > 0 ) {
                #pragma omp atomic
                sort.new_np[ target_tid ] += _npt[i];
            }
        }

        int * __restrict__ npt = &sort.npt[ 9*tid ];
        for( int i = 0; i < 9; i++ ) npt[ i ] = _npt[i];
        sort.nidx[ tid ] = _nout;
    }
}

/**
 * @brief Prepare particle receive buffers and start receive
 * 
 * @param sort      Temporary sort index 
 * @param recv      Receive message object 
 */
void particles::irecv_msg( part::particle_sort &sort, part::particle_message &recv ) {

    /// @brief Total size (bytes) of data to be received
    uint32_t total_size = 0;

    // Set individual message sizes:
    for( int i = 0; i < 9; i++) {
        if ( i != 4 ) {
            recv.size[i] = sort.recv.msg_np[i] * particle_size();
            total_size += recv.size[i];
        } else {
            recv.size[i] = 0;
        }
    }

    // Grow message buffer if need be
    recv.check_buffer( total_size );

    // Start receive
    recv.irecv();
}

/**
 * @brief Recalculates particle tile offset, leaving room for additional particles
 * 
 * @note The routine also leaves room for particles coming from other MPI nodes.
 *       The number of particles in each tile is set to 0
 * 
 * @param tmp           (out) Temp. Particle buffer
 * @param new_np        (in) New number of particles per tile, including edge
 *                      tiles (particles leaving the node), from `bnd_check()`
 * @param recv_np       (in) Number of incoming particles per edge tile, from
 *                      `particle_sort::exchange_np()`
 * @param extra         (in) Additional particles per local tile (optional)
 * @return uint32_t     (out) Total number of particles (including additional ones)
 */
uint32_t update_tile_info( 
    particles_view tmp, 
    const int * __restrict__ new_np,
    const int * __restrict__ recv_np,
    const int * __restrict__ extra = nullptr ) {

    // Include ghost tiles in calculations
    const int ntiles     = part::local_tiles( tmp.local_ntiles );
    const int ntiles_all = part::all_tiles( tmp.local_ntiles );

    int * __restrict__ offset = tmp.tile_offset;
    int * __restrict__ np     = tmp.tile_np;

    // Initialize offset[] with the new number of particles
    // (extra array only includes data for local tiles)
    for( int i = 0; i < ntiles_all; i++ ) {
        offset[i] = new_np[i];
        np[i] = 0;
    }

    if ( extra != nullptr ) {
        for( int i = 0; i < ntiles; i++ ) offset[i] += extra[i];
    }

    // Add incoming particles
    const int2 nt = make_int2( tmp.local_ntiles.x, tmp.local_ntiles.y );

    for( int dir = 0; dir < 9; dir++ ) {
        const int start = part::edge_tile_start( dir, nt );

        for( int k = 0; k < part::edge_ntiles( dir, nt ); k++ ) {
            offset[ part::local_edge_tid( dir, k, nt ) ] += recv_np[ start + k ];
        }
    }

    // Exclusive scan
    uint32_t total = 0;
    for( int i = 0; i < ntiles_all; i++ ) {
        const uint32_t n = offset[i];
        offset[i] = total;
        total += n;
    }

    // Total number of particles
    return total;
}

/**
 * @brief Copy outgoing particles to temporary buffer
 * 
 * @note particles leaving the tile are copied to a temporary particle buffer
 *       into the tile that will hold the data after the sort and that is
 *       currently empty.
 * 
 *       If particles are copied from the middle of the buffer, a particle will
 *       be copied from the end of the buffer to fill the hole.
 * 
 *       If the tile data position/limits in the main buffer will change,
 *       particles that stay in the tile but are now in invalid positions will
 *       be shifted.
 * 
 * @param data      Particle data
 * @param tmp       Temporary particle buffer (has new offsets)
 * @param sort      Sort data (new number of particles per tile, indices of
 *                  particles leaving the tile, etc.)
 * @param local_bnd Information on local node boundaries
 */
void copy_out( 
    particles_view data, particles_view tmp, const part::particle_sort_view sort,
    const part::bnd_type local_bnd )
{
    const int2 ntiles = make_int2( data.local_ntiles.x, data.local_ntiles.y );
    const int2 lim = make_int2( data.tile_dims.x, data.tile_dims.y );

    #pragma omp parallel for schedule(dynamic)
    for( int tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

        const int tx = tid % ntiles.x;
        const int ty = tid / ntiles.x;

        int const old_offset      = data.tile_offset[ tid ];
        int const * __restrict__ npt = &sort.npt[ 9*tid ];

        int2 const * __restrict__ ix = &data.ix[ old_offset ];

        int const * __restrict__ idx = &sort.idx[ old_offset ];
        int const nidx            = sort.nidx[ tid ];

        int const new_offset = tmp.tile_offset[ tid ];
        int const new_np     = sort.new_np[ tid ];

        /// @brief Offsets (in tmp buffer) for each of the 9 target tiles
        int _dir_offset[9];

        // Number of particles staying in tile
        const int n0 = npt[4];

        // Number of particles staying in the tile that need to be copied to temp memory
        // because tile position in memory has shifted
        int nshift;
        if ( new_offset >= old_offset ) {
            // Buffer has shifted right, copy particles left behind to end of buffer
            nshift = new_offset - old_offset;
        } else {
            // Buffer has shifted left, attempt to fill initial space with particles
            // coming from other tiles, use additional particles from end of buffer
            // if needed
            nshift = (old_offset + n0) - (new_offset + new_np);
            if ( nshift < 0 ) nshift = 0;
        }
        
        // At most n0 particles will be shifted
        if ( nshift > n0 ) nshift = n0;

        // Reserve space in the tmp array
        _dir_offset[4] = new_offset + omp::atomic_fetch_add( & tmp.tile_np[ tid ], nshift );

        // Find offsets on new buffer
        for( int i = 0; i < 9; i++ ) {
            if ( i != 4 ) {
                const int2 target = make_int2( tx + i % 3 - 1, ty + i / 3 - 1 );
                const int target_tid = part::tid_coords( target, ntiles, local_bnd );
                
                if ( target_tid >= 0 ) {
                    // If valid neighbour tile reserve space on tmp. array
                    _dir_offset[i] = tmp.tile_offset[ target_tid ] + 
                        omp::atomic_fetch_add( &tmp.tile_np[ target_tid ], npt[ i ] );
                } else {
                    // Otherwise mark offset as invalid
                    _dir_offset[i] = -1;
                }
            } 
        }

        // Copy particles moving away from tile and fill holes
        int _c = n0;
        for( int i = 0; i < nidx; i++ ) {
            
            const int k = idx[i];

            particle_t p = data.load( old_offset + k );

            const int xcross = ( p.ix.x >= lim.x ) - ( p.ix.x < 0 );
            const int ycross = ( p.ix.y >= lim.y ) - ( p.ix.y < 0 );

            const int dir = (ycross+1) * 3 + (xcross+1);

            // Check if particle crossed into a valid neighbor
            if ( _dir_offset[dir] >= 0 ) {        

                const int l = _dir_offset[dir]; _dir_offset[dir] += 1;

                // Correct positions - ix is ok for new tile
                p.ix.x -= xcross * lim.x;
                p.ix.y -= ycross * lim.y;

                tmp.store( l, p );
            }

            // Fill hole if needed
            if ( k < n0 ) {
                int c, invalid;

                do {
                    c = _c; _c += 1;
                    invalid = ( ix[c].x < 0 ) || ( ix[c].x >= lim.x ) || 
                              ( ix[c].y < 0 ) || ( ix[c].y >= lim.y );
                } while (invalid);

                data.store( old_offset + k, data.load( old_offset + c ) );
            }
        }

        // At this point all particles up to n0 are correct

        // Copy particles that need to be shifted
        // We've reserved space for nshift particles earlier
        const int new_idx = _dir_offset[4];

        // Copy from beginning (buffer shifted right) or end (shifted left) of buffer
        const int old_idx = ( new_offset >= old_offset ) ? 0 : n0 - nshift;
        for( int i = 0; i < nshift; i++ ) {
            tmp.store( new_idx + i, data.load( old_offset + old_idx + i ) );
        }

        // Store current number of local particles
        // These are already in the correct position in global buffer
        data.tile_np[ tid ] = n0 - nshift;
    }
}

/**
 * @brief Pack particles moving out of the node into a message buffer and start send
 * 
 * @note Edge (message) tiles of the same direction are contiguous in `tmp`,
 *       so each direction is packed with a single `pack()` call. The message
 *       layout is the same as in the GPU version: `np` cell indices followed
 *       by `np` values of each quantity.
 * 
 * @param tmp       Temporary buffer holding particles moving away from tiles
 * @param sort      Temporary sort index
 * @param send      Send message object
 */
void particles::isend_msg( particles &tmp, part::particle_sort &sort, part::particle_message &send ) {

    /// @brief Total number of particles being sent
    uint32_t send_np = 0;

    /// @brief Offset in message buffer for each direction (number of particles)
    uint32_t dir_off[9];

    // Get offsets and check send buffer size
    for( int i = 0; i < 9; i++ ) {
        dir_off[i] = send_np;
        if (i != 4) {
            send.size[i] = sort.send.msg_np[i] * particle_size();
            send_np += sort.send.msg_np[i];
        } else {
            send.size[i] = 0;
        }
    }

    if ( send_np > 0 ) {
        // Check if msg buffer is large enough
        send.check_buffer( send_np * particle_size() );

        const int2 nt = make_int2( tmp.local_ntiles.x, tmp.local_ntiles.y );
        const int  local_tiles = part::local_tiles( tmp.local_ntiles );
        const particles_view tmp_view = tmp.view();

        // Pack data
        #pragma omp parallel for schedule(dynamic)
        for( int dir = 0; dir < 9; dir++ ) {
            const int np = sort.send.msg_np[ dir ];
            if ( dir != 4 && np > 0 ) {
                ///@brief Offset of first edge tile of this direction in tmp buffer
                const int src = tmp_view.tile_offset[ local_tiles + part::edge_tile_start( dir, nt ) ];

                ///@brief send message buffer for this direction
                uint8_t * msg_buffer = &send.buffer[ dir_off[ dir ] * particle_size() ];

                tmp_view.pack( msg_buffer, packed_offsets( 0, np ), src, np );
            }
        }
    }

    // Start sending messages
    send.isend();
}

/**
 * @brief Copy incoming particles to main buffer. Buffer will be fully sorted after
 *        this step
 * 
 * @param data      Main particle data
 * @param tmp       Temporary particle data
 */
void copy_in( particles_view data, particles_view tmp )
{
    const int2 ntiles = make_int2( data.local_ntiles.x, data.local_ntiles.y );

    #pragma omp parallel for schedule(dynamic)
    for( int tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

        const int old_offset       =  data.tile_offset[ tid ];
        const int old_np           =  data.tile_np[ tid ];

        const int new_offset       =  tmp.tile_offset[ tid ];
        const int tmp_np           =  tmp.tile_np[ tid ];

        // Number of particles to add to the beginning of the buffer
        int np0 = 0;
        if ( new_offset < old_offset ) {
            np0 = old_offset - new_offset;
            if ( np0 > tmp_np ) np0 = tmp_np;
        }
            
        for( int i = 0; i < np0; i ++ ) {
            data.store( new_offset + i, tmp.load( new_offset + i ) );
        }

        // Add remaining particles to the end of the buffer
        for( int i = np0; i < tmp_np; i ++ ) {
            data.store( new_offset + old_np + i, tmp.load( new_offset + i ) );
        }

        // Store the new offset and number of particles
        data.tile_np[ tid ]     = old_np + tmp_np;
        data.tile_offset[ tid ] = new_offset;
    }
}

/**
 * @brief Unpack received particle data into main particle data buffer
 * 
 * @details Multiple message tiles may write to the same local tile (e.g. the
 *          corner tiles), so target positions are computed serially first;
 *          the data copy is then done in parallel.
 * 
 * @param sort      Temporary sort index
 * @param recv      Receive message object
 */
void particles::unpack_msg( part::particle_sort &sort, part::particle_message &recv ) {

    /// @brief number of particles per received tile
    const int * __restrict__ msg_tile_np = sort.recv.buffer;

    const int2 nt = make_int2( local_ntiles.x, local_ntiles.y );
    const int msg_tiles = part::msg_tiles( local_ntiles );

    /// @brief Destination position in particle buffer for each message tile
    std::vector<int> tgt_offset( msg_tiles );
    /// @brief Byte offset of the message (direction) holding each message tile
    std::vector<size_t> msg_offset( msg_tiles );
    /// @brief Packed offsets inside the message for each message tile
    std::vector<packed_offsets> src;
    src.reserve( msg_tiles );

    /// @brief Message offset in receive buffer (number of particles)
    int recv_off = 0;

    for( int dir = 0; dir < 9; dir++ ) {

        ///@brief number of particles in this message
        const int msg_np = sort.recv.msg_np[dir];

        ///@brief first edge tile of this direction in the received count buffer
        const int start = part::edge_tile_start( dir, nt );

        ///@brief number of particles unpacked from this message
        int np_unpack = 0;

        for( int k = 0; k < part::edge_ntiles( dir, nt ); k++ ) {
            const int e = start + k;
            const int recv_np = msg_tile_np[ e ];
            const int target_tid = part::local_edge_tid( dir, k, nt );

            tgt_offset[ e ] = tile_offset[ target_tid ] + tile_np[ target_tid ];
            msg_offset[ e ] = static_cast<size_t>( recv_off ) * particle_size();
            src.emplace_back( np_unpack, msg_np );

            tile_np[ target_tid ] += recv_np;
            np_unpack += recv_np;
        }
        recv_off += msg_np;
    }

    const particles_view data = view();

    #pragma omp parallel for schedule(dynamic)
    for( int e = 0; e < msg_tiles; e++ ) {
        const int recv_np = msg_tile_np[ e ];
        if ( recv_np > 0 ) {
            data.unpack( &recv.buffer[ msg_offset[e] ], src[e], tgt_offset[e], recv_np );
        }
    }
}

/**
 * @brief Copies all particles to correct tiles in another buffer, in a single
 *        pass
 * 
 * @details Every particle in the tile is read once and written once to its
 *          final position in `tmp`. Space for each of the 9 possible
 *          destinations (including the tile itself, dir = 4) is reserved up
 *          front using the counts from `bnd_check()` (`sort.npt`). Particles
 *          leaving the node are stored in the "edge" (message) tiles of
 *          `tmp`, exactly as in `copy_out()`, so that `isend_msg()` can pack
 *          them from `tmp`. Particles crossing a non-communicating,
 *          non-periodic boundary are dropped.
 * 
 *          Unlike `copy_out()` the source buffer is never modified (no hole
 *          filling), and the `sort.idx` / `sort.nidx` arrays are not used.
 * 
 * @note Requires that new buffer (`tmp`) already has the correct offset
 *       values, and number of particles set to 0, i.e., that
 *       `update_tile_info()` has been called on `tmp`.
 * 
 * @note After this call `tmp.tile_np` holds, for local tiles, the number of
 *       particles from this node (incoming MPI particles are added later by
 *       `unpack_msg()`), and, for edge tiles, the number of particles to be
 *       sent to other nodes.
 * 
 * @param data      Particle data
 * @param tmp       Temporary particle buffer (has new offsets)
 * @param sort      Sort data (number of particles moving in each direction)
 * @param local_bnd Information on local node boundaries
 */
void copy_sorted( 
    particles_view data, particles_view tmp, const part::particle_sort_view sort,
    const part::bnd_type local_bnd )
{
    const int2 ntiles = make_int2( data.local_ntiles.x, data.local_ntiles.y );
    const int2 lim = make_int2( data.tile_dims.x, data.tile_dims.y );

    #pragma omp parallel for schedule(dynamic)
    for( int tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

        const int tx = tid % ntiles.x;
        const int ty = tid / ntiles.x;

        const int offset = data.tile_offset[ tid ];
        const int np     = data.tile_np[ tid ];

        /// @brief Number of particles moving in each direction (npt[4] is staying)
        int const * __restrict__ npt = &sort.npt[ 9*tid ];

        /// @brief Offsets (in tmp buffer) for each of the 9 target tiles
        int _dir_offset[9];

        // Reserve space on new buffer for all 9 directions (including dir = 4)
        for( int i = 0; i < 9; i++ ) {
            const int n = npt[i];
            int off = -1;
            if ( n > 0 ) {
                const int2 target = make_int2( tx + i % 3 - 1, ty + i / 3 - 1 );
                const int target_tid = part::tid_coords( target, ntiles, local_bnd );

                // If valid target tile reserve space on tmp. array, otherwise
                // offset is left as invalid (-1) and particles will be dropped
                if ( target_tid >= 0 ) {
                    off = tmp.tile_offset[ target_tid ] + 
                          omp::atomic_fetch_add( &tmp.tile_np[ target_tid ], n );
                }
            }
            _dir_offset[i] = off;
        }

        for( int i = 0; i < np; i++ ) {
            particle_t p = data.load( offset + i );

            const int xcross = ( p.ix.x >= lim.x ) - ( p.ix.x < 0 );
            const int ycross = ( p.ix.y >= lim.y ) - ( p.ix.y < 0 );
            const int dir = ( ycross + 1 ) * 3 + ( xcross + 1 );

            if ( _dir_offset[dir] >= 0 ) {
                // Correct positions - ix is ok for new tile
                p.ix.x -= xcross * lim.x;
                p.ix.y -= ycross * lim.y;

                tmp.store( _dir_offset[dir]++, p );
            }
        }
    }
}

/**
 * @brief Gets total number of particles
 * 
 * @note This will work even if the buffer is not compact, i.e., if there is
 *       free space between tiles.
 * 
 * @return uint32_t 
 */
uint32_t particles::local_np() const {
    uint32_t np = 0;
    for( unsigned i = 0; i < local_ntiles.x * local_ntiles.y; i++ )
        np += tile_np[i];
    return np;
}

/**
 * @brief Gets maximum number of particles in a single tile
 * 
 * @return uint32_t 
 */
uint32_t particles::tile_np_max() const {
    int max_np = tile_np[0];
    for( unsigned i = 1; i < local_ntiles.x * local_ntiles.y; i++ ) {
        if ( tile_np[i] > max_np ) max_np = tile_np[i];
    }
    return max_np;
}

/**
 * @brief Gets minimum number of particles in a single tile
 * 
 * @return uint32_t 
 */
uint32_t particles::tile_np_min() const {
    int min_np = tile_np[0];
    for( unsigned i = 1; i < local_ntiles.x * local_ntiles.y; i++ ) {
        if ( tile_np[i] < min_np ) min_np = tile_np[i];
    }
    return min_np;
}

/**
 * @brief Moves particles to the correct tiles
 * 
 * @note particles are only expected to have moved no more than 1 tile
 *       in each direction.
 * 
 * @param tmp       Temporary particle buffer
 * @param sort      Temporary sort index 
 * @param extra     Additional space to add to each tile. Leaves  room for
 *                  particles to be injected later.
 */
void particles::tile_sort( particles & tmp, part::particle_sort & sort, const int * __restrict__ extra ) {

    // Reset sort data
    sort.reset();

    // Get new number of particles per tile
    bnd_check ( view(), sort.view(), local_bnd );

    // Exchange number of particles in edge cells
    sort.exchange_np();

    // Post particle data receives
    irecv_msg( sort, recv );

    // Get new offsets, including:
    // - Incoming particles from other MPI nodes
    // - New particles that will be injected (if any)
    auto total_np = update_tile_info ( tmp.view(), sort.view().new_np, sort.recv.buffer, extra );

    // Both buffers must be able to hold all particles in the staging path
    if ( total_np > max_part || total_np > tmp.max_part ) { 

        // Grow tmp particle buffer (tile_offset / tile_np are preserved)
        tmp.grow_buffer( total_np );

        // Copy all particles to correct tiles in tmp buffer. Particles
        // leaving the node are placed in the edge tiles of tmp
        copy_sorted( view(), tmp.view(), sort.view(), local_bnd );

        // Pack particle data and start sending (from tmp edge tiles)
        isend_msg( tmp, sort, send );

        // Swap buffers, *this now holds the sorted data and new tile info
        swap_buffers( *this, tmp );

        // Grow tmp particle buffer for future use (old data is discarded)
        tmp.grow_buffer( max_part );

        // sort.idx is indexed by particle buffer position, so it must also
        // be able to hold max_part values for the next bnd_check()
        sort.check_buffer( max_part );

    } else {
        // Copy outgoing particles (and particles needing shifting) to staging area
        copy_out ( view(), tmp.view(), sort.view(), local_bnd );

        // Pack particle data and start sending
        isend_msg( tmp, sort, send );

        // Copy local particles from staging area into final positions in partile buffer
        copy_in ( view(), tmp.view() );
    }

    // Wait for receive messages to complete
    recv.wait();

    // Unpack received data
    unpack_msg( sort, recv );

    // Wait for sends to complete
    send.wait();

    // For debug only, remove from production code
    // parallel.barrier();
    // validate( "after tile_sort" );
}

/**
 * @brief Shifts particle cells by the required amount
 * 
 * Cells are shifted by adding the parameter `shift` to the particle cell
 * indexes.
 * 
 * Note that this routine does not check if the particles are still inside the
 * tile.
 * 
 * @param shift     Cell shift in both directions
 */
void particles::cell_shift( int2 const shift ) {

    #pragma omp parallel for schedule(dynamic)
    for( unsigned tid = 0; tid < local_ntiles.y * local_ntiles.x; tid++ ) {
        const auto offset = tile_offset[ tid ];
        const auto np     = tile_np[ tid ];

        int2 * const __restrict__ t_ix = &ix[ offset ];

        for( int i = 0; i < np; i++ ) {
            int2 cell = t_ix[i];
            cell.x += shift.x;
            cell.y += shift.y;
            t_ix[i] = cell;
        }
    }
}

/**
 * @brief Checks particle buffer data for error
 * 
 * @warning This routine is meant for debug only and should not be called 
 *          for production code.
 * 
 * The routine will check for:
 *      1. Bad tile np values (< 0)
 *      2. Inconsistent tile offset values (the buffer should be packed)
 *      3. Invalid cell data (out of tile bounds)
 *      4. Invalid position data (out of [-0.5,0.5[), if a quantity named
 *         `x` is available
 *      5. Invalid floating point values (nan, inf) in any quantity
 * 
 * If there are any errors found the routine will exit the code.
 * 
 * @param msg       Message to print in case error is found
 * @param over      Amount of extra cells indices beyond limit allowed. Used
 *                  when checking the buffer before tile_sort()
 */
void particles::validate( std::string msg, int const over ) {

    if ( msg.empty() ) {
        mpi::cout << "validating particle set...";
    } else {
        mpi::cout << "validating particle set (" << msg << ")...";
    }

    int err = 0;

    // Check offset / np buffer
    for( unsigned tile_id = 0; tile_id < local_ntiles.x * local_ntiles.y; ++tile_id ) {
        if ( tile_np[tile_id] < 0 ) {
            mpi::cout << "\ntile[" << tile_id << "] - bad np (" << tile_np[ tile_id ] << "), should be >= 0";
            err = 1;
        }

        if ( tile_id > 0 ) {
            auto prev = tile_offset[ tile_id-1] + tile_np[ tile_id-1];
            if ( prev != tile_offset[ tile_id ] ) {
                mpi::cout << "\ntile[" << tile_id << "] - bad offset (" << tile_offset[ tile_id ] << ")"
                          << ", does not match previous tile info, should be " << prev;
                err = 1;
            }
        } else {
            if ( tile_offset[ tile_id ] != 0 ) {
                mpi::cout << "\ntile[" << tile_id << "] - bad offset (" << tile_offset[ tile_id ] << "), should be 0";
                err = 1;
            }
        }   
    }

    if ( err ) {
        mpi::fatal("\nInvalid tile information");
    }

    const int rank = parallel.get_rank();
    int2 const lb = make_int2( -over, -over );
    int2 const ub = make_int2( tile_dims.x + over, tile_dims.y + over ); 
    const particles_view data = view();

    // Loop over tiles
    for( unsigned tile_id = 0; tile_id < local_ntiles.x * local_ntiles.y; ++tile_id ) {
        const int2 tile_idx = make_int2( tile_id % local_ntiles.x, tile_id / local_ntiles.x );
        const int offset = tile_offset[ tile_id ];
        const int np     = tile_np[ tile_id ];

        for( int i = 0; i < np; i++ ) {
            int perr = 0;

            particle_t const p = data.load( offset + i );

            if ( (p.ix.x < lb.x) || (p.ix.x >= ub.x )) {
                std::printf("[%d] [%d,%d] Invalid ix[%d].x position (%d), range = [%d,%d[\n", rank, tile_idx.x, tile_idx.y, i, p.ix.x, lb.x, ub.x );
                perr = 1;
            }

            if ( (p.ix.y < lb.y) || (p.ix.y >= ub.y )) {
                std::printf("[%d] [%d,%d] Invalid ix[%d].y position (%d), range = [%d,%d[\n", rank, tile_idx.x, tile_idx.y, i, p.ix.y, lb.y, ub.y );
                perr = 1;
            }

            // Position inside cell (if available)
            if ( particle_detail::check_cell_position( p, i, rank, tile_idx ) ) perr = 1;

            // All floating point values must be finite
            if ( data.check_finite( offset + i, rank, tile_idx ) ) perr = 1;

            // Report only the first invalid particle in each tile
            if ( perr ) { err = 1; break; }
        }
    }

    if ( err ) {
        mpi::fatal("Invalid particle(s) found" );
    } else {
        mpi::cout << " particle set ok.\n";
    }
}

} // namespace PARTICLES_NAMESPACE

#undef PARTICLES_NAMESPACE
