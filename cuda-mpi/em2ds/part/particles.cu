#include "particles.hpp"

#include <iostream>
#include <sstream>
#include <string>
#include <cmath>
#include <vector>


/**
 * Optimized parameters for NVIDIA A100
 */
constexpr int opt_bnd_check_block        = 1024;
constexpr int opt_update_tile_info_block = 1024;
constexpr int opt_copy_out_block         = 1024;
constexpr int opt_copy_in_block          = 1024;


namespace kernel {

__global__
/**
 * @brief   Check which particles have left the tile and determine new number
 *          of particles per tile.
 * 
 * @warning This kernel expects that sort.new_np has been zeroed before being
 *          called.
 * 
 * @param part      (in) Particle data
 * @param sort      (out) Sort data (new number of particles per tile, indices
 *                  particles leaving the tile, etc.)
 * @param local_bnd     (in) Information on local node boundaries
 */
void __launch_bounds__(opt_bnd_check_block) bnd_check( 
    part::particles_view part, 
    part::particle_sort_view sort, 
    const part::bnd_type local_bnd)
{
    // ntiles needs to be set to signed because of the comparisons below
    const int2 ntiles = make_int2( part.local_ntiles.x, part.local_ntiles.y );
    const int2 lim = make_int2( part.tile_dims.x, part.tile_dims.y );

    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int tile_id = tile_idx.y * ntiles.x + tile_idx.x;

    const auto offset  = part.tile_offset[ tile_id ];
    const auto np      = part.tile_np[ tile_id ];

    int2 * __restrict__ ix    = &part.ix[ offset ];

    /// @brief Indices of particles leaving tile
    int  * __restrict__ idx   = &sort.idx[ offset ];

    /// @brief [shared] Number of particles moving in each direction
    __shared__ int _npt[9];
    for( auto i = gpu::block::thread_rank(); i < 9; i+= gpu::block::num_threads() ) _npt[i] = 0;

    /// @brief [shared] Number of particle leaving tile
    __shared__ int _nout;
    if ( gpu::block::thread_rank() == 0 ) _nout = 0;

    gpu::block::sync();

    // Count particles according to their motion and store indices of particles leaving tile
    for( int i = gpu::block::thread_rank(); i < np; i += gpu::block::num_threads() ) {
        int2 ipos = ix[i];
        int xcross = ( ipos.x >= lim.x ) - ( ipos.x < 0 );
        int ycross = ( ipos.y >= lim.y ) - ( ipos.y < 0 );
        
        if ( xcross || ycross ) {
            gpu::block::atomic_fetch_add( &_npt[ (ycross+1) * 3 + (xcross+1) ], 1 );
            idx[ gpu::block::atomic_fetch_add( &_nout, 1 ) ] = i;
        }
    }
    gpu::block::sync();

    // Store number of particles staying / leaving the tile
    if (gpu::block::thread_rank() == 0 ){
        // Particles remaining on the tile
        _npt[4] = np - _nout;
        // Store number of particles leaving tile
        sort.nidx[ tile_id ] = _nout;
    }
    gpu::block::sync();

    for( int i = gpu::block::thread_rank(); i < 9; i += gpu::block::num_threads() ) {
        
        // Store number of particles leaving tile in each direction
        sort.npt[ 9*tile_id + i ] = _npt[i];

        // Add number of particles to target neighboring node

        // Find target node
        int2 target = make_int2( 
                tile_idx.x + i % 3 - 1,
                tile_idx.y + i / 3 - 1
            );
        int target_tid = part::tid_coords( target, ntiles, local_bnd );
        
        if ( target_tid >= 0 ) {
            gpu::device::atomic_fetch_add( & sort.new_np[ target_tid ], _npt[i] );
        }
    }
}

} // namespace kernel

/**
 * @brief   Check which particles have left the tile and determine new number
 *          of particles per tile.
 * 
 * @warning This kernel expects that sort.new_np has been zeroed before being
 *          called.
 * 
 * @param part          (in) Particle data
 * @param sort          (out) Sort data (new number of particles per tile, indices
 *                      particles leaving the tile, etc.)
 * @param local_bnd     (in) Information on local node boundaries
 */
void bnd_check( 
    part::particles_view data, part::particle_sort_view sort, const part::bnd_type local_bnd)
{
    dim3 grid( data.local_ntiles.x, data.local_ntiles.y );
    auto block = opt_bnd_check_block;    
    kernel::bnd_check <<<grid,block>>> ( data, sort, local_bnd );
}


/**
 * @brief Prepare particle receive buffers and start receive
 * 
 * @param sort      Temporary sort index 
 * @param recv      Receive message object 
 */
void part::particles::irecv_msg( particle_sort &sort, particle_message &recv ) {

    /// @brief Total size (bytes) of data to be received
    uint32_t total_size = 0;

    // Copy msg_np from device memory
    gpu::device::memcpy_tohost( sort.recv.h_msg_np, sort.recv.d_msg_np, 9 );

    // Set individual message sizes:
    for( int i = 0; i < 9; i++) {
        if ( i != 4 ) {
            recv.size[i] = sort.recv.h_msg_np[i] * particle_size();
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

namespace kernel {

__global__
/**
 * @brief Recalculates particle tile offset, leaving room for additional particles
 * 
 * @note The routine must be called using a single block so that all threads
 *       involved in the calculation have access to the same shared memory 
 *       block
 * 
 * @note The number of particles in each tile is set to 0
 * 
 * @param tmp_part      (out) Particle buffer
 * @param new_np        (in/out) New number of particles per tile. Set to 0 after calculation.
 * @param extra         (in) Additional incoming particles
 * @param dev_np        (out) Total number of particles (including additional ones)
 */
void __launch_bounds__(opt_update_tile_info_block) update_tile_info( 
    part::particles_view tmp,
    part::particle_sort_view sort,
    const int * __restrict__ recv_buffer,
    const int * __restrict__ extra,
    uint32_t * __restrict__ dev_np ) {

    // New number of particles per tile
    const int * __restrict__ new_np = sort.new_np;

    const auto ntiles     = part::local_tiles( tmp.local_ntiles );
    const auto ntiles_all = part::all_tiles( tmp.local_ntiles );  

    int * __restrict__ offset = gpu::block::shared_mem<int>();

    // Initialize offset[] with the new number of particles
    if ( extra != nullptr ) {
        for( auto i = gpu::block::thread_rank(); i < ntiles_all; i += gpu::block::num_threads() ) {
            offset[i] = new_np[i];
            // extra array only includes data for local tiles
            if ( i < ntiles ) offset[i] += extra[i];

            // This will be set below
            // np[i] = 0
        }

    } else {
        for( auto i = gpu::block::thread_rank(); i < ntiles_all; i += gpu::block::num_threads() ) {
            offset[i] = new_np[i];
            // This will be set below
            // np[i] = 0
        }
    }

    gpu::block::sync();

    // Add incoming particles

    // Loop over directions, then 1 thread per msg tile
    const int2 nt = make_int2(tmp.local_ntiles.x, tmp.local_ntiles.y);
    for( int dir = 0; dir < 9; dir++ ) {
        const int start = part::edge_tile_start( dir, nt );

        for( int k = gpu::block::thread_rank(); k < part::edge_ntiles( dir, nt); k+= gpu::block::num_threads() ) {
            const int tid = part::local_edge_tid( dir, k, nt );
            gpu::block::atomic_fetch_add( & offset[ tid ], recv_buffer[ start + k ] );
        }
    }

    gpu::block::sync();

    // Block-wide exclusive scan 
    int total = gpu::block::exscan_add( offset, ntiles_all );

    // Copy result to device memory and set np to 0
    int * __restrict__ tmp_np  = tmp.tile_np;
    int * __restrict__ tmp_off = tmp.tile_offset;
    for( int i = gpu::block::thread_rank(); i < ntiles_all; i += gpu::block::num_threads() ) {
        tmp_np [i] = 0;
        tmp_off[i] = offset[i];
    }

    // Return total number of particles
    if ( gpu::block::thread_rank() == 0 ) {
        dev_np[0] = total;
    }
}

} // namespace kernel

/**
 * @brief Recalculates particle tile offset, leaving room for additional particles
 * 
 * @note The routine also leaves room for particles coming from other MPI nodes.
 *       The number of particles in each tile is set to 0
 * @param tmp           (out) Temp. Particle buffer
 * @param sort          (in) Sort data (includes information from other MPI nodes)
 * @param extra         (in) Additional particles (optional)
 * @return uint32_t     (out) Total number of particles (including additional ones)
 */
uint32_t update_tile_info( 
    part::particles_view tmp, 
    part::particle_sort & sort,
    gpu::device::var<uint32_t> & dev_np,
    const int * __restrict__ extra = nullptr ) {

    // Get optimal block size
    const auto ntiles_all   = part::all_tiles( tmp.local_ntiles );
    auto block = ( ntiles_all < opt_update_tile_info_block ) ? ntiles_all : opt_update_tile_info_block;

    size_t shm_size = ntiles_all * sizeof(int);
    gpu::block::set_shmem_size( kernel::update_tile_info, shm_size );
    kernel::update_tile_info <<< 1, block, shm_size >>> (
        tmp, sort.view(), sort.recv.buffer, 
        extra, dev_np.ptr() );

    return dev_np.get();
}


namespace kernel {

__global__
/**
 * @brief CUDA kernel for copying outgoing particles to temporary buffer
 * 
 * @note Particles leaving the tile are copied to a temporary particle buffer
 *       into the tile that will hold the data after the sort and that is
 *       currently empty.
 * 
 *       If particles are copyed from the middle of the buffer, a particle will
 *       be copied from the end of the buffer to fill the hole.
 * 
 *       If the tile data position/limits in the main buffer will change,
 *       particles that stay in the tile but are now in invalid positions will
 *       be shifted.
 * 
 * @param part      Particle data
 * @param tmp       Temporary particle buffer (has new offsets)
 * @param sort      Sort data (new number of particles per tile, indices of
 *                  particles leaving the tile, etc.)
 * @param periodic  Correct for periodic boundaries
 */
void __launch_bounds__(opt_copy_out_block) copy_out( 
    part::particles_view part, 
    part::particles_view tmp,
    const part::particle_sort_view sort,
    const part::bnd_type local_bnd )
{
    const int2 ntiles = make_int2( part.local_ntiles.x, part.local_ntiles.y );
    const int2 lim = make_int2( part.tile_dims.x, part.tile_dims.y );

    /// @brief [shared] offsets in target buffer
    __shared__ int _dir_offset[9];

    /// @brief [shared] index of particle used to fill hole
    __shared__ int _c;

    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id = tile_idx.y * ntiles.x + tile_idx.x;

    int const old_offset      = part.tile_offset[ tile_id ];
    int * __restrict__ npt    = &sort.npt[ 9*tile_id ];

    auto * __restrict__ ix  = &part.ix[ old_offset ];
    auto * __restrict__ x   = &part.x[ old_offset ];
    auto * __restrict__ u   = &part.u[ old_offset ];

    auto * __restrict__ idx = &sort.idx[ old_offset ];
    uint32_t const nidx     = sort.nidx[ tile_id ];

    int const new_offset = tmp.tile_offset[ tile_id ];
    int const new_np     = sort.new_np[ tile_id ];

    // The _dir_offset variable holds the offset for each of the 9 target
    // tiles so the tmp_* variables just point to the beggining of the buffers
    auto * __restrict__ tmp_ix = tmp.ix;
    auto * __restrict__ tmp_x  = tmp.x;
    auto * __restrict__ tmp_u  = tmp.u;

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
    if ( gpu::block::thread_rank() == 0 ) {
        _dir_offset[4] = new_offset + gpu::device::atomic_fetch_add( & tmp.tile_np[ tile_id ], nshift );
    }
    gpu::block::sync();

    // Find offsets on new buffer
    for( int i = gpu::block::thread_rank(); i < 9; i += gpu::block::num_threads() ) {
        
        if ( i != 4 ) {
            // Find target node
            int dx = i % 3 - 1;
            int dy = i / 3 - 1;

            int2 target = make_int2( tile_idx.x + dx, tile_idx.y + dy);
            int target_tid = part::tid_coords( target, ntiles, local_bnd );

            if ( target_tid >= 0 ) {
                // If valid neighbour tile reserve space on tmp. array
                _dir_offset[i] = tmp.tile_offset[ target_tid ] + 
                                gpu::device::atomic_fetch_add( &tmp.tile_np[ target_tid ], npt[ i ] );
            } else {
                // Otherwise mark offset as invalid
                _dir_offset[i] = -1;
            }
        } 
    }

    // First candidate to fill holes
    _c = n0;

    gpu::block::sync();

    // Copy particles moving away from tile and fill holes
    for( int i = gpu::block::thread_rank(); i < nidx; i += gpu::block::num_threads() ) {
        
        int k = idx[i];

        auto nix  = ix[k];
        auto nx   = x[k];
        auto nu   = u[k];
        
        int xcross = ( nix.x >= lim.x ) - ( nix.x < 0 );
        int ycross = ( nix.y >= lim.y ) - ( nix.y < 0 );

        const int dir = (ycross+1) * 3 + (xcross+1);

        // Check if particle crossed into a valid neighbor
        if ( _dir_offset[dir] >= 0 ) {        

            // _dir_offset[] includes the offset in the global tmp particle buffer
            int l = gpu::block::atomic_fetch_add( & _dir_offset[dir], 1 );

            // Correct positions - nix is ok for new tile
            nix.x -= xcross * lim.x;
            nix.y -= ycross * lim.y;

            tmp_ix[ l ] = nix;
            tmp_x[ l ] = nx;
            tmp_u[ l ] = nu;
        }

        // Fill hole if needed
        if ( k < n0 ) {
            int c, invalid;

            do {
                c = gpu::block::atomic_fetch_add( &_c, 1 );
                invalid = ( ix[c].x < 0 ) || ( ix[c].x >= lim.x ) || 
                          ( ix[c].y < 0 ) || ( ix[c].y >= lim.y );
            } while (invalid);

            ix[ k ] = ix[ c ];
            x [ k ] = x [ c ];
            u [ k ] = u [ c ];
        }
    }

    gpu::block::sync();

    // At this point all particles up to n0 are correct

    // Copy particles that need to be shifted
    // We've reserved space for nshift particles earlier
    const int new_idx = _dir_offset[4];

    if ( new_offset >= old_offset ) {
        // Copy from beggining of buffer
        for( int i = gpu::block::thread_rank(); i < nshift; i += gpu::block::num_threads() ) {
            tmp_ix[ new_idx + i ] = ix[ i ];
            tmp_x[ new_idx + i ]  = x [ i ];
            tmp_u[ new_idx + i ]  = u [ i ];
        }

    } else {

        // Copy from end of buffer
        const int old_idx = n0 - nshift;
        for( int i = gpu::block::thread_rank(); i < nshift; i += gpu::block::num_threads() ) {
            tmp_ix[ new_idx + i ] = ix[ old_idx + i ];
            tmp_x[ new_idx + i ]  = x [ old_idx + i ];
            tmp_u[ new_idx + i ]  = u [ old_idx + i ];
        }
    }

    // Store current number of local particles
    // These are already in the correct position in global buffer
    if ( gpu::block::thread_rank() == 0 ) {
        part.tile_np[ tile_id ] = n0 - nshift;
    }

}

} // namespace kernel


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
 * @param periodic  Correct for periodic boundaries
 */
void copy_out( 
    part::particles_view data, part::particles_view tmp, const part::particle_sort_view sort,
    const part::bnd_type local_bnd )
{
    dim3 grid( tmp.local_ntiles.x, tmp.local_ntiles.y );
    auto block = opt_copy_out_block;

    kernel::copy_out <<< grid, block >>> ( data, tmp, sort, local_bnd );
}

namespace kernel {

/**
 * @brief Pack send messages
 * 
 * @note This kernel should be launched with grid( msg_tiles )
 *
 * @param tmp 
 * @param send_buffer 
 * @param send_msg_np 
 */
__global__
void pack_msg( part::particles_view tmp, uint8_t * __restrict__ send_buffer, int * __restrict__ send_msg_np ) {

    ///@brief Number of particles per message (shared)
    __shared__ int msg_np[9];
    
    for( int i = gpu::block::thread_rank(); i < 9; i+= gpu::block::num_threads() ) {
        msg_np[i] = send_msg_np[i];
    }

    ///@brief Tile index in message buffer
    int msg_idx = blockIdx.x;

    gpu::block::sync();

    const int2 ntiles = make_int2( tmp.local_ntiles.x, tmp.local_ntiles.y );

    ///@brief message direction
    int dir = part::edge_tile_dir(msg_idx, ntiles);

    ///@brief Tile position in main particle buffer
    const int source_tid = tmp.local_ntiles.x * tmp.local_ntiles.y + msg_idx;

    ///@brief Number of particles in "communication" tile to be sent
    const auto send_np     = tmp.tile_np[ source_tid ];

    // If any particles are sent from this tile
    if ( send_np > 0 ) {
        ///@brief Offset to first particle in "communication" tile
        const auto src_offset = tmp.tile_offset[ source_tid ];

        ///@brief offset for this direction message data (number of particles)
        int dir_offset = 0;
        
        // This will always be very small (dir < 9), no point in parallelizing
        for( int i = 0; i < dir; i++ ) dir_offset += msg_np[ i ];

        ///@brief send message buffer for this direction
        uint8_t * msg_buffer = &send_buffer[ dir_offset * part::particles::particle_size() ];

        // Compute number of particles used by all previous tiles in the same direction
        // Might be possible to use a block::exscan
        size_t msg_buffer_off = 0;
        for( int i = part::edge_tile_start(dir,ntiles); i < msg_idx; i++ ) {
            msg_buffer_off += tmp.tile_np[ tmp.local_ntiles.x * tmp.local_ntiles.y + i];
        }

        // Compute offsets inside destination message
        const part::particles::packed_offsets dst( msg_buffer_off, msg_np[ dir ] );

        // Due to alignment issues copying as int2/float2 may fail
        gpu::block::memcpy( reinterpret_cast<int *>( &msg_buffer[ dst.ix ] ),   
                            reinterpret_cast<int *>( &tmp.ix[ src_offset ]), 2*send_np);
        gpu::block::memcpy( reinterpret_cast<float *>( &msg_buffer[ dst.x  ] ), 
                            reinterpret_cast<float *>( &tmp.x[ src_offset ]), 2*send_np );
        gpu::block::memcpy( reinterpret_cast<float *>( &msg_buffer[ dst.u  ] ), 
                            reinterpret_cast<float *>( &tmp.u[ src_offset ]), 3*send_np );
    }

}

} // namespace kernel

/**
 * @brief Pack particles moving out of the node into a message buffer and start send
 * 
 * @param tmp       Temporary buffer holding particles moving away from tiles
 * @param sort      Temporary sort index
 * @param send      Send message object
 */
void part::particles::isend_msg( particles &tmp, particle_sort &sort, particle_message &send ) {

    /// @brief Total number of particles being sent
    uint32_t send_np = 0;

    // Copy msg_np from device memory
    gpu::device::memcpy_tohost( sort.send.h_msg_np, sort.send.d_msg_np, 9 );

    // Get offsets and check send buffer size
    for( int i = 0; i < 9; i++ ) {
        if (i != 4) {
            send.size[i] = sort.send.h_msg_np[i] * particle_size();
            send_np += sort.send.h_msg_np[i];
        } else {
            send.size[i] = 0;
        }
    }

    if ( send_np > 0 ) {
        // Check if msg buffer is large enough
        send.check_buffer( send_np * particle_size() );

        // Pack data
        dim3 grid( part::msg_tiles( tmp.local_ntiles ) );
        dim3 block( 256 );
        kernel::pack_msg <<< grid, block >>> ( tmp.view(), send.buffer, sort.send.d_msg_np );
        gpu::device::sync();
    }

    // Start sending messages
    send.isend();
}



namespace kernel {

__global__
/**
 * @brief CUDA kernel for copying incoming particles to main buffer.
 * 
 * @param data      Main particle data
 * @param tmp       Temporary particle data
 */
void __launch_bounds__(opt_copy_in_block) copy_in(
    part::particles_view data,
    part::particles_view tmp
) {
    const int2 ntiles = make_int2( data.local_ntiles.x, data.local_ntiles.y );

    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id = tile_idx.y * ntiles.x + tile_idx.x;

    const int old_offset       =  data.tile_offset[ tile_id ];
    const int old_np           =  data.tile_np[ tile_id ];

    const int new_offset       =  tmp.tile_offset[ tile_id ];
    const int tmp_np           =  tmp.tile_np[ tile_id ];

    // Notice that we are already working with the new offset
    auto * __restrict__ ix  = &data.ix[ new_offset ];
    auto * __restrict__ x   = &data.x [ new_offset ];
    auto * __restrict__ u   = &data.u [ new_offset ];

    auto * __restrict__ tmp_ix = &tmp.ix[ new_offset ];
    auto * __restrict__ tmp_x  = &tmp.x [ new_offset ];
    auto * __restrict__ tmp_u  = &tmp.u [ new_offset ];

    if ( new_offset >= old_offset ) {

        // Add particles to the end of the buffer
        for( int i = gpu::block::thread_rank(); i < tmp_np; i += gpu::block::num_threads() ) {
            ix[ old_np + i ] = tmp_ix[ i ];
            x [ old_np + i ] = tmp_x[ i ];
            u [ old_np + i ] = tmp_u[ i ];
        }

    } else {

        // Add particles to the beggining of buffer
        int np0 = old_offset - new_offset;
        if ( np0 > tmp_np ) np0 = tmp_np;
        
        for( int i = gpu::block::thread_rank(); i < np0; i += gpu::block::num_threads() ) {
            ix[ i ] = tmp_ix[ i ];
            x [ i ] = tmp_x [ i ];
            u [ i ] = tmp_u [ i ];
        }

        // If any particles left, add particles to the end of the buffer
        for( int i = np0 + gpu::block::thread_rank(); i < tmp_np; i += gpu::block::num_threads() ) {
            ix[ old_np + i ] = tmp_ix[ i ];
            x [ old_np + i ] = tmp_x [ i ];
            u [ old_np + i ] = tmp_u [ i ];
        }

    }

    gpu::block::sync();

    // Store the new offset and number of particles
    if ( gpu::block::thread_rank() == 0 ) {
        data.tile_np[ tile_id ]     = old_np + tmp_np;
        data.tile_offset[ tile_id ] = new_offset;
    }

}

} // namespace kernel


/**
 * @brief Copy incoming particles to main buffer. Buffer will be fully sorted after
 *        this step
 * 
 * @param data      Main particle data
 * @param tmp       Temporary particle data
 */
void copy_in( part::particles_view data, part::particles_view tmp )
{
    dim3 grid( data.local_ntiles.x, data.local_ntiles.y );
    auto block = opt_copy_in_block;

    kernel::copy_in <<< grid, block >>> ( data, tmp );
}


namespace kernel {

__global__
/**
* @brief Kernel for unpacking received message data 
* 
* @note Must be called with a grid size equal to the number of message tiles
* 
* @param part              Particle data
* @param recv_buffer       Message receive buffer (all messages)
* @param recv_msg_np,      Number of particles per message
* @param recv_msg_tile_np  Number of particles per received tile
*/
void unpack_msg( part::particles_view data, 
    uint8_t * __restrict__ recv_buffer, 
    int * __restrict__ recv_msg_np,
    int * __restrict__ recv_msg_tile_np ) {

    ///@brief Number of message tiles
    const int msg_tiles = gridDim.x;

    ///@brief tile id in message array
    int msg_idx = blockIdx.x;

    const int2 ntiles = make_int2( data.local_ntiles.x, data.local_ntiles.y );
    
    // Copy recv_msg_np and recv_msg_tile_np to shared memory using coalescent access
    auto * msg_tile_np = gpu::block::shared_mem<int>();
    
    for( int i = gpu::block::thread_rank(); i < msg_tiles; i += gpu::block::num_threads() ) {
        msg_tile_np[i] = recv_msg_tile_np[i];
    }

    __shared__ int msg_np[9];
    for( int i = gpu::block::thread_rank(); i < 9; i += gpu::block::num_threads() ) {
        msg_np[i] = recv_msg_np[i];
    }

    gpu::block::sync();

    ///@brief number of particles received on this tile 
    int recv_np = msg_tile_np[msg_idx];

    // If any particles received in this tile
    if ( recv_np > 0 ) {

        ///@brief message direction
        int dir = part::edge_tile_dir(msg_idx, ntiles);

        ///@brief target tile on particle buffer
        int target_tid = part::local_edge_tid(dir, msg_idx - part::edge_tile_start( dir, ntiles ), ntiles);

        // destination buffers (in main particle buffer)
        __shared__ int tgt_offset;
        if ( gpu::block::thread_rank() == 0 ) {
            tgt_offset = data.tile_offset[ target_tid ] + 
                gpu::device::atomic_fetch_add( &data.tile_np[ target_tid ], recv_np );
        }
        gpu::block::sync();

        ///@brief offset for this direction message data (number of particles)
        int dir_offset = 0;
        
        // This will always be very small (dir < 9), no point in parallelizing
        for( int i = 0; i < dir; i++ ) dir_offset += msg_np[ i ];

        ///@brief receive message buffer for this direction
        uint8_t * msg_buffer = &recv_buffer[ dir_offset * part::particles::particle_size() ];

        // Compute number of particles used by all previous tiles in the same direction
        // Might be possible to use a block::exscan
        size_t msg_buffer_off = 0;
        for( int i = part::edge_tile_start(dir,ntiles); i < msg_idx; i++ ) {
            msg_buffer_off += msg_tile_np[i];
        }

        // Compute offsets inside source message
        const part::particles::packed_offsets src( msg_buffer_off, msg_np[ dir ] );

        // Note that due to alignment issues copying as int2/float2/float3 may fail
        gpu::block::memcpy( reinterpret_cast<int *>( &data.ix[ tgt_offset ]),
                            reinterpret_cast<int *>( &msg_buffer[ src.ix ] ),   
                            2*recv_np );
        gpu::block::memcpy( reinterpret_cast<float *>( &data.x[ tgt_offset ]),
                            reinterpret_cast<float *>( &msg_buffer[ src.x  ]), 
                            2*recv_np );
        gpu::block::memcpy( reinterpret_cast<float *>( &data.u[ tgt_offset ]),
                            reinterpret_cast<float *>( &msg_buffer[ src.u  ]), 
                             3*recv_np );

    }

}

} // namespace kernel

/**
 * @brief Unpack received particle data into main particle data buffer
 * 
 * @param sort      Temporary sort index
 * @param recv      Receive message object
 */
void part::particles::unpack_msg( particle_sort &sort, particle_message &recv ) {
    if ( recv.buffer != nullptr ) {
        // Unpack all data - multiple message tiles may write to the same local tile
        dim3 grid( part::msg_tiles( local_ntiles ) );
        dim3 block( 256 );

        size_t shm_size = part::msg_tiles( local_ntiles ) * sizeof(int);
        gpu::block::set_shmem_size( kernel::unpack_msg, shm_size );
        kernel::unpack_msg <<< grid, block, shm_size >>> ( 
            view(),                      // Particle data
            recv.buffer,                // Message receive buffer (all messages)
            sort.recv.d_msg_np,         // Number of received particles per message
            sort.recv.buffer            // Number of received particles per tile
        );
    }
}


namespace kernel {

__global__
/**
 * @brief CUDA kernel for copying all particles to correct tiles in another buffer
 *
 * @param part      Particle data
 * @param tmp       Temporary particle buffer (has new offsets)
 * @param sort      Sort data (indices of particles leaving the tile, etc.)
 * @param periodic  Correct for periodic boundaries
 */
void copy_sorted( 
    part::particles_view data,
    part::particles_view tmp,
    const part::particle_sort_view sort,
    const int2 periodic
) {
    const int2 ntiles = make_int2( data.local_ntiles.x, data.local_ntiles.y );
    const int2 lim = make_int2( data.tile_dims.x, data.tile_dims.y );

    /// @brief [shared] offsets in target buffer
    __shared__ int _dir_offset[9];

    /// @brief [shared] index of particle used to fill hole
    __shared__ int _c;

    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id = tile_idx.y * data.local_ntiles.x + tile_idx.x;


    int const old_offset      = data.tile_offset[ tile_id ];
    int * __restrict__ npt    = &sort.npt[ 9*tile_id ];

    auto * __restrict__ ix  = &data.ix[ old_offset ];
    auto * __restrict__ x   = &data.x[ old_offset ];
    auto * __restrict__ u   = &data.u[ old_offset ];

    int * __restrict__ idx    = &sort.idx[ old_offset ];
    uint32_t const nidx       = sort.nidx[ tile_id ];

    // The _dir_offset variables hold the offset for each of the 9 target
    // tiles so the tmp_* variables just point to the beggining of the buffers
    auto * __restrict__ tmp_ix = tmp.ix;
    auto * __restrict__ tmp_x  = tmp.x;
    auto * __restrict__ tmp_u  = tmp.u;

    // Find offsets on new buffer
    for( int i = gpu::block::thread_rank(); i < 9; i += gpu::block::num_threads() ) {
        
        // Find target node
        int target_tx = tile_idx.x + i % 3 - 1;
        int target_ty = tile_idx.y + i / 3 - 1;

        bool valid = true;

        // Correct for periodic boundaries
        if ( periodic.x ) {
            if ( target_tx < 0 )         target_tx += ntiles.x; 
            if ( target_tx >= ntiles.x ) target_tx -= ntiles.x;
        } else {
            valid &= ( target_tx >= 0 ) && ( target_tx < ntiles.x ); 
        }

        if ( periodic.y ) {
            if ( target_ty < 0 )         target_ty += ntiles.y;
            if ( target_ty >= ntiles.y ) target_ty -= ntiles.y;
        } else {
            valid &= ( target_ty >= 0 ) && ( target_ty < ntiles.y ); 
        }

        if ( valid ) {
            // If valid neighbour tile reserve space on tmp. array
            int target_tid = target_ty * ntiles.x + target_tx;

            _dir_offset[i] = tmp.tile_offset[ target_tid ] + 
                gpu::device::atomic_fetch_add( & tmp.tile_np[ target_tid ], npt[ i ] );
        
        } else {
            // Otherwise mark offset as invalid
            _dir_offset[i] = -1;
        }
    }

    // Particles remaining on tile
    const int n0 = npt[4];

    // First candidate to fill holes
    _c = n0;

    gpu::block::sync();

    // Copy particles moving away from tile and fill holes
    for( int i = gpu::block::thread_rank(); i < nidx; i += gpu::block::num_threads() ) {
        
        int k = idx[i];

        auto nix = ix[k];
        auto nx  = x[k];
        auto nu  = u[k];
        
        int xcross = ( nix.x >= lim.x ) - ( nix.x < 0 );
        int ycross = ( nix.y >= lim.y ) - ( nix.y < 0 );

        const int dir = (ycross+1) * 3 + (xcross+1);

        // Check if particle crossed into a valid neighbor
        if ( _dir_offset[dir] >= 0 ) {        

            // _dir_offset[] includes the offset in the global tmp particle buffer
            int l = gpu::block::atomic_fetch_add( & _dir_offset[dir], 1 );

            nix.x -= xcross * lim.x;
            nix.y -= ycross * lim.y;

            tmp_ix[ l ] = nix;
            tmp_x[ l ]  = nx;
            tmp_u[ l ]  = nu;
        }

        // Fill hole if needed
        if ( k < n0 ) {
            int c, invalid;

            do {
                c = gpu::block::atomic_fetch_add( &_c, 1 );
                invalid = ( ix[c].x < 0 ) || ( ix[c].x >= lim.x ) || 
                          ( ix[c].y < 0 ) || ( ix[c].y >= lim.y );
            } while (invalid);

            ix[ k ] = ix[ c ];
            x [ k ] = x [ c ];
            u [ k ] = u [ c ];
        }
    }

    gpu::block::sync();

    // Copy particles staying in tile
    const int start = _dir_offset[4];

    for( int i = gpu::block::thread_rank(); i < n0; i += gpu::block::num_threads() ) {
        tmp_ix[ start + i ] = ix[i];
        tmp_x [ start + i ] = x[i];
        tmp_u [ start + i ] = u[i];
    }
}

} // namespace kernel

/**
 * @brief Copies copy all particles to correct tiles in another buffer
 * 
 * @note Requires that new buffer (`tmp`) already has the correct offset
 *       values, and number of particles set to 0.
 * 
 * @param data      Particle data
 * @param tmp       Temporary particle buffer (has new offsets)
 * @param sort      Sort data (indices of particles leaving the tile, etc.)
 * @param periodic  Correct for periodic boundaries
 */
void copy_sorted( 
    part::particles_view data, part::particles_view tmp, const part::particle_sort_view sort,
    const part::bnd_type local_bnd )
{
    mpi::fatal( "copy_sorted() is not implemented yet, aborting...\n" );

    // dim3 grid( part.ntiles.x, part.ntiles.y );
    // dim3 block( 1024 );
    // kernel::copy_sorted <<< grid, block >>> ( part, tmp, sort, ... );
}


namespace kernel {

/**
 * @brief CUDA kernel for getting total number of particles
 * 
 * @note block size must be a multiple of gpu::warp_size so that all threads inside the warp are active
 *
 * @param tile_np   Number of particles per tile
 * @param ntiles    Total number of tiles
 * @param total     Output
 */
__global__
void total_np( 
    int const * const __restrict__ tile_np,
    uint32_t const ntiles, 
    uint32_t * const __restrict__ total
) {
    __shared__ uint32_t block_np;
    if ( threadIdx.x == 0 ) block_np = 0;
    gpu::block::sync();

    // Out of range threads contribute 0
    uint32_t thread_np = 0;
    for( uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < ntiles; i += gridDim.x * blockDim.x )
        thread_np += tile_np[ i ];

    thread_np = gpu::warp::reduce_add( thread_np );  
    if ( gpu::warp::thread_rank() == 0 ) 
        gpu::block::atomic_fetch_add( &block_np, thread_np );
    gpu::block::sync();

    if ( threadIdx.x == 0 ) 
        gpu::device::atomic_fetch_add( total, block_np );
}

/**
 * @brief CUDA kernel for getting maximum number of particles in a single tile
 * 
 * @note block size must be a multiple of gpu::warp_size so that all threads inside the warp are active
 *
 * @param tile_np   Number of particles per tile
 * @param ntiles    Total number of tiles
 * @param total     Output
 */
__global__
void tile_np_max( 
    int const * const __restrict__ tile_np,
    uint32_t const ntiles, 
    uint32_t * const __restrict__ total
) {
    __shared__ uint32_t block_max;
    if ( threadIdx.x == 0 ) block_max = 0;
    gpu::block::sync();

    uint32_t thread_np = 0;
    for( uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < ntiles; i += gridDim.x * blockDim.x ) {
        uint32_t np = tile_np[i];
        if ( np > thread_np ) thread_np = np;
    }

    thread_np = gpu::warp::reduce_max( thread_np );  
    if ( gpu::warp::thread_rank() == 0 ) 
        gpu::block::atomic_fetch_max( &block_max, thread_np );
    gpu::block::sync();

    if ( threadIdx.x == 0 ) 
        gpu::device::atomic_fetch_max( total, block_max );
}

/**
 * @brief CUDA kernel for getting minimum number of particles in a single tile
 *
 * @note block size must be a multiple of gpu::warp_size so that all threads inside the warp are active
 * 
 * @param tile_np   Number of particles per tile
 * @param ntiles    Total number of tiles
 * @param total     Output
 */
__global__
void tile_np_min( 
    int const * const __restrict__ tile_np,
    uint32_t const ntiles, 
    uint32_t * const __restrict__ total
) {
    __shared__ uint32_t block_min;
    if ( threadIdx.x == 0 ) block_min = UINT32_MAX;
    gpu::block::sync();

    uint32_t thread_np = UINT32_MAX;
    for( uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < ntiles; i += gridDim.x * blockDim.x ) {
        uint32_t np = tile_np[i];
        if ( np < thread_np ) thread_np = np;
    }

    thread_np = gpu::warp::reduce_min( thread_np );  
    if ( gpu::warp::thread_rank() == 0 )
        gpu::block::atomic_fetch_min( &block_min, thread_np );
    gpu::block::sync();

    if ( threadIdx.x == 0 ) 
        gpu::device::atomic_fetch_min( total, block_min );
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
uint32_t part::particles::local_np() {

    dev_tmp_uint32.set(0);

    auto size = local_ntiles.x*local_ntiles.y;
    auto block = ( size < 1024 ) ? roundup< gpu::warp_size > ( size ) : 1024 ;
    auto grid = (size + block -1 ) / block;
    kernel::total_np <<< grid, block >>> ( tile_np, size, dev_tmp_uint32.ptr() );

    return dev_tmp_uint32.get();
}

/**
 * @brief Gets maximum number of particles in a single tile
 * 
 * @return uint32_t 
 */
uint32_t part::particles::tile_np_max() {

    dev_tmp_uint32.set( 0 );

    auto size = local_ntiles.x*local_ntiles.y;
    auto block = ( size < 1024 ) ? roundup< gpu::warp_size > ( size ) : 1024 ;
    auto grid = (size + block -1 ) / block;
    kernel::tile_np_max <<< grid, block >>> ( tile_np, size, dev_tmp_uint32.ptr() );

    return dev_tmp_uint32.get();
}

/**
 * @brief Gets minimum number of particles in a single tile
 * 
 * @return uint32_t 
 */
uint32_t part::particles::tile_np_min() {

    dev_tmp_uint32.set( std::numeric_limits<uint32_t>::max() );

    auto size = local_ntiles.x*local_ntiles.y;
    auto block = ( size < 1024 ) ? roundup< gpu::warp_size > ( size ) : 1024 ;
    auto grid = (size + block -1 ) / block;
    kernel::tile_np_min <<< grid, block >>> ( tile_np, size, dev_tmp_uint32.ptr() );

    return dev_tmp_uint32.get();
}



namespace kernel {

/**
 * @brief Gather particle data
 * 
 * @tparam quant    Quantiy to gather
 * @param src       Particle data
 * @param d_data    Output data
 */
template < part::quantity quant >
__global__
void gather( 
    part::particles_view src,
    float * const __restrict__ d_data )
{
    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int   tile_id  = tile_idx.y * src.local_ntiles.x + tile_idx.x;
    
    // Offset and number of particles on particle buffer
    const auto  offset = src.tile_offset[ tile_id ];
    const auto  np     = src.tile_np[ tile_id ];

    // Global spatial offsets of local tile
    const int offx = (src.local_tile_start.x + tile_idx.x) * src.tile_dims.x;
    const int offy = (src.local_tile_start.y + tile_idx.y) * src.tile_dims.y;

    auto const * __restrict__ const ix = & src.ix[ offset ];
    auto const * __restrict__ const x  = & src.x [ offset ];
    auto const * __restrict__ const u  = & src.u [ offset ];

    for( int idx = gpu::block::thread_rank(); idx < np; idx += gpu::block::num_threads() ) {
        float val;
        if constexpr( quant == part::quantity::x  ) val = ( offx + ix[idx].x ) + (0.5f + x[idx].x);
        if constexpr( quant == part::quantity::y  ) val = ( offy + ix[idx].y ) + (0.5f + x[idx].y);
        if constexpr( quant == part::quantity::ux ) val = u[idx].x;
        if constexpr( quant == part::quantity::uy ) val = u[idx].y;
        if constexpr( quant == part::quantity::uz ) val = u[idx].z;
        d_data[ offset + idx ] = val;
    }
}

} // namespace kernel

/**
 * @brief Gather data from a specific particle quantity in a device buffer
 * 
 * @param quant         Quantity to gather
 * @param d_data        Output data buffer, assumed to have size >= np
 */
void part::particles::gather( part::quantity quant, float * const d_data )
{
    const dim3 grid( local_ntiles.x, local_ntiles.y );
    const dim3 block( 1024 );

    // Gather data on device
    switch (quant) {
    case part::quantity ::x : 
        kernel::gather<part::quantity ::x> <<<grid,block>>> ( view(), d_data );
        break;
    case part::quantity ::y:
        kernel::gather<part::quantity ::y> <<<grid,block>>> ( view(), d_data );
        break;
    case part::quantity ::ux:
        kernel::gather<part::quantity ::ux> <<<grid,block>>> ( view(), d_data );
        break;
    case part::quantity ::uy:
        kernel::gather<part::quantity ::uy> <<<grid,block>>> ( view(), d_data );
        break;
    case part::quantity ::uz:
        kernel::gather<part::quantity ::uz> <<<grid,block>>> ( view(), d_data );
        break;
    }
}

namespace kernel {

/**
 * @brief Gather particle data, scaling values
 * 
 * @note Data (val) will be returned as `scale.x * val + scale.y`
 *
 * @tparam quant    Quantiy to gather
 * @param src       Particle data
 * @param scale     Scale factor
 * @param d_data    Output data
 */
template < part::quantity quant >
__global__
void gather( 
    part::particles_view src,
    const float2 scale,
    float * const __restrict__ d_data
) {
    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int   tile_id  = tile_idx.y * src.local_ntiles.x + tile_idx.x;

    // Spatial offsets of local tile
    const int offx = (src.local_tile_start.x + tile_idx.x) * src.tile_dims.x;
    const int offy = (src.local_tile_start.y + tile_idx.y) * src.tile_dims.y;

    // Offset and number of particles on particle buffer
    const auto  offset = src.tile_offset[ tile_id ];
    const auto  np     = src.tile_np[ tile_id ];

    auto const * __restrict__ const ix = & src.ix[ offset ];
    auto const * __restrict__ const x  = & src.x [ offset ];
    auto const * __restrict__ const u  = & src.u [ offset ];

    for( int idx = gpu::block::thread_rank(); idx < np; idx += gpu::block::num_threads() ) {
        float val;
        if constexpr( quant == part::quantity::x  ) val = (offx + ix[idx].x) + (0.5f + x[idx].x);
        if constexpr( quant == part::quantity::y  ) val = (offy + ix[idx].y) + (0.5f + x[idx].y);
        if constexpr( quant == part::quantity::ux ) val = u[idx].x;
        if constexpr( quant == part::quantity::uy ) val = u[idx].y;
        if constexpr( quant == part::quantity::uz ) val = u[idx].z;
        d_data[ offset + idx ] = fma( scale.x, val, scale.y );
    }
}

} // namespace kernel


/**
 * @brief Gather data from a specific particle quantity in a device buffer, scaling values
 * 
 * @warning This expects the particle buffer to be compact, it will fail if 
 *          called with a non-compact buffer.
 *
 * @param quant     Quantity to gather
 * @param d_data    Output data buffer, assumed to have size >= np
 * @param scale     Scale factor for data
 */
void part::particles::gather( part::quantity quant, const float2 scale, float * const __restrict__ d_data )
{
    const dim3 grid( local_ntiles.x, local_ntiles.y );
    const dim3 block( 1024 );

    // Gather data on device
    switch (quant) {
    case part::quantity::x : 
        kernel::gather<part::quantity::x>  <<<grid,block>>> ( view(), scale, d_data );
        break; 
    case part::quantity::y: 
        kernel::gather<part::quantity::y> <<<grid,block>>> ( view(), scale, d_data );
        break; 
    case part::quantity::ux: 
        kernel::gather<part::quantity::ux> <<<grid,block>>> ( view(), scale, d_data );
        break; 
    case part::quantity::uy: 
        kernel::gather<part::quantity::uy> <<<grid,block>>> ( view(), scale, d_data );
        break; 
    case part::quantity::uz: 
        kernel::gather<part::quantity::uz> <<<grid,block>>> ( view(), scale, d_data );
        break;
    }
}


/**
 * @brief Save particle data to disk
 * 
 * @warning This expects the particle buffer to be compact, it will fail if 
 *          called with a non-compact buffer.
 *
 * @param quants    Quantities to save
 * @param metadata  Particle metadata (name, labels, units, etc.). Information is used to
 *                  set file name
 * @param iter      Iteration metadata
 * @param path      Path where to save the file
 */
void part::particles::save( const part::quantity quants[], zdf::part_info &metadata, zdf::iteration &iter, std::string path ) {

    // Get total number of particles to save
    uint64_t local = local_np();
    uint64_t global = 0;

    parallel.allreduce( &local, &global, 1, mpi::sum );

    // Update metadata entry
    metadata.np = global;

    if ( global > 0 ) {
        // Create a communicator including only the nodes with local particles
        int color = ( local > 0 ) ? 1 : MPI_UNDEFINED;
        MPI_Comm comm;
        MPI_Comm_split( parallel.get_comm(), color, 0, & comm );

        // Only nodes with particles are involved in this section
        if ( local > 0 ) {

            // Get rank in new communicator
            int rank;
            MPI_Comm_rank( comm, & rank );

            // Open file
            zdf::par_file part_file;
            zdf::open_part_file( part_file, metadata, iter, path+"/"+metadata.name, comm );

            // create the datasets
            std::vector< zdf::dataset > dsets( metadata.nquants );

            for( uint32_t i = 0; i < metadata.nquants; i++ ) {
                dsets[i].name      = metadata.quants[i];
                dsets[i].data_type = zdf::data_type<float>();
                dsets[i].ndims     = 1;
                dsets[i].data      = nullptr;
                dsets[i].count[0]  = global; 

                if ( !zdf::start_cdset( part_file, dsets[i] ) ) {
                    mpi::fatal( "particles::save() - Unable to create chunked dataset " + 
                        std::string(dsets[i].name) );
                }
            }

            // Allocate buffer for gathering particle data
            float *d_data = gpu::device::malloc<float>( local );
            float *h_data = gpu::host::malloc<float>( local );

            // Get offsets - this avoids recalculating offsets for each quantity
            uint64_t file_off;
            MPI_Exscan( &local, &file_off, 1, MPI_UINT64_T, MPI_SUM, comm );
            if ( rank == 0 ) file_off = 0;

            // Local data chunk
            zdf::chunk chunk;
            chunk.count[0] = local;
            chunk.start[0] = file_off;
            chunk.stride[0] = 1;
            chunk.data = h_data;

            // Write the data
            for ( uint32_t i = 0; i < metadata.nquants; i ++) {
                gather( quants[i], d_data );
                gpu::device::memcpy_tohost( h_data, d_data, local );
                zdf::write_cdset( part_file, dsets[i], chunk, file_off );
            }

            // Free temporary data
            gpu::device::free( d_data );
            gpu::host::free( h_data );

            // close the datasets
            for( uint32_t i = 0; i < metadata.nquants; i++ ) 
                zdf::end_cdset( part_file, dsets[i] );

            // Close the file
            zdf::close_file( part_file );
    
            MPI_Comm_free(&comm);
        }

    } else {
        // No particles - root node creates an empty file
        if ( parallel.root() ) {
            zdf::file part_file;
            zdf::open_part_file( part_file, metadata, iter, path+"/"+metadata.name );

            for ( uint32_t i = 0; i < metadata.nquants; i ++) {
                zdf::add_quant_part_file( part_file, metadata.quants[i],  nullptr, 0 );
            }

            zdf::close_file( part_file );
        }
    }
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
void part::particles::tile_sort( particles & tmp, particle_sort & sort, const int * __restrict__ extra ) {

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
    auto total_np = update_tile_info ( tmp.view(), sort, dev_tmp_uint32, extra );

    if ( total_np > max_part ) { 
        std::ostringstream msg;
        msg << "particles::tile_sort() - particle buffer requires growing,"
                  << "max_part: " << max_part << ", total_np: " << total_np
                  << ", not implemented yet.";
        mpi::fatal(msg.str());
    }

    // Copy outgoing particles (and particles needing shifting) to staging area
    copy_out ( view(), tmp.view(), sort.view(), local_bnd );

    // Pack particle data and start sending
    isend_msg( tmp, sort, send );

    // Copy local particles from staging area into final positions in partile buffer
    copy_in ( view(), tmp.view() );

    // Wait for receive messages to complete
    recv.wait();

    // Wait for messages to be received and unpack data
    unpack_msg( sort, recv );

    // Wait for sends to complete
    send.wait();

    // For debug only, remove from production code
    // parallel.barrier();
    // validate( "after tile_sort" );
}


namespace kernel {

__global__
/**
 * @brief CUDA kernel for shifting particle cells by the required amount
 * 
 * @param part      Particle data
 * @param shift     Cell shift
 */
void cell_shift( part::particles_view part, int2 const shift ) {

    const int2 ntiles = make_int2( part.local_ntiles.x, part.local_ntiles.y );
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id = tile_idx.y * ntiles.x + tile_idx.x;

    const auto tile_off = part.tile_offset[ tile_id ];
    const auto tile_np  = part.tile_np[ tile_id ];

    int2   * const __restrict__ ix = &part.ix[ tile_off ];

    for( int i = gpu::block::thread_rank(); i < tile_np; i += gpu::block::num_threads() ) {
        int2 cell = ix[i];
        cell.x += shift.x;
        cell.y += shift.y;
        ix[i] = cell;
    }
}

} // namespace kernel

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
void part::particles::cell_shift( int2 const shift ) {

    dim3 grid( local_ntiles.x, local_ntiles.y );
    dim3 block( 1024 );

    kernel::cell_shift <<< grid, block >>> ( view(), shift );
}



/**
 * @brief Maximum allowed value for u
 * 
 */
//#define __ULIM std::numeric_limits<float>::max()
#define __ULIM __FLT_MAX__

namespace kernel {

__global__
void validate( part::particles_view data, const int over, const int rank, uint32_t * out ) {
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int2 ntiles = make_int2( data.local_ntiles.x, data.local_ntiles.y );
    const int tile_id = tile_idx.y * ntiles.x + tile_idx.x;

    int const offset = data.tile_offset[ tile_id ];
    int const np     = data.tile_np[ tile_id ];
    auto * const __restrict__ ix = &data.ix[ offset ];
    auto * const __restrict__ x  = &data.x[ offset ];
    auto * const __restrict__ u  = &data.u[ offset ];

    __shared__ int _err; _err = 0;
    gpu::block::sync();

    int2 const lb = make_int2( -over, -over );
    int2 const ub = make_int2( data.tile_dims.x + over, data.tile_dims.y + over ); 

    for( int i = gpu::block::thread_rank(); i < np; i += gpu::block::num_threads() ) {
        int err = 0;

        if ( (ix[i].x < lb.x) || (ix[i].x >= ub.x )) {
            printf("[%d] [%d,%d] Invalid ix[%d].x position (%d), range = [%d,%d[\n", rank, tile_idx.x, tile_idx.y, i, ix[i].x, lb.x, ub.x );
            err = 1;
        }
        if ( x[i].x < -0.5f || x[i].x >= 0.5f ) {
            printf("[%d] [%d,%d] Invalid x[%d].x position (%f), range = [-0.5,0.5[\n", rank, tile_idx.x, tile_idx.y, i, x[i].x );
            err = 1;
        }

        if ( (ix[i].y < lb.y) || (ix[i].y >= ub.y )) {
            printf("[%d] [%d,%d] Invalid ix[%d].y position (%d), range = [%d,%d[\n", rank, tile_idx.x, tile_idx.y, i, ix[i].y, lb.y, ub.y );
            err = 1;
        }
        if ( x[i].y < -0.5f || x[i].y >= 0.5f ) {
            printf("[%d] [%d,%d] Invalid x[%d].y position (%f), range = [-0.5,0.5[\n", rank, tile_idx.x, tile_idx.y, i, x[i].y );
            err = 1;
        }

        if ( isnan(u[i].x) || isinf(u[i].x) || fabsf(u[i].x) >= __ULIM ) {
            printf("[%d] [%d,%d] Invalid u[%d].x gen. velocity (%f)\n", rank, tile_idx.x, tile_idx.y, i, u[i].x );
            err = 1;
        }

        if ( isnan(u[i].y) || isinf(u[i].y) || fabsf(u[i].y) >= __ULIM ) {
            printf("[%d] [%d,%d] Invalid u[%d].y gen. velocity (%f)\n", rank, tile_idx.x, tile_idx.y, i, u[i].y );
            err = 1;
        }

        if ( isnan(u[i].z) || isinf(u[i].z) || fabsf(u[i].z) >= __ULIM ) {
            printf("[%d] [%d,%d] Invalid u[%d].z gen. velocity (%f)\n", rank, tile_idx.x, tile_idx.y, i, u[i].z );
            err = 1;
        }

        if ( err ) _err = 1;

        if ( _err ) {
            *out = 1;
            break;
        }
    }
}

} // namespace kernel

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
 *      4. Invalid position data (out of [-0.5,0.5[)
 *      5. Invalid momenta (nan, inf or above __ULIM macro value)
 * 
 * If there are any errors found the routine will exit the code.
 * 
 * @param msg       Message to print in case error is found
 * @param over      Amount of extra cells indices beyond limit allowed. Used
 *                  when checking the buffer before tile_sort()
 */
void part::particles::validate( std::string msg, int const over ) {

    if ( msg.empty() ) {
        mpi::cout << "validating particle set...";
    } else {
        mpi::cout << "validating particle set (" << msg << ")...";
    }

    uint32_t err = 0;

    // Check offset / np buffer

    // For simplicity the test is done on the host
    int * h_np = gpu::host::malloc<int>( local_ntiles.x * local_ntiles.y );
    int * h_offset = gpu::host::malloc<int>( local_ntiles.x * local_ntiles.y );
    gpu::device::memcpy_tohost( h_np, tile_np, local_ntiles.x * local_ntiles.y );
    gpu::device::memcpy_tohost( h_offset, tile_offset, local_ntiles.x * local_ntiles.y );

    for( unsigned tile_id = 0; tile_id < local_ntiles.x * local_ntiles.y; ++tile_id ) {
        if ( h_np[tile_id] < 0 ) {
            mpi::cout << "tile[" << tile_id << "] - bad np (" << h_np[ tile_id ] << "), should be >= 0\n";
            err = 1;
        }

        if ( tile_id > 0 ) {
            auto prev = h_offset[ tile_id-1] + h_np[ tile_id-1];
            if ( prev != h_offset[ tile_id ] ) {
                mpi::cout << "tile[" << tile_id << "] - bad offset (" << h_offset[ tile_id ] << ")"
                          << ", does not match previous tile info, should be " << prev << '\n';
                err = 1;
            }
        } else {
            if ( h_offset[ tile_id ] != 0 ) {
                mpi::cout << "tile[" << tile_id << "] - bad offset (" << h_offset[ tile_id ] << "), should be 0\n";
                err = 1;
            }
        }   
    }
    gpu::host::free( h_np );
    gpu::host::free( h_offset );

    if ( err ) {
        mpi::fatal("\nInvalid tile information");
    }

    // Check particle data directly on device
    dev_tmp_uint32.set( 0 );

    dim3 grid( local_ntiles.x, local_ntiles.y );
    dim3 block( 32 );
    kernel::validate <<< grid, block >>> ( view(), over, parallel.get_rank(), dev_tmp_uint32.ptr() );
    err = dev_tmp_uint32.get();

    if ( err ) {
        mpi::fatal("Invalid particle(s) found" );
    } else {
        mpi::cout << " particle set ok.\n";
    }
}

#undef __ULIM
