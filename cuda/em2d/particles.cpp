#include "particles.h"

#include <iostream>
#include <string>
#include <cmath>

#include "timer.h"

#define opt_bnd_check_block 1024
#define opt_update_tile_info_block 64
#define opt_copy_out_block 32
#define opt_copy_in_block 1024

namespace kernel {

__global__
/**
 * @brief CUDA kernel for getting total number of particles
 * 
 * @param np        Number of particles per tile
 * @param ntiles    Total number of tiles
 * @param total     Output
 */
void np_total( 
    int const * const __restrict__ np,
    uint32_t const ntiles, 
    uint32_t * const __restrict__ total
) {
    __shared__ uint32_t block_np;
    block_np = 0;
    uint32_t thread_np = np[ blockIdx.x * blockDim.x + threadIdx.x ];
    block::sync();

    thread_np = warp::reduce_add( thread_np );  
    if ( warp::thread_rank() == 0 ) block::atomic_fetch_add( &block_np, thread_np );
    block::sync();

    if ( threadIdx.x == 0 ) device::atomic_fetch_add( total, block_np );
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
uint32_t Particles::np_total() {

    dev_tmp_uint32 = 0;

    auto size = ntiles.x*ntiles.y;
    auto block = ( size < 1024 ) ? size : 1024 ;
    auto grid = (size-1)/block + 1;
    kernel::np_total <<< grid, block >>> ( np, size, dev_tmp_uint32.ptr() );

    return dev_tmp_uint32.get();
}

namespace kernel {

__global__
/**
 * @brief CUDA kernel for getting maximum number of particles in a single tile
 * 
 * @param np        Number of particles per tile
 * @param ntiles    Total number of tiles
 * @param total     Output
 */
void np_max_tile( 
    int const * const __restrict__ np,
    uint32_t const ntiles, 
    uint32_t * const __restrict__ total
) {
    __shared__ uint32_t block_max;
    block_max = 0;
    uint32_t thread_np = np[ blockIdx.x * blockDim.x + threadIdx.x ];
    block::sync();

    thread_np = warp::reduce_max( thread_np );  
    if ( warp::thread_rank() == 0 ) block::atomic_fetch_max( &block_max, thread_np );
    block::sync();

    if ( threadIdx.x == 0 ) device::atomic_fetch_max( total, block_max );
}

}

/**
 * @brief Gets maximum number of particles in a single tile
 * 
 * @return uint32_t 
 */
uint32_t Particles::np_max_tile() {

    dev_tmp_uint32 = 0;

    auto size = ntiles.x*ntiles.y;
    auto block = ( size < 1024 ) ? size : 1024 ;
    auto grid = (size-1)/block + 1;
    kernel::np_max_tile <<< grid, block >>> ( np, size, dev_tmp_uint32.ptr() );

    return dev_tmp_uint32.get();
}


namespace kernel {

__global__
/**
 * @brief CUDA kernel for getting minimum number of particles in a single tile
 * 
 * @param np        Number of particles per tile
 * @param ntiles    Total number of tiles
 * @param total     Output
 */
void np_min_tile( 
    int const * const __restrict__ np,
    uint32_t const ntiles, 
    uint32_t * const __restrict__ total
) {
    __shared__ uint32_t block_min;
    block_min = UINT32_MAX;
    uint32_t thread_np = np[ blockIdx.x * blockDim.x + threadIdx.x ];
    block::sync();

    thread_np = warp::reduce_min( thread_np );  
    if ( warp::thread_rank() == 0 ) block::atomic_fetch_min( &block_min, thread_np );
    block::sync();

    if ( threadIdx.x == 0 ) device::atomic_fetch_min( total, block_min );
}

}

/**
 * @brief Gets minimum number of particles in a single tile
 * 
 * @return uint32_t 
 */
uint32_t Particles::np_min_tile() {

    dev_tmp_uint32 = std::numeric_limits<uint32_t>::max();

    auto size = ntiles.x*ntiles.y;
    auto block = ( size < 1024 ) ? size : 1024 ;
    auto grid = (size-1)/block + 1;
    kernel::np_min_tile <<< grid, block >>> ( np, size, dev_tmp_uint32.ptr() );

    return dev_tmp_uint32.get();
}

namespace kernel {

/**
 * @brief Gather particle data
 * 
 * @tparam quant    Quantiy to gather
 * @param part      Particle data
 * @param d_data    Output data
 */
template < part::quant quant >
__global__
void gather( 
    ParticleData part,
    float * const __restrict__ d_data
) {
    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int   tile_id  = tile_idx.y * part.ntiles.x + tile_idx.x;
    
    const auto  tile_off = part.offset[ tile_id ];
    const auto  tile_np  = part.np[ tile_id ];

    int2   * const __restrict__ ix       = & part.ix[ tile_off ];
    float2 const * __restrict__ const x  = & part.x[ tile_off ];
    float3 const * __restrict__ const u  = & part.u[ tile_off ];

    for( int idx = block_thread_rank(); idx < tile_np; idx += block_num_threads() ) {
        float val;
        if ( quant == part::x )  val = (tile_idx.x * part.nx.x + ix[idx].x) + (0.5f + x[idx].x);
        if ( quant == part::y )  val = (tile_idx.y * part.nx.y + ix[idx].y) + (0.5f + x[idx].y);
        if ( quant == part::ux ) val = u[idx].x;
        if ( quant == part::uy ) val = u[idx].y;
        if ( quant == part::uz ) val = u[idx].z;
        d_data[ tile_off + idx ] = val;
    }
}

}

/**
 * @brief Gather data from a specific particle quantity in a device buffer
 * 
 * @warning The output will use the same offset as the data buffer, so the
 *          particle buffer must be compact
 * 
 * @param quant         Quantity to gather
 * @param d_data        Output data buffer (assumed to have sufficient size)
 */
void Particles::gather( part::quant quant, float * const __restrict__ d_data )
{
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 1024 );
    
    // Gather data on device
    switch (quant) {
    case part::x : 
        kernel::gather<part::x> <<<grid,block>>>( *this, d_data );
        break;
    case part::y:
        kernel::gather<part::y> <<<grid,block>>>( *this, d_data );
        break;
    case part::ux:
        kernel::gather<part::ux> <<<grid,block>>>( *this, d_data );
        break;
    case part::uy:
        kernel::gather<part::uy> <<<grid,block>>>( *this, d_data );
        break;
    case part::uz:
        kernel::gather<part::uz> <<<grid,block>>>( *this, d_data );
        break;
    }
}

namespace kernel {

/**
 * @brief Gather particle data
 * 
 * @tparam quant    Quantiy to gather
 * @param part      Particle data
 * @param d_data    Output data
 */
template < part::quant quant >
__global__
void gather( 
    ParticleData part,
    float * const __restrict__ d_data,
    const float2 scale
) {
    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int   tile_id  = tile_idx.y * part.ntiles.x + tile_idx.x;
    
    const auto  tile_off = part.offset[ tile_id ];
    const auto  tile_np  = part.np[ tile_id ];

    int2   * const __restrict__ ix       = & part.ix[ tile_off ];
    float2 const * __restrict__ const x  = & part.x[ tile_off ];
    float3 const * __restrict__ const u  = & part.u[ tile_off ];

    for( int idx = block_thread_rank(); idx < tile_np; idx += block_num_threads() ) {
        float val;
        if ( quant == part::x )  val = (tile_idx.x * part.nx.x + ix[idx].x) + (0.5f + x[idx].x);
        if ( quant == part::y )  val = (tile_idx.y * part.nx.y + ix[idx].y) + (0.5f + x[idx].y);
        if ( quant == part::ux ) val = u[idx].x;
        if ( quant == part::uy ) val = u[idx].y;
        if ( quant == part::uz ) val = u[idx].z;
        d_data[ tile_off + idx ] = fma( scale.x, val, scale.y );
    }
}

}

/**
 * @brief Gather particle data, scaling values
 * 
 * @note Data (val) will be returned as `scale.x * val + scale.y`
 * 
 * @warning The output will use the same offset as the data buffer, so the
 *          particle buffer must be compact
 * 
 * @param quant         Quantity to gather
 * @param d_data        Output data buffer (assumed to have sufficient size)
 * @param scale     Scale factor for data
 */
void Particles::gather( part::quant quant, float * const __restrict__ d_data, const float2 scale )
{
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 1024 );
    
    // Gather data on device
    switch (quant) {
    case part::x : 
        kernel::gather<part::x> <<<grid,block>>>( *this, d_data, scale );
        break;
    case part::y:
        kernel::gather<part::y> <<<grid,block>>>( *this, d_data, scale );
        break;
    case part::ux:
        kernel::gather<part::ux> <<<grid,block>>>( *this, d_data, scale );
        break;
    case part::uy:
        kernel::gather<part::uy> <<<grid,block>>>( *this, d_data, scale );
        break;
    case part::uz:
        kernel::gather<part::uz> <<<grid,block>>>( *this, d_data, scale );
        break;
    }
}

/**
 * @brief Save particle data to disk
 * 
 * @param metadata  Particle metadata (name, labels, units, etc.). Information is used to
 *                  set file name
 * @param iter      Iteration metadata
 * @param path      Path where to save the file
 */
void Particles::save( zdf::part_info &metadata, zdf::iteration &iter, std::string path ) {

    uint32_t np = np_total();
    metadata.np = np;

    // Open file
    zdf::file part_file;
    zdf::open_part_file( part_file, metadata, iter, path+"/"+metadata.name );

    // Gather and save each quantity
    float *d_data = nullptr;
    float *h_data = nullptr;
    if( np > 0 ) {
        d_data = device::malloc<float>( np );
        h_data = host::malloc<float>( np );
    }

    if ( np > 0 ) {
        gather( part::quant::x, d_data );
        device::memcpy_tohost( h_data, d_data, np );
    }
    zdf::add_quant_part_file( part_file, "x", h_data, np );

    if ( np > 0 ) {
        gather( part::quant::y, d_data );
        device::memcpy_tohost( h_data, d_data, np );
    }
    zdf::add_quant_part_file( part_file, "y", h_data, np );

    if ( np > 0 ) {
        gather( part::quant::ux, d_data );
        device::memcpy_tohost( h_data, d_data, np );
    }
    zdf::add_quant_part_file( part_file, "ux", h_data, np );

    if ( np > 0 ) {
        gather( part::quant::uy, d_data );
        device::memcpy_tohost( h_data, d_data, np );
    }
    zdf::add_quant_part_file( part_file, "uy", h_data, np );

    if ( np > 0 ) {
        gather( part::quant::uz, d_data );
        device::memcpy_tohost( h_data, d_data, np );
    }
    zdf::add_quant_part_file( part_file, "uz", h_data, np );

    // Close the file
    zdf::close_file( part_file );

    // Cleanup
    if ( np > 0 ) {
        device::free( d_data );
        host::free( h_data );
    }
}

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
 * @param periodic  (in) Correct for periodic boundaries
 */
void __launch_bounds__(opt_bnd_check_block) bnd_check( 
    ParticleData part, ParticleSortData sort, 
    int2 const periodic
) {
    int2 ntiles = make_int2( part.ntiles.x, part.ntiles.y );
    int2 lim = make_int2( part.nx.x, part.nx.y );

    /// @brief [shared] Number of particles moving in each direction
    __shared__ int _npt[9];

    /// @brief [shared] Number of particle leaving tile
    __shared__ int _nout;

    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int tile_id = tile_idx.y * ntiles.x + tile_idx.x;

    const auto offset  = part.offset[ tile_id ];
    const auto np      = part.np[ tile_id ];

    int2 * __restrict__ ix    = &part.ix[ offset ];

    /// @brief Indices of particles leaving tile
    int  * __restrict__ idx   = &sort.idx[ offset ];

    // Initialize block shared values
    for( auto i = block_thread_rank(); i < 9; i+= block_num_threads() ) _npt[i] = 0;
    if ( block_thread_rank() == 0 ) _nout = 0;

    block_sync();

    // Count particles according to their motion and store indices of particles leaving tile
    for( int i = block_thread_rank(); i < np; i += block_num_threads() ) {
        int2 ipos = ix[i];
        int xcross = ( ipos.x >= lim.x ) - ( ipos.x < 0 );
        int ycross = ( ipos.y >= lim.y ) - ( ipos.y < 0 );
        
        if ( xcross || ycross ) {
            block::atomic_fetch_add( &_npt[ (ycross+1) * 3 + (xcross+1) ], 1 );
            idx[ block::atomic_fetch_add( &_nout, 1 ) ] = i;
        }
    }
    block_sync();

    // Store number of particles staying / leaving the tile
    if (block_thread_rank() == 0 ){
        // Particles remaining on the tile
        _npt[4] = np - _nout;
        // Store number of particles leaving tile
        sort.nidx[ tile_id ] = _nout;
    }
    block_sync();

    for( int i = block_thread_rank(); i < 9; i += block_num_threads() ) {
        
        // Store number of particles leaving tile in each direction
        sort.npt[ 9*tile_id + i ] = _npt[i];

        // Add number of particles to target neighboring node

        // Find target node
        int target_tx = tile_idx.x + ( i % 3 - 1 );
        int target_ty = tile_idx.y + ( i / 3 - 1 );

        // Correct for periodic boundaries
        if ( periodic.x ) {
            if ( target_tx < 0 )         target_tx += ntiles.x; 
            if ( target_tx >= ntiles.x ) target_tx -= ntiles.x;
        }
        
        if ( periodic.y ) {
            if ( target_ty < 0 )         target_ty += ntiles.y;
            if ( target_ty >= ntiles.y ) target_ty -= ntiles.y;
        }
        
        if ( ( target_tx >= 0 ) && ( target_tx < ntiles.x ) &&
                ( target_ty >= 0 ) && ( target_ty < ntiles.y ) ) {
            int target_tid = target_ty * ntiles.x + target_tx;
            device::atomic_fetch_add( & sort.new_np[ target_tid ], _npt[i] );
        }
    }
}

}

void Particles::bnd_check( ParticleSort & sort ) {

    dim3 grid( ntiles.x, ntiles.y );
    auto block = opt_bnd_check_block;    
    kernel::bnd_check <<<grid,block>>> ( *this, sort, periodic );
}


namespace kernel {

__global__
/**
 * @brief Recalculates particle tile offset and sets np to 0
 * 
 * @note The number of particles in each tile is set to 0
 * 
 * @param tmp           (out) Particle buffer
 * @param new_np        (in)  New number of particles per tile.
 */
void update_tile_info(
    ParticleData tmp,
    const int * __restrict__ new_np )
{
    /// @brief [shared] Sum of previous warp
    __shared__ int _prev;
    /// @brief [shared] Temporary results from each warp
    __shared__ int _tmp[ MAX_WARPS ];

    const int ntiles = tmp.ntiles.x * tmp.ntiles.y;

    _prev = 0;

    for( int i = block_thread_rank(); i < ntiles; i += block_num_threads() ) {
        auto s = new_np[i];
        auto v = warp::exscan_add( s );

        if ( warp::thread_rank() == warp::num_threads() - 1 )
            _tmp[ warp::group_rank() ] = v + s;

        block_sync();

        // Only 1 warp does this
        if ( warp::group_rank() == 0 ) {
            auto t = _tmp[ warp::thread_rank() ];
            t = warp::exscan_add(t);
            _tmp[ warp::thread_rank() ] = t + _prev;
        }
        block_sync();

        // Add in contribution from previous threads
        v += _tmp[ warp::group_rank() ];

        tmp.offset[i] = v;
        tmp.np[i] = 0;

        if ( block_thread_rank() == block_num_threads()-1 )
            _prev = v+s;

        block_sync();
    }
}

}


/**
 * @brief Recalculates particle tile offset
 * 
 * @note The number of particles in each tile is set to 0
 * 
 * @param tmp           (out) Particle buffer
 * @param new_np        (in)  New number of particles per tile.
 */
void update_tile_info(
    ParticleData & tmp,
    const int * __restrict__ new_np )
{
    const int ntiles = tmp.ntiles.x * tmp.ntiles.y;
    dim3 block( ( ntiles < 1024 ) ? ntiles : 1024 );    

    kernel::update_tile_info <<< 1, block >>> ( tmp, new_np );
}


namespace kernel {

__global__
/**
 * @brief Recalculates particle tile offset, leaving room for additional particles
 * 
 * @note The number of particles in each tile is set to 0
 * 
 * @param tmp           (out) Particle buffer
 * @param new_np        (in/out) New number of particles per tile. Set to 0 after calculation.
 * @param extra         (in) Additional incoming particles
 * @param dev_np        (out) Total number of particles (including additional ones)
 */
void __launch_bounds__(opt_update_tile_info_block) update_tile_info( 
    ParticleData tmp,
    const int * __restrict__ new_np, 
    const int * __restrict__ extra,
    uint32_t * dev_np
) {

    /// @brief [shared] Sum of previous warp
    __shared__ int _prev;
    /// @brief [shared] Temporary results from each warp
    __shared__ int _tmp[ MAX_WARPS ];

    const int ntiles = tmp.ntiles.x * tmp.ntiles.y;

    _prev = 0;

    for( int i = block_thread_rank(); i < ntiles; i += block_num_threads() ) {
        auto s = new_np[i] + extra[i];
        auto v = warp::exscan_add( s );

        if ( warp::thread_rank() == warp::num_threads() - 1 )
            _tmp[ warp::group_rank() ] = v + s;

        block_sync();

        // Only 1 warp does this
        if ( warp::group_rank() == 0 ) {
            auto t = _tmp[ warp::thread_rank() ];
            t = warp::exscan_add(t);
            _tmp[ warp::thread_rank() ] = t + _prev;
        }
        block_sync();

        // Add in contribution from previous threads
        v += _tmp[ warp::group_rank() ];

        tmp.offset[i] = v;
        tmp.np[i] = 0;

        if ( ( block_thread_rank() == block_num_threads()-1 ) || ( i + 1 == ntiles ) )
            _prev = v+s;
        block_sync();
    }

    if ( block_thread_rank() == 0 ) {
        dev_np[0] = _prev;
    }
}

}

/**
 * @brief Recalculates particle tile offset, leaving room for additional particles
 * 
 * @note The number of particles in each tile is set to 0
 * 
 * @param tmp           (out) Particle buffer
 * @param new_np        (in)  New number of particles per tile.
 * @param extra         (in) Additional incoming particles
 * @param dev_np        (out) Temporary variable to get total number of particles
 */
uint32_t update_tile_info(
    ParticleData & tmp,
    const int * __restrict__ new_np,
    const int * __restrict__ extra,
    device::Var<uint32_t> & dev_np )
{
    const int ntiles = tmp.ntiles.x * tmp.ntiles.y;
    auto block = ( ntiles < opt_update_tile_info_block ) ? ntiles : opt_update_tile_info_block;

    kernel::update_tile_info <<<1,block>>> ( tmp, new_np, extra, dev_np.ptr() );

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
    ParticleData part, 
    ParticleData tmp,
    const ParticleSortData sort,
    const int2 periodic
) {
    const int2 ntiles = make_int2( part.ntiles.x, part.ntiles.y );
    const int2 lim = make_int2( part.nx.x, part.nx.y );

    /// @brief [shared] offsets in target buffer
    __shared__ int _dir_offset[9];

    /// @brief [shared] index of particle used to fill hole
    __shared__ int _c;

    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id = tile_idx.y * ntiles.x + tile_idx.x;

    int const old_offset      = part.offset[ tile_id ];
    int * __restrict__ npt    = &sort.npt[ 9*tile_id ];

    int2   * __restrict__ ix  = &part.ix[ old_offset ];
    float2 * __restrict__ x   = &part.x[ old_offset ];
    float3 * __restrict__ u   = &part.u[ old_offset ];

    int * __restrict__ idx    = &sort.idx[ old_offset ];
    uint32_t const nidx       = sort.nidx[ tile_id ];

    int const new_offset = tmp.offset[ tile_id ];
    int const new_np     = sort.new_np[ tile_id ];

    // The _dir_offset variable holds the offset for each of the 9 target
    // tiles so the tmp_* variables just point to the beggining of the buffers
    int2* __restrict__  tmp_ix  = tmp.ix;
    float2* __restrict__ tmp_x  = tmp.x;
    float3* __restrict__ tmp_u  = tmp.u;

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
    if ( block_thread_rank() == 0 ) {
        _dir_offset[4] = new_offset + device::atomic_fetch_add( & tmp.np[ tile_id ], nshift );
    }
    block_sync();

    // Find offsets on new buffer
    for( int i = block_thread_rank(); i < 9; i += block_num_threads() ) {
        
        if ( i != 4 ) {
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

                _dir_offset[i] = tmp.offset[ target_tid ] + 
                                device::atomic_fetch_add( &tmp.np[ target_tid ], npt[ i ] );
            } else {
                // Otherwise mark offset as invalid
                _dir_offset[i] = -1;
            }
        } 
    }

    // First candidate to fill holes
    _c = n0;

    block_sync();

    // Copy particles moving away from tile and fill holes
    for( int i = block_thread_rank(); i < nidx; i += block_num_threads() ) {
        
        int k = idx[i];

        int2 nix  = ix[k];
        float2 nx = x[k];
        float3 nu = u[k];
        
        int xcross = ( nix.x >= lim.x ) - ( nix.x < 0 );
        int ycross = ( nix.y >= lim.y ) - ( nix.y < 0 );

        const int dir = (ycross+1) * 3 + (xcross+1);

        // Check if particle crossed into a valid neighbor
        if ( _dir_offset[dir] >= 0 ) {        

            // _dir_offset[] includes the offset in the global tmp particle buffer
            int l = block::atomic_fetch_add( & _dir_offset[dir], 1 );

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
                c = block::atomic_fetch_add( &_c, 1 );
                invalid = ( ix[c].x < 0 ) || ( ix[c].x >= lim.x ) || 
                          ( ix[c].y < 0 ) || ( ix[c].y >= lim.y );
            } while (invalid);

            ix[ k ] = ix[ c ];
            x [ k ] = x [ c ];
            u [ k ] = u [ c ];
        }
    }

    block_sync();

    // At this point all particles up to n0 are correct

    // Copy particles that need to be shifted
    // We've reserved space for nshift particles earlier
    const int new_idx = _dir_offset[4];

    if ( new_offset >= old_offset ) {
        // Copy from beggining of buffer
        for( int i = block_thread_rank(); i < nshift; i += block_num_threads() ) {
            tmp_ix[ new_idx + i ] = ix[ i ];
            tmp_x[ new_idx + i ]  = x [ i ];
            tmp_u[ new_idx + i ]  = u [ i ];
        }

    } else {

        // Copy from end of buffer
        const int old_idx = n0 - nshift;
        for( int i = block_thread_rank(); i < nshift; i += block_num_threads() ) {
            tmp_ix[ new_idx + i ] = ix[ old_idx + i ];
            tmp_x[ new_idx + i ]  = x [ old_idx + i ];
            tmp_u[ new_idx + i ]  = u [ old_idx + i ];
        }
    }

    // Store current number of local particles
    // These are already in the correct position in global buffer
    if ( block_thread_rank() == 0 ) {
        part.np[ tile_id ] = n0 - nshift;
    }

}

}

/**
 * @brief Copy outgoing particles to temporary buffer
 * 
 * @param tmp   Temporary particle buffer (has new offsets)
 * @param sort  Sort data (new number of particles per tile, indices of
 *              particles leaving the tile, etc.)
 */
void Particles::copy_out( ParticleData & tmp, const ParticleSortData & sort )
{
    dim3 grid( ntiles.x, ntiles.y );
    auto block = opt_copy_out_block;

    kernel::copy_out <<< grid, block >>> ( *this, tmp, sort, periodic );
}


namespace kernel {

__global__
/**
 * @brief CUDA kernel for copying incoming particles to main buffer.
 * 
 * @param part      Main particle data
 * @param tmp       Temporary particle data
 */
void __launch_bounds__(opt_copy_in_block) copy_in(
    ParticleData part,
    ParticleData tmp
) {
    const int2 ntiles = make_int2( part.ntiles.x, part.ntiles.y );

    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id = tile_idx.y * ntiles.x + tile_idx.x;

    const int old_offset       =  part.offset[ tile_id ];
    const int old_np           =  part.np[ tile_id ];

    const int new_offset       =  tmp.offset[ tile_id ];
    const int tmp_np           =  tmp.np[ tile_id ];

    // Notice that we are already working with the new offset
    int2   * __restrict__ ix  = &part.ix[ new_offset ];
    float2 * __restrict__ x   = &part.x [ new_offset ];
    float3 * __restrict__ u   = &part.u [ new_offset ];

    int2   * __restrict__ tmp_ix = &tmp.ix[ new_offset ];
    float2 * __restrict__ tmp_x  = &tmp.x [ new_offset ];
    float3 * __restrict__ tmp_u  = &tmp.u [ new_offset ];

    if ( new_offset >= old_offset ) {

        // Add particles to the end of the buffer
        for( int i = block_thread_rank(); i < tmp_np; i += block_num_threads() ) {
            ix[ old_np + i ] = tmp_ix[ i ];
            x[ old_np + i ]  = tmp_x[ i ];
            u[ old_np + i ]  = tmp_u[ i ];
        }

    } else {

        // Add particles to the beggining of buffer
        int np0 = old_offset - new_offset;
        if ( np0 > tmp_np ) np0 = tmp_np;
        
        for( int i = block_thread_rank(); i < np0; i += block_num_threads() ) {
            ix[ i ] = tmp_ix[ i ];
            x[ i ]  = tmp_x[ i ];
            u[ i ]  = tmp_u[ i ];
        }

        // If any particles left, add particles to the end of the buffer
        for( int i = np0 + block_thread_rank(); i < tmp_np; i += block_num_threads() ) {
            ix[ old_np + i ] = tmp_ix[ i ];
            x[ old_np + i ]  = tmp_x[ i ];
            u[ old_np + i ]  = tmp_u[ i ];
        }

    }

    block_sync();

    // Store the new offset and number of particles
    if ( block_thread_rank() == 0 ) {
        part.np[ tile_id ]     = old_np + tmp_np;
        part.offset[ tile_id ] = new_offset;
    }

}

}

/**
 * @brief Copy incoming particles to main buffer. Buffer will be fully sorted after
 *        this step
 *
 * @param tmp       Temporary particle data
 */
void Particles::copy_in( ParticleData & tmp ) {
    dim3 grid( ntiles.x, ntiles.y );
    auto block = opt_copy_in_block;

    kernel::copy_in <<< grid, block >>> ( *this, tmp );
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
    ParticleData part,
    ParticleData tmp,
    const ParticleSortData sort,
    const int2 periodic
) {
    const int2 ntiles = make_int2( part.ntiles.x, part.ntiles.y );
    const int2 lim = make_int2( part.nx.x, part.nx.y );

    /// @brief [shared] offsets in target buffer
    __shared__ int _dir_offset[9];

    /// @brief [shared] index of particle used to fill hole
    __shared__ int _c;

    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id = tile_idx.y * part.ntiles.x + tile_idx.x;


    int const old_offset      = part.offset[ tile_id ];
    int * __restrict__ npt    = &sort.npt[ 9*tile_id ];

    int2   * __restrict__ ix  = &part.ix[ old_offset ];
    float2 * __restrict__ x   = &part.x[ old_offset ];
    float3 * __restrict__ u   = &part.u[ old_offset ];

    int * __restrict__ idx    = &sort.idx[ old_offset ];
    uint32_t const nidx       = sort.nidx[ tile_id ];

    // The _dir_offset variables hold the offset for each of the 9 target
    // tiles so the tmp_* variables just point to the beggining of the buffers
    int2* __restrict__  tmp_ix  = tmp.ix;
    float2* __restrict__ tmp_x  = tmp.x;
    float3* __restrict__ tmp_u  = tmp.u;

    // Find offsets on new buffer
    for( int i = block_thread_rank(); i < 9; i += block_num_threads() ) {
        
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

            _dir_offset[i] = tmp.offset[ target_tid ] + 
                device::atomic_fetch_add( & tmp.np[ target_tid ], npt[ i ] );
        
        } else {
            // Otherwise mark offset as invalid
            _dir_offset[i] = -1;
        }
    }

    // Particles remaining on tile
    const int n0 = npt[4];

    // First candidate to fill holes
    _c = n0;

    block_sync();

    // Copy particles moving away from tile and fill holes
    for( int i = block_thread_rank(); i < nidx; i += block_num_threads() ) {
        
        int k = idx[i];

        int2 nix  = ix[k];
        float2 nx = x[k];
        float3 nu = u[k];
        
        int xcross = ( nix.x >= lim.x ) - ( nix.x < 0 );
        int ycross = ( nix.y >= lim.y ) - ( nix.y < 0 );

        const int dir = (ycross+1) * 3 + (xcross+1);

        // Check if particle crossed into a valid neighbor
        if ( _dir_offset[dir] >= 0 ) {        

            // _dir_offset[] includes the offset in the global tmp particle buffer
            int l = block::atomic_fetch_add( & _dir_offset[dir], 1 );

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
                c = block::atomic_fetch_add( &_c, 1 );
                invalid = ( ix[c].x < 0 ) || ( ix[c].x >= lim.x ) || 
                          ( ix[c].y < 0 ) || ( ix[c].y >= lim.y );
            } while (invalid);

            ix[ k ] = ix[ c ];
            x [ k ] = x [ c ];
            u [ k ] = u [ c ];
        }
    }

    block_sync();

    // Copy particles staying in tile
    const int start = _dir_offset[4];

    for( int i = block_thread_rank(); i < nidx; i += block_num_threads() ) {
        tmp_ix[ start + i ] = ix[i];
        tmp_x [ start + i ] = x[i];
        tmp_u [ start + i ] = u[i];
    }
}

}

/**
 * @brief Copies copy all particles to correct tiles in another buffer
 * 
 * @note Requires that new buffer (`tmp`) already has the correct offset
 *       values, and number of particles set to 0.
 * 
 * @param tmp       Temporary particle buffer (has new offsets)
 * @param sort      Sort data (indices of particles leaving the tile, etc.)
 */
void Particles::copy_sorted( ParticleData & tmp, const ParticleSortData & sort ) {

    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 1024 );

    kernel::copy_sorted <<< grid, block >>> ( *this, tmp, sort, periodic );
}


/**
 * @brief Moves particles to the correct tiles
 * 
 * @note Particles are only expected to have moved no more than 1 tile
 *       in each direction. If necessary the code will grow the particle buffer
 * 
 * @param tmp       Temporary particle buffer
 * @param sort      Temporary sort index 
 * @param extra     Additional space to add to each tile. Leaves  room for
 *                  particles to be injected later.
 */
void Particles::tile_sort( Particles & tmp, ParticleSort & sort, const int * __restrict__ extra ) {

    // Reset sort data
    sort.reset();

    // Get new number of particles per tile
    bnd_check( sort );

    if ( extra ) {
        // Get new offsets, including extra values in offset calculations
        // Used to reserve space in particle buffer for later injection
        auto total_np = update_tile_info ( tmp, sort.new_np, extra, dev_tmp_uint32 );

        if ( total_np > max_part ) { 

            // grow tmp particle buffer
            tmp.grow_buffer( total_np );

            // copy all particles to correct tiles in tmp buffer
            copy_sorted( tmp, sort );

            // swap buffers
            swap_buffers( *this, tmp );

            // grow tmp particle buffer for future use
            tmp.grow_buffer( max_part );

        } else {
            // Copy outgoing particles (and particles needing shifting) to staging area
            copy_out ( tmp, sort );

            // Copy particles from staging area into final positions in partile buffer
            copy_in ( tmp );
        }

    } else {
        // Get new offsets
        update_tile_info ( tmp, sort.new_np );

        // Copy outgoing particles (and particles needing shifting) to staging area
        copy_out ( tmp, sort );

        // Copy particles from staging area into final positions in partile buffer
        copy_in ( tmp );
    }

    // For debug only, remove from production code
    // validate( "After tile_sort" );
}

namespace kernel {

__global__
/**
 * @brief CUDA kernel for shifting particle cells by the required amount
 * 
 * @param part      Particle data
 * @param shift     Cell shift
 */
void cell_shift( ParticleData part, int2 const shift ) {

    const int2 ntiles = make_int2( part.ntiles.x, part.ntiles.y );
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id = tile_idx.y * ntiles.x + tile_idx.x;

    const auto tile_off = part.offset[ tile_id ];
    const auto tile_np  = part.np[ tile_id ];

    int2   * const __restrict__ ix = &part.ix[ tile_off ];

    for( int i = block_thread_rank(); i < tile_np; i += block_num_threads() ) {
        int2 cell = ix[i];
        cell.x += shift.x;
        cell.y += shift.y;
        ix[i] = cell;
    }
}

}

/**
 * @brief Shifts particle cells by the required amount
 * 
 * @note cells are shited by adding the parameter `shift` to the particle cell
 *       indexes.
 * 
 * @warning This routine does not check if the particles are still inside the
 *          tile.
 * 
 * @param shift     Cell shift in both directions
 */
void Particles::cell_shift( int2 const shift ) {

    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 1024 );

    kernel::cell_shift <<< grid, block >>> ( *this, shift );
}

#define __ULIM __FLT_MAX__

namespace kernel {

__global__
void validate( ParticleData part, int const over, uint32_t * out ) {
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int2 ntiles = make_int2( part.ntiles.x, part.ntiles.y );
    const int tile_id = tile_idx.y * ntiles.x + tile_idx.x;

    int const offset = part.offset[ tile_id ];
    int const np     = part.np[ tile_id ];
    int2   const * const __restrict__ ix = &part.ix[ offset ];
    float2 const * const __restrict__ x  = &part.x[ offset ];
    float3 const * const __restrict__ u  = &part.u[ offset ];

    __shared__ int _err;
    _err = 0;

    int2 const lb = make_int2( -over, -over );
    int2 const ub = make_int2( part.nx.x + over, part.nx.y + over ); 

    block_sync();

    for( int i = block_thread_rank(); i < np; i += block_num_threads() ) {

        int err = 0;

        if ( (ix[i].x < lb.x) || (ix[i].x >= ub.x )) {
            printf("[%d,%d] Invalid ix[%d].x position (%d), range = [%d,%d[\n", tile_idx.x, tile_idx.y, i, ix[i].x, lb.x, ub.x );
            err = 1;
        }
        if ( x[i].x < -0.5f || x[i].x >= 0.5f ) {
            printf("[%d,%d] Invalid x[%d].x position (%f), range = [-0.5,0.5[\n", tile_idx.x, tile_idx.y, i, x[i].x );
            err = 1;
        }

        if ( (ix[i].y < lb.y) || (ix[i].y >= ub.y )) {
            printf("[%d,%d] Invalid ix[%d].y position (%d), range = [%d,%d[\n", tile_idx.x, tile_idx.y, i, ix[i].y, lb.y, ub.y );
            err = 1;
        }
        if ( x[i].y < -0.5f || x[i].y >= 0.5f ) {
            printf("[%d,%d] Invalid x[%d].y position (%f), range = [-0.5,0.5[\n", tile_idx.x, tile_idx.y, i, x[i].y );
            err = 1;
        }

        if ( isnan(u[i].x) || isinf(u[i].x) || fabsf(u[i].x) >= __ULIM ) {
            printf("[%d,%d] Invalid u[%d].x gen. velocity (%f)\n", tile_idx.x, tile_idx.y, i, u[i].x );
            err = 1;
        }

        if ( isnan(u[i].y) || isinf(u[i].y) || fabsf(u[i].x) >= __ULIM ) {
            printf("[%d,%d] Invalid u[%d].y gen. velocity (%f)\n", tile_idx.x, tile_idx.y, i, u[i].y );
            err = 1;
        }

        if ( isnan(u[i].z) || isinf(u[i].z) || fabsf(u[i].x) >= __ULIM ) {
            printf("[%d,%d] Invalid u[%d].z gen. velocity (%f)\n", tile_idx.x, tile_idx.y, i, u[i].z );
            err = 1;
        }

        if ( err ) _err = 1;

        if ( _err ) {
            *out = 1;
            break;
        }
    }
}

}

/**
 * @brief Checks particle buffer data for error
 * 
 * @warning This routine is meant for debug only and should not be called 
 *          for production code.
 * 
 * @note
 * 
 * The routine will check for:
 *      1. Invalid cell data (out of tile bounds)
 *      2. Invalid position data (out of [-0.5,0.5[)
 *      3. Invalid momenta (nan, inf or above __ULIM macro value)
 * 
 * If there are any errors found the routine will exit the code.
 * 
 * @param msg       Message to print in case error is found
 * @param over      Amount of extra cells indices beyond limit allowed. Used
 *                  when checking the buffer before tile_sort()
 */
void Particles::validate( std::string msg, int const over ) {

    // Errors detected
    dev_tmp_uint32 = 0;

    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 32 );
    kernel::validate <<< grid, block >>> ( *this, over, dev_tmp_uint32.ptr() );

    if ( dev_tmp_uint32.get() ) {
        std::cerr << "(*error*) " << msg << " (np = " << np_total() << ")\n";
        ABORT( "invalid particle found, aborting..." );
    } else {
        std::cout << "(*info) " << msg << " particles ok\n";
    }
}

#undef __ULIM
