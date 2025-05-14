#include "particles.h"

#include <iostream>
#include <iomanip>

#include <string>
#include <cmath>

#include "timer.h"

/**
 * Optimized parameters for NVIDIA A100
 */
#define opt_bnd_check_block 1024
#define opt_update_tile_info_block 64
#define opt_copy_out_block 32
#define opt_copy_in_block 1024



namespace kernel {

    __global__
    /**
     * @brief Kernel for updating msg_np from msg_np_tile
     * 
     * @note Must be called with a grid size of (3,3)
     * 
     * @param recv_msg_np_tile      Number of particles in receive message tiles
     * @param recv_msg_np           Total number of particles per receive message
     * @param send_msg_np_tile      Number of particles in receive message tiles
     * @param send_msg_np           Total number of particles per receive message
     * @param ntiles                Grid tile dimension
     */
    void update_msg_np( 
        int const * const __restrict__ recv_msg_np_tile, int * const __restrict__ recv_msg_np, 
        int const * const __restrict__ send_msg_np_tile, int * const __restrict__ send_msg_np, 
        const uint2 ntiles ) {

        __shared__ unsigned int tile_off[9];
        tile_off[0] = 0;
        tile_off[1] = 1;
        tile_off[2] = 1 +   ntiles.x;
        tile_off[3] = 2 +   ntiles.x;
        tile_off[4] = 2 +   ntiles.x +   ntiles.y;
        tile_off[5] = 2 +   ntiles.x +   ntiles.y;
        tile_off[6] = 2 +   ntiles.x + 2*ntiles.y;
        tile_off[7] = 3 +   ntiles.x + 2*ntiles.y;
        tile_off[8] = 3 + 2*ntiles.x + 2*ntiles.y;

        __shared__ int recv_np_dir; recv_np_dir = 0;
        __shared__ int send_np_dir; send_np_dir = 0;

        int dir = 3 * blockIdx.y + blockIdx.x;

        // Number of tiles in message from direction
        int size = 1;                                   // corners
        if ( dir == 1 || dir == 7 ) size = ntiles.x;    // y boundary
        if ( dir == 3 || dir == 5 ) size = ntiles.y;    // x boundary

        block_sync();

        // tile offset in message
        int off = tile_off[ dir ];

        if ( dir != 4 ) {

            // Add contribution from all tiles in message
            int recv_np_tmp = 0;
            int send_np_tmp = 0;

            for( int i = block_thread_rank(); i < size; i+= block_num_threads() ) {
                recv_np_tmp += recv_msg_np_tile[ off + i ];
                send_np_tmp += send_msg_np_tile[ off + i ];
            }
            
            // These require compute capability 8.x
            recv_np_tmp = __reduce_add_sync( 0xffffffff, recv_np_tmp );
            send_np_tmp = __reduce_add_sync( 0xffffffff, send_np_tmp  );

            if ( warp::thread_rank() == 0 ) {
                // if block_num_threads() <= WARP_SIZE this could be replaced by a copy
                block::atomic_fetch_add( &recv_np_dir, recv_np_tmp );
                block::atomic_fetch_add( &send_np_dir, send_np_tmp );
            }
        }

        block_sync();

        // Store result in device memory
        if ( block_thread_rank() == 0 ) {
            recv_msg_np[ dir ] = recv_np_dir;
            send_msg_np[ dir ] = send_np_dir;
        }
    }
}

/**
 * @brief Exchange number of particles in edge cells
 * 
 */
void ParticleSort::exchange_np() {

    // Size of message according to direction
    const int ntx = ntiles.x;
    const int nty = ntiles.y;

    auto size = [ ntx, nty ]( int dir ) -> unsigned int {
        unsigned int s = 1;                   // corners
        if ( dir == 1 || dir == 7 ) s = ntx;  // y boundary
        if ( dir == 3 || dir == 5 ) s = nty;  // x boundary
        return s;
    };

    // Post receives
    unsigned int idx = 0;
    for( auto dir = 0; dir < 9; dir++ ) {            
        if ( dir != 4 ) {
            MPI_Irecv( &recv.buffer[idx], size(dir), MPI_INT, neighbor[dir],
                    source_tag(dir), comm, &recv.requests[dir]);
            idx += size(dir);
        } else {
            recv.requests[dir] = MPI_REQUEST_NULL;
        }
    }

    // Post sends
    idx = 0;
    for( auto dir = 0; dir < 9; dir++ ) {
        if ( dir != 4 ) {
            MPI_Isend( &send.buffer[idx], size(dir), MPI_INT, neighbor[dir],
                dest_tag(dir), comm, &send.requests[dir]);
            idx += size(dir);
        } else {
            send.requests[dir] = MPI_REQUEST_NULL;
        }
    }

    // Wait for receives to complete
    MPI_Waitall( 9, recv.requests, MPI_STATUSES_IGNORE );

    // update send.msg_np[] and recv.msg_np[]
    // This is only to avoid copying send.buffer and recv.buffer to host
    dim3 grid( 3, 3 );
    auto block = 32;
    kernel::update_msg_np <<< grid, block >>> ( 
        send.buffer, send.msg_np, 
        recv.buffer, recv.msg_np,
        ntiles
    );

    // Wait for sends to complete
    MPI_Waitall( 9, send.requests, MPI_STATUSES_IGNORE );

    // Ensure kernel::update_msg_np has completed     
    cudaDeviceSynchronize();
}


namespace kernel {

__global__
/**
 * @brief CUDA kernel for getting node local number of particles
 * 
 * @param np        Number of particles per tile
 * @param ntiles    Total number of tiles
 * @param total     Output
 */
void np_local( 
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
uint32_t Particles::np_local() {

    dev_tmp_uint32 = 0;

    auto size = ntiles.x*ntiles.y;
    auto block = ( size < 1024 ) ? size : 1024 ;
    auto grid = (size-1)/block + 1;
    kernel::np_local <<< grid, block >>> ( np, size, dev_tmp_uint32.ptr() );

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
void gather_quant( 
    ParticleData part,
    float * const __restrict__ d_data )
{

    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int   tile_id  = tile_idx.y * part.ntiles.x + tile_idx.x;
    
    // Offset and number of particles on particle buffer
    const auto  offset = part.offset[ tile_id ];
    const auto  np     = part.np[ tile_id ];

    // Global spatial offsets of local tile
    const int offx = (part.tile_off.x + tile_idx.x) * part.nx.x;
    const int offy = (part.tile_off.y + tile_idx.y) * part.nx.y;

    int2   const * __restrict__ const ix = & part.ix[ offset ];
    float2 const * __restrict__ const x  = & part.x [ offset ];
    float3 const * __restrict__ const u  = & part.u [ offset ];

    for( int idx = block_thread_rank(); idx < np; idx += block_num_threads() ) {
        float val;
        if constexpr( quant == part::x  ) val = ( offx + ix[idx].x ) + (0.5f + x[idx].x);
        if constexpr( quant == part::y  ) val = ( offy + ix[idx].y ) + (0.5f + x[idx].y);
        if constexpr( quant == part::ux ) val = u[idx].x;
        if constexpr( quant == part::uy ) val = u[idx].y;
        if constexpr( quant == part::uz ) val = u[idx].z;
        d_data[ offset + idx ] = val;
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
        kernel::gather_quant<part::x> <<<grid,block>>>( *this, d_data );
        break;
    case part::y:
        kernel::gather_quant<part::y> <<<grid,block>>>( *this, d_data );
        break;
    case part::ux:
        kernel::gather_quant<part::ux> <<<grid,block>>>( *this, d_data );
        break;
    case part::uy:
        kernel::gather_quant<part::uy> <<<grid,block>>>( *this, d_data );
        break;
    case part::uz:
        kernel::gather_quant<part::uz> <<<grid,block>>>( *this, d_data );
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
void gather_quant( 
    ParticleData part,
    float * const __restrict__ d_data,
    const float2 scale
) {
    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int   tile_id  = tile_idx.y * part.ntiles.x + tile_idx.x;

    // Spatial offsets of local tile
    const int offx = (part.tile_off.x + tile_idx.x) * part.nx.x;
    const int offy = (part.tile_off.y + tile_idx.y) * part.nx.y;

    // Offset and number of particles on particle buffer
    const auto  offset = part.offset[ tile_id ];
    const auto  np     = part.np[ tile_id ];

    int2   * const __restrict__ ix       = & part.ix[ offset ];
    float2 const * __restrict__ const x  = & part.x[ offset ];
    float3 const * __restrict__ const u  = & part.u[ offset ];

    for( int idx = block_thread_rank(); idx < np; idx += block_num_threads() ) {
        float val;
        if constexpr( quant == part::x  ) val = (offx + ix[idx].x) + (0.5f + x[idx].x);
        if constexpr( quant == part::y  ) val = (offy + ix[idx].y) + (0.5f + x[idx].y);
        if constexpr( quant == part::ux ) val = u[idx].x;
        if constexpr( quant == part::uy ) val = u[idx].y;
        if constexpr( quant == part::uz ) val = u[idx].z;
        d_data[ offset + idx ] = fma( scale.x, val, scale.y );
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
 * @param d_data        Output (device) data buffer (assumed to have sufficient size)
 * @param scale         Scale factor for data
 */
void Particles::gather( part::quant quant, const float2 scale, float * const __restrict__ d_data )
{
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 1024 );
    
    // Gather data on device
    switch (quant) {
    case part::x : 
        kernel::gather_quant<part::x> <<<grid,block>>>( *this, d_data, scale );
        break;
    case part::y:
        kernel::gather_quant<part::y> <<<grid,block>>>( *this, d_data, scale );
        break;
    case part::ux:
        kernel::gather_quant<part::ux> <<<grid,block>>>( *this, d_data, scale );
        break;
    case part::uy:
        kernel::gather_quant<part::uy> <<<grid,block>>>( *this, d_data, scale );
        break;
    case part::uz:
        kernel::gather_quant<part::uz> <<<grid,block>>>( *this, d_data, scale );
        break;
    }
}

/**
 * @brief Save particle data to disk
 * 
 * @param quants    Quantities to save
 * @param metadata  Particle metadata (name, labels, units, etc.). Information is used to
 *                  set file name
 * @param iter      Iteration metadata
 * @param path      Path where to save the file
 */
void Particles::save( const part::quant quants[], zdf::part_info &metadata, zdf::iteration &iter, std::string path ) {

    // Get total number of particles to save
    uint64_t local = np_local();
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
            zdf::dataset dsets[ metadata.nquants ];

            for( uint32_t i = 0; i < metadata.nquants; i++ ) {
                dsets[i].name      = metadata.quants[i];
                dsets[i].data_type = zdf::data_type<float>();
                dsets[i].ndims     = 1;
                dsets[i].data      = nullptr;
                dsets[i].count[0]  = global; 

                if ( !zdf::start_cdset( part_file, dsets[i] ) ) {
                    std::cerr << "Particles::save() - Unable to create chunked dataset " << metadata.quants[i] << '\n';
                    exit(1);
                }
            }

            // Allocate buffers for gathering particle data
            float *d_data = device::malloc<float>( local );
            float *h_data = host::malloc<float>( local );

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
                device::memcpy_tohost( h_data, d_data, local );
                zdf::write_cdset( part_file, dsets[i], chunk, file_off );
            }

            // Free temporary data
            device::free( d_data );
            host::free( h_data );

            // close the datasets
            for( uint32_t i = 0; i < metadata.nquants; i++ ) 
                zdf::end_cdset( part_file, dsets[i] );

            // Close the file
            zdf::close_file( part_file );
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
    ParticleData part, 
    ParticleSortData sort, 
    const part::bnd_type local_bnd)
{
    // ntiles needs to be set to signed because of the comparisons below
    const int2 ntiles = make_int2( part.ntiles.x, part.ntiles.y );
    const int2 lim = make_int2( part.nx.x, part.nx.y );

    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int tile_id = tile_idx.y * ntiles.x + tile_idx.x;

    const auto offset  = part.offset[ tile_id ];
    const auto np      = part.np[ tile_id ];

    int2 * __restrict__ ix    = &part.ix[ offset ];

    /// @brief Indices of particles leaving tile
    int  * __restrict__ idx   = &sort.idx[ offset ];

    /// @brief [shared] Number of particles moving in each direction
    __shared__ int _npt[9];
    for( auto i = block_thread_rank(); i < 9; i+= block_num_threads() ) _npt[i] = 0;

    /// @brief [shared] Number of particle leaving tile
    __shared__ int _nout;
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
        int2 target = make_int2( tile_idx.x + i % 3 - 1, tile_idx.y + i / 3 - 1 );

        int target_tid = part::tid_coords( target, ntiles, local_bnd );
        
        if ( target_tid >= 0 ) {
            device::atomic_fetch_add( & sort.new_np[ target_tid ], _npt[i] );
        }
    }
}

}

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
void bnd_check( ParticleData part, ParticleSortData sort, 
    const part::bnd_type local_bnd )
{
    
    dim3 grid( part.ntiles.x, part.ntiles.y );
    auto block = opt_bnd_check_block;    
    kernel::bnd_check <<<grid,block>>> ( part, sort, local_bnd );
}

namespace kernel {

__global__
/**
 * @brief Recalculates particle tile offset, leaving room for additional particles
 * 
 * @note The number of particles in each tile is set to 0
 * 
 * @param tmp_part      (out) Particle buffer
 * @param new_np        (in/out) New number of particles per tile. Set to 0 after calculation.
 * @param dev_np        (out) Total number of particles (including additional ones)
 * @param extra         (in) Additional incoming particles
 */
void __launch_bounds__(opt_update_tile_info_block) update_tile_info( 
    ParticleData tmp_part,
    ParticleSortData sort,
    const int * __restrict__ recv_buffer,
    uint32_t * __restrict__ dev_np,
    const int * __restrict__ extra = nullptr ) {

    // New number of particles per tile
    const int * __restrict__ new_np = sort.new_np;

    // Include ghost tiles in calculations
    const auto ntiles_all   = part::all_tiles( tmp_part.ntiles );
    const auto ntiles_msg   = part::msg_tiles( tmp_part.ntiles );
    const auto ntiles_local = part::local_tiles( tmp_part.ntiles );

    /// @brief [shared] Sum of previous warp
    __shared__ int _prev; _prev = 0;
    /// @brief [shared] Temporary results from each warp
    __shared__ int _tmp[ MAX_WARPS ];

    __shared__ unsigned int msg_tile_off[9];
    msg_tile_off[0] = 0;
    msg_tile_off[1] = 1;
    msg_tile_off[2] = 1 +   tmp_part.ntiles.x;
    msg_tile_off[3] = 2 +   tmp_part.ntiles.x;
    msg_tile_off[4] = 2 +   tmp_part.ntiles.x +   tmp_part.ntiles.y;
    msg_tile_off[5] = 2 +   tmp_part.ntiles.x +   tmp_part.ntiles.y;
    msg_tile_off[6] = 2 +   tmp_part.ntiles.x + 2*tmp_part.ntiles.y;
    msg_tile_off[7] = 3 +   tmp_part.ntiles.x + 2*tmp_part.ntiles.y;
    msg_tile_off[8] = 3 + 2*tmp_part.ntiles.x + 2*tmp_part.ntiles.y;
   

    int * __restrict__ offset = tmp_part.offset;
    int * __restrict__ np     = tmp_part.np;

    // Initialize offset[] with the new number of particles
    if ( extra != nullptr ) {
        for( auto i = block_thread_rank(); i < ntiles_all; i += block_num_threads() ) {
            offset[i] = new_np[i] + extra[i];
            np[i] = 0;
        }
    } else {
        for( auto i = block_thread_rank(); i < ntiles_all; i += block_num_threads() ) {
            offset[i] = new_np[i];
            np[i] = 0;
        }
    }

    block_sync();

    // Add incoming particles
    for( int msg_idx = block_thread_rank(); msg_idx < ntiles_msg; msg_idx += block_num_threads() ) {

        ///@brief message direction
        int dir = -1;
        for( int i = 0; i < 8; i++ ) {
            if ( i != 4 && msg_idx < msg_tile_off[i+1] ) { 
                dir = i; 
                break;
            }
        }
        if ( dir < 0 ) dir = 8;

        int dir_idx = msg_idx - msg_tile_off[dir];       

        ///@brief Tile stride for storing received data according to direction
        int tile_stride = 1;
        if ( dir == 3 || dir == 5 ) tile_stride = tmp_part.ntiles.x;

        // Tile offset for storing received data according to direction
        int tile_offset;
        {
            int y = dir / 3;
            int x = dir % 3;
            int xoff = 0; int yoff = 0;
            if ( x == 2 ) xoff = tmp_part.ntiles.x-1;
            if ( y == 2 ) yoff = (tmp_part.ntiles.y-1) * tmp_part.ntiles.x;
            tile_offset = yoff + xoff;
        };

        ///@brief target tile on particle buffer
        int target_tid = dir_idx * tile_stride + tile_offset;

        // Some tiles (corner) will receive particles from multiple messages
        // so we need to use an atomic operation
        device::atomic_fetch_add( & offset[ target_tid ], recv_buffer[ msg_idx ] );
    }

    block_sync();

    // Block-wide exclusive scan
    for( int i = block_thread_rank(); i < ntiles_all; i += block_num_threads() ) {
        auto s = offset[i];
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

        offset[i] = v;

        if ( ( block_thread_rank() == block_num_threads()-1 ) || ( i + 1 == ntiles_all ) )
            _prev = v+s;
        block_sync();
    }

    // Total number of particles
    if ( block_thread_rank() == 0 ) {
        dev_np[0] = _prev;
    }
}

}

 /**
 * @brief Recalculates particle tile offset, leaving room for additional particles
 * 
 * @note The routine also leaves room for particles coming from other MPI nodes.
 *        The number of particles in each tile is set to 0
 * 
 * @param tmp           (out) Temp. Particle buffer
 * @param sort          (in) Sort data (includes information from other MPI nodes)
 * @param dev_np        (out) Temporary variable to get total number of particles
 * @param extra         (in) Additional particles (optional)
 * @return uint32_t     (out) Total number of particles (including additional ones)
 */
uint32_t update_tile_info( ParticleData tmp, ParticleSortData sort, 
    const int * __restrict__ recv_buffer,
    device::Var<uint32_t> & dev_np,  
    const int * __restrict__ extra = nullptr ) {

    const auto ntiles_all   = part::all_tiles( tmp.ntiles );
    auto block = ( ntiles_all < opt_update_tile_info_block ) ? ntiles_all : opt_update_tile_info_block;

    kernel::update_tile_info <<<1,block>>> ( tmp, sort, recv_buffer, dev_np.ptr(), extra );

/*
    {
        int * np = host::malloc<int>( tmp.ntiles.x * tmp.ntiles.y );
        int * offset = host::malloc<int>( tmp.ntiles.x * tmp.ntiles.y );
        device::memcpy_tohost( np, tmp.np, tmp.ntiles.x * tmp.ntiles.y );
        device::memcpy_tohost( offset, tmp.offset, tmp.ntiles.x * tmp.ntiles.y );
    
        for( int j = 0; j < tmp.ntiles.y; j++ ) {
            mpi::cout << '[' << std::setw(2) << j << ']';
            for( int i = 0; i < tmp.ntiles.x; i++ ) {
                mpi::cout << ' ' <<  std::setw(3) << offset[ j * tmp.ntiles.x + i ];
            }
            mpi::cout << std::endl;
        }

    }
*/
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
    const part::bnd_type local_bnd )
{
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
            int dx = i % 3 - 1;
            int dy = i / 3 - 1;

            int2 target = make_int2( tile_idx.x + dx, tile_idx.y + dy);
            int target_tid = part::tid_coords( target, ntiles, local_bnd );

            if ( target_tid >= 0 ) {
                // If valid neighbour tile reserve space on tmp. array
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
void copy_out( Particles & part, ParticleData & tmp, const ParticleSortData & sort, const part::bnd_type local_bnd )
{
    dim3 grid( tmp.ntiles.x, tmp.ntiles.y );
    auto block = opt_copy_out_block;

    kernel::copy_out <<< grid, block >>> ( part, tmp, sort, local_bnd );
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
 *        this step, with room for incoming MPI particles
 *
 * @param tmp       Temporary particle data
 */
void copy_in( Particles & part, ParticleData & tmp ) {
    dim3 grid( part.ntiles.x, part.ntiles.y );
    auto block = opt_copy_in_block;

    kernel::copy_in <<< grid, block >>> ( part, tmp );
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
void copy_sorted( Particles & part, ParticleData & tmp, const ParticleSortData & sort ) {


    std::cerr << "copy_sorted() is not implemented yet, aborting...\n";
    mpi::abort(1);

    // dim3 grid( part.ntiles.x, part.ntiles.y );
    // dim3 block( 1024 );
    // kernel::copy_sorted <<< grid, block >>> ( part, tmp, sort, ... );
}

#if 0
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
#endif

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

    /*
    auto local_np = np_local();
    mpi::cout << "(*info*) local_np = " << local_np << std::endl;
    */

    // Reset sort data
    sort.reset();

    // Get new number of particles per tile
    bnd_check ( *this, sort, local_bnd );

    // Exchange number of particles in edge cells
    sort.exchange_np();

    // Post particle data receives
    irecv_msg( sort, recv );

    // Get new offsets, including:
    // - Incoming particles from other MPI nodes
    // - New particles that will be injected (if any)
    auto total_np = update_tile_info ( tmp, sort, sort.recv.buffer, dev_tmp_uint32, extra );

    if ( total_np > max_part ) { 
        std::cerr << "Particles::tile_sort() - particle buffer requires growing,";
        std::cerr << " not implemented yet.";
        mpi::abort(1);
    }

    // Copy outgoing particles (and particles needing shifting) to staging area
    copy_out ( *this, tmp, sort, local_bnd );

    // Pack particle data and start sending
    isend_msg( tmp, sort, send );

    // Copy local particles from staging area into final positions in partile buffer
    copy_in ( *this, tmp );

    // Wait for receive messages to complete
    recv.wait();

    // Unpack received message data
    unpack_msg( sort, sort.recv.msg_np, sort.recv.buffer, recv );

    // Wait for sends to complete
    send.wait();

    // For debug only, remove from production code
    parallel.barrier();
    deviceCheck();
    validate( "after tile_sort" );
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
void Particles::validate( std::string msg, int const over ) {

    // Check offset / np buffer
    // For simplicity the test is done on the host
    int * h_np = host::malloc<int>( ntiles.x * ntiles.y );
    int * h_offset = host::malloc<int>( ntiles.x * ntiles.y );
    device::memcpy_tohost( h_np, np, ntiles.x * ntiles.y );
    device::memcpy_tohost( h_offset, offset, ntiles.x * ntiles.y );

    int err = 0;
    for( unsigned tile_id = 0; tile_id < ntiles.x * ntiles.y; ++tile_id ) {
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
    host::free( h_np );
    host::free( h_offset );

    if ( err ) {
        mpi::cout << "(*error*) Invalid tile information, aborting..." << std::endl;
        mpi::abort(1);
    }

    // Errors detected
    dev_tmp_uint32 = 0;

    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 32 );
    kernel::validate <<< grid, block >>> ( *this, over, dev_tmp_uint32.ptr() );

    if ( dev_tmp_uint32.get() ) {
        std::cerr << "(*error*) " << msg << " (np = " << np_local() << ")\n";
        ABORT( "invalid particle found, aborting..." );
    } else {
        mpi::cout << "(*info*) " << msg << " particles ok\n";
    }
}

#undef __ULIM

/**
 * @brief Prepare particle receive buffers and start receive
 * 
 * @param sort      Temporary sort index 
 * @param recv      Receive message object 
 */
void Particles::irecv_msg( ParticleSort &sort, ParticleMessage &recv ) {

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

namespace kernel {

    /**
     * @brief Pack send messages
     * 
     * @note This kernel should be launched with grid( msg_tiles )
     * 
     * @return __global__ 
     */
    __global__
    void pack_msg( ParticleData tmp, uint8_t * __restrict__ send_buffer, int * __restrict__ send_msg_np ) {

        ///@brief Message tile offset for each direction
        __shared__ unsigned int msg_tile_off[9];
        msg_tile_off[0] = 0;
        msg_tile_off[1] = 1;
        msg_tile_off[2] = 1 +   tmp.ntiles.x;
        msg_tile_off[3] = 2 +   tmp.ntiles.x;
        msg_tile_off[4] = 2 +   tmp.ntiles.x +   tmp.ntiles.y;
        msg_tile_off[5] = 2 +   tmp.ntiles.x +   tmp.ntiles.y;
        msg_tile_off[6] = 2 +   tmp.ntiles.x + 2*tmp.ntiles.y;
        msg_tile_off[7] = 3 +   tmp.ntiles.x + 2*tmp.ntiles.y;
        msg_tile_off[8] = 3 + 2*tmp.ntiles.x + 2*tmp.ntiles.y;

        ///@brief Number of particles per message (shared)
        __shared__ int msg_np[9];
        
        for( int i = block_thread_rank(); i < 9; i+= block_num_threads() ) {
            msg_np[i] = send_msg_np[i];
        }

        ///@brief Tile index in message buffer
        int msg_idx = blockIdx.x;

        block_sync();

        ///@brief message direction
        int dir = -1;
        for( int i = 0; i < 8; i++ ) {
            if ( i != 4 && msg_idx < msg_tile_off[i+1] ) { 
                dir = i; 
                break;
            }
        }
        if ( dir < 0 ) dir = 8;

        ///@brief Tile position in main particle buffer
        const int source_tid = tmp.ntiles.x * tmp.ntiles.y + msg_idx;

        ///@brief Offset to first particle in "communication" tile
        const auto offset = tmp.offset[ source_tid ];

        ///@brief Number of particles in "communication" tile
        const auto np     = tmp.np[ source_tid ];

        ///@brief Size of single particle data
        constexpr size_t particle_size = sizeof(int2) + sizeof(float2) + sizeof(float3);        

        ///@brief offset for this direction message data (number of particles)
        int dir_offset = 0;
        
        // This will always be very small (dir < 9), no point in parallelizing
        for( int i = 0; i < dir; i++ ) dir_offset += msg_np[ i ];

        // This should be equivalent
        // dir_offset = tmp.offset[ tmp.ntiles.x * tmp.ntiles.y + msg_tile_off[dir] ] -
        //             tmp.offset[ tmp.ntiles.x * tmp.ntiles.y ];

        ///@brief send message buffer for this direction
        uint8_t * msg_buffer = &send_buffer[ dir_offset * particle_size ];

        ///@brief offset inside message (number of particles)
        size_t msg_buffer_off = 0;
        
        for( int i = msg_tile_off[ dir ]; i < msg_idx; i++ ) {
            msg_buffer_off += tmp.np[ tmp.ntiles.x * tmp.ntiles.y + i];
        }
        
        // This should be equivalent
        // msg_buffer_off = offset - tmp.offset[ tmp.ntiles.x * tmp.ntiles.y + msg_tile_off[dir] ];

        // Note that due to alignment issues copying as int2/float2 may fail

        {   // Pack ix position data
            size_t pos = msg_buffer_off * sizeof(int2);
            int * const __restrict__ src = (int *) &tmp.ix[ offset ];
            int * const __restrict__ tgt = (int *) &msg_buffer[ pos ];
            for( int i = block_thread_rank(); i < 2 * np; i += block_num_threads() )
                tgt[i] = src[i];
        }

        {   // Pack x position data
            size_t pos = msg_np[dir] * sizeof( int2 ) + msg_buffer_off * sizeof(int2);
            float * const __restrict__ src = (float *) &tmp.x[ offset ];
            float * const __restrict__ tgt = (float *) &msg_buffer[ pos ];
            for( int i = block_thread_rank(); i < 2 * np; i += block_num_threads() )
                tgt[i] = src[i];
        }

        {   // Pack u momentum data
            size_t pos = msg_np[dir] * (sizeof( int2 )+sizeof( float2 )) + msg_buffer_off * sizeof(float3);
            float * const __restrict__ src = (float *) &tmp.u[ offset ];
            float * const __restrict__ tgt = (float *) &msg_buffer[ pos ];
            for( int i = block_thread_rank(); i < 3 * np; i += block_num_threads() )
                tgt[i] = src[i];
        }
    }
}

/**
 * @brief Pack particles moving out of the node into a message buffer and start send
 * 
 * @param tmp       Temporary buffer holding particles moving away from tiles
 * @param sort      Temporary sort index
 * @param send      Send message object
 */
void Particles::isend_msg( Particles &tmp, ParticleSort &sort, ParticleMessage &send ) {

    /// @brief Total number of particles being sent
    uint32_t send_np = 0;

    // Get offsets and check send buffer size
    for( int i = 0; i < 9; i++ ) {
        if (i != 4) {
            send.size[i] = sort.send.msg_np[i] * particle_size();
            send_np += sort.send.msg_np[i];
        } else {
            sort.send.msg_np[i] = 0;    // this should not be necessary
            send.size[i] = 0;
        }
    }

    // Check if msg buffer is large enough
    send.check_buffer( send_np * particle_size() );

/*
    // debug
    auto all_np = send_np;
    parallel.reduce( &all_np, 1, mpi::sum );
    if ( parallel.root() ) {
        std::cout << "total #particles in send message: " << all_np << '\n';
    }
*/
    // Pack data
    dim3 grid( part::msg_tiles( tmp.ntiles ) );
    dim3 block( 256 );
    kernel::pack_msg <<< grid, block >>> ( tmp, send.buffer, sort.send.msg_np );

    // Remove from production code
    deviceCheck();

    // Check send buffers
    uint32_t offset = 0;
    for( int i = 0; i < 9; i++ ) {
        if ( i != 4 && send.size[i] > 0 ) {
            int ierr_ix = 0;
            constexpr int maxerr = 3;

            int2   * ix = host::malloc<int2> ( sort.send.msg_np[i] );
            device::memcpy_tohost( ix, (int2 *) & send.buffer[ offset ], sort.send.msg_np[i] );

            /*
            mpi::cout << "send ix[] = ";
            for( int j = 0; j < sort.send.msg_np[i]; j++ ) {
                mpi::cout << ix[j] << " ";
            }
            mpi::cout << std::endl;
            */
            for( int j = 0; j < sort.send.msg_np[i]; j++ ) {
                if (ix[j].x < 0 || ix[j].x >= nx.x  ||
                    ix[j].y < 0 || ix[j].y >= nx.y){
                    if ( ierr_ix >= maxerr ) {
                        mpi::cout << "over " << maxerr << " send ix errors, skipping...\n";
                        break;
                    }
                    mpi::cout << "bad send message ix[" << j << "] = " << ix[j] << '\n';
                    ierr_ix++;
                }
            }
            host::free( ix );

            int ierr_x = 0;
            float2 * x  = host::malloc<float2> ( sort.send.msg_np[i] );
            device::memcpy_tohost( x, (float2 *) & send.buffer[ offset + sort.send.msg_np[i] * sizeof(int2)], sort.send.msg_np[i] );
            /*
            mpi::cout << "send x [] = ";
            for( int j = 0; j < sort.send.msg_np[i]; j++ ) {
                mpi::cout << x[j] << " ";
            }
            mpi::cout << std::endl;
            */
            for( int j = 0; j < sort.send.msg_np[i]; j++ ) {
                if (x[j].x < -0.5f || x[j].x >= +0.5f ||
                    x[j].y < -0.5f || x[j].y >= +0.5f ){
                    if ( ierr_x >= maxerr ) {
                        mpi::cout << "over " << maxerr << " send x errors, skipping...\n";
                        break;
                    }
                    mpi::cout << "bad send message x[" << j << "] = " << x[j] << '\n';
                    ierr_x++;
                }
            }
            host::free( x );

            if ( ierr_ix > 0 || ierr_x > 0 ) {
                mpi::cout << "bad particle data inside send message, aborting\n";
                exit(1);
            }

            /*
            float3 * u  = host::malloc<float3> ( sort.send.msg_np[i] );
            device::memcpy_tohost( u, (float3 *) & send.buffer[ offset + sort.send.msg_np[i] * (sizeof(int2) + sizeof(float2)) ], sort.send.msg_np[i] );

            mpi::cout << "send u [] = ";
            for( int j = 0; j < sort.send.msg_np[i]; j++ ) {
                mpi::cout << u[j] << " ";
            }
            mpi::cout << std::endl;
            */

            offset += send.size[i];
        }
    }

    // Start sending messages
    send.isend();
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
    void unpack_msg( ParticleData part, 
        uint8_t * __restrict__ recv_buffer, 
        int * __restrict__ recv_msg_np,
        int * __restrict__ recv_msg_tile_np ) {

        ///@brief Number of message tiles
        const int   msg_tiles = blockDim.x;

        ///@brief tile id in message array
        int msg_idx = blockIdx.x;

        const uint2 ntiles = part.ntiles;
        
        constexpr size_t particle_size = sizeof(int2) + sizeof(float2) + sizeof(float3);

        ///@brief Message tile offset for each direction
        __shared__ unsigned int msg_tile_off[9];
        msg_tile_off[0] = 0;
        msg_tile_off[1] = 1;
        msg_tile_off[2] = 1 +   ntiles.x;
        msg_tile_off[3] = 2 +   ntiles.x;
        msg_tile_off[4] = 2 +   ntiles.x +   ntiles.y;
        msg_tile_off[5] = 2 +   ntiles.x +   ntiles.y;
        msg_tile_off[6] = 2 +   ntiles.x + 2*ntiles.y;
        msg_tile_off[7] = 3 +   ntiles.x + 2*ntiles.y;
        msg_tile_off[8] = 3 + 2*ntiles.x + 2*ntiles.y;

        // Copy recv_msg_np and recv_msg_tile_np to shared memory using coalescent access
        __shared__ int msg_np[9];
        for( int i = block_thread_rank(); i < 9; i += block_num_threads() ) {
            msg_np[i] = recv_msg_np[i];
        }

        extern __shared__ int msg_tile_np[];
        for( int i = block_thread_rank(); i < msg_tiles; i += block_num_threads() ) {
            msg_tile_np[i] = recv_msg_tile_np[i];
        }

        block_sync();

        ///@brief number of particles received on this tile 
        int recv_np = msg_tile_np[msg_idx];

        // If any particles received in this tile
        if ( recv_np > 0 ) {
            ///@brief message direction
            int dir = -1;
            for( int i = 0; i < 8; i++ ) {
                if ( i != 4 && msg_idx < msg_tile_off[i+1] ) { 
                    dir = i; 
                    break;
                }
            }
            if ( dir < 0 ) dir = 8;

            ///@brief Tile index in message direction
            int dir_idx = msg_idx - msg_tile_off[ dir ];

            ///@brief Tile stride for storing received data according to direction
            int tile_stride = 1;
            if ( dir == 3 || dir == 5 ) tile_stride = ntiles.x;

            // Tile offset for storing received data according to direction
            int tile_offset;
            {
                int y = dir / 3;
                int x = dir % 3;
                int xoff = 0; int yoff = 0;
                if ( x == 2 ) xoff = ntiles.x-1;
                if ( y == 2 ) yoff = (ntiles.y-1) * ntiles.x;
                tile_offset = yoff + xoff;
            };

            ///@brief target tile on particle buffer
            int target_tid = dir_idx * tile_stride + tile_offset;

            // destination buffers (in main particle buffer)
            __shared__ int tgt_offset;
            if ( block_thread_rank() == 0 ) {
                tgt_offset = part.offset[ target_tid ] + 
                    device::atomic_fetch_add( &part.np[ target_tid ], recv_np );
            }
            block_sync();

            ///@brief offset for this direction message data (number of particles)
            int dir_offset = 0;
            
            // This will always be very small (dir < 9), no point in parallelizing
            for( int i = 0; i < dir; i++ ) dir_offset += msg_np[ i ];

            ///@brief receive message buffer for this direction
            uint8_t * msg_buffer = &recv_buffer[ dir_offset * particle_size ];

            ///@brief offset inside message (number of particles)
            size_t msg_buffer_off = 0;

            // A possible optimization is to do a block::exscan of msg_tile_np
            for( int i = msg_tile_off[ dir ]; i < msg_idx; i++ ) {
                msg_buffer_off += msg_tile_np[i];
            }

            // Note that due to alignment issues copying as int2/float2 may fail

            {   // Unpack ix position data
                size_t pos = msg_buffer_off * sizeof(int2);
                int * const __restrict__ src = (int *) &msg_buffer[ pos ];
                int * const __restrict__ tgt = (int *) &part.ix[ tgt_offset ];
                for( int i = block_thread_rank(); i < 2 * recv_np; i += block_num_threads() )
                    tgt[i] = src[i];
            }

            {   // Unpack x position data
                size_t pos = msg_np[dir] * sizeof( int2 ) + msg_buffer_off * sizeof(int2);
                float * const __restrict__ src = (float *) &msg_buffer[ pos ];
                float * const __restrict__ tgt = (float *) &part.x[ tgt_offset ];
                for( int i = block_thread_rank(); i < 2 * recv_np; i += block_num_threads() )
                    tgt[i] = src[i];
            }

            {   // Unpack u position data
                size_t pos = msg_np[dir] * (sizeof( int2 )+sizeof( float2 )) + msg_buffer_off * sizeof(float3);
                float * const __restrict__ src = (float *) &msg_buffer[ pos ];
                float * const __restrict__ tgt = (float *) &part.u[ tgt_offset ];
                for( int i = block_thread_rank(); i < 3 * recv_np; i += block_num_threads() )
                    tgt[i] = src[i];
            }

        }

    }
}

/**
 * @brief Unpack received particle data into main particle data buffer
 * 
 * @param sort                  Temporary sort index
 * @param recv_msg_np           Number of particles per message
 * @param recv_msg_tile_np      Number of particles per tile
 * @param recv                  Receive particle data message object
 */
void Particles::unpack_msg( ParticleSortData &sort, 
    int * __restrict__ recv_msg_np, int * __restrict__ recv_msg_tile_np,
    ParticleMessage &recv ) {

/*
    // debug
    int incoming = 0;
    for( int i = 0; i < 9; i++ ) {
        if ( i != 4 ) incoming += recv_msg_np[i];
    }
    mpi::cout << "Incoming particles: " << incoming << std::endl;
*/

    // Check receive buffers (remove from production code)
    uint32_t offset = 0;
    for( int i = 0; i < 9; i++ ) {
        if ( i != 4 && recv.size[i] > 0 ) {

            int ierr = 0;

            int2   * ix = host::malloc<int2> ( recv_msg_np[i] );
            device::memcpy_tohost( ix, (int2 *) & recv.buffer[ offset ], recv_msg_np[i] );
            /*
            mpi::cout << "recv ix[] = ";
            for( int j = 0; j < recv_msg_np[i]; j++ ) {
                mpi::cout << ix[j] << " ";
            }
            mpi::cout << std::endl;
            */
            for( int j = 0; j < recv_msg_np[i]; j++ ) {
                if (ix[j].x < 0 || ix[j].x >= nx.x  ||
                    ix[j].y < 0 || ix[j].y >= nx.y){
                    mpi::cout << "bad recv message ix[" << j << "] = " << ix[j] << '\n';
                    ierr++;
                }
            }
            host::free( ix );

            float2 * x  = host::malloc<float2> ( recv_msg_np[i] );
            device::memcpy_tohost( x, (float2 *) & recv.buffer[ offset + recv_msg_np[i] * sizeof(int2)], recv_msg_np[i] );
            
            /*
            mpi::cout << "recv x[] = ";
            for( int j = 0; j < recv_msg_np[i]; j++ ) {
                mpi::cout << x[j] << " ";
            }
            mpi::cout << std::endl;
            */
            for( int j = 0; j < recv_msg_np[i]; j++ ) {
                if (x[j].x < -0.5f || x[j].x >= +0.5f ||
                    x[j].y < -0.5f || x[j].y >= +0.5f ){
                    mpi::cout << "bad recv message x[" << j << "] = " << x[j] << '\n';
                    ierr++;
                }
            }
            host::free( x );

            if ( ierr > 0 ) {
                mpi::cout << "bad particle data inside recv message, aborting\n";
                exit(1);
            }

            offset += recv.size[i];
        }
    }

    // Unpack all data - multiple message tiles may write to the same local tile
    dim3 grid( part::msg_tiles( ntiles ) );
    dim3 block( 32 );

    kernel::unpack_msg <<< grid, block >>> ( 
        *this,                      // Particle data
        recv.buffer,                // Message receive buffer (all messages)
        recv_msg_np,                // Number of particles per message
        recv_msg_tile_np            // Number of particles per tile
    );

    // Remove from production code
    deviceCheck();
}
