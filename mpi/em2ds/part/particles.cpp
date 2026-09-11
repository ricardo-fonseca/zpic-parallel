#include "particles.hpp"

#include <iostream>
#include <sstream>
#include <string>
#include <cmath>
#include <vector>

/**
 * @brief Exchange number of particles in edge cells
 *
 */
void part::particle_sort::exchange_np() {

    const int2 nt = make_int2( ntiles.x, ntiles.y );

    // Post receives
    for( int dir = 0; dir < 9; dir++ ) {
        const int start = part::edge_tile_start( dir, nt );
        const int size  = part::edge_ntiles( dir, nt );

        if ( neighbor[dir] >= 0 ) {
            MPI_Irecv( &recv.buffer[start], size, MPI_INT, neighbor[dir],
                       source_tag(dir), comm, &recv.requests[dir] );
        } else {
            recv.requests[dir] = MPI_REQUEST_NULL;
        }
    }

    // Post sends
    for( int dir = 0; dir < 9; dir++ ) {
        const int start = part::edge_tile_start( dir, nt );
        const int size  = part::edge_ntiles( dir, nt );

        if ( neighbor[dir] >= 0 ) {
            MPI_Isend( &send.buffer[start], size, MPI_INT, neighbor[dir],
                       dest_tag(dir), comm, &send.requests[dir] );
        } else {
            send.requests[dir] = MPI_REQUEST_NULL;
        }
    }

    // Wait for receives to complete
    MPI_Waitall( 9, recv.requests, MPI_STATUSES_IGNORE );

    // update send.msg_np[] and recv.msg_np[]
    for( int dir = 0; dir < 9; dir++ ) {
        const int start = part::edge_tile_start( dir, nt );
        const int size  = part::edge_ntiles( dir, nt );

        int send_np = 0;
        int recv_np = 0;
        for( int k = 0; k < size; k++ ) {
            send_np += send.buffer[ start + k ];
            recv_np += recv.buffer[ start + k ];
        }
        send.msg_np[dir] = send_np;
        recv.msg_np[dir] = recv_np;
    }

    // Wait for sends to complete
    MPI_Waitall( 9, send.requests, MPI_STATUSES_IGNORE );
}

/**
 * @brief Gather particle data
 * 
 * @tparam quant    Quantity to gather
 * @param part      Particle data
 * @param d_data    Output data
 */
template < part::quantity quant >
void gather_quant( 
    part::particles_view src,
    float * const __restrict__ d_data )
{
    const int2 ntiles = make_int2( src.local_ntiles.x, src.local_ntiles.y );

    #pragma omp parallel for schedule(dynamic)
    for( auto tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

        auto tx = tid % ntiles.x;
        auto ty = tid / ntiles.x;

        // Global spatial offsets of local tile
        const int offx = (src.local_tile_start.x + tx) * src.tile_dims.x;
        const int offy = (src.local_tile_start.y + ty) * src.tile_dims.y;

        const auto offset = src.tile_offset[tid];
        const auto np     = src.tile_np[tid];

        int2   const * __restrict__ const ix = &src.ix[ offset ];
        float2 const * __restrict__ const x  = &src.x[ offset ];
        float3 const * __restrict__ const u  = &src.u[ offset ];
        
        for( int idx = 0; idx < np; idx ++ ) {
            float val;
            if constexpr( quant == part::quantity::x  ) val = ( offx + ix[idx].x ) + (0.5f + x[idx].x);
            if constexpr( quant == part::quantity::y  ) val = ( offy + ix[idx].y ) + (0.5f + x[idx].y);
            if constexpr( quant == part::quantity::ux ) val = u[idx].x;
            if constexpr( quant == part::quantity::uy ) val = u[idx].y;
            if constexpr( quant == part::quantity::uz ) val = u[idx].z;
            d_data[ offset + idx ] = val;
        }
    }
};

/**
 * @brief Gather data from a specific particle quantity in a device buffer
 * 
 * @param quant         Quantity to gather
 * @param d_data        Output data buffer, assumed to have size >= np
 */
void part::particles::gather( part::quantity quant, float * const d_data )
{
    
    // Gather data on device
    switch (quant) {
    case part::quantity ::x : 
        gather_quant<part::quantity ::x>( *this, d_data );
        break;
    case part::quantity ::y:
        gather_quant<part::quantity ::y>( *this, d_data );
        break;
    case part::quantity ::ux:
        gather_quant<part::quantity ::ux>( *this, d_data );
        break;
    case part::quantity ::uy:
        gather_quant<part::quantity ::uy>( *this, d_data );
        break;
    case part::quantity ::uz:
        gather_quant<part::quantity ::uz>( *this, d_data );
        break;
    }
}

/**
 * @brief Gather particle data, scaling values
 * 
 * @warning This expects the particle buffer to be compact, it will fail if 
 *          called with a non-compact buffer.
 *
 * @note Data (val) will be returned as `scale.x * val + scale.y`
 * 
 * @tparam quant    Quantity to gather
 * @param part      Particle data
 * @param scale     Scale factor for data
 * @param d_data    Scaled output data
 */
template < part::quantity quant >
void gather_quant( 
    part::particles_view src,
    const float2 scale, 
    float * const __restrict__ d_data )
{
    const int2 ntiles = make_int2( src.local_ntiles.x, src.local_ntiles.y );

    #pragma omp parallel for schedule(dynamic)
    for( auto tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

        auto tx = tid % ntiles.x;
        auto ty = tid / ntiles.x;

        // Global spatial offsets of local tile
        const int offx = (src.local_tile_start.x + tx) * src.tile_dims.x;
        const int offy = (src.local_tile_start.y + ty) * src.tile_dims.y;

        const auto offset = src.tile_offset[tid];
        const auto np     = src.tile_np[tid];

        int2   const * __restrict__ const ix = &src.ix[ offset ];
        float2 const * __restrict__ const x  = &src.x[ offset ];
        float3 const * __restrict__ const u  = &src.u[ offset ];
        
        for( int idx = 0; idx < np; idx ++ ) {
            float val;
            if constexpr ( quant == part::quantity::x )  val = ( offx + ix[idx].x) + (0.5f + x[idx].x);
            if constexpr ( quant == part::quantity::y )  val = ( offy + ix[idx].y) + (0.5f + x[idx].y);
            if constexpr ( quant == part::quantity::ux ) val = u[idx].x;
            if constexpr ( quant == part::quantity::uy ) val = u[idx].y;
            if constexpr ( quant == part::quantity::uz ) val = u[idx].z;
            d_data[ offset + idx ] = ops::fma( scale.x, val, scale.y );
        }
    }
};

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
    
    // Gather data on device
    switch (quant) {
    case part::quantity::x : 
        gather_quant<part::quantity::x> ( *this, scale, d_data );
        break; 
    case part::quantity::y: 
        gather_quant<part::quantity::y> ( *this, scale, d_data );
        break; 
    case part::quantity::ux: 
        gather_quant<part::quantity::ux>( *this, scale, d_data );
        break; 
    case part::quantity::uy: 
        gather_quant<part::quantity::uy>( *this, scale, d_data );
        break; 
    case part::quantity::uz: 
        gather_quant<part::quantity::uz>( *this, scale, d_data );
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
            //zdf::dataset dsets[ metadata.nquants ];
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
            float *data = memory::malloc<float>( local );

            // Get offsets - this avoids recalculating offsets for each quantity
            uint64_t file_off;
            MPI_Exscan( &local, &file_off, 1, MPI_UINT64_T, MPI_SUM, comm );
            if ( rank == 0 ) file_off = 0;

            // Local data chunk
            zdf::chunk chunk;
            chunk.count[0] = local;
            chunk.start[0] = file_off;
            chunk.stride[0] = 1;
            chunk.data = data;

            // Write the data
            for ( uint32_t i = 0; i < metadata.nquants; i ++) {
                gather( quants[i], data );
                zdf::write_cdset( part_file, dsets[i], chunk, file_off );
            }

            // Free temporary data
            memory::free( data );

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
    // ntiles needs to be set to signed because of the comparisons below
    const int2 ntiles = make_int2( data.local_ntiles.x, data.local_ntiles.y );
    const int2 lim = make_int2( data.tile_dims.x, data.tile_dims.y );

    #pragma omp parallel for schedule(dynamic)
    for( auto tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

        auto tx = tid % ntiles.x;
        auto ty = tid / ntiles.x;

        auto const np     = data.tile_np[ tid ];
        auto const offset = data.tile_offset[ tid ];

        int2 * __restrict__ ix    = &data.ix[ offset ];

        /// @brief Indices of particles leaving tile
        int  * __restrict__ idx   = &sort.idx[ offset ];

        /// @brief Number of particles moving in each direction
        int _npt[9];
        for( auto i = 0; i < 9; ++i ) _npt[i] = 0;
        
        /// @brief Number of particle leaving tile
        int _nout;
        _nout = 0;

        // sync

        // Count particles according to their motion
        // Store indices of particles leaving tile

        for( auto i = 0; i < np; ++i ) {
            int2 ipos = ix[i];
            int xcross = ( ipos.x >= lim.x ) - ( ipos.x < 0 );
            int ycross = ( ipos.y >= lim.y ) - ( ipos.y < 0 );
            
            if ( xcross || ycross ) {
                _npt[ (ycross+1) * 3 + (xcross+1) ] += 1;
                idx[ _nout ] = i; _nout += 1;
            }
        }

        // sync

        // only one thread per tile does this
        {
            // particles remaining on the tile
            _npt[4] = np - _nout;
        }

        // sync

        for( int i =0; i < 9; ++i ) {
            
            // Find target node
            int2 target = make_int2( tx + i % 3 - 1, ty + i / 3 - 1 );

            int target_tid = part::tid_coords( target, ntiles, local_bnd );
            
            if ( target_tid >= 0 && _npt[i] > 0 ) {
                #pragma omp atomic
                sort.new_np[ target_tid ] += _npt[i];
            }
        }

        {   // only one thread per tile does this
            int  * __restrict__ npt   = &sort.npt[ 9*tid ];

            for( int i = 0; i < 9; i++ ) npt[ i ] = _npt[i];
            sort.nidx[ tid ] = _nout;
        }
    }
}

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
    part::particles_view & tmp, 
    part::particle_sort & sort,
    const int * __restrict__ extra = nullptr ) {

    const int * __restrict__ recv_buffer = sort.recv.buffer;
    const int * __restrict__ new_np = sort.new_np;

    // Include ghost tiles in calculations
    const auto ntiles     = part::local_tiles( tmp.local_ntiles );
    const auto ntiles_all = part::all_tiles( tmp.local_ntiles );

    int * __restrict__ offset = tmp.tile_offset;
    int * __restrict__ np     = tmp.tile_np;

    // Initialize offset[] with the new number of particles
    if ( extra != nullptr ) {
        // extra array only includes data for local tiles
        for( auto i = 0; i < ntiles; i++ ) {
            offset[i] = new_np[i] + extra[i];
            np[i] = 0;
        }

        for( auto i = ntiles; i < ntiles_all; i++ ) {
            offset[i] = new_np[i];
            np[i] = 0;
        }
    } else {
        for( auto i = 0; i < ntiles_all; i++ ) {
            offset[i] = new_np[i];
            np[i] = 0;
        }
    }

    // Add incoming particles
    const int2 nt = make_int2(tmp.local_ntiles.x, tmp.local_ntiles.y);

    for( int dir = 0; dir < 9; dir++ ) {
        const int start = part::edge_tile_start( dir, nt );

        for( unsigned k = 0; k < part::edge_ntiles( dir, nt); k++ ) {
            offset[ part::local_edge_tid( dir, k, nt ) ] += recv_buffer[ start + k ];
        }
    }

    // Exclusive scan
    uint32_t total = 0;
    for( auto i = 0; i < ntiles_all; i++ ) {
        uint32_t tmp = offset[i];
        offset[i] = total;
        total += tmp;
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
 * @param part      Particle data
 * @param tmp       Temporary particle buffer (has new offsets)
 * @param sort      Sort data (new number of particles per tile, indices of
 *                  particles leaving the tile, etc.)
 * @param periodic  Correct for periodic boundaries
 */
void copy_out( 
    part::particles_view data, part::particles_view tmp, const part::particle_sort_view sort,
    const part::bnd_type local_bnd )
{
    const int2 ntiles = make_int2( data.local_ntiles.x, data.local_ntiles.y );
    const int2 lim = make_int2( data.tile_dims.x, data.tile_dims.y );

    #pragma omp parallel for schedule(dynamic)
    for( auto tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

        auto tx = tid % ntiles.x;
        auto ty = tid / ntiles.x;

        int const old_offset      = data.tile_offset[ tid ];
        int * __restrict__ npt    = &sort.npt[ 9*tid ];

        int2   * __restrict__ ix  = &data.ix[ old_offset ];
        float2 * __restrict__ x   = &data.x[ old_offset ];
        float3 * __restrict__ u   = &data.u[ old_offset ];

        int * __restrict__ idx    = &sort.idx[ old_offset ];
        uint32_t const nidx       = sort.nidx[ tid ];

        int const new_offset = tmp.tile_offset[ tid ];
        int const new_np     = sort.new_np[ tid ];
        
        int _dir_offset[9];

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
        _dir_offset[4] = new_offset + omp::atomic_fetch_add( & tmp.tile_np[ tid ], nshift );

        // Find offsets on new buffer
        for( int i = 0; i < 9; i++ ) {
            
            if ( i != 4 ) {
                // Find target node
                int dx = i % 3 - 1;
                int dy = i / 3 - 1;

                int2 target = make_int2( tx + dx, ty + dy);

                int target_tid = part::tid_coords( target, ntiles, local_bnd );
                
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
        for( unsigned i = 0; i < nidx; i++ ) {
            
            int k = idx[i];

            int2 nix  = ix[k];
            float2 nx = x[k];
            float3 nu = u[k];

            int xcross = ( nix.x >= lim.x ) - ( nix.x < 0 );
            int ycross = ( nix.y >= lim.y ) - ( nix.y < 0 );

            const int dir = (ycross+1) * 3 + (xcross+1);

            // Check if particle crossed into a valid neighbor
            if ( _dir_offset[dir] >= 0 ) {        

                int l = _dir_offset[dir]; _dir_offset[dir] += 1;

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
                    c = _c; _c += 1;
                    invalid = ( ix[c].x < 0 ) || ( ix[c].x >= lim.x ) || 
                              ( ix[c].y < 0 ) || ( ix[c].y >= lim.y );
                } while (invalid);

                ix[ k ] = ix[ c ];
                x [ k ] = x [ c ];
                u [ k ] = u [ c ];
            }
        }

        // At this point all particles up to n0 are correct


        // Copy particles that need to be shifted
        // We've reserved space for nshift particles earlier
        const int new_idx = _dir_offset[4];

        if ( new_offset >= old_offset ) {
            // Copy from begining of buffer
            for( int i = 0; i < nshift; i++ ) {
                tmp_ix[ new_idx + i ] = ix[ i ];
                tmp_x[ new_idx + i ]  = x [ i ];
                tmp_u[ new_idx + i ]  = u [ i ];
            }

        } else {

            // Copy from end of buffer
            const int old_idx = n0 - nshift;
            for( int i = 0; i < nshift; i++ ) {
                tmp_ix[ new_idx + i ] = ix[ old_idx + i ];
                tmp_x[ new_idx + i ]  = x [ old_idx + i ];
                tmp_u[ new_idx + i ]  = u [ old_idx + i ];
            }
        }

        // Store current number of local particles
        // These are already in the correct position in global buffer
        data.tile_np[ tid ] = n0 - nshift;

    }
}

/**
 * @brief Copy incoming particles to main buffer. Buffer will be fully sorted after
 *        this step
 * 
 * @param part      Main particle data
 * @param tmp       Temporary particle data
 */
void copy_in( part::particles_view data, part::particles_view tmp )
{
    const int2 ntiles = make_int2( data.local_ntiles.x, data.local_ntiles.y );

    #pragma omp parallel for schedule(dynamic)
    for( auto tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

        const int old_offset       =  data.tile_offset[ tid ];
        const int old_np           =  data.tile_np[ tid ];

        const int new_offset       =  tmp.tile_offset[ tid ];
        const int tmp_np           =  tmp.tile_np[ tid ];

        // Notice that we are already working with the new offset
        int2   * __restrict__ ix  = &data.ix[ new_offset ];
        float2 * __restrict__ x   = &data.x [ new_offset ];
        float3 * __restrict__ u   = &data.u [ new_offset ];

        int2   * __restrict__ tmp_ix = &tmp.ix[ new_offset ];
        float2 * __restrict__ tmp_x  = &tmp.x [ new_offset ];
        float3 * __restrict__ tmp_u  = &tmp.u [ new_offset ];

        if ( new_offset >= old_offset ) {

            // Add particles to the end of the buffer
            for( int i = 0; i < tmp_np; i++ ) {
                ix[ old_np + i ] = tmp_ix[ i ];
                x[ old_np + i ]  = tmp_x[ i ];
                u[ old_np + i ]  = tmp_u[ i ];
            }

        } else {

            // Add particles to the beggining of buffer
            int np0 = old_offset - new_offset;
            if ( np0 > tmp_np ) np0 = tmp_np;
            
            for( int i = 0; i < np0; i ++ ) {
                ix[ i ] = tmp_ix[ i ];
                x[ i ]  = tmp_x[ i ];
                u[ i ]  = tmp_u[ i ];
            }

            // If any particles left, add particles to the end of the buffer
            for( int i = np0; i < tmp_np; i ++ ) {
                ix[ old_np + i ] = tmp_ix[ i ];
                x[ old_np + i ]  = tmp_x[ i ];
                u[ old_np + i ]  = tmp_u[ i ];
            }
        }

        // Store the new offset and number of particles
        data.tile_np[ tid ]     = old_np + tmp_np;
        data.tile_offset[ tid ] = new_offset;
    }
}


/**
 * @brief Copies copy all particles to correct tiles in another buffer
 * 
 * @note Requires that new buffer (`tmp`) already has the correct offset
 *       values, and number of particles set to 0.
 * 
 * @param part      Particle data
 * @param tmp       Temporary particle buffer (has new offsets)
 * @param sort      Sort data (indices of particles leaving the tile, etc.)
 * @param periodic  Correct for periodic boundaries
 */
void copy_sorted( 
    part::particles_view data, part::particles_view tmp, const part::particle_sort_view sort,
    const part::bnd_type local_bnd )
{
    // Copy all particles to correct tile in tmp buffer
    const int2 ntiles = make_int2( data.local_ntiles.x, data.local_ntiles.y );
    const int2 lim = make_int2( data.tile_dims.x, data.tile_dims.y );

    for( int ty = 0; ty < ntiles.y; ++ty ) {
        for( int tx = 0; tx < ntiles.x; ++tx ) {

            int const tid = ty * ntiles.x + tx;
    
            int const old_offset      = data.tile_offset[ tid ];
            int * __restrict__ npt    = &sort.npt[ 9*tid ];

            int2   * __restrict__ ix  = &data.ix[ old_offset ];
            float2 * __restrict__ x   = &data.x[ old_offset ];
            float3 * __restrict__ u   = &data.u[ old_offset ];

            int * __restrict__ idx    = &sort.idx[ old_offset ];
            uint32_t const nidx       = sort.nidx[ tid ];
            
            int _dir_offset[9];

            // The _dir_offset variables hold the offset for each of the 9 target
            // tiles so the tmp_* variables just point to the beggining of the buffers
            int2* __restrict__  tmp_ix  = tmp.ix;
            float2* __restrict__ tmp_x  = tmp.x;
            float3* __restrict__ tmp_u  = tmp.u;

            // sync

            // Find offsets on new buffer
            for( int i = 0; i < 9; i++ ) {
                
                // Find target node
                int2 target = make_int2( tx + i % 3 - 1, ty + i / 3 - 1 );

                int target_tid = part::tid_coords( target, ntiles, local_bnd );

                if ( target_tid >= 0 ) {
                    // If valid neighbour tile reserve space on tmp. array
 
                    // _dir_offset[i] = atomicAdd( & tmp_tiles.offset2[ tid2 ], npt[ i ] );
                    _dir_offset[i] = tmp.tile_offset[ target_tid ] + tmp.tile_np[ target_tid ]; tmp.tile_np[ target_tid ] += npt[ i ];

                } else {
                    // Otherwise mark offset as invalid
                    _dir_offset[i] = -1;
                }
            }

            const int n0 = npt[4];
            int _c; _c = n0;

            // sync

            // Copy particles moving away from tile and fill holes
            for( int i = 0; i < nidx; i++ ) {
                
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
                    int l = _dir_offset[dir]; _dir_offset[dir] += 1;

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
                        c = _c; _c += 1;
                        invalid = ( ix[c].x < 0 ) || ( ix[c].x >= lim.x ) || 
                                  ( ix[c].y < 0 ) || ( ix[c].y >= lim.y );
                    } while (invalid);

                    ix[ k ] = ix[ c ];
                    x [ k ] = x [ c ];
                    u [ k ] = u [ c ];
                }
            }

            // sync

            // Copy particles staying in tile
            const int start = _dir_offset[4];

            for( int i = 0; i < n0; i ++ ) {
                tmp_ix[ start + i ] = ix[i];
                tmp_x [ start + i ] = x[i];
                tmp_u [ start + i ] = u[i];
            }
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
    bnd_check ( *this, sort, local_bnd );

    // Exchange number of particles in edge cells
    sort.exchange_np();

    // Post particle data receives
    irecv_msg( sort, recv );

    // Get new offsets, including:
    // - Incoming particles from other MPI nodes
    // - New particles that will be injected (if any)
    auto total_np = update_tile_info ( tmp, sort, extra );

    if ( total_np > max_part ) { 
        std::ostringstream msg;
        msg << "particles::tile_sort() - particle buffer requires growing,"
                  << "max_part: " << max_part << ", total_np: " << total_np
                  << ", not implemented yet.";
        mpi::fatal(msg.str());
    }

    // Copy outgoing particles (and particles needing shifting) to staging area
    copy_out ( *this, tmp, sort, local_bnd );

    // Pack particle data and start sending
    isend_msg( tmp, sort, send );

    // Copy local particles from staging area into final positions in partile buffer
    copy_in ( *this, tmp );

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

    // Loop over tiles
    #pragma omp parallel for schedule(dynamic)
    for( unsigned tid = 0; tid < local_ntiles.y * local_ntiles.x; tid++ ) {
        const auto offset = tile_offset[ tid ];
        const auto np  = tile_np[ tid ];

        int2 * const __restrict__ t_ix = &ix[ offset ];

        for( int i = 0; i < np; i++ ) {
            int2 cell = t_ix[i];
            cell.x += shift.x;
            cell.y += shift.y;
            t_ix[i] = cell;
        }
    }
}


#if 1

/**
 * @brief Maximum allowed value for u
 * 
 */
#define __ULIM std::numeric_limits<float>::max()

/**
 * @brief Checks particle buffer data for error
 * 
 * @warning This routine is meant for debug only and should not be called 
 *          for production code.
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
void part::particles::validate( std::string msg, int const over ) {

    if ( msg.empty() ) {
        mpi::cout << "validating particle set...";
    } else {
        mpi::cout << "validating particle set (" << msg << ")...";
    }

    uint32_t err = 0;
    int2 const lb = make_int2( -over, -over );
    int2 const ub = make_int2( tile_dims.x + over, tile_dims.y + over ); 

    // Check offset / np buffer
    for( unsigned tile_id = 0; tile_id < local_ntiles.x * local_ntiles.y; ++tile_id ) {
        if ( tile_np[tile_id] < 0 ) {
            mpi::cout << "\n tile[" << tile_id << "] - bad np (" << tile_np[ tile_id ] << "), should be >= 0";
            err = 1;
        }

        if ( tile_id > 0 ) {
            auto prev = tile_offset[ tile_id-1] + tile_np[ tile_id-1];
            if ( prev != tile_offset[ tile_id ] ) {
                mpi::cout << "\n tile[" << tile_id << "] - bad offset (" << tile_offset[ tile_id ] << ")"
                          << ", does not match previous tile info, should be " << prev;
                err = 1;
            }
        } else {
            if ( tile_offset[ tile_id ] != 0 ) {
                mpi::cout << "tile[" << tile_id << "] - bad offset (" << tile_offset[ tile_id ] << "), should be 0";
                err = 1;
            }
        }   
    }

    if ( err ) {
        mpi::fatal("\nInvalid tile information");
    }

    // Loop over tiles
    for( unsigned tile_id = 0; tile_id < local_ntiles.x * local_ntiles.y; ++tile_id ) {
        const auto offset = tile_offset[ tile_id ];
        const auto np  = tile_np[ tile_id ];

        int2   * const __restrict__ t_ix = &ix[ offset ];
        float2 * const __restrict__ t_x  = &x [ offset ];
        float3 * const __restrict__ t_u  = &u [ offset ];

        for( int i = 0; i < np; i++ ) {
            if ((t_ix[i].x < lb.x) || (t_ix[i].x >= ub.x )) { 
                mpi::cout << "\ntile[" << tile_id << "] Invalid ix[" << i << "].x position (" << t_ix[i].x << ")"
                          << ", range = [" << lb.x << "," << ub.x << "]";
                err=1; break;
            }
            if ((t_ix[i].y < lb.y) || (t_ix[i].y >= ub.y )) { 
                mpi::cout << "\ntile[" << tile_id << "] Invalid ix[" << i << "].y position (" << t_ix[i].y << ")"
                          << ", range = [" << lb.y << "," << ub.y << "]";
                err=1; break;
            }
            if ( std::isnan(t_u[i].x) || std::isinf(t_u[i].x) || std::abs(t_u[i].x) >= __ULIM ) {
                mpi::cout << "\ntile[" << tile_id << "] Invalid u[" << i << "].x gen. velocity (" << t_u[i].x <<")";
                err=1; break;
            }
            if ( std::isnan(t_u[i].y) || std::isinf(t_u[i].y) || std::abs(t_u[i].y) >= __ULIM ) {
                mpi::cout << "\ntile[" << tile_id << "] Invalid u[" << i << "].y gen. velocity (" << t_u[i].y <<")";
                err=1; break;
            }
            if ( std::isnan(t_u[i].z) || std::isinf(t_u[i].z) || std::abs(t_u[i].z) >= __ULIM ) {
                mpi::cout << "\ntile[" << tile_id << "] Invalid u[" << i << "].z gen. velocity (" << t_u[i].z <<")";
                err=1; break;
            }
            if ( t_x[i].x < -0.5f || t_x[i].x >= 0.5f ) {
                mpi::cout << "\ntile[" << tile_id << "] Invalid x[" << i << "].x position (" 
                          << t_x[i].x << "), range = [-0.5,0.5[";
                err=1; break;
            }
            if ( t_x[i].y < -0.5f || t_x[i].y >= 0.5f ) {
                mpi::cout << "\ntile[" << tile_id << "] Invalid x[" << i << "].y position ("
                          << t_x[i].y << "), range = [-0.5,0.5[\n";
                err=1; break;
            }
        }
    }

    if ( err ) {
        mpi::fatal("Invalid particle(s) found" );
    } else {
        mpi::cout << " particle set ok.\n";
    }
}

#else

/**
 * @brief Validates particle data in buffer
 * 
 * Routine checks for valid positions (both cell index and cell position) and
 * for valid velocities
 * 
 * @param msg       Message to print in case error is found
 * @param over      Amount of extra cells indices beyond limit allowed. Used
 *                  when checking the buffer before tile_sort()
 */
void particles::validate( std::string msg, int const over ) {

    uint32_t nerr = 0;
    int2 const lb = make_int2( -over, -over );
    int2 const ub = make_int2( tiles.tile_dims.x + over, tiles.tile_dims.y + over );
    const uint2 ntiles  = tiles.ntiles;


    // Loop over tiles
    for( int ty = 0; ty < ntiles.y; ty ++ ) {
        for( int tx = 0; tx < ntiles.x; tx ++ ) {
            const auto tid  = ty * ntiles.x + tx;
            const auto tile_off = tiles.offset[ tid ];
            const auto tile_np  = tiles.np[ tid ];

            int2   * const __restrict__ ix = &data.ix[ tile_off ];
            float2 * const __restrict__ x  = &data.x[ tile_off ];
            float3 * const __restrict__ u  = &data.u[ tile_off ];

            for( int i = 0; i < tile_np; i++ ) {
                if ( (ix[i].x < lb.x) || (ix[i].x >= ub.x )) { nerr++; break; }
                if ( (ix[i].y < lb.y) || (ix[i].y >= ub.y )) { nerr++; break; }
                if ( std::isnan(u[i].x) || std::isinf(u[i].x) || std::abs(u[i].x) >= __ULIM ) { nerr++; break; }
                if ( std::isnan(u[i].y) || std::isinf(u[i].y) || std::abs(u[i].y) >= __ULIM ) { nerr++; break; }
                if ( std::isnan(u[i].z) || std::isinf(u[i].z) || std::abs(u[i].z) >= __ULIM ) { nerr++; break; }
                if ( x[i].x < -0.5f || x[i].x >= 0.5f ) { nerr++; break; }
                if ( x[i].y < -0.5f || x[i].y >= 0.5f ) { nerr++; break; }
            }
        }
    }

    if ( nerr > 0 ) {
        mpi::fatal( msg + ": invalid particle(s) found" );
    }
}

#endif

/**
 * @brief Prepare particle receive buffers and start receive
 * 
 * @param sort      Temporary sort index 
 * @param recv      Receive message object 
 */
void part::particles::irecv_msg( particle_sort &sort, particle_message &recv ) {

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
 * @brief Pack particles moving out of the node into a message buffer and start send
 * 
 * @param tmp       Temporary buffer holding particles moving away from tiles
 * @param sort      Temporary sort index
 * @param send      Send message object
 */
void part::particles::isend_msg( particles &tmp, particle_sort &sort, particle_message &send ) {

    /// @brief Total number of particles being sent
    uint32_t send_np = 0;
    /// @brief Offset in particle buffer for each message data
    uint32_t off[9];

    // Get offsets and check send buffer size
    for( int i = 0; i < 9; i++ ) {
        off[i] = send_np;
        if (i != 4) {
            send.size[i] = sort.send.msg_np[i] * particle_size();
            send_np += sort.send.msg_np[i];
        } else {
            sort.send.msg_np[i] = 0;    // this should not be necessary
            send.size[i] = 0;
        }
    }

    send.check_buffer( send_np * particle_size() );

    // Pack data

    // Offset to first "communication" tile
    const auto tile_off = tmp.tile_offset[ local_ntiles.x * local_ntiles.y ];

    int2   * const __restrict__ ix = &tmp.ix[ tile_off ];
    float2 * const __restrict__ x  = &tmp.x[ tile_off ];
    float3 * const __restrict__ u  = &tmp.u[ tile_off ];

    #pragma omp parallel for schedule(dynamic)
    for( int dir = 0; dir < 9; dir++) {
        if ( sort.send.msg_np[dir] > 0 ) {
            uint8_t * __restrict__ buffer = &send.buffer[ off[dir] * particle_size() ];
            
            uint32_t np = sort.send.msg_np[dir];
            const packed_offsets dst( np );

            std::memcpy( &buffer[ dst.ix ], &ix[ off[dir] ], np * sizeof(*ix) );
            std::memcpy( &buffer[ dst.x  ], &x [ off[dir] ], np * sizeof(*x) );
            std::memcpy( &buffer[ dst.u  ], &u [ off[dir] ], np * sizeof(*u) );
        }
    }

    // Start sending messages
    send.isend();
}

/**
 * @brief Unpack received particle data into main particle data buffer
 * 
 * @param sort      Temporary sort index
 * @param recv      Receive message object
 */
void part::particles::unpack_msg( particle_sort &sort, particle_message &recv ) {

    /// @brief number of particles per received tile
    int * __restrict__ msg_tile_np = sort.recv.buffer;

    // Unpack all data - multiple message tiles may write to the same local tile
    // This version of the unpack algorithm does not work in OpenMP parallel
    int recv_off = 0;

    const int2 nt = make_int2( local_ntiles.x, local_ntiles.y );

    // loop over messages
    for( int dir = 0; dir < 9; dir++ ) {

        ///@brief number of particles in this message
        const int msg_np = sort.recv.msg_np[dir];

        ///@brief first edge tile of this direction in the received count buffer
        const int start = part::edge_tile_start( dir, nt );

        ///@brief receive message buffer for this direction
        uint8_t * msg_buffer = & recv.buffer[ recv_off * particle_size() ];
        
        ///@brief byte offsets of each quantity block in this message
        const packed_offsets src( msg_np );

        ///@brief number of particles unpacked from this message
        int np_unpack = 0;

        for( unsigned k = 0; k < part::edge_ntiles(dir, nt); k++ ) {
            
            ///@brief number of particles received on this tile 
            const int recv_np  =  msg_tile_np[ start + k ];
            
            // If any particles received in that tile
            if ( recv_np > 0 ) {
                // Get target tile for msg data
                int target_tid = part::local_edge_tid( dir, k, nt );

                // Position in destination buffers
                int tgt_offset =  tile_offset[ target_tid ] + tile_np[ target_tid ];

                // Copy message data
                std::memcpy( &ix[ tgt_offset ],
                                &msg_buffer[ src.ix + np_unpack * sizeof(*ix) ],
                                recv_np * sizeof(*ix) );

                std::memcpy( &x[ tgt_offset ],
                                &msg_buffer[ src.x  + np_unpack * sizeof(*x) ],
                                recv_np * sizeof(*x) );

                std::memcpy( &u[ tgt_offset ],
                                &msg_buffer[ src.u  + np_unpack * sizeof(*u) ],
                                recv_np * sizeof(*u) );

                // Update number of messages in tile
                tile_np[ target_tid ] += recv_np;

                // Update number of unpacked particles
                np_unpack += recv_np;
            }
        }
        recv_off += msg_np;
    }
}
