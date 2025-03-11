#include "particles.h"

#include <iostream>
#include <string>
#include <cmath>

#include "timer.h"

/**
 * @brief Gather particle data
 * 
 * @tparam quant    Quantiy to gather
 * @param part      Particle data
 * @param d_data    Output data
 */
template < part::quant quant >
void gather_quant( 
    ParticleData part,
    float * const __restrict__ d_data,
    int * const __restrict__ d_off )
{
    const uint2 tile_nx = part.nx;
    const int2 ntiles = make_int2( part.ntiles.x, part.ntiles.y );

    #pragma omp parallel for schedule(dynamic)
    for( auto tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

        auto tx = tid % ntiles.x;
        auto ty = tid / ntiles.x;

        const auto offset = part.offset[tid];
        const auto np     = part.np[tid];
        const auto data_offset = d_off[tid];

        int2   const * __restrict__ const ix = &part.ix[ offset ];
        float2 const * __restrict__ const x  = &part.x[ offset ];
        float3 const * __restrict__ const u  = &part.u[ offset ];
        
        for( int idx = 0; idx < np; idx ++ ) {
            float val;
            if constexpr ( quant == part::x )  val = (tx * tile_nx.x + ix[idx].x) + (0.5f + x[idx].x);
            if constexpr ( quant == part::y )  val = (ty * tile_nx.y + ix[idx].y) + (0.5f + x[idx].y);
            if constexpr ( quant == part::ux ) val = u[idx].x;
            if constexpr ( quant == part::uy ) val = u[idx].y;
            if constexpr ( quant == part::uz ) val = u[idx].z;
            d_data[ data_offset + idx ] = val;
        }
    }
};

/**
 * @brief Gather data from a specific particle quantity in a device buffer
 * 
 * @param quant         Quantity to gather
 * @param d_data        Output data buffer, assumed to have size >= np
 */
void Particles::gather( part::quant quant, float * const d_data, int * d_off )
{
    // If d_off was not supplied use the same offset as particle data
    if ( d_off == nullptr ) d_off = offset;
    
    // Gather data on device
    switch (quant) {
    case part::x : 
        gather_quant<part::x>( *this, d_data, d_off );
        break;
    case part::y:
        gather_quant<part::y>( *this, d_data, d_off );
        break;
    case part::ux:
        gather_quant<part::ux>( *this, d_data, d_off );
        break;
    case part::uy:
        gather_quant<part::uy>( *this, d_data, d_off );
        break;
    case part::uz:
        gather_quant<part::uz>( *this, d_data, d_off );
        break;
    }
}

/**
 * @brief Gather particle data, scaling values
 * 
 * @note Data (val) will be returned as `scale.x * val + scale.y`
 * 
 * @tparam quant    Quantiy to gather
 * @param part      Particle data
 * @param d_data    Scaled output data
 * @param scale     Scale factor for data
 */
template < part::quant quant >
void gather_quant( 
    ParticleData part,
    const float2 scale, 
    float * const __restrict__ d_data,
    int * const __restrict__ d_off )
{
    const uint2 tile_nx = part.nx;
    const int2 ntiles = make_int2( part.ntiles.x, part.ntiles.y );

    #pragma omp parallel for schedule(dynamic)
    for( auto tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

        auto tx = tid % ntiles.x;
        auto ty = tid / ntiles.x;

        const auto offset = part.offset[tid];
        const auto np     = part.np[tid];

        int2   const * __restrict__ const ix = &part.ix[ offset ];
        float2 const * __restrict__ const x  = &part.x[ offset ];
        float3 const * __restrict__ const u  = &part.u[ offset ];
        
        for( int idx = 0; idx < np; idx ++ ) {
            float val;
            if constexpr ( quant == part::x )  val = (tx * tile_nx.x + ix[idx].x) + (0.5f + x[idx].x);
            if constexpr ( quant == part::y )  val = (ty * tile_nx.y + ix[idx].y) + (0.5f + x[idx].y);
            if constexpr ( quant == part::ux ) val = u[idx].x;
            if constexpr ( quant == part::uy ) val = u[idx].y;
            if constexpr ( quant == part::uz ) val = u[idx].z;
            d_data[ offset + idx ] = ops::fma( scale.x, val, scale.y );
        }
    }
};

/**
 * @brief Gather data from a specific particle quantity in a device buffer, scaling values
 * 
 * @param quant     Quantity to gather
 * @param d_data    Output data buffer, assumed to have size >= np
 * @param scale     Scale factor for data
 */
void Particles::gather( part::quant quant, const float2 scale, float * const __restrict__ d_data, int * const __restrict__ d_off )
{
    
    // Gather data on device
    switch (quant) {
    case part::x : 
        gather_quant<part::x> ( *this, scale, d_data, d_off );
        break; 
    case part::y: 
        gather_quant<part::y> ( *this, scale, d_data, d_off );
        break; 
    case part::ux: 
        gather_quant<part::ux>( *this, scale, d_data, d_off );
        break; 
    case part::uy: 
        gather_quant<part::uy>( *this, scale, d_data, d_off );
        break; 
    case part::uz: 
        gather_quant<part::uz>( *this, scale, d_data, d_off );
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
    uint64_t global;
    MPI_Allreduce( &local, &global, 1, MPI_UINT64_T, MPI_SUM, parallel.get_comm() );

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

            for( int i = 0; i < metadata.nquants; i++ ) {
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
            for ( int i = 0; i < metadata.nquants; i ++) {
                gather( quants[i], data, offset );
                zdf::write_cdset( part_file, dsets[i], chunk, file_off );
            }

            // Free temporary data
            memory::free( data );

            // close the datasets
            for( int i = 0; i < metadata.nquants; i++ ) 
                zdf::end_cdset( part_file, dsets[i] );

            // Close the file
            zdf::close_file( part_file );
        }
    } else {
        // No particles - root node creates an empty file
        if ( parallel.get_rank() == 0 ) {
            zdf::file part_file;
            zdf::open_part_file( part_file, metadata, iter, path+"/"+metadata.name );

            for ( int i = 0; i < metadata.nquants; i ++) {
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
    ParticleData part, ParticleSortData sort, const part::bnd_type local_bnd)
{
    // ntiles needs to be set to signed because of the comparisons below
    const int2 ntiles = make_int2( part.ntiles.x, part.ntiles.y );
    const int2 lim = make_int2( part.nx.x, part.nx.y );

    #pragma omp parallel for schedule(dynamic)
    for( auto tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

        auto tx = tid % ntiles.x;
        auto ty = tid / ntiles.x;

        auto const np     = part.np[ tid ];
        auto const offset = part.offset[ tid ];

        int2 * __restrict__ ix    = &part.ix[ offset ];

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
            // Particles remaining on the tile
            _npt[4] = np - _nout;
        }

        // sync

        for( int i =0; i < 9; ++i ) {
            
            // Find target node
            int2 target = make_int2( tx + i % 3 - 1, ty + i / 3 - 1 );

            int target_tid = part::tid_coords( target, ntiles, local_bnd );
            
            if ( target_tid >= 0 ) {
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
 * @brief Recalculates particle tile offset
 * 
 * @note The number of particles in each tile is set to 0
 * 
 * @param tmp           (out) Particle buffer
 * @param new_np        (in)  New number of particles per tile.
 * @return uint32_t     (out) Total number of particles
 */
uint32_t update_tile_info( ParticleData tmp, const int * __restrict__ new_np ) {

    // Include ghost tiles in calculations
    const auto ntiles = part::all_tiles( tmp.ntiles );

    int * __restrict__ offset = tmp.offset;
    int * __restrict__ np     = tmp.np;

    uint32_t total = 0;
    for( unsigned i = 0; i < ntiles; i++ ) {
        offset[i] = total;
        np[i] = 0;
        total += new_np[i];
    }

    return total;
}

/**
 * @brief Recalculates particle tile offset, leaving room for additional particles
 * 
 * @note The number of particles in each tile is set to 0
 * 
 * @param tmp           (out) Particle buffer
 * @param new_np        (in/out) New number of particles per tile. Set to 0 after calculation.
 * @param extra         (in) Additional incoming particles
 * @return uint32_t     (out) Total number of particles (including additional ones)
 */
uint32_t update_tile_info( ParticleData tmp, const int * __restrict__ new_np, 
    const int * __restrict__ extra ) {

    // Include ghost tiles in calculations
    const auto ntiles = part::all_tiles( tmp.ntiles );

    int * __restrict__ offset = tmp.offset;
    int * __restrict__ np     = tmp.np;

    uint32_t total = 0;
    for( unsigned i = 0; i < ntiles; i++ ) {
        offset[i] = total;
        np[i] = 0;
        total += new_np[i] + extra[i];
    }

    return total;
}

/**
 * @brief Copy outgoing particles to temporary buffer
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
void copy_out( 
    ParticleData part, ParticleData tmp, const ParticleSortData sort,
    const part::bnd_type local_bnd )
{
    const int2 ntiles = make_int2( part.ntiles.x, part.ntiles.y );
    const int2 lim = make_int2( part.nx.x, part.nx.y );

    #pragma omp parallel for schedule(dynamic)
    for( auto tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

        auto tx = tid % ntiles.x;
        auto ty = tid / ntiles.x;

        int const old_offset      = part.offset[ tid ];
        int * __restrict__ npt    = &sort.npt[ 9*tid ];

        int2   * __restrict__ ix  = &part.ix[ old_offset ];
        float2 * __restrict__ x   = &part.x[ old_offset ];
        float3 * __restrict__ u   = &part.u[ old_offset ];

        int * __restrict__ idx    = &sort.idx[ old_offset ];
        uint32_t const nidx       = sort.nidx[ tid ];

        int const new_offset = tmp.offset[ tid ];
        int const new_np     = sort.new_np[ tid ];
        
        int _dir_offset[9];

        // The _dir_offset variables hold the offset for each of the 9 target
        // tiles so the tmp_* variables just point to the beginning of the buffers
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
        _dir_offset[4] = new_offset + omp::atomic_fetch_add( & tmp.np[ tid ], nshift );

        // Find offsets on new buffer
        for( int i = 0; i < 9; i++ ) {
            
            if ( i != 4 ) {
                // Find target node
                int2 target = make_int2( tx + i % 3 - 1, ty + i / 3 - 1 );

                int target_tid = part::tid_coords( target, ntiles, local_bnd );
                
                if ( target_tid >= 0 ) {
                    // If valid neighbour tile reserve space on tmp. array
                    _dir_offset[i] = tmp.offset[ target_tid ] + 
                        omp::atomic_fetch_add( &tmp.np[ target_tid ], npt[ i ] );
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

                    invalid = ( ix[c].x < 0 ) || ( ix[c].x >= lim.x) || 
                                ( ix[c].y < 0 ) || ( ix[c].y >= lim.y);
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
            // Copy from beggining of buffer
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
        part.np[ tid ] = n0 - nshift;

    }
}

/**
 * @brief Copy incoming particles to main buffer. Buffer will be fully sorted after
 *        this step
 * 
 * @param part      Main particle data
 * @param tmp       Temporary particle data
 */
void copy_in( ParticleData part, ParticleData tmp )
{
    const int2 ntiles = make_int2( part.ntiles.x, part.ntiles.y );

    #pragma omp parallel for schedule(dynamic)
    for( auto tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

        const int old_offset       =  part.offset[ tid ];
        const int old_np           =  part.np[ tid ];

        const int new_offset       =  tmp.offset[ tid ];
        const int tmp_np           =  tmp.np[ tid ];

        // Notice that we are already working with the new offset
        int2   * __restrict__ ix  = &part.ix[ new_offset ];
        float2 * __restrict__ x   = &part.x [ new_offset ];
        float3 * __restrict__ u   = &part.u [ new_offset ];

        int2   * __restrict__ tmp_ix = &tmp.ix[ new_offset ];
        float2 * __restrict__ tmp_x  = &tmp.x [ new_offset ];
        float3 * __restrict__ tmp_u  = &tmp.u [ new_offset ];

        if ( new_offset >= old_offset ) {

            // Add particles to the end of the buffer
            for( int i = 0; i < tmp_np; i ++ ) {
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
        part.np[ tid ]     = old_np + tmp_np;
        part.offset[ tid ] = new_offset;
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
    ParticleData part, ParticleData tmp, const ParticleSortData sort,
    const part::bnd_type local_bnd )
{
    // Copy all particles to correct tile in tmp buffer
    const int2 ntiles = make_int2( part.ntiles.x, part.ntiles.y );
    const int2 lim = make_int2( part.nx.x, part.nx.y );

    for( int ty = 0; ty < ntiles.y; ++ty ) {
        for( int tx = 0; tx < ntiles.x; ++tx ) {

            int const tid = ty * ntiles.x + tx;
    
            int const old_offset      = part.offset[ tid ];
            int * __restrict__ npt    = &sort.npt[ 9*tid ];

            int2   * __restrict__ ix  = &part.ix[ old_offset ];
            float2 * __restrict__ x   = &part.x[ old_offset ];
            float3 * __restrict__ u   = &part.u[ old_offset ];

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
                    _dir_offset[i] = tmp.offset[ target_tid ] + tmp.np[ target_tid ]; tmp.np[ target_tid ] += npt[ i ];

                } else {
                    // Otherwise mark offset as invalid
                    _dir_offset[i] = -1;
                }
            }

            const int n0 = npt[4];
            int _c; _c = n0;

            // sync

            // Copy particles moving away from tile and fill holes
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

                    // _dir_offset[] includes the offset in the global tmp particle buffer

                    // int l = atomicAdd( & _dir_offset[dir], 1 );
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
                        // c = atomicAdd( &_c, 1 );
                        c = _c; _c += 1;

                        invalid = ( ix[c].x < 0 ) || ( ix[c].x >= lim.x) || 
                                  ( ix[c].y < 0 ) || ( ix[c].y >= lim.y);
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
    bnd_check ( *this, sort, periodic );

    if ( extra ) {
        // Get new offsets, including extra values in offset calculations
        // Used to reserve space in particle buffer for later injection
        auto total_np = update_tile_info ( tmp, sort.new_np, extra );

        if ( total_np > max_part ) { 

            // grow tmp particle buffer
            tmp.grow_buffer( total_np );

            // copy all particles to correct tiles in tmp buffer
            copy_sorted( *this, tmp, sort, periodic );

            // swap buffers
            swap_buffers( *this, tmp );

            // grow tmp particle buffer for future use
            grow_buffer( max_part );

        } else {
            // Copy outgoing particles (and particles needing shifting) to staging area
            copy_out ( *this, tmp, sort, periodic );

            // Copy particles from staging area into final positions in partile buffer
            copy_in ( *this, tmp );
        }

    } else {
        // Get new offsets
        update_tile_info ( tmp, sort.new_np );

        // Copy outgoing particles (and particles needing shifting) to staging area
        copy_out ( *this, tmp, sort, periodic );

        // Copy particles from staging area into final positions in partile buffer
        copy_in ( *this, tmp );
    }


    // For debug only, remove from production code
    // validate( "After tile_sort" );
}
#endif

#if 1
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
    bnd_check ( *this, sort, local_bnd );

    // Exchange number of particles in edge cells
    sort.exchange_np();

    // Post particle data receives
    recv.irecv();

    // Get new offsets, including extra values in offset calculations
    // Used to reserve space in particle buffer for later injection
    auto total_np = update_tile_info ( tmp, sort.new_np, extra );

    if ( total_np > max_part ) { 
        std::cerr << "Particles::tile_sort() - particle buffer requires growing,";
        std::cerr << " not implemented yet.";
        exit(1);
    }

    // Copy outgoing particles (and particles needing shifting) to staging area
    copy_out ( *this, tmp, sort, local_bnd );

    // Pack particle data and start sending
    pack_msg( tmp, sort, send );
    send.isend();

    // Copy local particles from staging area into final positions in partile buffer
    copy_in ( *this, tmp );

    // Wait for messages to be received and unpack data
    recv.wait();
    unpack_msg( sort, recv );

    // Wait for sends to complete
    recv.wait();

    // For debug only, remove from production code
    // validate( "After tile_sort" );
}
#endif

/**
 * @brief Shifts particle cells by the required amount
 * 
 * Cells are shited by adding the parameter `shift` to the particle cell
 * indexes.
 * 
 * Note that this routine does not check if the particles are still inside the
 * tile.
 * 
 * @param shift     Cell shift in both directions
 */
void Particles::cell_shift( int2 const shift ) {

    // Loop over tiles
    for( unsigned ty = 0; ty < ntiles.y; ++ty ) {
        for( unsigned tx = 0; tx < ntiles.x; ++tx ) {
            const auto tid  = ty * ntiles.x + tx;
            const auto tile_off = offset[ tid ];
            const auto tile_np  = np[ tid ];

            int2 * const __restrict__ t_ix = &ix[ tile_off ];

            for( int i = 0; i < tile_np; i++ ) {
                int2 cell = t_ix[i];
                cell.x += shift.x;
                cell.y += shift.y;
                t_ix[i] = cell;
            }
        }
    }
}

#define __ULIM __FLT_MAX__


#if 1

/**
 * @brief Checks particle buffer data for error
 * 
 * WARNING: This routine is meant for debug only and should not be called 
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
void Particles::validate( std::string msg, int const over ) {

    std::cout << "Validating particle set, " << msg << "..." << std::endl;

    uint32_t err = 0;
    int2 const lb = make_int2( -over, -over );
    int2 const ub = make_int2( nx.x + over, nx.y + over ); 

    // Check offset / np buffer
    for( unsigned tile_id = 0; tile_id < ntiles.x * ntiles.y; ++tile_id ) {
        if ( np[tile_id] < 0 ) {
            std::cerr << "tile[" << tile_id << "] - bad np (" << np[ tile_id ] << "), should be >= 0\n";
            err = 1;
        }

        if ( tile_id > 0 ) {
            auto prev = offset[ tile_id-1] + np[ tile_id-1];
            if ( prev != offset[ tile_id ] ) {
                std::cerr << "tile[" << tile_id << "] - bad offset (" << offset[ tile_id ] << ")"
                    << ", does not match previous tile info, should be " << prev << '\n';
                err = 1;
            }
        } else {
            if ( offset[ tile_id ] != 0 ) {
                std::cerr << "tile[" << tile_id << "] - bad offset (" << offset[ tile_id ] << "), should be 0\n";
                err = 1;
            }
        }   
    }

    if ( err ) {
        std::cerr << "(*error*) Invalid tile information, aborting..." << std::endl;
        abort();
    }

    // Loop over tiles
    for( unsigned tile_id = 0; tile_id < ntiles.x * ntiles.y; ++tile_id ) {
        const auto tile_off = offset[ tile_id ];
        const auto tile_np  = np[ tile_id ];

        int2   * const __restrict__ t_ix = &ix[ tile_off ];
        float2 * const __restrict__ t_x  = &x[ tile_off ];
        float3 * const __restrict__ t_u  = &u[ tile_off ];

        for( int i = 0; i < tile_np; i++ ) {
            if ((t_ix[i].x < lb.x) || (t_ix[i].x >= ub.x )) { 
                std::cerr << "tile[" << tile_id << "] Invalid ix[" << i << "].x position (" << t_ix[i].x << ")"
                            << ", range = [" << lb.x << "," << ub.x << "]\n";
                err=1; break;
            }
            if ((t_ix[i].y < lb.y) || (t_ix[i].y >= ub.y )) { 
                std::cerr << "tile[" << tile_id << "] Invalid ix[" << i << "].y position (" << t_ix[i].y << ")"
                            << ", range = [" << lb.y << "," << ub.y << "]\n";
                err=1; break;
            }
            if ( std::isnan(t_u[i].x) || std::isinf(t_u[i].x) || std::abs(t_u[i].x) >= __ULIM ) {
                std::cerr << "tile[" << tile_id << "] Invalid u[" << i << "].x gen. velocity (" << t_u[i].x <<")\n";
                err=1; break;
            }
            if ( std::isnan(t_u[i].y) || std::isinf(t_u[i].y) || std::abs(t_u[i].y) >= __ULIM ) {
                std::cerr << "tile[" << tile_id << "] Invalid u[" << i << "].y gen. velocity (" << t_u[i].y <<")\n";
                err=1; break;
            }
            if ( std::isnan(t_u[i].z) || std::isinf(t_u[i].z) || std::abs(t_u[i].z) >= __ULIM ) {
                std::cerr << "tile[" << tile_id << "] Invalid u[" << i << "].z gen. velocity (" << t_u[i].z <<")\n";
                err=1; break;
            }
            if ( t_x[i].x < -0.5f || t_x[i].x >= 0.5f ) {
                std::cerr << "tile[" << tile_id << "] Invalid x[" << i << "].x position (" << t_x[i].x << "), range = [-0.5,0.5[\n";
                err=1; break;
            }
            if ( t_x[i].y < -0.5f || t_x[i].y >= 0.5f ) {
                std::cerr << "tile[" << tile_id << "] Invalid x[" << i << "].y position (" << t_x[i].y << "), range = [-0.5,0.5[\n";
                err=1; break;
            }
        }
    }

    if ( err ) {
        std::cerr << "(*error*) Invalid particle(s) found, aborting..." << std::endl;
        exit(1);
    } else {
        std::cout << "Particle set ok." << std::endl;
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
void Particles::validate( std::string msg, int const over ) {

    uint32_t nerr = 0;
    int2 const lb = make_int2( -over, -over );
    int2 const ub = make_int2( tiles.nx.x + over, tiles.nx.y + over );
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
        std::cerr << "(*error*) " << msg << ": invalid particle, aborting...\n";
        exit(1);
    }
}

#endif


/**
 * @brief Pack particles moving out of the node into a message buffer
 * 
 * @param tmp       Temporary buffer holding particles moving away from tiles
 * @param sort      Temporary sort index
 * @param send      Send message object
 */
void Particles::pack_msg( Particles &tmp, ParticleSortData &sort, ParticleMessage &send ) {

    /// @brief Total number of particles being sent
    uint32_t send_np = 0;
    /// @brief Offset in particle buffer for each message data
    uint32_t off[8];

    // Get offsets and check send buffer size
    for( int i = 0; i < 8; i++ ) {
        off[i] = send_np;
        send.size[i] = sort.send.msg_np[i] * particle_size();
        send_np += sort.send.msg_np[i];
    }
    send.check_buffer( send_np * particle_size() );

    // Pack data

    const auto tile_off = ntiles.x * ntiles.y;
    int2   * const __restrict__ ix = &tmp.ix[ tile_off ];
    float2 * const __restrict__ x  = &tmp.x[ tile_off ];
    float3 * const __restrict__ u  = &tmp.u[ tile_off ];

    #pragma omp parallel for schedule(dynamic)
    for( int i = 0; i < 8; i++) {
        uint8_t * __restrict__ buffer = &send.data[ off[i] * particle_size() ];
        size_t pos = 0;
        size_t n;

        n = sort.send.msg_np[i] * sizeof(int2);
        memcpy( &buffer[pos], &ix[ off[i] ], n );
        pos += n;

        n = sort.send.msg_np[i] * sizeof(float2);
        memcpy( &buffer[pos],  &x[ off[i] ], n);
        pos += n;

        n = sort.send.msg_np[i] * sizeof(float3);
        memcpy( &buffer[pos],  &u[ off[i] ], n);
        // pos += n; // unnecessary
    }

    // Start sending messages
    send.irecv();
}

/**
 * @brief Unpack received particle data into main particle data buffer
 * 
 * @param sort      Temporary sort index
 * @param recv      Receive message object
 */
void Particles::unpack_msg( ParticleSortData &sort, ParticleMessage &recv ) {

    // Total number of tiles in receive message buffer
    auto ntiles_msg = part::msg_tiles( ntiles );

    /// @brief number of particles per received tile
    int * msg_tile_np = sort.recv.buffer;
    
    /// @brief offset (in number of particles) in message buffer per tile
    int msg_offset[ ntiles_msg ];

    int sum = 0;
    for( int i = 0; i < ntiles_msg; ) {
        msg_offset[i] = sum; sum += msg_tile_np[i];
    }

    // Wait for messages to complete
    recv.wait();

    // Unpack all data - multiple message tiles may write to the same local tile
    #pragma omp parallel for schedule(dynamic)
    for( auto i = 0; i < ntiles_msg; i ++ ) {

        // Number of particles received
        int recv_np = msg_tile_np[i];

        // If any particles received in that tile
        if ( recv_np > 0 ) {
            // Get target tile for msg data
            int msg_target = part::tid_recv_target( i, ntiles );

            // destination buffers (in main particle buffer)
            int tgt_offset  = offset[ msg_target ] + 
                omp::atomic_fetch_add( &np[ msg_target ], recv_np );

            int2   * __restrict__ const dst_ix = &ix[ tgt_offset ];
            float2 * __restrict__ const dst_x  = &x [ tgt_offset ];
            float3 * __restrict__ const dst_u  = &u [ tgt_offset ];

            // source buffers (in packed data buffer)
            int src_offset = msg_offset[ i ] * particle_size();
            int2   * __restrict__ const src_ix = (int2 *)   & recv.data[ src_offset ];
            float2 * __restrict__ const src_x  = (float2 *) & recv.data[ src_offset + recv_np * sizeof(int2) ];
            float3 * __restrict__ const src_u  = (float3 *) & recv.data[ src_offset + recv_np * (sizeof(int2) + sizeof(float2))];

            // Copy data
            for( int j = 0; j < recv_np; j++ ) {
                dst_ix[j] = src_ix[j];
                dst_x [j] = src_x [j];
                dst_u [j] = src_u [j];
            }
        }
    }
}

#undef __ULIM
