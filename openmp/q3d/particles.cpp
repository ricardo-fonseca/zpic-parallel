#include "particles.h"

#include <iostream>
#include <string>
#include <cmath>

#include "timer.h"


/**
 * @brief Gather particle data
 * 
 * @tparam quant    Quantity to gather
 * @param part      Particle data
 * @param d_data    Output data
 */
template < part::quant quant >
void gather_quant( 
    ParticleData part,
    float * const __restrict__ d_data )
{
    const int2 ntiles = make_int2( part.ntiles.x, part.ntiles.y );

    #pragma omp parallel for schedule(dynamic)
    for( auto tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

        auto tx = tid % ntiles.x;
        auto ty = tid / ntiles.x;

        // Global spatial offsets of local tile
        const int offx = tx * part.nx.x;
        const int offy = ty * part.nx.y;

        const auto offset = part.offset[tid];
        const auto np     = part.np[tid];

        auto const * __restrict__ const ix = &part.ix[ offset ];
        auto const * __restrict__ const x  = &part.x[ offset ];
        auto const * __restrict__ const u  = &part.u[ offset ];
        auto const * __restrict__ const q  = &part.q[ offset ];
        auto const * __restrict__ const θ  = &part.θ[ offset ];
        
        for( int idx = 0; idx < np; idx ++ ) {
            float val;
            if constexpr( quant == part::z    ) val = ( offx + ix[idx].x ) + (0.5f + x[idx].x);
            if constexpr( quant == part::r    ) val = ( offy + ix[idx].y ) + x[idx].y;
            if constexpr( quant == part::q    ) val = q[idx];
            if constexpr( quant == part::cosθ ) val = θ[idx].x;
            if constexpr( quant == part::sinθ ) val = θ[idx].y;
            if constexpr( quant == part::ux   ) val = u[idx].x;
            if constexpr( quant == part::uy   ) val = u[idx].y;
            if constexpr( quant == part::uz   ) val = u[idx].z;
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
void Particles::gather( part::quant quant, float * const d_data )
{

    // Gather data on device
    switch (quant) {
    case part::z : 
        gather_quant<part::z>( *this, d_data );
        break;
    case part::r : 
        gather_quant<part::r>( *this, d_data );
        break;
    case part::q : 
        gather_quant<part::q>( *this, d_data );
        break;
    case part::cosθ : 
        gather_quant<part::cosθ>( *this, d_data );
        break;
    case part::sinθ:
        gather_quant<part::sinθ>( *this, d_data );
        break;
    case part::ux:
        gather_quant<part::ux>( *this, d_data );
        break;
    case part::uy:
        gather_quant<part::uy>( *this, d_data );
        break;
    case part::uz:
        gather_quant<part::uz>( *this, d_data );
        break;
    }
}

/**
 * @brief Gather particle data, scaling values
 * 
 * @note Data (val) will be returned as `scale.x * val + scale.y`
 * 
 * @tparam quant    Quantity to gather
 * @param part      Particle data
 * @param scale     Scale factor for data
 * @param d_data    Scaled output data
 */
template < part::quant quant >
void gather_quant( 
    ParticleData part,
    const float2 scale, 
    float * const __restrict__ d_data )
{
    const int2 ntiles = make_int2( part.ntiles.x, part.ntiles.y );

    #pragma omp parallel for schedule(dynamic)
    for( auto tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

        auto tx = tid % ntiles.x;
        auto ty = tid / ntiles.x;

        // Global spatial offsets of local tile
        const int offx = tx * part.nx.x;
        const int offy = ty * part.nx.y;

        const auto offset = part.offset[tid];
        const auto np     = part.np[tid];

        auto const * __restrict__ const ix = &part.ix[ offset ];
        auto const * __restrict__ const x  = &part.x[ offset ];
        auto const * __restrict__ const u  = &part.u[ offset ];
        auto const * __restrict__ const q  = &part.q[ offset ];
        auto const * __restrict__ const θ  = &part.θ[ offset ];
        
        for( int idx = 0; idx < np; idx ++ ) {
            float val;
            if constexpr( quant == part::z     ) val = ( offx + ix[idx].x ) + (0.5f + x[idx].x);
            if constexpr( quant == part::r     ) val = ( offy + ix[idx].y ) + x[idx].y;
            if constexpr( quant == part::q     ) val = q[idx];
            if constexpr( quant == part::cosθ  ) val = θ[idx].x;
            if constexpr( quant == part::sinθ  ) val = θ[idx].y;
            if constexpr( quant == part::ux    ) val = u[idx].x;
            if constexpr( quant == part::uy    ) val = u[idx].y;
            if constexpr( quant == part::uz    ) val = u[idx].z;
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
void Particles::gather( part::quant quant, const float2 scale, float * const __restrict__ d_data )
{
    
    // Gather data on device
    switch (quant) {
    case part::z : 
        gather_quant<part::z>( *this, scale, d_data );
        break;
    case part::r : 
        gather_quant<part::r>( *this, scale, d_data );
        break;
    case part::q : 
        gather_quant<part::q>( *this, scale, d_data );
        break;
    case part::cosθ : 
        gather_quant<part::cosθ>( *this, scale, d_data );
        break;
    case part::sinθ:
        gather_quant<part::sinθ>( *this, scale, d_data );
        break;
    case part::ux:
        gather_quant<part::ux>( *this, scale, d_data );
        break;
    case part::uy:
        gather_quant<part::uy>( *this, scale, d_data );
        break;
    case part::uz:
        gather_quant<part::uz>( *this, scale, d_data );
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

    uint32_t out_np = np_total();

    // Update metadata entry
    metadata.np = out_np;

    // Open file
    zdf::file part_file;
    zdf::open_part_file( part_file, metadata, iter, path+"/"+metadata.name );

    // Gather and save each quantity
    float *data = nullptr;

    if( out_np > 0 ) {
        data = memory::malloc<float>( out_np );
    }

    gather( part::quant::z, data );
    zdf::add_quant_part_file( part_file, "z", data, out_np );

    gather( part::quant::r, data );
    zdf::add_quant_part_file( part_file, "r", data, out_np );

    gather( part::quant::q, data );
    zdf::add_quant_part_file( part_file, "q", data, out_np );

    gather( part::quant::cosθ, data );
    zdf::add_quant_part_file( part_file, "cosθ", data, out_np );

    gather( part::quant::sinθ, data );
    zdf::add_quant_part_file( part_file, "sinθ", data, out_np );

    gather( part::quant::ux, data );
    zdf::add_quant_part_file( part_file, "ux", data, out_np );

    gather( part::quant::uy, data );
    zdf::add_quant_part_file( part_file, "uy", data, out_np );

    gather( part::quant::uz, data );
    zdf::add_quant_part_file( part_file, "uz", data, out_np );

    // Close the file
    zdf::close_file( part_file );

    // Cleanup
    if ( out_np > 0 ) {
        memory::free( data );
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
 * @param periodic_z    (in) Periodic boundaries along z
 */
void bnd_check( 
    ParticleData part, ParticleSortData sort, const int periodic_z )
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

            int target_tid = part::tid_coords( target, ntiles, periodic_z );
            
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
    ParticleData & tmp, 
    ParticleSort & sort,
    const int * __restrict__ extra = nullptr ) {

    const int * __restrict__ new_np = sort.new_np;

    // Include ghost tiles in calculations
    const auto ntiles     = tmp.ntiles.y * tmp.ntiles.x;

    int * __restrict__ offset = tmp.offset;
    int * __restrict__ np     = tmp.np;

    // Initialize offset[] with the new number of particles
    if ( extra != nullptr ) {
        // extra array only includes data for local tiles
        for( unsigned i = 0; i < ntiles; i++ ) {
            offset[i] = new_np[i] + extra[i];
            np[i] = 0;
        }
    } else {
        for( unsigned i = 0; i < ntiles; i++ ) {
            offset[i] = new_np[i];
            np[i] = 0;
        }
    }

    // Exclusive scan
    uint32_t total = 0;
    for( unsigned i = 0; i < ntiles; i++ ) {
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
 * @note Particles leaving the tile are copied to a temporary particle buffer
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
 * @param part        Particle data
 * @param tmp         Temporary particle buffer (has new offsets)
 * @param sort        Sort data (new number of particles per tile, indices of
 *                    particles leaving the tile, etc.)
 * @param periodic_z  Correct for periodic boundaries along z
 */
void copy_out( 
    ParticleData part, ParticleData tmp, const ParticleSortData sort, int const periodic_z )
{
    const int2 ntiles = make_int2( part.ntiles.x, part.ntiles.y );
    const int2 lim = make_int2( part.nx.x, part.nx.y );

    #pragma omp parallel for schedule(dynamic)
    for( auto tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

        auto tile_x = tid % ntiles.x;
        auto tile_y = tid / ntiles.x;

        int const old_offset      = part.offset[ tid ];
        int * __restrict__ npt    = &sort.npt[ 9*tid ];

        auto * __restrict__ ix  = &part.ix[ old_offset ];
        auto * __restrict__ x   = &part.x[ old_offset ];
        auto * __restrict__ u   = &part.u[ old_offset ];
        auto * __restrict__ q   = &part.q[ old_offset ];
        auto * __restrict__ θ   = &part.θ[ old_offset ];

        auto * __restrict__ idx   = &sort.idx[ old_offset ];
        uint32_t const nidx       = sort.nidx[ tid ];

        int const new_offset = tmp.offset[ tid ];
        int const new_np     = sort.new_np[ tid ];
        
        int _dir_offset[9];

        // The _dir_offset variable holds the offset for each of the 9 target
        // tiles so the tmp_* variables just point to the begining of the buffers
        auto * __restrict__ tmp_ix  = tmp.ix;
        auto * __restrict__ tmp_x  = tmp.x;
        auto * __restrict__ tmp_u  = tmp.u;
        auto * __restrict__ tmp_q  = tmp.q;
        auto * __restrict__ tmp_θ  = tmp.θ;

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
                int dx = i % 3 - 1;
                int dy = i / 3 - 1;

                int2 target = make_int2( tile_x + dx, tile_y + dy);

                int target_tid = part::tid_coords( target, ntiles, periodic_z );
                
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

            auto nix  = ix[k];
            auto nx   = x[k];
            auto nu   = u[k];
            auto nq   = q[k];
            auto nθ   = θ[k];

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
                tmp_q[ l ] = nq;
                tmp_θ[ l ] = nθ;
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
                q [ k ] = q [ c ];
                θ [ k ] = θ [ c ];
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
                tmp_q[ new_idx + i ]  = q [ i ];
                tmp_θ[ new_idx + i ]  = θ [ i ];
            }

        } else {

            // Copy from end of buffer
            const int old_idx = n0 - nshift;
            for( int i = 0; i < nshift; i++ ) {
                tmp_ix[ new_idx + i ] = ix[ old_idx + i ];
                tmp_x[ new_idx + i ]  = x [ old_idx + i ];
                tmp_u[ new_idx + i ]  = u [ old_idx + i ];
                tmp_q[ new_idx + i ]  = q [ old_idx + i ];
                tmp_θ[ new_idx + i ]  = θ [ old_idx + i ];
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
        auto * __restrict__ ix  = &part.ix[ new_offset ];
        auto * __restrict__ x   = &part.x [ new_offset ];
        auto * __restrict__ u   = &part.u [ new_offset ];
        auto * __restrict__ q   = &part.q [ new_offset ];
        auto * __restrict__ θ   = &part.θ[ new_offset ];

        auto * __restrict__ tmp_ix = &tmp.ix[ new_offset ];
        auto * __restrict__ tmp_x  = &tmp.x [ new_offset ];
        auto * __restrict__ tmp_u  = &tmp.u [ new_offset ];
        auto * __restrict__ tmp_q  = &tmp.q [ new_offset ];
        auto * __restrict__ tmp_θ  = &tmp.θ[ new_offset ];

        if ( new_offset >= old_offset ) {

            // Add particles to the end of the buffer
            for( int i = 0; i < tmp_np; i++ ) {
                ix[ old_np + i ] = tmp_ix[ i ];
                x [ old_np + i ] = tmp_x[ i ];
                u [ old_np + i ] = tmp_u[ i ];
                q [ old_np + i ] = tmp_q[ i ];
                θ [ old_np + i ] = tmp_θ[ i ];
            }
        } else {

            // Add particles to the begining of buffer
            int np0 = old_offset - new_offset;
            if ( np0 > tmp_np ) np0 = tmp_np;
            
            for( int i = 0; i < np0; i ++ ) {
                ix[ i ] = tmp_ix[ i ];
                x [ i ] = tmp_x [ i ];
                u [ i ] = tmp_u [ i ];
                q [ i ] = tmp_q [ i ];
                θ [ i ] = tmp_θ [ i ];
            }

            // If any particles left, add particles to the end of the buffer
            for( int i = np0; i < tmp_np; i ++ ) {
                ix[ old_np + i ] = tmp_ix[ i ];
                x [ old_np + i ] = tmp_x [ i ];
                u [ old_np + i ] = tmp_u [ i ];
                q [ old_np + i ] = tmp_q [ i ];
                θ [ old_np + i ] = tmp_θ [ i ];
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
 * @param part        Particle data
 * @param tmp         Temporary particle buffer (has new offsets)
 * @param sort        Sort data (indices of particles leaving the tile, etc.)
 * @param periodic_z  Correct for periodic boundaries along z
 */
void copy_sorted( 
    ParticleData part, ParticleData tmp, const ParticleSortData sort,
    const int periodic_z )
{
    // Copy all particles to correct tile in tmp buffer
    const int2 ntiles = make_int2( part.ntiles.x, part.ntiles.y );
    const int2 lim = make_int2( part.nx.x, part.nx.y );

    #pragma omp parallel for schedule(dynamic)
    for( auto tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

        const int tile_y = tid / ntiles.x;
        const int tile_x = tid % ntiles.x;
    
        int const old_offset      = part.offset[ tid ];
        int * __restrict__ npt    = &sort.npt[ 9*tid ];

        auto * __restrict__ ix  = &part.ix[ old_offset ];
        auto * __restrict__ x   = &part.x[ old_offset ];
        auto * __restrict__ u   = &part.u[ old_offset ];
        auto * __restrict__ q   = &part.q[ old_offset ];
        auto * __restrict__ θ   = &part.θ[ old_offset ];

        int * __restrict__ idx    = &sort.idx[ old_offset ];
        uint32_t const nidx       = sort.nidx[ tid ];
        
        int _dir_offset[9];

        // The _dir_offset variables hold the offset for each of the 9 target
        // tiles so the tmp_* variables just point to the begining of the buffers
        auto * __restrict__ tmp_ix = tmp.ix;
        auto * __restrict__ tmp_x  = tmp.x;
        auto * __restrict__ tmp_u  = tmp.u;
        auto * __restrict__ tmp_q  = tmp.q;
        auto * __restrict__ tmp_θ  = tmp.θ;

        // sync

        // Find offsets on new buffer
        for( int i = 0; i < 9; i++ ) {
            
            // Find target node
            int2 target = make_int2( tile_x + i % 3 - 1, tile_y + i / 3 - 1 );

            int target_tid = part::tid_coords( target, ntiles, periodic_z );

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

            auto nix = ix[k];
            auto nx  = x[k];
            auto nu  = u[k];
            auto nq  = q[k];
            auto nθ  = θ[k];
            
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
                tmp_x[ l ]  = nx;
                tmp_u[ l ]  = nu;
                tmp_q[ l ]  = nq;
                tmp_θ[ l ]  = nθ;
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
                q [ k ] = q [ c ];
                θ [ k ] = θ [ c ];
            }
        }

        // sync

        // Copy particles staying in tile
        const int start = _dir_offset[4];

        for( int i = 0; i < n0; i ++ ) {
            tmp_ix[ start + i ] = ix[i];
            tmp_x [ start + i ] = x[i];
            tmp_u [ start + i ] = u[i];
            tmp_q [ start + i ] = q[i];
            tmp_θ [ start + i ] = θ[i];
        }
    }
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
    bnd_check ( *this, sort, periodic_z );

    // Get new offsets, including:
    // - Incoming particles from other MPI nodes
    // - New particles that will be injected (if any)
    auto total_np = update_tile_info ( tmp, sort, extra );

    if ( total_np > max_part ) { 
        std::cerr << "Particles::tile_sort() - particle buffer requires growing,"
                  << "max_part: " << max_part << ", total_np: " << total_np
                  << ", not implemented yet.\n";
        std::exit(1);
    }

    // Copy outgoing particles (and particles needing shifting) to staging area
    copy_out ( *this, tmp, sort, periodic_z );

    // Copy local particles from staging area into final positions in particle buffer
    copy_in ( *this, tmp );

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
void Particles::cell_shift( int2 const shift ) {

    // Loop over tiles
    #pragma omp parallel for schedule(dynamic)
    for( unsigned tid = 0; tid < ntiles.y * ntiles.x; tid++ ) {
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

#define __ULIM __FLT_MAX__


#if 1

/**
 * @brief Checks particle buffer data for error
 * 
 * @warning This routine is meant for debug only and should not be called 
 *          for production code.
 * 
 * The routine will check for:
 *      1. Invalid (z,r) cell data (out of tile bounds)
 *      2. Invalid (z,r) position data (out of [-0.5,0.5[)
 *      3. Invalid (cosθ, sinθ) data (out of [-1,1])
 *      3. Invalid momenta (nan, inf or above __ULIM macro value)
 * 
 * If there are any errors found the routine will exit the code.
 * 
 * @param msg       Message to print in case error is found
 * @param over      Amount of extra cells indices beyond limit allowed. Used
 *                  when checking the buffer before tile_sort()
 */
void Particles::validate( std::string msg, int const over ) {

    if ( msg.empty() ) {
        std::cout << "validating particle set...";
    } else {
        std::cout << "validating particle set (" << msg << ")...";
    }

    uint32_t err = 0;
    int2 const lb = make_int2( -over, -over );
    int2 const ub = make_int2( nx.x + over, nx.y + over ); 

    // Check offset / np buffer
    for( unsigned tile_id = 0; tile_id < ntiles.x * ntiles.y; ++tile_id ) {
        if ( np[tile_id] < 0 ) {
            std::cout << "\n tile[" << tile_id << "] - bad np (" << np[ tile_id ] << "), should be >= 0";
            err = 1;
        }

        if ( tile_id > 0 ) {
            auto prev = offset[ tile_id-1] + np[ tile_id-1];
            if ( prev != offset[ tile_id ] ) {
                std::cout << "\n tile[" << tile_id << "] - bad offset (" << offset[ tile_id ] << ")"
                          << ", does not match previous tile info, should be " << prev;
                err = 1;
            }
        } else {
            if ( offset[ tile_id ] != 0 ) {
                std::cout << "tile[" << tile_id << "] - bad offset (" << offset[ tile_id ] << "), should be 0";
                err = 1;
            }
        }   
    }

    if ( err ) {
        std::cout << "\n(*error*) Invalid tile information, aborting..." << std::endl;
        std::exit(1);
    }

    // Loop over tiles
    for( unsigned tile_id = 0; tile_id < ntiles.x * ntiles.y; ++tile_id ) {
        const auto tile_off = offset[ tile_id ];
        const auto tile_np  = np[ tile_id ];

        auto * const __restrict__ t_ix = &ix[ tile_off ];
        auto * const __restrict__ t_x  = &x[ tile_off ];
        auto * const __restrict__ t_u  = &u[ tile_off ];
        auto * const __restrict__ t_q  = &q[ tile_off ];
        auto * const __restrict__ t_θ  = &θ[ tile_off ];

        for( int i = 0; i < tile_np; i++ ) {
            if ((t_ix[i].x < lb.x) || (t_ix[i].x >= ub.x )) { 
                std::cout << "\ntile[" << tile_id << "] Invalid ix[" << i << "] z position (" << t_ix[i].x << ")"
                          << ", range = [" << lb.x << "," << ub.x << "]\n";
                err=1; break;
            }
            if ( t_x[i].x < -0.5f || t_x[i].x >= 0.5f ) {
                std::cout << "\ntile[" << tile_id << "] Invalid x[" << i << "] z position (" << t_x[i].x << ")"
                          << ", range =  [-0.5,0.5[\n";
                err = 1;
            }

            if ((t_ix[i].y < lb.y) || (t_ix[i].y >= ub.y )) { 
                std::cout << "\ntile[" << tile_id << "] Invalid ix[" << i << "] r position (" << t_ix[i].y << ")"
                          << ", range = [" << lb.y << "," << ub.y << "]\n";
                err=1; break;
            }
            if ( t_x[i].y < -0.5f || t_x[i].y >= 0.5f ) {
                std::cout << "\ntile[" << tile_id << "] Invalid y[" << i << "] r position (" << t_x[i].y << ")"
                          << ", range =  [-0.5,0.5[\n";
                err = 1;
            }

            if ( std::isnan(t_u[i].x) || std::isinf(t_u[i].x) || std::abs(t_u[i].x) >= __ULIM ) {
                std::cout << "\ntile[" << tile_id << "] Invalid u[" << i << "].x gen. velocity (" << t_u[i].x <<")\n";
                err=1; break;
            }
            if ( std::isnan(t_u[i].y) || std::isinf(t_u[i].y) || std::abs(t_u[i].y) >= __ULIM ) {
                std::cout << "\ntile[" << tile_id << "] Invalid u[" << i << "].y gen. velocity (" << t_u[i].y <<")\n";
                err=1; break;
            }
            if ( std::isnan(t_u[i].z) || std::isinf(t_u[i].z) || std::abs(t_u[i].z) >= __ULIM ) {
                std::cout << "\ntile[" << tile_id << "] Invalid u[" << i << "].z gen. velocity (" << t_u[i].z <<")\n";
                err=1; break;
            }

            if ( t_q[i] == 0 || std::isnan(t_q[i]) || std::isinf(t_q[i]) || std::abs(t_q[i]) >= __ULIM ) {
                std::cout << "\ntile[" << tile_id << "] Invalid q[" << i << "] charge (" << t_q[i] <<")\n";
                err=1; break;
            }


            if ( t_θ[i].x < -1 || t_θ[i].x > 1 ) {
                std::cout << "\ntile[" << tile_id << "] Invalid cosθ[" << i << "] value (" 
                          << t_θ[i].x << "), range = [-1,1]\n";
                err=1; break;
            }
            if ( t_θ[i].y < -1 || t_θ[i].y > 1 ) {
                std::cout << "\ntile[" << tile_id << "] Invalid sinθ[" << i << "] value ("
                          << t_θ[i].y << "), range = [-1,1]\n";
                err=1; break;
            }
        }
    }

    if ( err ) {
        std::cout << "\n(*error*) Invalid particle(s) found, aborting..." << std::endl;
        std::exit(1);
    } else {
        std::cout << " particle set ok.\n";
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

            auto * const __restrict__ ix = &data.ix[ tile_off ];
            auto * const __restrict__ x  = &data.x[ tile_off ];
            auto * const __restrict__ u  = &data.u[ tile_off ];
            auto * const __restrict__ q  = &data.q[ tile_off ];
            auto * const __restrict__ θ  = &data.θ[ tile_off ];

            for( int i = 0; i < tile_np; i++ ) {
                if ( (ix[i].x < lb.x) || (ix[i].x >= ub.x )) { nerr++; break; }
                if ( (ix[i].y < lb.y) || (ix[i].y >= ub.y )) { nerr++; break; }
                if ( x[i].x < -0.5f || x[i].x >= 0.5f ) { nerr++; break; }
                if ( x[i].y < -0.5f || x[i].y >= 0.5f ) { nerr++; break; }
                if ( std::isnan(u[i].x) || std::isinf(u[i].x) || std::abs(u[i].x) >= __ULIM ) { nerr++; break; }
                if ( std::isnan(u[i].y) || std::isinf(u[i].y) || std::abs(u[i].y) >= __ULIM ) { nerr++; break; }
                if ( std::isnan(u[i].x) || std::isinf(u[i].x) || std::abs(u[i].x) >= __ULIM ) { nerr++; break; }
                if ( θ[i].x < -1 || θ[i].x > 1 ) { nerr++; break; }
                if ( θ[i].y < -1 || θ[i].y > 1 ) { nerr++; break; }
                if ( std::isnan(q[i]) || std::isinf(q[i]) || std::abs(q[i]) >= __ULIM ) { nerr++; break; }
            }
        }
    }

    if ( nerr > 0 ) {
        std::cerr << "(*error*) " << msg << ": invalid particle, aborting...\n";
        exit(1);
    }
}

#endif
