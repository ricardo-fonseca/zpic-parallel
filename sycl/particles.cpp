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
    sycl::queue & q )
{
    auto tile_nx = part.nx;
    auto ntiles  = part.ntiles;

    // 8×1 work items per group
    sycl::range<2> local{ 8, 1 };

    // ntiles.x × ntiles.y groups
    sycl::range<2> global{ ntiles.x, ntiles.y };

    q.submit([&](sycl::handler &h) {

        h.parallel_for( 
            sycl::nd_range{ global * local, local },
            [=](sycl::nd_item<2> it) { 
            
            const int2 tile_idx = make_int2( it.get_group(0), it.get_group(1));
            const int tile_id = tile_idx.y * part.ntiles.x + tile_idx.x;
            const auto tile_off = part.offset[ tile_id ];
            const auto tile_np  = part.np[ tile_id ];

            int2   * const __restrict__ ix       = & part.ix[ tile_off ];
            float2 const * __restrict__ const x  = & part.x[ tile_off ];
            float3 const * __restrict__ const u  = & part.u[ tile_off ];

            for( int idx = it.get_local_id(0); idx < tile_np; idx += it.get_local_range(0) ) {
                float val;
                if ( quant == part::x )  val = (it.get_group(0) * tile_nx.x + ix[idx].x) + (0.5f + x[idx].x);
                if ( quant == part::y )  val = (it.get_group(1) * tile_nx.y + ix[idx].y) + (0.5f + x[idx].y);
                if ( quant == part::ux ) val = u[idx].x;
                if ( quant == part::uy ) val = u[idx].y;
                if ( quant == part::uz ) val = u[idx].z;
                d_data[ tile_off + idx ] = val;
            }
        });
    });
    q.wait();
};

/**
 * @brief Gather data from a specific particle quantity in a device buffer
 * 
 * @param quant         Quantity to gather
 * @param d_data        Output data buffer, assumed to have size >= np
 */
void Particles::gather( part::quant quant, float * const __restrict__ d_data )
{
    // Gather data on device
    switch (quant) {
    case part::x : 
        gather_quant<part::x>( *this, d_data, queue );
        break;
    case part::y:
        gather_quant<part::y>( *this, d_data, queue );
        break;
    case part::ux:
        gather_quant<part::ux>( *this, d_data, queue );
        break;
    case part::uy:
        gather_quant<part::uy>( *this, d_data, queue );
        break;
    case part::uz:
        gather_quant<part::uz>( *this, d_data, queue );
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
    float * const __restrict__ d_data,
    const float2 scale,
    sycl::queue & q )
{
    auto tile_nx = part.nx;

    // 8×1 work items per group
    sycl::range<2> local{ 8, 1 };

    // ntiles.x × ntiles.y groups
    sycl::range<2> global{ part.ntiles.x, part.ntiles.y };

    q.submit([&](sycl::handler &h) {

        h.parallel_for( 
            sycl::nd_range{ global * local, local },
            [=](sycl::nd_item<2> it) { 
            
            const int2 tile_idx = make_int2( it.get_group(0), it.get_group(1));
            const int tile_id = tile_idx.y * part.ntiles.x + tile_idx.x;

            const auto tile_off = part.offset[ tile_id ];
            const auto tile_np  = part.np[ tile_id ];

            int2   * const __restrict__ ix = &part.ix[ tile_off ];
            float2 const * __restrict__ const x  = &part.x[ tile_off ];
            float3 const * __restrict__ const u  = &part.u[ tile_off ];

            for( int idx = it.get_local_id(0); idx < tile_np; idx += it.get_local_range(0) ) {
                float val;
                if ( quant == part::x )  val = (it.get_group(0) * tile_nx.x + ix[idx].x) + (0.5f + x[idx].x);
                if ( quant == part::y )  val = (it.get_group(1) * tile_nx.y + ix[idx].y) + (0.5f + x[idx].y);
                if ( quant == part::ux ) val = u[idx].x;
                if ( quant == part::uy ) val = u[idx].y;
                if ( quant == part::uz ) val = u[idx].z;
                d_data[ tile_off + idx ] = ops::fma( scale.x, val, scale.y );
            }
        });
    }); 
    q.wait();
}

/**
 * @brief Gather data from a specific particle quantity in a device buffer, scaling values
 * 
 * @param quant     Quantity to gather
 * @param d_data    Output data buffer, assumed to have size >= np
 * @param scale     Scale factor for data
 */
void Particles::gather( part::quant quant, float * const __restrict__ d_data, const float2 scale )
{
    // Gather data on device
    switch (quant) {
    case part::x : 
        gather_quant<part::x>( *this, d_data, scale, queue );
        break;
    case part::y:
        gather_quant<part::y>( *this, d_data, scale, queue );
        break;
    case part::ux:
        gather_quant<part::ux>( *this, d_data, scale, queue );
        break;
    case part::uy:
        gather_quant<part::uy>( *this, d_data, scale, queue );
        break;
    case part::uz:
        gather_quant<part::uz>( *this, d_data, scale, queue );
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
        d_data = device::malloc<float>( np, queue );
        h_data = host::malloc<float>( np, queue );
    }

    if ( np > 0 ) {
        gather( part::quant::x, d_data );
        device::memcpy_tohost( h_data, d_data, np, queue );
    }
    zdf::add_quant_part_file( part_file, "x", h_data, np );

    if ( np > 0 ) {
        gather( part::quant::y, d_data );
        device::memcpy_tohost( h_data, d_data, np, queue );
    }
    zdf::add_quant_part_file( part_file, "y", h_data, np );

    if ( np > 0 ) {
        gather( part::quant::ux, d_data );
        device::memcpy_tohost( h_data, d_data, np, queue );
    }
    zdf::add_quant_part_file( part_file, "ux", h_data, np );

    if ( np > 0 ) {
        gather( part::quant::uy, d_data );
        device::memcpy_tohost( h_data, d_data, np, queue );
    }
    zdf::add_quant_part_file( part_file, "uy", h_data, np );

    if ( np > 0 ) {
        gather( part::quant::uz, d_data );
        device::memcpy_tohost( h_data, d_data, np, queue );
    }
    zdf::add_quant_part_file( part_file, "uz", h_data, np );

    // Close the file
    zdf::close_file( part_file );

    // Cleanup
    if ( np > 0 ) {
        device::free( d_data, queue );
        host::free( h_data, queue );
    }
}

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
void bnd_check( 
    ParticleData part, ParticleSortData sort, 
    int2 const periodic,
    sycl::queue & q )
{
    int2 ntiles = make_int2( part.ntiles.x, part.ntiles.y );
    int2 lim = make_int2( part.nx.x, part.nx.y );

    // 256×1 work items per group
    sycl::range<2> local{ 256, 1 };

    // ntiles.x × ntiles.y groups
    sycl::range<2> global{ part.ntiles.x, part.ntiles.y };

    q.submit([&](sycl::handler &h) {

        /// @brief [shared] Number of particles moving in each direction
        auto _npt = sycl::local_accessor< int, 1 > ( 9, h );

        /// @brief [shared] Number of particle leaving tile
        auto _nout = sycl::local_accessor< int, 1 > ( 1, h );

        h.parallel_for( 
            sycl::nd_range{ global * local, local },
            [=](sycl::nd_item<2> it) { 

            const int2 tile_idx = make_int2( it.get_group(0), it.get_group(1));
            const int tile_id = tile_idx.y * ntiles.x + tile_idx.x;

            const auto offset  = part.offset[ tile_id ];
            const auto np      = part.np[ tile_id ];

            int2 * __restrict__ ix    = &part.ix[ offset ];

            /// @brief Indices of particles leaving tile
            int  * __restrict__ idx   = &sort.idx[ offset ];

            for( auto i = 0; i < 9; ++i ) _npt[i] = 0;
            _nout[0] = 0;

            it.barrier();

            // Count particles according to their motion
            // Store indices of particles leaving tile

            for( int i = it.get_local_id(0); i < np; i += it.get_local_range(0) ) {
                int2 ipos = ix[i];
                int xcross = ( ipos.x >= lim.x ) - ( ipos.x < 0 );
                int ycross = ( ipos.y >= lim.y ) - ( ipos.y < 0 );
                
                if ( xcross || ycross ) {
                    device::local::atomicAdd( &_npt[ (ycross+1) * 3 + (xcross+1) ], 1 );
                    idx[ device::local::atomicAdd( &_nout[0], 1 ) ] = i;
                }
            }

            it.barrier();

            if (it.get_local_id(0) == 0 ){
                // Particles remaining on the tile
                _npt[4] = np - _nout[0];
                // Store number of particles leaving tile
                sort.nidx[ tile_id ] = _nout[0];
            }

            it.barrier();

            for( int i = it.get_local_id(0); i < 9; i += it.get_local_range(0) ) {
                
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
                    device::global::atomicAdd( & sort.new_np[ target_tid ], _npt[i] );
                }
            }
        });
    });
    q.wait();
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
    ParticleData tmp,
    const int * __restrict__ new_np,
    sycl::queue & q )
{
    const int ntiles = tmp.ntiles.x * tmp.ntiles.y;
    const int max_num_sub_groups = q.get_device().get_info<sycl::info::device::max_num_sub_groups>();
    
    unsigned group_size = ( ntiles < 256 ) ? ntiles : 256;
    sycl::range<1> local{ group_size };

    q.submit([&](sycl::handler &h) {

        /// @brief [shared] Sum of previous group
        auto group_prev = sycl::local_accessor< int, 1 > ( 1, h );

        /// @brief [shared] Sum of previous sub-group
        auto sg_prev = sycl::local_accessor< int, 1 > ( 1, h );

        /// @brief [shared] Temporary results from each sub group
        auto _tmp = sycl::local_accessor< int, 1 > ( max_num_sub_groups, h );

        h.parallel_for( 
            sycl::nd_range{ local, local },
            [=](sycl::nd_item<1> it) {

            group_prev[0] = 0;
            auto sg = it.get_sub_group();

            for( int i = it.get_local_id(0); i < ntiles; i += it.get_local_range(0) ) {
                auto s = new_np[i];
                auto v = device::subgroup::exscan_add( sg, s );

                if ( sg.get_local_id() == sg.get_local_range() - 1 )
                    _tmp[ sg.get_group_linear_id() ] = v + s;

                it.barrier();

                // Only 1 sub-group does this
                if ( sg.get_group_linear_id() == 0 ) {
                    sg_prev[0] = group_prev[0];
                    for( auto j = 0; j < sg.get_group_linear_range(); j += sg.get_local_linear_range() ) {
                        int t = _tmp[ j + sg.get_local_id() ];
                        int e = device::subgroup::exscan_add( sg, t ) + sg_prev[0];
                        _tmp[ j + sg.get_local_id() ] = e;
                        if ( sg.get_local_id() == sg.get_local_linear_range() - 1 ) 
                            sg_prev[0] = e + t;
                    }
                }
                it.barrier();

                // Add in contribution from previous threads
                v += _tmp[ sg.get_group_linear_id() ];

                tmp.offset[i] = v;
                tmp.np[i] = 0;

                if ( it.get_local_id(0) == it.get_local_range(0)-1 ) {
                    group_prev[0] = v+s;
                }
                it.barrier();
            }
        });
    });
    q.wait();
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
uint32_t update_tile_info( 
    ParticleData tmp,
    const int * __restrict__ new_np, 
    const int * __restrict__ extra,
    device::Var<uint32_t> tmp_dev_np,
    sycl::queue & q )
{
    const int ntiles = tmp.ntiles.x * tmp.ntiles.y;
    const int max_num_sub_groups = q.get_device().get_info<sycl::info::device::max_num_sub_groups>();
    
    unsigned group_size = ( ntiles < 256 ) ? ntiles : 256;
    sycl::range<1> local{ group_size };
 
    q.submit([&](sycl::handler &h) {

        auto dev_np = tmp_dev_np.ptr();

        /// @brief [shared] Sum of previous group
        auto _prev = sycl::local_accessor< int, 1 > ( 1, h );

        /// @brief [shared] Temporary results from each sub group
        auto _tmp = sycl::local_accessor< int, 1 > ( max_num_sub_groups, h );

        h.parallel_for( 
            sycl::nd_range{ local, local },
            [=](sycl::nd_item<1> it) {

            _prev[0] = 0;
            auto sg = it.get_sub_group();

            for( int i = it.get_local_id(0); i < ntiles; i += it.get_local_range(0) ) {
                auto s = new_np[i] + extra[i];
                auto v = device::subgroup::exscan_add( sg, s );

                if ( sg.get_local_id() == sg.get_local_range() - 1 )
                    _tmp[ sg.get_group_linear_id() ] = v + s;

                it.barrier();

                // Only 1 warp does this
                if ( sg.get_group_linear_id() == 0 ) {
                    auto t = _tmp[ sg.get_local_id() ];
                    t = device::subgroup::exscan_add(sg, t);
                    _tmp[ sg.get_local_id() ] = t + _prev[0];
                }
                it.barrier();


                // Add in contribution from previous threads
                v += _tmp[ sg.get_group_linear_id() ];

                tmp.offset[i] = v;
                tmp.np[i] = 0;

                if (( it.get_local_id(0) == it.get_local_range(0)-1 ) || (i+1 == ntiles)) {
                    _prev[0] = v+s;
                }
                it.barrier();
            }

            if ( it.get_global_id(0) == 0 ) {
                dev_np[0] = _prev[0];
            }
        });
    });
    q.wait();

    return tmp_dev_np.get();
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
    ParticleData part, 
    ParticleData tmp,
    const ParticleSortData sort,
    const int2 periodic,
    sycl::queue & q )
{
    const int2 ntiles = make_int2( part.ntiles.x, part.ntiles.y );
    const int2 lim = make_int2( part.nx.x, part.nx.y );

    // 8×1 work items per group
    sycl::range<2> local{ 256, 1 };

    // ntiles.x × ntiles.y groups
    sycl::range<2> global{ part.ntiles.x, part.ntiles.y };
 
    q.submit([&](sycl::handler &h) {

        /// @brief [shared] offsets in target buffer
        auto _dir_offset = sycl::local_accessor< int, 1 > ( 9, h );

        /// @brief [shared] index of particle used to fill hole
        auto _c = sycl::local_accessor< int, 1 > ( 1, h );

        h.parallel_for( 
            sycl::nd_range{ global * local, local },
            [=](sycl::nd_item<2> it) { 

            const int2 tile_idx = make_int2( it.get_group(0), it.get_group(1) );
            const int tile_id = tile_idx.y * part.ntiles.x + tile_idx.x;

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
            if ( it.get_local_id(0) == 0 ) {
                _dir_offset[4] = new_offset + device::global::atomicAdd( & tmp.np[ tile_id ], nshift );
            }

            it.barrier();

            // Find offsets on new buffer
            for( int i = it.get_local_id(0); i < 9; i += it.get_local_range(0) ) {
                
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
                                        device::global::atomicAdd( &tmp.np[ target_tid ], npt[ i ] );
                    } else {
                        // Otherwise mark offset as invalid
                        _dir_offset[i] = -1;
                    }
                } 
            }

            _c[0] = n0;

            it.barrier();

            // Copy particles moving away from tile and fill holes
            for( int i = it.get_local_id(0); i < nidx; i += it.get_local_range(0) ) {
                
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
                    int l = device::local::atomicAdd( & _dir_offset[dir], 1 );

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
                        c = device::local::atomicAdd( &_c[0], 1 );
                        invalid = ( ix[c].x < 0 ) || ( ix[c].x >= lim.x) || 
                                  ( ix[c].y < 0 ) || ( ix[c].y >= lim.y);
                    } while (invalid);

                    ix[ k ] = ix[ c ];
                    x [ k ] = x [ c ];
                    u [ k ] = u [ c ];
                }
            }

            it.barrier();

            // At this point all particles up to n0 are correct

            // Copy particles that need to be shifted
            // We've reserved space for nshift particles earlier
            const int new_idx = _dir_offset[4];

            if ( new_offset >= old_offset ) {
                // Copy from beggining of buffer
                for( int i = it.get_local_id(0); i < nshift; i += it.get_local_range(0) ) {
                    tmp_ix[ new_idx + i ] = ix[ i ];
                    tmp_x[ new_idx + i ]  = x [ i ];
                    tmp_u[ new_idx + i ]  = u [ i ];
                }

            } else {

                // Copy from end of buffer
                const int old_idx = n0 - nshift;
                for( int i = it.get_local_id(0); i < nshift; i += it.get_local_range(0) ) {
                    tmp_ix[ new_idx + i ] = ix[ old_idx + i ];
                    tmp_x[ new_idx + i ]  = x [ old_idx + i ];
                    tmp_u[ new_idx + i ]  = u [ old_idx + i ];
                }
            }

            // Store current number of local particles
            // These are already in the correct position in global buffer
            if ( it.get_local_id(0) == 0 ) {
                part.np[ tile_id ] = n0 - nshift;
            }

        });
    });
    q.wait();
}

/**
 * @brief Copy incoming particles to main buffer. Buffer will be fully sorted after
 *        this step
 * 
 * @param part      Main particle data
 * @param tmp       Temporary particle data
 */
void copy_in(
    ParticleData part,
    ParticleData tmp,
    sycl::queue & q ) 
{
    // 8×1 work items per group
    sycl::range<2> local{ 256, 1 };

    // ntiles.x × ntiles.y groups
    sycl::range<2> global{ part.ntiles.x, part.ntiles.y };

    q.submit([&](sycl::handler &h) {

        h.parallel_for( 
            sycl::nd_range{ global * local, local },
            [=](sycl::nd_item<2> it) { 

            const int2 tile_idx = make_int2( it.get_group(0), it.get_group(1));
            const int tile_id = tile_idx.y * part.ntiles.x + tile_idx.x;

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
                for( int i = it.get_local_id(0); i < tmp_np; i += it.get_local_range(0) ) {
                    ix[ old_np + i ] = tmp_ix[ i ];
                    x[ old_np + i ]  = tmp_x[ i ];
                    u[ old_np + i ]  = tmp_u[ i ];
                }

            } else {

                // Add particles to the beggining of buffer
                int np0 = old_offset - new_offset;
                if ( np0 > tmp_np ) np0 = tmp_np;
                
                for( int i = it.get_local_id(0); i < np0; i += it.get_local_range(0) ) {
                    ix[ i ] = tmp_ix[ i ];
                    x[ i ]  = tmp_x[ i ];
                    u[ i ]  = tmp_u[ i ];
                }

                // If any particles left, add particles to the end of the buffer
                for( int i = np0 + it.get_local_id(0); i < tmp_np; i += it.get_local_range(0) ) {
                    ix[ old_np + i ] = tmp_ix[ i ];
                    x[ old_np + i ]  = tmp_x[ i ];
                    u[ old_np + i ]  = tmp_u[ i ];
                }

            }

            it.barrier();

            // Store the new offset and number of particles
            if ( it.get_local_id(0) == 0 ) {
                part.np[ tile_id ]     = old_np + tmp_np;
                part.offset[ tile_id ] = new_offset;
            }
        });
    });
    q.wait();
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
    ParticleData part,
    ParticleData tmp,
    const ParticleSortData sort,
    const int2 periodic,
    sycl::queue & q )
{
    const int2 ntiles = make_int2( part.ntiles.x, part.ntiles.y );
    const int2 lim = make_int2( part.nx.x, part.nx.y );

    // 8×1 work items per group
    sycl::range<2> local{ 8, 1 };

    // ntiles.x × ntiles.y groups
    sycl::range<2> global{ part.ntiles.x, part.ntiles.y };

    q.submit([&](sycl::handler &h) {

        /// @brief [shared] offsets in target buffer
        auto _dir_offset = sycl::local_accessor< int, 1 > ( 9, h );

        /// @brief [shared] index of particle used to fill hole
        auto _c = sycl::local_accessor< int, 1 > ( 1, h );

        h.parallel_for( 
            sycl::nd_range{ global * local, local },
            [=](sycl::nd_item<2> it) { 

            const int2 tile_idx = make_int2( it.get_group(0), it.get_group(1));
            const int tile_id = tile_idx.y * part.ntiles.x + tile_idx.x;

    
            int const old_offset      = part.offset[ tile_id ];
            int * __restrict__ npt    = &sort.npt[ 9*tile_id ];

            int2   * __restrict__ ix  = &part.ix[ old_offset ];
            float2 * __restrict__ x   = &part.x[ old_offset ];
            float3 * __restrict__ u   = &part.u[ old_offset ];

            int * __restrict__ idx    = &sort.idx[ old_offset ];
            uint32_t const nidx       = sort.nidx[ tile_id ];
            
            int _dir_offset[9];

            // The _dir_offset variables hold the offset for each of the 9 target
            // tiles so the tmp_* variables just point to the beggining of the buffers
            int2* __restrict__  tmp_ix  = tmp.ix;
            float2* __restrict__ tmp_x  = tmp.x;
            float3* __restrict__ tmp_u  = tmp.u;

            // Find offsets on new buffer
            for( int i = it.get_local_id(0); i < 9; i += it.get_local_range(0) ) {

                int tx = it.get_group(0);
                int ty = it.get_group(1);
                
                // Find target node
                int target_tx = tx + i % 3 - 1;
                int target_ty = ty + i / 3 - 1;

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
                        device::global::atomicAdd( & tmp.np[ target_tid ], npt[ i ] );
                
                } else {
                    // Otherwise mark offset as invalid
                    _dir_offset[i] = -1;
                }
            }

            const int n0 = npt[4];
            _c[0] = n0;

            it.barrier();

            // Copy particles moving away from tile and fill holes
            for( int i = it.get_local_id(0); i < nidx; i += it.get_local_range(0) ) {
                
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
                    int l = device::local::atomicAdd( & _dir_offset[dir], 1 );

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
                        c = device::local::atomicAdd( &_c[0], 1 );

                        invalid = ( ix[c].x < 0 ) || ( ix[c].x >= lim.x) || 
                                  ( ix[c].y < 0 ) || ( ix[c].y >= lim.y);
                    } while (invalid);

                    ix[ k ] = ix[ c ];
                    x [ k ] = x [ c ];
                    u [ k ] = u [ c ];
                }
            }

            it.barrier();

            // Copy particles staying in tile
            const int start = _dir_offset[4];

            for( int i = it.get_local_id(0); i < nidx; i += it.get_local_range(0) ) {
                tmp_ix[ start + i ] = ix[i];
                tmp_x [ start + i ] = x[i];
                tmp_u [ start + i ] = u[i];
            }
        });
    });
    q.wait();
}

#if 0
/**
 * @brief Moves particles to the correct tiles
 * 
 * @note Particles are only expected to have moved no more than 1 tile
 *       in each direction
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

    // Get new offsets (prefix scan of np)
    if ( extra ) {
        // Includes extra values in offset calculations
        // Used to reserve space in particle buffer for later injection
        update_tile_info ( tmp, sort.new_np, extra );
    } else {
        update_tile_info ( tmp, sort.new_np );
    }

    // Copy outgoing particles (and particles needing shifting) to staging area
    copy_out ( *this, tmp, sort, periodic );

    // Copy particles from staging area into final positions in partile buffer
    copy_in ( *this, tmp );

    // For debug only, remove from production code
    // validate( "After tile_sort");
}
#endif

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

    // Get new offsets
    if ( extra ) {
        update_tile_info ( tmp, sort.new_np, extra );
    } else {
        update_tile_info ( tmp, sort.new_np );
    }

    // copy all particles to correct tiles in tmp buffer
    copy_sorted( *this, tmp, sort, periodic );

    // swap buffers
    swap_buffers( *this, tmp );

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
    bnd_check ( *this, sort, periodic, queue );

    if ( extra ) {
        // Get new offsets, including extra values in offset calculations
        // Used to reserve space in particle buffer for later injection
        auto total_np = update_tile_info ( tmp, sort.new_np, extra, _dev_tmp_uint32, queue );

        if ( total_np > max_part ) { 

            // grow tmp particle buffer
            tmp.grow_buffer( total_np );

            // copy all particles to correct tiles in tmp buffer
            copy_sorted( *this, tmp, sort, periodic, queue );

            // swap buffers
            swap_buffers( *this, tmp );

            // grow tmp particle buffer for future use
            grow_buffer( max_part );

        } else {
            // Copy outgoing particles (and particles needing shifting) to staging area
            copy_out ( *this, tmp, sort, periodic, queue );

            // Copy particles from staging area into final positions in partile buffer
            copy_in ( *this, tmp, queue );
        }

    } else {
        // Get new offsets
        update_tile_info ( tmp, sort.new_np, queue );

        // Copy outgoing particles (and particles needing shifting) to staging area
        copy_out ( *this, tmp, sort, periodic, queue );

        // Copy particles from staging area into final positions in partile buffer
        copy_in ( *this, tmp, queue );
    }


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

    // 8×1 work items per group
    sycl::range<2> local{ 8, 1 };

    // ntiles.x × ntiles.y groups
    sycl::range<2> global{ ntiles.x, ntiles.y };

    queue.submit([&](sycl::handler &h) {

        auto part_ix = ix;

        auto offset = this->offset;
        auto np = this->np;
        auto ntiles = this -> ntiles;

        h.parallel_for( 
            sycl::nd_range{ global * local, local },
            [=](sycl::nd_item<2> it) { 
            
            const int2 tile_idx = make_int2( it.get_group(0), it.get_group(1));
            const int tile_id = tile_idx.y * ntiles.x + tile_idx.x;

            const auto tile_off = offset[ tile_id ];
            const auto tile_np  = np[ tile_id ];

            int2   * const __restrict__ ix = &part_ix[ tile_off ];

            for( int i = it.get_local_id(0); i < tile_np; i += it.get_local_range(0) ) {
                int2 cell = ix[i];
                cell.x += shift.x;
                cell.y += shift.y;
                ix[i] = cell;
            }
        });
    });
    queue.wait();
}

#define __ULIM __FLT_MAX__

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

    int2 const lb = make_int2( -over, -over );
    int2 const ub = make_int2( nx.x + over, nx.y + over ); 

    std::cout << "Validating particle set, " << msg << "..." << std::endl;

    _dev_tmp_uint32.set(0);

    // Check offset / np buffer
    queue.submit([&](sycl::handler &h) {

        sycl::range<1> global{ ntiles.x * ntiles. y };
        sycl::stream out(8192, 1024, h);
        auto dev_err = _dev_tmp_uint32.ptr();
        auto np = this -> np;
        auto offset = this -> offset;

        h.parallel_for( 
            sycl::nd_range{ global, global },
            [=](sycl::nd_item<1> it) { 
                int tile_id = it.get_global_linear_id();

                if ( np[tile_id] < 0 ) {
                    out << "tile[" << tile_id << "] - bad np (" << np[ tile_id ] << "), should be >= 0\n";
                    dev_err[0] = 1;
                }

                if ( tile_id > 0 ) {
                    auto prev = offset[ tile_id-1] + np[ tile_id-1];
                    if ( prev != offset[ tile_id ] ) {
                        out << "tile[" << tile_id << "] - bad offset (" << offset[ tile_id ] << ")"
                            << ", does not match previous tile info, should be " << prev << '\n';
                        dev_err[0] = 1;
                    }
                } else {
                    if ( offset[ tile_id ] != 0 ) {
                        out << "tile[" << tile_id << "] - bad offset (" << offset[ tile_id ] << "), should be 0\n";
                        dev_err[0] = 1;
                    }
                }        
            });
    });
    queue.wait();
    if ( _dev_tmp_uint32.get() ) {
        std::cerr << "(*error*) " << msg << ": invalid tile information, aborting...\n";
        abort();
    }

    // Check positions
    queue.submit([&](sycl::handler &h) {
        // 8×1 work items per group
        sycl::range<2> local{ 8, 1 };

        // ntiles.x × ntiles.y groups
        sycl::range<2> global{ ntiles.x, ntiles.y };

        sycl::stream out(8192, 1024, h);

        auto dev_err = _dev_tmp_uint32.ptr();
        auto offset = this -> offset;
        auto np = this -> np;

        auto ix = this -> ix;
        auto x  = this -> x;
        //auto u  = this -> u;
        auto ntiles = this -> ntiles;

        h.parallel_for( 
            sycl::nd_range{ global * local, local },
            [=](sycl::nd_item<2> it) { 
            
            const int2 tile_idx = make_int2( it.get_group(0), it.get_group(1));
            const int tile_id = tile_idx.y * ntiles.x + tile_idx.x;

            const auto tile_off = offset[ tile_id ];
            const auto tile_np  = np[ tile_id ];

            int2   const * const __restrict__ t_ix = &ix[ tile_off ];
            float2 const * const __restrict__ t_x  = &x[ tile_off ];
            // float3 const * const __restrict__ t_u  = &u[ tile_off ];

            for( int i = it.get_local_id(0); i < tile_np; i += it.get_local_range(0) ) {
                int err = 0;

                if ((t_ix[i].x < lb.x) || (t_ix[i].x >= ub.x )) { 
                    out << "tile[" << tile_id << "] Invalid ix[" << i << "].x position (" << t_ix[i].x << ")"
                        << ", range = [" << lb.x << "," << ub.x << "]\n";
                    err = 1;
                }
                if ((t_ix[i].y < lb.y) || (t_ix[i].y >= ub.y )) { 
                    out << "tile[" << tile_id << "] Invalid ix[" << i << "].y position (" << t_ix[i].y << ")"
                        << ", range = [" << lb.y << "," << ub.y << "]\n";
                    err = 1;
                }
#if 0
                if ( std::isnan(t_u[i].x) || std::isinf(t_u[i].x) || sycl::fabs(t_u[i].x) >= __ULIM ) {
                    out << "tile[" << tile_id << "] Invalid u[" << i << "].x gen. velocity (" << t_u[i].x <<")\n";
                    err = 1;
                }
                if ( std::isnan(t_u[i].y) || std::isinf(t_u[i].y) || sycl::fabs(t_u[i].y) >= __ULIM ) {
                    out << "tile[" << tile_id << "] Invalid u[" << i << "].y gen. velocity (" << t_u[i].y <<")\n";
                    err = 1;
                }
                if ( std::isnan(t_u[i].z) || std::isinf(t_u[i].z) || sycl::fabs(t_u[i].z) >= __ULIM ) {
                    out << "tile[" << tile_id << "] Invalid u[" << i << "].z gen. velocity (" << t_u[i].z <<")\n";
                    err = 1;
                }
#endif
                if ( t_x[i].x < -0.5f || t_x[i].x >= 0.5f ) {
                    out << "tile[" << tile_id << "] Invalid x[" << i << "].x position (" << t_x[i].x << "), range = [-0.5,0.5[\n";
                    err = 1;
                }
                if ( t_x[i].y < -0.5f || t_x[i].y >= 0.5f ) {
                    out << "tile[" << tile_id << "] Invalid x[" << i << "].y position (" << t_x[i].y << "), range = [-0.5,0.5[\n";
                    err = 1;
                }

                if ( err ) {
                    dev_err[0] = 1;
                    break;
                }
            }
        });
    });
    queue.wait();

    if ( _dev_tmp_uint32.get() ) {
        std::cerr << "(*error*) Invalid particle(s) found, aborting..." << std::endl;
        abort();
    } else {
        std::cout << "Particle set ok." << std::endl;
    }
}

#undef __ULIM
