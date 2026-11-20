#include "current.hpp"
#include <iostream>

namespace current_bc {

/**
 * @brief Normal / tangential current components for walls normal to x
 *        (dir = 0) or y (dir = 1)
 */
template< coord::cart dir > struct comp;

template<> struct comp< coord::x > {
    static float & n ( float3 & v ) { return v.x; }
    static float & t1( float3 & v ) { return v.y; }
    static float & t2( float3 & v ) { return v.z; }
};

template<> struct comp< coord::y > {
    static float & n ( float3 & v ) { return v.y; }
    static float & t1( float3 & v ) { return v.x; }
    static float & t2( float3 & v ) { return v.z; }
};

/**
* @brief Check if boundary condition is a conducting wall
* 
* @param t     Boundary condition type
* @return      True if t is PEC, PMC or PEC (absorbing), false otherwise
*/
inline bool is_conducting( current::bc::type t ) {
    return t == current::bc::pec || t == current::bc::pmc || t == current::bc::pec_absorbing;
}

/**
 * @brief Lower wall (pec / pmc) boundary condition for one line of cells
 * 
 * The wall sits on node 0. Layout along the line (stride s):
 * 
 *  p[-s]   normal: -1/2  tangential: -1 (guard)
 *  p[ 0]   normal: +1/2  tangential: wall node
 *  p[+s]   normal: +3/2  tangential: +1
 * 
 * Fold (particle reflection, same for pec and pmc): the current deposited
 * beyond the wall is mirrored into the domain, with the normal component
 * reversed. The deposit on the wall node is its own mirror and is unchanged.
 * pec_absorbing walls skip the fold: the current deposited beyond the wall
 * is overwritten by the image values below.
 * 
 * Images (wall type dependent):
 *  pec - opposite sign image charges: normal J even, tangential J odd, so
 *        the tangential J on the wall node is 0
 *  pmc - same sign image charges: normal J odd, tangential J even, so the
 *        tangential J on the wall node is doubled (fold step only)
 * 
 * @param p     Pointer to the wall node
 * @param s     Stride along the line normal to the wall (1 or ystride)
 * @param type  Wall type (pec, pmc or pec_absorbing)
 * @param fold  Fold guard cell current and correct wall node (once per step)
 */
template< coord::cart dir >
void lower_conducting( float3 * const p, int const s, current::bc::type const type, bool const fold ) {
    using C = comp<dir>;
    // pec_absorbing walls use pec images, but particles are not reflected
    const bool pec     = ( type != current::bc::pmc );
    const bool reflect = ( type != current::bc::pec_absorbing );

    if ( fold ) {
        if ( reflect ) {
            C::n ( p[ 0 ] ) -= C::n ( p[ -s ] );
            C::t1( p[ s ] ) += C::t1( p[ -s ] );
            C::t2( p[ s ] ) += C::t2( p[ -s ] );
        }

        if ( ! pec ) {
            C::t1( p[ 0 ] ) *= 2;
            C::t2( p[ 0 ] ) *= 2;
        }
    }

    if ( pec ) {
        C::t1( p[ 0 ] ) = 0;
        C::t2( p[ 0 ] ) = 0;
    }

    const float pn = pec ? +1.0f : -1.0f;   // normal component parity
    const float pt = -pn;                   // tangential component parity

    C::n ( p[ -s ] ) = pn * C::n ( p[ 0 ] );
    C::t1( p[ -s ] ) = pt * C::t1( p[ s ] );
    C::t2( p[ -s ] ) = pt * C::t2( p[ s ] );
}

/**
 * @brief Upper wall (pec / pmc) boundary condition for one line of cells
 * 
 * The wall sits on node n (first upper guard cell). Layout along the line
 * (stride s):
 * 
 *  p[-2s]  normal: n-3/2         tangential: n-2
 *  p[ -s]  normal: n-1/2         tangential: n-1
 *  p[  0]  normal: n+1/2 (guard) tangential: wall node (guard)
 *  p[ +s]  normal: n+3/2 (guard) tangential: n+1 (guard)
 * 
 * See lower() for the fold / image rules.
 * 
 * @param p     Pointer to the wall node
 * @param s     Stride along the line normal to the wall (1 or ystride)
 * @param type  Wall type (pec, pmc or pec_absorbing)
 * @param fold  Fold guard cell current and correct wall node (once per step)
 */
template< coord::cart dir >
void upper_conducting( float3 * const p, int const s, current::bc::type const type, bool const fold ) {
    using C = comp<dir>;
    // pec_absorbing walls use pec images, but particles are not reflected
    const bool pec     = ( type != current::bc::pmc );
    const bool reflect = ( type != current::bc::pec_absorbing );

    if ( fold ) {
        if ( reflect ) {
            C::n ( p[  -s ] ) -= C::n ( p[ 0 ] );
            C::n ( p[ -2*s] ) -= C::n ( p[ s ] );
            C::t1( p[  -s ] ) += C::t1( p[ s ] );
            C::t2( p[  -s ] ) += C::t2( p[ s ] );
        }

        if ( ! pec ) {
            C::t1( p[ 0 ] ) *= 2;
            C::t2( p[ 0 ] ) *= 2;
        }
    }

    if ( pec ) {
        C::t1( p[ 0 ] ) = 0;
        C::t2( p[ 0 ] ) = 0;
    }

    const float pn = pec ? +1.0f : -1.0f;   // normal component parity
    const float pt = -pn;                   // tangential component parity

    C::n ( p[ 0 ] ) = pn * C::n ( p[  -s  ] );
    C::n ( p[ s ] ) = pn * C::n ( p[ -2*s ] );
    C::t1( p[ s ] ) = pt * C::t1( p[  -s  ] );
    C::t2( p[ s ] ) = pt * C::t2( p[  -s  ] );
}

} // namespace current_bc


/**
 * @brief Physical boundary conditions for the x direction 
 * 
 * @param bnd_x         Boundary to process, 0 - lower, 1 - upper
 * @param tile_idx_y    Tile index along y direction
 * @param current       View of tiled current grid
 * @param local_bc            Boundary condition
 */
void current_bcx(
    unsigned bnd_x, unsigned tile_idx_y,
    const grid::tiled_view<float3> J_grid,
    const current::bc_type local_bc,
    const bool fold ) {

    const int ystride = J_grid.tile_ystride();

    if ( bnd_x == 0 ) {
        // Lower boundary
        const auto type = local_bc.x.lower;
        if ( ! current_bc::is_conducting( type ) ) return;

        // Point to ix = 0 on the first (extended) row of the lower x boundary tile
        float3 * const J = & J_grid.tile_buffer( 0, tile_idx_y )[ J_grid.gc.x.lower ];

        for( unsigned idx = 0;
             idx < J_grid.tile_ext_dims.y; 
             idx ++ ) {
            current_bc::lower_conducting< coord::x >( & J[ idx * ystride ], 1, type, fold );
        }
    } else {
        // Upper boundary
        const auto type = local_bc.x.upper;
        if ( ! current_bc::is_conducting( type ) ) return;

        // Point to ix = nx (first upper guard cell) on the first (extended) row
        // of the upper x boundary tile
        float3 * const J = & J_grid.tile_buffer( J_grid.local_ntiles.x - 1, tile_idx_y )
            [ J_grid.gc.x.lower + J_grid.tile_dims.x ];

        for( unsigned idx = 0;
             idx < J_grid.tile_ext_dims.y; 
             idx ++ ) {
            current_bc::upper_conducting< coord::x >( & J[ idx * ystride ], 1, type, fold );
        }
    }
}

/**
 * @brief Physical boundary conditions for the y direction 
 * 
 * @param bnd_y         Boundary to process, 0 - lower, 1 - upper
 * @param tile_idx_x    Tile index along x direction
 * @param current       View of tiled current grid
 * @param local_bc      Boundary condition
 */
void current_bcy( 
    unsigned bnd_y, unsigned tile_idx_x,
    const grid::tiled_view<float3> J_grid,
    const current::bc_type local_bc,
    const bool fold ) {

    const int ystride = J_grid.tile_ystride();
    
    if ( bnd_y == 0 ) {
        // Lower boundary
        const auto type = local_bc.y.lower;
        if ( ! current_bc::is_conducting( type ) ) return;

        // Point to iy = 0 on the first (extended) column of the lower y boundary tile
        float3 * const J = & J_grid.tile_buffer( tile_idx_x, 0 )[ J_grid.gc.y.lower * ystride ];

        for( unsigned idx = 0;
             idx < J_grid.tile_ext_dims.x; 
             idx ++ ) {
            current_bc::lower_conducting< coord::y >( & J[ idx ], ystride, type, fold );
        }
    } else {
        // Upper boundary
        const auto type = local_bc.y.upper;
        if ( ! current_bc::is_conducting( type ) ) return;

        // Point to iy = ny (first upper guard row) on the first (extended)
        // column of the upper y boundary tile
        float3 * const J = & J_grid.tile_buffer( tile_idx_x, J_grid.local_ntiles.y - 1 )
            [ ( J_grid.gc.y.lower + J_grid.tile_dims.y ) * ystride ];
        
        for( unsigned idx = 0;
             idx < J_grid.tile_ext_dims.x; 
             idx ++ ) {
            current_bc::upper_conducting< coord::y >( & J[ idx ], ystride, type, fold );
        }
    }
}

/**
 * @brief Processes "physical" boundary conditions
 * 
 */
void current::process_bc( const bool fold ) {

    const uint2 ntiles          = J -> get_local_ntiles();

    // x boundaries
    if ( local_bc.x.lower > current::bc::periodic || local_bc.x.upper > current::bc::periodic ) {
        #pragma omp parallel for collapse(2)
        for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
            for( unsigned bnd_x : {0,1} ) {
                current_bcx( bnd_x, ty, J -> view(), local_bc, fold );
            }
        }
    }

    // y boundaries
    if ( local_bc.y.lower > current::bc::periodic || local_bc.y.upper > current::bc::periodic ) {
        #pragma omp parallel for collapse(2)
        for( unsigned bnd_y : { 0,1 } ) {
            for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {
                current_bcy( bnd_y, tx, J -> view(), local_bc, fold );
            }
        }
    }
}


/**
 * @brief Advance electric current to next iteration
 * 
 * Adds up current deposited on guard cells and (optionally) applies digital filtering
 * 
 */
void current::advance() {

    // Add up current deposited on guard cells
    J -> add_from_gc( );
    J -> copy_to_gc( );

    // Fold current deposited beyond walls back into the domain and set the
    // image currents
    process_bc( true );

    // Apply filtering. Every filter pass ends with copy_to_gc(), which leaves
    // physical edge guard cells untouched, so the image currents at walls
    // are refreshed after each pass
    filter -> apply( *J, true, [this]{ process_bc( false ); } );

    // Advance iteration count
    iter++;

    // Moving window is checked after the iteration count is advanced so
    // that the window position matches the time label used by save()
    // (iter * dt). This only affects the axis range on output data and must
    // follow the same ordering as emf::advance()
    if ( mov_window.needs_move( iter * dt ) )
        mov_window.advance();
}

/**
 * @brief Save electric current data to diagnostic file
 * 
 * @param jc        Current component to save
 */
void current::save( const fcomp::cart jc ) const {

    std::string vfname = "J";      // Dataset name
    std::string vflabel = "J_";    // Dataset label (for plots)

    switch ( jc ) {
        case( fcomp::x ) :
            vfname  += 'x';
            vflabel += 'x';
            break;
        case( fcomp::y ) :
            vfname  += 'y';
            vflabel += 'y';
            break;
        case( fcomp::z ) :
            vfname  += 'z';
            vflabel += 'z';
            break;
        default:
            mpi::fatal( "Invalid field component (fc) selected" );
    }

    zdf::grid_info info = {
        .name = (char *) vfname.c_str(),
        .ndims = 2,
        .label = (char *) vflabel.c_str(),
        .units = (char *) "e \\omega_n^2 / c",
    };

    zdf::iteration iteration = {
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_n"
    };

    zdf::grid_axis axis[2];

    // x axis follows the moving window (motion() is 0 if inactive)
    axis[0] = zdf::grid_axis {
        .name = (char *) "x",
        .min = mov_window.motion(),
        .max = box.x + mov_window.motion(),
        .label = (char *) "x",
        .units = (char *) "c/\\omega_n"
    };

    axis[1] = zdf::grid_axis {
        .name = (char *) "y",
        .min = 0.0,
        .max = box.y,
        .label = (char *) "y",
        .units = (char *) "c/\\omega_n"
    };

    info.axis = axis;
    J -> save( jc, info, iteration, "CURRENT" );
}
