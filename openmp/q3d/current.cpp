#include "current.h"

#include <iostream>

#include "current.h"

#include <iostream>

/**
 * @brief Physical boundary conditions for the z direction
 * 
 * @param tile      Tile position on grid
 * @param J         Tile current density & d_J[ gc.x.lower ]
 * @param nx        Number of cells
 * @param ext_nx    Number of cells including guard cells
 * @param bc        Boundary condition
 */
template< class T >
void current_bcz(
    const uint2 tile_idx,
    T * const __restrict__ J,
    uint2 const nx, uint2 const ext_nx,
    const current::bc_type bc ) {

    const int jstride = ext_nx.x;

    if ( tile_idx.x == 0 ) {
        // Lower boundary
        switch( bc.x.lower ) {
        case( current::bc::reflecting ):
            for( unsigned idx = 0; idx < ext_nx.y; idx ++ ) {
                // j includes the y-stride
                const int j = idx * jstride;

                auto jz0 = -J[ -1 + j ].z + J[ 0 + j ].z; 
                auto jr1 =  J[ -1 + j ].r + J[ 1 + j ].r;
                auto jθ1 =  J[ -1 + j ].θ + J[ 1 + j ].θ;

                J[ -1 + j ].z = J[ 0 + J ].z = jz0;
                J[ -1 + j ].r = J[ 1 + J ].r = jr1;
                J[ -1 + j ].θ = J[ 1 + J ].θ = jθ1;
            }
            break;
        default:
            break;
        }
    } else {
        // Upper boundary
        switch( bc.x.upper ) {
        case( current::bc::reflecting ):
            for( unsigned idx = 0; idx < ext_nx.y; idx ++ ) {
                const int j = idx * jstride;

                auto jz0 =  J[ nx.x-1 + j ].z - J[ nx.x + 0 + j ].z; 
                auto jr1 =  J[ nx.x-1 + j ].r + J[ nx.x + 1 + j ].r;
                auto jθ1 =  J[ nx.x-1 + j ].θ + J[ nx.x + 1 + j ].θ;

                J[ nx.x-1 + j ].z = J[ nx.x + 0 + j ].x = jz0;
                J[ nx.x-1 + j ].r = J[ nx.x + 1 + j ].y = jr1;
                J[ nx.x-1 + j ].θ = J[ nx.x + 1 + j ].z = jθ1;
            }
            break;
        default:
            break;
        }
    }
}


/**
 * @brief Physical boundary conditions for the radial direction
 * 
 * @note This must only be called for the upper radial boundary (not the
 * @note axial boundary)
 * 
 * @param tile      Tile position on grid
 * @param J         Tile current density & d_J[ gc.y.lower * ystride ]
 * @param nx        Number of cells
 * @param ext_nx    Number of cells including guard cells
 * @param bc        Boundary condition
 */
template< class T >
void current_bcr( 
    const uint2 tile_idx,
    T * const __restrict__ J,
    uint2 const nx, uint2 const ext_nx,
    const current::bc_type bc ) {

    const int jstride = ext_nx.x;
    
    // Upper boundary
    switch( bc.y.upper ) {
    case( current::bc::reflecting ):
        for( unsigned idx = 0; idx < ext_nx.x; idx ++ ) {
            const int i = idx;

            auto jz1 =  J[ i + (nx.y-1)*jstride ].z + J[ i + (nx.y + 1)*jstride ].z; 
            auto jr0 =  J[ i + (nx.y-1)*jstride ].r - J[ i + (nx.y + 0)*jstride ].r;
            auto jθ1 =  J[ i + (nx.y-1)*jstride ].θ + J[ i + (nx.y + 1)*jstride ].θ;

            J[ i + (nx.y-1)*jstride ].z = J[ i + (nx.y + 1)*jstride ].z = jz1;
            J[ i + (nx.y-1)*jstride ].r = J[ i + (nx.y + 0)*jstride ].r = jr0;
            J[ i + (nx.y-1)*jstride ].θ = J[ i + (nx.y + 1)*jstride ].θ = jθ1;
        }
        break;
    default:
        break;
    }
}

/**
 * @brief Processes "physical" boundary conditions
 * 
 */
void Current::process_bc() {

    NOT_IMPLEMENTED

#if 0
    const uint2 ntiles          = J -> ntiles;
    const unsigned int tile_vol = J -> tile_vol;
    const uint2 nx              = J -> nx;
    const uint2 ext_nx          = J -> ext_nx;

    // z boundaries
    if ( bc.x.lower > current::bc::periodic || bc.x.upper > current::bc::periodic ) {
        // Loop over tiles
        //  Only lower (0) and upper ( ntiles.x - 1 ) tiles have physical x boundaries

        #pragma omp parallel for
        for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
            for( unsigned tx : { 0u, ntiles.x-1 } ) {

                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                // Start at z cell 0
                const auto z_offset = J -> gc.x.lower;

                current_bcz( tile_idx, J -> mode0().d_buffer[tile_off + z_offset], nx, ext_nx, bc );

                for( int m = 1; m < nmodes; m++ )
                    current_bcz( tile_idx, J -> mode(m).d_buffer[tile_off + z_offset], nx, ext_nx, bc );
            }
        }
    }

    // y boundaries
    if ( bc.y.upper > current::bc::periodic ) {

        // Loop over tiles
        //  Only upper ( ntiles.y - 1 ) tiles have physical r boundaries
        unsigned ty = ntiles.y-1;

        for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {

            const auto tile_idx = make_uint2( tx, ty );
            const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
            const auto tile_off = tid * tile_vol;

            // Start at r cell 0
            const auto r_offset = J -> gc.y.lower * ext_nx.x;

            current_bcr( tile_idx, J -> mode0().d_buffer[ tile_off + r_offset ], nx, ext_nx, bc );
            for( int m = 1; m < nmodes; m++ )
                current_bcr( tile_idx, J -> mode(m).d_buffer[tile_off + r_offset], nx, ext_nx, bc );
        }
    }
#endif

}

/**
 * @brief Normalize current grid (m = 0)
 * 
 * @param m             Azymuthal mode
 * @param tile_idx      Tile index (x,y)
 * @param ntiles        Number of tiles in grid (x,y)
 * @param d_current     Pointer to current grid
 * @param offset        Offset to position (0,0) on the grid
 * @param nx            Tile grid size
 * @param ext_nx        External tile grid size
 * @param dr            Radial cell size (in simulation units)
 */
void current_norm_0(
    uint2 const tile_idx,
    uint2 const ntiles,
    cyl3<float> * const __restrict__ d_current, int offset, 
    uint2 const nx, uint2 const ext_nx,
    float2 const dx, double dt
) {

    auto tid = tile_idx.y * ntiles.x + tile_idx.x;
    const int tile_off = tid * roundup4( ext_nx.x * ext_nx.y );
    const int jstride = ext_nx.x;

    auto * __restrict__ current = &  d_current[ tile_off + offset ];

    ///@brief radial cell size
    auto dr = dx.y;

    float const dz_dt = dx.x / dt; 
    float const dr_dt = dx.y / dt; 

    int ir0 = tile_idx.y * nx.y;
    for( int j = -1; j < static_cast<int>(nx.y+2); j++ ){
        /// @brief r at center of cell
        float rc   = abs( ir0 + j        ) * dr;
        /// @brief r at lower edge of cell
        float rm   = abs( ir0 + j - 0.5f ) * dr;
        
        float norm_r  = ( ir0 + j == 0 )? 0 : 1.0f / rc;
        float norm_zθ = 1.0f / rm;

        for( int i = -1; i < static_cast<int>(nx.x+2); i++ ){
            current[ j * jstride +i ].z *= dz_dt * norm_zθ;
            current[ j * jstride +i ].r *= dr_dt * norm_r;
            current[ j * jstride +i ].θ *= dr_dt * norm_zθ;
        }
    }

    // Axial boundary
    // Fold values for r < 0 back into simulation domain
    if ( ir0 == 0 ) {

        // alternative, signθ = -(-1)^m
        const auto signθ = -1.f;

        for( int i = -1; i < static_cast<int>(nx.x+2); i++ ){
            current[ i + 1 * jstride ].z += current[ i +   0  * jstride ].z;
            current[ i + 2 * jstride ].z += current[ i + (-1) * jstride ].z;

            current[ i + 1 * jstride ].r -= current[ i +   (-1) * jstride ].r;

            current[ i + 1 * jstride ].θ += signθ * current[ i +   0  * jstride ].θ;
            current[ i + 2 * jstride ].θ += signθ * current[ i + (-1) * jstride ].θ;

            // The following values are used for diagnostic output only
            current[ i + 0 * jstride ].z  = current[ i + 1 * jstride ].z;
            current[ i + 0 * jstride ].r  = 0;
            current[ i + 0 * jstride ].θ  = signθ * current[ i +   1  * jstride ].θ;
        }
    }
}

/**
 * @brief Normalize current grid, modes m > 0
 * 
 * @param m 
 * @param tile_idx 
 * @param ntiles 
 * @param d_current 
 * @param offset 
 * @param nx 
 * @param ext_nx 
 * @param dx 
 * @param dt 
 */
void current_norm_m(
    unsigned const m,
    uint2 const tile_idx,
    uint2 const ntiles,
    cyl3< std::complex<float> > * const __restrict__ d_current, int offset, 
    uint2 const nx, uint2 const ext_nx,
    float2 const dx, double dt
) {

    auto tid = tile_idx.y * ntiles.x + tile_idx.x;
    const int tile_off = tid * roundup4( ext_nx.x * ext_nx.y );
    const int jstride = ext_nx.x;

    auto * __restrict__ current = &  d_current[ tile_off + offset ];

    ///@brief radial cell size
    auto dr = dx.y;

    float const dz_dt = dx.x / dt; 
    float const dr_dt = dx.y / dt; 

    ///@brief Normalization for jθ
    const std::complex<float> norm_θ{0,-2/(m*static_cast<float>(dt))};

    int ir0 = tile_idx.y * nx.y;
    for( int j = -1; j < static_cast<int>(nx.y+2); j++ ){
        /// @brief r at center of cell
        float rc   = abs( ir0 + j        ) * dr;
        /// @brief r at lower edge of cell
        float rm   = abs( ir0 + j - 0.5f ) * dr;
        
        float norm_r  = ( ir0 + j == 0 )? 0 : 2.f / rc;
        float norm_z  = 2.f / rm;

        for( int i = -1; i < static_cast<int>(nx.x+2); i++ ){
            current[ j * jstride +i ].z *= dz_dt * norm_z ;
            current[ j * jstride +i ].r *= dr_dt * norm_r;
            current[ j * jstride +i ].θ *= norm_θ;
        }
    }

    // Axial boundary
    // Fold values for r < 0 back into simulation domain
    if ( ir0 == 0 ) {

        // alternative, signθ = -(-1)^m
        const auto signθ = ( m & 1 ) ? 1.f : -1.f;

        for( int i = -1; i < static_cast<int>(nx.x+2); i++ ){
            current[ i + 1 * jstride ].z += current[ i +   0  * jstride ].z;
            current[ i + 2 * jstride ].z += current[ i + (-1) * jstride ].z;

            current[ i + 1 * jstride ].r -= current[ i +   (-1) * jstride ].r;

            current[ i + 1 * jstride ].θ += signθ * current[ i +   0  * jstride ].θ;
            current[ i + 2 * jstride ].θ += signθ * current[ i + (-1) * jstride ].θ;

            // The following values are used for diagnostic output only
            current[ i + 0 * jstride ].z  = current[ i + 1 * jstride ].z;
            current[ i + 0 * jstride ].r  = 0;
            current[ i + 0 * jstride ].θ  = signθ * current[ i +   1  * jstride ].θ;
        }
    }
}


/**
 * @brief Normalize grid values for ring particles
 * 
 */
void Current::normalize() {

    auto & J0 = J -> mode0();
    const auto ntiles  = J0.ntiles;
    const auto offset  = J0.offset;
    const auto nx      = J0.nx;
    const auto ext_nx  = J0.ext_nx;

    // Normalize mode m = 0
    #pragma omp parallel for
    for( unsigned tid = 0; tid < ntiles.x * ntiles.y; tid ++ ) {
        auto tile_idx = uint2{ tid % ntiles.x, tid / ntiles.x };
        current_norm_0( tile_idx, ntiles, J0.d_buffer, offset, nx, ext_nx, dx, dt );
    }

    // Normalize higher order modes
    #pragma omp parallel for
    for( unsigned tid = 0; tid < ntiles.x * ntiles.y; tid ++ ) {
        auto tile_idx = uint2{ tid % ntiles.x, tid / ntiles.x };
        for( unsigned m = 1; m < nmodes; m++ ) {
            auto & Jm = J -> mode(m);
            current_norm_m( m, tile_idx, ntiles, Jm.d_buffer, offset, nx, ext_nx, dx, dt );
        }
    }
}

/**
 * @brief Advance electric current to next iteration
 * 
 * Adds up current deposited on guard cells and (optionally) applies digital filtering
 * 
 */
void Current::advance() {

    // Add up current deposited on guard cells
    J -> add_from_gc( );
    J -> copy_to_gc( );

    // Do additional bc calculations if needed
    // Process_bc();

    // Normalize for ring particles
    normalize();

    // Apply filtering
    // filter -> apply( *J );

    // Advance iteration count
    iter++;

    // I'm not sure if this should be before or after `iter++`
    // Note that it only affects the axis range on output data
    if ( moving_window.needs_move( iter * dt ) )
        moving_window.advance();
}

/**
 * @brief Save current density to file
 * 
 * @param fc        Which current component to save (r, θ or z)
 * @param m         Mode
 */
void Current::save( const fcomp::cyl jc, unsigned m ) {

    std::string vfname  = "J" + std::to_string(m);      // Dataset name
    std::string vflabel = "J^" + std::to_string(m) + "_";    // Dataset label (for plots)
    std::string path{"CURRENT"};

    switch ( jc ) {
        case( fcomp::z ) :
            vfname  += 'z';
            vflabel += 'z';
            break;
        case( fcomp::r ) :
            vfname  += 'r';
            vflabel += 'r';
            break;
        case( fcomp::θ ) :
            vfname  += "θ";
            vflabel += "\\theta";
            break;
        default:
            std::cerr << "Invalid current component (jc) selected, aborting\n";
            std::exit(1);
    }

    zdf::grid_axis axis[2];
    axis[0] = (zdf::grid_axis) {
        .name = (char *) "z",
        .min = 0.0 + moving_window.motion(),
        .max = box.x + moving_window.motion(),
        .label = (char *) "z",
        .units = (char *) "c/\\omega_n"
    };

    axis[1] = (zdf::grid_axis) {
        .name = (char *) "r",
        .min = -dx.y/2,
        .max = box.y-dx.y/2,
        .label = (char *) "r",
        .units = (char *) "c/\\omega_n"
    };

    zdf::grid_info info = {
        .name = (char *) vfname.c_str(),
    	.label = (char *) vflabel.c_str(),
    	.units = (char *) "e \\omega_n^2 / c",
    	.axis = axis
    };

    zdf::iteration iteration = {
    	.n = iter,
    	.t = iter * dt,
    	.time_units = (char *) "1/\\omega_n"
    };

    J -> save( m, jc, info, iteration, path );
}