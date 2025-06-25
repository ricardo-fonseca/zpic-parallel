#include "current.h"

#include <iostream>

/**
 * @brief Physical boundary conditions for the x direction
 * 
 * @param tile      Tile position on grid
 * @param J         Tile current density & d_J[ gc.x.lower ]
 * @param nx        Number of cells
 * @param ext_nx    Number of cells including guard cells
 * @param bc        Boundary condition
 */
void current_bcx(
    const uint2 tile_idx,
    float3 * const __restrict__ J,
    uint2 const nx, uint2 const ext_nx,
    const current::bc_type bc ) {

    const int ystride = ext_nx.x;

    if ( tile_idx.x == 0 ) {
        // Lower boundary
        switch( bc.x.lower ) {
        case( current::bc::reflecting ):
            for( unsigned idx = 0; idx < ext_nx.y; idx ++ ) {
                // iy includes the y-stride
                const int iy = idx * ystride;

                float jx0 = -J[ -1 + iy ].x + J[ 0 + iy ].x; 
                float jy1 =  J[ -1 + iy ].y + J[ 1 + iy ].y;
                float jz1 =  J[ -1 + iy ].z + J[ 1 + iy ].z;

                J[ -1 + iy ].x = J[ 0 + iy ].x = jx0;
                J[ -1 + iy ].y = J[ 1 + iy ].y = jy1;
                J[ -1 + iy ].z = J[ 1 + iy ].z = jz1;
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
                const int iy = idx * ystride;

                float jx0 =  J[ nx.x-1 + iy ].x - J[ nx.x + 0 + iy ].x; 
                float jy1 =  J[ nx.x-1 + iy ].y + J[ nx.x + 1 + iy ].y;
                float jz1 =  J[ nx.x-1 + iy ].z + J[ nx.x + 1 + iy ].z;

                J[ nx.x-1 + iy ].x = J[ nx.x + 0 + iy ].x = jx0;
                J[ nx.x-1 + iy ].y = J[ nx.x + 1 + iy ].y = jy1;
                J[ nx.x-1 + iy ].z = J[ nx.x + 1 + iy ].z = jz1;
            }
            break;
        default:
            break;
        }
    }
}


/**
 * @brief Physical boundary conditions for the y direction
 * 
 * @param tile      Tile position on grid
 * @param J         Tile current density & d_J[ gc.y.lower * ystride ]
 * @param nx        Number of cells
 * @param ext_nx    Number of cells including guard cells
 * @param bc        Boundary condition
 */
void current_bcy( 
    const uint2 tile_idx,
    float3 * const __restrict__ J,
    uint2 const nx, uint2 const ext_nx,
    const current::bc_type bc ) {

    const int ystride = ext_nx.x;
    
    if ( tile_idx.y == 0 ) {
        // Lower boundary
        switch( bc.y.lower ) {
        case( current::bc::reflecting ):
            for( unsigned idx = 0; idx < ext_nx.x; idx ++ ) {
                const int ix = idx;

                auto jx1 =  J[ ix - ystride ].x + J[ ix + ystride ].x; 
                auto jy0 = -J[ ix - ystride ].y + J[ ix +       0 ].y;
                auto jz1 =  J[ ix - ystride ].z + J[ ix + ystride ].z;

                J[ ix - ystride ].x = J[ ix + ystride ].x = jx1;
                J[ ix - ystride ].y = J[ ix +       0 ].y = jy0;
                J[ ix - ystride ].z = J[ ix + ystride ].z = jz1;
            }
            break;
        default:
            break;
        }
    } else {
        // Upper boundary
        switch( bc.y.upper ) {
        case( current::bc::reflecting ):
            for( unsigned idx = 0; idx < ext_nx.x; idx ++ ) {
                const int ix = idx;

                auto jx1 =  J[ ix + (nx.y-1)*ystride ].x + J[ ix + (nx.y + 1)*ystride ].x; 
                auto jy0 =  J[ ix + (nx.y-1)*ystride ].y - J[ ix + (nx.y + 0)*ystride ].y;
                auto jz1 =  J[ ix + (nx.y-1)*ystride ].z + J[ ix + (nx.y + 1)*ystride ].z;

                J[ ix + (nx.y-1)*ystride ].x = J[ ix + (nx.y + 1)*ystride ].x = jx1;
                J[ ix + (nx.y-1)*ystride ].y = J[ ix + (nx.y + 0)*ystride ].y = jy0;
                J[ ix + (nx.y-1)*ystride ].z = J[ ix + (nx.y + 1)*ystride ].z = jz1;
            }
            break;
        default:
            break;
        }
    }
}

/**
 * @brief Processes "physical" boundary conditions
 * 
 */
void Current::process_bc() {

    const uint2 ntiles          = J -> ntiles;
    const unsigned int tile_vol = J -> tile_vol;
    const uint2 nx              = J -> nx;
    const uint2 ext_nx          = J -> ext_nx;

    // x boundaries
    if ( bc.x.lower > current::bc::periodic || bc.x.upper > current::bc::periodic ) {
        // Loop over tiles
        //  Only lower (0) and upper ( ntiles.x - 1 ) tiles have physical x boundaries

        for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
            for( unsigned tx : { 0u, ntiles.x-1 } ) {

                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                // Start at x cell 0
                const auto x_offset = J -> gc.x.lower;

                float3 * const __restrict__ tile_J = & J->d_buffer[ tile_off + x_offset ];

                current_bcx( tile_idx, tile_J, nx, ext_nx, bc );
            }
        }
    }

    // y boundaries
    if ( bc.y.lower > current::bc::periodic || bc.y.upper > current::bc::periodic ) {

        // Loop over tiles
        //  Only lower (0) and upper ( ntiles.y - 1 ) tiles have physical y boundaries

        for( unsigned ty : { 0u, ntiles.y-1 } ) {
            for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {

                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                // Start at y cell 0
                const auto y_offset = J -> gc.y.lower * ext_nx.x;

                float3 * const __restrict__ tile_J = & J->d_buffer[ tile_off + y_offset ];

                current_bcy( tile_idx, tile_J, nx, ext_nx, bc );
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
void Current::advance() {

    // Add up current deposited on guard cells
    J -> add_from_gc( );
    // J -> copy_to_gc( );

    // Do additional bc calculations if needed
    // Process_bc();

    // Calculate fJ
    fft_forward -> transform( *J, *fJ );

    // Apply filtering
    filter -> apply( *fJ );

    // Advance iteration count
    iter++;
}

/**
 * @brief Save electric current data to diagnostic file
 * 
 * @param jc        Current component to save (0, 1 or 2)
 */
void Current::save( fcomp::cart const jc ) {

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
            std::cerr << "Invalid field component (fc) selected, aborting\n";
            std::exit(1);
    }

    zdf::grid_axis axis[2];
    axis[0] = (zdf::grid_axis) {
    	.name = (char *) "x",
    	.min = 0.0,
    	.max = box.x,
    	.label = (char *) "x",
    	.units = (char *) "c/\\omega_n"
    };

    axis[1] = (zdf::grid_axis) {
        .name = (char *) "y",
    	.min = 0.0,
    	.max = box.y,
    	.label = (char *) "y",
    	.units = (char *) "c/\\omega_n"
    };

    zdf::grid_info info = {
        .name = (char *) vfname.c_str(),
    	.ndims = 2,
    	.label = (char *) vflabel.c_str(),
    	.units = (char *) "e \\omega_n^2 / c",
    	.axis = axis
    };

    info.count[0] = J -> dims.x;
    info.count[1] = J -> dims.y;

    zdf::iteration iteration = {
    	.n = iter,
    	.t = iter * dt,
    	.time_units = (char *) "1/\\omega_n"
    };

    J -> save( jc, info, iteration, "CURRENT" );
}