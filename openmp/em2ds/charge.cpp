#include "charge.h"

#include <iostream>

/**
 * @brief Physical boundary conditions for the x direction
 * 
 * @param tile      Tile position on grid
 * @param rho       Tile charge density & d_rho[ gc.y.lower * ystride ]
 * @param nx        Number of cells
 * @param ext_nx    Number of cells including guard cells
 * @param bc        Boundary condition
 */
void charge_bcx( 
    const uint2 tile_idx,
    float * const __restrict__ rho,
    uint2 const nx, uint2 const ext_nx,
    const charge::bc_type bc ) {

    const int ystride = ext_nx.x;

    if ( tile_idx.x == 0 ) {
        // Lower boundary
        switch( bc.x.lower ) {
        case( charge::bc::reflecting ):
            for( unsigned idx = 0; idx < ext_nx.y; idx ++ ) {
                // iy includes the y-stride
                const int iy = idx * ystride;

                auto tmp = rho[ -1 + iy ] + rho[ 1 + iy ];
                rho[ -1 + iy ] = rho[ 1 + iy ] = tmp;
            }
            break;
        default:
            break;
        }
    } else {
        // Upper boundary
        switch( bc.x.upper ) {
        case( charge::bc::reflecting ):
            for( unsigned idx = 0; idx < ext_nx.y; idx ++ ) {
                const int iy = idx * ystride;

                auto tmp =  rho[ nx.x-1 + iy ] + rho[ nx.x + 1 + iy ];
                rho[ nx.x-1 + iy ] = rho[ nx.x + 1 + iy ] = tmp;
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
 * @param rho       Tile charge density & d_rho[ gc.y.lower * ystride ]
 * @param nx        Number of cells
 * @param ext_nx    Number of cells including guard cells
 * @param bc        Boundary condition
 */
void charge_bcy( 
    const uint2 tile_idx,
    float * const __restrict__ rho,
    uint2 const nx, uint2 const ext_nx,
    const charge::bc_type bc ) {

    const int ystride = ext_nx.x;
    
    if ( tile_idx.y == 0 ) {
        // Lower boundary
        switch( bc.y.lower ) {
        case( charge::bc::reflecting ):
            for( unsigned idx = 0; idx < ext_nx.x; idx ++ ) {
                const int ix = idx;

                auto tmp =  rho[ ix - ystride ] + rho[ ix + ystride ];
                rho[ ix - ystride ] = rho[ ix + ystride ] = tmp;
            }
            break;
        default:
            break;
        }
    } else {
        // Upper boundary
        switch( bc.y.upper ) {
        case( charge::bc::reflecting ):
            for( unsigned idx = 0; idx < ext_nx.x; idx ++ ) {
                const int ix = idx;

                auto tmp =  rho[ ix + (nx.y-1)*ystride ] + rho[ ix + (nx.y + 1)*ystride ];
                rho[ ix + (nx.y-1)*ystride ] = rho[ ix + (nx.y + 1)*ystride ] = tmp;
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
void Charge::process_bc() {
    const uint2 ntiles          = rho -> ntiles;
    const unsigned int tile_vol = rho -> tile_vol;
    const uint2 nx              = rho -> nx;
    const uint2 ext_nx          = rho -> ext_nx;

    // x boundaries
    if ( bc.x.lower > charge::bc::periodic || bc.x.upper > charge::bc::periodic ) {
        // Loop over tiles
        //  Only lower (0) and upper ( ntiles.x - 1 ) tiles have physical x boundaries

        for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
            for( unsigned tx : { 0u, ntiles.x-1 } ) {

                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                // Start at x cell 0
                const auto x_offset = rho -> gc.x.lower;

                float * const __restrict__ tile_rho = & rho->d_buffer[ tile_off + x_offset ];

                charge_bcx( tile_idx, tile_rho, nx, ext_nx, bc );
            }
        }
    }

    // y boundaries
    if ( bc.y.lower > charge::bc::periodic || bc.y.upper > charge::bc::periodic ) {

        // Loop over tiles
        //  Only lower (0) and upper ( ntiles.y - 1 ) tiles have physical y boundaries

        for( unsigned ty : { 0u, ntiles.y-1 } ) {
            for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {

                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                // Start at y cell 0
                const auto y_offset = rho -> gc.y.lower * ext_nx.x;

                float * const __restrict__ tile_rho = & rho->d_buffer[ tile_off + y_offset ];

                charge_bcx( tile_idx, tile_rho, nx, ext_nx, bc );
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
void Charge::advance() {

    // Add up current deposited on guard cells
    rho ->  add_from_gc( );
    // rho ->  copy_to_gc( );

    // Do additional bc calculations if needed
    // process_bc();

	// Add neutralizing background
	// This is preferable to initializing rho to this value before charge deposition
	// because it leads to less roundoff errors
    if ( neutral ) rho -> add( *neutral );
    
    // Calculate frho
    fft_forward -> transform( *rho, *frho );

    // Filter charge
    filter -> apply( *frho );

    // Advance iteration count
    iter++;
}

/**
 * @brief Save charge density data to diagnostic file
 * 
 */
void Charge::save( ) {

    std::string name = "rho";      // Dataset name
    std::string label = "\\rho";    // Dataset label (for plots)

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
        .name = (char *) name.c_str(),
    	.ndims = 2,
    	.label = (char *) label.c_str(),
    	.units = (char *) "e \\omega_n^2 / c",
    	.axis = axis
    };

    info.count[0] = rho ->  dims.x;
    info.count[1] = rho ->  dims.y;

    zdf::iteration iteration = {
    	.n = iter,
    	.t = iter * dt,
    	.time_units = (char *) "1/\\omega_n"
    };

    rho ->  save( info, iteration, "CHARGE" );
}