#include "charge.hpp"

/**
 * @brief Physical boundary conditions for the x direction 
 * 
 * @param bnd_x         Boundary to process, 0 - lower, 1 - upper
 * @param tile_idx_y    Tile index along y direction
 * @param charge        View of tiled charge grid
 * @param bc            Boundary condition
 */
 void charge_bcx( 
    int bnd_x, int tile_idx_y,
    const grid::tiled_view<float> charge,
    const charge::bc_type bc ) {

    const int ystride = charge.tile_ystride();

    if ( bnd_x == 0 ) {
        // Lower boundary
        float * __restrict__ rho = & charge.tile_buffer
            (0,tile_idx_y)      // lower x boundary tile
            [ charge.gc.x.lower ];     // point to first x cell (ix = 0)

        switch( bc.x.lower ) {
        case( charge::bc::reflecting ):
            for( unsigned idx = 0; idx < charge.tile_ext_dims.y; idx ++ ) {
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
        float * __restrict__ rho = & charge.tile_buffer
            (charge.local_ntiles.x-1,tile_idx_y)    // upper x boundary tile
            [ charge.gc.x.lower + charge.tile_dims.x ];    // point to first upper gc (ix = tile_dims.x)

        switch( bc.x.upper ) {
        case( charge::bc::reflecting ):
            for( unsigned idx = 0; idx < charge.tile_ext_dims.y; idx ++ ) {
                const int iy = idx * ystride;

                auto tmp =  rho[ -1 + iy ] + rho[ 1 + iy ];
                rho[ -1 + iy ] = rho[ 1 + iy ] = tmp;
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
 * @param bnd_y         Boundary to process, 0 - lower, 1 - upper
 * @param tile_idx_x    Tile index along x direction
 * @param charge        View of tiled charge grid
 * @param bc            Boundary condition
 */
void charge_bcy( 
    int bnd_y, int tile_idx_x,
    const grid::tiled_view<float> charge,
    const charge::bc_type bc ) {

    const int ystride = charge.tile_ystride();
    
    if ( bnd_y == 0 ) {
        // Lower boundary
        float * __restrict__ rho = & charge.tile_buffer
            (tile_idx_x,0)              // lower y boundary tiles
            [ charge.gc.y.lower * ystride ];   // point to first y cell (iy = 0)

        switch( bc.y.lower ) {
        case( charge::bc::reflecting ):
            for( unsigned idx = 0; idx < charge.tile_ext_dims.x; idx ++ ) {
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
        float * __restrict__ rho = & charge.tile_buffer
            (tile_idx_x,charge.local_ntiles.y-1)   // upper y boundary tiles
            [ (charge.gc.y.lower + charge.tile_dims.y ) * ystride ];  // point to first upper gc (iy = tile_dims.y)

        switch( bc.y.upper ) {
        case( charge::bc::reflecting ):
            for( unsigned idx = 0; idx < charge.tile_ext_dims.x; idx ++ ) {
                const int ix = idx;

                auto tmp =  rho[ ix + (-1)*ystride ] + rho[ ix + (+1)*ystride ];
                rho[ ix + (-1)*ystride ] = rho[ ix + (+1)*ystride ] = tmp;
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
void charge::process_bc() {
    const uint2 ntiles          = rho -> get_local_ntiles();

    // x boundaries
    if ( bc.x.lower > charge::bc::periodic || bc.x.upper > charge::bc::periodic ) {
        #pragma omp parallel for collapse(2)
        for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
            for( unsigned bnd_x : {0,1} ) {
                charge_bcx( bnd_x, ty, rho -> view(), bc );
            }
        }
    }

    // y boundaries
    if ( bc.y.lower > charge::bc::periodic || bc.y.upper > charge::bc::periodic ) {
        #pragma omp parallel for collapse(2)
        for( unsigned bnd_y : { 0,1 } ) {
            for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {
                charge_bcy( bnd_y, tx, rho -> view(), bc );
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
void charge::advance() {

    // Add up current deposited on guard cells
    rho ->  add_from_gc( );

    // Do additional bc calculations if needed
    process_bc();

    // Add neutralizing background
    // This is preferable to initializing rho to this value before charge deposition
    // because it leads to less roundoff errors
    if ( neutral ) rho -> add( *neutral );
    
    // Calculate frho
    fft_forward -> transform( *frho, *rho );

    // Filter charge
    filter -> apply( *frho );

    // Advance iteration count
    iter++;
}

/**
 * @brief Save charge density data to diagnostic file
 * 
 */
void charge::save( const quantity quant ) {

    std::string name = "rho";      // Dataset name
    std::string label = "\\rho";    // Dataset label (for plots)

    grid::tiled<float> * f = nullptr;
    grid::flat<std::complex<float>> * cf = nullptr;

    switch (quant) {
        case quantity::rho :
            f = rho;
            name = "rho";
            label = "\\rho";
            break;
        case quantity::frho :
            cf = frho;
            name = "frho";
            label = "\\mathcal{F}\\,\\rho";
            break;
    }

    zdf::grid_info info = {
        .name = (char *) name.c_str(),
        .ndims = 2,
        .label = (char *) label.c_str(),
        .units = (char *) "e \\omega_n^2 / c",
    };

    zdf::iteration iteration = {
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_n"
    };

    zdf::grid_axis axis[2];

    if ( f != nullptr ) {
        // Real field
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

        info.axis = axis;

        f -> save( info, iteration, "CHARGE" );

    } else {
        // Complex field
        float2 dk = grid::fft::dk( box );

        axis[0] = (zdf::grid_axis) {
            .name = (char *) "ky",
            .min =  - dk.y * ( cf -> get_global_dims().x / 2 ),
            .max =    dk.y * ( cf -> get_global_dims().x / 2 - 1 ),
            .label = (char *) "k_y"
        };

        axis[1] = (zdf::grid_axis) {
            .name = (char *) "kx",
            .min = 0.0,
            .max = (cf -> get_global_dims().y - 1) * dk.x,
            .label = (char *) "k_x"
        };

        info.axis = axis;

        cf -> save( info, iteration, "CHARGE" );
    }
}