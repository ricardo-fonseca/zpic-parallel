#include "charge.hpp"

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
void charge::process_bc() {
    const uint2 ntiles          = rho -> get_local_ntiles();
    const uint2 tile_dims       = rho -> tile_dims;
    const uint2 tile_ext_dims   = rho -> tile_ext_dims;

    // x boundaries
    if ( bc.x.lower > charge::bc::periodic || bc.x.upper > charge::bc::periodic ) {
        // Loop over tiles
        //  Only lower (0) and upper ( ntiles.x - 1 ) tiles have physical x boundaries

        #pragma omp parallel for collapse(2)
        for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
            for( unsigned tx : { 0u, ntiles.x-1 } ) {

                const auto tile_idx = make_uint2( tx, ty );

                // Start at x cell 0
                const auto x_offset = rho -> gc.x.lower;

                float * const __restrict__ tile_rho = & rho->tile_buffer(tx,ty)[ x_offset ];

                charge_bcx( tile_idx, tile_rho, tile_dims, tile_ext_dims, bc );
            }
        }
    }

    // y boundaries
    if ( bc.y.lower > charge::bc::periodic || bc.y.upper > charge::bc::periodic ) {

        // Loop over tiles
        //  Only lower (0) and upper ( ntiles.y - 1 ) tiles have physical y boundaries

        #pragma omp parallel for collapse(2)
        for( unsigned ty : { 0u, ntiles.y-1 } ) {
            for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {

                const auto tile_idx = make_uint2( tx, ty );

                // Start at y cell 0
                const auto y_offset = rho -> gc.y.lower * tile_ext_dims.x;

                float * const __restrict__ tile_rho = & rho->tile_buffer(tx,ty)[ y_offset ];

                charge_bcy( tile_idx, tile_rho, tile_dims, tile_ext_dims, bc );
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
    // Currently disabled
    // process_bc();

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