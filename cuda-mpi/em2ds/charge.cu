#include "charge.hpp"
#include "core/gpu.cuh"

namespace kernel {

/**
 * @brief Physical boundary conditions for the x direction
 * 
 * @note Launch with grid( 2, local_ntiles.y ): blockIdx.x selects the
 *       boundary (0 - lower, 1 - upper), blockIdx.y the tile row
 * 
 * @param rho_grid      View of tiled charge grid
 * @param local_bc      Local rank boundary conditions
 */
__global__
void charge_bcx( 
    const grid::tiled_view<float> rho_grid,
    const charge::bc_type local_bc ) {

    int bnd_x = blockIdx.x;
    int tile_idx_y = blockIdx.y;

    const int ystride = rho_grid.tile_ystride();

    if ( bnd_x == 0 ) {
        // Lower boundary
        float * __restrict__ rho = & rho_grid.tile_buffer
            (0,tile_idx_y)      // lower x boundary tile
            [ rho_grid.gc.x.lower ];     // point to first x cell (ix = 0)

        switch( local_bc.x.lower ) {
        case( charge::bc::reflecting ):
            for( unsigned idx = gpu::block::thread_rank();
                 idx < rho_grid.tile_ext_dims.y; 
                 idx += gpu::block::num_threads() ) {
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
        float * __restrict__ rho = & rho_grid.tile_buffer
            (rho_grid.local_ntiles.x-1,tile_idx_y)    // upper x boundary tile
            [ rho_grid.gc.x.lower + rho_grid.tile_dims.x ];    // point to first upper gc (ix = tile_dims.x)

        switch( local_bc.x.upper ) {
        case( charge::bc::reflecting ):
            for( unsigned idx = gpu::block::thread_rank();
                idx < rho_grid.tile_ext_dims.y; 
                idx += gpu::block::num_threads() ) {
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
 * @note Launch with grid( local_ntiles.x, 2 ): blockIdx.x selects the tile
 *       column, blockIdx.y the boundary (0 - lower, 1 - upper)
 * 
 * @param rho_grid      View of tiled charge grid
 * @param local_bc      Local rank boundary conditions
 */
__global__
void charge_bcy( 
    const grid::tiled_view<float> rho_grid,
    const charge::bc_type local_bc ) {

    int tile_idx_x = blockIdx.x;
    int bnd_y = blockIdx.y;

    const int ystride = rho_grid.tile_ystride();
    
    if ( bnd_y == 0 ) {
        // Lower boundary
        float * __restrict__ rho = & rho_grid.tile_buffer
            (tile_idx_x,0)              // lower y boundary tiles
            [ rho_grid.gc.y.lower * ystride ];   // point to first y cell (iy = 0)

        switch( local_bc.y.lower ) {
        case( charge::bc::reflecting ):
            for( unsigned idx = gpu::block::thread_rank();
                 idx < rho_grid.tile_ext_dims.x; 
                 idx += gpu::block::num_threads() ) {
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
        float * __restrict__ rho = & rho_grid.tile_buffer
            (tile_idx_x,rho_grid.local_ntiles.y-1)   // upper y boundary tiles
            [ (rho_grid.gc.y.lower + rho_grid.tile_dims.y ) * ystride ];  // point to first upper gc (iy = tile_dims.y)

        switch( local_bc.y.upper ) {
        case( charge::bc::reflecting ):
            for( unsigned idx = gpu::block::thread_rank();
                idx < rho_grid.tile_ext_dims.x;
                idx += gpu::block::num_threads() ) {
                const int ix = idx;

                auto tmp =  rho[ ix - ystride ] + rho[ ix + ystride ];
                rho[ ix - ystride ] = rho[ ix + ystride ] = tmp;
            }
            break;
        default:
            break;
        }
    }
}

}

/**
 * @brief Processes "physical" boundary conditions
 * 
 * @note Physical boundaries are only applied on ranks that sit on the
 *       corresponding edge of the global domain; internal partition
 *       boundaries were already handled by add_from_gc()
 */
void charge::process_bc() {

    dim3 block( 256 );
    const uint2 ntiles          = rho -> get_local_ntiles();

    // x boundaries
    if ( local_bc.x.lower > charge::bc::periodic || local_bc.x.upper > charge::bc::periodic ) {
        dim3 grid( 2, ntiles.y );
        kernel::charge_bcx <<< grid, block >>> ( 
            rho -> view(), local_bc
        );
    }

    // y boundaries
    if ( local_bc.y.lower > charge::bc::periodic || local_bc.y.upper > charge::bc::periodic ) {
        dim3 grid( ntiles.x, 2 );
        kernel::charge_bcy <<< grid, block >>> ( 
            rho -> view(), local_bc
        );
    }
}


/**
 * @brief Advance charge density to next iteration
 * 
 * Adds up charge deposited on guard cells, applies physical boundary
 * conditions and the neutralizing background, transforms to k-space and
 * filters the result.
 * 
 */
void charge::advance() {

    // Add up charge deposited on guard cells
    rho ->  add_from_gc( );

    // Do additional bc calculations if needed
    process_bc();

    // Add neutralizing background
    // This is preferable to initializing rho to this value before charge deposition
    // because it leads to less roundoff errors
    if ( neutral ) rho -> add( *neutral );
    
    // Calculate frho
    fft_forward -> transform( *frho, *rho );

    // Filter charge (k-space only, rho is left unfiltered)
    filter -> apply( *frho );

    // Advance iteration count
    iter++;
}

/**
 * @brief Save charge density data to diagnostic file
 * 
 * @param quant     Which quantity to save (rho or frho)
 */
void charge::save( const quantity quant ) {

    std::string name = "rho";      // Dataset name
    std::string label = "\\rho";    // Dataset label (for plots)

    grid::tiled<float> * f = nullptr;
    grid::flat<charge::complex_t> * cf = nullptr;

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
        axis[0] = zdf::grid_axis {
            .name = (char *) "x",
            .min = 0.0,
            .max = box.x,
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

        f -> save( info, iteration, "CHARGE" );

    } else {
        // Complex field
        // kx: 0 ... nx/2 (r2c half spectrum)
        // ky: kspace_save() rotates by ceil(ny/2) rows, so it runs from
        //     -floor(ny/2) to floor((ny-1)/2) modes
        const float2 dk = grid::fft::dk( box );
        const int nky = cf -> get_global_dims().y;

        axis[0] = zdf::grid_axis {
            .name = (char *) "kx",
            .min = 0.0,
            .max = (cf -> get_global_dims().x - 1) * dk.x,
            .label = (char *) "k_x",
            .units = (char *) "\\omega_n/c"
        };

        axis[1] = zdf::grid_axis {
            .name = (char *) "ky",
            .min =  - dk.y * ( nky / 2 ),
            .max =    dk.y * ( (nky - 1) / 2 ),
            .label = (char *) "k_y",
            .units = (char *) "\\omega_n/c"
        };

        info.axis = axis;

        grid::fft::kspace_save( *cf, info, iteration, "CHARGE");
    }
}
