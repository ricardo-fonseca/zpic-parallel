#include "current.hpp"

namespace kernel {

/**
 * @brief Physical boundary conditions for the x direction
 * 
 * @warning This assumes that the current is staggered
 * 
 * @note Launch with grid( 2, local_ntiles.y ): blockIdx.x selects the
 *       boundary (0 - lower, 1 - upper), blockIdx.y the tile row
 * 
 * @param J_grid        View of tiled current grid
 * @param local_bc      Local rank boundary conditions
 */
__global__
void current_bcx(
    const grid::tiled_view<float3> J_grid,
    const current::bc_type local_bc ) {

    int bnd_x = blockIdx.x;
    int tile_idx_y = blockIdx.y;
    const int ystride = J_grid.tile_ystride();

    if ( bnd_x == 0 ) {
        // Lower boundary
        float3 * __restrict__ J = & J_grid.tile_buffer
            (0,tile_idx_y)      // lower x boundary tile
            [ J_grid.gc.x.lower ];     // point to first x cell (ix = 0)
        
        switch( local_bc.x.lower ) {
        case( current::bc::reflecting ):
            for( unsigned idx = gpu::block::thread_rank();
                 idx < J_grid.tile_ext_dims.y; 
                 idx += gpu::block::num_threads() ) {
                // iy includes the y-stride
                const int iy = idx * ystride;

                const float jx0 = -J[ -1 + iy ].x + J[ 0 + iy ].x; 
                const float jy1 =  J[ -1 + iy ].y + J[ 1 + iy ].y;
                const float jz1 =  J[ -1 + iy ].z + J[ 1 + iy ].z;

                // Normal component (odd): guard is the negative mirror
                J[  0 + iy ].x =  jx0;
                J[ -1 + iy ].x = -jx0;

                // Tangential components (even)
                J[ -1 + iy ].y = J[ 1 + iy ].y = jy1;
                J[ -1 + iy ].z = J[ 1 + iy ].z = jz1;
            }
            break;
        default:
            break;
        }
    } else {
        // Upper boundary
        float3 * __restrict__ J = & J_grid.tile_buffer
            (J_grid.local_ntiles.x-1,tile_idx_y)    // upper x boundary tile
            [ J_grid.gc.x.lower + J_grid.tile_dims.x ];    // point to first upper gc (ix = tile_dims.x)
        
        switch( local_bc.x.upper ) {
        case( current::bc::reflecting ):
            for( unsigned idx = gpu::block::thread_rank();
                 idx < J_grid.tile_ext_dims.y; 
                 idx += gpu::block::num_threads() ) {
                const int iy = idx * ystride;

                const float jx0 =  J[ -1 + iy ].x - J[ 0 + iy ].x; 
                const float jy1 =  J[ -1 + iy ].y + J[ 1 + iy ].y;
                const float jz1 =  J[ -1 + iy ].z + J[ 1 + iy ].z;

                // Normal component (odd): interior cell is -1, guard is 0
                J[ -1 + iy ].x =  jx0;
                J[  0 + iy ].x = -jx0;

                // Tangential components (even)
                J[ -1 + iy ].y = J[ 1 + iy ].y = jy1;
                J[ -1 + iy ].z = J[ 1 + iy ].z = jz1;
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
 * @warning this assumes that the current is staggered
 * 
 * @note Launch with grid( local_ntiles.x, 2 ): blockIdx.x selects the tile
 *       column, blockIdx.y the boundary (0 - lower, 1 - upper)
 * 
 * @param J_grid        View of tiled current grid
 * @param local_bc      Local rank boundary conditions
 */
__global__
void current_bcy( 
    const grid::tiled_view<float3> J_grid,
    const current::bc_type local_bc ) {

    int tile_idx_x = blockIdx.x;
    int bnd_y = blockIdx.y;
    const int ystride = J_grid.tile_ystride();
    
    if ( bnd_y == 0 ) {
        // Lower boundary
        float3 * __restrict__ J = & J_grid.tile_buffer
            (tile_idx_x,0)              // lower y boundary tiles
            [ J_grid.gc.y.lower * ystride ];   // point to first y cell (iy = 0)

        switch( local_bc.y.lower ) {
        case( current::bc::reflecting ):
            for( unsigned idx = gpu::block::thread_rank();
                 idx < J_grid.tile_ext_dims.x; 
                 idx += gpu::block::num_threads() ) {
                const int ix = idx;

                const float jx1 =  J[ ix - ystride ].x + J[ ix + ystride ].x; 
                const float jy0 = -J[ ix - ystride ].y + J[ ix           ].y;
                const float jz1 =  J[ ix - ystride ].z + J[ ix + ystride ].z;

                // Normal component (odd): guard is the negative mirror
                J[ ix           ].y =  jy0;
                J[ ix - ystride ].y = -jy0;

                // Tangential components (even)
                J[ ix - ystride ].x = J[ ix + ystride ].x = jx1;
                J[ ix - ystride ].z = J[ ix + ystride ].z = jz1;
            }
            break;
        default:
            break;
        }
    } else {
        // Upper boundary
        float3 * __restrict__ J = & J_grid.tile_buffer
            (tile_idx_x,J_grid.local_ntiles.y-1)   // upper y boundary tiles
            [ (J_grid.gc.y.lower + J_grid.tile_dims.y ) * ystride ];  // point to first upper gc (iy = tile_dims.y)
        
        switch( local_bc.y.upper ) {
        case( current::bc::reflecting ):
            for( unsigned idx = gpu::block::thread_rank();
                 idx < J_grid.tile_ext_dims.x; 
                 idx += gpu::block::num_threads() ) {
                const int ix = idx;

                const float jx1 =  J[ ix - ystride ].x + J[ ix + ystride ].x; 
                const float jy0 =  J[ ix - ystride ].y - J[ ix           ].y;
                const float jz1 =  J[ ix - ystride ].z + J[ ix + ystride ].z;

                // Normal component (odd): interior cell is -1, guard is 0
                J[ ix - ystride ].y =  jy0;
                J[ ix           ].y = -jy0;

                // Tangential components (even)
                J[ ix - ystride ].x = J[ ix + ystride ].x = jx1;
                J[ ix - ystride ].z = J[ ix + ystride ].z = jz1;
            }
            break;
        default:
            break;
        }
    }
}

} // namespace kernel

/**
 * @brief Processes "physical" boundary conditions
 * 
 * @note Physical boundaries are only applied on ranks that sit on the
 *       corresponding edge of the global domain; internal partition
 *       boundaries were already handled by add_from_gc()
 */
void current::process_bc() {

    dim3 block( 256 );
    const uint2 ntiles          = J -> get_local_ntiles();

    // x boundaries
    if ( local_bc.x.lower > current::bc::periodic || local_bc.x.upper > current::bc::periodic ) {
        dim3 grid( 2, ntiles.y );
        kernel::current_bcx <<< grid, block >>> ( 
            J -> view(), local_bc
        );
    }

    // y boundaries
    if ( local_bc.y.lower > current::bc::periodic || local_bc.y.upper > current::bc::periodic ) {
        dim3 grid( ntiles.x, 2 );
        kernel::current_bcy <<< grid, block >>> ( 
            J -> view(), local_bc
        );
    }
}

/**
 * @brief Advance electric current to next iteration
 * 
 * Adds up current deposited on guard cells, applies physical boundary
 * conditions, transforms to k-space and filters the result.
 * 
 */
void current::advance() {

    // Add up current deposited on guard cells
    J -> add_from_gc( );

    // Do additional bc calculations if needed
    process_bc();

    // Calculate fJ
    fft_forward -> transform( *fJ, *J );

    // Apply filtering (k-space only, J is left unfiltered)
    filter -> apply( *fJ );

    // Advance iteration count
    iter++;
}

/**
 * @brief Save electric current data to diagnostic file
 * 
 * @param quant     Which quantity to save (j or fj)
 * @param jc        Current component to save
 */
void current::save( const quantity quant, const fcomp::cart jc ) {

    std::string vfname;     // Dataset name
    std::string vflabel;    // Dataset label (for plots)

    grid::tiled_vec3<float> * f = nullptr;
    grid::flat3<current::complex_t> * cf = nullptr;

    switch (quant) {
        case quantity::j :
            f = J;
            vfname = "J";
            vflabel = "J_";
            break;
        case quantity::fj :
            cf = fJ;
            vfname = "fJ";
            vflabel = "\\mathcal{F}\\,J_";
            break;
        default:
            mpi::fatal( "Invalid field type selected" );
    }

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

        f -> save( jc, info, iteration, "CURRENT" );

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
            .min = - dk.y * ( nky / 2 ),
            .max =   dk.y * ( (nky - 1) / 2 ),
            .label = (char *) "k_y",
            .units = (char *) "\\omega_n/c"
        };

        info.axis = axis;

        grid::fft::kspace_save( *cf, jc, info, iteration, "CURRENT" );
    }
}
