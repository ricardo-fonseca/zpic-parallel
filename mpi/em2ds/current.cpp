#include "current.hpp"

/**
 * @brief Physical boundary conditions for the x direction 
 * 
 * @param bnd_x         Boundary to process, 0 - lower, 1 - upper
 * @param tile_idx_y    Tile index along y direction
 * @param current       View of tiled current grid
 * @param local_bc            Boundary condition
 */
void current_bcx(
    int bnd_x, int tile_idx_y,
    const grid::tiled_view<float3> current,
    const current::bc_type local_bc ) {

    const int ystride = current.tile_ystride();

    if ( bnd_x == 0 ) {
        // Lower boundary
        float3 * __restrict__ J = & current.tile_buffer
            (0,tile_idx_y)      // lower x boundary tile
            [ current.gc.x.lower ];     // point to first x cell (ix = 0)
        
        switch( local_bc.x.lower ) {
        case( current::bc::reflecting ):
            for( unsigned idx = 0; idx < current.tile_ext_dims.y; idx ++ ) {
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
        float3 * __restrict__ J = & current.tile_buffer
            (current.local_ntiles.x-1,tile_idx_y)    // upper x boundary tile
            [ current.gc.x.lower + current.tile_dims.x ];    // point to first upper gc (ix = tile_dims.x)
        
        switch( local_bc.x.upper ) {
        case( current::bc::reflecting ):
            for( unsigned idx = 0; idx < current.tile_ext_dims.y; idx ++ ) {
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
 * @param bnd_y         Boundary to process, 0 - lower, 1 - upper
 * @param tile_idx_x    Tile index along x direction
 * @param current       View of tiled current grid
 * @param local_bc      Boundary condition
 */
void current_bcy( 
    int bnd_y, int tile_idx_x,
    const grid::tiled_view<float3> current,
    const current::bc_type local_bc ) {

    const int ystride = current.tile_ystride();
    
    if ( bnd_y == 0 ) {
        // Lower boundary
        float3 * __restrict__ J = & current.tile_buffer
            (tile_idx_x,0)              // lower y boundary tiles
            [ current.gc.y.lower * ystride ];   // point to first y cell (iy = 0)

        switch( local_bc.y.lower ) {
        case( current::bc::reflecting ):
            for( unsigned idx = 0; idx < current.tile_ext_dims.x; idx ++ ) {
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
        float3 * __restrict__ J = & current.tile_buffer
            (tile_idx_x,current.local_ntiles.y-1)   // upper y boundary tiles
            [ (current.gc.y.lower + current.tile_dims.y ) * ystride ];  // point to first upper gc (iy = tile_dims.y)
        
        switch( local_bc.y.upper ) {
        case( current::bc::reflecting ):
            for( unsigned idx = 0; idx < current.tile_ext_dims.x; idx ++ ) {
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

/**
 * @brief Processes "physical" boundary conditions
 * 
 */
void current::process_bc() {

    const uint2 ntiles          = J -> get_local_ntiles();

    // x boundaries
    if ( local_bc.x.lower > current::bc::periodic || local_bc.x.upper > current::bc::periodic ) {
        #pragma omp parallel for collapse(2)
        for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
            for( unsigned bnd_x : {0,1} ) {
                current_bcx( bnd_x, ty, J -> view(), local_bc );
            }
        }
    }

    // y boundaries
    if ( local_bc.y.lower > current::bc::periodic || local_bc.y.upper > current::bc::periodic ) {
       #pragma omp parallel for collapse(2)
        for( unsigned bnd_y : { 0,1 } ) {
            for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {
                current_bcy( bnd_y, tx, J -> view(), local_bc );
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

    // Do additional bc calculations if needed
    process_bc();

    // Calculate fJ
    fft_forward -> transform( *fJ, *J );

    // Apply filtering
    filter -> apply( *fJ );

    // Advance iteration count
    iter++;
}

/**
 * @brief Save electric current data to diagnostic file
 * 
 * @param jc        current component to save (0, 1 or 2)
 */
void current::save( const quantity quant, const fcomp::cart jc ) {

    std::string vfname;      // Dataset name
    std::string vflabel;    // Dataset label (for plots)

    grid::tiled_vec3<float> * f = nullptr;
    grid::flat3<complex_t> * cf = nullptr;

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

        f -> save( jc, info, iteration, "CURRENT" );

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

        cf -> save( jc, info, iteration, "CURRENT" );
    }
}