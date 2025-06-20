#include "charge.h"

#include <iostream>

namespace kernel {

__global__
void charge_bcx(
    float * const __restrict__ rho_buffer,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx, bnd<unsigned int> gc, 
    const charge::bc_type bc
) {
    const uint2  tile_idx = { blockIdx.x * ( ntiles.x - 1 ), blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    const int ystride = ext_nx.x;
    // Start at x cell 0
    const auto x_offset = gc.x.lower;

    float * const __restrict__ rho = & rho_buffer[ tile_off + x_offset ];

    if ( tile_idx.x == 0 ) {
        // Lower boundary
        switch( bc.x.lower ) {
        case( charge::bc::reflecting ):
            for( unsigned idx = block_thread_rank(); idx < ext_nx.y; idx += block_num_threads() ) {
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
            for( unsigned idx = block_thread_rank(); idx < ext_nx.y; idx += block_num_threads() ) {
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

__global__
void charge_bcy( 
    float * const __restrict__ rho_buffer,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx, bnd<unsigned int> gc, 
    const charge::bc_type bc ) {

    const auto tile_idx = uint2{ 
        blockIdx.x ,
        blockIdx.y * (ntiles.y-1)
    };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    const int ystride = ext_nx.x;
    // Start at y cell 0
    const auto y_offset = gc.y.lower * ext_nx.x;

    float * const __restrict__ rho = & rho_buffer[ tile_off + y_offset ];
    
    if ( tile_idx.y == 0 ) {
        // Lower boundary
        switch( bc.y.lower ) {
        case( charge::bc::reflecting ):
            for( unsigned idx = block_thread_rank(); idx < ext_nx.x; idx += block_num_threads() ) {
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
            for( unsigned idx = block_thread_rank(); idx < ext_nx.x; idx += block_num_threads() ) {
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

}

/**
 * @brief Processes "physical" boundary conditions
 * 
 */
void Charge::process_bc() {

    dim3 block( 64 );

    // x boundaries
    if ( bc.x.lower > charge::bc::periodic || bc.x.upper > charge::bc::periodic ) {
         dim3 grid( 2, rho -> ntiles.y );

        kernel::charge_bcx <<< grid, block >>> ( 
            rho -> d_buffer, 
            rho -> ntiles, rho -> nx, rho -> ext_nx, rho -> gc, 
            bc
        );
    }

    // y boundaries
    if ( bc.y.lower > charge::bc::periodic || bc.y.upper > charge::bc::periodic ) {

        dim3 grid( rho -> ntiles.x, 2 );

        kernel::charge_bcy <<< grid, block >>> ( 
            rho ->  d_buffer, 
            rho ->  ntiles, rho ->  nx, rho ->  ext_nx, rho ->  gc,
            bc
        );
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
    
    // rho ->  copy_to_gc( ); // not needed

    // Do additional bc calculations if needed
    // process_bc();

	// Add neutralizing background
	// This is preferable to initializing rho to this value before charge deposition
	// because it leads to less roundoff errors
    if ( neutral ) rho -> add( *neutral );
    
    // Calculate frho
    fft_forward -> transform( *rho, reinterpret_cast< fft::complex64 * > ( frho->d_buffer ) );

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

    info.count[0] = rho ->  global_nx.x;
    info.count[1] = rho ->  global_nx.y;

    zdf::iteration iteration = {
    	.n = iter,
    	.t = iter * dt,
    	.time_units = (char *) "1/\\omega_n"
    };

    rho ->  save( info, iteration, "CHARGE" );
}