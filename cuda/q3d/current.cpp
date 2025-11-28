#include "current.h"
#include <iostream>

namespace kernel {

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
__global__
void current_bcz(
    T * const __restrict__ J_buffer,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx, bnd<unsigned int> gc, 
    const current::bc_type bc )
{
    const uint2  tile_idx = { blockIdx.x * ( ntiles.x - 1 ), blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    const int jstride = ext_nx.x;
    // Start at z cell 0
    const auto z_offset = gc.x.lower;

    auto * const __restrict__ J = & J_buffer[ tile_off + z_offset ];

    if ( tile_idx.x == 0 ) {
        // Lower boundary
        switch( bc.x.lower ) {
        case( current::bc::reflecting ):
            for( unsigned idx = block_thread_rank(); idx < ext_nx.y; idx += block_num_threads() ) {
                // j includes the y-stride
                const int j = idx * jstride;

                auto jz0 = -J[ -1 + j ].z + J[ 0 + j ].z; 
                auto jr1 =  J[ -1 + j ].r + J[ 1 + j ].r;
                auto jth1 =  J[ -1 + j ].th + J[ 1 + j ].th;

                J[ -1 + j ].z = J[ 0 + j ].z = jz0;
                J[ -1 + j ].r = J[ 1 + j ].r = jr1;
                J[ -1 + j ].th = J[ 1 + j ].th = jth1;
            }
            break;
        default:
            break;
        }
    } else {
        // Upper boundary
        switch( bc.x.upper ) {
        case( current::bc::reflecting ):
            for( unsigned idx = block_thread_rank(); idx < ext_nx.y; idx += block_num_threads() ) {
                const int j = idx * jstride;

                auto jz0 =  J[ nx.x-1 + j ].z - J[ nx.x + 0 + j ].z; 
                auto jr1 =  J[ nx.x-1 + j ].r + J[ nx.x + 1 + j ].r;
                auto jth1 =  J[ nx.x-1 + j ].th + J[ nx.x + 1 + j ].th;

                J[ nx.x-1 + j ].z = J[ nx.x + 0 + j ].z = jz0;
                J[ nx.x-1 + j ].r = J[ nx.x + 1 + j ].r = jr1;
                J[ nx.x-1 + j ].th = J[ nx.x + 1 + j ].th = jth1;
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
__global__
void current_bcr( 
    T * const __restrict__ J_buffer,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx, bnd<unsigned int> gc, 
    const current::bc_type bc )
{
    const auto tile_idx = uint2{ 
        blockIdx.x ,
        blockIdx.y * (ntiles.y-1)
    };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    const int jstride = ext_nx.x;
    // Start at r cell 0
    const auto r_offset = gc.y.lower * ext_nx.x;

    auto * const __restrict__ J = & J_buffer[ tile_off + r_offset ];
 
    // Upper boundary
    switch( bc.y.upper ) {
    case( current::bc::reflecting ):
        for( unsigned idx = block_thread_rank(); idx < ext_nx.x; idx += block_num_threads() ) {
            const int i = idx;

            auto jz1 =  J[ i + (nx.y-1)*jstride ].z + J[ i + (nx.y + 1)*jstride ].z; 
            auto jr0 =  J[ i + (nx.y-1)*jstride ].r - J[ i + (nx.y + 0)*jstride ].r;
            auto jth1 =  J[ i + (nx.y-1)*jstride ].th + J[ i + (nx.y + 1)*jstride ].th;

            J[ i + (nx.y-1)*jstride ].z = J[ i + (nx.y + 1)*jstride ].z = jz1;
            J[ i + (nx.y-1)*jstride ].r = J[ i + (nx.y + 0)*jstride ].r = jr0;
            J[ i + (nx.y-1)*jstride ].th = J[ i + (nx.y + 1)*jstride ].th = jth1;
        }
        break;
    default:
        break;
    }
}

__global__
/**
 * @brief Normalize current grid (m = 0)
 * 
 * @param d_current     Pointer to current grid
 * @param offset        Offset to position (0,0) on the grid
 * @param nx            Tile grid size
 * @param ext_nx        External tile grid size
 * @param dr            Radial cell size (in simulation units)
 */
void current_norm_0(
    cyl3<float> * const __restrict__ d_current, int offset, 
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx,
    float2 const dx, double dt
) {
    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    auto tid  = tile_idx.y * ntiles.x + tile_idx.x;
    const int tile_off = tid * roundup4( ext_nx.x * ext_nx.y );
    const int jstride = ext_nx.x;

    auto * __restrict__ current = &  d_current[ tile_off + offset ];

    ///@brief radial cell size
    auto dr = dx.y;

    float const dz_dt = dx.x / dt; 
    float const dr_dt = dx.y / dt; 

    int ir0 = tile_idx.y * nx.y;

    int range_z = (nx.x + 2) + 1; // [-1 .. nx.x-2[
    int range_r = (nx.y + 2) + 1; // [-1 .. nx.y-2[

    for( int idx = block_thread_rank(); idx < range_z * range_r; idx += block_num_threads() ) {
        int i = idx % range_z - 1;
        int j = idx / range_z - 1;

        /// @brief r at center of cell
        float rc   = std::abs( ir0 + j        ) * dr;
        /// @brief r at lower edge of cell
        float rl   = std::abs( ir0 + j - 0.5f ) * dr;
        
        float norm_r  = ( ir0 + j == 0 )? 0 : 1.0f / rc;
        float norm_zth = 1.0f / rl;

        current[ j * jstride +i ].z *= dz_dt * norm_zth;
        current[ j * jstride +i ].r *= dr_dt * norm_r;
        current[ j * jstride +i ].th *= dr_dt * norm_zth;
    }

    // Axial boundary
    // Fold values for r < 0 back into simulation domain
    if ( ir0 == 0 ) {

        for( int i = block_thread_rank() - 1; i < nx.x + 2; i += block_num_threads() ) {

            current[ i + 1 * jstride ].z += current[ i +   0  * jstride ].z;
            current[ i + 2 * jstride ].z += current[ i + (-1) * jstride ].z;

            current[ i + 1 * jstride ].r -= current[ i + (-1) * jstride ].r;

            current[ i + 1 * jstride ].th -= current[ i +   0  * jstride ].th;
            current[ i + 2 * jstride ].th -= current[ i + (-1) * jstride ].th;

            // The following values are used for diagnostic output only
            current[ i + 0 * jstride ].z  = current[ i + 1 * jstride ].z;
            current[ i + 0 * jstride ].r  = current[ i + 1 * jstride ].r;
            current[ i + 0 * jstride ].th  = current[ i + 1 * jstride ].th;
        }
    }
}

__global__
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
    cyl3< ops::complex<float> > * const __restrict__ d_current, int offset, 
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx,
    float2 const dx, double dt
) {
    const uint2 tile_idx = { blockIdx.x, blockIdx.y };

    auto tid = tile_idx.y * ntiles.x + tile_idx.x;
    const int tile_off = tid * roundup4( ext_nx.x * ext_nx.y );
    const int jstride = ext_nx.x;

    auto * __restrict__ current = &  d_current[ tile_off + offset ];

    ///@brief radial cell size
    auto dr = dx.y;
    float const dz_dt = dx.x / dt; 
    float const dr_dt = dx.y / dt; 

    ///@brief Normalization for jth
    const ops::complex<float> norm_th{0,2/(m*static_cast<float>(dt))};

    int ir0 = tile_idx.y * nx.y;

    int range_z = (nx.x + 2) + 1; // [-1 .. nx.x-2[
    int range_r = (nx.y + 2) + 1; // [-1 .. nx.y-2[

    for( int idx = block_thread_rank(); idx < range_z * range_r; idx += block_num_threads() ) {
        int i = idx % range_z - 1;
        int j = idx / range_z - 1;

        /// @brief r at center of cell
        float rc   = std::abs( ir0 + j        ) * dr;
        /// @brief r at lower edge of cell
        float rl   = std::abs( ir0 + j - 0.5f ) * dr;
        
        float norm_r  = ( ir0 + j == 0 )? 0 : (2 * dr_dt) / rc;
        float norm_z  = ( 2 * dz_dt ) / rl;

        current[ i + j * jstride ].z *= norm_z;
        current[ i + j * jstride ].r *= norm_r;
        current[ i + j * jstride ].th *= norm_th;
    }

    // Axial boundary
    // Fold values for r < 0 back into simulation domain
    if ( ir0 == 0 ) {

        // alternative, signt = -(-1)^m
        // const auto signt = ( m & 1 ) ? 1.f : -1.f;

        for( int i = block_thread_rank() - 1; i < nx.x + 2; i += block_num_threads() ) {
            current[ i + 1 * jstride ].z += current[ i +   0  * jstride ].z;
            current[ i + 2 * jstride ].z += current[ i + (-1) * jstride ].z;

            current[ i + 1 * jstride ].r -= current[ i + (-1) * jstride ].r;

            current[ i + 1 * jstride ].th -= current[ i +   0  * jstride ].th;
            current[ i + 2 * jstride ].th -= current[ i + (-1) * jstride ].th;

            // The following values are used for diagnostic output only
            current[ i + 0 * jstride ].z  = current[ i + 1 * jstride ].z;
            current[ i + 0 * jstride ].r  = current[ i + 1 * jstride ].r;
            current[ i + 0 * jstride ].th  = current[ i + 1 * jstride ].th;
        }
    }
}

}

/**
 * @brief Processes "physical" boundary conditions
 * 
 */
void Current::process_bc() {

    auto & J0 = J -> mode0();
    dim3 block( 64 );

    // z boundaries
    if ( bc.x.lower > current::bc::periodic || bc.x.upper > current::bc::periodic ) {
        dim3 grid( 2, J0.ntiles.y );

        kernel::current_bcz <<< grid, block >>> ( 
            J0.d_buffer, 
            J0.ntiles, J0.nx, J0.ext_nx, J0.gc, 
            bc
        );

        for( int m = 1; m < nmodes; m++ ) {
            auto & Jm = J -> mode(m);
            kernel::current_bcz <<< grid, block >>> ( 
                Jm.d_buffer, 
                Jm.ntiles, Jm.nx, Jm.ext_nx, Jm.gc, 
                bc
            );
        }
    }

    // r boundaries
    // Only outer radial boundaries need be considered, axial boundaries
    // are treated elsewhere
    if ( bc.y.upper > current::bc::periodic ) {

        dim3 grid( J0.ntiles.x, 1 );

        kernel::current_bcr <<< grid, block >>> ( 
            J0.d_buffer, 
            J0.ntiles, J0.nx, J0.ext_nx, J0.gc,
            bc
        );

        for( int m = 1; m < nmodes; m++ ) {
            auto & Jm = J -> mode(m);
            kernel::current_bcr <<< grid, block >>> ( 
                Jm.d_buffer, 
                Jm.ntiles, Jm.nx, Jm.ext_nx, Jm.gc, 
                bc
            );
        }
    }
}

/**
 * @brief Normalize grid values for ring particles
 * 
 */
void Current::normalize() {

    auto & J0 = J -> mode0();

    dim3 grid( J0.ntiles.x, J0.ntiles.y );
    dim3 block( 128 );

    // Normalize mode 0
    kernel::current_norm_0 <<< grid, block >>> (
        J0.d_buffer, J0.offset, 
        J0.ntiles, J0.nx, J0.ext_nx, 
        dx, dt
    );

    // Normalize higher order modes
    for( int m = 1; m < nmodes; m++ ) {
        auto & Jm = J -> mode(m);
        kernel::current_norm_m <<< grid, block >>> (
            m, Jm.d_buffer, Jm.offset, 
            Jm.ntiles, Jm.nx, Jm.ext_nx, 
            dx, dt
        );
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
    apply_filter();

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
 * @param fc        Which current component to save (r, t or z)
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
        case( fcomp::th ) :
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