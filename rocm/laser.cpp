#include "laser.h"

#include <iostream>
#include <cassert>

#include "filter.h"

namespace kernel {

/**
 * @brief Gets longitudinal laser envelope a given position
 * 
 * @param z         position
 * @param start     Start position
 * @param rise      Rise length
 * @param flat      Flat length
 * @param fall      Fall length
 * @return float    laser envelope
 */
__device__ __inline__
float lon_env( Laser::Pulse & laser, float z ) {

    if ( z > laser.start ) {
        // Ahead of laser
        return 0.0;
    } else if ( z > laser.start - laser.rise ) {
        // Laser rise
        float csi = z - laser.start;
        float e = sin( M_PI_2 * csi / laser.rise );
        return e*e;
    } else if ( z > laser.start - (laser.rise + laser.flat) ) {
        // Flat-top
        return 1.0;
    } else if ( z > laser.start - (laser.rise + laser.flat + laser.fall) ) {
        // Laser fall
        float csi = z - (laser.start - laser.rise - laser.flat - laser.fall);
        float e = sin( M_PI_2 * csi / laser.fall );
        return e*e;
    }

    // Before laser
    return 0.0;
}

__global__
void div_corr_x_A (
    float3 * const __restrict__ E_buffer, 
    float3 * const __restrict__ B_buffer,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx, unsigned int const offset,
    double const dx_dy, double2 * const __restrict__ tmp )
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    const int ystride = ext_nx.x;
    
    auto * shm = block::shared_mem<float3>();
    float3 __restrict__ * E_local = &shm[0];
    float3 __restrict__ * B_local = &shm[tile_vol];

    for( unsigned idx = block_thread_rank(); idx < tile_vol; idx += block_num_threads() ) {
        E_local[idx] = E_buffer[ tile_off + idx ];
        B_local[idx] = B_buffer[ tile_off + idx ];
    }
    block_sync();

    float3 * const __restrict__ E = & E_local[ offset ];
    float3 * const __restrict__ B = & B_local[ offset ]; 

    auto tmp_off = tile_idx.y * nx.y * ntiles.x;

    for( int iy = block_thread_rank(); iy < nx.y; iy += block_num_threads() ) {
        // Find divergence at left edge
        double divEx = 0;
        double divBx = 0;
        for( int ix = nx.x - 1; ix >= 0; ix-- ) {
            divEx += dx_dy * (E[ix+1 + iy*ystride].y - E[ix+1 + (iy-1)*ystride ].y);
            divBx += dx_dy * (B[ix + (iy+1)*ystride].y - B[ix + iy*ystride ].y);
        }

        // Write result to tmp. array
        tmp[ tmp_off + iy * ntiles.x + tile_idx.x ] = double2{ divEx, divBx };
    }
}

__global__
void div_corr_x_B( double2 * const __restrict__ tmp, const uint2 ntiles )
{
    auto * buffer = block::shared_mem<double2>();

    int giy = blockIdx.x;

    for( int i = block_thread_rank(); i < ntiles.x; i += block_num_threads() ) {
        buffer[i] = tmp[ giy * ntiles.x + i ];
    }
    block_sync();

    // Perform scan operation (serial inside block)
    if ( block_thread_rank() == 0 ) {
        double2 a{ 0 };
        for( int i = ntiles.x-1; i >= 0; i--) {
            double2 b = buffer[i];
            buffer[i] = a;
            a.x += b.x;
            a.y += b.y;
        }
    }
    block_sync();

    // Copy data to global memory
    for( int i = block_thread_rank(); i < ntiles.x; i += block_num_threads() ) {
        tmp[ giy * ntiles.x + i ] = buffer[i];
    }
}

__global__
void div_corr_x_C( 
    float3 * const __restrict__ E_buffer,
    float3 * const __restrict__ B_buffer,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx, unsigned int const offset,
    double const dx_dy, double2 const * const __restrict__ tmp )
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    const int ystride = ext_nx.x;
    
    auto * shm = block::shared_mem<float3>();
    float3 __restrict__ * E_local = &shm[0];
    float3 __restrict__ * B_local = &shm[tile_vol];

    for( unsigned idx = block_thread_rank(); idx < tile_vol; idx += block_num_threads() ) {
        E_local[idx] = E_buffer[ tile_off + idx ];
        B_local[idx] = B_buffer[ tile_off + idx ];
    }
    block_sync();

    float3 * const __restrict__ E = & E_local[ offset ];
    float3 * const __restrict__ B = & B_local[ offset ];

    auto tmp_off = tile_idx.y * nx.y * ntiles.x;

    for( int iy = block_thread_rank(); iy < nx.y; iy += block_num_threads() ) {
        // Get divergence at right edge
        double2 div = tmp[ tmp_off + iy * ntiles.x + tile_idx.x ];
        double divEx = div.x;
        double divBx = div.y;

        for( int ix = nx.x - 1; ix >= 0; ix-- ) {
            divEx += dx_dy * (E[ix+1 + iy*ystride].y - E[ix+1 + (iy-1)*ystride ].y);
            E[ ix + iy * ystride].x = divEx;

            divBx += dx_dy * (B[ix + (iy+1)*ystride].y - B[ix + iy*ystride ].y);
            B[ ix + iy * ystride].x = divBx;
        }
    }
    block_sync();

    for( unsigned idx = block_thread_rank(); idx < tile_vol; idx += block_num_threads() ) {
        E_buffer[ tile_off + idx ] = E_local[idx];
        B_buffer[ tile_off + idx ] = B_local[idx];
    }   
}

}

/**
 * @brief Sets the longitudinal field components of E and B to ensure 0
 *        divergence
 * 
 * @note There are some opportunities for optimization, check the comments in
 *       the code
 * 
 * @warning The algorithm assumes 0 field at the right boundary of the box. If
 *          this is not true then the field values will be wrong.
 * 
 * @param E     E field
 * @param B     B field
 * @param dx    Cell size
 * @param q     Sycl queue
 */
void div_corr_x( vec3grid<float3>& E, vec3grid<float3>& B, const float2 dx ) {


    // Check that local memory can hold up to 2 times the tile buffer
    auto local_mem_size = block::shared_mem_size();
    if ( local_mem_size < 2 * E.tile_vol * sizeof( float3 ) ) {
        std::cerr << "(*error*) Tile size too large [" << E.nx.x << "," << E.nx.y << " (plus guard cells)\n";
        std::cerr << "(*error*) Insufficient local memory (" << local_mem_size << " B) for div_corr_x() function.\n";
        abort();
    }

    // Temporary buffer for divergence calculations
    size_t bsize = E.ntiles.x * (E.ntiles.y * E.nx.y);
    double2 * tmp = device::malloc<double2>( bsize );

    /**
     * Step A - Get per-tile E and B divergence at tile left edge starting
     *          from 0.0
     * 
     * This could (potentially) be improved by processing each line in a warp,
     * and replacing the divEx and divBx calculations by warp level reductions
     */

    dim3 grid( E.ntiles.x, E.ntiles.y );
    dim3 block( 32 );
    size_t shm_size = 2 * E.tile_vol * sizeof(float3);

    const double dx_dy = (double) dx.x / (double) dx.y;

    kernel::div_corr_x_A <<< grid, block, shm_size >>> ( 
        E.d_buffer, B.d_buffer, 
        E.ntiles, E.nx, E.ext_nx, E.offset,
        dx_dy, tmp
    );

    /**
     * Step B - Performs a left-going scan operation on the results from step A.
     * 
     * This could (potentially) be improved by processing each row of tiles in
     * a warp:
     * - Copy the data from tmp and store it in shared memory in reverse order
     * - Do a normal ex-scan operation using warp accelerated code
     * - Copy the data in reverse order from shared memory and store it in tmp
     */

    dim3 grid_B( E.gnx.y );
    dim3 block_B( E.ntiles.x > 32 ? 32 : E.ntiles.x );
    size_t shm_size_B = E.ntiles.x * sizeof(double2);

    kernel::div_corr_x_B <<< grid_B, block_B, shm_size_B >>> (
        tmp, E.ntiles
    );

    /**
     * Step C - Starting from the results from step B, get the longitudinal
     *          components at each cell.
     * 
     * This could potentially be improved by processing each line within a warp
     * - Copying the data from global memory and storing it in reverse x order
     * - Get the divergence correction for each cell and perform a normal warp
     *   level ex-scan adding the values from step B and store in shared memory
     * - Copy the data back to global memory again reversing the order
     */

    kernel::div_corr_x_C <<< grid, block, shm_size >>> (
        E.d_buffer, B.d_buffer, 
        E.ntiles, E.nx, E.ext_nx, E.offset,
        dx_dy, tmp
    );

    // Free temporary memory
    device::free( tmp );

    // Correct longitudinal values on guard cells
    E.copy_to_gc( );
    B.copy_to_gc( );
}

/**
 * @brief Validates laser parameters
 * 
 * @return      0 on success, -1 on error
 */
int Laser::Pulse::validate() {

    if ( a0 <= 0 ) {
        std::cerr << "(*error*) Invalid laser a0, must be > 0\n";
        return -1;
    }    

    if ( omega0 <= 0 ) {
        std::cerr << "(*error*) Invalid laser OMEGA0, must be > 0\n";
        return -1;
    }    

    if ( fwhm > 0 ) {
        // The fwhm parameter overrides the rise/flat/fall parameters
        rise = fwhm;
        fall = fwhm;
        flat = 0.;
    } else {
        if ( rise <= 0 ) {
            std::cerr << "(*error*) Invalid laser RISE, must be > 0\n";
            return (-1);
        }

        if ( flat < 0 ) {
            std::cerr << "(*error*) Invalid laser FLAT, must be >= 0\n";
            return (-1);
        }

        if ( fall <= 0 ) {
            std::cerr << "(*error*) Invalid laser FALL, must be > 0\n";
            return (-1);
        }
    }

    return 0;
}


namespace kernel {

__global__
void plane_wave(
    Laser::PlaneWave laser, 
    float3 * __restrict__ E_buffer, float3 * __restrict__ B_buffer,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx, unsigned int const offset,
    float2 const dx )
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    float3 * const __restrict__ tile_E = & E_buffer[ tile_off + offset ];
    float3 * const __restrict__ tile_B = & B_buffer[ tile_off + offset ];

    const int ix0 = tile_idx.x * nx.x;
    const float k = laser.omega0;
    const float amp = laser.omega0 * laser.a0;
    const int ystride = ext_nx.x;

    for( unsigned idx = block_thread_rank(); idx < nx.y * nx.x; idx += block_num_threads() ) {
        const auto ix = idx % nx.x;
        const auto iy = idx / nx.x; 

        const float z   = ( ix0 + ix ) * dx.x;
        const float z_2 = ( ix0 + ix + 0.5 ) * dx.x;

        float lenv   = amp * lon_env( laser, z );
        float lenv_2 = amp * lon_env( laser, z_2 );

        tile_E[ ix + iy * ystride ] = make_float3(
            0,
            +lenv * cos( k * z ) * laser.cos_pol,
            +lenv * cos( k * z ) * laser.sin_pol
        );

        tile_B[ ix + iy * ystride ] = make_float3(
            0,
            -lenv_2 * cos( k * z_2 ) * laser.sin_pol,
            +lenv_2 * cos( k * z_2 ) * laser.cos_pol
        );
    }
}


}

/**
 * @brief Launches a plane wave
 * 
 * The E and B tiled grids have the complete laser field.
 * 
 * @param E     Electric field
 * @param B     Magnetic field
 * @param box   Box size
 * @return      Returns 0 on success, -1 on error (invalid laser parameters)
 */
int Laser::PlaneWave::launch( vec3grid<float3>& E, vec3grid<float3>& B, float2 box ) {

    if ( validate() < 0 ) return -1;

    if (( cos_pol == 0 ) && ( sin_pol == 0 )) {
        cos_pol = cos( polarization );
        sin_pol = sin( polarization );
    }

    uint2 g_nx = E.gnx;

    float2 dx = make_float2(
        box.x / g_nx.x,
        box.y / g_nx.y
    );

    dim3 block( 64 );
    dim3 grid( E.ntiles.x, E.ntiles.y );

    kernel::plane_wave<<<grid, block>>> (
        *this, E.d_buffer, B.d_buffer,
        E.ntiles, E.nx, E.ext_nx, E.offset,
        dx
    );

    E.copy_to_gc( );
    B.copy_to_gc( );

    if ( filter > 0 ) {

        Filter::Compensated fcomp( coord::x, filter);
        fcomp.apply( E );
        fcomp.apply( B );
    }

    return 0;
}

/**
 * @brief Validate Gaussian laser parameters
 * 
 * @return      0 on success, -1 on error
 */
int Laser::Gaussian::validate() {
    
    if ( Laser::Pulse::validate() < 0 ) {
        return -1;
    }

    if ( W0 <= 0 ) {
        std::cerr << "(*error*) Invalid laser W0, must be > 0\n";
        return (-1);
    }

    return 0;
}

namespace kernel {

/**
 * @brief Returns local phase for a gaussian beamn
 * 
 * @param omega0    Beam frequency
 * @param W0        Beam waist
 * @param z         Position along focal line (focal plane at z = 0)
 * @param r         Position transverse to focal line (focal line at r = 0)
 * @return          Local field value
 */
__device__ float gauss_phase( const float omega0, const float W0, const float z, const float r ) {
    const float z0   = omega0 * ( W0 * W0 ) / 2;
    const float rho2 = r*r;
    const float curv = rho2 * z / (z0*z0 + z*z);
    const float rWl2 = (z0*z0)/(z0*z0 + z*z);
    const float gouy_shift = atan2( z, z0 );

    return sqrt( sqrt(rWl2) ) * 
           exp( - rho2 * rWl2/( W0 * W0 ) ) * 
           cos( omega0*( z + curv ) - gouy_shift );
}

__global__
void gaussian( 
    Laser::Gaussian beam, 
    float3 * __restrict__ E_buffer, float3 * __restrict__ B_buffer,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx, unsigned int const offset,
    float2 const dx
) {
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    float3 * const __restrict__ tile_E = & E_buffer[ tile_off + offset ];
    float3 * const __restrict__ tile_B = & B_buffer[ tile_off + offset ];

    const int ix0 = tile_idx.x * nx.x;
    const int iy0 = tile_idx.y * nx.y;

    const int ystride = ext_nx.x;

    const float amp = beam.omega0 * beam.a0;

    for( unsigned idx = block_thread_rank(); idx < nx.y * nx.x; idx += block_num_threads() ) {
        const auto ix = idx % nx.x;
        const auto iy = idx / nx.x; 

        const float z   = ( ix0 + ix ) * dx.x;
        const float z_2 = ( ix0 + ix + 0.5 ) * dx.x;

        const float r   = (iy0 + iy ) * dx.y - beam.axis;
        const float r_2 = (iy0 + iy + 0.5 ) * dx.y - beam.axis;

        const float lenv   = amp * lon_env( beam, z );
        const float lenv_2 = amp * lon_env( beam, z_2 );

        tile_E[ ix + iy * ystride ] = make_float3(
            0,
            +lenv * gauss_phase( beam.omega0, beam.W0, z - beam.focus, r_2 ) * beam.cos_pol,
            +lenv * gauss_phase( beam.omega0, beam.W0, z - beam.focus, r   ) * beam.sin_pol
        );
        tile_B[ ix + iy * ystride ] = make_float3(
            0,
            -lenv_2 * gauss_phase( beam.omega0, beam.W0, z_2 - beam.focus, r   ) * beam.sin_pol,
            +lenv_2 * gauss_phase( beam.omega0, beam.W0, z_2 - beam.focus, r_2 ) * beam.cos_pol
        );
    }
}

}

/**
 * @brief Launches a Gaussian pulse
 * 
 * The E and B tiled grids have the complete laser field.
 * 
 * @param E     Electric field
 * @param B     Magnetic field
 * @param dx    Cell size
 * @return      Returns 0 on success, -1 on error (invalid laser parameters)
 */
int Laser::Gaussian::launch(vec3grid<float3>& E, vec3grid<float3>& B, float2 const box ) {

    // std::cout << "Launching gaussian pulse...\n";

    if ( validate() < 0 ) return -1;

    if (( cos_pol == 0 ) && ( sin_pol == 0 )) {
        cos_pol = cos( polarization );
        sin_pol = sin( polarization );
    }

    uint2 g_nx = E.gnx;

    float2 dx = make_float2(
        box.x / g_nx.x,
        box.y / g_nx.y
    );

    dim3 block( 64 );
    dim3 grid( E.ntiles.x, E.ntiles.y );

    kernel::gaussian<<<grid, block>>> (
        *this, E.d_buffer, B.d_buffer,
        E.ntiles, E.nx, E.ext_nx, E.offset,
        dx
    );

    E.copy_to_gc( );
    B.copy_to_gc( );

    if ( filter > 0 ) {

        Filter::Compensated fcomp( coord::x, filter);
        fcomp.apply( E );
        fcomp.apply( B );
    }

    // Get longitudinal field components by enforcing div = 0
    div_corr_x( E, B, dx );

    return 0;
}