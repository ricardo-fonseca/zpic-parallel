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

/**
 * @brief Divergence correction - Prefix scan local data to get longitudinal components
 * 
 * @note Must be called with grid( local_nx.y ) and block( WARP_SIZE )
 * 
 * @param E_buffer      E field tile buffer (including offset to cell [0,0])
 * @param B_buffer      B field tile buffer (including offset to cell [0,0])
 * @param ntiles        Number of tiles in local node (x,y)
 * @param nx            Individual tile size
 * @param ext_nx        Individual tile size including guard cells
 */
__global__
void div_corr_z_scan (
    cyl_cfloat3 * const __restrict__ E_buffer, 
    cyl_cfloat3 * const __restrict__ B_buffer,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx,
    float2 const dx
)
{
    int grid_j = blockIdx.x;

    if ( grid_j == 0 ) return;

    const int tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const int jstride = ext_nx.x;

    /// @brief longitudinal (z) cell size
    const auto dz = dx.x;

    /// @brief transverse (r) cell size
    const auto dr = dx.y;

    ///@brief y tile index
    int ty = blockIdx.x / nx.y;

    ///@brief j coordinate inside tile
    int j = grid_j - ty * nx.y;

    /// @brief imaginary unit
    constexpr ops::complex<float> I{0,1};

    ///@brief r/Δr at center of cell
    const float rc  = grid_j;
    ///@brief r/Δr at center of lower (j-1) cell
    const float rcm = grid_j - 1;
    ///@brief r/Δr at upper edge of cell
    const float rp = grid_j + 0.5;
    ///@brief r/Δr at lower edge of cell
    const float rm = grid_j - 0.5;

    ///@brief Δz/r at center of cell
    const float dz_rc = dz / (rc * dr);
    ///@brief Δz/r at lower edge of cell
    const float dz_rm = dz / (rm * dr);

    __shared__ ops::complex<double> divEz; divEz = 0;
    __shared__ ops::complex<double> divBz; divBz = 0;

    block_sync();

    // Process tiles right to left
    for( int tx = ntiles.x-1; tx >=0; tx-- ) {

        const auto tile_idx = make_uint2( tx, ty );
        const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
        const auto tile_off = tid * tile_vol;

        auto * const __restrict__ tile_E = & E_buffer[ tile_off ];
        auto * const __restrict__ tile_B = & B_buffer[ tile_off ];

        int start = ( nx.x / WARP_SIZE ) * WARP_SIZE; 
        for( int i = start + block_thread_rank(); i >= 0; i -= WARP_SIZE ) {

            ops::complex<double> dEz = 0;
            ops::complex<double> dBz = 0;

            // If inside tile read x divergence
            if ( i < nx.x ) {
                dEz = dz_rm * (
                        ( rc * tile_E[i+1 + j*jstride].r - rcm * tile_E[i+1 + (j-1)*jstride].r) + 
                        I * tile_E[i+1 + j*jstride].th
                    ) ;
                dBz = dz_rc * (
                        ( rp * tile_B[i   + (j+1)*jstride].r - rm * tile_B[i + j*jstride].r ) + 
                        I * tile_B[i + j*jstride].th
                    ) ;
            }

            // Do a right to left inclusive scan
            {
                const int laneId = threadIdx.x & ( WARP_SIZE - 1 );
                #pragma unroll
                for( int k = 1; k < WARP_SIZE; k <<= 1 ) {
                    double re1 = __shfl_down_sync( 0xffffffff, real(dEz), k );
                    double im1 = __shfl_down_sync( 0xffffffff, imag(dEz), k );
                    double re2 = __shfl_down_sync( 0xffffffff, real(dBz), k );
                    double im2 = __shfl_down_sync( 0xffffffff, imag(dBz), k );
                    if ( laneId < WARP_SIZE - k ) {
                        dEz += ops::complex<double>{ re1, im1 };
                        dBz += ops::complex<double>{ re2, im2 };
                    }
                }
            }

            // If inside tile store longitudinal components
            if ( i < nx.x ) {
                tile_E[i + j * jstride].z = divEz + dEz;
                tile_B[i + j * jstride].z = divBz + dBz;
            }

            // Accumulate results for next loop
            if ( block_thread_rank() == 0 ) {
                divEz += dEz;
                divBz += dBz;
            }
            block_sync();
        }
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
 * @param E     E field (mode 1)
 * @param B     B field (mode 1)
 * @param dx    cell size (z,r)
 */
void div_corr_z( cyl3grid<ops::complex<float>>& E, cyl3grid<ops::complex<float>>& B, const float2 dx ) {

    dim3 grid( E.dims.y );
    dim3 block( WARP_SIZE );

    kernel::div_corr_z_scan <<< grid, block >>> ( 
        & E.d_buffer[ E.offset ], & B.d_buffer[ B.offset ], 
        E.get_ntiles(), E.nx, E.ext_nx, dx
    );

    // Correct longitudinal values on tile guard cells
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
    cyl_cfloat3 * __restrict__ E_buffer, cyl_cfloat3 * __restrict__ B_buffer,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx, unsigned int const offset,
    float2 const dx )
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    cyl_cfloat3 * const __restrict__ tile_E = & E_buffer[ tile_off + offset ];
    cyl_cfloat3 * const __restrict__ tile_B = & B_buffer[ tile_off + offset ];

    const int i0 = tile_idx.x * nx.x;
    const float k = laser.omega0;
    const float amp = laser.omega0 * laser.a0;

    const int jstride = ext_nx.x;

    ///@brief cell size in longitudinal direction
    const auto dz = dx.x;

    ///@brief imaginary unit
    constexpr ops::complex<float> I{0,1};
    ///@brief exp( I pol )
    const ops::complex<float> pol_r{ laser.cos_pol, -laser.sin_pol };
    ///@brief exp( I (pol-π/2) )
    const ops::complex<float> pol_th{ laser.sin_pol, +laser.cos_pol };

    for( unsigned idx = block_thread_rank(); idx < nx.y * nx.x; idx += block_num_threads() ) {
        const auto i = idx % nx.x;
        const auto j = idx / nx.x; 

        const float z   = ( i0 + i ) * dz;
        const float z_2 = ( i0 + i + 0.5 ) * dz;

        float lenv   = amp * lon_env( laser, z );
        float lenv_2 = amp * lon_env( laser, z_2 );

        tile_E[ i + j * jstride ].z = 0;
        tile_E[ i + j * jstride ].r = +lenv * std::cos( k * z ) * pol_r;
        tile_E[ i + j * jstride ].th = -lenv * std::cos( k * z ) * pol_th;

        tile_B[ i + j * jstride ].z = 0;
        tile_B[ i + j * jstride ].r = lenv_2 * std::cos( k * z_2 ) * pol_th;
        tile_B[ i + j * jstride ].th = lenv_2 * std::cos( k * z_2 ) * pol_r;
    }

    // Correct axial cell values, see field solver
    if ( tile_idx.y == 0 ) {
        // This is an m = 1 field
        for( int i = block_thread_rank(); i < nx.x; i += block_num_threads() ) {
            tile_E[ i + 0 * jstride ].z = 0;
            tile_E[ i + 0 * jstride ].r = ( 4.f * tile_E[ i + 1*jstride ].r - tile_E[ i + 2*jstride ].r ) / 3.f;
            tile_E[ i + 0 * jstride ].th = tile_E[ i + 1*jstride ].th;

            tile_E[ i + 0 * jstride ].z = 0;
            tile_B[ i + 0 * jstride ].r = + tile_B[ i + 1*jstride ].r;  
            tile_B[ i + 0 * jstride ].th = 0.125f * I * ( tile_B[ i + 2*jstride ].r - 9.f * tile_B[ i + 2*jstride ].r );
        }

        // values for j < 0 are unused for linear interpolation
        for( int i = block_thread_rank(); i < nx.x; i += block_num_threads() ) {
            tile_E[ i + (-1) * jstride ].z = 0;
            tile_E[ i + (-1) * jstride ].r = 0;
            tile_E[ i + (-1) * jstride ].th = 0;

            tile_B[ i + (-1) * jstride ].z = 0;
            tile_B[ i + (-1) * jstride ].r = 0;
            tile_B[ i + (-1) * jstride ].th = 0;
        }
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
int Laser::PlaneWave::launch( cyl3grid<ops::complex<float>>& E, cyl3grid<ops::complex<float>>& B, float2 box ) {

    if ( validate() < 0 ) return -1;

    if (( cos_pol == 0 ) && ( sin_pol == 0 )) {
        cos_pol = cos( polarization );
        sin_pol = sin( polarization );
    }

    uint2 g_nx = E.dims;

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

        Filter::Compensated fcomp( coord::z, filter);
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
    cyl_cfloat3 * __restrict__ E_buffer, cyl_cfloat3 * __restrict__ B_buffer,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx, unsigned int const offset,
    float2 const dx
) {
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    cyl_cfloat3 * const __restrict__ tile_E = & E_buffer[ tile_off + offset ];
    cyl_cfloat3 * const __restrict__ tile_B = & B_buffer[ tile_off + offset ];

    const int i0 = tile_idx.x * nx.x;
    const int j0 = tile_idx.y * nx.y;

    const int jstride = ext_nx.x;

    const float amp = beam.omega0 * beam.a0;

    ///@brief cell size in longitudinal direction
    const auto dz = dx.x;
    ///@brief cell size in radial direction
    const auto dr = dx.y;

    ///@brief imaginary unit
    constexpr ops::complex<float> I{0,1};
    ///@brief exp( I pol )
    ops::complex<float> pol_r{ beam.cos_pol, -beam.sin_pol };
    ///@brief exp( I (pol-π/2) )
    ops::complex<float> pol_th{ beam.sin_pol, +beam.cos_pol };

    const auto omega0 = beam.omega0;
    const auto W0 = beam.W0;
    const auto focus = beam.focus;

    for( unsigned idx = block_thread_rank(); idx < nx.y * nx.x; idx += block_num_threads() ) {
        const auto i = idx % nx.x;
        const auto j = idx / nx.x; 

        const float z   = ( i0 + i ) * dz;
        const float z_2 = ( i0 + i + 0.5 ) * dz;

        const float r   = ( j0 + j ) * dr;
        const float r_2 = ( j0 + j + 0.5 ) * dr;

        const float lenv   = amp * lon_env( beam, z );
        const float lenv_2 = amp * lon_env( beam, z_2 );

        tile_E[ i + j * jstride ].z = 0;
        tile_E[ i + j * jstride ].r = +lenv * gauss_phase( omega0, W0, z - focus, r_2 ) * pol_r;
        tile_E[ i + j * jstride ].th = +lenv * gauss_phase( omega0, W0, z - focus, r   ) * pol_th;

        tile_B[ i + j * jstride ].z = 0;
        tile_B[ i + j * jstride ].r = -lenv_2 * gauss_phase( omega0, W0, z_2 - focus, r   ) * pol_th;
        tile_B[ i + j * jstride ].th = +lenv_2 * gauss_phase( omega0, W0, z_2 - focus, r_2 ) * pol_r;
    }

    // Correct axial cell values, see field solver
    if ( tile_idx.y == 0 ) {
        // This is an m = 1 field
        for( int i = block_thread_rank(); i < static_cast<int>(nx.x); i+=block_num_threads() ) {
            tile_E[ i + 0 * jstride ].z = ops::complex<float> {0,0};
            tile_E[ i + 0 * jstride ].r = ( 4.f * tile_E[ i + 1*jstride ].r - tile_E[ i + 2*jstride ].r ) / 3.f;
            tile_E[ i + 0 * jstride ].th = tile_E[ i + 1*jstride ].th;

            tile_B[ i + 0 * jstride ].z = ops::complex<float> {0,0};
            tile_B[ i + 0 * jstride ].r = + tile_B[ i + 1*jstride ].r;  
            tile_B[ i + 0 * jstride ].th = 0.125f * I * ( 9.f * tile_B[ i + 1*jstride ].r - tile_B[ i + 2*jstride ].r );
        }

        // values for j < 0 are unused for linear interpolation
        for( int i = block_thread_rank(); i < static_cast<int>(nx.x); i+=block_num_threads() ) {
            tile_E[ i + (-1) * jstride ].z = 0;
            tile_E[ i + (-1) * jstride ].r = 0;
            tile_E[ i + (-1) * jstride ].th = 0;

            tile_B[ i + (-1) * jstride ].z = 0;
            tile_B[ i + (-1) * jstride ].r = 0;
            tile_B[ i + (-1) * jstride ].th = 0;
        }
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
int Laser::Gaussian::launch(cyl3grid<ops::complex<float>>& E, cyl3grid<ops::complex<float>>& B, float2 const box ) {

    // std::cout << "Launching gaussian pulse...\n";

    if ( validate() < 0 ) return -1;

    if (( cos_pol == 0 ) && ( sin_pol == 0 )) {
        cos_pol = cos( polarization );
        sin_pol = sin( polarization );
    }

    uint2 g_nx = E.dims;

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

        Filter::Compensated fcomp( coord::z, filter);
        fcomp.apply( E );
        fcomp.apply( B );
    }

    // Get longitudinal field components by enforcing div = 0
    div_corr_z( E, B, dx );

    return 0;
}