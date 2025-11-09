#include "laser.h"

#include <iostream>
#include <cassert>

#include "filter.h"

/**
 * @brief Sets the longitudinal field components of E and B to ensure 0 divergence
 * 
 * The algorithm assumes 0 field at the right boundary of the box
 * 
 * This version only allows parallelism on the outermost (ty) loop
 * 
 * @param E     E field (mode 1)
 * @param B     B field (mode 1)
 * @param dx    cell size (z,y)
 */
void div_corr_z( cyl3grid<std::complex<float>>& E, cyl3grid<std::complex<float>>& B, const float2 dx ) {

    const int2 ntiles   = make_int2( E.ntiles.x, E.ntiles.y );
    const auto tile_vol = E.tile_vol;
    const int2  nx      = make_int2( E.nx.x, E.nx.y );
    const auto offset   = E.offset;
    const int jstride   = E.ext_nx.x; // Make sure jstride is signed because j may be < 0

    const double dz = dx.x;
    const double dr = dx.y;

    auto rdim = E.get_dims().y;

    /// @brief imaginary unit
    constexpr std::complex<double> I{0,1};

    #pragma omp parallel for
    for( int grid_j = 1; grid_j < rdim; grid_j ++ ) {

        std::complex<double> divEz = 0;
        std::complex<double> divBz = 0;

        ///@brief y tile index
        int ty = grid_j / nx.y;
        
        ///@brief j coordinate inside tile
        int j = grid_j - ty * nx.y;
        
        ///@brief r/Δr at center of cell
        const double rc  = grid_j;
        ///@brief r/Δr at center of lower (j-1) cell
        const double rcm = grid_j - 1;
        ///@brief r/Δr at upper edge of cell
        const double rp = grid_j + 0.5;
        ///@brief r/Δr at lower edge of cell
        const double rm = grid_j - 0.5;

        ///@brief Δz/r at center of cell
        const double dz_rc = dz / (rc * dr);
        ///@brief Δz/r at lower edge of cell
        const double dz_rm = dz / (rm * dr);

        // Process tiles right to left
        for( int tx = ntiles.x-1; tx >=0; tx -- ) {

            const auto tid      = ty * ntiles.x + tx;
            const auto tile_off = tid * tile_vol;

            auto * const __restrict__ tile_E = & E.d_buffer[ tile_off + offset ];
            auto * const __restrict__ tile_B = & B.d_buffer[ tile_off + offset ];

            for( int i = nx.x - 1; i >= 0; i-- ) {

                divEz += dz_rm * (
                        ( rc * tile_E[i+1 + j*jstride].r - rcm * tile_E[i+1 + (j-1)*jstride].r) + 
                        I * tile_E[i+1 + j*jstride].θ
                    ) ;
                tile_E[ i + j * jstride].z = divEz;
                
                divBz += dz_rc * (
                        ( rp * tile_B[i   + (j+1)*jstride].r - rm * tile_B[i + j*jstride].r ) + 
                        I * tile_B[i + j*jstride].θ
                    ) ;
                tile_B[ i + j * jstride ].z = divBz;
            }
        }
    }

    // Axial values
    for( int tx = 0; tx < ntiles.x; tx++ ) {
        const auto tid = tx; // ty = 0;
    
        const auto tile_off = tid * tile_vol;

        auto * const __restrict__ tile_E = & E.d_buffer[ tile_off + offset ];
        auto * const __restrict__ tile_B = & B.d_buffer[ tile_off + offset ];

        for( int i = 0; i < nx.x; i++ ) {
            tile_E[ i + 0 * jstride ].z = - tile_E[ i + 1 * jstride ].z;
            tile_B[ i + 0 * jstride ].z = 0;
        }
    }

    E.copy_to_gc();
    B.copy_to_gc();
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
int Laser::PlaneWave::launch( cyl3grid<std::complex<float>>& E, cyl3grid<std::complex<float>>& B, float2 box ) {

    // std::cout << "Launching plane wave...\n";

    if ( validate() < 0 ) return -1;

    if (( cos_pol == 0 ) && ( sin_pol == 0 )) {
        cos_pol = std::cos( polarization );
        sin_pol = std::sin( polarization );
    }

    uint2 g_nx = E.dims;

    float2 dx = make_float2(
        box.x / g_nx.x,
        box.y / g_nx.y
    );

    // Grid tile parameters
    const auto ntiles   = E.ntiles;
    const auto tile_vol = E.tile_vol;
    const auto nx       = E.nx;
    const auto offset   = E.offset;
    const int  jstride  = E.ext_nx.x; // jstride should be signed

    const float k = omega0;
    const float amp = omega0 * a0;

    ///@brief cell size in longitudinal direction
    const auto dz = dx.x;

    ///@brief imaginary unit
    constexpr std::complex<float> I{0,1};
    ///@brief exp( I pol )
    const std::complex<float> pol_r{ cos_pol, -sin_pol };
    ///@brief exp( I (pol-π/2) )
    const std::complex<float> pol_θ{ sin_pol, +cos_pol };

    // Loop over tiles
    #pragma omp parallel for
    for( int tid = 0; tid < ntiles.y * ntiles.x; tid++ ) {

        const auto ty = tid / ntiles.x;
        const auto tx = tid % ntiles.x;

        const auto tile_off = tid * tile_vol;

        // Copy data to shared memory and block
        auto * const __restrict__ tile_E = & E.d_buffer[ tile_off + offset ];
        auto * const __restrict__ tile_B = & B.d_buffer[ tile_off + offset ];

        const int i0 = tx * nx.x;

        for( unsigned j = 0; j < nx.y; j++ ) {
            for( unsigned i = 0; i < nx.x; i++ ) {
                const float z   = ( i0 + i ) * dz;
                const float z_2 = ( i0 + i + 0.5 ) * dz;

                float lenv   = amp * lon_env( z   );
                float lenv_2 = amp * lon_env( z_2 );

                tile_E[ i + j * jstride ].z = 0;
                tile_E[ i + j * jstride ].r = +lenv * std::cos( k * z ) * pol_r;
                tile_E[ i + j * jstride ].θ = -lenv * std::cos( k * z ) * pol_θ;

                tile_B[ i + j * jstride ].z = 0;
                tile_B[ i + j * jstride ].r = lenv_2 * std::cos( k * z_2 ) * pol_θ;
                tile_B[ i + j * jstride ].θ = lenv_2 * std::cos( k * z_2 ) * pol_r;
            }
        }

        // Correct axial cell values, see field solver
        if ( ty == 0 ) {
            // This is an m = 1 field
            for( int i = 0; i < nx.x; i++ ) {
                tile_E[ i + 0 * jstride ].z = 0;
                tile_E[ i + 0 * jstride ].r = ( 4.f * tile_E[ i + 1*jstride ].r - tile_E[ i + 2*jstride ].r ) / 3.f;
                tile_E[ i + 0 * jstride ].θ = tile_E[ i + 1*jstride ].θ;

                tile_E[ i + 0 * jstride ].z = 0;
                tile_B[ i + 0 * jstride ].r = + tile_B[ i + 1*jstride ].r;  
                tile_B[ i + 0 * jstride ].θ = 0.125f * I * ( tile_B[ i + 2*jstride ].r - 9.f * tile_B[ i + 2*jstride ].r );
            }

            // values for j < 0 are unused for linear interpolation
            for( int i = 0; i < nx.x; i++ ) {
                tile_E[ i + (-1) * jstride ].z = 0;
                tile_E[ i + (-1) * jstride ].r = 0;
                tile_E[ i + (-1) * jstride ].θ = 0;

                tile_B[ i + (-1) * jstride ].z = 0;
                tile_B[ i + (-1) * jstride ].r = 0;
                tile_B[ i + (-1) * jstride ].θ = 0;
            }
        }

    }

    E.copy_to_gc();
    B.copy_to_gc();

    if ( filter > 0 ) {

        Filter::Compensated fcomp( coord::x, filter);
        fcomp.apply(E);
        fcomp.apply(B);
    }

    // std::cout << "Plane wave launched\n";

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


/**
 * @brief Returns local phase for a gaussian beamn
 * 
 * @param omega0    Beam frequency
 * @param W0        Beam waist
 * @param z         Position along focal line (focal plane at z = 0)
 * @param r         Position transverse to focal line (focal line at r = 0)
 * @return          Local field value
 */
inline float gauss_phase( const float omega0, const float W0, const float z, const float r ) {
    const float z0   = omega0 * ( W0 * W0 ) / 2;
    const float rho2 = r*r;
    const float curv = rho2 * z / (z0*z0 + z*z);
    const float rWl2 = (z0*z0)/(z0*z0 + z*z);
    const float gouy_shift = atan2( z, z0 );

    return std::sqrt( std::sqrt(rWl2) ) * 
        std::exp( - rho2 * rWl2/( W0 * W0 ) ) * 
        std::cos( omega0*( z + curv ) - gouy_shift );
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
int Laser::Gaussian::launch( cyl3grid<std::complex<float>>& E, cyl3grid<std::complex<float>>& B, float2 const box ) {

    if ( validate() < 0 ) return -1;

    if (( cos_pol == 0 ) && ( sin_pol == 0 )) {
        cos_pol = std::cos( polarization );
        sin_pol = std::sin( polarization );
    }

    uint2 g_nx = E.dims;

    float2 dx = make_float2(
        box.x / g_nx.x,
        box.y / g_nx.y
    );

    // Grid tile parameters
    const auto ntiles   = E.ntiles;
    const auto tile_vol = E.tile_vol;
    const auto nx       = E.nx;
    const auto offset   = E.offset;
    const int  jstride  = E.ext_nx.x;   // ystride must be signed

    const float amp = omega0 * a0;

    ///@brief cell size in longitudinal direction
    const auto dz = dx.x;
    ///@brief cell size in radial direction
    const auto dr = dx.y;

    ///@brief imaginary unit
    constexpr std::complex<float> I{0,1};
    ///@brief exp( I pol )
    std::complex<float> pol_r{ cos_pol, -sin_pol };
    ///@brief exp( I (pol-π/2) )
    std::complex<float> pol_θ{ sin_pol, +cos_pol };

    // Loop over tiles
    #pragma omp parallel for
    for( int tid = 0; tid < ntiles.y * ntiles.x; tid++ ) {

        const auto tx = tid % ntiles.x;
        const auto ty = tid / ntiles.x;
        const auto tile_off = tid * tile_vol;

        // Copy data to shared memory and block
        auto * const __restrict__ tile_E = & E.d_buffer[ tile_off + offset ];
        auto * const __restrict__ tile_B = & B.d_buffer[ tile_off + offset ];

        const int i0 = tx * nx.x;
        const int j0 = ty * nx.y;

        for( unsigned j = 0; j < nx.y; j++ ) {
            for( unsigned i = 0; i < nx.x; i++ ) {
                const float z   = ( i0 + i       ) * dz;
                const float z_2 = ( i0 + i + 0.5 ) * dz;

                const float r   = ( j0 + j - 0.5 ) * dr;
                const float r_2 = ( j0 + j       ) * dr;

                const float lenv   = amp * lon_env( z   );
                const float lenv_2 = amp * lon_env( z_2 );

                tile_E[ i + j * jstride ].z = 0;
                tile_E[ i + j * jstride ].r = +lenv * gauss_phase( omega0, W0, z - focus, r_2 ) * pol_r;
                tile_E[ i + j * jstride ].θ = +lenv * gauss_phase( omega0, W0, z - focus, r   ) * pol_θ;

                tile_B[ i + j * jstride ].z = 0;
                tile_B[ i + j * jstride ].r = -lenv_2 * gauss_phase( omega0, W0, z_2 - focus, r   ) * pol_θ;
                tile_B[ i + j * jstride ].θ = +lenv_2 * gauss_phase( omega0, W0, z_2 - focus, r_2 ) * pol_r;
            }
        }

        // Correct axial cell values, see field solver
        if ( ty == 0 ) {
            // This is an m = 1 field
            for( int i = 0; i < nx.x; i++ ) {
                tile_E[ i + 0 * jstride ].z = 0;
                tile_E[ i + 0 * jstride ].r = ( 4.f * tile_E[ i + 1*jstride ].r - tile_E[ i + 2*jstride ].r ) / 3.f;
                tile_E[ i + 0 * jstride ].θ = tile_E[ i + 1*jstride ].θ;

                tile_B[ i + 0 * jstride ].z = 0;
                tile_B[ i + 0 * jstride ].r = + tile_B[ i + 1*jstride ].r;  
                tile_B[ i + 0 * jstride ].θ = 0.125f * I * ( 9.f * tile_B[ i + 1*jstride ].r - tile_B[ i + 2*jstride ].r );
            }

            // values for j < 0 are unused for linear interpolation
            for( int i = 0; i < nx.x; i++ ) {
                tile_E[ i + (-1) * jstride ].z = 0;
                tile_E[ i + (-1) * jstride ].r = 0;
                tile_E[ i + (-1) * jstride ].θ = 0;

                tile_B[ i + (-1) * jstride ].z = 0;
                tile_B[ i + (-1) * jstride ].r = 0;
                tile_B[ i + (-1) * jstride ].θ = 0;
            }
        }
    }

    E.copy_to_gc();
    B.copy_to_gc();

    if ( filter > 0 ) {

        Filter::Compensated fcomp( coord::x, filter);
        fcomp.apply(E);
        fcomp.apply(B);
    }

    div_corr_z( E, B, dx );

    // std::cout << "Launched gaussian pulse\n";

    return 0;
}
