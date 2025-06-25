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
 * @param E     E field
 * @param B     B field
 * @param dx    cell size
 */
void div_corr_x( vec3grid<float3>& E, vec3grid<float3>& B, const float2 dx ) {

    const int2 ntiles = make_int2( E.ntiles.x, E.ntiles.y );
    const auto tile_vol = E.tile_vol;
    const auto nx       = E.nx;
    const auto offset   = E.offset;
    const int ystride   = E.ext_nx.x; // Make sure ystride is signed because iy may be < 0

    const double dx_dy = ((double) dx.x) / ((double) dx.y);

    for( int ty = 0; ty < ntiles.y; ty ++ ) {

        // If paralelizing over y tiles these should be defined by Y tile group (tiles with same tile.y) 
        double divEx[ nx.y ];
        double divBx[ nx.y ];

        for( unsigned iy = 0; iy < nx.y; iy++ ) {
            divEx[ iy ] = divBx[ iy ] = 0;
        }
        
        // Process tiles right to left
        for( int tx = ntiles.x-1; tx >=0; tx -- ) {

            const auto tile_idx = make_uint2( tx, ty );
            const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
            const auto tile_off = tid * tile_vol;

            // Copy data to shared memory and block
            float3 * const __restrict__ tile_E = & E.d_buffer[ tile_off + offset ];
            float3 * const __restrict__ tile_B = & B.d_buffer[ tile_off + offset ];

            for( unsigned iy = 0; iy < nx.y; iy++ ) {
                double tmpDivEx = divEx[ iy ];
                double tmpDivBx = divBx[ iy ];

                for( int ix = nx.x - 1; ix >= 0; ix-- ) {

                    tmpDivEx += dx_dy * (tile_E[ix+1 + iy*ystride].y - tile_E[ix+1 + (iy-1)*ystride].y);
                    tile_E[ ix + iy * ystride].x = tmpDivEx;
                    
                    tmpDivBx += dx_dy * (tile_B[ix   + (iy+1)*ystride].y - tile_B[ix + iy*ystride].y);
                    tile_B[ ix + iy * ystride ].x = tmpDivBx;
                }

                divEx[iy] = tmpDivEx;
                divBx[iy] = tmpDivBx;
            }
        }
    }

}

/**
 * @brief Sets the longitudinal field components of E and B to ensure 0 divergence
 * 
 * The algorithm assumes 0 field at the right boundary of the box
 * 
 * This version allows for more parallelism, similar to GPU versions
 * 
 * @param E     E field
 * @param B     B field
 * @param dx    cell size
 */
void div_corr_x_mk1( vec3grid<float3>& E, vec3grid<float3>& B, const float2 dx ) {

    const auto ntiles   = E.ntiles;
    const auto tile_vol = E.tile_vol;
    const int2 nx       = make_int2(E.nx.x, E.nx.y);
    const auto offset   = E.offset;
    const int ystride   = E.ext_nx.x;

    const double dx_dy = ((double) dx.x) / ((double) dx.y);

    size_t bsize = ntiles.x * (ntiles.y * nx.y);
    double tmpE[ bsize ];
    double tmpB[ bsize ];

    // Get divergence inside each tile
    for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
        for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {

            const auto tile_idx = make_uint2( tx, ty );
            const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
            const auto tile_off = tid * tile_vol;

            // Copy data to shared memory and block
            float3 * const __restrict__ tile_E = & E.d_buffer[ tile_off + offset ];
            float3 * const __restrict__ tile_B = & B.d_buffer[ tile_off + offset ];

            for( int iy = 0; iy < nx.y; iy++ ) {
                // Find divergence at left edge
                double divEx = 0;
                double divBx = 0;
                for( int ix = nx.x - 1; ix >= 0; ix-- ) {
                    divEx += dx_dy * (tile_E[ix+1 +     iy*ystride].y - tile_E[ix+1 + (iy-1)*ystride].y);
                    divBx += dx_dy * (tile_B[ix   + (iy+1)*ystride].y - tile_B[ix   +     iy*ystride].y);
                }

                const int idx = (tile_idx.y * nx.y + iy) * ntiles.x + tile_idx.x;
                tmpE[ idx ] = divEx;
                tmpB[ idx ] = divBx;
            }
        }
    }

    // Do a left scan to find accumulated divergence
    for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
        for( int iy = 0; iy < nx.y; iy++ ) {
            double divEx = 0;
            double divBx = 0;
            for( int tx = ntiles.x-1; tx >= 0; tx -- ) {
                auto tile_idx = make_uint2( tx, ty );
                const int idx = (tile_idx.y * nx.y + iy) * ntiles.x + tile_idx.x;

                auto tE = tmpE[ idx ] + divEx;
                auto tB = tmpB[ idx ] + divBx;

                tmpE[ idx ] = divEx;
                tmpB[ idx ] = divBx;

                divEx = tE;
                divBx = tB;
            }
        }
    }

    // Correct divergence
    for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
        for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {

            const auto tile_idx = make_uint2( tx, ty );
            const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
            const auto tile_off = tid * tile_vol;

            // Copy data to shared memory and block
            float3 * const __restrict__ tile_E = & E.d_buffer[ tile_off + offset ];
            float3 * const __restrict__ tile_B = & B.d_buffer[ tile_off + offset ];

            for( int iy = 0; iy < nx.y; iy++ ) {
                auto idx = (tile_idx.y * nx.y + iy) * ntiles.x + tile_idx.x;
                auto divEx = tmpE[ idx ];
                auto divBx = tmpB[ idx ];

                for( int ix = nx.x - 1; ix >= 0; ix-- ) {
                    divEx += dx_dy * (tile_E[ix+1 +     iy*ystride].y - tile_E[ix+1 + (iy-1)*ystride].y);
                    tile_E[ ix + iy * ystride].x = divEx;
                    
                    divBx += dx_dy * (tile_B[ix   + (iy+1)*ystride].y - tile_B[ix   +     iy*ystride].y);
                    tile_B[ ix + iy * ystride ].x = divBx;
                }
            }
        }
    }
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
int Laser::PlaneWave::launch( vec3grid<float3>& E, vec3grid<float3>& B, float2 box ) {

    // std::cout << "Launching plane wave...\n";

    if ( validate() < 0 ) return -1;

    if (( cos_pol == 0 ) && ( sin_pol == 0 )) {
        cos_pol = std::cos( polarization );
        sin_pol = std::sin( polarization );
    }

    uint2 g_nx = E.gnx;

    float2 dx = make_float2(
        box.x / g_nx.x,
        box.y / g_nx.y
    );

    // Grid tile parameters
    const auto ntiles   = E.ntiles;
    const auto tile_vol = E.tile_vol;
    const auto nx       = E.nx;
    const auto offset   = E.offset;
    const int  ystride  = E.ext_nx.x; // ystride should be signed

    const float k = omega0;
    const float amp = omega0 * a0;

    // Loop over tiles
    for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
        for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {

            const auto tile_idx = make_uint2( tx, ty );
            const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
            const auto tile_off = tid * tile_vol;

            // Copy data to shared memory and block
            float3 * const __restrict__ tile_E = & E.d_buffer[ tile_off + offset ];
            float3 * const __restrict__ tile_B = & B.d_buffer[ tile_off + offset ];

            const int ix0 = tile_idx.x * nx.x;

            for( unsigned iy = 0; iy < nx.y; iy++ ) {
                for( unsigned ix = 0; ix < nx.x; ix++ ) {
                    const float z   = ( ix0 + ix ) * dx.x;
                    const float z_2 = ( ix0 + ix + 0.5 ) * dx.x;

                    float lenv   = amp * lon_env( z   );
                    float lenv_2 = amp * lon_env( z_2 );

                    tile_E[ ix + iy * ystride ] = make_float3(
                        0,
                        +lenv * std::cos( k * z ) * cos_pol,
                        +lenv * std::cos( k * z ) * sin_pol
                    );

                    tile_B[ ix + iy * ystride ] = make_float3(
                        0,
                        -lenv_2 * std::cos( k * z_2 ) * sin_pol,
                        +lenv_2 * std::cos( k * z_2 ) * cos_pol
                    );
                }
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
int Laser::Gaussian::launch(vec3grid<float3>& E, vec3grid<float3>& B, float2 const box ) {

    // std::cout << "Launching gaussian pulse...\n";

    if ( validate() < 0 ) return -1;

    if (( cos_pol == 0 ) && ( sin_pol == 0 )) {
        cos_pol = std::cos( polarization );
        sin_pol = std::sin( polarization );
    }

    uint2 g_nx = E.gnx;

    float2 dx = make_float2(
        box.x / g_nx.x,
        box.y / g_nx.y
    );

    // Grid tile parameters
    const auto ntiles   = E.ntiles;
    const auto tile_vol = E.tile_vol;
    const auto nx       = E.nx;
    const auto offset   = E.offset;
    const int  ystride  = E.ext_nx.x;   // ystride must be signed

    const float amp = omega0 * a0;

    // Loop over tiles
    for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
        for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {

            const auto tile_idx = make_uint2( tx, ty );
            const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
            const auto tile_off = tid * tile_vol;

            // Copy data to shared memory and block
            float3 * const __restrict__ tile_E = & E.d_buffer[ tile_off + offset ];
            float3 * const __restrict__ tile_B = & B.d_buffer[ tile_off + offset ];
 
            const int ix0 = tile_idx.x * nx.x;
            const int iy0 = tile_idx.y * nx.y;

            for( unsigned iy = 0; iy < nx.y; iy++ ) {
                for( unsigned ix = 0; ix < nx.x; ix++ ) {
                    const float z   = ( ix0 + ix ) * dx.x;
                    const float z_2 = ( ix0 + ix + 0.5 ) * dx.x;

                    const float r   = (iy0 + iy ) * dx.y - axis;
                    const float r_2 = (iy0 + iy + 0.5 ) * dx.y - axis;

                    const float lenv   = amp * lon_env( z   );
                    const float lenv_2 = amp * lon_env( z_2 );

                    tile_E[ ix + iy * ystride ] = make_float3(
                        0,
                        +lenv * gauss_phase( omega0, W0, z - focus, r_2 ) * cos_pol,
                        +lenv * gauss_phase( omega0, W0, z - focus, r   ) * sin_pol
                    );
                    tile_B[ ix + iy * ystride ] = make_float3(
                        0,
                        -lenv_2 * gauss_phase( omega0, W0, z_2 - focus, r   ) * sin_pol,
                        +lenv_2 * gauss_phase( omega0, W0, z_2 - focus, r_2 ) * cos_pol
                    );

                }
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

    div_corr_x( E, B, dx );

    // std::cout << "Gaussian pulse launched\n";

    return 0;
}
