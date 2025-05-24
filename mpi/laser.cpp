#include "laser.h"

#include <iostream>
#include <cassert>

#include "filter.h"

#include "timer.h"

/**
 * @brief Sets the longitudinal field components of E and B to ensure 0 divergence
 * 
 * The algorithm assumes 0 field at the right boundary of the box
 * 
 * @param E     E field
 * @param B     B field
 * @param dx    cell size
 */
void div_corr_x( vec3grid<float3>& E, vec3grid<float3>& B, const float2 dx ) {

    const int2 ntiles   = make_int2( E.get_ntiles().x, E.get_ntiles().y );
    const auto tile_vol = E.tile_vol;
    const auto nx       = E.nx;
    const auto offset   = E.offset;
    const int ystride   = E.ext_nx.x; // Make sure ystride is signed because (iy-1) may be < 0

    const double dx_dy = ((double) dx.x) / ((double) dx.y);

    const auto local_nx = E.local_nx;

    double  * __restrict__ sendbuf = memory::malloc<double>( 2 * local_nx.y );

    #pragma omp parallel for
    for( int local_iy = 0; local_iy < local_nx.y; local_iy ++ ) {

        double divEx = 0;
        double divBx = 0;

        int ty = local_iy / nx.y;
        int iy = local_iy - ty * nx.y;
        
        // Process tiles right to left
        for( int tx = ntiles.x-1; tx >=0; tx -- ) {

            const auto tile_idx = make_uint2( tx, ty );
            const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
            const auto tile_off = tid * tile_vol;

            float3 * const __restrict__ tile_E = & E.d_buffer[ tile_off + offset ];
            float3 * const __restrict__ tile_B = & B.d_buffer[ tile_off + offset ];

            for( int ix = nx.x - 1; ix >= 0; ix-- ) {

                divEx += dx_dy * (tile_E[ix+1 + iy*ystride].y - tile_E[ix+1 + (iy-1)*ystride].y);
                tile_E[ ix + iy * ystride].x = divEx;
                
                divBx += dx_dy * (tile_B[ix   + (iy+1)*ystride].y - tile_B[ix + iy*ystride].y);
                tile_B[ ix + iy * ystride ].x = divBx;
            }
        }

        // Copy final divergence values to send buffer
        sendbuf[              local_iy ] = divEx;
        sendbuf[ local_nx.y + local_iy ] = divBx;
    }

    // If there is a parallel partition along x, add in contribution from nodes to the right
    if ( E.part.dims.x > 1 ) {
        double * __restrict__ recvbuf = memory::malloc<double>( 2 * local_nx.y );

        // Create a communicator with nodes having the same y coordinate
        int color = E.part.get_coords().y;
        // Reorder ranks right to left
        int key   = E.part.dims.x - E.part.get_coords().x;
        MPI_Comm newcomm;
        MPI_Comm_split( E.part.get_comm(), color, key, &newcomm );
        
        // Add contribution from all nodes to the right
        MPI_Exscan( sendbuf, recvbuf, 2 * local_nx.y, MPI_DOUBLE, MPI_SUM, newcomm );

        // Add result to local grid
        // Rightmost node does not need to do this
        if ( key > 0 ) {
            #pragma omp parallel for
            for( int local_iy = 0; local_iy < local_nx.y; local_iy ++ ) {

                double divEx = recvbuf[              local_iy ];
                double divBx = recvbuf[ local_nx.y + local_iy ];

                int ty = local_iy / nx.y;
                int iy = local_iy - ty * nx.y;
                
                for( int tx = 0; tx < ntiles.x; tx++ ) {

                    const auto tile_idx = make_uint2( tx, ty );
                    const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                    const auto tile_off = tid * tile_vol;

                    float3 * const __restrict__ tile_E = & E.d_buffer[ tile_off + offset ];
                    float3 * const __restrict__ tile_B = & B.d_buffer[ tile_off + offset ];

                    for( int ix = 0; ix < nx.x; ix++ ) {

                        tile_E[ ix + iy * ystride].x += divEx;
                        tile_B[ ix + iy * ystride].x += divBx;
                    }
                }
            }
        }

        MPI_Comm_free( &newcomm );
        memory::free( recvbuf );
    }

    // Free temporary memory
    memory::free( sendbuf );

    // Correct longitudinal values on guard cells
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
int Laser::PlaneWave::launch( vec3grid<float3>& E, vec3grid<float3>& B, float2 box ) {

    // std::cout << "Launching plane wave...\n";

    if ( validate() < 0 ) return -1;

    if (( cos_pol == 0 ) && ( sin_pol == 0 )) {
        cos_pol = std::cos( polarization );
        sin_pol = std::sin( polarization );
    }

    uint2 global_nx = E.get_global_nx();

    float2 dx = make_float2(
        box.x / global_nx.x,
        box.y / global_nx.y
    );

    // Grid tile parameters
    const auto ntiles     = E.get_ntiles();
    const auto tile_vol   = E.tile_vol;
    const auto nx         = E.nx;
    const auto offset     = E.offset;
    const int  ystride    = E.ext_nx.x; // ystride should be signed
    const uint2 global_tile_off  = E.get_tile_off();

    const float k = omega0;
    const float amp = omega0 * a0;

    // Loop over tiles
    #pragma omp parallel for
    for( int tid = 0; tid < ntiles.y * ntiles.x; tid++ ) {

        const auto ty = tid / ntiles.x;
        const auto tx = tid % ntiles.x;

        const auto tile_idx = make_uint2( tx, ty );
        const auto tile_off = tid * tile_vol;

        // Copy data to shared memory and block
        float3 * const __restrict__ tile_E = & E.d_buffer[ tile_off + offset ];
        float3 * const __restrict__ tile_B = & B.d_buffer[ tile_off + offset ];

        const int ix0 = ( global_tile_off.x + tile_idx.x ) * nx.x;

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
    const float curv = 0.5 * rho2 * z / (z0*z0 + z*z);
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

    E.zero();
    B.zero();

    uint2 global_nx = E.get_global_nx();;

    float2 dx = make_float2(
        box.x / global_nx.x,
        box.y / global_nx.y
    );

    // Grid tile parameters
    const auto ntiles   = E.get_ntiles();
    const auto tile_vol = E.tile_vol;
    const auto nx       = E.nx;
    const auto offset   = E.offset;
    const int  ystride  = E.ext_nx.x;   // ystride must be signed
    const uint2 global_tile_off  = E.get_tile_off();

    const float amp = omega0 * a0;

    // Loop over tiles
    #pragma omp parallel for
    for( int tid = 0; tid < ntiles.y * ntiles.x ; tid++ ) {
        const auto ty = tid / ntiles.x;
        const auto tx = tid % ntiles.x;
        const auto tile_off = tid * tile_vol;

        // Copy data to shared memory and block
        float3 * const __restrict__ tile_E = & E.d_buffer[ tile_off + offset ];
        float3 * const __restrict__ tile_B = & B.d_buffer[ tile_off + offset ];

        const int ix0 = ( global_tile_off.x + tx ) * nx.x;
        const int iy0 = ( global_tile_off.y + ty ) * nx.y;

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

    E.copy_to_gc();
    B.copy_to_gc();

    // Set longitudinal field components
    div_corr_x( E, B, dx );

    // Apply filtering if required
    if ( filter > 0 ) {
        Filter::Compensated fcomp( coord::x, filter);
        fcomp.apply(E);
        fcomp.apply(B);
    }

    return 0;
}