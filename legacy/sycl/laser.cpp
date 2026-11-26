#include "laser.h"

#include <iostream>
#include <cassert>

#include "filter.h"

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
inline float lon_env( const float z, const float start, const float rise, const float flat, const float fall ) {

    if ( z > start ) {
        // Ahead of laser
        return 0.0;
    } else if ( z > start - rise ) {
        // Laser rise
        float csi = z - start;
        float e = sycl::sin( M_PI_2 * csi / rise );
        return e*e;
    } else if ( z > start - (rise + flat) ) {
        // Flat-top
        return 1.0;
    } else if ( z > start - (rise + flat + fall) ) {
        // Laser fall
        float csi = z - (start - rise - flat - fall);
        float e = sycl::sin( M_PI_2 * csi / fall );
        return e*e;
    }

    // Before laser
    return 0.0;
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
void div_corr_x( vec3grid<float3>& E, vec3grid<float3>& B, const float2 dx, sycl::queue & q ) {

    auto tile_vol     = E.tile_vol;
    auto nx           = E.nx;
    auto ntiles       = E.ntiles;
    auto * E_d_buffer = E.d_buffer;
    auto * B_d_buffer = B.d_buffer;
    auto offset       = E.offset;
    int ystride       = E.ext_nx.x;

    const double dx_dy = (double) dx.x / (double) dx.y;

    // Check that local memory can hold up to 2 times the tile buffer
    auto local_mem_size = q.get_device().get_info<sycl::info::device::local_mem_size>();
    if ( local_mem_size < 2 * tile_vol * sizeof( float3 ) ) {
        std::cerr << "(*error*) Tile size too large " << nx << " (plus guard cells)\n";
        std::cerr << "(*error*) Insufficient local memory (" << local_mem_size << " B) for div_corr_x() function.\n";
        abort();
    }

    // Temporary buffer for divergence calculations
    size_t bsize = E.ntiles.x * (E.ntiles.y * E.nx.y);
    double2 * tmp = device::malloc<double2>( bsize, q );

    /**
     * Step A - Get per-tile E and B divergence at tile left edge starting
     *          from 0.0
     * 
     * This could (potentially) be improved by processing each line in a warp,
     * and replacing the divEx and divBx calculations by warp level reductions
     */
    q.submit([&](sycl::handler &h) {

        // Group shared memory
        auto E_local = sycl::local_accessor< float3, 1 > ( tile_vol, h );
        auto B_local = sycl::local_accessor< float3, 1 > ( tile_vol, h );

        // 8×1 work items per group
        sycl::range<2> local{ 8, 1 };

        // ntiles.x × ntiles.y groups
        sycl::range<2> global{ ntiles.x, ntiles.y };

        h.parallel_for( 
            sycl::nd_range{ global * local, local },
            [=](sycl::nd_item<2> it) { 

            const auto tile_idx = make_uint2( it.get_group(0), it.get_group(1) );
            const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
            const auto tile_off = tid * tile_vol;

            for( unsigned idx = it.get_local_id(0); idx < tile_vol; idx += it.get_local_range(0) ) {
                E_local[idx] = E_d_buffer[ tile_off + idx ];
                B_local[idx] = B_d_buffer[ tile_off + idx ];
            }
            it.barrier();

            float3 * const __restrict__ E = & E_local[ offset ];
            float3 * const __restrict__ B = & B_local[ offset ]; 

            auto tmp_off = tile_idx.y * nx.y * ntiles.x;

            for( int iy = it.get_local_id(0); iy < nx.y; iy += it.get_local_range(0) ) {
                // Find divergence at left edge
                double divEx = 0;
                double divBx = 0;
                for( int ix = nx.x - 1; ix >= 0; ix-- ) {
                    divEx += dx_dy * (E[ix+1 + iy*ystride].y - E[ix+1 + (iy-1)*ystride ].y);
                    divBx += dx_dy * (B[ix + (iy+1)*ystride].y - B[ix + iy*ystride ].y);
                }

                // Write result to tmp. array
                tmp[ tmp_off + iy * ntiles.x + tile_idx.x ] = make_double2( divEx, divBx );
            }
        });
    });
    q.wait();

    /**
     * Step B - Performs a left-going scan operation on the results from step A.
     * 
     * This could (potentially) be improved by processing each row of tiles in
     * a warp:
     * - Copy the data from tmp and store it in shared memory in reverse order
     * - Do a normal ex-scan operation using warp accelerated code
     * - Copy the data in reverse order from shared memory and store it in tmp
     */
    q.submit([&](sycl::handler &h) {

        // Group shared memory
        auto buffer = sycl::local_accessor< double2, 1 > ( ntiles.x, h );

        // 8 work items per group
        sycl::range<1> local{ 8 };

        // ntiles.x × ntiles.y groups
        sycl::range<1> global{ E.gnx.y };

        h.parallel_for( 
            sycl::nd_range{global * local , local},
            [=](sycl::nd_item<1> it) { 
            
            int giy = it.get_group(0);

            for( int i = it.get_local_id(0); i < ntiles.x; i += it.get_local_range(0) ) {
                buffer[i] = tmp[ giy * ntiles.x + i ];
            }
            it.barrier();

            // Perform scan operation (serial inside block)
            if ( it.get_local_id(0) == 0 ) {
                double2 a = make_double2( 0, 0 );
                for( int i = ntiles.x-1; i >= 0; i--) {
                    double2 b = buffer[i];
                    buffer[i] = a;
                    a.x += b.x;
                    a.y += b.y;
                }
            }
            it.barrier();

            // Copy data to global memory
            for( int i = it.get_local_id(0); i < ntiles.x; i += it.get_local_range(0) ) {
                tmp[ giy * ntiles.x + i ] = buffer[i];
            }
        });
    });
    q.wait();

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
    q.submit([&](sycl::handler &h) {
        // Group shared memory
        auto tmp_E = sycl::local_accessor< float3, 1 > ( tile_vol, h );
        auto tmp_B = sycl::local_accessor< float3, 1 > ( tile_vol, h );

        // 8×1 work items per group
        sycl::range<2> local{ 8, 1 };

        // ntiles.x × ntiles.y groups
        sycl::range<2> global{ ntiles.x, ntiles.y };

        h.parallel_for( 
            sycl::nd_range{global * local , local},
            [=](sycl::nd_item<2> it) { 

            const auto tile_idx = make_uint2( it.get_group(0), it.get_group(1) );
            const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
            const auto tile_off = tid * tile_vol;

            for( unsigned idx = it.get_local_id(0); idx < tile_vol; idx += it.get_local_range(0) ) {
                tmp_E[idx] = E_d_buffer[ tile_off + idx ];
                tmp_B[idx] = B_d_buffer[ tile_off + idx ];
            }
            it.barrier();

            float3 * const __restrict__ E = &tmp_E[0] + offset;
            float3 * const __restrict__ B = &tmp_B[0] + offset; 

            auto tmp_off = tile_idx.y * nx.y * ntiles.x;

            for( int iy = it.get_local_id(0); iy < nx.y; iy += it.get_local_range(0) ) {
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
            it.barrier();

            for( unsigned idx = it.get_local_id(0); idx < tile_vol; idx += it.get_local_range(0) ) {
                E_d_buffer[ tile_off + idx ] = tmp_E[idx];
                B_d_buffer[ tile_off + idx ] = tmp_B[idx];
            }        
        });
    });
    q.wait();

    // Free temporary memory
    device::free( tmp, q );

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
int Laser::PlaneWave::launch( vec3grid<float3>& E, vec3grid<float3>& B, float2 box, sycl::queue & q ) {

    if ( validate() < 0 ) return -1;

    if (( cos_pol == 0 ) && ( sin_pol == 0 )) {
        cos_pol = sycl::cos( polarization );
        sin_pol = sycl::sin( polarization );
    }

    uint2 g_nx = E.gnx;

    float2 dx = make_float2(
        box.x / g_nx.x,
        box.y / g_nx.y
    );

    q.submit([&](sycl::handler &h) {
        auto ntiles   = E.ntiles;
        auto tile_vol = E.tile_vol;
        auto nx       = E.nx;
        auto offset   = E.offset;
        int  ystride  = E.ext_nx.x; // ystride should be signed

        auto * E_buffer = E.d_buffer;
        auto * B_buffer = B.d_buffer;

        float k = omega0;
        float amp = omega0 * a0;

        auto cos_pol = this -> cos_pol;
        auto sin_pol = this -> sin_pol;

        auto start = this -> start;
        auto rise  = this -> rise;
        auto flat  = this -> flat;
        auto fall  = this -> fall;


        // 8×1 work items per group
        sycl::range<2> local{ 8, 1 };

        // ntiles.x × ntiles.y groups
        sycl::range<2> global{ ntiles.x, ntiles.y };

        h.parallel_for( 
            sycl::nd_range{global * local , local},
            [=](sycl::nd_item<2> it) { 

            const auto tile_idx = make_uint2( it.get_group(0), it.get_group(1) );
            const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
            const auto tile_off = tid * tile_vol;

            // Copy data to shared memory and block
            float3 * const __restrict__ tile_E = & E_buffer[ tile_off + offset ];
            float3 * const __restrict__ tile_B = & B_buffer[ tile_off + offset ];

            const int ix0 = tile_idx.x * nx.x;

            for( unsigned idx = it.get_local_id(0); idx < nx.y * nx.x; idx += it.get_local_range(0) ) {
                const auto ix = idx % nx.x;
                const auto iy = idx / nx.x; 

                const float z   = ( ix0 + ix ) * dx.x;
                const float z_2 = ( ix0 + ix + 0.5 ) * dx.x;

                float lenv   = amp * lon_env( z   , start, rise, flat, fall );
                float lenv_2 = amp * lon_env( z_2 , start, rise, flat, fall );

                tile_E[ ix + iy * ystride ] = make_float3(
                    0,
                    +lenv * sycl::cos( k * z ) * cos_pol,
                    +lenv * sycl::cos( k * z ) * sin_pol
                );

                tile_B[ ix + iy * ystride ] = make_float3(
                    0,
                    -lenv_2 * sycl::cos( k * z_2 ) * sin_pol,
                    +lenv_2 * sycl::cos( k * z_2 ) * cos_pol
                );
            }
        });
    });

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
    const float gouy_shift = std::atan2( z, z0 );

    return sycl::sqrt( sycl::sqrt(rWl2) ) * 
        sycl::exp( - rho2 * rWl2/( W0 * W0 ) ) * 
        sycl::cos( omega0*( z + curv ) - gouy_shift );
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
int Laser::Gaussian::launch(vec3grid<float3>& E, vec3grid<float3>& B, float2 const box, sycl::queue & q ) {

    // std::cout << "Launching gaussian pulse...\n";

    if ( validate() < 0 ) return -1;

    if (( cos_pol == 0 ) && ( sin_pol == 0 )) {
        cos_pol = sycl::cos( polarization );
        sin_pol = sin( polarization );
    }

    uint2 g_nx = E.gnx;

    float2 dx = make_float2(
        box.x / g_nx.x,
        box.y / g_nx.y
    );

    q.submit([&](sycl::handler &h) {
        const auto ntiles   = E.ntiles;
        const auto tile_vol = E.tile_vol;
        const auto nx       = E.nx;
        const auto offset   = E.offset;
        const int  ystride  = E.ext_nx.x;   // ystride must be signed

        auto * E_buffer = E.d_buffer;
        auto * B_buffer = B.d_buffer;

        const auto cos_pol = this -> cos_pol;
        const auto sin_pol = this -> sin_pol;

        const auto axis    = this -> axis;
        const auto omega0  = this -> omega0;
        const auto W0      = this -> W0;
        const auto focus   = this -> focus;

        const auto start = this -> start;
        const auto rise  = this -> rise;
        const auto flat  = this -> flat;
        const auto fall  = this -> fall;

        const float amp = omega0 * a0;

        // 8×1 work items per group
        sycl::range<2> local{ 8, 1 };

        // ntiles.x × ntiles.y groups
        sycl::range<2> global{ ntiles.x, ntiles.y };

        h.parallel_for( 
            sycl::nd_range{global * local , local},
            [=](sycl::nd_item<2> it) { 

            const auto tile_idx = make_uint2( it.get_group(0), it.get_group(1) );
            const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
            const auto tile_off = tid * tile_vol;

            float3 * const __restrict__ tile_E = & E_buffer[ tile_off + offset ];
            float3 * const __restrict__ tile_B = & B_buffer[ tile_off + offset ];
 
            const int ix0 = tile_idx.x * nx.x;
            const int iy0 = tile_idx.y * nx.y;

            for( unsigned idx = it.get_local_id(0); idx < nx.y * nx.x; idx += it.get_local_range(0) ) {
                const auto ix = idx % nx.x;
                const auto iy = idx / nx.x; 

                const float z   = ( ix0 + ix ) * dx.x;
                const float z_2 = ( ix0 + ix + 0.5 ) * dx.x;

                const float r   = (iy0 + iy ) * dx.y - axis;
                const float r_2 = (iy0 + iy + 0.5 ) * dx.y - axis;

                const float lenv   = amp * lon_env( z   , start, rise, flat, fall );
                const float lenv_2 = amp * lon_env( z_2 , start, rise, flat, fall );

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
        });
    });
    q.wait();

    E.copy_to_gc( );
    B.copy_to_gc( );

    if ( filter > 0 ) {

        Filter::Compensated fcomp( coord::x, filter);
        fcomp.apply( E );
        fcomp.apply( B );
    }

    // Get longitudinal field components by enforcing div = 0
    div_corr_x( E, B, dx, q );

    return 0;
}