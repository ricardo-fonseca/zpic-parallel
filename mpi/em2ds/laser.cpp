#include "laser.hpp"


#include "grid/fft.hpp"
#include "filter.hpp"
#include "parallel/mpi.hpp"

#include <iostream>
#include <cassert>

namespace laser {

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
inline float lon_env( laser::pulse & laser, float z ) {

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
 * @brief Ensures div F = 0 by modifying the x component of F
 * 
 * @param data      FFT of field data organized as a contiguous grid
 * @param dims      Dimensions of k-space grid
 * @param dk        k-space cell size
 */
inline void lon_x( 
    std::complex<float> * const __restrict__ fld_x, 
    const std::complex<float> * const __restrict__ fld_y, 
    const uint2 global_dims, const uint2 local_dims, const uint2 local_start, const float2 dk ) {

    #pragma omp for
    for( unsigned idx = 0; idx < local_dims.y * local_dims.x; idx ++ ){
        const int ix = local_start.x + idx % local_dims.x;
        const int iy = local_start.y + idx / local_dims.x;

        // Note that ky is along the x direction, and kx is along the y direction
        const float ky = (( 2 * ix < int(global_dims.x) ) ? ix : ( ix - int(global_dims.x) ) ) * dk.y;
        const float kx = iy * dk.x;

        fld_x[idx] = ( kx > 0 ) ? - ky * fld_y[idx] / kx : 0.f;
    }
}

}

/**
 * @brief Validates laser parameters
 * 
 * @return      0 on success, -1 on error
 */
int laser::pulse::validate() {

    if ( a0 <= 0 ) {
        mpi::fatal("Invalid laser parameter (a0), must be > 0");
    }    

    if ( omega0 <= 0 ) {
        mpi::fatal( "Invalid laser parameter (omega0), must be > 0" );
    }    

    if ( fwhm > 0 ) {
        // The fwhm parameter overrides the rise/flat/fall parameters
        rise = fwhm;
        fall = fwhm;
        flat = 0.;
    } else {
        if ( rise <= 0 ) {
            mpi::fatal( "Invalid laser parameter (rise), must be > 0" );
        }

        if ( flat < 0 ) {
            mpi::fatal("Invalid laser parameter (flat), must be >= 0" );
        }

        if ( fall <= 0 ) {
            mpi::fatal( "Invalid laser parameter (fall), must be > 0" );
        }
    }

    return 0;
}

/**
 * @brief Adds a new laser pulse onto an EMF object
 * 
 * @param emf   EMF object
 * @return      Returns 0 on success, -1 on error (invalid laser parameters)
 */
int laser::pulse::add( emf & emf ) {

    grid::tiled_vec3<float> tmp_E( emf.E -> global_ntiles, emf.E-> tile_dims, emf.E -> gc, emf.E -> part );
    grid::tiled_vec3<float> tmp_B( emf.E -> global_ntiles, emf.E-> tile_dims, emf.E -> gc, emf.E -> part );

    // Get laser fields
    int ierr = launch( tmp_E, tmp_B, emf.box );

    if ( ! ierr ) {

        // Add to k-space fields
        grid::fft::r2c_plan dft_forward( tmp_E );
        auto fft_tmp = grid::fft::complex_grid( tmp_E );
        const float2 dk = grid::fft::dk( emf.box );

        filter::lowpass filter( make_float2( 0.5, 0.5 ) );

        // transform tmp_E and add to fEt
        dft_forward.transform( fft_tmp, tmp_E );
        lon_x( fft_tmp, dk );
        filter.apply( fft_tmp );
        emf.fEt -> add( fft_tmp );

        emf.fft_backward -> transform( tmp_E, fft_tmp  );
        emf.E -> add( tmp_E );

        // transform tmp_B and add to fB 
        dft_forward.transform( fft_tmp, tmp_B );
        lon_x( fft_tmp, dk );
        filter.apply( fft_tmp );
        emf.fB  -> add( fft_tmp );

        emf.fft_backward -> transform( tmp_B, fft_tmp );
        emf.B -> add( tmp_B );
    }

    return ierr;
};

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
int laser::plane_wave::launch( grid::tiled_vec3<float>& E, grid::tiled_vec3<float>& B, float2 box ) {

    if ( validate() < 0 ) return -1;

    if (( cos_pol == 0 ) && ( sin_pol == 0 )) {
        cos_pol = std::cos( polarization );
        sin_pol = std::sin( polarization );
    }

    const float2 dx = make_float2(
        box.x / E.get_global_dims().x,
        box.y / E.get_global_dims().y
    );

    // Laser k
    const float k = omega0;
    // Amplitude
    const float amp = omega0 * a0;

    // Grid tile parameters
    auto local_ntiles = E.get_local_ntiles();
    auto tile_dims = E.tile_dims;
    auto tile_start = E.get_local_tile_start();
    int ystride = E.tile_ext_dims.x;

    // Loop over tiles
    #pragma omp parallel for collapse(2)
    for( unsigned ty = 0; ty < local_ntiles.y; ty++ ) {
        for( unsigned tx = 0; tx < local_ntiles.x; tx++ ) {
            // Copy data to shared memory and block
            float3 * const __restrict__ tile_E = E.tile_data( tx, ty );
            float3 * const __restrict__ tile_B = B.tile_data( tx, ty );

            const int ix0 = (tile_start.x + tx) * tile_dims.x;

            for( unsigned iy = 0; iy < tile_dims.y; iy++ ) {
                for( unsigned ix = 0; ix < tile_dims.x; ix++ ) {
                    const float z   = ( ix0 + ix ) * dx.x;

                    float lenv   = amp * laser::lon_env( *this, z ) * std::cos( k * z );

                    tile_E[ ix + iy * ystride ] = make_float3(
                        0,
                        +lenv * cos_pol,
                        +lenv * sin_pol
                    );

                    tile_B[ ix + iy * ystride ] = make_float3(
                        0,
                        -lenv * sin_pol,
                        +lenv * cos_pol
                    );
                }
            }
        }
    }

    E.copy_to_gc();
    B.copy_to_gc();

    return 0;
}


/**
 * @brief Validate Gaussian laser parameters
 * 
 * @return      0 on success, -1 on error
 */
int laser::gaussian::validate() {
    
    if ( laser::pulse::validate() < 0 ) {
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
int laser::gaussian::launch(grid::tiled_vec3<float>& E, grid::tiled_vec3<float>& B, const float2 box ) {

    if ( validate() < 0 ) return -1;

    if (( cos_pol == 0 ) && ( sin_pol == 0 )) {
        cos_pol = std::cos( polarization );
        sin_pol = std::sin( polarization );
    }

    const float2 dx = make_float2(
        box.x / E.get_global_dims().x,
        box.y / E.get_global_dims().y
    );

    // Laser amplitude
    const float amp = omega0 * a0;

    // Grid tile parameters
    auto local_ntiles = E.get_local_ntiles();
    auto tile_dims = E.tile_dims;
    auto tile_start = E.get_local_tile_start();
    auto ystride = E.tile_ext_dims.x;

    // Loop over tiles
    #pragma omp parallel for collapse(2)
    for( unsigned ty = 0; ty < local_ntiles.y; ty++ ) {
        for( unsigned tx = 0; tx < local_ntiles.x; tx++ ) {
            float3 * const __restrict__ tile_E = E.tile_data( tx, ty );
            float3 * const __restrict__ tile_B = B.tile_data( tx, ty );

            const int ix0 = (tile_start.x + tx) * tile_dims.x;
            const int iy0 = (tile_start.y + ty) * tile_dims.y;

            for( int iy = 0; iy < static_cast<int>(tile_dims.y); iy++ ) {
                for( int ix = 0; ix < static_cast<int>(tile_dims.x); ix++ ) {
                    const float z = ( ix0 + ix ) * dx.x;
                    const float r = ( iy0 + iy ) * dx.y - axis;

                    const float lenv   = amp * laser::lon_env( *this, z ) * gauss_phase( omega0, W0, z - focus, r );

                    tile_E[ ix + iy * ystride ] = make_float3(
                        0,
                        +lenv * cos_pol,
                        +lenv * sin_pol
                    );
                    tile_B[ ix + iy * ystride ] = make_float3(
                        0,
                        -lenv * sin_pol,
                        +lenv * cos_pol
                    );
                }
            }
        }
    }

    // Set guard cell values
    E.copy_to_gc();
    B.copy_to_gc();

    return 0;
}


/**
 * @brief 
 * 
 */
int laser::gaussian::lon_x( grid::flat3<std::complex<float>> & fld, const float2 dk ) {

    laser::lon_x( 
        fld.x(), fld.y(), 
        fld.get_global_dims(), fld.get_local_dims(), fld.get_local_start(), 
        dk );
    return 0;
}