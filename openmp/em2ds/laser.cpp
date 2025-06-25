#include "laser.h"

#include <iostream>
#include <cassert>

#include "fft.h"

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
inline float lon_env( Laser::Pulse & laser, float z ) {

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
    std::complex<float> * const __restrict__ data, 
    uint2 dims, float2 dk ) {
    
    std::complex<float> * const __restrict__ fld_x = & data[               0 ];
    std::complex<float> * const __restrict__ fld_y = & data[ dims.x * dims.y ];
    // std::complex<float> * const __restrict__ fld_z = & data[ 2 * dims.x * dims.y ];

    #pragma omp for
    for( unsigned idx = 0; idx < dims.y * dims.x; idx ++ ){
        int ix = idx % dims.x;
        int iy = idx / dims.x;

        const float ky = ((iy < int(dims.y)/2) ? iy : (iy - int(dims.y)) ) * dk.y;
        const float kx = ix * dk.x;

        fld_x[idx] = ( ix > 0 ) ? - ky * fld_y[idx] / kx : 0.f;
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
 * @brief Adds a new laser pulse onto an EMF object
 * 
 * @param emf   EMF object
 * @return      Returns 0 on success, -1 on error (invalid laser parameters)
 */
int Laser::Pulse::add( EMF & emf ) {

    vec3grid<float3> tmp_E( emf.E -> ntiles, emf.E-> nx, emf.E -> gc );
    vec3grid<float3> tmp_B( emf.B -> ntiles, emf.B-> nx, emf.B -> gc );

    // Get laser fields
    int ierr = launch( tmp_E, tmp_B, emf.box );

    // Add laser to simulation
    if ( ! ierr ) {

        // Add to k-space fields
        basic_grid3<std::complex<float>> fft_tmp( fft :: fdims( tmp_E.dims ) );
        
        fft::plan forward_E( tmp_E, fft_tmp );
        fft::plan forward_B( tmp_B, fft_tmp );

        fft::plan backward_E( fft_tmp, tmp_E );
        fft::plan backward_B( fft_tmp, tmp_B );

        const float2 dk = fft::dk( emf.box );

        Filter::Lowpass filter( make_float2( 0.5, 0.5 ) );

        // transform tmp_E and add to fEt
        forward_E.transform( tmp_E, fft_tmp );
        lon_x( fft_tmp, dk );
        filter.apply( fft_tmp );
        emf.fEt -> add( fft_tmp );

        backward_E.transform( fft_tmp, tmp_E );
        emf.E -> add( tmp_E );

        // transform tmp_B and add to fB 
        forward_B.transform( tmp_B, fft_tmp );
        lon_x( fft_tmp, dk );
        filter.apply( fft_tmp );
        emf.fB  -> add( fft_tmp );

        backward_B.transform( fft_tmp, tmp_B );
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
int Laser::PlaneWave::launch( vec3grid<float3>& E, vec3grid<float3>& B, float2 box ) {

    if ( validate() < 0 ) return -1;

    if (( cos_pol == 0 ) && ( sin_pol == 0 )) {
        cos_pol = std::cos( polarization );
        sin_pol = std::sin( polarization );
    }

    uint2 global_nx = E.dims;

    float2 dx = make_float2(
        box.x / global_nx.x,
        box.y / global_nx.y
    );

    // Grid tile parameters
    const auto ntiles     = E.ntiles;
    const auto tile_vol   = E.tile_vol;
    const auto nx         = E.nx;
    const auto offset     = E.offset;
    const int  ystride    = E.ext_nx.x; // ystride should be signed

    const float k = omega0;
    const float amp = omega0 * a0;

    // Loop over tiles
    #pragma omp parallel for
    for( unsigned tid = 0; tid < ntiles.y * ntiles.x; tid++ ) {

        const auto ty = tid / ntiles.x;
        const auto tx = tid % ntiles.x;

        const auto tile_idx = make_uint2( tx, ty );
        const auto tile_off = tid * tile_vol;

        // Copy data to shared memory and block
        float3 * const __restrict__ tile_E = & E.d_buffer[ tile_off + offset ];
        float3 * const __restrict__ tile_B = & B.d_buffer[ tile_off + offset ];

        const int ix0 = tile_idx.x * nx.x;

        for( unsigned iy = 0; iy < nx.y; iy++ ) {
            for( unsigned ix = 0; ix < nx.x; ix++ ) {
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

    E.copy_to_gc();
    B.copy_to_gc();

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

    if ( validate() < 0 ) return -1;

    if (( cos_pol == 0 ) && ( sin_pol == 0 )) {
        cos_pol = std::cos( polarization );
        sin_pol = std::sin( polarization );
    }

    uint2 global_nx = E.dims;

    float2 dx = make_float2(
        box.x / global_nx.x,
        box.y / global_nx.y
    );

    // Grid tile parameters
    const int2 ntiles   = make_int2( E.ntiles.x, E.ntiles.y );
    const int  tile_vol = E.tile_vol;
    const int2 nx       = make_int2( E.nx.x, E.nx.y );
    const int  offset   = E.offset;
    const int  ystride  = E.ext_nx.x;   // ystride must be signed

    const float amp = omega0 * a0;

    // Loop over tiles
    #pragma omp parallel for
    for( int tid = 0; tid < ntiles.y * ntiles.x ; tid++ ) {
        const int ty = tid / ntiles.x;
        const int tx = tid % ntiles.x;
        const int tile_off = tid * tile_vol;

        float3 * const __restrict__ tile_E = & E.d_buffer[ tile_off + offset ];
        float3 * const __restrict__ tile_B = & B.d_buffer[ tile_off + offset ];

        const int ix0 = tx * nx.x;
        const int iy0 = ty * nx.y;

        for( int iy = 0; iy < nx.y; iy++ ) {

            for( int ix = 0; ix < nx.x; ix++ ) {
                const float z   = ( ix0 + ix ) * dx.x;
                const float r   = ( iy0 + iy ) * dx.y       - axis;

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

    // Set guard cell values
    E.copy_to_gc();
    B.copy_to_gc();

    return 0;
}


/**
 * @brief 
 * 
 */
int Laser::Gaussian::lon_x( basic_grid3<std::complex<float>> & fld, const float2 dk ) {

    laser::lon_x( fld.d_buffer, fld.dims, dk );
    return 0;
}