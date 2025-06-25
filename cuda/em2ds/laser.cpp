#include "laser.h"

#include <iostream>
#include <cassert>

#include "fft.h"

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
 * @brief Ensures div F = 0 by modifying the x component of F
 * 
 * @param data      FFT of field data organized as a contiguous grid
 * @param dims      Dimensions of k-space grid
 * @param dk        k-space cell size
 */
__global__ 
void lon_x( 
    fft::complex64 * const __restrict__ data, 
    uint2 dims, float2 dk ) {
    
    const int iy = blockIdx.x;
    const float ky = ((iy < dims.y/2) ? iy : (iy - int(dims.y)) ) * dk.y;

    fft::complex64 * const __restrict__ fld_x = & data[               0 ];
    fft::complex64 * const __restrict__ fld_y = & data[ dims.x * dims.y ];
    // fft::complex64 * const __restrict__ fld_z = & data[ 2 * dims.x * dims.y ];

    const int stride = dims.x;
    for( auto ix = block_thread_rank(); ix < dims.x; ix += block_num_threads() ) {
        auto idx = iy * stride + ix;

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
        fft::plan dft_forward( tmp_E.dims, fft::r2c_v3 );
        basic_grid3<std::complex<float>> fft_tmp( fft::fdims( tmp_E.dims ) );
        const float2 dk = fft::dk( emf.box );

        Filter::Lowpass filter( make_float2( 0.5, 0.5 ) );

        // transform tmp_E and add to fEt
        dft_forward.transform( tmp_E, fft_tmp );
        lon_x( fft_tmp, dk );
        filter.apply( fft_tmp );
        emf.fEt -> add( fft_tmp );

        emf.fft_backward -> transform( fft_tmp, tmp_E );
        emf.E -> add( tmp_E );

        // transform tmp_B and add to fB 
        dft_forward.transform( tmp_B, fft_tmp );
        lon_x( fft_tmp, dk );
        filter.apply( fft_tmp );
        emf.fB  -> add( fft_tmp );

        emf.fft_backward -> transform( fft_tmp, tmp_B );
        emf.B -> add( tmp_B );
    }
    return ierr;
};


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

    float2 dx = make_float2(
        box.x / E.dims.x,
        box.y / E.dims.y
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
        const float r   = (iy0 + iy ) * dx.y - beam.axis;

        const float fld    = amp * lon_env( beam, z ) * 
                             gauss_phase( beam.omega0, beam.W0, z - beam.focus, r );

        tile_E[ ix + iy * ystride ] = make_float3(
            0,
            + fld * beam.cos_pol,
            + fld * beam.sin_pol
        );
        tile_B[ ix + iy * ystride ] = make_float3(
            0,
            - fld * beam.sin_pol,
            + fld * beam.cos_pol
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

    float2 dx = make_float2(
        box.x / E.dims.x,
        box.y / E.dims.y
    );

    dim3 block( 64 );
    dim3 grid( E.ntiles.x, E.ntiles.y );

    kernel::gaussian<<<grid, block>>> (
        *this, E.d_buffer, B.d_buffer,
        E.ntiles, E.nx, E.ext_nx, E.offset,
        dx
    );

    return 0;
}

/**
 * @brief 
 * 
 */
int Laser::Gaussian::lon_x( basic_grid3<std::complex<float>> & fld, const float2 dk ) {

    kernel::lon_x <<< fld.dims.y, 64 >>> ( 
        reinterpret_cast< fft::complex64 * > (fld.d_buffer),
        fld.dims, dk );

    return 0;
}
