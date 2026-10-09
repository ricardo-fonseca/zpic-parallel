#include "laser.hpp"


#include "grid/fft.cuh"
#include "filter.cuh"
#include "parallel/mpi.hpp"
#include "util/complex.hpp"

#include <iostream>
#include <cassert>

namespace laser {
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
__device__
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
        auto fft_tmp = dft_forward.kspace_grid3();
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

namespace laser {
namespace kernel {
__global__
void plane_wave(
    laser::plane_wave laser,
    grid::tiled_vec3_view<float> E_fld, grid::tiled_vec3_view<float> B_fld, 
    uint2 const local_tile_start, float2 const dx )
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };

    float3 * const __restrict__ E = E_fld.tile_data( tile_idx );
    float3 * const __restrict__ B = B_fld.tile_data( tile_idx );

    const int ix0 = ( local_tile_start.x + tile_idx.x ) * E_fld.tile_dims.x;
    const float k = laser.omega0;
    const float amp = laser.omega0 * laser.a0;
    const int ystride = E_fld.tile_ystride();

    auto nx = E_fld.tile_dims;

    for( unsigned idx = gpu::block::thread_rank(); idx < nx.y * nx.x; idx += gpu::block::num_threads() ) {
        const auto ix = idx % nx.x;
        const auto iy = idx / nx.x; 

        const float z   = ( ix0 + ix ) * dx.x;
        const float z_2 = ( ix0 + ix + 0.5 ) * dx.x;

        float lenv   = amp * lon_env( laser, z );
        float lenv_2 = amp * lon_env( laser, z_2 );

        E[ ix + iy * ystride ] = make_float3(
            0,
            +lenv * cos( k * z ) * laser.cos_pol,
            +lenv * cos( k * z ) * laser.sin_pol
        );

        B[ ix + iy * ystride ] = make_float3(
            0,
            -lenv_2 * cos( k * z_2 ) * laser.sin_pol,
            +lenv_2 * cos( k * z_2 ) * laser.cos_pol
        );
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

    dim3 block( 64 );
    dim3 grid( E.get_local_ntiles().x, E.get_local_ntiles().y );

    kernel::plane_wave<<<grid, block>>> (
        * this, E.view(), B.view(),
        E.get_local_tile_start(), dx
    );

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

namespace laser {
namespace kernel {

#if 0
/**
 * @brief Returns local phase for a gaussian beamn
 * 
 * @param omega0    Beam frequency
 * @param W0        Beam waist
 * @param z         Position along focal line (focal plane at z = 0)
 * @param r         Position transverse to focal line (focal line at r = 0)
 * @return          Local field value
 */
__device__
inline float gauss_phase( const float omega0, const float W0, const float z, const float r ) {
    const float z0   = omega0 * ( W0 * W0 ) / 2;
    const float rho2 = r*r;
    const float curv = 0.5f * rho2 * z / (z0*z0 + z*z);
    const float rWl2 = (z0*z0)/(z0*z0 + z*z);
    const float gouy_shift = atan2( z, z0 );

    return sqrt( sqrt(rWl2) ) * 
        exp( - rho2 * rWl2/( W0 * W0 ) ) * 
        cos( omega0*( z + curv ) - gouy_shift );
}
#endif

__global__
void gaussian( 
    laser::gaussian beam, 
    grid::tiled_vec3_view<float> E_fld, grid::tiled_vec3_view<float> B_fld, 
    uint2 const local_tile_start, float2 const dx )
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };

    float3 * const __restrict__ E = E_fld.tile_data( tile_idx );
    float3 * const __restrict__ B = B_fld.tile_data( tile_idx );

    const float omega0 = beam.omega0;
    const float W0 = beam.W0;

    // Gaussian beam phase (lambda version)
    auto gauss_phase = [ omega0, W0 ] ( const float z, const float r ) {
        const float z0   = omega0 * ( W0 * W0 ) / 2;
        const float rho2 = r*r;
        const float curv = 0.5f * rho2 * z / (z0*z0 + z*z);
        const float rWl2 = (z0*z0)/(z0*z0 + z*z);
        const float gouy_shift = atan2( z, z0 );

        return sqrt( sqrt(rWl2) ) * 
            exp( - rho2 * rWl2/( W0 * W0 ) ) * 
            cos( omega0*( z + curv ) - gouy_shift );
    };

    // Beam amplitude
    const float amp = beam.omega0 * beam.a0;

    const int ix0 = ( local_tile_start.x + tile_idx.x ) * E_fld.tile_dims.x;
    const int iy0 = ( local_tile_start.y + tile_idx.y ) * E_fld.tile_dims.y;
    const int ystride = E_fld.tile_ystride();

    auto nx = E_fld.tile_dims;

    for( unsigned idx = gpu::block::thread_rank(); idx < nx.y * nx.x; idx += gpu::block::num_threads() ) {
        const auto ix = idx % nx.x;
        const auto iy = idx / nx.x; 

        const float z   = ( ix0 + ix ) * dx.x;
        const float r   = (iy0 + iy ) * dx.y - beam.axis;

        const float fld    = amp * lon_env( beam, z ) * 
                             gauss_phase( z - beam.focus, r );

        E[ ix + iy * ystride ] = make_float3(
            0,
            + fld * beam.cos_pol,
            + fld * beam.sin_pol
        );
        B[ ix + iy * ystride ] = make_float3(
            0,
            - fld * beam.sin_pol,
            + fld * beam.cos_pol
        );
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

    dim3 block( 64 );
    dim3 grid( E.get_local_ntiles().x, E.get_local_ntiles().y );

    kernel::gaussian<<<grid, block>>> (
        * this, E.view(), B.view(),
        E.get_local_tile_start(), dx
    );

    E.copy_to_gc();
    B.copy_to_gc();

    return 0;
}

namespace laser {
namespace kernel {
/**
 * @brief Ensures div F = 0 by modifying the x component of F
 * 
 * @param view      View of k-space laser field data
 * @param dk        k-space cell size
 */
__global__
inline void lon_x( 
    grid::flat3_view<complex_t> view, const float2 dk ) {

    static_assert( ! grid::fft::kspace_transposed, "Normal k-space layout expected" );

    const int iy = blockIdx.x;
    const int kiy = view.local_start.y + iy;
    const float ky =  ((2 * kiy < view.global_dims.y ) ? kiy : (kiy - int( view.global_dims.y)) ) * dk.y;


    util::complex64 * __restrict__ fld_x = reinterpret_cast<util::complex64 *>(view.x_buffer);
    util::complex64 * __restrict__ fld_y = reinterpret_cast<util::complex64 *>(view.y_buffer);
    const int ystride = view.local_dims.x;

    for( auto ix = gpu::block::thread_rank(); ix < view.local_dims.x; ix += gpu::block::num_threads() ) {
        auto idx = iy * ystride + ix;

        const float kx = ( view.local_start.x + ix ) * dk.x;
        fld_x[idx] = ( ix > 0 ) ? - ky * fld_y[idx] / kx : 0.f;
    }
}
}
}

/**
* @brief Sets longitudinal component
* @note Enforces div.fld = 0 by setting x component
* 
* @param fld   Fourier transform of E or B field
* @param dk    Cell size in k space
* @return int  
*/
int laser::gaussian::lon_x( grid::flat3<complex_t> & fld, const float2 dk ) {

    laser::kernel::lon_x <<< fld.get_local_dims().y, 64 >>> ( 
        fld.view(), dk );

    return 0;
}