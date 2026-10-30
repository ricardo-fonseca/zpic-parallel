#include "core/gpu.cuh"
#include "grid/tiled_vec3.cuh"
#include "laser.hpp"


#include "grid/fft.cuh"
#include "filter.cuh"
#include "parallel/mpi.hpp"
#include "util/complex.hpp"

#include <iostream>
#include <cassert>
#include <cmath>

namespace laser {
namespace kernel {

// Note: Plain-data copies of the laser parameters are passed to the kernels
// because objects of classes with virtual functions may not be passed as
// arguments to __global__ functions.

/**
 * @brief Plane wave kernel parameters
 */
struct plane_wave_params {
    /// @brief Longitudinal envelope
    envelope env;
    /// @brief Wavenumber (= omega0, normalized units)
    float k;
    /// @brief Field amplitude (omega0 * a0)
    float amp;
    /// @brief Cosine of the polarization angle
    float cos_pol;
    /// @brief Sine of the polarization angle
    float sin_pol;
};

/**
 * @brief Gaussian beam kernel parameters
 */
struct gaussian_params {
    /// @brief Longitudinal envelope
    envelope env;
    /// @brief Laser frequency
    float omega0;
    /// @brief Beam waist
    float W0;
    /// @brief Focal plane position
    float focus;
    /// @brief Beam axis position
    float axis;
    /// @brief Field amplitude (omega0 * a0)
    float amp;
    /// @brief Cosine of the polarization angle
    float cos_pol;
    /// @brief Sine of the polarization angle
    float sin_pol;
};

/**
 * @brief Gets longitudinal laser envelope at a given position
 * 
 * @param env       Envelope parameters
 * @param z         Position
 * @return float    Laser envelope
 */
__device__
inline float lon_env( const laser::envelope & env, const float z ) {

    if ( z > env.start ) {
        // Ahead of laser
        return 0.0f;
    } else if ( z > env.start - env.rise ) {
        // Laser rise
        const float csi = z - env.start;
        const float e = sinf( static_cast<float>(M_PI_2) * csi / env.rise );
        return e*e;
    } else if ( z > env.start - (env.rise + env.flat) ) {
        // Flat-top
        return 1.0f;
    } else if ( z > env.start - (env.rise + env.flat + env.fall) ) {
        // Laser fall
        const float csi = z - (env.start - env.rise - env.flat - env.fall);
        const float e = sinf( static_cast<float>(M_PI_2) * csi / env.fall );
        return e*e;
    }

    // Before laser
    return 0.0f;
}

/**
 * @brief Ensures div F = 0 by modifying the x component of F
 * 
 * @note Solves kx Fx + ky Fy = 0 for Fx. The kx = 0 column is set to 0.
 *       Must be called with grid( local_dims.y ).
 * 
 * @param view      View of k-space laser field data
 * @param dk        k-space cell size
 */
__global__
void lon_x( 
    grid::flat3_view<complex_t> view, const float2 dk ) {

    static_assert( ! grid::fft::kspace_transposed, "Normal k-space layout expected" );

    // Row is block ID
    const int iy  = blockIdx.x;
    const int kiy = view.local_start.y + iy;
    const float ky = ( ( 2 * kiy < int( view.global_dims.y ) ) ? kiy : ( kiy - int( view.global_dims.y ) ) ) * dk.y;

    complex_t       * const __restrict__ fld_x = view.x_buffer;
    complex_t const * const __restrict__ fld_y = view.y_buffer;
    const int ystride = view.local_dims.x;

    for( auto ix = gpu::block::thread_rank(); ix < view.local_dims.x; ix += gpu::block::num_threads() ) {
        const auto idx = iy * ystride + ix;

        // r2c half spectrum, kx >= 0 so no wrap is needed. Use the global
        // index, otherwise nodes with local_start.x > 0 would zero their
        // first column
        const int   kix = view.local_start.x + ix;
        const float kx  = kix * dk.x;
        fld_x[idx] = ( kix > 0 ) ? - ky * fld_y[idx] / kx : 0.f;
    }
}

} // namespace kernel
} // namespace laser


namespace {

/**
 * @brief Get cell size from box size and global grid dimensions
 */
float2 cell_size( const grid::tiled_vec3<float> & E, const float2 box ) {
    return make_float2(
        box.x / E.get_global_dims().x,
        box.y / E.get_global_dims().y
    );
}

} // anonymous namespace

/**
 * @brief Validates laser parameters
 * 
 * @note Invalid parameters cause the simulation to stop (mpi::fatal)
 */
void laser::pulse::validate() const {

    if ( a0 <= 0 ) {
        mpi::fatal( "Invalid laser parameter (a0), must be > 0" );
    }    

    if ( omega0 <= 0 ) {
        mpi::fatal( "Invalid laser parameter (omega0), must be > 0" );
    }    

    // If fwhm > 0 it overrides the rise/flat/fall parameters, so these are
    // only checked when it is not set
    if ( fwhm <= 0 ) {
        if ( rise <= 0 ) {
            mpi::fatal( "Invalid laser parameter (rise), must be > 0" );
        }

        if ( flat < 0 ) {
            mpi::fatal( "Invalid laser parameter (flat), must be >= 0" );
        }

        if ( fall <= 0 ) {
            mpi::fatal( "Invalid laser parameter (fall), must be > 0" );
        }
    }
}

/**
 * @brief Get the longitudinal envelope
 * 
 * @note If fwhm > 0 this returns rise = fall = fwhm and flat = 0, otherwise
 *       the rise, flat and fall parameters are used
 * 
 * @return laser::envelope 
 */
laser::envelope laser::pulse::get_envelope() const {
    if ( fwhm > 0 ) {
        return laser::envelope{ start, fwhm, 0.0f, fwhm };
    }
    return laser::envelope{ start, rise, flat, fall };
}

/**
 * @brief Get the polarization direction
 * 
 * @note Uses cos_pol / sin_pol if either is set, otherwise these are
 *       calculated from the polarization angle
 * 
 * @return float2   { cos(polarization), sin(polarization) }
 */
float2 laser::pulse::get_polarization() const {
    if (( cos_pol == 0 ) && ( sin_pol == 0 )) {
        return make_float2( std::cos( polarization ), std::sin( polarization ) );
    }
    return make_float2( cos_pol, sin_pol );
}

/**
 * @brief Adds a new laser pulse onto an EMF object
 * 
 * @param emf   EMF object
 * @return      Returns 0 on success
 */
int laser::pulse::add( emf & emf ) const {

    grid::tiled_vec3<float> tmp_E( emf.E -> global_ntiles, emf.E -> tile_dims, emf.E -> gc, emf.E -> part );
    grid::tiled_vec3<float> tmp_B( emf.B -> global_ntiles, emf.B -> tile_dims, emf.B -> gc, emf.B -> part );

    // Get laser fields
    int ierr = launch( tmp_E, tmp_B, emf.box );

    // Add laser to simulation
    if ( ! ierr ) {

        // Add to k-space fields
        // Note: lon_x() and the k-space low-pass filter commute, and the
        // k-space data must be added to fEt / fB before the backward
        // transform, since c2r transforms may overwrite their input
        grid::fft::r2c_plan dft_forward( tmp_E );
        auto fft_tmp = dft_forward.kspace_grid3();
        const float2 dk = grid::fft::dk( emf.box );

        filter::lowpass lowpass( filter );

        // transform tmp_E and add to fEt
        dft_forward.transform( fft_tmp, tmp_E );
        lon_x( fft_tmp, dk );
        lowpass.apply( fft_tmp );
        emf.fEt -> add( fft_tmp );

        // This will also set the real space guard cells
        emf.fft_backward -> transform( tmp_E, fft_tmp  );
        emf.E -> add( tmp_E );

        // transform tmp_B and add to fB 
        dft_forward.transform( fft_tmp, tmp_B );
        lon_x( fft_tmp, dk );
        lowpass.apply( fft_tmp );
        emf.fB  -> add( fft_tmp );

        emf.fft_backward -> transform( tmp_B, fft_tmp );
        emf.B -> add( tmp_B );
    }

    return ierr;
}

namespace laser {
namespace kernel {

/**
 * @brief Sets plane wave laser fields
 * 
 * @note All field components are defined at the lower left corner of the
 *       cell (no staggering). Only the interior cells are set.
 * 
 * @param p                 Plane wave parameters
 * @param E_fld             E field grid
 * @param B_fld             B field grid
 * @param local_tile_start  Tile start position on global tile grid
 * @param dx                Cell size
 */
__global__
void plane_wave(
    plane_wave_params const p,
    grid::tiled_vec3_view<float> E_fld, grid::tiled_vec3_view<float> B_fld, 
    uint2 const local_tile_start, float2 const dx )
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };

    float3 * const __restrict__ E = E_fld.tile_data( tile_idx );
    float3 * const __restrict__ B = B_fld.tile_data( tile_idx );

    const int ix0 = ( local_tile_start.x + tile_idx.x ) * E_fld.tile_dims.x;
    const int ystride = E_fld.tile_ystride();

    const auto nx = E_fld.tile_dims;

    for( unsigned idx = gpu::block::thread_rank(); idx < nx.y * nx.x; idx += gpu::block::num_threads() ) {
        const int ix = idx % nx.x;
        const int iy = idx / nx.x; 

        const float z   = ( ix0 + ix ) * dx.x;

        const float fld   = p.amp * lon_env( p.env, z   ) * cosf( p.k * z   );

        E[ ix + iy * ystride ] = make_float3(
            0,
            +fld * p.cos_pol,
            +fld * p.sin_pol
        );

        B[ ix + iy * ystride ] = make_float3(
            0,
            -fld * p.sin_pol,
            +fld * p.cos_pol
        );
    }
}

} // namespace kernel
} // namespace laser

/**
 * @brief Launches a plane wave
 * 
 * The E and B tiled grids have the complete laser field. Guard cells are
 * not set, they are not used by the forward FFT.
 * 
 * @param E     Electric field
 * @param B     Magnetic field
 * @param box   Box size
 * @return      Returns 0 on success
 */
int laser::plane_wave::launch( grid::tiled_vec3<float>& E, grid::tiled_vec3<float>& B, const float2 box ) const {

    validate();

    const float2 dx  = cell_size( E, box );
    const float2 pol = get_polarization();

    const kernel::plane_wave_params p {
        get_envelope(), omega0, omega0 * a0, pol.x, pol.y
    };

    dim3 block( 64 );
    dim3 grid( E.get_local_ntiles().x, E.get_local_ntiles().y );

    kernel::plane_wave <<< grid, block >>> (
        p, E.view(), B.view(), E.get_local_tile_start(), dx
    );

    return 0;
}

/**
 * @brief Validate Gaussian laser parameters
 * 
 * @note Invalid parameters cause the simulation to stop (mpi::fatal)
 */
void laser::gaussian::validate() const {
    
    laser::pulse::validate();

    if ( W0 <= 0 ) {
        mpi::fatal( "Invalid laser parameter (W0), must be > 0" );
    }
}

namespace laser {
namespace kernel {

/**
 * @brief Gaussian beam transverse profile and phase
 * 
 * @note The amplitude factor (W0/W)^(1/2) corresponds to 2D slab geometry,
 *       but the Gouy phase atan(z/z0) is the 3D value (the 2D value is
 *       0.5 * atan(z/z0)). This is kept as in the original code.
 * 
 * @param p     Gaussian beam parameters
 * @param z     Distance from focal plane
 * @param r     Distance from beam axis
 * @return      Field profile (normalized)
 */
__device__
inline float gauss_phase( const gaussian_params & p, const float z, const float r ) {
    const float z0   = p.omega0 * ( p.W0 * p.W0 ) / 2;
    const float rho2 = r*r;
    const float curv = 0.5f * rho2 * z / ( z0*z0 + z*z );
    const float rWl2 = ( z0*z0 ) / ( z0*z0 + z*z );
    const float gouy_shift = 0.5f * atan2f( z, z0 );

    return sqrtf( sqrtf( rWl2 ) ) * 
        expf( - rho2 * rWl2 / ( p.W0 * p.W0 ) ) * 
        cosf( p.omega0 * ( z + curv ) - gouy_shift );
}

/**
 * @brief Sets Gaussian beam laser fields (transverse components)
 * 
 * @note All field components are defined at the lower left corner of the
 *       cell (no staggering). Only the interior cells are set.
 * 
 * @param p                 Gaussian beam parameters
 * @param E_fld             E field grid
 * @param B_fld             B field grid
 * @param local_tile_start  Tile start position on global tile grid
 * @param dx                Cell size
 */
__global__
void gaussian( 
    gaussian_params const p, 
    grid::tiled_vec3_view<float> E_fld, grid::tiled_vec3_view<float> B_fld, 
    uint2 const local_tile_start, float2 const dx )
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };

    float3 * const __restrict__ E = E_fld.tile_data( tile_idx );
    float3 * const __restrict__ B = B_fld.tile_data( tile_idx );

    const int ix0 = ( local_tile_start.x + tile_idx.x ) * E_fld.tile_dims.x;
    const int iy0 = ( local_tile_start.y + tile_idx.y ) * E_fld.tile_dims.y;
    const int ystride = E_fld.tile_ystride();

    const auto nx = E_fld.tile_dims;

    for( unsigned idx = gpu::block::thread_rank(); idx < nx.y * nx.x; idx += gpu::block::num_threads() ) {
        const int ix = idx % nx.x;
        const int iy = idx / nx.x; 

        const float z   = ( ix0 + ix ) * dx.x;
        const float r   = ( iy0 + iy ) * dx.y - p.axis;

        const float fld = p.amp * lon_env( p.env, z ) * gauss_phase( p, z - p.focus, r );

        E[ ix + iy * ystride ] = make_float3(
            0,
            + fld * p.cos_pol,
            + fld * p.sin_pol
        );

        B[ ix + iy * ystride ] = make_float3(
            0,
            - fld * p.sin_pol,
            + fld * p.cos_pol
        );
    }
}

} // namespace kernel
} // namespace laser

/**
 * @brief Launches a Gaussian pulse
 * 
 * Only the transverse field components are set, the longitudinal components
 * are set in k-space by lon_x() (called from pulse::add()). Guard cells are
 * not set, they are not used by the forward FFT.
 * 
 * @param E     Electric field
 * @param B     Magnetic field
 * @param box   Box size
 * @return      Returns 0 on success
 */
int laser::gaussian::launch( grid::tiled_vec3<float>& E, grid::tiled_vec3<float>& B, const float2 box ) const {

    validate();

    const float2 dx  = cell_size( E, box );
    const float2 pol = get_polarization();

    const kernel::gaussian_params p {
        get_envelope(), omega0, W0, focus, axis, omega0 * a0, pol.x, pol.y
    };

    dim3 block( 64 );
    dim3 grid( E.get_local_ntiles().x, E.get_local_ntiles().y );

    kernel::gaussian <<< grid, block >>> (
        p, E.view(), B.view(), E.get_local_tile_start(), dx
    );

    return 0;
}

/**
 * @brief Sets longitudinal component
 * @note Enforces div.fld = 0 by setting x component
 * 
 * @param fld   Fourier transform of E or B field
 * @param dk    Cell size in k space
 * @return int  Returns 0 on success
 */
int laser::gaussian::lon_x( grid::flat3<complex_t> & fld, const float2 dk ) const {

    laser::kernel::lon_x <<< fld.get_local_dims().y, 64 >>> ( 
        fld.view(), dk );

    return 0;
}
