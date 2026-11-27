#include "laser.hpp"


#include "grid/fft.hpp"
#include "filter.hpp"

#include <iostream>
#include <cassert>
#include <cmath>

namespace em2ds {

/**
 * @brief Gets longitudinal laser envelope at a given position
 * 
 * @param env       Envelope parameters
 * @param z         Position
 * @return float    Laser envelope
 */
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
 */
inline void lon_x( 
    laser::complex_t * const __restrict__ fld_x, 
    const laser::complex_t * const __restrict__ fld_y, 
    const uint2 dims, const float2 dk ) {

    static_assert( ! grid::fft::kspace_transposed, "Normal k-space layout expected" );

    #pragma omp for
    for( unsigned idx = 0; idx < dims.y * dims.x; idx ++ ){
        const int ix = idx % dims.x;
        const int iy = idx / dims.x;

        const float2 kvec = grid::fft::k( make_int2( ix, iy ), dims, dk );
        const float kx = kvec.x;
        const float ky = kvec.y;

        fld_x[idx] = ( kx > 0 ) ? - ky * fld_y[idx] / kx : 0.f;
    }
}

namespace {

/**
 * @brief Get cell size from box size and global grid dimensions
 */
float2 cell_size( const grid::tiled_vec3<float> & E, const float2 box ) {
    return make_float2(
        box.x / E.get_grid_dims().x,
        box.y / E.get_grid_dims().y
    );
}

} // anonymous namespace

void laser::pulse::validate() const {

    if ( a0 <= 0 ) {
        util::fatal( "Invalid laser parameter (a0), must be > 0" );
    }    

    if ( omega0 <= 0 ) {
        util::fatal( "Invalid laser parameter (omega0), must be > 0" );
    }    

    // If fwhm > 0 it overrides the rise/flat/fall parameters, so these are
    // only checked when it is not set
    if ( fwhm <= 0 ) {
        if ( rise <= 0 ) {
            util::fatal( "Invalid laser parameter (rise), must be > 0" );
        }

        if ( flat < 0 ) {
            util::fatal( "Invalid laser parameter (flat), must be >= 0" );
        }

        if ( fall <= 0 ) {
            util::fatal( "Invalid laser parameter (fall), must be > 0" );
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

    grid::tiled_vec3<float> tmp_E( emf.E -> grid_ntiles, emf.E -> tile_dims, emf.E -> gc, emf.E -> grid_periodic );
    grid::tiled_vec3<float> tmp_B( emf.B -> grid_ntiles, emf.B -> tile_dims, emf.B -> gc, emf.B -> grid_periodic );

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
};

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
int laser::plane_wave::launch( grid::tiled_vec3<float>& E, grid::tiled_vec3<float>& B, float2 box ) const {

    validate();

    const float2 dx  = cell_size( E, box );
    const float2 pol = get_polarization();

    laser::envelope env = get_envelope();

    const float amp = omega0 * a0;
    const float k = omega0;

    // Grid tile parameters
    auto grid_ntiles = E.get_grid_ntiles();
    auto tile_dims = E.tile_dims;
    int ystride = E.tile_ext_dims.x;

    // Loop over tiles
    #pragma omp parallel for collapse(2)
    for( unsigned ty = 0; ty < grid_ntiles.y; ty++ ) {
        for( unsigned tx = 0; tx < grid_ntiles.x; tx++ ) {
            // Copy data to shared memory and block
            float3 * const __restrict__ tile_E = E.tile_data( tx, ty );
            float3 * const __restrict__ tile_B = B.tile_data( tx, ty );

            const int ix0 = tx * tile_dims.x;

            for( unsigned iy = 0; iy < tile_dims.y; iy++ ) {
                for( unsigned ix = 0; ix < tile_dims.x; ix++ ) {
                    const float z   = ( ix0 + ix ) * dx.x;

                    float lenv   = amp * lon_env( env, z ) * std::cos( k * z );

                    tile_E[ ix + iy * ystride ] = make_float3(
                        0,
                        +lenv * pol.x,
                        +lenv * pol.y
                    );

                    tile_B[ ix + iy * ystride ] = make_float3(
                        0,
                        -lenv * pol.y,
                        +lenv * pol.x
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
 * @note Invalid parameters cause the simulation to stop (util::fatal)
 */
void laser::gaussian::validate() const {
    
    laser::pulse::validate();

    if ( W0 <= 0 ) {
        util::fatal( "Invalid laser parameter (W0), must be > 0" );
    }
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
int laser::gaussian::launch(grid::tiled_vec3<float>& E, grid::tiled_vec3<float>& B, const float2 box ) const 
{

    validate();

    const float2 dx  = cell_size( E, box );
    const float2 pol = get_polarization();

    laser::envelope env = get_envelope();

    // Laser amplitude
    const float amp = omega0 * a0;

    // Grid tile parameters
    auto grid_ntiles = E.get_grid_ntiles();
    auto tile_dims = E.tile_dims;
    auto ystride = E.tile_ext_dims.x;

    // Loop over tiles
    #pragma omp parallel for collapse(2)
    for( unsigned ty = 0; ty < grid_ntiles.y; ty++ ) {
        for( unsigned tx = 0; tx < grid_ntiles.x; tx++ ) {
            float3 * const __restrict__ tile_E = E.tile_data( tx, ty );
            float3 * const __restrict__ tile_B = B.tile_data( tx, ty );

            const int ix0 = tx * tile_dims.x;
            const int iy0 = ty * tile_dims.y;

            for( int iy = 0; iy < static_cast<int>(tile_dims.y); iy++ ) {
                for( int ix = 0; ix < static_cast<int>(tile_dims.x); ix++ ) {
                    const float z = ( ix0 + ix ) * dx.x;
                    const float r = ( iy0 + iy ) * dx.y - axis;

                    const float lenv   = amp * lon_env( env, z ) * gauss_phase( omega0, W0, z - focus, r );

                    tile_E[ ix + iy * ystride ] = make_float3(
                        0,
                        +lenv * pol.x,
                        +lenv * pol.y
                    );
                    tile_B[ ix + iy * ystride ] = make_float3(
                        0,
                        -lenv * pol.y,
                        +lenv * pol.x
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
int laser::gaussian::lon_x( grid::flat3<laser::complex_t> & fld, const float2 dk ) const {

    em2ds::lon_x( 
        fld.x(), fld.y(), 
        fld.get_grid_dims(), 
        dk );
    return 0;
}

} // namespace em2ds
