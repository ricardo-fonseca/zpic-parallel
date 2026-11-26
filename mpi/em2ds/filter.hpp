#pragma once

#include <complex>
#include <memory>
#include <algorithm>

#include "grid/flat.hpp"
#include "grid/flat3.hpp"
#include "grid/fft.hpp"

namespace em2ds {

namespace filter {

using complex_t = util::complex<float>;

class digital {
    public:
    virtual std::unique_ptr<digital> clone() const = 0;
    virtual void apply( grid::flat<complex_t> & fld )  = 0;
    virtual void apply( grid::flat3<complex_t> & fld ) = 0;
    virtual ~digital() = default;
};

class none : public digital {
    public:
    std::unique_ptr<digital> clone() const override {
        return std::make_unique<none>( *this );
    }

    void apply( grid::flat<complex_t> & ) override { /* do nothing */ };
    void apply( grid::flat3<complex_t> & ) override { /* do nothing */ };
};

namespace kernel {

/**
 * @brief Range of local ky indices to be zeroed by the lowpass filter on a
 *        k-space row
 * 
 * @note The FFTW k-space layout is transposed (x = ky, y = kx), split along kx
 *       across ranks and holds the full ky extent locally, so each local row
 *       corresponds to a single kx value. If kx is above the kx cutoff the
 *       whole row is zeroed; otherwise, since ky is stored in FFT order (see
 *       grid::fft::k()), the modes with |ky| above the ky cutoff form a single
 *       contiguous range in the middle of the row. The mask depends only on
 *       kx and |ky|, so Hermitian symmetry of the r2c half-spectrum is
 *       preserved.
 * 
 * @note Cutoff positions are the same as in the GPU (cuFFTMp) version.
 * 
 * @param global_dims   Global k-space grid dimensions (x = ky, y = kx)
 * @param local_dims    Local k-space grid dimensions
 * @param local_start   Position of local grid on global grid
 * @param iy            Local row (kx) index
 * @param cutoff        Cutoff as a fraction of the Nyquist mode along x and y
 * @param i0            (out) First local ky index to zero
 * @param i1            (out) Last local ky index to zero + 1 (i1 == i0 zeroes
 *                      nothing)
 */
inline void lowpass_range( 
    const uint2 global_dims, const uint2 local_dims, const uint2 local_start,
    const int iy, const float2 cutoff, int & i0, int & i1 )
{
    static_assert( grid::fft::kspace_transposed, "Transposed k-space layout expected" );

    const int nky = global_dims.x;
    const int nkx = global_dims.y;      // nx/2 + 1 (r2c half spectrum)
    const int nx  = local_dims.x;

    // Cutoff positions in global grid
    const int kcx = cutoff.x * ( nkx - 1 );
    const int kcy = cutoff.y * ( nky / 2 );

    // Global kx mode
    const int kx = int( local_start.y ) + iy;

    // Row above the kx cutoff: zero everything
    if ( kx > kcx ) {
        i0 = 0;
        i1 = nx;
        return;
    }

    // Otherwise zero |ky| > kcy, i.e. global ky indices [ kcy + 1, nky - kcy - 1 ],
    // converted to local indices
    i0 = std::clamp( kcy + 1   - int( local_start.x ), 0, nx );
    i1 = std::clamp( nky - kcy - int( local_start.x ), i0, nx );
}

/**
 * @brief Lowpass filter for scalar k-space grids
 * 
 * @param fld       k-space grid
 * @param cutoff    Cutoff as a fraction of the Nyquist mode along x and y
 */
inline void lowpass( grid::flat<complex_t> & fld, float2 const cutoff )
{
    const auto global_dims = fld.get_global_dims();
    const auto local_dims  = fld.get_local_dims();
    const auto local_start = fld.get_local_start();

    complex_t * __restrict__ data = fld.data();
    const int ystride = local_dims.x;
    const int nrows   = local_dims.y;

    // One row per iteration (one block per row in the GPU version)
    #pragma omp parallel for schedule(static)
    for( int iy = 0; iy < nrows; iy++ ) {
        int i0, i1;
        lowpass_range( global_dims, local_dims, local_start, iy, cutoff, i0, i1 );

        complex_t * const row = &data[ iy * ystride ];
        std::fill( row + i0, row + i1, complex_t(0) );
    }
}

/**
 * @brief Lowpass filter for vector (3 component) k-space grids
 * 
 * @param fld       k-space grid
 * @param cutoff    Cutoff as a fraction of the Nyquist mode along x and y
 */
inline void lowpass( grid::flat3<complex_t> & fld, float2 const cutoff )
{
    const auto global_dims = fld.get_global_dims();
    const auto local_dims  = fld.get_local_dims();
    const auto local_start = fld.get_local_start();

    complex_t * __restrict__ x_data = fld.x();
    complex_t * __restrict__ y_data = fld.y();
    complex_t * __restrict__ z_data = fld.z();

    const int ystride = local_dims.x;
    const int nrows   = local_dims.y;

    // One row per iteration (one block per row in the GPU version)
    #pragma omp parallel for schedule(static)
    for( int iy = 0; iy < nrows; iy++ ) {
        int i0, i1;
        lowpass_range( global_dims, local_dims, local_start, iy, cutoff, i0, i1 );

        const int off = iy * ystride;
        std::fill( x_data + off + i0, x_data + off + i1, complex_t(0) );
        std::fill( y_data + off + i0, y_data + off + i1, complex_t(0) );
        std::fill( z_data + off + i0, z_data + off + i1, complex_t(0) );
    }
}

} // namespace kernel

/**
 * @brief Sharp (brick-wall) spectral lowpass filter
 * 
 * Zeroes all modes with kx > cutoff.x * kx_Nyquist or |ky| > cutoff.y * ky_Nyquist
 */
class lowpass : public digital {
    protected:

    /// @brief Cutoff as a fraction of the Nyquist mode, in [0,1] along each direction
    const float2 cutoff;
    
    public:

    /**
     * @brief Construct a new lowpass filter
     * 
     * @param cutoff    Cutoff as a fraction of the Nyquist mode along x and y,
     *                  must be in [0,1] (1 keeps all modes)
     */
    lowpass( const float2 cutoff ) : cutoff( cutoff ) {
        if ( cutoff.x < 0.f || cutoff.x > 1.f || cutoff.y < 0.f || cutoff.y > 1.f ) {
            mpi::fatal( "filter::lowpass: cutoff values must be in the range [0,1]" );
        }
    };

    std::unique_ptr<digital> clone() const override {
        return std::make_unique<lowpass>( *this );
    }

    void apply( grid::flat<complex_t> & fld ) override {
        kernel::lowpass( fld, cutoff );
    }
    
    void apply( grid::flat3<complex_t> & fld ) override {
        kernel::lowpass( fld, cutoff );
    }
};

}

} // namespace em2ds
