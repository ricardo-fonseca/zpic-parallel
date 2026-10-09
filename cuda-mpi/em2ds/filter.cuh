#pragma once

#include "grid/flat.cuh"
#include "grid/flat3.cuh"
#include "grid/fft.cuh"

#include "core/gpu.cuh"
#include "util/complex.hpp"

namespace filter {

using complex_t = util::complex64;
    
class digital {
    public:
    virtual digital * clone() const = 0;
    virtual void apply( grid::flat<complex_t> & fld )  = 0;
    virtual void apply( grid::flat3<complex_t> & fld ) = 0;
    virtual ~digital() = default;
};

class none : public digital {
    public:
    none * clone() const override { return new none(); };
    void apply( grid::flat<complex_t> & ) override { /* do nothing */ };
    void apply( grid::flat3<complex_t> & ) override { /* do nothing */ };
};

namespace kernel {

/**
 * @brief First local kx index to be zeroed by the lowpass filter on a k-space row
 * 
 * @note k-space is split along kx across ranks (cuFFTMp slab layout) and holds
 *       the full ky extent locally. The mask depends only on |ky| and kx, so
 *       Hermitian symmetry of the r2c half-spectrum is preserved.
 * 
 * @tparam View     flat_view / flat3_view type
 * @param view      Grid view (uses local_start, local_dims and global_dims)
 * @param iy        Local row (ky) index
 * @param cutoff    Cutoff as a fraction of the Nyquist mode along x and y
 * @return          Local kx index from which all modes are zeroed (0 zeroes
 *                  the whole row, >= local_dims.x zeroes nothing)
 */
template< class View >
__device__ inline int lowpass_ix0( const View & view, const int iy, const float2 cutoff )
{
    static_assert( ! grid::fft::kspace_transposed, "Normal k-space layout expected" );

    const int nkx = view.global_dims.x;     // nx/2 + 1 (r2c half spectrum)
    const int nky = view.global_dims.y;

    // Global ky mode (FFT order, see grid::fft::k())
    const int kiy = int( view.local_start.y ) + iy;
    const int ky  = abs( ( 2 * kiy < nky ) ? kiy : ( kiy - nky ) );

    // Cutoff positions in global grid
    const int kcx = cutoff.x * ( nkx - 1 );
    const int kcy = cutoff.y * ( nky / 2 );

    // Row above the ky cutoff: zero everything
    if ( ky > kcy ) return 0;

    // Otherwise zero kx > kcx, converted to a local index
    const int ix0 = kcx + 1 - int( view.local_start.x );
    return ( ix0 > 0 ) ? ix0 : 0;
}

/**
 * @brief Lowpass filter kernel for scalar k-space grids
 * 
 * @note Launch with one block per local ky row
 */
__global__
inline void lowpass( grid::flat_view<complex_t> view, float2 const cutoff )
{
    const int iy  = blockIdx.x;
    const int nx  = view.local_dims.x;
    const int ix0 = lowpass_ix0( view, iy, cutoff );

    complex_t * __restrict__ data = reinterpret_cast<complex_t *>( view.d_buffer );
    const int ystride = nx;

    for( int ix = ix0 + int( gpu::block::thread_rank() ); ix < nx; ix += int( gpu::block::num_threads() ) ) {
        data[ iy * ystride + ix ] = 0;
    }
}

/**
 * @brief Lowpass filter kernel for vector (3 component) k-space grids
 * 
 * @note Launch with one block per local ky row
 */
__global__
inline void lowpass( grid::flat3_view<complex_t> view, float2 const cutoff )
{
    const int iy  = blockIdx.x;
    const int nx  = view.local_dims.x;
    const int ix0 = lowpass_ix0( view, iy, cutoff );

    complex_t * __restrict__ x_data = view.x_buffer;
    complex_t * __restrict__ y_data = view.y_buffer;
    complex_t * __restrict__ z_data = view.z_buffer;

    const int ystride = nx;

    for( int ix = ix0 + int( gpu::block::thread_rank() ); ix < nx; ix += int( gpu::block::num_threads() ) ) {
        const auto idx = iy * ystride + ix;
        x_data[ idx ] = 0;
        y_data[ idx ] = 0;
        z_data[ idx ] = 0;
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

    lowpass * clone() const override { return new lowpass ( cutoff ); };

    void apply( grid::flat<complex_t> & fld ) override {
        const auto nrows = fld.get_local_dims().y;
        if ( nrows > 0 )
            kernel::lowpass <<< nrows, 256 >>> ( fld.view(), cutoff );
    }
    
    void apply( grid::flat3<complex_t> & fld ) override {
        const auto nrows = fld.get_local_dims().y;
        if ( nrows > 0 )
            kernel::lowpass <<< nrows, 256 >>> ( fld.view(), cutoff );
    }
};

}
