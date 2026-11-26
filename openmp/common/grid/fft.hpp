#pragma once

#include <iostream>
#include <cstddef>
#include <cstring>
#include <string>
#include <algorithm>

#include <fftw3.h>

#include "../util/complex.hpp"
#include "../util/math.hpp"
#include "../util/error.hpp"
#include "zdf.hpp"

#include "tiled.hpp"
#include "flat.hpp"

#include "tiled_vec3.hpp"
#include "flat3.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Multi-threaded FFTW support
 * 
 * @note By default FFTW transforms run on a single thread: for small grids
 *       the threaded FFTW transforms were found to be slower than the serial
 *       ones. Define USE_FFTW_THREADS (e.g. CXXFLAGS += -DUSE_FFTW_THREADS)
 *       to have FFTW use omp_get_max_threads() threads. This requires
 *       OpenMP support and linking with the FFTW OpenMP library
 *       (-lfftw3f_omp), which is otherwise not needed.
 */
#if defined( USE_FFTW_THREADS ) && defined( _OPENMP )
#define FFTW_THREADS_ENABLED 1
#endif

namespace grid {

namespace fft {

using complex_t = util::complex<float>;

static_assert( sizeof( complex_t ) == sizeof( fftwf_complex ),
    "complex_t must be layout compatible with fftwf_complex" );

/**
 * @brief kspace grid layout
 *
 * @note Serial FFTW uses the normal (non-transposed) layout, same as the
 *       cuFFTMp (GPU) version: x is kx (nx/2+1 values), y is ky (ny values)
 */
inline constexpr bool kspace_transposed = false;

/**
 * @brief Returns k-space coordinate from k-space grid coordinates
 * 
 * @param gix       k-space grid coordinate
 * @param gdims     k-space grid dimensions
 * @param dk        K-space cell size
 * @return float2 
 */
inline float2 k( const int2 gix, const uint2 gdims, const float2 dk ) {
    static_assert( kspace_transposed == false , "this only works for normal k-space layout");
    return {
        gix.x * dk.x,
        (( 2 * gix.y < int(gdims.y) ) ? gix.y : ( gix.y - static_cast<int>(gdims.y) ) ) * dk.y
    };
}

/**
 * @brief Initialize FFTW library
 * 
 * @note Must be called before creating any plan
 */
inline void init( ) {
    #ifdef FFTW_THREADS_ENABLED
    fftwf_init_threads();
    #endif
}

/**
 * @brief Deallocate all persistent data and reset FFTW
 * 
 * @note All plans must be destroyed before calling this function. k-space
 *       grids may outlive it (their buffers are released with fftwf_free())
 */
inline void cleanup( ) {
    #ifdef FFTW_THREADS_ENABLED
    fftwf_cleanup_threads();
    #endif

    fftwf_cleanup();
}

/**
* @brief Get cell size in k-space
* 
* @param box       Box size in real space (not number of cells)
* @return float2   Cell size in k space
*/
inline float2 dk( float2 box ) {
    // Value from GLIBC math.h M_PIf
    constexpr float pi = 3.14159265358979323846f;
    return float2{ 2 * pi / box.x, 2 * pi / box.y };
}

/**
 * @brief Dimensions of complex grid for r2c and c2r transforms
 *
 * @note The k-space grid uses the normal layout: x is kx (nx/2+1 values),
 *       y is ky (ny values)
 * 
 * @param dims      Dimensions of real grid
 * @return uint2 
 */
inline uint2 grid_kdims( const uint2 dims ) {
    static_assert( kspace_transposed == false , "this only works for normal k-space layout");
    return make_uint2( dims.x/2 + 1, dims.y );
}

/**
 * @brief Size of k-space buffers for a 2D R2C/C2R transform
 *
 * @note The buffers hold either the k-space data or the padded real space
 *       data (in-place transforms). The size is rounded up to a multiple
 *       of 8 complex values (64 bytes), so that vec3 buffers placed
 *       contiguously all share the alignment of the first one, as required
 *       by FFTW new-array execute functions
 * 
 * @param dims          Real space dimensions
 * @return std::size_t  Required capacity (number of complex elements)
 */
inline std::size_t size_2d( uint2 const dims ) {
    const auto kdims = grid_kdims( dims );
    return roundup<8>( static_cast<std::size_t>( kdims.x ) * kdims.y );
}

// see https://www.fftw.org/fftw3_doc/Planner-Flags.html

struct plan { 
    enum rigor { 
        estimate   = FFTW_ESTIMATE,
        measure    = FFTW_MEASURE,
        patient    = FFTW_PATIENT,
        exhaustive = FFTW_EXHAUSTIVE
    };
};

namespace detail {

/**
 * @brief Copy complex buffer (OpenMP parallel)
 * 
 * @param dst   Destination buffer
 * @param src   Source buffer
 * @param n     Number of elements
 */
inline void copy( complex_t * const __restrict__ dst,
    const complex_t * const __restrict__ src, const std::size_t n ) {
    #pragma omp parallel
    {
        #ifdef _OPENMP
        const std::size_t nt  = omp_get_num_threads();
        const std::size_t tid = omp_get_thread_num();
        #else
        const std::size_t nt = 1, tid = 0;
        #endif
        const std::size_t chunk = ( n + nt - 1 ) / nt;
        const std::size_t i0 = std::min( n, tid * chunk );
        const std::size_t i1 = std::min( n, i0 + chunk );
        if ( i1 > i0 ) std::memcpy( dst + i0, src + i0, ( i1 - i0 ) * sizeof( complex_t ) );
    }
}

}

class fft_plan {

    protected:

    /// @brief Transform type
    enum class kind { r2c, c2r };

    /// @brief FFTW plan (single grid)
    fftwf_plan plan = nullptr;

    /**
     * @brief FFTW plan for vec3 grids (3 transforms in one call)
     * 
     * @note Only created for plans built from a tiled_vec3 grid. It requires
     *       the 3 components to be stored contiguously, `capacity` elements
     *       apart, as allocated by kspace_grid3()
     */
    fftwf_plan plan3 = nullptr;

    /**
     * @brief Size (in elements) of k-space buffers (complex values)
     *
     * @note This may be larger than the k-space grid size (real space data
     *       and alignment)
     */
    std::size_t capacity = 0;

    /**
     * @brief Allocate k-space buffer for use with transform
     *
     * @note The buffer will have space for `count * capacity` complex values
     *       and the alignment required by FFTW. Free it with fftwf_free()
     * 
     * @param count     Number of grids (1 or 3)
     * @return complex_t* 
     */
    complex_t * kspace_buffer( const std::size_t count = 1 ) const {
        complex_t * ptr = reinterpret_cast< complex_t * >( fftwf_alloc_complex( count * capacity ) );
        if ( ptr == nullptr ) {
            util::fatal( "fftwf_alloc_complex() failed for " + 
                    std::to_string( count * capacity * sizeof( complex_t ) ) + " bytes" );
        }
        return ptr;
    }

    /**
     * @brief Checks if the 3 buffers of a vec3 k-space grid are stored
     *        contiguously (as required by plan3)
     * 
     * @param x     x component buffer
     * @param y     y component buffer
     * @param z     z component buffer
     * @return true if plan3 can be used on these buffers
     */
    bool use_plan3( const complex_t * x, const complex_t * y, const complex_t * z ) const noexcept {
        return ( plan3 != nullptr ) && ( y == x + capacity ) && ( z == x + 2 * capacity );
    }

    private:

    /**
     * @brief Create in-place FFTW plan
     * 
     * @param type      Transform type (r2c/c2r)
     * @param howmany   Number of transforms (1 or 3)
     * @param flag      Planning-rigor flag for creating the plan
     * @return fftwf_plan 
     */
    fftwf_plan create( const kind type, const int howmany, const plan::rigor flag ) {

        // Temporary buffer, used only to create the (in-place) plan
        // Using kspace_buffer() ensures the same alignment as k-space grids
        complex_t * buffer = kspace_buffer( howmany );
        float * data_r = reinterpret_cast< float * >( buffer );
        fftwf_complex * data_c = reinterpret_cast< fftwf_complex * >( buffer );

        // C order: slowest dimension (y) first
        const int nx  = grid_dims.x;
        const int ny  = grid_dims.y;
        const int nxc = nx/2 + 1;

        int n[]           = { ny, nx };
        int real_embed[]  = { ny, 2 * nxc };    // padded real layout (in-place)
        int cplx_embed[]  = { ny, nxc };

        // Distance between transforms, in units of the respective type
        const int cdist = static_cast<int>( capacity );
        const int rdist = 2 * cdist;

        #ifdef FFTW_THREADS_ENABLED
        // Multi-threaded transforms, see USE_FFTW_THREADS note at the top
        // of this file, and the note in Makefile for macOS homebrew builds
        fftwf_plan_with_nthreads( omp_get_max_threads() );
        #endif

        fftwf_plan p = nullptr;
        switch( type ) {
        case kind::r2c:
            p = fftwf_plan_many_dft_r2c( 2, n, howmany,
                data_r, real_embed, 1, rdist,
                data_c, cplx_embed, 1, cdist,
                flag );
            break;
        case kind::c2r:
            p = fftwf_plan_many_dft_c2r( 2, n, howmany,
                data_c, cplx_embed, 1, cdist,
                data_r, real_embed, 1, rdist,
                flag );
            break;
        }

        // Free the memory used for creating the plan
        fftwf_free( buffer );

        if ( p == nullptr ) {
            util::fatal( "FFTW plan creation failed" );
        }

        return p;
    }

    /**
     * @brief Verify that a source grid matches the FFT plan
     * 
     * @param in_dims     Source grid dimensions
     */
    void check_source( const uint2 in_dims ) const {
        if (( in_dims.x != grid_dims.x ) ||
            ( in_dims.y != grid_dims.y )) {
            util::fatal( "Source grid dimensions " + to_string( in_dims ) +
                        " and FFT plan dimensions " + to_string( grid_dims ) +
                        " don't match" );
        }
    }

    public:

    /// @brief Real space data dimensions
    const uint2 grid_dims;

    /**
     * @brief Construct a new fft plan object
     * 
     * @param dims      Real space dimensions
     * @param type      Transform type (r2c/c2r)
     * @param flag      Planning-rigor flag for creating the plan
     * @param vec3      Also create the vec3 (3 transforms) plan
     */
    fft_plan( const uint2 dims, const kind type, const plan::rigor flag,
        const bool vec3 = false ) : grid_dims( dims ) {

        if ( dims.x == 0 || dims.y == 0 ) {
            util::fatal( "Invalid FFT dimensions: " + to_string( dims ) );
        }

        // Size of k-space buffers
        capacity = size_2d( grid_dims );

        // Single grid plan
        plan = create( type, 1, flag );

        // Vec3 grid plan
        if ( vec3 ) plan3 = create( type, 3, flag );
    }

    /**
     * @brief Construct a new fft plan object from a tiled grid source
     * 
     * @param source    Tiled grid source
     * @param type      Transform type (r2c/c2r)
     * @param flag      Planning-rigor flag for creating the plan
     */
    fft_plan( const grid::tiled<float> & source, const kind type, const plan::rigor flag ):
        fft_plan( source.get_grid_dims(), type, flag ) {

        // Verify that the FFT dimensions match the source
        check_source( source.get_grid_dims() );
    }

    /**
     * @brief Construct a new fft plan object from a tiled_vec3 grid source
     * 
     * @param source    Tiled grid source
     * @param type      Transform type (r2c/c2r)
     * @param flag      Planning-rigor flag for creating the plan
     */
    fft_plan( const grid::tiled_vec3<float> & source, const kind type, const plan::rigor flag ):
        fft_plan( source.get_grid_dims(), type, flag, true ) {

        // Verify that the FFT dimensions match the source
        check_source( source.get_grid_dims() );
    }

    fft_plan( const fft_plan & ) = delete;
    fft_plan & operator=( const fft_plan & ) = delete;

    /**
     * @brief Destroy the plan object
     * 
     * @note Must be called before grid::fft::cleanup()
     */
    ~fft_plan() {
        if ( plan  != nullptr ) fftwf_destroy_plan( plan );
        if ( plan3 != nullptr ) fftwf_destroy_plan( plan3 );
    }

    inline uint2 get_grid_dims() const noexcept {
        return grid_dims;
    }

    /**
    * @brief Dimension of complex grid (x = kx, y = ky)
    * @return uint2 
    */
    inline uint2 get_grid_kdims( ) const noexcept {
        return ::grid::fft::grid_kdims( grid_dims );
    }

    inline std::size_t get_capacity() const noexcept {
        return capacity;
    }

    /**
     * @brief Create a flat grid for the k-space data
     * 
     * @note The grid does not reference the plan, so it may outlive it
     * 
     * @return grid::flat<complex_t> 
     */
    inline grid::flat<complex_t> kspace_grid( ) const {

        // Build buffer destructor
        auto release = [ ] ( complex_t * p ){ 
            fftwf_free( p );
        };

        // Build flat grid object
        return grid::flat<complex_t> ( 
            ::grid::fft::grid_kdims( grid_dims ), 
            kspace_buffer(), capacity, release
        );
    }

    /**
     * @brief Allocate a new flat grid for the k-space data
     * 
     * @return grid::flat<complex_t>* 
     */
    inline grid::flat<complex_t>* new_kspace_grid () const {
        return new grid::flat<complex_t>( kspace_grid() );
    }

    /**
     * @brief Create a flat3 grid for the k-space data
     * 
     * @note The 3 buffers are allocated as a single block (x, y, z,
     *       `capacity` elements apart) so that vec3 transforms can be done
     *       with a single FFTW call. The grid does not reference the plan,
     *       so it may outlive it
     * 
     * @return grid::flat3<complex_t> 
     */
    inline grid::flat3<complex_t> kspace_grid3( ) const {

        complex_t * base = kspace_buffer( 3 );

        // Build buffer destructor, only the x buffer (start of the block)
        // is released
        auto release = [ base ] ( complex_t * p ){ 
            if ( p == base ) fftwf_free( p );
        };

        // Build flat grid object
        return grid::flat3<complex_t> ( 
            ::grid::fft::grid_kdims( grid_dims ),
            base, base + capacity, base + 2 * capacity,
            capacity, release
        );
    }

    /**
     * @brief Allocate a new flat3 grid for the k-space data
     * 
     * @return grid::flat3<complex_t>* 
     */
    inline grid::flat3<complex_t>* new_kspace_grid3() const {
        return new grid::flat3<complex_t>( kspace_grid3() );
    }
};

/**
 * @brief Real to complex FFT plan
 * 
 */
class r2c_plan : public fft_plan {

    public:

    /**
     * @brief Construct a new r2c plan object
     * 
     * @param dims      Dimensions of real space data
     * @param flag      Planning-rigor flag for creating the plan, defaults to
     *                  `plan::estimate`
     */
    r2c_plan( const uint2 dims, const plan::rigor flag = plan::estimate ) :
        fft_plan( dims, kind::r2c, flag ) { }

    /**
     * @brief Construct an FFT plan object from a `grid::tiled<float>` object
     * 
     * @param source    Source object. Only dimensions are used (source data
     *                  is not modified)
     * @param flag      Planning-rigor flag for creating the plan, defaults to
     *                  `plan::estimate`
     */
    r2c_plan( const grid::tiled<float> & source, const plan::rigor flag = plan::estimate ):
        fft_plan( source, kind::r2c, flag ) { }

    /**
     * @brief Construct an FFT plan object from a `grid::tiled_vec3<float>` object
     * 
     * @param source    Source object. Only dimensions are used (source data
     *                  is not modified)
     * @param flag      Planning-rigor flag for creating the plan, defaults to
     *                  `plan::estimate`
     */
    r2c_plan( const grid::tiled_vec3<float> & source, const plan::rigor flag = plan::estimate ):
        fft_plan( source, kind::r2c, flag ) { }

    /**
     * @brief Stream extraction
     * 
     * @param os    Output stream
     * @param obj   r2c_plan object
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, const r2c_plan & obj) {
        return os << "FFT r2c plan, dims: " << obj.grid_dims;
    }

    /**
     * @brief Perform real to complex transform
     * 
     * @param output        Ouput grid (complex), must have been created by
     *                      kspace_grid() / new_kspace_grid()
     * @param input         Input grid (real)
     */
    void transform( grid::flat<complex_t> & output, const grid::tiled<float> & input ) {

        float * data = reinterpret_cast<float *>( output.data() );

        // Gather input data (padded real layout)
        input.gather( data, make_uint2(1, 2 * (grid_dims.x / 2 + 1)) );

        // Perform the transform in-place
        fftwf_execute_dft_r2c( plan, data, reinterpret_cast<fftwf_complex *>( data ) );
    }
    
    /**
     * @brief Perform real to complex transform
     * 
     * @param output        Ouput grid (complex), must have been created by
     *                      kspace_grid3() / new_kspace_grid3()
     * @param input         Input grid (real)
     */
    void transform( grid::flat3<complex_t> & output, const grid::tiled_vec3<float> & input ) {

        float * x_data = reinterpret_cast<float *>(output.x());
        float * y_data = reinterpret_cast<float *>(output.y());
        float * z_data = reinterpret_cast<float *>(output.z());
        
        // Gather input data (padded real layout)
        input.gather( x_data, y_data, z_data, make_uint2(1, 2 * (grid_dims.x / 2 + 1)) );

        // Perform the transforms in-place
        if ( use_plan3( output.x(), output.y(), output.z() ) ) {
            fftwf_execute_dft_r2c( plan3, x_data, reinterpret_cast<fftwf_complex*>(x_data) );
        } else {
            fftwf_execute_dft_r2c( plan, x_data, reinterpret_cast<fftwf_complex*>(x_data) );
            fftwf_execute_dft_r2c( plan, y_data, reinterpret_cast<fftwf_complex*>(y_data) );
            fftwf_execute_dft_r2c( plan, z_data, reinterpret_cast<fftwf_complex*>(z_data) );
        }
    }
};

/**
 * @brief Complex to real FFT plan
 * 
 */
class c2r_plan : public fft_plan {
    private:

    /// @brief Scratch buffer, so transform() preserves its input
    complex_t * scratch = nullptr;

    /// @brief Scratch buffer for vec3 transforms (3 contiguous buffers)
    complex_t * scratch3 = nullptr;

    public:

    /**
     * @brief Construct a new c2r plan object
     * 
     * @param dims      Dimensions of real space data
     * @param flag      Planning-rigor flag for creating the plan, defaults to
     *                  `plan::estimate`
     */
    c2r_plan( const uint2 dims, const plan::rigor flag = plan::estimate ) :
        fft_plan( dims, kind::c2r, flag ) { }

    /**
     * @brief Construct a new c2r plan object from a `grid::tiled<float>` object
     * 
     * @param dest      Destination object. Only dimensions are used (data is
     *                  not modified)
     * @param flag      Planning-rigor flag for creating the plan, defaults to
     *                  `plan::estimate`
     */
    c2r_plan( const grid::tiled<float> & dest, const plan::rigor flag = plan::estimate ):
        fft_plan( dest, kind::c2r, flag ) { }

    /**
     * @brief Construct a new c2r plan object from a `grid::tiled_vec3<float>` object
     * 
     * @param dest      Destination object. Only dimensions are used (data is
     *                  not modified)
     * @param flag      Planning-rigor flag for creating the plan, defaults to
     *                  `plan::estimate`
     */
    c2r_plan( const grid::tiled_vec3<float> & dest, const plan::rigor flag = plan::estimate ):
        fft_plan( dest, kind::c2r, flag ) { }

    /**
     * @brief Destroy the c2r plan object
     * 
     */
    ~c2r_plan( ){
        // Destroy internal scratch space
        if ( scratch  != nullptr ) fftwf_free( scratch  );
        if ( scratch3 != nullptr ) fftwf_free( scratch3 );
    }

    /**
     * @brief Normalization factor
     * 
     * @return float    1/(dims.x * dims.y)
     */
    inline float norm( ) const {
        return 1.f / ( static_cast<float>( grid_dims.x ) * grid_dims.y );
    }

    /**
     * @brief Stream extraction
     * 
     * @param os 
     * @param obj 
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, const c2r_plan& obj) {
        return os << "FFT c2r plan, dims: " << obj.grid_dims;
    }

    /**
     * @brief Perform complex to real transform
     * 
     * @note Input data is preserved
     * 
     * @param output        Output grid (real)
     * @param input         Input grid (complex)
     */
    void transform( grid::tiled<float> & output, const grid::flat<complex_t> & input ) {

        // If scratch buffer has not been allocated yet, do so now
        if ( scratch == nullptr ) scratch = kspace_buffer();

        // Copy input data to scratch
        detail::copy( scratch, input.data(), input.buffer_size() );

        // Perform in-place transform of scratch
        float * data = reinterpret_cast<float *>( scratch );
        fftwf_execute_dft_c2r( plan, reinterpret_cast<fftwf_complex *>( scratch ), data );

        // Buffer now holds padded real-space data, scatter and normalize
        output.scatter( data, norm(), make_uint2(1, 2 * (grid_dims.x / 2 + 1)) );
    }

    /**
     * @brief Perform complex to real transform
     * 
     * @note Input data is preserved
     * 
     * @param output        Output grid (real)
     * @param input         Input grid (complex)
     */
    void transform( grid::tiled_vec3<float> & output, const grid::flat3<complex_t> & input ) {

        // If scratch buffers have not been allocated yet, do so now
        if ( scratch3 == nullptr ) scratch3 = kspace_buffer( 3 );

        complex_t * scratch_x = scratch3;
        complex_t * scratch_y = scratch3 + capacity;
        complex_t * scratch_z = scratch3 + 2 * capacity;

        // Copy input data to scratch
        detail::copy( scratch_x, input.x(), input.buffer_size() );
        detail::copy( scratch_y, input.y(), input.buffer_size() );
        detail::copy( scratch_z, input.z(), input.buffer_size() );

        float * data_x = reinterpret_cast<float *>( scratch_x );
        float * data_y = reinterpret_cast<float *>( scratch_y );
        float * data_z = reinterpret_cast<float *>( scratch_z );

        // Perform in-place transforms of scratch buffers
        if ( plan3 != nullptr ) {
            fftwf_execute_dft_c2r( plan3, reinterpret_cast<fftwf_complex *>( scratch_x ), data_x );
        } else {
            fftwf_execute_dft_c2r( plan, reinterpret_cast<fftwf_complex *>( scratch_x ), data_x );
            fftwf_execute_dft_c2r( plan, reinterpret_cast<fftwf_complex *>( scratch_y ), data_y );
            fftwf_execute_dft_c2r( plan, reinterpret_cast<fftwf_complex *>( scratch_z ), data_z );
        }

        // Scatter data into output grid and normalize
        output.scatter( data_x, data_y, data_z, norm(), make_uint2(1, 2 * (grid_dims.x / 2 + 1)) );
    }
};

/**
 * @brief Copy a k-space buffer to a new buffer, rotating ky so that negative
 *        modes come first
 * 
 * @note The output layout matches the cuFFTMp (GPU) and FFTW MPI versions so
 *       that files are identical: x is kx, y is ky. k-space data is stored
 *       in FFT order along ky (see grid::fft::k()). Rotating by ceil(ny/2)
 *       rows gives a monotonic ky axis running from -floor(ny/2) to
 *       floor((ny-1)/2) modes.
 * 
 * @param data          k-space buffer (x = kx, y = ky)
 * @param dims          k-space grid dimensions (x = kx, y = ky)
 * @return              Buffer with rotated data, of size dims.x * dims.y;
 *                      free it with memory::free()
 */
inline complex_t * kspace_to_host( const complex_t * const __restrict__ data, const uint2 dims ) {

    /// @brief Number of kx modes
    const std::size_t nkx = dims.x;
    /// @brief Number of ky modes
    const std::size_t nky = dims.y;

    complex_t * __restrict__ h_buffer = memory::malloc<complex_t>( nkx * nky );

    // Rotate whole rows: out[j] = in[(j + yroll) % ny]
    const std::size_t yroll = ( nky + 1 ) / 2;     // ceil(ny/2)

    #pragma omp parallel for
    for( std::size_t j = 0; j < nky; j++ ) {
        const std::size_t iky = ( j + yroll ) % nky;
        std::memcpy( h_buffer + j * nkx, data + iky * nkx, nkx * sizeof( complex_t ) );
    }

    return h_buffer;
}

/**
 * @brief Select one component of a vector k-space grid
 * 
 * @param cf    Vector k-space grid
 * @param fc    Field component
 * @return      Pointer to the component data
 */
inline const complex_t * kspace_component( const grid::flat3<complex_t> & cf, const fcomp::cart fc ) {
    switch( fc ) {
    case fcomp::cart::x : return cf.x();
    case fcomp::cart::y : return cf.y();
    case fcomp::cart::z : return cf.z();
    }
    util::fatal( "kspace_save(): invalid field component" );
}

namespace detail {

/**
 * @brief Saves k-space data with full metadata, correcting the layout
 * 
 * @param data          k-space data
 * @param dims          k-space grid dimensions (x = kx, y = ky)
 * @param info          Grid metadata
 * @param iter          Iteration metadata
 * @param path          File path
 */
inline void kspace_save( const complex_t * data, const uint2 dims,
    zdf::grid_info &info, const zdf::iteration &iter, const std::string &path ) {

    complex_t * h_buffer = kspace_to_host( data, dims );

    // Fill in grid dimensions (x = kx, y = ky)
    info.ndims = 2;
    info.count[0] = dims.x;
    info.count[1] = dims.y;

    zdf::save_grid<complex_t>( h_buffer, info, iter, path );

    memory::free( h_buffer );
}

/**
 * @brief Saves k-space data (no metadata), correcting the layout
 * 
 * @param data          k-space data
 * @param dims          k-space grid dimensions (x = kx, y = ky)
 * @param name          Grid name
 * @param filename      Output file name (includes path)
 */
inline void kspace_save( const complex_t * data, const uint2 dims,
    const std::string & name, const std::string & filename ) {

    // x = kx, y = ky
    uint64_t count[2] = { dims.x, dims.y };

    complex_t * h_buffer = kspace_to_host( data, dims );

    zdf::save_grid( h_buffer, 2, count, name, filename );

    memory::free( h_buffer );
}

}

/**
 * @brief Saves a k-space grid with full metadata, correcting the layout
 * 
 * @note Data is saved in (kx, ky) order, matching the GPU version. The ky
 *       axis metadata in `info` must describe the rotated layout, i.e. run
 *       from -floor(ny/2)*dk.y to floor((ny-1)/2)*dk.y
 * 
 * @param cf        k-space grid
 * @param info      Grid metadata
 * @param iter      Iteration metadata
 * @param path      File path
 */
inline void kspace_save( const grid::flat<complex_t> & cf, zdf::grid_info &info, const zdf::iteration &iter, const std::string &path ) {
    detail::kspace_save( cf.data(), cf.get_grid_dims(), info, iter, path );
}

/**
 * @brief Saves one component of a vector k-space grid with full metadata,
 *        correcting the layout
 * 
 * @note Data is saved in (kx, ky) order, matching the GPU version. The ky
 *       axis metadata in `info` must describe the rotated layout, i.e. run
 *       from -floor(ny/2)*dk.y to floor((ny-1)/2)*dk.y
 * 
 * @param cf        Vector k-space grid
 * @param fc        Field component to save
 * @param info      Grid metadata
 * @param iter      Iteration metadata
 * @param path      File path
 */
inline void kspace_save( const grid::flat3<complex_t> & cf, const fcomp::cart fc, zdf::grid_info &info, const zdf::iteration &iter, const std::string &path ) {
    detail::kspace_save( kspace_component( cf, fc ), cf.get_grid_dims(), info, iter, path );
}

/**
 * @brief Saves a k-space grid (no metadata), correcting the layout
 * 
 * @note Data is saved in (kx, ky) order, matching the GPU version
 * 
 * @param cf        k-space grid
 * @param filename  Output file name (includes path)
 */
inline void kspace_save( const grid::flat<complex_t> & cf, const std::string & filename ) {
    detail::kspace_save( cf.data(), cf.get_grid_dims(), cf.name, filename );
}

/**
 * @brief Saves one component of a vector k-space grid (no metadata),
 *        correcting the layout
 * 
 * @note Data is saved in (kx, ky) order, matching the GPU version
 * 
 * @param cf        Vector k-space grid
 * @param fc        Field component to save
 * @param filename  Output file name (includes path)
 */
inline void kspace_save( const grid::flat3<complex_t> & cf, const fcomp::cart fc, const std::string & filename ) {
    detail::kspace_save( kspace_component( cf, fc ), cf.get_grid_dims(), cf.name, filename );
}

} // namespace fft

} // namespace grid
