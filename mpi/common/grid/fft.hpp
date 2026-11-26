#pragma once

#include <iostream>
#include <cstddef>
#include <string>
#include <algorithm>

#include <mpi.h>
#include <fftw3.h>
#include <fftw3-mpi.h>

#include "../util/complex.hpp"
#include "../parallel/parallel.hpp"
#include "../zdf/zdf-parallel.hpp"

#include "tiled.hpp"
#include "flat.hpp"

#include "tiled_vec3.hpp"
#include "flat3.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace grid {

namespace fft {

using complex_t = util::complex<float>;

static_assert( sizeof( complex_t ) == sizeof( fftwf_complex ),
    "complex_t must be layout compatible with fftwf_complex" );

/**
 * @brief kspace grid layout
 *
 * @note FFTW uses a transposed layout (unlike cuFFTMp): x is ky (ny values),
 *       y is kx (nx/2+1 values)
 */
inline constexpr bool kspace_transposed = true;

/**
 * @brief Returns k-space coordinate from global k-space grid coordinates
 *
 * @note This expects a tranposed k-space
 * 
 * @param gix       Global k-space grid coordinate
 * @param gdims     Global k-space grid dimensions
 * @param dk        K-space cell size
 * @return float2 
 */
inline float2 k( const int2 gix, const uint2 gdims, const float2 dk ) {
    static_assert( kspace_transposed == true , "this only works for a transposed k-space layout");
    return {
        gix.y * dk.x,
        (( 2 * gix.x < int(gdims.x) ) ? gix.x : ( gix.x - static_cast<int>(gdims.x) ) ) * dk.y
    };
}

/**
 * @brief Initialize FFTW library
 * 
 * @note Must be called after MPI initialization and before creating any plan
 */
inline void init( ) {
    #ifdef _OPENMP
    fftwf_init_threads();
    #endif

    fftwf_mpi_init();
}

/**
 * @brief Deallocate all persistent data and reset FFTW
 * 
 * @note All plans must be destroyed before calling this function. k-space
 *       grids may outlive it (their buffers are released with fftwf_free())
 */
inline void cleanup( ) {
    fftwf_mpi_cleanup();

    #ifdef _OPENMP
    fftwf_cleanup_threads();
    #endif
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
 * @brief Global dimension of complex grid for r2c and c2r transforms
 *
 * @note The k-space grid is transposed: x is ky (ny values), y is kx
 *       (nx/2+1 values). FFTW distributes it along kx, i.e. along y, so it
 *       uses the same parallel partition as the real space grid
 * 
 * @param global_dims   Global dimensions of real grid (not transposed)
 * @return uint2 
 */
inline uint2 global_kdims( const uint2 global_dims ) {
    static_assert( kspace_transposed == true , "this only works for a transposed k-space layout");
    return make_uint2( global_dims.y, global_dims.x/2 + 1 );
}

/**
 * @brief Get local dimensions for a 2D R2C/C2R transform
 *
 * @note Real space is split along y; k-space (x = ky, y = kx) is also split
 *       along y (i.e. along kx), following the FFTW transposed MPI
 *       distributions
 * 
 * @param global_dims   (in) Global real space dimensions
 * @param part          (in) Parallel partition
 * @param local_dims    (out) Local real grid dimensions
 * @param local_start   (out) Start position of local real grid
 * @param local_kdims   (out) Local k-space grid dimension
 * @param local_kstart  (out) Start position of local k-space grid
 * @return std::size_t  Required capacity (number of complex elements)
 */
inline std::size_t local_size_2d( 
    uint2 const global_dims, const mpi::cart2d & part,
    uint2 & local_dims, uint2 & local_start, 
    uint2 & local_kdims, uint2 & local_kstart ) {

    static_assert( kspace_transposed == true , "this only works for a transposed k-space layout");

    if ( part.dims.x != 1 ) {
        mpi::fatal("FFT operations require that the domain is only"
                    " partitioned along the y direction" );
    }
    
    ///@brief number of parallel nodes (along y)
    int size = part.dims.y;
    ///@brief node rank (along y)
    int rank = part.get_coords().y;

    // FFTW assigns blocks by communicator rank, which must match the y coordinate
    int comm_rank;
    MPI_Comm_rank( part.get_comm(), & comm_rank );
    if ( comm_rank != rank ) {
        mpi::fatal( "FFT: communicator rank does not match partition y coordinate" );
    }

    if ( global_dims.y < static_cast<unsigned>(size) ) {
        mpi::fatal( "FFT: global y dimension must be at least the number of parallel nodes" );
    }

    ///@brief Number of complex elements along x (and kx)
    const ptrdiff_t nxc = global_dims.x/2 + 1;

    ptrdiff_t local_n0,  local_0_start;
    ptrdiff_t local_n1c, local_1_start;

    // For r2c / c2r transforms the returned size is in complex elements
    const ptrdiff_t local_size = fftwf_mpi_local_size_2d_transposed(
            global_dims.y, nxc, part.get_comm(),
            &local_n0,  &local_0_start,
            &local_n1c, &local_1_start
        );

    // Spatial coordinate partition (partition along y)
    local_dims   = make_uint2( global_dims.x, local_n0 );
    local_start  = make_uint2( 0, local_0_start );

    // k-space partition (transposed, x = ky, y = kx, partition along kx)
    local_kdims  = make_uint2( global_dims.y, local_n1c );
    local_kstart = make_uint2( 0, local_1_start );

    return static_cast< std::size_t >( local_size );
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

    /// @brief FFTW plan
    fftwf_plan plan = nullptr;

    /**
     * @brief Size (in elements) of k-space buffers (complex values)
     *
     * @note This is the FFTW local size, which may be larger than the
     *       local k-space grid size (real space data and workspace)
     */
    std::size_t capacity = 0;

    /**
     * @brief Allocate k-space buffer for use with transform
     *
     * @note The buffer will have space for `capacity` complex values and the
     *       alignment required by FFTW. Free it with fftwf_free()
     * 
     * @return complex_t* 
     */
    complex_t * kspace_buffer() const {
        complex_t * ptr = reinterpret_cast< complex_t * >( fftwf_alloc_complex( capacity ) );
        if ( ptr == nullptr ) {
            mpi::fatal( "fftwf_alloc_complex() failed for " + 
                    std::to_string( capacity * sizeof( complex_t ) ) + " bytes" );
        }
        return ptr;
    }

    private:

    /// @brief Local real space dimension
    uint2 local_dims;

    /// @brief Position of local real space grid on global grid
    uint2 local_start;

    /// @brief Local k-space grid dimension
    uint2 local_kdims;

    /// @brief Position of local k-space grid on global grid
    uint2 local_kstart;

    /**
     * @brief Verify that a source grid partition matches the FFT plan
     * 
     * @param in_local_dims     Source grid local dimensions
     * @param in_local_start    Source grid local start position
     */
    void check_source( const uint2 in_local_dims, const uint2 in_local_start ) const {
        if (( in_local_dims.x != local_dims.x ) ||
            ( in_local_dims.y != local_dims.y )) {
            mpi::fatal( "Source grid local dimensions " + to_string( in_local_dims ) +
                        " and FFT plan local dimensions " + to_string( local_dims ) +
                        " don't match. Check global dimensions and parallel partition" );
        }

        if (( in_local_start.x != local_start.x ) ||
            ( in_local_start.y != local_start.y )) {
            mpi::fatal( "Source grid local start position " + to_string( in_local_start ) +
                        " and FFT plan local start " + to_string( local_start ) +
                        " don't match. Check global dimensions and parallel partition" );
        }
    }

    public:

    /// @brief Real space global data dimensions
    const uint2 global_dims;

    /// @brief Parallel partition (used for both real and k-space grids)
    const mpi::cart2d & part;

    /**
     * @brief Construct a new fft plan object
     * 
     * @note FFTW duplicates the MPI communicator internally, so there is no
     *       communicator lifetime requirement
     * 
     * @param global_dims   Global real space dimensions
     * @param part          Parallel partition
     * @param type          Transform type (r2c/c2r)
     * @param flag          Planning-rigor flag for creating the plan
     */
    fft_plan( const uint2 global_dims, const mpi::cart2d & part, const kind type,
        const plan::rigor flag ) :
        global_dims( global_dims ), part( part ) {

        // Initialize dimensions of local real / k-space grids
        capacity = local_size_2d( global_dims, part,
            local_dims, local_start, local_kdims, local_kstart );

        // Temporary buffer, used only to create the (in-place) plan
        // Using kspace_buffer() ensures the same alignment as k-space grids
        complex_t * buffer = kspace_buffer();
        float * data_r = reinterpret_cast< float * >( buffer );
        fftwf_complex * data_c = reinterpret_cast< fftwf_complex * >( buffer );

        // C order: slowest dimension (y) first
        const ptrdiff_t nx = global_dims.x;
        const ptrdiff_t ny = global_dims.y;

        #ifdef _OPENMP
        // See note in Makefile for macOS homebrew builds
        fftwf_plan_with_nthreads( omp_get_max_threads() );
        #endif

        switch( type ) {
        case kind::r2c:
            plan = fftwf_mpi_plan_dft_r2c_2d( ny, nx, data_r, data_c,
                part.get_comm(), plan::measure | FFTW_MPI_TRANSPOSED_OUT );
            break;
        case kind::c2r:
            plan = fftwf_mpi_plan_dft_c2r_2d( ny, nx, data_c, data_r,
                part.get_comm(), plan::measure | FFTW_MPI_TRANSPOSED_IN );
            break;
        }

        // Free the memory used for creating the plan
        fftwf_free( buffer );

        if ( plan == nullptr ) {
            mpi::fatal( "FFTW plan creation failed" );
        }

        if ( part.root() )  {
            std::cout << "Created FFTW plan:\n";
            fftwf_print_plan( plan ); 
            std::printf("\n");
        }
    }

    /**
     * @brief Construct a new fft plan object from a tiled grid source
     * 
     * @param source    Tiled grid source
     * @param type      Transform type (r2c/c2r)
     * @param flag      Planning-rigor flag for creating the plan
     */
    fft_plan( const grid::tiled<float> & source, const kind type, const plan::rigor flag ):
        fft_plan( source.get_global_dims(), source.get_part(), type, flag ) {

        // Verify that the FFT partition matches the source partition
        check_source( source.get_local_dims(),
                      source.get_local_tile_start() * source.tile_dims );
    }

    /**
     * @brief Construct a new fft plan object from a tiled_vec3 grid source
     * 
     * @param source    Tiled grid source
     * @param type      Transform type (r2c/c2r)
     * @param flag      Planning-rigor flag for creating the plan
     */
    fft_plan( const grid::tiled_vec3<float> & source, const kind type, const plan::rigor flag ):
        fft_plan( source.get_global_dims(), source.get_part(), type, flag ) {

        // Verify that the FFT partition matches the source partition
        check_source( source.get_local_dims(),
                      source.get_local_tile_start() * source.tile_dims );
    }

    fft_plan( const fft_plan & ) = delete;
    fft_plan & operator=( const fft_plan & ) = delete;

    /**
     * @brief Destroy the plan object
     * 
     * @note Must be called before grid::fft::cleanup()
     */
    ~fft_plan() {
        if ( plan != nullptr ) fftwf_destroy_plan( plan );
    }

    inline uint2 get_global_dims() const noexcept {
        return global_dims;
    }

    /**
    * @brief Global dimension of complex grid (x = ky, y = kx)
    * @return uint2 
    */
    inline uint2 get_global_kdims( ) const noexcept {
        return ::grid::fft::global_kdims( global_dims );
    }

    inline uint2 get_local_dims( ) const noexcept {
        return local_dims;
    }

    inline uint2 get_local_start( ) const noexcept {
        return local_start;
    }

    inline uint2 get_local_kdims( ) const noexcept {
        return local_kdims;
    }

    inline uint2 get_local_kstart( ) const noexcept {
        return local_kstart;
    }

    inline std::size_t get_capacity() const noexcept {
        return capacity;
    }

    /**
     * @brief Create a flat grid for the k-space data
     * 
     * @note The grid references the real space partition (not the plan), so
     *       it may outlive the plan
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
            ::grid::fft::global_kdims( global_dims ), local_kdims, local_kstart, part, 
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
     * @note The grid references the real space partition (not the plan), so
     *       it may outlive the plan
     * 
     * @return grid::flat3<complex_t> 
     */
    inline grid::flat3<complex_t> kspace_grid3( ) const {

        // Build buffer destructor
        auto release = [ ] ( complex_t * p ){ 
            fftwf_free( p );
        };

        // Build flat grid object
        return grid::flat3<complex_t> ( 
            ::grid::fft::global_kdims( global_dims ), local_kdims, local_kstart, part, 
            kspace_buffer(), kspace_buffer(), kspace_buffer(),
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
     * @param global_dims   Global dimensions of real space data
     * @param part          Parallel partition
     * @param flag          Planning-rigor flag for creating the plan, defaults to
     *                      `plan::estimate`
     */
    r2c_plan( const uint2 global_dims, const mpi::cart2d & part,
        const plan::rigor flag = plan::estimate ) :
        fft_plan( global_dims, part, kind::r2c, flag ) { }

    /**
     * @brief Construct an FFT plan object from a `grid::tiled<float>` object
     * 
     * @param source    Source object. Only global dimensions and partition are
     *                  used (source data is not modified)
     * @param flag      Planning-rigor flag for creating the plan, defaults to
     *                  `plan::estimate`
     */
    r2c_plan( const grid::tiled<float> & source, const plan::rigor flag = plan::estimate ):
        fft_plan( source, kind::r2c, flag ) { }

    /**
     * @brief Construct an FFT plan object from a `grid::tiled_vec3<float>` object
     * 
     * @param source    Source object. Only global dimensions and partition are
     *                  used (source data is not modified)
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
        return os << "FFT r2c plan, dims: " << obj.global_dims;
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
        input.gather( data, make_uint2(1, 2 * (global_dims.x / 2 + 1)) );

        // Perform the transform in-place
        fftwf_mpi_execute_dft_r2c( plan, data, reinterpret_cast<fftwf_complex *>( data ) );
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
        input.gather( x_data, y_data, z_data, make_uint2(1, 2 * (global_dims.x / 2 + 1)) );

        // Perform the transforms in-place
        fftwf_mpi_execute_dft_r2c( plan, x_data, reinterpret_cast<fftwf_complex*>(x_data) );
        fftwf_mpi_execute_dft_r2c( plan, y_data, reinterpret_cast<fftwf_complex*>(y_data) );
        fftwf_mpi_execute_dft_r2c( plan, z_data, reinterpret_cast<fftwf_complex*>(z_data) );
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

    /// @brief Additional scratch memory for vec3 transform (y)
    complex_t * scratch_y = nullptr;

    /// @brief Additional scratch memory for vec3 transform (z)
    complex_t * scratch_z = nullptr;

    public:

    /**
     * @brief Construct a new c2r plan object
     * 
     * @param global_dims   Global dimensions of real space data
     * @param part          Parallel partition
     * @param flag          Planning-rigor flag for creating the plan, defaults to
     *                      `plan::estimate`
     */
    c2r_plan( const uint2 global_dims, const mpi::cart2d & part,
        const plan::rigor flag = plan::estimate ) :
        fft_plan( global_dims, part, kind::c2r, flag ) { }

    /**
     * @brief Construct a new c2r plan object from a `grid::tiled<float>` object
     * 
     * @param dest      Destination object. Only global dimensions and partition
     *                  are used (data is not modified)
     * @param flag      Planning-rigor flag for creating the plan, defaults to
     *                  `plan::estimate`
     */
    c2r_plan( const grid::tiled<float> & dest, const plan::rigor flag = plan::estimate ):
        fft_plan( dest, kind::c2r, flag ) { }

    /**
     * @brief Construct a new c2r plan object from a `grid::tiled_vec3<float>` object
     * 
     * @param dest      Destination object. Only global dimensions and partition
     *                  are used (data is not modified)
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
        if ( scratch   != nullptr ) fftwf_free( scratch   );
        if ( scratch_y != nullptr ) fftwf_free( scratch_y );
        if ( scratch_z != nullptr ) fftwf_free( scratch_z );
    }

    /**
     * @brief Normalization factor
     * 
     * @return float    1/(dims.x * dims.y)
     */
    inline float norm( ) const {
        return 1.f / ( static_cast<float>( global_dims.x ) * global_dims.y );
    }

    /**
     * @brief Stream extraction
     * 
     * @param os 
     * @param obj 
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, const c2r_plan& obj) {
        return os << "FFT c2r plan, dims: " << obj.global_dims;
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
        fftwf_mpi_execute_dft_c2r( plan, reinterpret_cast<fftwf_complex *>( scratch ), data );

        // Buffer now holds padded real-space slabs, scatter and normalize
        output.scatter( data, norm(), make_uint2(1, 2 * (global_dims.x / 2 + 1)) );
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
        if ( scratch   == nullptr ) scratch   = kspace_buffer();
        if ( scratch_y == nullptr ) scratch_y = kspace_buffer();
        if ( scratch_z == nullptr ) scratch_z = kspace_buffer();

        // Copy input data to scratch
        detail::copy( scratch,   input.x(), input.buffer_size() );
        detail::copy( scratch_y, input.y(), input.buffer_size() );
        detail::copy( scratch_z, input.z(), input.buffer_size() );

        float * data_x = reinterpret_cast<float *>( scratch   );
        float * data_y = reinterpret_cast<float *>( scratch_y );
        float * data_z = reinterpret_cast<float *>( scratch_z );

        // Perform in-place transforms of scratch buffers
        fftwf_mpi_execute_dft_c2r( plan, reinterpret_cast<fftwf_complex *>( scratch   ), data_x );
        fftwf_mpi_execute_dft_c2r( plan, reinterpret_cast<fftwf_complex *>( scratch_y ), data_y );
        fftwf_mpi_execute_dft_c2r( plan, reinterpret_cast<fftwf_complex *>( scratch_z ), data_z );

        // Scatter data into output grid and normalize
        output.scatter( data_x, data_y, data_z, norm(), make_uint2(1, 2 * (global_dims.x / 2 + 1)) );
    }
};

/**
 * @brief Copy a local k-space buffer to a new buffer, transposing it to a
 *        (kx, ky) layout and rotating ky so that negative modes come first
 * 
 * @note The output layout matches the cuFFTMp (GPU) version so that files are
 *       identical: x is kx, y is ky. k-space data is stored in FFT order along
 *       ky (see grid::fft::k()). Rotating by ceil(ny/2) rows gives a monotonic
 *       ky axis running from -floor(ny/2) to floor((ny-1)/2) modes. This
 *       requires the full ky extent to be local, which holds for the FFTW
 *       transposed layout (k-space is split along kx only)
 * 
 * @param data          k-space buffer (local data, transposed layout)
 * @param local_dims    Local grid dimensions (transposed, x = ky, y = kx)
 * @param global_dims   Global grid dimensions (transposed, x = ky, y = kx)
 * @return              Buffer with transposed and rotated data, of size
 *                      local_dims.x * local_dims.y; free it with memory::free()
 */
inline complex_t * kspace_to_host( const complex_t * const __restrict__ data,
    const uint2 local_dims, const uint2 global_dims ) {

    if ( local_dims.x != global_dims.x ) {
        mpi::fatal( "kspace_to_host(): the full ky extent must be local to each rank" );
    }

    /// @brief Number of ky modes (full extent)
    const std::size_t nky = local_dims.x;
    /// @brief Number of local kx modes
    const std::size_t nkx = local_dims.y;

    // Avoid zero-size allocations on ranks with no local kx modes
    complex_t * __restrict__ h_buffer = memory::malloc<complex_t>( std::max<std::size_t>( nkx * nky, 1 ) );

    // out[j][i] = in[i][(j + yroll) % ny]
    const std::size_t yroll = ( nky + 1 ) / 2;     // ceil(ny/2)

    #pragma omp parallel for
    for( std::size_t j = 0; j < nky; j++ ) {
        const std::size_t iky = ( j + yroll ) % nky;
        for( std::size_t i = 0; i < nkx; i++ ) {
            h_buffer[ j * nkx + i ] = data[ i * nky + iky ];
        }
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
    default:
        mpi::fatal( "kspace_save(): invalid field component" );
    }
    return nullptr;
}

namespace detail {

/**
 * @brief Saves k-space data with full metadata, correcting the layout
 * 
 * @param data          Local k-space data (transposed layout)
 * @param global_dims   Global grid dimensions (transposed, x = ky, y = kx)
 * @param local_dims    Local grid dimensions (transposed)
 * @param local_start   Local grid start position (transposed)
 * @param comm          MPI communicator
 * @param info          Grid metadata
 * @param iter          Iteration metadata
 * @param path          File path
 */
inline void kspace_save( const complex_t * data, 
    const uint2 global_dims, const uint2 local_dims, const uint2 local_start,
    const MPI_Comm comm,
    zdf::grid_info &info, const zdf::iteration &iter, const std::string &path ) {

    complex_t * h_buffer = kspace_to_host( data, local_dims, global_dims );

    // Fill in global grid dimensions (x = kx, y = ky)
    info.ndims = 2;
    info.count[0] = global_dims.y;
    info.count[1] = global_dims.x;

    // Information on local chunk of grid data
    zdf::chunk chunk;
    chunk.data = h_buffer;
    chunk.count[0] = local_dims.y;
    chunk.count[1] = local_dims.x;
    chunk.start[0] = local_start.y;
    chunk.start[1] = local_start.x;
    chunk.stride[0] = chunk.stride[1] = 1;

    zdf::save_grid<complex_t>( chunk, info, iter, path, comm );

    memory::free( h_buffer );
}

/**
 * @brief Saves k-space data (no metadata), correcting the layout
 * 
 * @param data          Local k-space data (transposed layout)
 * @param global_dims   Global grid dimensions (transposed, x = ky, y = kx)
 * @param local_dims    Local grid dimensions (transposed)
 * @param local_start   Local grid start position (transposed)
 * @param comm          MPI communicator
 * @param name          Grid name
 * @param filename      Output file name (includes path)
 */
inline void kspace_save( const complex_t * data, 
    const uint2 global_dims, const uint2 local_dims, const uint2 local_start,
    const MPI_Comm comm,
    const std::string & name, const std::string & filename ) {

    // x = kx, y = ky
    uint64_t global[2] = { global_dims.y, global_dims.x };
    uint64_t start[2]  = { local_start.y, local_start.x };
    uint64_t local[2]  = { local_dims.y,  local_dims.x };

    complex_t * h_buffer = kspace_to_host( data, local_dims, global_dims );

    zdf::save_grid( h_buffer, 2, global, start, local, name, filename, comm );

    memory::free( h_buffer );
}

}

/**
 * @brief Saves a k-space grid with full metadata, correcting the layout
 * 
 * @note Collective. Data is saved in (kx, ky) order, matching the GPU
 *       version. The ky axis metadata in `info` must describe the rotated
 *       layout, i.e. run from -floor(ny/2)*dk.y to floor((ny-1)/2)*dk.y
 * 
 * @param cf        k-space grid
 * @param info      Grid metadata
 * @param iter      Iteration metadata
 * @param path      File path
 */
inline void kspace_save( const grid::flat<complex_t> & cf, zdf::grid_info &info, const zdf::iteration &iter, const std::string &path ) {
    detail::kspace_save( cf.data(), 
        cf.get_global_dims(), cf.get_local_dims(), cf.get_local_start(), cf.get_part().get_comm(),
        info, iter, path );
}

/**
 * @brief Saves one component of a vector k-space grid with full metadata,
 *        correcting the layout
 * 
 * @note Collective. Data is saved in (kx, ky) order, matching the GPU
 *       version. The ky axis metadata in `info` must describe the rotated
 *       layout, i.e. run from -floor(ny/2)*dk.y to floor((ny-1)/2)*dk.y
 * 
 * @param cf        Vector k-space grid
 * @param fc        Field component to save
 * @param info      Grid metadata
 * @param iter      Iteration metadata
 * @param path      File path
 */
inline void kspace_save( const grid::flat3<complex_t> & cf, const fcomp::cart fc, zdf::grid_info &info, const zdf::iteration &iter, const std::string &path ) {
    detail::kspace_save( kspace_component( cf, fc ), 
        cf.get_global_dims(), cf.get_local_dims(), cf.get_local_start(), cf.get_part().get_comm(),
        info, iter, path );
}

/**
 * @brief Saves a k-space grid (no metadata), correcting the layout
 * 
 * @note Collective. Data is saved in (kx, ky) order, matching the GPU version
 * 
 * @param cf        k-space grid
 * @param filename  Output file name (includes path)
 */
inline void kspace_save( const grid::flat<complex_t> & cf, const std::string & filename ) {
    detail::kspace_save( cf.data(), 
        cf.get_global_dims(), cf.get_local_dims(), cf.get_local_start(), cf.get_part().get_comm(),
        cf.name, filename );
}

/**
 * @brief Saves one component of a vector k-space grid (no metadata),
 *        correcting the layout
 * 
 * @note Collective. Data is saved in (kx, ky) order, matching the GPU version
 * 
 * @param cf        Vector k-space grid
 * @param fc        Field component to save
 * @param filename  Output file name (includes path)
 */
inline void kspace_save( const grid::flat3<complex_t> & cf, const fcomp::cart fc, const std::string & filename ) {
    detail::kspace_save( kspace_component( cf, fc ), 
        cf.get_global_dims(), cf.get_local_dims(), cf.get_local_start(), cf.get_part().get_comm(),
        cf.name, filename );
}

} // namespace fft

} // namespace grid
