#pragma once

#include <cstddef>
#include <fftw3.h>
#include <fftw3-mpi.h>
#include <mpi.h>
#include <complex>

#include "tiled.hpp"
#include "flat.hpp"

#include "vec3_tiled.hpp"
#include "flat3.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace grid {

namespace fft {

/**
* @brief Initialize FFTW library
* 
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
 */
inline void cleanup( ) {
    #ifdef _OPENMP
    fftwf_cleanup_threads();
    #endif
    fftwf_mpi_cleanup();
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

class r2c_plan {

    private:

    /// @brief FFTW Real to complex plan
    fftwf_plan plan = nullptr;

    /// @brief Real space global data dimensions
    const uint2 dims;

    /// @brief Local dimensions of real space grid
    int2 local_dims;

    public:

    /**
     * @brief Construct a new r2c plan object
     * 
     * @param dims      Global dimensions of real space data
     * @param comm      MPI Communicator
     * @param flag      Planning-rigor flag for creating the plan, defaults to
     *                  plan::estimate
     */
    r2c_plan( const uint2 dims, const MPI_Comm comm, 
            plan::rigor flag = plan::estimate ) :
        dims( dims ){

        int comm_size;
        if ( MPI_Comm_size( comm, & comm_size ) != MPI_SUCCESS ) {
            std::cerr << "Unable to get MPI communicator size\n";
            MPI_Abort( comm, 1 );
        }

        if ( dims.y % comm_size != 0 ) {
            std::cerr << "Y dimension does is not evenly divisible by the number of processes\n";
            MPI_Abort( comm, 1 );
        }

        local_dims = make_int2( dims.x, dims.y / comm_size );

        const ptrdiff_t nx = dims.x;
        const ptrdiff_t ny = dims.y;
        const ptrdiff_t nxc = nx/2 + 1;

        ptrdiff_t local_n0,  local_0_start;
        ptrdiff_t local_n1c, local_1_start;

        const ptrdiff_t local_size = fftwf_mpi_local_size_2d_transposed(
            ny, nxc, comm,
            &local_n0,  &local_0_start,
            &local_n1c, &local_1_start
        );

        // This is used only to create the plan
        float * data = fftwf_alloc_real( 2 * local_size );
        
        // Plan for an in-place transform
        fftwf_complex * data_c = reinterpret_cast<fftwf_complex *>( data );

        #ifdef _OPENMP
        // See note in Makefile for macOS homebrew builds
        fftwf_plan_with_nthreads( omp_get_max_threads() );
        #endif

        plan = fftwf_mpi_plan_dft_r2c_2d(
            ny, nx,
            data, data_c,
            comm,
            flag | FFTW_MPI_TRANSPOSED_OUT
        );

        // Free the memory used for creating the plan
        fftwf_free( data );
    }

    /**
     * @brief Construct an FFT plan object from a `grid::tiled<float>` object
     * 
     * @param source    Source object. Only global dimensions and MPI Comm are
     *                  used.
     * @param flag      Planning-rigor flag for creating the plan, defaults to
     *                  `plan::estimate`
     */
    r2c_plan( const grid::tiled<float> & source, plan::rigor flag = plan::estimate ):
        r2c_plan( source.get_global_dims(), source.get_part().get_comm(), flag ) {
    }

    /**
     * @brief Construct an FFT plan object from a grid::tiled<float> object
     *
     * @note If flag is different from plan::estimate the source object data
     *       will be overwritten
     * 
     * @param dims      Global dimensions of real space data
     * @param comm      MPI Communicator
     */
    r2c_plan( const grid::vec3_tiled<float> & source, plan::rigor flag = plan::estimate ):
        r2c_plan( source.get_global_dims(), source.get_part().get_comm(), flag ) {
    }

    /**
     * @brief Destroy the plan object
     * 
     */
    ~r2c_plan() {
        fftwf_destroy_plan( plan );
    }

    /**
     * @brief Stream extraction
     * 
     * @param os    Output stream
     * @param obj   r2c_plan object
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, r2c_plan & obj) {
        os << "FFT r2c plan, dims: " << obj.dims;
        return os;
    }

    /**
     * @brief Perform real to complex transform
     * 
     * @param output        Ouput grid (complex)
     * @param input         Input grid (real)
     * @param in_ystride    Input ystride
     */
    void transform( grid::flat<std::complex<float>> & output, const grid::tiled<float> & input ) {

        float * data           = reinterpret_cast<float *>( output.data() );
        fftwf_complex * data_c = reinterpret_cast<fftwf_complex *>( data );

        // Gather input data 
        input.gather( data, make_uint2(1, 2 * (dims.x / 2 + 1)) );

        // Perform the transform in-place
        fftwf_mpi_execute_dft_r2c( plan, data, data_c );
    }
    
        /**
     * @brief Perform real to complex transform
     * 
     * @param output        Ouput grid (complex)
     * @param input         Input grid (real)
     * @param in_ystride    Input ystride
     */
    void transform( grid::flat3<std::complex<float>> & output, const grid::vec3_tiled<float> & input ) {

        auto dims = input.get_global_dims();
        float * x_data = reinterpret_cast<float*>(output.x());
        float * y_data = reinterpret_cast<float*>(output.y());
        float * z_data = reinterpret_cast<float*>(output.z());
        
        input.gather( x_data, y_data, z_data, make_uint2(1, 2 * (dims.x / 2 + 1)) );

        // Perform the transforms in-place
        fftwf_mpi_execute_dft_r2c( plan, x_data, reinterpret_cast<fftwf_complex*>(x_data) );
        fftwf_mpi_execute_dft_r2c( plan, y_data, reinterpret_cast<fftwf_complex*>(y_data) );
        fftwf_mpi_execute_dft_r2c( plan, z_data, reinterpret_cast<fftwf_complex*>(z_data) );
    }

    /**
     * @brief Stream extraction
     * 
     * @param os 
     * @param obj 
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, const r2c_plan& obj) {
        return os << fftwf_sprint_plan( obj.plan );
    }
};


class c2r_plan {
    private:

    /// @brief Complex to real plan
    fftwf_plan plan = nullptr;

    /// @brief Real space global data dimensions
    const uint2 dims;

    /// @brief Local dimensions of real space grid
    int2 local_dims;

    /// @brief Scratch buffer size
    std::size_t scratch_size;

    /// @brief Scratch memory for transform
    float * scratch_x = nullptr;

    /// @brief Additional scratch memory for vec3 transform (x)
    float * scratch_y = nullptr;

    /// @brief Additional scratch memory for vec3 transform (y)
    float * scratch_z = nullptr;

    public:

    /**
     * @brief Construct a new c2r plan object
     * 
     * @param dims      Global dimensions of real space data
     * @param comm      MPI Communicator
     * @param flag      Planning-rigor flag for creating the plan, defaults to
     *                  plan::estimate
     */
    c2r_plan( const uint2 dims, const MPI_Comm comm, 
            plan::rigor flag = plan::estimate ) :
        dims( dims ){

        int comm_size;
        if ( MPI_Comm_size( comm, & comm_size ) != MPI_SUCCESS ) {
            std::cerr << "Unable to get MPI communicator size\n";
            MPI_Abort( comm, 1 );
        }

        if ( dims.y % comm_size != 0 ) {
            std::cerr << "Y dimension does is not evenly divisible by the number of processes\n";
            MPI_Abort( comm, 1 );
        }

        local_dims = make_int2( dims.x, dims.y / comm_size );

        const ptrdiff_t nx = dims.x;
        const ptrdiff_t ny = dims.y;
        const ptrdiff_t nxc = nx/2 + 1;

        ptrdiff_t local_n0,  local_0_start;
        ptrdiff_t local_n1c, local_1_start;

        const ptrdiff_t local_size = fftwf_mpi_local_size_2d_transposed(
            ny, nxc, comm,
            &local_n0,  &local_0_start,
            &local_n1c, &local_1_start
        );

        // Allocate scratch memory
        scratch_size = 2 * local_size;
        scratch_x = fftwf_alloc_real( scratch_size );
        
        // This is used only to create the out of place plan
        fftwf_complex * data_c = fftwf_alloc_complex( local_size );

        #ifdef _OPENMP
        // See note in Makefile for macOS homebrew builds
        fftwf_plan_with_nthreads( omp_get_max_threads() );
        #endif

        plan = fftwf_mpi_plan_dft_c2r_2d(
            ny, nx,
            data_c, scratch_x,
            comm,
            flag | FFTW_PRESERVE_INPUT | FFTW_MPI_TRANSPOSED_IN
        );

        // Free the memory used for creating the plan
        fftwf_free( data_c );
    }

    /**
     * @brief Construct a new c2r plan object
     * 
     * @param dest 
     * @param flag 
     */
    c2r_plan( const grid::tiled<float> & dest, plan::rigor flag = plan::estimate ):
        c2r_plan( dest.get_global_dims(), dest.get_part().get_comm(), flag ) { }

    /**
     * @brief Construct a new c2r plan object
     * 
     * @param dest 
     * @param flag 
     */
    c2r_plan( const grid::vec3_tiled<float> & dest, plan::rigor flag = plan::estimate ):
        c2r_plan( dest.get_global_dims(), dest.get_part().get_comm(), flag ) { }

    /**
     * @brief Destroy the c2r plan object
     * 
     */
    ~c2r_plan( ){
        if ( scratch_x != nullptr ) fftwf_free( scratch_x );
        if ( scratch_y != nullptr ) fftwf_free( scratch_y );
        if ( scratch_z != nullptr ) fftwf_free( scratch_z );

        fftwf_destroy_plan( plan );
    }

    /**
     * @brief Normalization factor
     * 
     * @return float    1/(dims.x * dims.y)
     */
    inline float norm( ) const {
        return 1.f / (dims.x * dims.y);
    }

    /**
     * @brief Stream extraction
     * 
     * @param os 
     * @param obj 
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, const c2r_plan& obj) {
        return os << fftwf_sprint_plan( obj.plan );
    }

    /**
     * @brief Perform complex to real transform
     * 
     * @param output        Output grid (real)
     * @param out_ystride   Output ystride
     * @param input         Input grid (complex)
     */
    void transform( grid::tiled<float> & output, const grid::flat<std::complex<float>> & input ) {

        fftwf_complex * data_c = reinterpret_cast<fftwf_complex *>( input.data() );
        float * data           = reinterpret_cast<float *>( scratch_x );

        // Perform out of place transform from input to scratch
        fftwf_mpi_execute_dft_c2r( plan, data_c, data );

        // Scatter data into output grid and normalize
        output.scatter( data, norm(), make_uint2(1, 2 * (dims.x / 2 + 1)) );
    }

    /**
     * @brief Perform complex to real transform
     * 
     * @param output        Output grid (real)
     * @param out_ystride   Output ystride
     * @param input         Input grid (complex)
     */
    void transform( grid::vec3_tiled<float> & output, const grid::flat3<std::complex<float>> & input ) {

        fftwf_complex * data_x_c = reinterpret_cast<fftwf_complex *>( input.x() );
        fftwf_complex * data_y_c = reinterpret_cast<fftwf_complex *>( input.y() );
        fftwf_complex * data_z_c = reinterpret_cast<fftwf_complex *>( input.z() );

        // If y and z scratch buffers have not been allocated yet, do so now
        if ( scratch_y == nullptr ) scratch_y = fftwf_alloc_real( 2 * scratch_size );
        if ( scratch_z == nullptr ) scratch_z = fftwf_alloc_real( 2 * scratch_size );

        float * data_x = reinterpret_cast<float *>( scratch_x );
        float * data_y = reinterpret_cast<float *>( scratch_y );
        float * data_z = reinterpret_cast<float *>( scratch_z );
        
        // Perform out of place transform from input to scratch
        fftwf_mpi_execute_dft_c2r( plan, data_x_c, data_x );
        fftwf_mpi_execute_dft_c2r( plan, data_y_c, data_y );
        fftwf_mpi_execute_dft_c2r( plan, data_z_c, data_z );

        // Scatter data into output grid and normalize
        output.scatter( data_x, data_y, data_z, norm(), make_uint2(1, 2 * (dims.x / 2 + 1)) );
    }

};

/**
 * @brief Global dimension of complex grid for r2c and c2r transposed transforms
 * 
 * @param global_dims   Global dimensions of real grid (not transposed)
 * @return int2 
 */
inline uint2 global_kdims_transposed( uint2 global_dims ) {
    return make_uint2( global_dims.y, global_dims.x/2 + 1 );
}

/**
 * @brief Local dimension of complex grid for r2c and c2r transposed transforms
 * 
 * @note The function also computes dimensions and start position of the local
 *       complex grid
 * 
 * @param global_dims   Global dimensions of real grid (not transposed)
 * @param local_kdims   Local dimensions of complex grid
 * @param local_kstart  Local start positions in parallel domain of complex grid
 * @return size_t 
 */
inline size_t local_kdims_tranposed( 
    uint2 global_dims, MPI_Comm comm,
    uint2 & local_kdims, uint2 & local_kstart ) {

    const ptrdiff_t nxc = global_dims.x/2 + 1;
    ptrdiff_t local_n0,  local_0_start;
    ptrdiff_t local_n1c, local_1_start;

    size_t local_size = fftwf_mpi_local_size_2d_transposed(
            global_dims.y, nxc, comm,
            &local_n0,  &local_0_start,
            &local_n1c, &local_1_start
        );

    local_kdims  = make_uint2( global_dims.y, local_n1c );
    local_kstart = make_uint2( 0, local_1_start );

    return local_size;
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
 * @brief Creates a grid::flat<std::complex<float>> from the parameters of the
 *        input grid. This grid can be used in r2c and c2r operations with the
 *        input grid (or others with similar dimensions)
 * 
 * @param in    Grid object describing the real data
 * @return grid::flat<std::complex<float>>
 */
inline grid::flat<std::complex<float>> complex_grid( const grid::tiled<float>& in ) {

    auto global_dims = in.get_global_dims();

    // Ouput grid global dims
    const uint2 out_global_dims = fft::global_kdims_transposed( global_dims );
    
    // Output grid local dims and start position
    uint2 out_local_dims, out_local_start;
    fft::local_kdims_tranposed( 
        global_dims, 
        in.get_part().get_comm(), 
        out_local_dims, 
        out_local_start );

    return grid::flat<std::complex<float>> ( 
        out_global_dims, out_local_dims, out_local_start,
        in.get_part()
    );
}

/**
 * @brief Creates a grid::flat3<std::complex<float>> from the parameters of the
 *        input grid. This grid can be used in r2c and c2r operations with the
 *        input grid (or others with similar dimensions)
 * 
 * @param in    Grid object describing the real data
 * @return grid::flat<std::complex<float>>
 */
inline grid::flat3<std::complex<float>> complex_grid( const grid::vec3_tiled<float>& in ) {

    auto global_dims = in.get_global_dims();

    // Ouput grid global dims
    const uint2 out_global_dims = fft::global_kdims_transposed( global_dims );
    
    // Output grid local dims and start position
    uint2 out_local_dims, out_local_start;
    size_t out_local_size = fft::local_kdims_tranposed( 
        global_dims, 
        in.get_part().get_comm(), 
        out_local_dims, 
        out_local_start );

    return grid::flat3<std::complex<float>> ( 
        out_global_dims, out_local_dims, out_local_start,
        in.get_part(), out_local_size
    );
}

inline grid::flat3<std::complex<float>>* new_complex_grid( const grid::vec3_tiled<float>& in ) {
    return new grid::flat3<std::complex<float>>( complex_grid( in ) );
}

}

}
