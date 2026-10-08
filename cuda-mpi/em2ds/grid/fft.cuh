#pragma once

#include <iostream>
#include <cstddef>
#include <string>
#include <complex>
#include <memory>

#include "device_types.h"
#include "tiled.cuh"
#include "flat.cuh"

#include "tiled_vec3.cuh"
#include "flat3.cuh"

#include <cufftmp/cufftMp.h>

#include <nvshmem.h>
#include <nvshmemx.h>

#include "../core/vec_types.cuh"
#include "../parallel/parallel.hpp"
#include "../grid/tiled.cuh"
#include "../grid/flat.cuh"

#include "../util/complex.hpp"

namespace grid {

namespace fft {

using complex_t = util::complex64;

static_assert( sizeof( complex_t ) == sizeof( cufftComplex ),
    "complex_t must be layout compatible with cufftComplex" );


/**
 * @brief kspace grid layout
 *
 * @note cuFFTMp does not used a transposed layout (unlike FFTW)
 */
inline constexpr bool kspace_transposed = false;

/**
 * @brief Returns k-space coordinate from global k-space grid coordinates
 * 
 * @param gix       Global k-space grid coordinate
 * @param gdims     Global k-space grid dimensions
 * @param dk        K-space cell size
 * @return float2 
 */
__host__ __device__
inline float2 k( const int2 gix, const uint2 gdims, const float2 dk ) {
    static_assert( kspace_transposed == false , "this only works for normal k-space layout");
    return {
        gix.x * dk.x,
        (( 2 * gix.y < int(gdims.y) ) ? gix.y : ( gix.y - static_cast<int>(gdims.y) ) ) * dk.y
    };
}

namespace detail {

/// @brief Communicator used to initialize NVSHMEM (must outlive NVSHMEM)
inline MPI_Comm nvshmem_comm = MPI_COMM_NULL;

/**
 * @brief Returns a string containing the name of an error code in the enum. 
 *
 * If the error code is not recognized, “unrecognized error code” is returned.
 * 
 * @param error             Error code to convert to string
 * @return const char*      pointer to a NULL-terminated string
 */
inline const char* getCufftErrStr(cufftResult error) {
  switch (error) {
    case CUFFT_SUCCESS:
      return "The cuFFT operation was successful.";
    case CUFFT_INVALID_PLAN:
      return "cuFFT was passed an invalid plan handle.";
    case CUFFT_ALLOC_FAILED:
      return "cuFFT failed to allocate GPU or CPU memory.";
    case CUFFT_INVALID_VALUE:
      return "User specified an invalid pointer or parameter.";
    case CUFFT_INTERNAL_ERROR:
      return "Driver or internal cuFFT library error.";
    case CUFFT_EXEC_FAILED:
      return "Failed to execute an FFT on the GPU.";
    case CUFFT_SETUP_FAILED:
      return "The cuFFT library failed to initialize.";
    case CUFFT_INVALID_SIZE:
      return "User specified an invalid transform size.";
    case CUFFT_INVALID_DEVICE:
      return "Execution of a plan was on different GPU than plan creation.";
    case CUFFT_NO_WORKSPACE:
      return "No workspace has been provided prior to plan execution.";
    case CUFFT_NOT_IMPLEMENTED:
      return "Function does not implement functionality for parameters given.";
    case CUFFT_NOT_SUPPORTED:
      return "Operation is not supported for parameters given.";
    default:
      return "Unknown error.";
  }
}

}

// The CUFFT_CHECK macro definition
#define CUFFT_CHECK(call)                                                    \
    do {                                                                     \
        cufftResult status = (call);                                         \
        if (status != CUFFT_SUCCESS) {                                       \
            mpi::fatal( std::string( #call ) + " failed with " + ::grid::fft::detail::getCufftErrStr( status ) ); \
        }                                                                    \
    } while (0)

/**
 * @brief Initialize NVSHMEM on `comm`. Call after cudaSetDevice() and before
 *        creating any symmetric grid or cuFFTMp plan.
 */
inline void init( MPI_Comm comm = MPI_COMM_WORLD ) {
    detail::nvshmem_comm = comm;
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    attr.mpi_comm = static_cast<void *>( & detail::nvshmem_comm );

    nvshmemx_init_attr( NVSHMEMX_INIT_WITH_MPI_COMM, & attr );
}

/**
 * @brief Finalize NVSHMEM. All symmetric grids and plans must be destroyed.
 */
inline void cleanup( ) {
    nvshmem_finalize();
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
* @note The k-space grid is NOT transposed: x is kx (nx/2+1 values), y is ky
*       (ny values). cuFFTMp's built-in shuffled format distributes it along
*       kx, each rank holding a natural-order [ky][kx_local] block.
* @return uint2 
*/
inline uint2 global_kdims( const uint2 global_dims ) {
    return make_uint2( global_dims.x/2 + 1, global_dims.y );
}

/**
 * @brief Get local dimensions for a 2D R2C/C2R transform
 *
 * @note Real space is split along y; k-space (x = kx, y = ky) is split along
 *       kx, following the cuFFTMp built-in slab distributions
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
    if ( part.dims.x != 1 ) {
        mpi::fatal("FFT operations require that the domain is only"
                    " partitioned along the y direction" );
    }
    
    ///@brief number of parallel nodes (along y)
    int size = part.dims.y;
    ///@brief node rank (along y)
    int rank = part.get_coords().y;

    // cuFFTMp assigns slabs by communicator rank, which must match the y coordinate
    int comm_rank;
    MPI_Comm_rank( part.get_comm(), & comm_rank );
    if ( comm_rank != rank ) {
        mpi::fatal( "FFT: communicator rank does not match partition y coordinate" );
    }

    ///@brief Number of complex elements along x (and kx)
    unsigned int nxc = global_dims.x / 2 + 1;

    /**
    * @brief Dimension of k-space grid (not transposed)
    * @note x component corresponds to kx, y component to ky
    */
    uint2 global_kdims { nxc, global_dims.y };

    if ( global_dims.y < static_cast<unsigned>(size) || 
            global_kdims.x < static_cast<unsigned>(size) ) {
        if ( rank == 0 ) std::cerr << "cuFFTMp requires at least one row / kx mode per rank\n";
        mpi::abort( 1 );
    }

    /**
    * @brief cuFFTMp built-in slab distribution of n items over P ranks
    * 
    */
    auto slab = [ size, rank ] ( const int n, unsigned & count, unsigned & start ) {
        const int q = n / size;
        const int r = n % size;
        count = q + ( rank < r ? 1 : 0 );
        start = rank * q + std::min( rank, r );
    };

    // Spatial coordinate partition (partition along y)
    local_dims.x = global_dims.x;
    local_start.x = 0;
    slab( global_dims.y, local_dims.y, local_start.y );

    // k-space partition (along kx, i.e. along x)
    slab( global_kdims.x, local_kdims.x, local_kstart.x );
    local_kdims.y = global_kdims.y;
    local_kstart.y = 0;

    // Minimum number of complex elements to allocate
    return std::max(
            static_cast<std::size_t>( local_dims.y )  * nxc,            // padded real data
            static_cast<std::size_t>( local_kdims.x ) * local_kdims.y   // complex data
        );
}

class fft_plan {

    protected:

    /// @brief cufftMp Real to complex plan
    cufftHandle plan = 0;

    /// @brief Size (in elements) of k-space grid
    
    /**
     * @brief Size (in elements) of k-space grid (complex values)
     *
     * @note NVSHMEM requires the same allocation size on every PE, so the
     *       size is the maximum local size over all ranks
     * 
     */
    std::size_t capacity = 0;

    /**
     * @brief Allocate k-space buffer for use with transform
     *
     * @note The buffer will have space for `capacity` complex values. This value
     *       is initialized in the constructor and must be the same on all PEs.
     * 
     * @return complex_t* 
     */
    complex_t * kspace_buffer() {
        const std::size_t bytes = capacity * sizeof( complex_t );
        complex_t * ptr = static_cast<complex_t *>( nvshmem_malloc( bytes ) );
        if ( ptr == nullptr ) {
            mpi::fatal( "nvshmem_malloc failed for " + 
                    std::to_string( bytes ) +
                    " bytes, consider increasing NVSHMEM_SYMMETRIC_SIZE" );
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

    /// @brief Internal workspace used
    std::size_t workspace;

    /// @brief MPI communicator (cuFFTMp keeps a pointer to it, must outlive the plan)
    MPI_Comm comm;

    public:

    /// @brief Real space global data dimensions
    const uint2 global_dims;

    /// @brief Parallel partition
    const mpi::cart2d & part;

    /**
     * @brief Transposed parallel partition for k-space grids
     *
     * @note k-space grids created by this plan keep a reference to it, so
     *       they must be destroyed before the plan
     */
    std::unique_ptr<mpi::cart2d> part_transp;

    /**
     * @brief Construct a new fft plan object
     * 
     * @param global_dims   Global real space dimensions
     * @param part          Parallel partition
     * @param type          Transform type (R2C/C2R)
     */
    fft_plan( const uint2 global_dims, const mpi::cart2d & part, const cufftType type ) :
        global_dims( global_dims ), part( part ) {
    
        // Initialize cuFFTMp plan
        comm = part.get_comm();
        // C order: slowest dimension (y) first
        
        // Create the plan
        CUFFT_CHECK( cufftCreate(&plan) );
        // Attach the MPI communicator to the plan
        CUFFT_CHECK( cufftMpAttachComm( plan, CUFFT_COMM_MPI, &comm));
        
        // Set the stream
        // CUFFT_CHECK(cufftSetStream( plan, stream));

        // Set default subformats
        CUFFT_CHECK(cufftXtSetSubformatDefault(plan, CUFFT_XT_FORMAT_INPLACE, CUFFT_XT_FORMAT_INPLACE_SHUFFLED));

        // Make the 2D transform plan
        CUFFT_CHECK(cufftMakePlan2d(plan, global_dims.y, global_dims.x, type, &workspace) );

        // Initialize dimensions of local k-space grid
        capacity = local_size_2d( global_dims, part,
            local_dims, local_start, local_kdims, local_kstart );

        // For NVSHMEM, capacity must be the maximum of all nodes
        part.allreduce( &capacity, 1, mpi::max );

        // This grid will use a tranposed parallel partition
        part_transp = std::make_unique<mpi::cart2d>( part, mpi::cart2d::transpose_t {} );

    }

    /**
     * @brief Construct a new fft plan object from a tiled grid source
     * 
     * @param source    Tiled grid source
     * @param type      Transform type (R2C/C2R)
     */
    fft_plan( const grid::tiled<float> & source, const cufftType type ):
        fft_plan( source.get_global_dims(), source.get_part(), type ) {

        // Verify that the FFT partition matches the source partition
        uint2 in_local_dims = source.get_local_dims();
        if (( in_local_dims.x != local_dims.x ) ||
            ( in_local_dims.y != local_dims.y )) {
            mpi::cout << "Source grid local dimensions (" << in_local_dims
                      << ") and FFT plan local dimensions (" << local_dims
                      << ") don't match. Check global dimensions and parallel partition"
                         ", aborting.\n";
            mpi::abort(1);
        }

        uint2 in_local_start = source.get_local_tile_start() * source.tile_dims;
        if (( in_local_start.x != local_start.x ) ||
            ( in_local_start.y != local_start.y )) {
            mpi::cout << "Source grid local start positions don't match FFT plan."
                         " Check global dimensions and parallel partition"
                         ", aborting.\n";
            mpi::abort(1);
        }
    }

    /**
     * @brief Construct a new fft plan object from a tiled grid source
     * 
     * @param source    Tiled grid source
     * @param type      Transform type (R2C/C2R)
     */
    fft_plan( const grid::tiled_vec3<float> & source, const cufftType type ):
        fft_plan( source.get_global_dims(), source.get_part(), type ) {

        // Verify that the FFT partition matches the source partition
        uint2 in_local_dims = source.get_local_dims();
        if (( in_local_dims.x != local_dims.x ) ||
            ( in_local_dims.y != local_dims.y )) {
            mpi::cout << "Source grid local dimensions (" << in_local_dims
                      << ") and FFT plan local dimensions (" << local_dims
                      << ") don't match. Check global dimensions and parallel partition"
                         ", aborting.\n";
            mpi::abort(1);
        }

        uint2 in_local_start = source.get_local_tile_start() * source.tile_dims;
        if (( in_local_start.x != local_start.x ) ||
            ( in_local_start.y != local_start.y )) {
            mpi::cout << "Source grid local start positions don't match FFT plan."
                         " Check global dimensions and parallel partition"
                         ", aborting.\n";
            mpi::abort(1);
        }
    }


    fft_plan( const fft_plan & ) = delete;
    fft_plan & operator=( const fft_plan & ) = delete;

    /**
     * @brief Destroy the plan object
     * 
     */
    ~fft_plan() {
        cufftDestroy( plan );
    }
    
    /**
     * @brief Set stream for the transforms
     * 
     * @param stream    CUDA stream
     */
    void set_stream( const cudaStream_t stream ) {
        auto res = cufftSetStream( plan, stream );
        if (res != CUFFT_SUCCESS ) {
            mpi::fatal( "cufftSetStream() failed");
        }
    }

    inline uint2 get_global_dims() const noexcept {
        return global_dims;
    }

    /**
    * @brief Global dimension of complex grid (x = kx, y = ky)
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
     * @warning Collective (symmetric allocation). The grid references this
     *          plan's transposed partition, so it must not outlive the plan.
     * 
     * @return grid::flat<complex_t> 
     */
    inline grid::flat<complex_t> kspace_grid( ) {

        // Build buffer destructor
        auto release = [ ] (complex_t * p ){ 
            nvshmem_free( p );
        };

        // Build flat grid object
        return grid::flat<complex_t> ( 
            global_kdims( global_dims ), local_kdims, local_kstart, *part_transp, 
            kspace_buffer(), capacity, release
        );
    }

    /**
     * @brief Allocate a new flat grid for the k-space data
     * 
     * @return grid::flat<complex_t>* 
     */
    inline grid::flat<complex_t>* new_kspace_grid () {
        return new grid::flat<complex_t>( kspace_grid() );
    }

    /**
     * @brief Create a flat3 grid for the k-space data
     * 
     * @warning Collective (symmetric allocation). The grid references this
     *          plan's transposed partition, so it must not outlive the plan.
     * 
     * @return grid::flat3<complex_t> 
     */
    inline grid::flat3<complex_t> kspace_grid3( ) {

        // Build buffer destructor
        auto release = [ ] (complex_t * p ){ 
            nvshmem_free( p );
        };

        // Build flat grid object
        return grid::flat3<complex_t> ( 
            global_kdims( global_dims ), local_kdims, local_kstart, *part_transp, 
            kspace_buffer(), kspace_buffer(), kspace_buffer(),
            capacity, release
        );
    }

    /**
     * @brief Allocate a new flat grid for the k-space data
     * 
     * @return grid::flat<complex_t>* 
     */
    inline grid::flat3<complex_t>* new_kspace_grid3() {
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
     */
    r2c_plan( const uint2 global_dims, const mpi::cart2d & part ) :
        fft_plan( global_dims, part, CUFFT_R2C ) { }

    /**
     * @brief Construct an FFT plan object from a `grid::tiled<float>` object
     * 
     * @param source    Source object. Only global dimensions and MPI Comm are
     *                  used.
     * @param flag      Planning-rigor flag for creating the plan, defaults to
     *                  `plan::estimate`
     */
    r2c_plan( const grid::tiled<float> & source ):
        fft_plan( source, CUFFT_R2C ) { }

    r2c_plan( const grid::tiled_vec3<float> & source ):
        fft_plan( source, CUFFT_R2C ) { }

    /**
     * @brief Stream extraction
     * 
     * @param os    Output stream
     * @param obj   r2c_plan object
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, r2c_plan & obj) {
        os << "FFT r2c plan, dims: " << obj.global_dims;
        return os;
    }

    /**
     * @brief Perform real to complex transform
     * 
     * @param output        Ouput grid (complex)
     * @param input         Input grid (real)
     * @param in_ystride    Input ystride
     */
    void transform( grid::flat<complex_t> & output, const grid::tiled<float> & input ) {

        float * data          = reinterpret_cast<float *>( output.data() );

        // Gather input data 
        input.gather( data, make_uint2(1, 2 * (global_dims.x / 2 + 1)) );
        
        CUFFT_CHECK( 
            cufftExecR2C( plan,
            reinterpret_cast<cufftReal *>( data ),
            reinterpret_cast<cufftComplex *>( data ) ) 
        );
    }
    
    /**
     * @brief Perform real to complex transform
     * 
     * @param output        Ouput grid (complex)
     * @param input         Input grid (real)
     * @param in_ystride    Input ystride
     */
    void transform( grid::flat3<complex_t> & output, const grid::tiled_vec3<float> & input ) {

        auto dims = input.get_global_dims();
        float * x_data = reinterpret_cast<float *>(output.x());
        float * y_data = reinterpret_cast<float *>(output.y());
        float * z_data = reinterpret_cast<float *>(output.z());
        
        input.gather( x_data, y_data, z_data, make_uint2(1, 2 * (dims.x / 2 + 1)) );

        // Perform the transforms in-place
        CUFFT_CHECK( cufftExecR2C( plan, reinterpret_cast<cufftReal *>(x_data), reinterpret_cast<cufftComplex*>(x_data) ) );
        CUFFT_CHECK( cufftExecR2C( plan, reinterpret_cast<cufftReal *>(y_data), reinterpret_cast<cufftComplex*>(y_data) ) );
        CUFFT_CHECK( cufftExecR2C( plan, reinterpret_cast<cufftReal *>(z_data), reinterpret_cast<cufftComplex*>(z_data) ) );
    }

};

/**
 * @brief Complex to real FFT plan
 * 
 */
class c2r_plan : public fft_plan {
    private:

    /// @brief Scratch buffer (symmetric), so transform() preserves its input
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
     */
    c2r_plan( const uint2 global_dims, const mpi::cart2d & part ) :
        fft_plan( global_dims, part, CUFFT_C2R ) { }

    /**
     * @brief Construct a new c2r plan object
     * 
     * @param dest 
     * @param flag 
     */
    c2r_plan( const grid::tiled<float> & dest ):
        fft_plan( dest, CUFFT_C2R ) { }


    c2r_plan( const grid::tiled_vec3<float> & dest ):
        fft_plan( dest, CUFFT_C2R ) { }

    /**
     * @brief Destroy the c2r plan object
     * 
     */
    ~c2r_plan( ){
        // Destroy internal scratch space
        if ( scratch   != nullptr ) nvshmem_free( scratch  );
        if ( scratch_y != nullptr ) nvshmem_free( scratch_y );
        if ( scratch_z != nullptr ) nvshmem_free( scratch_z );
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
     * @param output        Output grid (real)
     * @param input         Input grid (complex)
     */
    void transform( grid::tiled<float> & output, const grid::flat<complex_t> & input ) {

        // If scratch buffers have not been allocated yet, do so now
        if ( scratch   == nullptr ) scratch   = kspace_buffer();

        // Copy input data to scratch
        gpu::device::memcpy_todevice( 
            reinterpret_cast<complex_t *>( scratch ),
            input.data(),
            input.buffer_size()
        );

        // Perform in-place transform of scratch
        CUFFT_CHECK( cufftExecC2R( plan, reinterpret_cast<cufftComplex *>( scratch ), reinterpret_cast<cufftReal *>( scratch ) ) );
        
        // Buffer now holds padded real-space slabs
        output.scatter( reinterpret_cast<float *>(scratch), 
            norm(), make_uint2(1, 2 * (global_dims.x / 2 + 1)) );
    }


    /**
     * @brief Perform complex to real transform
     * 
     * @param output        Output grid (real)
     * @param out_ystride   Output ystride
     * @param input         Input grid (complex)
     */
    void transform( grid::tiled_vec3<float> & output, const grid::flat3<complex_t> & input ) {

        // If scratch buffers have not been allocated yet, do so now
        if ( scratch   == nullptr ) scratch   = kspace_buffer();
        if ( scratch_y == nullptr ) scratch_y = kspace_buffer();
        if ( scratch_z == nullptr ) scratch_z = kspace_buffer();

        // Copy input data to scratch
        gpu::device::memcpy_todevice( reinterpret_cast<complex_t *>( scratch   ), input.x(), input.buffer_size() );
        gpu::device::memcpy_todevice( reinterpret_cast<complex_t *>( scratch_y ), input.y(), input.buffer_size() );
        gpu::device::memcpy_todevice( reinterpret_cast<complex_t *>( scratch_z ), input.z(), input.buffer_size() );
        
        // Perform in-place transform from input to scratch
        CUFFT_CHECK( cufftExecC2R( plan, reinterpret_cast<cufftComplex *>( scratch   ), reinterpret_cast<cufftReal *>( scratch   ) ) );
        CUFFT_CHECK( cufftExecC2R( plan, reinterpret_cast<cufftComplex *>( scratch_y ), reinterpret_cast<cufftReal *>( scratch_y ) ) );
        CUFFT_CHECK( cufftExecC2R( plan, reinterpret_cast<cufftComplex *>( scratch_z ), reinterpret_cast<cufftReal *>( scratch_z ) ) );

        // Scatter data into output grid and normalize
        output.scatter( 
            reinterpret_cast<float *>( scratch   ),
            reinterpret_cast<float *>( scratch_y ),
            reinterpret_cast<float *>( scratch_z ), 
            norm(), make_uint2(1, 2 * ( global_dims.x / 2 + 1)) );
    }

};


} // namespace fft

} // namespace grid
