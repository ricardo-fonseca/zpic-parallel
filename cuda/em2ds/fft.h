#ifndef FFT_H_
#define FFT_H_

#include "gpu.h"
#include "cufft.h"

#include "grid.h"
#include "vec3grid.h"

#include "basic_grid.h"
#include "basic_grid3.h"

#include "complex.h"

namespace fft {

/// @brief Transform type
enum type  { r2c = 0, c2r, r2c_v3, c2r_v3 };

/**
 * @brief FFT plan
 * 
 */
class plan {
    private:

    /// @brief  FFT plan
    cufftHandle fft_plan;

    /// @brief Temporary real grid (for r2c / c2r transforms)
    basic_grid< float > * tmp_real = nullptr;
    
    /// @brief Temporary complex grid (for c2r transforms)
    basic_grid< std::complex< float > > * tmp_complex = nullptr;

    /// @brief Temporary real grid (for r2c / c2r transforms)
    basic_grid3< float > * tmp_real3 = nullptr;
    
    /// @brief Temporary complex grid (for c2r transforms)
    basic_grid3< std::complex< float > > * tmp_complex3 = nullptr;

    /// @brief FFT type
    const fft::type fft_type;

    /// @brief Data dimensions
    const uint2 dims;

    public:

    /**
     * @brief Construct a new plan object
     * 
     * @param dims      Data dimensions. When using C2R transformations this refers to the
     *                  real data dimensions
     * @param fft_type  Type of transformation
     */
    plan( const uint2 dims, fft::type fft_type ):
        dims( dims ), fft_type( fft_type ) {

        if ( dims.x == 0 || dims.y == 0 ) {
            std::cerr << "Invalid dimensions for FFT plan " << dims << "aborting...\n";
            device::exit(1);
        }

        uint2 fdims{ dims.x/2 + 1, dims.y };
        cufftType_t tmp_type;
        int batch;

        switch( fft_type ) {
            case fft::type::r2c: 
                tmp_type = CUFFT_R2C;
                tmp_real = new basic_grid< float > ( dims );
                batch = 1;
                break;
            case fft::type::c2r: 
                tmp_type = CUFFT_C2R; 
                tmp_real = new basic_grid< float > ( dims );
                tmp_complex = new basic_grid< std::complex<float> > ( fdims );
                batch = 1;
                break;
            case fft::type::r2c_v3: 
                tmp_type = CUFFT_R2C;
                tmp_real3 = new basic_grid3< float > ( dims );
                batch = 3;
                break;
            case fft::type::c2r_v3: 
                tmp_type = CUFFT_C2R; 
                tmp_real3 = new basic_grid3< float > ( dims );
                tmp_complex3 = new basic_grid3< std::complex<float> > ( fdims );
                batch = 3;
                break;
            default:
                std::cerr << "Invalid fft type, aborting\n";
                device::exit(1);
        }

        // cuFFT defines this in the opposite order
        int fft_dims[] = { (int) dims.y, (int) dims.x };
        cufftResult ierr = cufftPlanMany( 
                            & fft_plan, 2, fft_dims, 
                            nullptr, 1, 0, // *inembed, istride, idist
                            nullptr, 1, 0, // *onembed, ostride, odist
                            tmp_type, batch );

        if ( CUFFT_SUCCESS != ierr ) {
            std::cerr << "Unable to create FFT plan, cufftPlanMany() failed with code " << ierr << '\n';
            std::cerr << "aborting...\n";
            device::exit(1);
        }

    }

    /**
     * @brief Destroy the plan object
     * 
     */
    ~plan() {
        cufftDestroy( fft_plan );

        delete( tmp_complex );
        delete( tmp_real );
        delete( tmp_complex3 );
        delete( tmp_real3 );
    }

    /**
     * @brief Dimensions of real data
     * 
     * @return uint2 
     */
    inline uint2 get_dims() { return dims; }

    /**
     * @brief Dimensions of complex data
     * 
     * @return uint2 
     */
    inline uint2 get_fdims() { return uint2{ dims.x/2 + 1, dims.y }; }

    /**
     * @brief Perform a real to complex transform from grid<float> data
     * 
     * @warning This will allocate / deallocate a temporary array
     * 
     * @param real      (in) Tiled grid real data
     * @param complex   (out) Contiguous complex data
     */
    void transform( grid<float>& real, basic_grid< std::complex<float> > & complex ) {
        if ( fft::type::r2c != fft_type ) {
            std::cerr << "FFT was not configured for real to complex tranform, aborting\n";
            device::exit(1);
        }

        // Gather data into basic_grid
        real.gather( tmp_real->d_buffer );

        // Do transform
        cufftExecR2C( fft_plan, 
            reinterpret_cast< cufftReal * > ( tmp_real->d_buffer ),
            reinterpret_cast< cufftComplex * > ( complex.d_buffer )
        );
    }

    /**
     * @brief Perform a complex to real transform to grid<float> data
     * 
     * @warning This will allocate / deallocate a temporary array
     * 
     * @param complex   (in) Contiguous complex data
     * @param real      (out) Tiled grid real data
     */
    void transform( basic_grid< std::complex<float> > & complex, grid<float>& real ) {
        if ( fft::type::c2r != fft_type ) {
            std::cerr << "FFT was not configured for complex to real tranform, aborting\n";
            device::exit(1);
        }

        // Copy original data to temporary buffer
        device::memcpy( tmp_complex -> d_buffer, complex.d_buffer, complex.buffer_size() );

        cufftExecC2R( fft_plan, 
            reinterpret_cast< cufftComplex * > ( tmp_complex -> d_buffer),
            reinterpret_cast< cufftReal * > ( tmp_real -> d_buffer )
        );
        real.scatter( tmp_real -> d_buffer, norm() );
    }

    /**
     * @brief Perform a real to complex transform from vec3grid<float3> data
     * 
     * @warning This will allocate / deallocate a temporary array
     * 
     * @param real      (in) Tiled vec3grid real data
     * @param complex   (out) Contiguous complex data
     */
    void transform( vec3grid<float3>& real3, basic_grid3< std::complex<float> > & complex3 ) {
        if ( fft::type::r2c_v3 != fft_type ) {
            std::cerr << "FFT was not configured for vec3 real to complex tranform, aborting\n";
            device::exit(1);
        }
        
        real3.gather( tmp_real3 -> d_buffer );

        cufftExecR2C( fft_plan, 
            reinterpret_cast< cufftReal * > ( tmp_real3 -> d_buffer ), 
            reinterpret_cast< cufftComplex * > ( complex3.d_buffer )
        );
    }



    /**
     * @brief Perform a complex to real transform to grid<float> data
     * 
     * @warning This will allocate / deallocate a temporary array
     * 
     * @param complex   (in) Contiguous complex data
     * @param real      (out) Tiled grid real data
     */
    void transform( basic_grid3< std::complex<float> > & complex3, vec3grid<float3>& real3 ) {
        
        // From
        // https://docs.nvidia.com/cuda/cufft/index.html#data-layout
        // (...)
        // "Out-of-place complex-to-real FFT will always overwrite input buffer."

        if ( fft::type::c2r_v3 != fft_type ) {
            std::cerr << "FFT was not configured for vec3 complex to real tranform, aborting\n";
            device::exit(1);
        }

        // Copy original data to temporary buffer
        device::memcpy( tmp_complex3 -> d_buffer, complex3.d_buffer, complex3.buffer_size() );
    
        cufftExecC2R( fft_plan, 
            reinterpret_cast< cufftComplex * > ( tmp_complex3 -> d_buffer ), 
            reinterpret_cast< cufftReal * >    ( tmp_real3->d_buffer ) );

        real3.scatter( tmp_real3->d_buffer, norm() );
    }

    /**
     * @brief Stream extraction
     * 
     * @param os 
     * @param obj 
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, fft::plan & obj) {
        os << "FFT plan, dims: " << obj.dims << ", type: ";
        
        switch (obj.fft_type) {
        case r2c:
            os << "real to complex"; break;
        case c2r:
            os << "complex to real"; break;
        case r2c_v3:
            os << "real to complex (vec3)"; break;
        case c2r_v3:
            os << "complex to real (vec3)"; break;
        default:
            os << "unknown"; break;
        }
        return os;
    }

    /**
     * @brief Normalization factor
     * 
     * @return float    1/(dims.x * dims.y)
     */
    inline float norm( ) {
        return 1.f / (dims.x * dims.y);
    }
};

/**
 * @brief Get cell size in k-space
 * 
 * @param box       Box size in real space (not number of cells)
 * @return float2   Cell size in k space
 */
static inline float2 dk( float2 box ) {
    // Value from GLIBC math.h M_PIf
    constexpr float pi = 3.14159265358979323846f;
    return float2{ 2 * pi / box.x, 2 * pi / box.y };
}

/**
 * @brief Dimensions in Fourier space for a real DFT operation
 * 
 * @param dims      Grid dimensions in real space
 * @return uint2 
 */
static inline uint2 fdims( uint2 dims ) {
    return uint2{ dims.x/2 + 1, dims.y };   
}

}

#endif