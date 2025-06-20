#ifndef FFT_H_
#define FFT_H_

#include "gpu.h"
#include "cufft.h"
#include "grid.h"
#include "vec3grid.h"

#include "complex.h"

namespace fft {

/// @brief Transform type
enum type  { r2c = 0, c2r, c2c };

/**
 * @brief FFT plan
 * 
 */
class plan {
    private:

    /// @brief  FFT plan
    cufftHandle fft_plan;

    public:

    /// @brief FFT type
    const fft::type fft_type;

    /// @brief Data dimensions
    const uint2 dims;

    /// @brief Batch size (number of simultaneous transforms)
    const int batch;

    /**
     * @brief Construct a new plan object
     * 
     * @param dims      Data dimensions. When using C2R transformations this refers to the
     *                  real data dimensions
     * @param fft_type  Type of transformation
     */
    plan( const uint2 dims, fft::type fft_type, int batch = 1 ):
        dims( dims ), fft_type( fft_type ), batch( batch ) {

        if ( dims.x == 0 || dims.y == 0 ) {
            std::cerr << "Invalid dimensions for FFT plan " << dims << "aborting...\n";
            device::exit(1);
        }

        if ( batch < 1 ) {
            std::cerr << "Invalid batch size for FFT plan " << batch << "aborting...\n";
            device::exit(1);
        }

        cufftType_t tmp_type;

        switch( fft_type ) {
            case fft::type::r2c: 
                tmp_type = CUFFT_R2C; break;
            case fft::type::c2r: 
                tmp_type = CUFFT_C2R; break;
            case fft::type::c2c: 
                tmp_type = CUFFT_R2C; break;
            default:
                std::cerr << "Invalid fft type, aborting\n";
                device::exit(1);
        }
        
        // cuFFT defines this in the opposite order
        int fft_dims[] = { (int) dims.y, (int) dims.x };

        cufftResult ierr = cufftPlanMany( & fft_plan, 2, fft_dims, 
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
    }

    /**
     * @brief Dimensions of input data for transform
     * 
     * @return uint2 
     */
    inline uint2 input_dims() {
        switch( fft_type ) {
            case fft::type::c2r: 
                return uint2{ dims.x/2 + 1, dims.y };

            case fft::type::r2c: 
            case fft::type::c2c:
            default:
                return dims;
        }
    }

    /**
     * @brief Dimensions of output data for transform
     * 
     * @return uint2 
     */
    inline uint2 output_dims() {
        switch( fft_type ) {
            case fft::type::r2c: 
                return uint2{ dims.x/2 + 1, dims.y };

            case fft::type::c2r: 
            case fft::type::c2c:
            default:
                return dims;
        }
    }

    /**
     * @brief Perform a real to complex transform from grid<float> data
     * 
     * @warning This will allocate / deallocate a temporary array
     * 
     * @param real      (in) Tiled grid real data
     * @param complex   (out) Contiguous complex data
     */
    void transform( grid<float>& real, fft::complex64 * const __restrict__ complex ) {
        if ( fft::type::r2c != fft_type ) {
            std::cerr << "FFT was not configured for real to complex tranform, aborting\n";
            device::exit(1);
        }

        if ( batch != 1 ) {
            std::cerr << "Invalid batch size for grid<float> r2c transform, aborting\n";
            device::exit(1);
        }

        // Temporary array
        float * tmp = device::malloc<float>( dims.x * dims.y );
        real.gather( tmp );
        cufftExecR2C( fft_plan, tmp, complex );
        device::free( tmp );
    }

    /**
     * @brief Perform a real to complex transform from vec3grid<float3> data
     * 
     * @warning This will allocate / deallocate a temporary array
     * 
     * @param real      (in) Tiled vec3grid real data
     * @param complex   (out) Contiguous complex data
     */
    void transform( vec3grid<float3>& real, fft::complex64 * const __restrict__ complex ) {
        if ( fft::type::r2c != fft_type ) {
            std::cerr << "FFT was not configured for real to complex tranform, aborting\n";
            device::exit(1);
        }

        if ( batch != 3 ) {
            std::cerr << "Invalid batch size for grid<float> r2c transform, aborting\n";
            device::exit(1);
        }

        // Temporary array
        auto * tmp = device::malloc<float>( dims.x * dims.y * 3 );
        real.gather( tmp );
        cufftExecR2C( fft_plan, tmp, complex );
        device::free( tmp );
    }

    /**
     * @brief Perform a complex to real transform to grid<float> data
     * 
     * @warning This will allocate / deallocate a temporary array
     * 
     * @param complex   (in) Contiguous complex data
     * @param real      (out) Tiled grid real data
     */
    void transform( fft::complex64 * const __restrict__ complex, grid<float>& real ) {
        if ( fft::type::c2r != fft_type ) {
            std::cerr << "FFT was not configured for complex to real tranform, aborting\n";
            device::exit(1);
        }

        if ( batch != 1 ) {
            std::cerr << "Invalid batch size for grid<float> c2r transform, aborting\n";
            device::exit(1);
        }

        // Temporary array
        float * tmp = device::malloc<float>( dims.x * dims.y );
        cufftExecC2R( fft_plan, complex, tmp );
        real.scatter( tmp, norm() );
        device::free( tmp );
    }

    /**
     * @brief Perform a complex to real transform to grid<float> data
     * 
     * @warning This will allocate / deallocate a temporary array
     * 
     * @param complex   (in) Contiguous complex data
     * @param real      (out) Tiled grid real data
     */
    void transform( fft::complex64 * const __restrict__ complex, vec3grid<float3>& real ) {
        if ( fft::type::c2r != fft_type ) {
            std::cerr << "FFT was not configured for complex to real tranform, aborting\n";
            device::exit(1);
        }

        if ( batch != 3 ) {
            std::cerr << "Invalid batch size for grid<float> c2r transform, aborting\n";
            device::exit(1);
        }

        // Temporary array
        float * tmp = device::malloc<float>( dims.x * dims.y * 3 );
        cufftExecC2R( fft_plan, complex, tmp );
        real.scatter( tmp, norm() );
        device::free( tmp );
    }

    /**
     * @brief Perform a real to complex transform from grid<float> data
     *  
     * @param real      (in) Tiled grid real data
     * @param complex   (out) Contiguous complex data
     * @param d_tmp     (out) Temporary real data in device memory. Must be >= ndims.x * ndims.y
     */
    void transform( grid<float>& real, 
                    fft::complex64 * const __restrict__ complex, 
                    float * const __restrict__ d_tmp ) {
        if ( fft::type::r2c != fft_type ) {
            std::cerr << "FFT was not configured for real to complex tranform, aborting\n";
            device::exit(1);
        }
        real.gather( d_tmp );
        cufftExecR2C( fft_plan, d_tmp, complex );
    }

    /**
     * @brief Perform a complex to real transform to grid<float> data
     * 
     * @param complex   (in) Contiguous complex data
     * @param real      (out) Tiled grid real data
     * @param d_tmp     (out) Temporary real data in device memory. Must be >= ndims.x * ndims.y
     */
    void transform( fft::complex64 * const __restrict__ complex, 
                    grid<float>& real, 
                    float * const __restrict__ d_tmp ) {
        if ( fft::type::c2r != fft_type ) {
            std::cerr << "FFT was not configured for complex to real tranform, aborting\n";
            device::exit(1);
        }

        cufftExecC2R( fft_plan, complex, d_tmp );
        real.scatter( d_tmp );
    }

    /**
     * @brief Perform a complex to complex transform
     * 
     * @param input         Input data
     * @param output        Output data
     * @param direction     Transform direction, should be one of CUFFT_FORWARD or CUFFT_INVERSE
     */
    void transform( fft::complex64 * const __restrict__ input, 
                    fft::complex64 * const __restrict__ output, 
                    int direction ) {
        if ( fft::type::c2c != fft_type ) {
            std::cerr << "FFT was not configured for complex to real tranform, aborting\n";
            device::exit(1);
        }

        // transform out-of-place
        cufftExecC2C( fft_plan, input, output, direction );
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
    return float2{ 2 * M_PIf / box.x, 2 * M_PIf / box.y };
}

}

#endif