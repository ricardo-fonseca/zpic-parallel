#ifndef FFT_H_
#define FFT_H_


#ifdef _OPENMP
#include <omp.h>
#endif

#include <fftw3.h>

#include "grid.h"
#include "basic_grid.h"

#include "vec3grid.h"
#include "basic_grid3.h"

#include "complex.h"

namespace fft {

inline void init( ) {
    #ifdef _OPENMP
    fftwf_init_threads();
    #endif
}

inline void cleanup( ) {
    #ifdef _OPENMP
    fftwf_cleanup_threads();
    #endif
    fftwf_cleanup();
}


/// @brief Transform type
enum type  { r2c = 0, c2r, r2c_v3, c2r_v3 };

/**
 * @brief FFT plan
 * 
 */
class plan {
    private:

    /// @brief  FFT plan
    fftwf_plan fft_plan;

    /// @brief Temporary real grid (for r2c / c2r transforms)
    basic_grid< float > * tmp_real = nullptr;
    
    /// @brief Temporary complex grid (for c2r transforms)
    basic_grid< std::complex< float > > * tmp_complex = nullptr;

    /// @brief Temporary real grid (for r2c / c2r transforms)
    basic_grid3< float > * tmp_real3 = nullptr;
    
    /// @brief Temporary complex grid (for c2r transforms)
    basic_grid3< std::complex< float > > * tmp_complex3 = nullptr;

    /// @brief Data dimensions
    const uint2 dims;

    /// @brief FFT type
    const fft::type fft_type;

    public:


    /**
     * @brief Construct a new FFT plan object for real data transform
     * 
     * @param dims      Real data dimensions
     * @param fft_type  Type of transformation
     */
    plan( const uint2 dims, fft::type fft_type ):
        dims( dims ), fft_type( fft_type ) {

        if ( dims.x == 0 || dims.y == 0 ) {
            std::cerr << "Invalid dimensions for FFT plan " << dims << "aborting...\n";
            std::exit(1);
        }

        uint2 fdims{ dims.x/2 + 1, dims.y };

        int n[] = { (int) dims.y, (int) dims.x };

        // Creating an FFTW plan requires an existing buffer
        fftwf_complex * tmp_buffer = nullptr;

        #ifdef _OPENMP
        fftwf_plan_with_nthreads( omp_get_max_threads() );
        #endif

        switch( fft_type ) {
            case fft::type::r2c: 
                tmp_real = new basic_grid< float > ( dims );
                tmp_buffer = fftwf_alloc_complex( fdims.y * fdims.x );
                fft_plan = fftwf_plan_many_dft_r2c( 
                    2, n, 1,
                    tmp_real -> d_buffer, nullptr, 1, dims.y * dims.x,
                    tmp_buffer, nullptr, 1, fdims.y * fdims.x,
                    FFTW_ESTIMATE 
                );
                break;
            case fft::type::c2r: 
                tmp_real = new basic_grid< float > ( dims );
                tmp_complex = new basic_grid< std::complex<float> > ( fdims );
                fft_plan = fftwf_plan_many_dft_c2r( 
                    2, n, 1,
                    reinterpret_cast< fftwf_complex * > (tmp_complex -> d_buffer), nullptr, 1, fdims.y*fdims.x,
                    tmp_real -> d_buffer, nullptr, 1, dims.y*dims.x,
                    FFTW_ESTIMATE 
                );
                break;
            case fft::type::r2c_v3: 
                tmp_real3 = new basic_grid3< float > ( dims );
                tmp_buffer = fftwf_alloc_complex( fdims.y * fdims.x * 3 );
                fft_plan = fftwf_plan_many_dft_r2c( 
                    2, n, 3,
                    tmp_real3 -> d_buffer, nullptr, 1, dims.y * dims.x,
                    tmp_buffer, nullptr, 1, fdims.y * fdims.x,
                    FFTW_ESTIMATE 
                );
                break;
            case fft::type::c2r_v3: 
                tmp_real3 = new basic_grid3< float > ( dims );
                tmp_complex3 = new basic_grid3< std::complex<float> > ( fdims );
                fft_plan = fftwf_plan_many_dft_c2r( 
                    2, n, 3,
                    reinterpret_cast< fftwf_complex * > (tmp_complex3 -> d_buffer), nullptr, 1, fdims.y*fdims.x,
                    tmp_real3 -> d_buffer, nullptr, 1, dims.y*dims.x,
                    FFTW_ESTIMATE 
                );
                break;
            default:
                std::cerr << "Invalid fft type, aborting\n";
                std::exit(1);
        }

        // Free temporary memory
        fftwf_free( tmp_buffer );
    }

    /**
     * @brief Destroy the plan object
     * 
     */
    ~plan() {
        fftwf_destroy_plan( fft_plan );

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
            std::cerr << "FFT was not configured for real to complex transform, aborting\n";
            std::exit(1);
        }

        // Gather data into basic_grid
        real.gather( tmp_real->d_buffer );

        // Do transform
        fftwf_execute_dft_r2c( fft_plan,
            tmp_real -> d_buffer,
            reinterpret_cast< fftwf_complex * > (complex.d_buffer )
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
            std::cerr << "FFT was not configured for complex to real transform, aborting\n";
            std::exit(1);
        }

        // Copy original data to temporary buffer
        memory::memcpy( tmp_complex->d_buffer, complex.d_buffer, complex.buffer_size() );

        fftwf_execute_dft_c2r( 
            fft_plan,
            reinterpret_cast< fftwf_complex * > ( tmp_complex->d_buffer ),
            tmp_real->d_buffer
        );

        real.scatter( tmp_real->d_buffer, norm() );
    }


    /**
     * @brief Perform a real to complex transform of vec3 data
     * 
     * @param real 
     * @param complex 
     */
    void transform( vec3grid<float3> & real3, basic_grid3< std::complex<float> > & complex3 ) {

        if ( fft::type::r2c_v3 != fft_type ) {
            std::cerr << "FFT was not configured for vec3 real to complex transform, aborting\n";
            std::exit(1);
        }

        // Gather data into basic_grid
        real3.gather( tmp_real3->d_buffer );

        // Do transform
        fftwf_execute_dft_r2c( fft_plan,
            tmp_real3 -> d_buffer,
            reinterpret_cast< fftwf_complex * > ( complex3.d_buffer )
        );
    }   

    void transform( basic_grid3< std::complex<float> > & complex3, vec3grid<float3> & real3 ) {

        if ( fft::type::c2r_v3 != fft_type ) {
            std::cerr << "FFT was not configured for complex to real transform, aborting\n";
            std::exit(1);
        }

        // Copy original data to temporary buffer
        memory::memcpy( tmp_complex3->d_buffer, complex3.d_buffer, complex3.buffer_size() );

        fftwf_execute_dft_c2r( fft_plan,
            reinterpret_cast< fftwf_complex * > ( tmp_complex3 -> d_buffer ),
            tmp_real3 -> d_buffer
        );

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