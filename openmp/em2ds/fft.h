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

constexpr int forward  = FFTW_FORWARD;
constexpr int backward = FFTW_BACKWARD;

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
enum type  { r2c = 0, c2r, c2c, r2c_v3, c2r_v3 };

/**
 * @brief FFT plan
 * 
 */
class plan {
    private:

    void * fft_in;
    void * fft_out;

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
     * @brief Construct a new real to complex FFT plan
     * 
     * @param real      Real data grid
     * @param complex   Complex data
     */
    plan( grid<float>& real, basic_grid< std::complex<float> > & complex ) :
        dims( real.dims ), fft_type( fft::type::r2c ) {

        uint2 fdims{ dims.x/2 + 1, dims.y };

        if ( complex.dims.x != fdims.x ||  complex.dims.y != fdims.y ) {
            std::cerr << "Invalid dimensions for r2c plan\n";
            std::exit(1);
        }

        tmp_real    = new basic_grid< float > ( dims );
        tmp_complex = nullptr;

        #ifdef _OPENMP
        fftwf_plan_with_nthreads( omp_get_max_threads() );
        #endif

        fft_plan = fftwf_plan_dft_r2c_2d(
            dims.y, dims.x,
            tmp_real -> d_buffer, reinterpret_cast< fftwf_complex * > (complex.d_buffer),
            FFTW_ESTIMATE
        );

        if ( fft_plan == nullptr ) {
            std::cerr << "Error creating FFTW R2C plan, aborting...\n";
            std::exit(1);
        }

        fft_in  = reinterpret_cast< void * > (tmp_real->d_buffer);
        fft_out = reinterpret_cast< void * > (complex.d_buffer);
    }

    /**
     * @brief Construct a new complex to real FFT plan
     * 
     * @param complex 
     * @param real 
     */
    plan( basic_grid< std::complex<float> > & complex, grid<float>& real ) :
        dims( real.dims ), fft_type( fft::type::c2r ) {

        uint2 fdims{ dims.x/2 + 1, dims.y };

        if ( complex.dims.x != fdims.x ||  complex.dims.y != fdims.y ) {
            std::cerr << "Invalid dimensions for c2r plan\n";
            std::exit(1);
        }

        tmp_real    = new basic_grid< float > ( dims );
        tmp_complex = new basic_grid< std::complex<float> > ( dims );

        #ifdef _OPENMP
        fftwf_plan_with_nthreads( omp_get_max_threads() );
        #endif

        fft_plan = fftwf_plan_dft_c2r_2d(
            dims.y, dims.x,
            reinterpret_cast< fftwf_complex * > (tmp_complex -> d_buffer), tmp_real->d_buffer,
            FFTW_ESTIMATE
        );

        if ( fft_plan == nullptr ) {
            std::cerr << "Error creating FFTW C2R plan, aborting...\n";
            std::exit(1);
        }

        fft_in  = reinterpret_cast< void * > (tmp_complex -> d_buffer);
        fft_out = reinterpret_cast< void * > (tmp_real->d_buffer);
    }

    /**
     * @brief Construct a new complex to complex FFT plan
     * 
     * @param in 
     * @param out 
     * @param direction 
     */
    plan( basic_grid< std::complex<float> > & in, basic_grid< std::complex<float> > & out, int direction ) :
        dims( in.dims ), fft_type( fft::type::c2c ) {
        
        if ( in.dims.x != out.dims.x ||  in.dims.y != out.dims.y ) {
            std::cerr << "Invalid dimensions for c2r plan\n";
            std::exit(1);
        }

        #ifdef _OPENMP
        fftwf_plan_with_nthreads( omp_get_max_threads() );
        #endif

        fft_plan = fftwf_plan_dft_2d(
            dims.y, dims.x,
            reinterpret_cast< fftwf_complex * > (in.d_buffer), 
            reinterpret_cast< fftwf_complex * > (out.d_buffer),
            direction,
            FFTW_ESTIMATE
        );

        fft_in  = reinterpret_cast< void * > (in.d_buffer);
        fft_out = reinterpret_cast< void * > (out.d_buffer);

    }

    /**
     * @brief Construct a new vec3 real to complex FFT plan
     * 
     * @warning Not implemented yet
     * 
     * @param real 
     * @param complex 
     */
    plan( vec3grid<float3> & real3, basic_grid3< std::complex<float> > & complex3 ) :
        dims( real3.dims ), fft_type( fft::type::r2c_v3 ){

        uint2 fdims{ dims.x/2 + 1, dims.y };

        if ( complex3.dims.x != fdims.x ||  complex3.dims.y != fdims.y ) {
            std::cerr << "Invalid dimensions for r2c plan\n";
            std::exit(1);
        }

        tmp_real3    = new basic_grid3< float > ( dims );

        #ifdef _OPENMP
        fftwf_plan_with_nthreads( omp_get_max_threads() );
        #endif

        int fft_dims[] = { (int) dims.y, (int) dims.x };
        fft_plan = fftwf_plan_many_dft_r2c( 
            2, fft_dims, 3,
            tmp_real3 -> d_buffer, nullptr, 1, dims.y * dims.x,
            reinterpret_cast< fftwf_complex * > (complex3.d_buffer), nullptr, 1, fdims.y * fdims.x,
            FFTW_ESTIMATE 
        );

        if ( fft_plan == nullptr ) {
            std::cerr << "Error creating FFTW R2C(3) plan, aborting...\n";
            std::exit(1);
        }

        fft_in  = reinterpret_cast< void * > (tmp_real3->d_buffer);
        fft_out = reinterpret_cast< void * > (complex3.d_buffer);        
    }   

    /**
     * @brief Construct a new vec3 complex to real FFT plan
     * 
     * @warning Not implemented yet
     * 
     * @param complex 
     * @param real 
     */
    plan( basic_grid3< std::complex<float> > & complex, vec3grid<float3> & real ) :
        dims( real.dims ), fft_type( fft::type::c2r_v3 ) {

        uint2 fdims{ dims.x/2 + 1, dims.y };

        if ( complex.dims.x != fdims.x ||  complex.dims.y != fdims.y ) {
            std::cerr << "Invalid dimensions for c2r plan\n";
            std::exit(1);
        }

        tmp_real3    = new basic_grid3< float > ( dims );
        tmp_complex3 = new basic_grid3< std::complex<float> > ( fdims );

        #ifdef _OPENMP
        fftwf_plan_with_nthreads( omp_get_max_threads() );
        #endif

        int real_dims[] = { (int) dims.y, (int) dims.x };
        fft_plan = fftwf_plan_many_dft_c2r( 
            2, real_dims, 3,
            reinterpret_cast< fftwf_complex * > (tmp_complex3 -> d_buffer), nullptr, 1, fdims.y*fdims.x,
            tmp_real3 -> d_buffer, nullptr, 1, dims.y*dims.x,
            FFTW_ESTIMATE 
        );

        if ( fft_plan == nullptr ) {
            std::cerr << "Error creating FFTW C2R(3) plan, aborting...\n";
            std::exit(1);
        }

        fft_in  = reinterpret_cast< void * > (tmp_complex3 -> d_buffer);
        fft_out = reinterpret_cast< void * > (tmp_real3->d_buffer);
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

        if ( fft_out != reinterpret_cast< void * > (complex.d_buffer) ) {
            std::cerr << "Invalid output grid, must be the same used for creating plan, aborting\n";
            std::exit(1);            
        };

        real.gather( tmp_real->d_buffer );
        fftwf_execute( fft_plan );
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

        fftwf_execute( fft_plan );

        real.scatter( tmp_real->d_buffer, norm() );
    }


    /**
     * @brief Perform a complex to complex transform
     * 
     * @param input         Input data
     * @param output        Output data
     */
    void transform( std::complex<float> * const __restrict__ input, 
                    std::complex<float> * const __restrict__ output ) {

        if ( fft::type::c2c != fft_type ) {
            std::cerr << "FFT was not configured for complex to complex transform, aborting\n";
            std::exit(1);
        }

        if ( fft_in != reinterpret_cast< void * > (input) ) {
            std::cerr << "Invalid input grid, must be the same used for creating plan, aborting\n";
            std::exit(1);            
        };

        if ( fft_out != reinterpret_cast< void * > (output) ) {
            std::cerr << "Invalid output grid, must be the same used for creating plan, aborting\n";
            std::exit(1);            
        };

        fftwf_execute( fft_plan );
    }

    /**
     * @brief Perform a real to complex transform of vec3 data
     * 
     * @param real 
     * @param complex 
     */
    void transform( vec3grid<float3> & real3, basic_grid3< std::complex<float> > & complex3 ) {

        if ( fft::type::r2c_v3 != fft_type ) {
            std::cerr << "FFT was not configured for real to complex transform, aborting\n";
            std::exit(1);
        }

        if ( fft_out != reinterpret_cast< void * > (complex3.d_buffer) ) {
            std::cerr << "Invalid output grid, must be the same used for creating plan, aborting\n";
            std::exit(1);            
        };

        real3.gather( tmp_real3->d_buffer );
        fftwf_execute( fft_plan );
    }   

    void transform( basic_grid3< std::complex<float> > & complex3, vec3grid<float3> & real3 ) {

        if ( fft::type::c2r_v3 != fft_type ) {
            std::cerr << "FFT was not configured for complex to real transform, aborting\n";
            std::exit(1);
        }

        // Copy original data to temporary buffer
        memory::memcpy( tmp_complex3->d_buffer, complex3.d_buffer, complex3.buffer_size() );

        fftwf_execute( fft_plan );

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
        os << "FFT plan, dims: " << obj.dims << ", type: " << obj.fft_type;
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
    constexpr float pi = 3.14159265358979323846264338327950288f;
    
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