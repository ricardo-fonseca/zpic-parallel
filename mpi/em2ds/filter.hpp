#pragma once

#include  "grid/flat.hpp"
#include  "grid/flat3.hpp"

namespace filter {

using complex_t = std::complex<float>;

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
    void apply( grid::flat<complex_t> & fld ) override { /* do nothing */ };
    void apply( grid::flat3<complex_t> & fld ) override { /* do nothing */ };
};

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
        complex_t * __restrict__ data = fld.data();

        auto dims = fld.get_local_dims();
        auto start = fld.get_local_start();
        auto global = fld.get_global_dims();

        // Note that ky is along the x direction, and kx is along the y direction
        const int kcx = cutoff.x * ( global.y - 1 );
        const int kcy = cutoff.y * ( global.x / 2 );

        #pragma omp parallel for
        for( unsigned idx = 0; idx < dims.y * dims.x; idx++ ) {
            const int ix = start.x + idx % dims.x;
            const int iy = start.y + idx / dims.x;

            // Note that ky is along the x direction, and kx is along the y direction
            const float ky = std::abs(( 2 * ix < int(global.x) ) ? ix : ( ix - int(global.x) ) );
            const float kx = iy;

            if ( ky > kcy || kx > kcx ) {
                data[ idx ] = 0;
            }
        }
    };
    
    void apply( grid::flat3<complex_t> & fld ) override {
        complex_t * __restrict__ data_x = fld.x();
        complex_t * __restrict__ data_y = fld.y();
        complex_t * __restrict__ data_z = fld.z();
        

        auto dims = fld.get_local_dims();
        auto start = fld.get_local_start();
        auto global = fld.get_global_dims();

        // Note that ky is along the x direction, and kx is along the y direction
        const int kcx = cutoff.x * global.y;
        const int kcy = cutoff.y * global.x / 2;

        #pragma omp parallel for
        for( unsigned idx = 0; idx < dims.y * dims.x; idx++ ) {
            const int ix = start.x + idx % dims.x;
            const int iy = start.y + idx / dims.x;

            // Note that ky is along the x direction, and kx is along the y direction
            const float ky = std::abs(( 2 * ix < int(global.x) ) ? ix : ( ix - int(global.x) ) );
            const float kx = iy;

            if ( ky > kcy || kx > kcx ) {
                data_x[ idx ] = 0;
                data_y[ idx ] = 0;
                data_z[ idx ] = 0;
            }
        }
    };
};


}
