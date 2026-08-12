#pragma once

#include  "grid/flat.hpp"
#include  "grid/flat3.hpp"

namespace filter {

class digital {
    public:
    virtual digital * clone() const = 0;
    virtual void apply( grid::flat<std::complex<float>> & fld )  = 0;
    virtual void apply( grid::flat3<std::complex<float>> & fld ) = 0;
    virtual ~digital() = default;
};

class none : public digital {
    public:
    none * clone() const override { return new none(); };
    void apply( grid::flat<std::complex<float>> & fld ) override { /* do nothing */ };
    void apply( grid::flat3<std::complex<float>> & fld ) override { /* do nothing */ };
};

class lowpass : public digital {
    protected:

    const float2 cutoff;
    
    public:

    lowpass( const float2 cutoff ) : cutoff( cutoff ) {};

    lowpass * clone() const override { return new lowpass ( cutoff ); };

    void apply( grid::flat<std::complex<float>> & fld ) override {
        std::complex<float> * __restrict__ data = fld.data();

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
    
    void apply( grid::flat3<std::complex<float>> & fld ) override {
        std::complex<float> * __restrict__ data_x = fld.x();
        std::complex<float> * __restrict__ data_y = fld.y();
        std::complex<float> * __restrict__ data_z = fld.z();
        

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
