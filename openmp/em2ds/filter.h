#ifndef FILTER_H_
#define FILTER_H_

#include "zpic.h"
#include  "basic_grid.h"
#include  "basic_grid3.h"

namespace Filter {

class Digital {
    public:
    virtual Digital * clone() const = 0;
    virtual void apply( basic_grid<std::complex<float>> & fld )  = 0;
    virtual void apply( basic_grid3<std::complex<float>> & fld ) = 0;
    virtual ~Digital() = default;
};

class None : public Digital {
    public:
    None * clone() const override { return new None(); };
    void apply( basic_grid<std::complex<float>> & fld ) override { /* do nothing */ };
    void apply( basic_grid3<std::complex<float>> & fld ) override { /* do nothing */ };
};

class Lowpass : public Digital {
    protected:

    const float2 cutoff;
    
    public:

    Lowpass( const float2 cutoff ) : cutoff( cutoff ) {};

    Lowpass * clone() const override { return new Lowpass ( cutoff ); };

    void apply( basic_grid<std::complex<float>> & fld ) override {
        std::complex<float> * __restrict__ data = fld.d_buffer;
        
        const int kcx = cutoff.x * ( fld.dims.x - 1 );
        const int kcy = cutoff.y * ( fld.dims.y / 2 );

        #pragma omp parallel for
        for( unsigned idx = 0; idx < fld.dims.y * fld.dims.x; idx++ ) {
            const int ix = idx % fld.dims.x;
            const int iy = idx / fld.dims.x;

            const int kx  = ix;
            const int ky  = abs( ((iy < int(fld.dims.y)/2) ? iy : (iy - int(fld.dims.y)) ) );

            if ( ky > kcy && kx > kcx ) data[ idx ] = 0;
        }
    };
    
    void apply( basic_grid3<std::complex<float>> & fld ) override {
        std::complex<float> * __restrict__ data_x = fld.x;
        std::complex<float> * __restrict__ data_y = fld.y;
        std::complex<float> * __restrict__ data_z = fld.z;
        
        const int kcx = cutoff.x * ( fld.dims.x - 1 );
        const int kcy = cutoff.y * ( fld.dims.y / 2 );

        #pragma omp parallel for
        for( unsigned idx = 0; idx < fld.dims.y * fld.dims.x; idx++ ) {
            const int ix = idx % fld.dims.x;
            const int iy = idx / fld.dims.x;

            const int kx  = ix;
            const int ky  = abs( ((iy < int(fld.dims.y)/2) ? iy : (iy - int(fld.dims.y)) ) );

            if ( ky > kcy && kx > kcx ) {
                data_x[ idx ] = 0;
                data_y[ idx ] = 0;
                data_z[ idx ] = 0;
            }
        }
    };
};


}


#endif
