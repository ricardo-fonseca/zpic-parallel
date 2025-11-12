#ifndef FILTER_H_
#define FILTER_H_

#include "zpic.h"
#include "cyl3grid.h"

namespace Filter {

class Digital {
    public:
    virtual Digital * clone() const = 0;
    virtual ~Digital() = default;

    virtual void apply( cyl3grid<float> & fld ) = 0;
    virtual void apply( cyl3grid<std::complex<float>> & fld ) = 0;
};

class None : public Digital {
    public:
    None * clone() const override { return new None(); };

    void apply( cyl3grid<float> & fld ) override { };
    void apply( cyl3grid<std::complex<float>> & fld ) override { };
};

class Binomial : public Digital {
    protected:

    unsigned int order;
    coord::cyl dir;
    
    public:

    Binomial( coord::cyl dir, unsigned int order = 0 ) : 
        order( (order > 0) ? order: 1 ),
        dir(dir) { };

    Binomial * clone() const override { return new Binomial ( dir, order); };

    void apply( cyl3grid<float> & fld ) override {
        switch( dir ) {
        case( coord::z ):
            for( unsigned i = 0; i < order; i++ )
                fld.kernel3_x( 0.25f, 0.5f, 0.25f );
            break;
        case( coord::r ):
            for( unsigned i = 0; i < order; i++ )
                fld.kernel3_y( 0.25f, 0.5f, 0.25f );
            break;
        }
    }

    void apply( cyl3grid<std::complex<float>> & fld ) override {
        switch( dir ) {
        case( coord::z ):
            for( unsigned i = 0; i < order; i++ )
                fld.kernel3_x( 0.25f, 0.5f, 0.25f );
            break;
        case( coord::r ):
            for( unsigned i = 0; i < order; i++ )
                fld.kernel3_y( 0.25f, 0.5f, 0.25f );
            break;
        }
    }
};

class Compensated : public Binomial{
    
    public:

    Compensated( coord::cyl dir, unsigned int order = 0 ) : Binomial ( dir, order ) {};
    Compensated * clone() const override { return new Compensated ( dir, order); };

    void apply( cyl3grid<float> & fld ) override {

        // Calculate compensator values
        float a = -1.0f;
        float b = (4.0 + 2.0*order) / order;
        float norm = 2*a+b;

        switch( dir ) {
        case( coord::z ):
            for( unsigned i = 0; i < order; i++ )
                fld.kernel3_x( 0.25f, 0.5f, 0.25f );
            fld.kernel3_x( a/norm, b/norm, a/norm );
            break;
        case( coord::r ):
            for( unsigned i = 0; i < order; i++ )
                fld.kernel3_y( 0.25f, 0.5f, 0.25f );
            fld.kernel3_y( a/norm, b/norm, a/norm );
            break;
        }
    };

    void apply( cyl3grid<std::complex<float>> & fld ) override {

        // Calculate compensator values
        float a = -1.0f;
        float b = (4.0 + 2.0*order) / order;
        float norm = 2*a+b;

        switch( dir ) {
        case( coord::z ):
            for( unsigned i = 0; i < order; i++ )
                fld.kernel3_x( 0.25f, 0.5f, 0.25f );
            fld.kernel3_x( a/norm, b/norm, a/norm );
            break;
        case( coord::r ):
            for( unsigned i = 0; i < order; i++ )
                fld.kernel3_y( 0.25f, 0.5f, 0.25f );
            fld.kernel3_y( a/norm, b/norm, a/norm );
            break;
        }
    };
};


}


#endif
