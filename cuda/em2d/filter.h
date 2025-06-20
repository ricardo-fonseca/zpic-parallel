#ifndef FILTER_H_
#define FILTER_H_

#include "zpic.h"
#include "vec3grid.h"

namespace Filter {

class Digital {
    public:
    virtual Digital * clone() const = 0;
    virtual void apply( vec3grid<float3> & fld ) = 0;
    virtual ~Digital() = default;
};

class None : public Digital {
    public:
    None * clone() const override { return new None(); };
    void apply( vec3grid<float3> & fld ) override { /* do nothing */ };
};

class Binomial : public Digital {
    protected:

    unsigned int order;
    coord::cart dir;
    
    public:

    Binomial( coord::cart dir, unsigned int order = 0 ) : 
        order( (order > 0) ? order: 1 ),
        dir(dir) { };

    Binomial * clone() const override { return new Binomial ( dir, order); };

    void apply( vec3grid<float3> & fld ) override {
        switch( dir ) {
        case( coord::x ):
            for( unsigned i = 0; i < order; i++ )
                fld.kernel3_x( 0.25f, 0.5f, 0.25f );
            break;
        case( coord::y ):
            for( unsigned i = 0; i < order; i++ )
                fld.kernel3_y( 0.25f, 0.5f, 0.25f );
            break;
        }
    }
};

class Compensated : public Binomial{
    
    public:

    Compensated( coord::cart dir, unsigned int order_ = 0 ) : Binomial ( dir, order_ ) {};

    Compensated * clone() const override { return new Compensated ( dir, order); };

    void apply( vec3grid<float3> & fld ) override {

        // Calculate compensator values
        float a = -1.0f;
        float b = (4.0 + 2.0*order) / order;
        float norm = 2*a+b;

        switch( dir ) {
        case( coord::x ):
            for( unsigned i = 0; i < order; i++ )
                fld.kernel3_x( 0.25f, 0.5f, 0.25f );
            fld.kernel3_x( a/norm, b/norm, a/norm );
            break;
        case( coord::y ):
            for( unsigned i = 0; i < order; i++ )
                fld.kernel3_y( 0.25f, 0.5f, 0.25f );
            fld.kernel3_y( a/norm, b/norm, a/norm );
            break;
        }
    };
};


}


#endif
