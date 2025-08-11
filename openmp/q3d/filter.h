#ifndef FILTER_H_
#define FILTER_H_

#include "zpic.h"
#include "cyl3grid.h"

namespace Filter {

class Digital {
    public:
    virtual Digital * clone() const = 0;
    virtual ~Digital() = default;

    template < class T >
    void apply( cyl3grid<T> & fld ) { 
        static_assert(0, "Filter class must override apply method");
    };
};

class None : public Digital {
    public:
    None * clone() const override { return new None(); };
    template < class T >
    void apply( cyl3grid<T> & fld ) { };
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

    template < class T >
    void apply( cyl3grid<T> & fld ) {
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

    Compensated( coord::cart dir, unsigned int order = 0 ) : Binomial ( dir, order ) {};

    Compensated * clone() const override { return new Compensated ( dir, order); };

    template < class T >
    void apply( cyl3grid<T> & fld ) {

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
