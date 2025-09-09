#include <iostream>

#include "utils.h"

#include <complex>

#include "vec_types.h"
#include "cyl3.h"

#include "cyl3grid.h"
#include "cyl3modes.h"

#include "emf.h"
#include "laser.h"
#include "species.h"

void test_cylgrid( void ) {

    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;

    uint2 ntiles{ 8, 8 };
    uint2 nx{ 16, 16 };
    bnd<unsigned int> gc{ 0 };

    CylGrid<float> field( 4, ntiles, nx, gc );

    std::cout << "field: " << field << '\n';

    std::cout << "mode0: " << field.mode0() << '\n';
    std::cout << "mode1: " << field.mode(1) << '\n';
    std::cout << "mode2: " << field.mode(2) << '\n';
    std::cout << "mode3: " << field.mode(3) << '\n';

    std::cout << ansi::bold;
    std::cout << "Done!\n";
    std::cout << ansi::reset;

}

void test_cyl3_cylgrid( void ) {

    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;

    uint2 ntiles{ 8, 8 };
    uint2 nx{ 16, 16 };
    bnd<unsigned int> gc{ 0 };

    Cyl3CylGrid<float> field( 4, ntiles, nx, gc );

    std::cout << "field: " << field << '\n';

    std::cout << "mode0: " << field.mode0() << '\n';
    std::cout << "mode1: " << field.mode(1) << '\n';
    std::cout << "mode2: " << field.mode(2) << '\n';
    std::cout << "mode3: " << field.mode(3) << '\n';

    std::cout << ansi::bold;
    std::cout << "Done!\n";
    std::cout << ansi::reset;

}

void test_pvec3_cylgrid( void ) {

    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;

    uint2 ntiles{ 8, 8 };
    uint2 nx{ 16, 16 };
    bnd<unsigned int> gc{ 0 };

    auto A = new Cyl3CylGrid<float>( 4, ntiles, nx, gc );
    auto B = new Cyl3CylGrid<float>( 4, ntiles, nx, gc );

    A -> set_name( "field A" );
    B -> set_name( "field B" );

    A -> zero();
    B -> zero();

    std::cout << "A: " << * A << '\n';

    std::cout << "mode0: " << A->mode0() << '\n';
    std::cout << "mode1: " << B->mode(1) << '\n';
    std::cout << "mode2: " << A->mode(2) << '\n';
    std::cout << "mode3: " << B->mode(3) << '\n';

    delete( A );
    delete( B );

    std::cout << ansi::bold;
    std::cout << "Done!\n";
    std::cout << ansi::reset;

}


void test_emf( void ) {

    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;

    uint2 ntiles{ 8, 8 };
    uint2 nx{ 16, 16 };

    float2 box{ 12.8, 12.8 };
    double dt{ 0.07 };

    EMF emf( 4, ntiles, nx, box, dt );

    std::cout << "field: " << emf << '\n';

    std::cout << "mode0: " << emf.E->mode0() << '\n';
    std::cout << "mode1: " << emf.B->mode(1) << '\n';
    std::cout << "mode2: " << emf.E->mode(2) << '\n';
    std::cout << "mode3: " << emf.B->mode(3) << '\n';

    std::cout << ansi::bold
              << "Completed " << __func__ << "()\n"
              << ansi::reset;
}

void test_laser( void ) {

    std::cout << ansi::bold
              << "Running " << __func__ << "()...\n"
              << ansi::reset;

    uint2 ntiles{ 16, 8 };
    uint2 nx{ 64, 32 };

    float2 box{ 20.48, 25.6 };
    double dt = 0.01;

    EMF emf( 4, ntiles, nx, box, dt );


//    Laser::PlaneWave laser;

    Laser::Gaussian laser;
    laser.W0 = 4.0;
    laser.focus = 20.48;

    // Common laser parameters
    laser.start = 10.2;
    laser.fwhm = 4.0;
    laser.a0 = 1.0;
    laser.omega0 = 10.0;
    laser.sin_pol = 1;
    laser.cos_pol = 0;


    laser.add( emf );

    auto save_emf = [& emf ]( ) {
        emf.save( emf::e, fcomp::r, 1 );
        emf.save( emf::e, fcomp::θ, 1 );
        emf.save( emf::e, fcomp::z, 1 );

        emf.save( emf::b, fcomp::r, 1 );
        emf.save( emf::b, fcomp::θ, 1 );
        emf.save( emf::b, fcomp::z, 1 );
    };

    save_emf();

    auto niter = 700;
    for( int i = 0; i < niter; i ++) {
        emf.advance();
    }
 
    save_emf( );

    std::cout << ansi::bold
              << "Completed " << __func__ << "()\n"
              << ansi::reset;

}

void test_inj( void ) {

    std::cout << ansi::bold
            << "Running " << __func__ << "()...\n"
            << ansi::reset;

    uint2 ntiles{ 4, 4 };
    uint2 nx{ 16, 16 };
    float2 box{ 4, 4 };

    auto dt = 0.99 * zpic::courant( ntiles, nx, box );

    uint3 ppc{ 2, 2, 8 };
    Species electrons( "electrons", -1.0f, ppc );

//    electrons.set_density(Density::Step(coord::z, 1.0, 2.0));
//    electrons.set_density( Density::Slab(coord::z, 1.0, 2.0, 3.0)); 
    electrons.set_density( Density::Sphere(1.0, float2{2,2}, 1.0)); 

    electrons.initialize( box, ntiles, nx, dt, 0 );
    electrons.save();
    electrons.save_charge( 0 );

    std::cout << ansi::bold
              << "Completed " << __func__ << "()\n"
              << ansi::reset;

}


int main( void ) {

    std::cout << ansi::bold 
              << "quasi-3D tests\n"
              << ansi::reset ;

    // test_cylgrid();
    // test_vec3_cylgrid();
    // test_pvec3_cylgrid();

    // test_emf();
    // test_laser();

    test_inj();
}
