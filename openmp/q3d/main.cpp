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

#include "timer.h"
#include "simulation.h"

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

    auto dt = 0.99 * zpic::courant( 2, ntiles, nx, box );

    uint3 ppc{ 2, 2, 8 };
    Species electrons( "electrons", -1.0f, ppc );

//    electrons.set_density(Density::Step(coord::z, 1.0, 2.0));
//    electrons.set_density( Density::Slab(coord::z, 1.0, 2.0, 3.0)); 
    electrons.set_density( Density::Sphere(1.0, float2{2,2}, 1.0)); 

    electrons.initialize( 2, box, ntiles, nx, dt, 0 );
    electrons.save();
    electrons.save_charge( 0 );

    std::cout << ansi::bold
              << "Completed " << __func__ << "()\n"
              << ansi::reset;

}

void test_mov( void ) {

    std::cout << ansi::bold
            << "Running " << __func__ << "()...\n"
            << ansi::reset;


    uint2 ntiles{ 4, 4 };
    uint2 nx{ 32, 32 };
    float2 box{ 12.8, 12.8 };

    auto dt = 0.99 * zpic::courant( 2, ntiles, nx, box );

    uint3 ppc{ 2, 2, 8 };
    Species electrons( "electrons", -1.0f, ppc );

    electrons.set_density( Density::Sphere(1.0, float2{2,2}, 1.0)); 
    electrons.set_udist( UDistribution::Cold( float3{ 0, 0, 1.e6 } ) );

    electrons.initialize( 2, box, ntiles, nx, dt, 0 );

    electrons.save_charge(0);

    int niter = 100;
    for( auto i = 0; i < niter; i ++ ) {
        electrons.advance();
    }

    electrons.save_charge(0);
    electrons.save();


    std::cout << ansi::bold
              << "Completed " << __func__ << "()\n"
              << ansi::reset;

}


void test_current( void ) {

    std::cout << ansi::bold
            << "Running " << __func__ << "()...\n"
            << ansi::reset;


    uint2 ntiles{ 4, 4 };
    uint2 nx{ 32, 32 };
    float2 box{ 12.8, 12.8 };

    // auto dt = 0.99 * zpic::courant( ntiles, nx, box );
    
    auto dt = 0.06;

    Current current( 2, ntiles, nx, box, dt );

    uint3 ppc{ 2, 2, 8 };
    Species electrons( "electrons", -1.0f, ppc );

    electrons.set_density( Density::Sphere(1.0, float2{6.4,6.4}, 3.2)); 

    electrons.set_udist( UDistribution::Cold( float3{ 1e6, 1e6, 1.e6 } ) );
//    electrons.set_udist( UDistribution::Cold( float3{ 0, 0, 1e6 } ) );

    electrons.initialize( 2, box, ntiles, nx, dt, 0 );

    electrons.save_charge(0);

    current.zero();

    electrons.advance( current );
    
    current.advance();

    // Save mode 0
    current.save( fcomp::z, 0 );
    current.save( fcomp::r, 0 );
    current.save( fcomp::θ, 0 );

    // Save mode 1
    current.save( fcomp::z, 1 );
    current.save( fcomp::r, 1 );
    current.save( fcomp::θ, 1 );

    std::cout << ansi::bold
              << "Completed " << __func__ << "()\n"
              << ansi::reset;

}

void test_beam( void ) {

    std::cout << ansi::bold
            << "Running " << __func__ << "()...\n"
            << ansi::reset;

    uint2 dims { 256, 256 };
    float2 box { 25.6, 25.6 };

    // auto dt = 0.99 * zpic::courant( ntiles, nx, box );

    uint2 nx { 32, 32 };
    uint2 ntiles { dims.x / nx.x, dims.y / nx.y };

    auto dt = zpic::courant( 2, dims, box ) * 0.5f;

    std::cout << "dt     = " << dt << '\n';
    std::cout << "ntiles = " << ntiles << '\n';

    // Create simulation
    Simulation sim( 2, ntiles, nx, box, dt );


    uint3 ppc{ 2, 2, 1 };
    Species electrons( "electrons", -1.0f, ppc );

    electrons.set_density( Density::Sphere(1.0, float2{20.0,0.0}, 1.6)); 
    electrons.set_udist( UDistribution::Cold( float3{ 0, 0, 1e6 } ) );

    sim.add_species( electrons );
    sim.set_moving_window();

    electrons.save();

    auto diag = [& sim, & electrons ]( ) {
        // Save mode 0
        electrons.save_charge(0);

        sim.emf.save(emf::e, fcomp::z, 0);
        sim.emf.save(emf::e, fcomp::r, 0);
        sim.emf.save(emf::e, fcomp::θ, 0);

        sim.emf.save(emf::b, fcomp::z, 0);
        sim.emf.save(emf::b, fcomp::r, 0);
        sim.emf.save(emf::b, fcomp::θ, 0);

        sim.current.save( fcomp::z, 0 );
        sim.current.save( fcomp::r, 0 );
        sim.current.save( fcomp::θ, 0 );
    };

    for( int i = 0; i < 1000; i++ ) {
        if ( i % 10 == 0 ) diag();
        sim.advance_mov_window();
    }
    diag();


    std::cout << ansi::bold
              << "Completed " << __func__ << "()\n"
              << ansi::reset;

}

void test_pwfa( void ) {

    std::cout << ansi::bold
            << "Running " << __func__ << "()...\n"
            << ansi::reset;

    uint2 dims { 256, 256 };
    float2 box { 25.6, 25.6 };

    // auto dt = 0.99 * zpic::courant( ntiles, nx, box );

    uint2 nx { 32, 32 };
    uint2 ntiles { dims.x / nx.x, dims.y / nx.y };

    auto dt = zpic::courant( 2, dims, box ) * 0.5f;

    std::cout << "dt     = " << dt << '\n';
    std::cout << "ntiles = " << ntiles << '\n';

    // Create simulation
    Simulation sim( 1, ntiles, nx, box, dt );

    uint3 ppc{ 2, 2, 1 };

    Species beam( "beam", -1.0f, ppc );
    beam.set_density( Density::Sphere(1.0, float2{23.0,0.0}, 1.6)); 
    beam.set_udist( UDistribution::Cold( float3{ 0, 0, 1e6 } ) );
    sim.add_species( beam );

    Species plasma( "plasma", -1.0f, ppc );
    plasma.set_density( Density::Step( coord::z, 1.0, 40.96 ) ); 
    sim.add_species( plasma );

    sim.set_moving_window();

    auto diag = [& sim, & beam, & plasma ]( ) {
        // Save mode 0
        beam.save_charge(0);
        plasma.save_charge(0);
        plasma.save();

        sim.emf.save(emf::e, fcomp::z, 0);
        sim.emf.save(emf::e, fcomp::r, 0);
        sim.emf.save(emf::e, fcomp::θ, 0);

        sim.emf.save(emf::b, fcomp::z, 0);
        sim.emf.save(emf::b, fcomp::r, 0);
        sim.emf.save(emf::b, fcomp::θ, 0);

        sim.current.save( fcomp::z, 0 );
        sim.current.save( fcomp::r, 0 );
        sim.current.save( fcomp::θ, 0 );
    };

    while ( sim.get_t() <= 61.5 ) {
        if ( sim.get_iter() % 10 == 0 ) diag();
        sim.advance_mov_window();
    }

    std::cout << ansi::bold
              << "Completed " << __func__ << "()\n"
              << ansi::reset;
}

void test_lwfa() {

    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;


    uint2 dims { 1024, 128 };
    float2 box { 20.48, 12.8 };

    uint2 nx { 32, 32 };
    uint2 ntiles { dims.x / nx.x, dims.y / nx.y };

    auto dt = zpic::courant( 2, dims, box ) * 0.9f;

    // Create simulation
    Simulation sim( 2, ntiles, nx, box, dt);

    // Add electrons
    Species electrons("electrons", -1.0f, make_uint3( 2, 2, 8 ));
    electrons.set_density( Density::Step( coord::z, 1.0, 20.48 ) );
//    electrons.set_density( Density::Step( coord::z, 1.0, 10.24 ) );
    sim.add_species( electrons );
    
    // Add Laser
    Laser::Gaussian laser;
    laser.start   = 20.0;
    laser.fwhm    = 2.0;
    laser.a0      = 1.0;
    laser.omega0  = 10.0;    
    laser.W0      = 4.0;
    laser.focus   = 20.48;
    laser.sin_pol = 1;

    laser.add( sim.emf );


    // Set moving window and current filtering
    sim.set_moving_window();
//    sim.current.set_filter( Filter::Compensated( coord::z, 4 ));

    auto diag = [& sim, & electrons ]( ) {
        sim.emf.save(emf::e, fcomp::z, 0);
        sim.emf.save(emf::e, fcomp::r, 0);
        sim.emf.save(emf::e, fcomp::θ, 0);

        sim.emf.save(emf::e, fcomp::z, 1);
        sim.emf.save(emf::e, fcomp::r, 1);
        sim.emf.save(emf::e, fcomp::θ, 1);

        sim.current.save( fcomp::z, 0);
        sim.current.save( fcomp::r, 0);
        sim.current.save( fcomp::θ, 0);

        sim.current.save( fcomp::z, 1 );
        sim.current.save( fcomp::r, 1 );
        sim.current.save( fcomp::θ, 1 );

    
        electrons.save_charge( 0 );
        electrons.save_charge( 1 );

    };

#if 0
    Timer timer;
                  
    timer.start();
                                 
    while (sim.get_t() < tmax )
    {
        if ( sim.get_iter() % 10 == 0 ) {
            diag();
        }

        // std::cout << "n = " << sim.get_iter() << '\n';
        sim.advance_mov_window();
    }
    
    diag();

    timer.stop();

    std::cout << "Simulation run up to t = " << sim.get_t()
              << " in " << timer.elapsed(timer::s) << " s\n";


#else

    std::cout << "Starting simulation, dt = " << dt << '\n';

    while( sim.get_t() <= 1.05 * box.x ) {
        if ( sim.get_iter() % 10 == 0 ) {
            std::cout << "i = " << sim.get_iter() << '\n';
            diag();
        }
        sim.advance_mov_window();
    }

#endif



    std::cout << ansi::bold;
    std::cout << "Done!\n";
    std::cout << ansi::reset; 
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

   // test_inj();
   // test_mov();
   // test_current();


   // test_beam();
   test_pwfa();
   // test_lwfa();
}
