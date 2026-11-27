#include <iostream>

#include "em2d.hpp"

#include "udist.hpp"
#include "util/term.hpp"
#include "util/timer.hpp"

#include "core/bounds.hpp"
#include "grid/grid.hpp"
#include "laser.hpp"

void test_grid( void ) {
    
    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;

    // Global number of tiles
    // const uint2 ntiles = { 4, 4 };
    const uint2 ntiles = { 3, 2 };


    // const uint2 tile_dims     = { 12, 12 };
    const uint2 tile_dims     = { 11, 17 };

    bounds_2d<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {1,2};

    int2 periodic{1,1};

    grid::tiled<float> data( ntiles, tile_dims, gc, periodic );

    std::cout << "Setting values...\n";

    data.zero( );
    data.set( 1.0 );

    #pragma omp parallel for
    for( unsigned int tid = 0; tid < ntiles.y * ntiles.x; tid++ ) {
        const uint2 tile_idx = { tid % ntiles.x, tid / ntiles.x  };
        float * const __restrict__ tile_data = data.tile_data(tid);

        const auto   tile_val = ( tile_idx.y ) * ntiles.x + ( tile_idx.x );

        for( unsigned int idx = 0; idx < tile_dims.y * tile_dims.x; idx ++ ) {
            const auto ix = idx % tile_dims.x;
            const auto iy = idx / tile_dims.x; 
            tile_data[iy * data.tile_ext_dims.x + ix] = tile_val;
        }
    }
    
    data.add_from_gc();
    data.copy_to_gc();

    for( auto i = 0; i < 5; i++)
       data.x_shift_left( 1 );

    data.kernel3_x( 1., 2., 1. );
    data.kernel3_y( 1., 2., 1. );

    std::cout << "Saving data...\n";

    data.save( "mpi/mpi.zdf" );

    std::cout << ansi::bold;
    std::cout << "Done!\n";
    std::cout << ansi::reset;
}

void test_vec3grid( ) {

    std::cout << ansi::bold;
    std::cout << "Running test_grid()\n";
    std::cout << ansi::reset;
    std::cout << "Declaring test_vec3grid::tiled<float> data...\n";

    const uint2 ntiles = { 8, 8 };
    const uint2 tile_dims = { 16,16 };
    
    bounds_2d<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {1,2};

    int2 periodic{1,1};

    grid::tiled_vec3< float > data( ntiles, tile_dims, gc, periodic );

    // Set zero
    // data.zero( );

    // Set constant
    // data.set( float3{1.0, 2.0, 3.0} );

    // Set different value per tile
    #pragma omp parallel for
    for( unsigned int tid = 0; tid < ntiles.y * ntiles.x; tid++ ) {
        const uint2 tile_idx = { tid % ntiles.x, tid / ntiles.x  };
        auto * const __restrict__ tile_data = data.tile_data( tid );

        const auto   tile_val = ( tile_idx.y ) * ntiles.x + ( tile_idx.x );

        for( unsigned int idx = 0; idx < tile_dims.y * tile_dims.x; idx += 1 ) {
            const auto iy = idx / tile_dims.x; 
            const auto ix = idx % tile_dims.x;
            tile_data[iy * data.tile_ext_dims.x + ix] = make_float3( 1 + tile_val, 2 + tile_val, 3 + tile_val );
        }
    }
    data.copy_to_gc( );

    data.add_from_gc( );
    data.copy_to_gc( );

    for( int i = 0; i < 5; i++ ) {
        data.x_shift_left( 1 );
    }

    data.kernel3_x( 1., 2., 1. );
    data.kernel3_y( 1., 2., 1. );

    std::cout << "Saving data...\n";

    data.save( fcomp::x, "mpi/mpi-vec3-x.zdf" );
    data.save( fcomp::y, "mpi/mpi-vec3-y.zdf" );
    data.save( fcomp::z, "mpi/mpi-vec3-z.zdf" );

    std::cout << ansi::bold;
    std::cout << "Done!\n";
    std::cout << ansi::reset;
}


void test_laser( ) {

    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;

    uint2 ntiles = { 64, 16 };
    uint2 tile_dims = { 16, 16 };

    float2 box = { 20.48, 25.6 };
    double dt = 0.014;

    int2 periodic{1,1};


    em2d::emf emf( ntiles, tile_dims, box, dt, periodic );

    auto save_emf = [ & emf ]( ) {
        emf.save( em2d::emf::quantity::e, fcomp::x );
        emf.save( em2d::emf::quantity::e, fcomp::y );
        emf.save( em2d::emf::quantity::e, fcomp::z );

        emf.save( em2d::emf::quantity::b, fcomp::x );
        emf.save( em2d::emf::quantity::b, fcomp::y );
        emf.save( em2d::emf::quantity::b, fcomp::z );
    };

/*
    em2d::laser::plane_wave laser;
    laser.start = 10.2;
    laser.fwhm = 4.0;
    laser.a0 = 1.0;
    laser.omega0 = 10.0;
*/

    em2d::laser::gaussian laser;
    laser.start = 10.2;
    laser.fwhm = 4.0;
    laser.a0 = 1.0;
    laser.omega0 = 10.0;
    laser.W0 = 1.5;
    laser.focus = 20.48;
    laser.axis = 12.8;

    laser.sin_pol = 0;
    laser.cos_pol = 1;

    laser.add( emf );

    save_emf();

    int niter = 20.48 / dt / 2;

    std::cout << "Starting test - " << niter << " iterations...\n";

    timer::clock t0("test");

    t0.start();

    for( int i = 0; i < niter; i ++) {
        emf.advance( );
    }

    t0.stop();

    save_emf();

    t0.report( std::to_string(niter) + " iterations.");
    std::cout << ansi::bold;
    std::cout << "Done!\n";
    std::cout << ansi::reset;
}

void test_inj( ) {

    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;

    int2 periodic{1,1};

    uint2 ntiles{ 4, 4 };
    uint2 nx{ 32, 32 };

    float2 box{ 12.8, 12.8 };

    auto dt = 0.99 * em2d::simulation::courant( ntiles, nx, box );

    uint2 ppc{ 8, 8 };
    em2d::species electrons( "electrons", -1.0f, ppc );

    std::cout << "Created species\n";

    //electrons.set_density(em2d::density::step(coord::x, 1.0, 5.0));
    // electrons.set_density(em2d::density::slab(coord::y, 1.0, 5.0, 8.0));
    electrons.set_density( em2d::density::sphere( 1.0, float2{5.0, 7.0}, 2.0 ) );

    std::cout << "Density set\n";

    electrons.set_udist( em2d::udist::thermal( float3{ 0.1, 0.2, 0.3 }, float3{1,0,0} ) );

    electrons.initialize( box, ntiles, nx, dt, 0, periodic );

    electrons.save_charge();
    electrons.save();
    electrons.save_phasespace(
        em2d::phasespace::quantity::ux, float2{-1, 3}, 256,
        em2d::phasespace::quantity::uz, float2{-1, 1}, 128
    );

    std::cout << ansi::bold;
    std::cout << __func__ << "() complete!\n";
    std::cout << ansi::reset;
}

void test_mov( ) {

    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;

    int2 periodic{1,1};

    uint2 ntiles{ 4, 4 };
    uint2 nx{ 32, 32 };

    float2 box{ 12.8, 12.8 };

    auto dt = 0.99 * em2d::simulation::courant( ntiles, nx, box );

    uint2 ppc{ 8, 8 };
    em2d::species electrons( "electrons", -1.0f, ppc );

    electrons.set_density( em2d::density::sphere( 1.0, float2{2.1, 2.1}, 2.0 ) );
    electrons.set_udist( em2d::udist::cold( float3{ -1, -2, -3 } ) );
    electrons.initialize( box, ntiles, nx, dt, 0, periodic );

    electrons.save_charge();
    electrons.save();

    int niter = 200; //200
    for( auto i = 0; i < niter; i ++ ) {
        auto np = electrons.total_np();
        std::cout << "i = " << i << ", total particles: " << np << '\n';
        electrons.advance();
    }

    electrons.save_charge();
    electrons.save();

    std::cout << ansi::bold;
    std::cout << __func__ << "() complete!\n";
    std::cout << ansi::reset;
}

void test_current( ) {

    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;

    int2 periodic{1,1};

    uint2 ntiles{ 4, 4 };
    uint2 nx{ 32, 32 };

    float2 box{ 12.8, 12.8 };

    auto dt = 0.99 * em2d::simulation::courant( ntiles, nx, box );

    uint2 ppc{ 8, 8 };
    em2d::species electrons( "electrons", -1.0f, ppc );

    electrons.set_density( em2d::density::sphere( 1.0, float2{6.4, 6.4}, 5.0 ) );
    electrons.set_udist( em2d::udist::cold( float3{ 1, 2, 3 } ) );

    electrons.initialize( box, ntiles, nx, dt, 0, periodic );

    em2d::current current( ntiles, nx, box, dt, periodic );

    electrons.save_charge();

    electrons.advance( current );
    current.advance( );
    
    current.save( fcomp::x );
    current.save( fcomp::y );
    current.save( fcomp::z );

    std::cout << ansi::bold;
    std::cout << __func__ << "() complete!\n";
    std::cout << ansi::reset;
}

void test_mov_window_emf( ) {

    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;

    // Disable periodic boundaries along x
    int2 periodic{0,1};

    // Simulation grid and time step
    uint2 ntiles = { 64, 16 };
    uint2 nx = { 16, 16 };

    float2 box = { 20.48, 25.6 };
    double dt = 0.014;

    // Create simulation object
    em2d::simulation sim( ntiles, nx, box, dt, periodic );

    // Set moving window along x
    sim.set_moving_window();


    auto save_emf = [ & sim ]( ) {
        sim.emf.save( em2d::emf::quantity::e, fcomp::x );
        sim.emf.save( em2d::emf::quantity::e, fcomp::y );
        sim.emf.save( em2d::emf::quantity::e, fcomp::z );

        sim.emf.save( em2d::emf::quantity::b, fcomp::x );
        sim.emf.save( em2d::emf::quantity::b, fcomp::y );
        sim.emf.save( em2d::emf::quantity::b, fcomp::z );
    };

    em2d::laser::gaussian laser;
    laser.start = 20.0;
    laser.fwhm = 4.0;
    laser.a0 = 1.0;
    laser.omega0 = 10.0;
    laser.W0 = 1.5;
    laser.focus = 20.48;
    laser.axis = 12.8;

    laser.sin_pol = 0;
    laser.cos_pol = 1;

    laser.add( sim.emf );

    save_emf();

    int niter = 20.48 / dt / 2 ;
    // int niter = 0;


    std::cout << "Starting test - " << niter << " iterations...\n";

    auto t0 = timer::clock("test");

    t0.start();

    for( int i = 0; i < niter; i ++) {
        if ( i % 100 == 0 ) save_emf();
        sim.advance();
    }
        save_emf();

    t0.stop();

    t0.report( std::to_string(niter) + " iterations.");

    std::cout << ansi::bold;
    std::cout << "Done!\n";
    std::cout << ansi::reset;
}


void test_mov_window_inj( ) {

    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;

    // Disable periodic boundaries along x
    int2 periodic = { 0, 1 };


    // Simulation grid and time step
    uint2 ntiles{ 48,8 };
    uint2 nx{ 32, 32 };

    float2 box{ 30.72, 25.6 };

    auto dt = 0.99 * em2d::simulation::courant( ntiles, nx, box );


    // Create simulation object
    em2d::simulation sim( ntiles, nx, box, dt, periodic );

    // 
    uint2 ppc{ 8, 8 };
    em2d::species electrons( "electrons", -1.0f, ppc );
    electrons.set_density(em2d::density::step(coord::x, 1.0, box.x - 0.25));
    sim.add_species( electrons );

    // Set moving window along x
    // Currently this must happen after sim.add_species()
    sim.set_moving_window();

    electrons.save_charge();

    int niter = ( box.x + 1.0 ) / dt;

    std::cout << "Starting test - " << niter << " iterations...\n";

    auto t0 = timer::clock("test");

    t0.start();

    for( int i = 0; i < niter; i ++) {
        if ( sim.get_iter() % 100 == 0 )
            electrons.save_charge();
        sim.advance_mov_window();
    }

    electrons.save_charge();

    t0.stop();

    t0.report( std::to_string(niter) + " iterations.");

    std::cout << ansi::bold;
    std::cout << "Done!\n";
    std::cout << ansi::reset;
}

void test_lwfa()
{
    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;

    // Disable periodic boundaries along x
    int2 periodic = { 0, 1 };

    // Simulation grid and time step
    uint2 ntiles{ 48,8 };
    uint2 nx{ 32, 32 };

    float2 box{ 30.72, 25.6 };

    auto dt = 0.99 * em2d::simulation::courant( ntiles, nx, box );

    std::cout << "(*info*) dt = " << dt << '\n';

    // Create simulation object
    em2d::simulation sim( ntiles, nx, box, dt, periodic );

    // Add background plasma
    uint2 ppc{ 4, 4 };
    em2d::species electrons( "electrons", -1.0f, ppc );
    electrons.set_density(em2d::density::step(coord::x, 1.0, box.x - 0.25));
    sim.add_species( electrons );
    
    // Add laser pulse
    em2d::laser::gaussian laser;
    laser.start = box.x - .25;
    laser.fwhm = 2.0;
    laser.a0 = 2.0;
    laser.omega0 = 10.0;
    laser.W0 = 4.0;
    laser.focus = box.x;
    laser.axis = box.y / 2;
    
    laser.sin_pol = 1;
    laser.cos_pol = 0;

    laser.add( sim.emf );
    
    std::cout << "(*info*) Added " << laser << '\n';

    // Set moving window and current filtering
    sim.set_moving_window();
    sim.current.set_filter(em2d::filter::compensated(em2d::filter::axis::x, 4));

    // Diagnostics
    auto save = [ & sim, & electrons ]( ) {
        sim.emf.save( em2d::emf::quantity::e, fcomp::x );
        sim.emf.save( em2d::emf::quantity::e, fcomp::y );
        sim.emf.save( em2d::emf::quantity::e, fcomp::z );

        sim.current.save( fcomp::x );
        sim.current.save( fcomp::y );
        sim.current.save( fcomp::z );

        electrons.save_charge();
    };


    auto t0 = timer::clock("test");

    t0.start();

    // Run simulation dumping diagnostics at every 100 timesteps
    while ( sim.get_t() <= box.x * 1.5 ) {
        if ( sim.get_iter() % 100 == 0 ) {
            std::cout << "i = " << sim.get_iter() << '\n';
            save();
        }
        sim.advance_mov_window();
    }

    save();

    t0.stop();

    t0.report( std::to_string(sim.get_iter()) + " iterations.");
    
    std::cout << ansi::bold;
    std::cout << "Done!\n";
    std::cout << ansi::reset;

}
void test_weibel( ) {
   
    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;

    int2 periodic{1,1};

    // Create simulation box
    uint2 ntiles{16, 16};
    uint2 tile_dims{32, 32};
    float2 box = {tile_dims.x * ntiles.x * 0.1f, tile_dims.y * ntiles.y * 0.1f};
    float dt = 0.07;
                                        
    em2d::simulation sim( ntiles, tile_dims, box, dt, periodic);
                            
    uint2 ppc{4, 4};

    em2d::species electrons("electrons", -1.0f, ppc);
    electrons.set_udist(
        em2d::udist::thermal_corr( 
            float3{ 0.1, 0.1, 0.1 },
            float3{ 0, 0, 0.6 }
        )
    );
    electrons.push_type = em2d::species::pusher::euler;

    sim.add_species( electrons );

    em2d::species positrons("positrons", +1.0f, ppc);
    positrons.set_udist(
        em2d::udist::thermal_corr( 
            float3{ 0.1, 0.1, 0.1 },
            float3{ 0, 0, -0.6 }
        )
    );
    positrons.push_type = em2d::species::pusher::euler;

    sim.add_species( positrons );

    // Disable diagnostics for performance benchmark
#if 1
    // Lambda function for diagnostic output
    auto diag = [ & ]( ) {
        sim.emf.save(em2d::emf::quantity::b, fcomp::x);
        sim.emf.save(em2d::emf::quantity::b, fcomp::y);
        sim.emf.save(em2d::emf::quantity::b, fcomp::z);

        electrons.save_charge();
        positrons.save_charge();

        electrons.save_phasespace(
            em2d::phasespace::quantity::ux, float2{-2, 2}, 256,
            em2d::phasespace::quantity::uy, float2{-2, 2}, 256
        );

        sim.energy_info();
    };
#else
    auto diag = [ ]( ) {};
#endif

    // Run simulation    
    int const imax = 500;
        
    std::cout << "Running large Weibel test up to n = " << imax << "...\n";
                
    timer::clock timer;
                  
    timer.start();

    while (sim.get_iter() < imax)
    {     
        if ( sim.get_iter() % 50 == 0 ) {
            diag();
            std::cout << "i = " << sim.get_iter() << '\n';    
        }       
        sim.advance();
    }

    timer.stop();

    diag();

    auto nmove = sim.get_nmove();

    std::cout << "simulation complete at i = " << sim.get_iter() << '\n';
    auto time = timer.elapsed(timer::units::s);
    std::cout << "Elapsed time: " << time << " s\n";
    auto perf = nmove / time / 1.e9;
    std::cerr << "Performance : " << perf << " GPart/s\n";
}

int main( int argc, char *argv[] ) {

    // Initialize SIMD support
    simd::init();

    em2d::sys_info();

    // test_grid( );  
    // test_vec3grid( );
    // test_laser( );
    // test_inj( );
    // test_mov( );
    // test_current( );
    // test_mov_window_emf();
    // test_mov_window_inj();

    // test_weibel();
    test_lwfa();
}
