#include "em2d.hpp"

#include <iostream>

#include "udist.hpp"
#include "util/term.hpp"
#include "util/timer.hpp"

#include "core/bounds.hpp"
#include "grid/grid.hpp"
#include "laser.hpp"

void test_grid( void ) {
    
    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    // Parallel partition
    uint2 partition = make_uint2( 2, 2 );

    // Global number of tiles
    // const uint2 global_ntiles = { 4, 4 };
    const uint2 global_ntiles = { 3, 2 };


    // const uint2 tile_dims     = { 12, 12 };
    const uint2 tile_dims     = { 11, 17 };

    bounds_2d<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {1,2};

    mpi::cart2d parallel( partition );

    grid::tiled<float> data( global_ntiles, tile_dims, gc, parallel );

    // Get local number of tiles
    const auto ntiles   = data.get_local_ntiles();

    uint2 const local_tile_start = data.get_local_tile_start();

    if ( mpi::root() ) {
        std::cout << "Setting values...\n";
    }

    data.zero( );
    data.set( 1.0 );

    #pragma omp parallel for
    for( unsigned int tid = 0; tid < ntiles.y * ntiles.x; tid++ ) {
        const uint2 tile_idx = { tid % ntiles.x, tid / ntiles.x  };
        float * const __restrict__ tile_data = data.tile_data(tid);

        const auto   tile_val = ( local_tile_start.y + tile_idx.y ) * global_ntiles.x + ( local_tile_start.x + tile_idx.x );

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

    parallel.barrier();
    if ( mpi::root() )
        std::cout << "Saving data...\n";

    data.save( "mpi/mpi.zdf" );

    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Done!\n";
        std::cout << ansi::reset;
    }      
}

void test_vec3grid( ) {

    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Running test_grid()\n";
        std::cout << ansi::reset;
        std::cout << "Declaring test_vec3grid::tiled<float> data...\n";
    }

    // Parallel partition
    uint2 partition = make_uint2( 2, 2 );

    const uint2 global_ntiles = { 8, 8 };
    const uint2 tile_dims = { 16,16 };
    
    bounds_2d<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {1,2};

    mpi::cart2d parallel( partition );

    grid::tiled_vec3< float > data( global_ntiles, tile_dims, gc, parallel );

    // Get local number of tiles
    const auto local_ntiles = data.get_local_ntiles();

    uint2 const local_tile_start = data.get_local_tile_start();

    // Set zero
    // data.zero( );

    // Set constant
    // data.set( float3{1.0, 2.0, 3.0} );

    // Set different value per tile
    for( unsigned int tid = 0; tid < local_ntiles.y * local_ntiles.x; tid++ ) {
        const uint2 tile_idx = { tid % local_ntiles.x, tid / local_ntiles.x  };
        auto * const __restrict__ tile_data = data.tile_data( tid );

        const auto   tile_val = ( local_tile_start.y + tile_idx.y ) * global_ntiles.x + ( local_tile_start.x + tile_idx.x );

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

    if ( mpi::root() )
        std::cout << "Saving data...\n";

    data.save( fcomp::x, "mpi/mpi-vec3-x.zdf" );
    data.save( fcomp::y, "mpi/mpi-vec3-y.zdf" );
    data.save( fcomp::z, "mpi/mpi-vec3-z.zdf" );

    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Done!\n";
        std::cout << ansi::reset;
    }
}


void test_laser( ) {

    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    // Parallel partition
//    uint2 partition = make_uint2( 1, 1 );
    uint2 partition = make_uint2( 4, 2 );

    uint2 ntiles = { 64, 16 };
    uint2 tile_dims = { 16, 16 };

    float2 box = { 20.48, 25.6 };
    double dt = 0.014;

    mpi::cart2d parallel( partition );

    em2d::emf emf( ntiles, tile_dims, box, dt, parallel );

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

    if ( mpi::root() ) {
        std::cout << "Starting test - " << niter << " iterations...\n";
    }

    timer::clock t0("test");

    t0.start();

    for( int i = 0; i < niter; i ++) {
        emf.advance( );
    }

    t0.stop();

    save_emf();

    if ( mpi::root() ) {
        t0.report( std::to_string(niter) + " iterations.");
        std::cout << ansi::bold;
        std::cout << "Done!\n";
        std::cout << ansi::reset;
    }
}

void test_inj( ) {

    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    // Parallel partition
    uint2 partition = make_uint2( 2, 2 );

    mpi::cart2d parallel( partition );

    uint2 ntiles{ 4, 4 };
    uint2 nx{ 32, 32 };

    float2 box{ 12.8, 12.8 };

    auto dt = 0.99 * em2d::simulation::courant( ntiles, nx, box );

    uint2 ppc{ 8, 8 };
    em2d::species electrons( "electrons", -1.0f, ppc );

    parallel.barrier();
    if ( mpi::root() ) std::cout << "Created species\n";

    //electrons.set_density(em2d::density::step(coord::x, 1.0, 5.0));
    // electrons.set_density(em2d::density::slab(coord::y, 1.0, 5.0, 8.0));
    electrons.set_density( em2d::density::sphere( 1.0, float2{5.0, 7.0}, 2.0 ) );

    parallel.barrier();
    if ( mpi::root() ) std::cout << "Density set\n";

    electrons.set_udist( em2d::udist::thermal( float3{ 0.1, 0.2, 0.3 }, float3{1,0,0} ) );

    electrons.initialize( box, ntiles, nx, dt, 0, parallel );

    electrons.save_charge();
    electrons.save();
    electrons.save_phasespace(
        em2d::phasespace::quantity::ux, float2{-1, 3}, 256,
        em2d::phasespace::quantity::uz, float2{-1, 1}, 128
    );

    parallel.barrier();
    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << __func__ << "() complete!\n";
        std::cout << ansi::reset;
    }
}

void test_mov( ) {

    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    // Parallel partition
    uint2 partition = make_uint2( 2, 2 );

    mpi::cart2d parallel( partition );

    uint2 ntiles{ 4, 4 };
    uint2 nx{ 32, 32 };

    float2 box{ 12.8, 12.8 };

    auto dt = 0.99 * em2d::simulation::courant( ntiles, nx, box );

    uint2 ppc{ 8, 8 };
    em2d::species electrons( "electrons", -1.0f, ppc );

    electrons.set_density( em2d::density::sphere( 1.0, float2{2.1, 2.1}, 2.0 ) );
    electrons.set_udist( em2d::udist::cold( float3{ -1, -2, -3 } ) );
    electrons.initialize( box, ntiles, nx, dt, 0, parallel );

    electrons.save_charge();
    electrons.save();

    int niter = 200; //200
    for( auto i = 0; i < niter; i ++ ) {
        auto np_global = electrons.global_np();
        if ( parallel.root() ) std::cout << "i = " << i << ", total particles: " << np_global << '\n';
        electrons.advance();
    }

    electrons.save_charge();
    electrons.save();

    parallel.barrier();
    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << __func__ << "() complete!\n";
        std::cout << ansi::reset;
    }
}

void test_current( ) {

    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    // Parallel partition
    uint2 partition = make_uint2( 2, 2 );

    mpi::cart2d parallel( partition );

    uint2 ntiles{ 4, 4 };
    uint2 nx{ 32, 32 };

    float2 box{ 12.8, 12.8 };

    auto dt = 0.99 * em2d::simulation::courant( ntiles, nx, box );

    uint2 ppc{ 8, 8 };
    em2d::species electrons( "electrons", -1.0f, ppc );

    electrons.set_density( em2d::density::sphere( 1.0, float2{6.4, 6.4}, 5.0 ) );
    electrons.set_udist( em2d::udist::cold( float3{ 1, 2, 3 } ) );

    electrons.initialize( box, ntiles, nx, dt, 0, parallel );

    em2d::current current( ntiles, nx, box, dt, parallel );

    electrons.save_charge();

    electrons.advance( current );
    current.advance( );
    
    current.save( fcomp::x );
    current.save( fcomp::y );
    current.save( fcomp::z );

    parallel.barrier();
    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << __func__ << "() complete!\n";
        std::cout << ansi::reset;
    }
}

void test_mov_window_emf( ) {

    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    // Parallel partition
//    uint2 partition = make_uint2( 1, 1 );
    uint2 partition = make_uint2( 4, 2 );

    // Simulation grid and time step
    uint2 ntiles = { 64, 16 };
    uint2 nx = { 16, 16 };

    float2 box = { 20.48, 25.6 };
    double dt = 0.014;

    // Disable periodic boundaries along x
    int2 periodic = { 0, 1 };

    // Create simulation object
    em2d::simulation sim( ntiles, nx, box, dt, partition, periodic );

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


    if ( mpi::root() ) {
        std::cout << "Starting test - " << niter << " iterations...\n";
    }

    auto t0 = timer::clock("test");

    t0.start();

    for( int i = 0; i < niter; i ++) {
        if ( i % 100 == 0 ) save_emf();
        sim.advance();
    }
        save_emf();

    t0.stop();

    if ( mpi::root() ) {
        t0.report( std::to_string(niter) + " iterations.");

        std::cout << ansi::bold;
        std::cout << "Done!\n";
        std::cout << ansi::reset;
    }
}


void test_mov_window_inj( ) {

    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    // Parallel partition
    uint2 partition = make_uint2( 1, 1 );
//    uint2 partition = make_uint2( 1, 2 );
//    uint2 partition = make_uint2( 2, 1 );
//    uint2 partition = make_uint2( 2, 2 );

    // Simulation grid and time step
    uint2 ntiles{ 48,8 };
    uint2 nx{ 32, 32 };

    float2 box{ 30.72, 25.6 };

    auto dt = 0.99 * em2d::simulation::courant( ntiles, nx, box );

    // Disable periodic boundaries along x
    int2 periodic = { 0, 1 };

    // Create simulation object
    em2d::simulation sim( ntiles, nx, box, dt, partition, periodic );

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

    if ( mpi::root() ) {
        std::cout << "Starting test - " << niter << " iterations...\n";
    }

    auto t0 = timer::clock("test");

    t0.start();

    for( int i = 0; i < niter; i ++) {
        if ( sim.get_iter() % 100 == 0 )
            electrons.save_charge();
        sim.advance_mov_window();
    }

    electrons.save_charge();

    t0.stop();

    if ( mpi::root() ) {
        t0.report( std::to_string(niter) + " iterations.");

        std::cout << ansi::bold;
        std::cout << "Done!\n";
        std::cout << ansi::reset;
    }
}

void test_lwfa()
{
    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    // Parallel partition
//    uint2 partition = make_uint2( 1, 1 );
    uint2 partition = make_uint2( 2, 2 );

    // Simulation grid and time step
    uint2 ntiles{ 48,8 };
    uint2 nx{ 32, 32 };

    float2 box{ 30.72, 25.6 };

    auto dt = 0.99 * em2d::simulation::courant( ntiles, nx, box );

    if ( mpi::root() ) {
        std::cout << "(*info*) dt = " << dt << '\n';
    }

    // Disable periodic boundaries along x
    int2 periodic = { 0, 1 };

    // Create simulation object
    em2d::simulation sim( ntiles, nx, box, dt, partition, periodic );

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
    
    if ( mpi::root() ) {
        std::cout << "(*info*) Added " << laser << '\n';
    }

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
            if ( sim.parallel.root() ) std::cout << "i = " << sim.get_iter() << '\n';
            save();
        }
        sim.advance_mov_window();
    }

    save();

    t0.stop();

    if ( mpi::root() ) {
        t0.report( std::to_string(sim.get_iter()) + " iterations.");
        
        std::cout << ansi::bold;
        std::cout << "Done!\n";
        std::cout << ansi::reset;
    }

}
void test_weibel( ) {
   
    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    // Parallel partition, must be 1 along x
//    uint2 partition = make_uint2( 1, 1 );
    uint2 partition = make_uint2( 2, 2 );

    // Create simulation box
    uint2 ntiles{16, 16};
    uint2 tile_dims{32, 32};
    float2 box = {tile_dims.x * ntiles.x * 0.1f, tile_dims.y * ntiles.y * 0.1f};
    float dt = 0.07;
                                        
    em2d::simulation sim( ntiles, tile_dims, box, dt, partition );
                            
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
        
    if ( sim.parallel.root() )
        std::cout << "Running large Weibel test up to n = " << imax << "...\n";
                
    timer::clock timer;
                  
    timer.start();

    while (sim.get_iter() < imax)
    {     
        if ( sim.get_iter() % 50 == 0 ) {
            diag();
            if ( sim.parallel.root() ) 
                std::cout << "i = " << sim.get_iter() << '\n';    
        }       
        sim.advance();
    }

    timer.stop();

    diag();

    auto nmove = sim.get_nmove();
    if ( sim.parallel.root() ) {
        std::cout << "simulation complete at i = " << sim.get_iter() << '\n';
        auto time = timer.elapsed(timer::units::s);
        std::cout << "Elapsed time: " << time << " s\n";
        auto perf = nmove / time / 1.e9;
        std::cerr << "Performance : " << perf << " GPart/s\n";
    }                  
}

int main( int argc, char *argv[] ) {

    // Initialize the MPI environment
    mpi::init( & argc, & argv );

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

    test_weibel();
    // test_lwfa();

    // Finalize the MPI environment
    mpi::finalize();

}
