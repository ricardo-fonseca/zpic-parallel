#include <unistd.h> // For getopt
#include <iostream>

#include "core/gpu.cuh"
#include "grid/tiled.cuh"
#include "grid/tiled_vec3.cuh"

#include "parallel/mpi.hpp"
#include "util/term.hpp"

#include "core/bounds.hpp"
#include "core/vec_types.cuh"
#include "grid/grid.cuh"

#include "zpic.hpp"

namespace kernel {

__global__
void test_tiled( 
    grid::tiled_view<float> tiles,
    uint2 const global_ntiles, uint2 const local_tile_start )
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    auto * const __restrict__ tile_data = tiles.tile_data( tile_idx );

    const float tile_val = ( local_tile_start.y + tile_idx.y ) * global_ntiles.x + 
                           ( local_tile_start.x + tile_idx.x );

    for( auto idx = gpu::block::thread_rank(); 
        idx < tiles.tile_dims.y * tiles.tile_dims.x; 
        idx += gpu::block::num_threads() ) {
        const auto iy = idx / tiles.tile_dims.x; 
        const auto ix = idx % tiles.tile_dims.x;
        tile_data[iy * tiles.tile_ystride() + ix] = tile_val;
    }
}

} // namespace kernel

void test_tiled_grid( void ) {
    
    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    // Parallel partition
    // uint2 partition = make_uint2( 1, 4 );
    // uint2 partition = make_uint2( 4, 1 );
    uint2 partition = make_uint2( 2, 2 );

    // Global number of tiles
    const uint2 global_ntiles = { 4, 4 };
    //const uint2 global_ntiles = { 3, 2 };


    const uint2 tile_dims     = { 12, 12 };

    bounds_2d<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {1,2};

    mpi::cart2d parallel( partition );

    grid::tiled<float> data( global_ntiles, tile_dims, gc, parallel );
    data.name = "Test grid";

    // Get local number of tiles
    const auto ntiles   = data.get_local_ntiles();

    const uint2 local_tile_start = data.get_local_tile_start();

    if ( mpi::root() ) {
        std::cout << "Setting values...\n";
    }

    data.zero( );
    data.set( 32.0 );

    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 64 );

    kernel::test_tiled <<< grid, block >>>
        (data.view(), global_ntiles, local_tile_start );

    data.add_from_gc();
    data.copy_to_gc();

    for( auto i = 0; i < 5; i++)
       data.x_shift_left( 1 );

    data.kernel3_x( 1., 2., 1. );
    data.kernel3_y( 1., 2., 1. );

    if ( mpi::root() )
        std::cout << "Saving data...\n";

    data.save( "cuda/cuda.zdf" );
    gpu::device::check();

    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Done!\n";
        std::cout << ansi::reset;
    }
        
}

namespace kernel {

__global__
void test_vec3( 
    grid::tiled_vec3_view<float> tiles,
    uint2 const global_ntiles, uint2 const local_tile_start )
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    auto * const __restrict__ tile_data = tiles.tile_data( tile_idx );

    const float tile_val = ( local_tile_start.y + tile_idx.y ) * global_ntiles.x + 
                           ( local_tile_start.x + tile_idx.x );

    for( auto idx = gpu::block::thread_rank(); 
        idx < tiles.tile_dims.y * tiles.tile_dims.x; 
        idx += gpu::block::num_threads() ) {
        const auto iy = idx / tiles.tile_dims.x; 
        const auto ix = idx % tiles.tile_dims.x;
        tile_data[iy * tiles.tile_ystride() + ix] = make_float3( 1 + tile_val, 2 + tile_val, 3 + tile_val );;
    }
}

}

void test_tiled_vec3_grid( ) {

    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Running test_grid()\n";
        std::cout << ansi::reset;
        std::cout << "Declaring test_vec3grid<float> data...\n";
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
    data.zero( );

    // Set constant
    // data.set( float3{1.0, 2.0, 3.0} );

    // Set different value per tile
    dim3 grid( local_ntiles.x, local_ntiles.y );
    dim3 block( 64 );
    kernel::test_vec3 <<< grid, block >>>
        (data.view(), global_ntiles, local_tile_start );

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

    data.save( fcomp::x, "cuda/mpi-vec3-x.zdf" );
    data.save( fcomp::y, "cuda/mpi-vec3-y.zdf" );
    data.save( fcomp::z, "cuda/mpi-vec3-z.zdf" );

    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Done!\n";
        std::cout << ansi::reset;
    }
}

#include "emf.hpp"
#include "laser.hpp"

#include "util/timer.hpp"

void test_laser( ) {

    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    // Parallel partition
//    uint2 partition = make_uint2( 1, 1 );
//    uint2 partition = make_uint2( 1, 2 );
//    uint2 partition = make_uint2( 4, 1 );
    uint2 partition = make_uint2( 4, 2 );

    uint2 ntiles = { 64, 16 };
    uint2 nx = { 16, 16 };

    float2 box = { 20.48, 25.6 };
    double dt = 0.014;

    mpi::cart2d parallel( partition );

    emf emf( ntiles, nx, box, dt, parallel );

    auto save_emf = [ & emf ]( ) {
        emf.save( emf::quantity::e, fcomp::x );
        emf.save( emf::quantity::e, fcomp::y );
        emf.save( emf::quantity::e, fcomp::z );

        emf.save( emf::quantity::b, fcomp::x );
        emf.save( emf::quantity::b, fcomp::y );
        emf.save( emf::quantity::b, fcomp::z );
    };

/*
    laser::plane_wave laser;
    laser.start = 10.2;
    laser.fwhm = 4.0;
    laser.a0 = 1.0;
    laser.omega0 = 10.0;
*/

    laser::gaussian laser;
    laser.start = 10.2;
    laser.fwhm = 4.0;
    laser.a0 = 1.0;
    laser.omega0 = 10.0;
    laser.W0 = 1.5;
    laser.focus = 20.48;
    laser.axis = 12.8;

    laser.sin_pol = 0;
    laser.cos_pol = 1;

    if ( mpi::root() ) std::cout << "Adding laser...\n";
    laser.add( emf );

    if ( mpi::root() ) std::cout << "Saving initial fields...\n";
    save_emf();

    int niter = 20.48 / dt / 2;
    // int niter{ 10 };

    if ( mpi::root() ) std::cout << "Starting test - " << niter << " iterations...\n";

    timer::clock t0("test");

    t0.start();

    for( int i = 0; i < niter; i ++) {
        emf.advance( );
    }

    t0.stop();
    
    save_emf();

    if ( mpi::root() ) {
        std::cout << niter << " iterations completed in " << t0 << '\n';
    }

    if ( mpi::root() ) {
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

    auto dt = 0.99 * zpic::simulation::courant( ntiles, nx, box );

    uint2 ppc{ 8, 8 };
    species electrons( "electrons", -1.0f, ppc );

    parallel.barrier();
    if ( mpi::root() ) std::cout << "Created species\n";

    // electrons.set_density(density::step(coord::x, 1.0, 5.0));
    // electrons.set_density(density::slab(coord::y, 1.0, 5.0, 8.0));
    electrons.set_density( density::sphere( 1.0, float2{5.0, 7.0}, 2.0 ) );

    electrons.set_udist( udist::thermal( float3{ 0.1, 0.2, 0.3 }, float3{1,0,0} ) );

    parallel.barrier();
    if ( mpi::root() ) std::cout << "Initializing species...\n";

    electrons.initialize( box, ntiles, nx, dt, 0, parallel );

    parallel.barrier();
    if ( mpi::root() ) std::cout << "Species initialized.\n";

    electrons.save_charge();

    electrons.save();
    electrons.save_phasespace(
        phasespace::quantity::ux, float2{-1, 3}, 256,
        phasespace::quantity::uz, float2{-1, 1}, 128
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

    auto dt = 0.99 * zpic::simulation::courant( ntiles, nx, box );

    uint2 ppc{ 8, 8 };
    species electrons( "electrons", -1.0f, ppc );

    electrons.set_density( density::sphere( 1.0, float2{2.1, 2.1}, 2.0 ) );
    electrons.set_udist( udist::cold( float3{ -1, -2, -3 } ) );
    electrons.initialize( box, ntiles, nx, dt, 0, parallel );

    electrons.save_charge();
    electrons.save();

    int niter = 200; //200
    for( auto i = 0; i < niter; i ++ ) {
        auto global_np = electrons.global_np();
        if ( parallel.root() ) std::cout << "i = " << i << ", total particles: " << global_np << '\n';
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
    uint2 partition = make_uint2( 1, 4 );

    mpi::cart2d parallel( partition );

    uint2 ntiles{ 4, 4 };
    uint2 nx{ 32, 32 };

    float2 box{ 12.8, 12.8 };

    auto dt = 0.99 * zpic::simulation::courant( ntiles, nx, box );

    uint2 ppc{ 8, 8 };
    species electrons( "electrons", -1.0f, ppc );

    electrons.set_density( density::sphere( 1.0, float2{6.4, 6.4}, 5.0 ) );
    electrons.set_udist( udist::cold( float3{ 1, 2, 3 } ) );

    electrons.initialize( box, ntiles, nx, dt, 0, parallel );

    current current( ntiles, nx, box, dt, parallel );

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

void test_weibel( )
{
   
    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    // Parallel partition, must be 1 along x
    uint2 partition = make_uint2( 1, 4 );

    // Create simulation box
    uint2 ntiles{16, 16};
    uint2 nx{32, 32};                                                                                                                                                                     
    float2 box = {nx.x * ntiles.x * 0.1f, nx.y * ntiles.y * 0.1f};
    float dt = 0.07;
                                        
    zpic::simulation sim( ntiles, nx, box, dt, partition );
                            
    uint2 ppc{4, 4};

    species electrons("electrons", -1.0f, ppc);
    electrons.set_udist(
        udist::thermal_corr( 
            float3{ 0.1, 0.1, 0.1 },
            float3{ 0, 0, 0.6 }
        )
    );
    electrons.push_type = species::pusher::euler;

    sim.add_species( electrons );

    species positrons("positrons", +1.0f, ppc);
    positrons.set_udist(
        udist::thermal_corr( 
            float3{ 0.1, 0.1, 0.1 },
            float3{ 0, 0, -0.6 }
        )
    );
    positrons.push_type = species::pusher::euler;

    sim.add_species( positrons );

    // Lambda function for diagnostic output
    auto diag = [ & ]( ) {
        sim.emf.save(emf::quantity::b, fcomp::x);
        sim.emf.save(emf::quantity::b, fcomp::y);
        sim.emf.save(emf::quantity::b, fcomp::z);

        electrons.save_charge();
        positrons.save_charge();

        electrons.save_phasespace(
            phasespace::quantity::ux, float2{-2, 2}, 256,
            phasespace::quantity::uy, float2{-2, 2}, 256
        );

        sim.energy_info();
    };

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
        std::cout << "Performance : " << perf << " GPart/s\n";
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
    zpic::simulation sim( ntiles, nx, box, dt, partition, periodic );

    // Set moving window along x
    sim.set_moving_window();


    auto save_emf = [ & sim ]( ) {
        sim.emf.save( emf::quantity::e, fcomp::x );
        sim.emf.save( emf::quantity::e, fcomp::y );
        sim.emf.save( emf::quantity::e, fcomp::z );

        sim.emf.save( emf::quantity::b, fcomp::x );
        sim.emf.save( emf::quantity::b, fcomp::y );
        sim.emf.save( emf::quantity::b, fcomp::z );
    };

    laser::gaussian laser;
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
        std::cout << niter << " iterations in "
                  << t0.elapsed(timer::units::s) << " s.\n";

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
//    uint2 partition = make_uint2( 1, 1 );
//    uint2 partition = make_uint2( 1, 2 );
//    uint2 partition = make_uint2( 2, 1 );
    uint2 partition = make_uint2( 2, 2 );

    // Simulation grid and time step
    uint2 ntiles{ 48,8 };
    uint2 nx{ 32, 32 };

    float2 box{ 30.72, 25.6 };

    auto dt = 0.99 * zpic::simulation::courant( ntiles, nx, box );

    // Disable periodic boundaries along x
    int2 periodic = { 0, 1 };

    if ( mpi::root() ) std::cout << "Creating simulation object...\n";
    zpic::simulation sim( ntiles, nx, box, dt, partition, periodic );

    if ( mpi::root() ) std::cout << "Creating electron species...\n";
    uint2 ppc{ 1, 1 };
    species electrons( "electrons", -1.0f, ppc );

    if ( mpi::root() ) std::cout << "Setting species density...\n";
    // electrons.set_density(Density::Step(coord::x, 1.0, box.x - 0.25));
    electrons.set_density( density::sphere( 1.0, float2{ box.x, 12.8}, 6.4 ) );

    if ( mpi::root() ) std::cout << "Adding electron species to simulation...\n";
    sim.add_species( electrons );

    if ( mpi::root() ) std::cout << "Turning moving window on...\n";

    // Set moving window along x
    sim.set_moving_window();

    electrons.save_charge();

    int niter = ( box.x + 1.0 ) / dt;

    if ( mpi::root() )
        std::cout << "Starting test - " << niter << " iterations...\n";

    auto t0 = timer::clock("test");

    t0.start();

    for( int i = 0; i < niter; i ++) {
        if ( sim.get_iter() % 100 == 0 )
            electrons.save_charge();
        sim.advance_mov_window();

        auto np_global = electrons.global_np();
        if ( sim.parallel.root() ) 
            std::cout << "# particles in sim: " << np_global << '\n';
    }

    electrons.save_charge();

    t0.stop();

    if ( mpi::root() ) {
        std::cout << niter << " iterations in "
                  << t0.elapsed(timer::units::s) << " s.\n";

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
//    uint2 partition = make_uint2( 2, 2 );
    uint2 partition = make_uint2( 4, 2 );

    // Simulation grid and time step
    uint2 ntiles{ 48,8 };
    uint2 nx{ 32, 32 };

    float2 box{ 30.72, 25.6 };

    auto dt = 0.99 * zpic::simulation::courant( ntiles, nx, box );

    if ( mpi::root() ) 
        std::cout << "(*info*) dt = " << dt << '\n';

    // Disable periodic boundaries along x
    int2 periodic = { 0, 1 };

    // Create simulation object
    zpic::simulation sim( ntiles, nx, box, dt, partition, periodic );

    // Add background plasma
    uint2 ppc{ 4, 4 };
    species electrons( "electrons", -1.0f, ppc );
    electrons.set_density(density::step(coord::x, 1.0, box.x - 0.25));
    sim.add_species( electrons );
    
    // Add laser pulse
    laser::gaussian laser;
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
    
    if ( mpi::root() )
        std::cout << "(*info*) Added " << laser << '\n';

    // Set moving window and current filtering
    sim.set_moving_window();
    // sim.current.set_filter(Filter::Compensated(coord::x, 4));

    // Diagnostics
    auto save = [ & sim, & electrons ]( ) {
        sim.emf.save( emf::quantity::e, fcomp::x );
        sim.emf.save( emf::quantity::e, fcomp::y );
        sim.emf.save( emf::quantity::e, fcomp::z );

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
        std::cout << ansi::bold;
        std::cout << "Done!\n";
        std::cout << ansi::reset;
        std::cout << "After " << sim.get_iter() << " iterations,";
        t0.report();
    }

}

#if 0

void test_weibel( std::string param ) {
    
    // param is expected to be in the form "px, py, ntx, nty"
    // Where (px,py) are the parallel MPI dimensions and (ntx,nty) the global number of tiles

    // Parse parameters
    uint2 partition { 0 };
    uint2 ntiles { 0 };

    if ( std::sscanf( param.c_str(), "%d, %d, %d, %d", 
        &partition.x, &partition.y, 
        &ntiles.x, &ntiles.y ) != 4 ) {
        
        if ( mpi::world_root() ) 
            std::cerr << "Invalid test parameters: '" << param << "'\n";
        mpi::abort(1);
    };

    if ( mpi::world_root() ) {
        std::cout << ansi::bold;
        std::cout << "Running Weibel test\n";
        std::cout << ansi::reset;
        std::cout << "MPI partition : " << partition << '\n';
        std::cout << "Global tiles  : " << ntiles << '\n';
    }


    // Create simulation box
    uint2 nx {32, 32};
    uint2 ppc {8, 8};
                                                                                                                                                                      
    float2 box {nx.x * ntiles.x * 0.1f, nx.y * ntiles.y * 0.1f};
                    
    float dt = 0.07;
                                        
    Simulation sim(ntiles, nx, box, dt, partition );
                            
    Species electrons("electrons", -1.0f, ppc);
    electrons.set_udist(
        UDistribution::ThermalCorr( 
            make_float3( 0.1, 0.1, 0.1 ),
            make_float3(0, 0, 0.6 )
        )
    );

    sim.add_species( electrons );

    Species positrons("positrons", +1.0f, ppc);
    positrons.set_udist(
        UDistribution::ThermalCorr( 
            make_float3( 0.1, 0.1, 0.1 ),
            make_float3( 0, 0, -0.6 )
        )
    );

    sim.add_species( positrons );
                     
    // Run simulation    
    int const imax = 500;
                
    if ( mpi::world_root() )
        std::cout << "Running test up to n = " << imax << "...\n";

    Timer timer;
                  
    timer.start();
                                 
    while (sim.get_iter() < imax)
    {
        // std::cout << "n = " << sim.get_iter() << '\n';
        sim.advance();
    }
                 
    timer.stop();
    
    if ( mpi::world_root() )
        std::cout << "Simulation complete at i = " << sim.get_iter() << '\n';
                      
    sim.energy_info();

/*
    sim.emf.save(emf::b, fcomp::x);
    sim.emf.save(emf::b, fcomp::y);
    sim.emf.save(emf::b, fcomp::z);
*/                                                                  
    auto perf = sim.get_nmove() / timer.elapsed(timer::s) / 1.e9;

    if ( mpi::world_root() ) {
        std::cout << "Elapsed time: " << timer.elapsed(timer::s) << " s\n";
        std::cout << "Performance: " << perf << " GPart/s\n";
    }   

}

#if 0

void test_weibel( ) {
    uint2 gnx{ 128, 128 };
    uint2 ntiles{ 2, 2 };

    uint2 nx{ gnx.x/ntiles.x, gnx.y/ntiles.y };
    float2 box = make_float2( gnx.x/10.0, gnx.y/10.0 );
    float dt = 0.07;

    Simulation sim( ntiles, nx, box, dt );

    uint2 ppc{ 4, 4 };

    Species electrons("electrons", -1.0f, ppc);
    electrons.set_udist(
        UDistribution::ThermalCorr( 
            float3{ 0.1, 0.1, 0.1 },
            float3{ 0, 0, 0.6 }
        )
    );

    sim.add_species( electrons );

    Species positrons("positrons", +1.0f, ppc);
    positrons.set_udist(
        UDistribution::ThermalCorr( 
            float3{ 0.1, 0.1, 0.1 },
            float3{ 0, 0, -0.6 }
        )
    );

    sim.add_species( positrons );

    // Lambda function for diagnostic output
    auto diag = [ & ]( ) {
        sim.emf.save(emf::b, fcomp::x);
        sim.emf.save(emf::b, fcomp::y);
        sim.emf.save(emf::b, fcomp::z);

        electrons.save_charge();
        positrons.save_charge();

        sim.energy_info();
    };

    Timer timer; timer.start();

    while ( sim.get_t() <= 35.0 ) {
        //if ( sim.get_iter() % 10 == 0 )
        //    std::cout << "t = " << sim.get_t() << '\n';
        sim.advance();
    }

    timer.stop();

    diag();

    auto perf = sim.get_nmove() / timer.elapsed(timer::s) / 1.e9;

    std::cerr << "Elapsed time: " << timer.elapsed(timer::s) << " s"
              << ", Performance: " << perf << " GPart/s\n";
}


void test_weibel_large( )
{
                            
    // Create simulation box
    uint2 ntiles{64, 64};
    uint2 nx{32, 32};
    uint2 ppc{8, 8};
                                                                                                    
    uint64_t vol = static_cast<uint64_t>(nx.x * nx.y) *  static_cast<uint64_t>(ntiles.x * ntiles.y);
                                             
    std::cout << "** Large Weibel test **\n";                                   
    std::cout << " # tiles          : " << ntiles.x << ", " << ntiles.y << "\n";
    std::cout << " tile size        : " << nx.x << ", " << nx.y << "\n";                      
    std::cout << " global size      : " << nx.x * ntiles.x << ", " << nx.y * ntiles.y << "\n";
    std::cout << " # part / species : " << vol * (ppc.x * ppc.y) / (1048576.0) << " M \n";
                                                                  
    float2 box = {nx.x * ntiles.x * 0.1f, nx.y * ntiles.y * 0.1f};
                    
    float dt = 0.07;
                                        
    Simulation sim( ntiles, nx, box, dt );
                            
    Species electrons("electrons", -1.0f, ppc);
    electrons.set_udist(
        UDistribution::ThermalCorr( 
            float3{ 0.1, 0.1, 0.1 },
            float3{ 0, 0, 0.6 }
        )
    );

    sim.add_species( electrons );

    Species positrons("positrons", +1.0f, ppc);
    positrons.set_udist(
        UDistribution::ThermalCorr( 
            float3{ 0.1, 0.1, 0.1 },
            float3{ 0, 0, -0.6 }
        )
    );

    sim.add_species( positrons );
                     
    // Run simulation    
    int const imax = 500;
                                                          
    printf("Running large Weibel test up to n = %d...\n", imax);
                
    Timer timer;
                  
    timer.start();
                                 
    while (sim.get_iter() < imax)
    {                 
        sim.advance();
    }
                 
    timer.stop();
                                                              
    std::cout << "Simulation complete at i = " << sim.get_iter() << '\n';
                      
    sim.energy_info();
                                    
    sim.emf.save(emf::b, fcomp::x);
    sim.emf.save(emf::b, fcomp::y);
    sim.emf.save(emf::b, fcomp::z);
                                                                  
    auto perf = sim.get_nmove() / timer.elapsed(timer::s) / 1.e9;

    // std::cerr << "Elapsed time: " << timer.elapsed(timer::s) << " s"
    //          << ", Performance: " << perf << " GPart/s\n";

    std::cout << "[benchmark] " << perf << " GPart/s\n";
}

void test_warm( ) 
{

    Simulation sim(         
        uint2{32, 32},        // ntiles
        uint2{32, 32},        // nx
        float2{102.4, 102.4}, // box
        0.07                  // dt
    );               

    uint2  ppc{8, 8};       
    float3 uth{0.01, 0.01, 0.01};
    float3 ufl{0.0, 0.0, 0.0};
                                      
    UDistribution::Thermal udist(uth, ufl);
                            
    Species electrons("electrons", -1.0f, ppc);
    electrons.set_udist(udist);
                       
    sim.add_species(electrons);
                   
    electrons.save();   
                   
    int const imax = 500;
                      
    std::cout <<  "Running warm test up to iteration = " << imax <<"...\n";
                      
    Timer timer;      
    timer.start();

    while (sim.get_iter() < imax)
    {
        sim.advance();
    }

    timer.stop();

    sim.current.save(fcomp::x);
    sim.current.save(fcomp::y);
    sim.current.save(fcomp::z);
    electrons.save();

    std::cout << "Simulation complete at i = " << sim.get_iter() << '\n';

    auto perf = sim.get_nmove() / timer.elapsed(timer::s) / 1.e9;
    std::cerr << "Elapsed time: " << timer.elapsed(timer::s) << " s"
              << ", Performance: " << perf << " GPart/s\n";
}

void benchmark_weibel()
{

    auto bench_weibel_sim = [ ]( std::ostream &output, uint2 ppc, uint2 nx, uint2 ntiles ) {
        Simulation sim(
            ntiles,                                                      // ntiles
            nx,                                                          // nx
            make_float2(ntiles.x * nx.x * 0.1f, ntiles.y * nx.y * 0.1f), // box
            0.07                                                         // dt
        );

        float3 ufl = float3{0., 0., 0.6};
        float3 uth = float3{0.1, 0.1, 0.1};

        Species electrons("electrons", -1.0f, ppc);
        electrons.set_udist( UDistribution::Thermal (uth, ufl) );

        Species positrons("positrons", +1.0f, ppc);
        ufl.z = - ufl.z;
        positrons.set_udist( UDistribution::Thermal (uth, ufl) );

        sim.add_species(electrons);
        sim.add_species(positrons);

        float const imax = 500;

        Timer timer;

        timer.start();

        while (sim.get_iter() < imax)
        {
            sim.advance();
        }

        timer.stop();

        output         << ppc.x << ", " << ppc.y 
               << ", " << nx.x<< ", " << nx.y
               << ", " << ntiles.x << ", " << ntiles.y
               << ", " << timer.elapsed(timer::s)
               << ", " << sim.get_nmove() / timer.elapsed(timer::s) / 1.e9
               << ", " << std::endl;
    };

    std::vector<uint2> ppc_list{
        {1, 1}, {2, 1}, {2, 2}, {4, 2}, {4, 4}, {8, 4}, {8, 8}, {16, 8}, {16, 16}, {32, 16}, {32, 32}};

    std::vector<uint2> ntiles_list{
        {8, 8}, {16, 8}, {16, 16}, {32, 16}, {32, 32}, {64, 32}, {64, 64}, {128, 64}, {128, 128}, {256, 128}, {256, 256}, {512, 256}, {512, 512}, {1024, 512}, {1024, 1024}};

    std::vector<uint2> nx_list{
        {4, 4}, {8, 4}, {8, 8}, {16, 8}, {16, 16}, {32, 16}, {32, 32}, {64, 32}, {64, 64}, { 80, 64 }, { 80, 80 } };

/* 
    std::vector<uint2> ppc_list{ {4, 4} };
    std::vector<uint2> ntiles_list{ { 8, 8 } };
    std::vector<uint2> nx_list{ {80, 80} };
 */

    std::ofstream output;
    output.open("benchmark_weibel.csv");
    output << "ppc.x, ppc.y, nx.x, nx.y, ntiles.x, ntiles.y, time, perf" << std::endl;

    int ntests = ppc_list.size() *
                nx_list.size() *
                ntiles_list.size();

    int i = 0;
    for (uint2 ppc : ppc_list)
        for (uint2 nx : nx_list)
            for (uint2 ntiles : ntiles_list)
            {
                // The casts avoid integer overflows (because 64^2 * 1024^2 = 2^32)
                int64_t vol = static_cast<int64_t>(nx.x * nx.y) * 
                              static_cast<int64_t>(ntiles.x * ntiles.y);
                int64_t npart = vol * static_cast<int64_t>(ppc.x * ppc.y);

                printf("[%3d/%3d] %d, %d, %d, %d, %d, %d (%ld)\n", i++, ntests,
                       ppc.x, ppc.y, nx.x, nx.y, ntiles.x, ntiles.y, npart);

                // Limit the test to 512 M part / species
                if ( npart < 512 * 1024 * 1024 )
                {
                    deviceReset();
                    bench_weibel_sim(output, ppc, nx, ntiles);
                }
            }

    output.close();
}

#endif

/**
 * @brief Print information about the environment
 * 
 * @note Only MPI root node prints this
 */
void info( void ) {

    if ( mpi::world_root() ) {

        std::cout << ansi::bold;
        std::cout << "Environment\n";
        std::cout << ansi::reset;

        char name[MPI_MAX_PROCESSOR_NAME];
        int len; MPI_Get_processor_name(name, &len);

        std::cout << "MPI running on " << mpi::world_size() << " processes\n";

        std::cout << "GPU devices on rank 0 (" << name << "):\n";
        print_gpu_info();

    }
}

/**
 * @brief Print command line help
 * 
 * @param argv0     Program name (from argv[0])
 */
void cli_help( char * argv0 ) {
    std::cerr << "Usage: " << argv0 << " [-h] [-s] [-t name] [-n parameter]\n";

    std::cerr << '\n';
    std::cerr << "Options:\n";
    std::cerr << "  -h                  Display this message and exit\n";
    std::cerr << "  -s                  Silence information about MPI/CUDA device\n";
    std::cerr << "  -t <name>           Name of the test to run. Defaults to 'weibel'\n";
    std::cerr << "  -p <parameters>     Test parameters (string). Purpose will depend on the \n";
    std::cerr << "                      test chosen. Defaults to '2,2,16,16'\n";
    std::cerr << '\n';
}


/**
 * @brief Initialize GPU device
 * 
 * @note Selects and initializes a GPU device on the parallel node
 * 
 * @param comm      MPI communicator
 */
void gpu_init( MPI_Comm comm ) {

    MPI_Comm local_comm;
    int global_rank, local_rank;

    // Create a communicator with processes sharing the same hardware node
    MPI_Comm_rank( comm, &global_rank );
    MPI_Comm_split_type( comm, MPI_COMM_TYPE_SHARED, global_rank,  MPI_INFO_NULL, &local_comm );
    
    // Get rank in local communicator
    MPI_Comm_rank(local_comm, &local_rank);

    // Free the communicator
    MPI_Comm_free(&local_comm);

    // Get number of GPU devices on node
    int num_devices; cudaGetDeviceCount(&num_devices);

    // Use local_rank to select GPU device on node usina a round-robin algorithm
    int device = local_rank % num_devices;
    cudaSetDevice( device );

    // mpi::cout << "GPU device: " << device << '\n';

    // Reset current device
    // deviceReset();
}

#endif

int main( int argc, char *argv[] ) {

    // Initialize the MPI environment
    mpi::init( & argc, & argv );

    // Initialize the gpu device
    gpu::parallel_init( MPI_COMM_WORLD );

#if 0
    // Process command line arguments
    int opt;
    int silent = 0;
    std::string test = "weibel";
    std::string param = "2,2,16,16";
    while ((opt = getopt(argc, argv, "ht:p:s")) != -1) {
        switch (opt) {
            case 't':
            test = optarg;
            break;
        case 'p':
            param = optarg;
            break;
        case 's':
            silent = 1;
            break;
        case 'h':
        case '?':
            cli_help( argv[0] );
            mpi::finalize();
            exit(0);
        default:
            if ( mpi::world_root() ) cli_help( argv[0] );    
            mpi::abort(1);
        }
    }
    
    // Print information about the environment
    if ( ! silent ) info();
#endif

    // test_tiled_grid();
    // test_tiled_vec3_grid();

    // test_laser( );
    // test_inj( );
    // test_mov( );
    // test_current( );
    test_weibel();

    // test_mov_window_emf();
    // test_mov_window_inj();
    // test_lwfa();


    // test_warm( );
    /*
    if ( test == "weibel" ) {
        test_weibel( param );
    } else {
        if ( mpi::world_root() ) 
            std::cerr << "Unknonw test '" << test << "', aborting...\n";
        mpi::abort(1);
    }
    */
    // benchmark_weibel();

    // Finalize the MPI environment
    mpi::finalize();
}
