
// For getopt
#include <unistd.h>

#include <iostream>


#include "utils.h"
#include "vec_types.h"
#include "grid.h"

#include "vec3grid.h"
#include "emf.h"
#include "laser.h"

#include "timer.h"
#include "simulation.h"
#include "cathode.h"

/**
 * OpemMP support
 */
#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * SIMD support
 */
#include "simd/simd.h"

/**
 * MPI support 
 */
#include "parallel.h"


void test_grid( void ) {
    
    if ( mpi::world_root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    // Parallel partition
    uint2 partition = make_uint2( 2, 2 );

    // Global number of tiles
    const uint2 global_ntiles = { 4, 4 };
    const uint2 nx     = { 12, 12 };

    bnd<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {1,2};

    Partition parallel( partition );

    grid<float> data( global_ntiles, nx, gc, parallel );

    // Get local number of tiles
    const auto ntiles   = data.get_ntiles();

    uint2 const global_off = data.get_tile_off();

    if ( mpi::world_root() ) {
        std::cout << "Setting values...\n";
    }

    data.zero( );
    data.set( 32.0 );

    auto ext_nx = data.ext_nx;

    for( unsigned int tid = 0; tid < ntiles.y * ntiles.x; tid++ ) {
        const uint2 tile_idx = { tid % ntiles.x, tid / ntiles.x  };
        const size_t tile_off = tid * roundup4( ext_nx.x * ext_nx.y );
        auto * const __restrict__ tile_data = & data.d_buffer[ tile_off + data.offset ];

        const auto   tile_val = ( global_off.y + tile_idx.y ) * global_ntiles.x + ( global_off.x + tile_idx.x );

        for( unsigned int idx = 0; idx < nx.y * nx.x; idx ++ ) {
            const auto ix = idx % nx.x;
            const auto iy = idx / nx.x; 
            tile_data[iy * data.ext_nx.x + ix] = tile_val;
        }
    }
    
    data.add_from_gc();
    data.copy_to_gc();

    for( auto i = 0; i < 5; i++)
       data.x_shift_left( 1 );

    data.kernel3_x( 1., 2., 1. );
    data.kernel3_y( 1., 2., 1. );

    parallel.barrier();
    if ( mpi::world_root() )
        std::cout << "Saving data...\n";

    data.save( "mpi/mpi.zdf" );

    if ( mpi::world_root() ) {
        std::cout << ansi::bold;
        std::cout << "Done!\n";
        std::cout << ansi::reset;
    }      
}

void test_vec3grid( ) {

    if ( mpi::world_root() ) {
        std::cout << ansi::bold;
        std::cout << "Running test_grid()\n";
        std::cout << ansi::reset;
        std::cout << "Declaring test_vec3grid<float> data...\n";
    }

    // Parallel partition
    uint2 partition = make_uint2( 2, 2 );

    const uint2 global_ntiles = { 8, 8 };
    const uint2 nx = { 16,16 };
    
    bnd<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {1,2};

    Partition parallel( partition );

    vec3grid<float3> data( global_ntiles, nx, gc, parallel );

    // Get local number of tiles
    const auto ntiles = data.get_ntiles();

    uint2 const global_off = data.get_tile_off();

    // Set zero
    // data.zero( );

    // Set constant
    // data.set( float3{1.0, 2.0, 3.0} );

    // Set different value per tile
    for( unsigned int tid = 0; tid < ntiles.y * ntiles.x; tid++ ) {
        const uint2 tile_idx = { tid % ntiles.x, tid / ntiles.x  };
        const size_t tile_off = tid * roundup4( data.ext_nx.x * data.ext_nx.y );
        auto * const __restrict__ tile_data = & data.d_buffer[ data.offset + tile_off ];

        const auto   tile_val = ( global_off.y + tile_idx.y ) * global_ntiles.x + ( global_off.x + tile_idx.x );

        for( unsigned int idx = 0; idx < nx.y * nx.x; idx += 1 ) {
            const auto iy = idx / nx.x; 
            const auto ix = idx % nx.x;
            tile_data[iy * data.ext_nx.x + ix] = make_float3( 1 + tile_val, 2 + tile_val, 3 + tile_val );
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

    if ( mpi::world_root() )
        std::cout << "Saving data...\n";

    data.save( fcomp::x, "mpi/mpi-vec3-x.zdf" );
    data.save( fcomp::y, "mpi/mpi-vec3-y.zdf" );
    data.save( fcomp::z, "mpi/mpi-vec3-z.zdf" );

    if ( mpi::world_root() ) {
        std::cout << ansi::bold;
        std::cout << "Done!\n";
        std::cout << ansi::reset;
    }
}


void test_laser( ) {

    if ( mpi::world_root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    // Parallel partition
    uint2 partition = make_uint2( 4, 2 );

    uint2 ntiles = { 64, 16 };
    uint2 nx = { 16, 16 };

    float2 box = { 20.48, 25.6 };
    double dt = 0.014;

    Partition parallel( partition );

    EMF emf( ntiles, nx, box, dt, parallel );

    auto save_emf = [ & emf ]( ) {
        emf.save( emf::e, fcomp::x );
        emf.save( emf::e, fcomp::y );
        emf.save( emf::e, fcomp::z );

        emf.save( emf::b, fcomp::x );
        emf.save( emf::b, fcomp::y );
        emf.save( emf::b, fcomp::z );
    };

/*
    Laser::PlaneWave laser;
    laser.start = 10.2;
    laser.fwhm = 4.0;
    laser.a0 = 1.0;
    laser.omega0 = 10.0;
*/

    Laser::Gaussian laser;
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

    if ( mpi::world_root() ) {
        std::cout << "Starting test - " << niter << " iterations...\n";
    }

    auto t0 = Timer("test");

    t0.start();

    for( int i = 0; i < niter; i ++) {
        emf.advance( );
    }

    t0.stop();

    save_emf();

    if ( mpi::world_root() ) {
        char buffer[128];
        snprintf( buffer, 127, "%d iterations: ", niter );
        t0.report( buffer );

        std::cout << ansi::bold;
        std::cout << "Done!\n";
        std::cout << ansi::reset;
    }
}

void test_inj( ) {

    if ( mpi::world_root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    // Parallel partition
    uint2 partition = make_uint2( 2, 2 );

    Partition parallel( partition );

    uint2 ntiles{ 4, 4 };
    uint2 nx{ 32, 32 };

    float2 box{ 12.8, 12.8 };

    auto dt = 0.99 * zpic::courant( ntiles, nx, box );

    uint2 ppc{ 8, 8 };
    Species electrons( "electrons", -1.0f, ppc );

    parallel.barrier();
    if ( mpi::world_root() ) std::cout << "Created species\n";

    //electrons.set_density(Density::Step(coord::x, 1.0, 5.0));
    // electrons.set_density(Density::Slab(coord::y, 1.0, 5.0, 8.0));
    electrons.set_density( Density::Sphere( 1.0, float2{5.0, 7.0}, 2.0 ) );

    parallel.barrier();
    if ( mpi::world_root() ) std::cout << "Density set\n";

    electrons.set_udist( UDistribution::Thermal( float3{ 0.1, 0.2, 0.3 }, float3{1,0,0} ) );

    electrons.initialize( box, ntiles, nx, dt, 0, parallel );

    electrons.save_charge();
    electrons.save();
    electrons.save_phasespace(
        phasespace::ux, float2{-1, 3}, 256,
        phasespace::uz, float2{-1, 1}, 128
    );

    parallel.barrier();
    if ( mpi::world_root() ) {
        std::cout << ansi::bold;
        std::cout << __func__ << "() complete!\n";
        std::cout << ansi::reset;
    }
}

void test_mov( ) {

    if ( mpi::world_root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    // Parallel partition
    uint2 partition = make_uint2( 2, 2 );

    Partition parallel( partition );

    uint2 ntiles{ 4, 4 };
    uint2 nx{ 32, 32 };

    float2 box{ 12.8, 12.8 };

    auto dt = 0.99 * zpic::courant( ntiles, nx, box );

    uint2 ppc{ 8, 8 };
    Species electrons( "electrons", -1.0f, ppc );

    electrons.set_density( Density::Sphere( 1.0, float2{2.1, 2.1}, 2.0 ) );
    electrons.set_udist( UDistribution::Cold( float3{ -1, -2, -3 } ) );
    electrons.initialize( box, ntiles, nx, dt, 0, parallel );

    electrons.save_charge();
    electrons.save();

    int niter = 200; //200
    for( auto i = 0; i < niter; i ++ ) {
        auto np_global = electrons.np_global();
        if ( parallel.root() ) std::cout << "i = " << i << ", total particles: " << np_global << '\n';
        electrons.advance();
    }

    electrons.save_charge();
    electrons.save();

    parallel.barrier();
    if ( mpi::world_root() ) {
        std::cout << ansi::bold;
        std::cout << __func__ << "() complete!\n";
        std::cout << ansi::reset;
    }
}

void test_current( ) {

    if ( mpi::world_root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    // Parallel partition
    uint2 partition = make_uint2( 2, 2 );

    Partition parallel( partition );

    uint2 ntiles{ 4, 4 };
    uint2 nx{ 32, 32 };

    float2 box{ 12.8, 12.8 };

    auto dt = 0.99 * zpic::courant( ntiles, nx, box );

    uint2 ppc{ 8, 8 };
    Species electrons( "electrons", -1.0f, ppc );

    electrons.set_density( Density::Sphere( 1.0, float2{6.4, 6.4}, 5.0 ) );
    electrons.set_udist( UDistribution::Cold( float3{ 1, 2, 3 } ) );

    electrons.initialize( box, ntiles, nx, dt, 0, parallel );

    Current current( ntiles, nx, box, dt, parallel );

    electrons.save_charge();

    electrons.advance( current );
    current.advance( );
    
    current.save( fcomp::x );
    current.save( fcomp::y );
    current.save( fcomp::z );

    parallel.barrier();
    if ( mpi::world_root() ) {
        std::cout << ansi::bold;
        std::cout << __func__ << "() complete!\n";
        std::cout << ansi::reset;
    }
}

void test_weibel_96( )
{
    // MPI partition
    uint2 partition { 2, 2 };
                            
    // Create simulation box
    uint2 ntiles {16, 16};
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
        std::cout << "Running Weibel(96) test up to n = " << imax << "...\n";

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
                                    
    sim.emf.save(emf::b, fcomp::x);
    sim.emf.save(emf::b, fcomp::y);
    sim.emf.save(emf::b, fcomp::z);
                                                                  
    auto perf = sim.get_nmove() / timer.elapsed(timer::s) / 1.e9;

    if ( mpi::world_root() ) {
        std::cerr << "Elapsed time: " << timer.elapsed(timer::s) << " s"
                  << ", Performance: " << perf << " GPart/s\n";
    }
}

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
        std::cout << "Running Weibel test\n";
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
        std::cerr << "Elapsed time: " << timer.elapsed(timer::s) << " s"
                  << ", Performance: " << perf << " GPart/s\n";
    }   

}


void test_weibel_debug( )
{
   
    if ( mpi::world_root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    // Parallel partition
    uint2 partition = make_uint2( 2, 2 );

    // Create simulation box
    uint2 ntiles{8, 8};
    uint2 nx{32, 32};                                                                                                                                                                     
    float2 box = {nx.x * ntiles.x * 0.1f, nx.y * ntiles.y * 0.1f};
    float dt = 0.07;
                                        
    Simulation sim( ntiles, nx, box, dt, partition );
                            
    uint2 ppc{4, 4};

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

    // Run simulation    
    int const imax = 500;
        
    if ( sim.parallel.root() )
        std::cout << "Running large Weibel test up to n = " << imax << "...\n";
                
    Timer timer;
                  
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
        std::cout << "Simulation complete at i = " << sim.get_iter() << '\n';
        auto time = timer.elapsed(timer::s);
        std::cout << "Elapsed time: " << time << " s\n";
        auto perf = nmove / time / 1.e9;
        std::cout << "Performance : " << perf << " GPart/s\n";
    }                  
}


/**
 * @brief Print information about the environment
 * 
 */
void info( void ) {

    if ( mpi::world_root() ) {

        std::cout << "MPI running on " << mpi::world_size() << " processes\n";

        #ifdef SIMD
            std::cout << "SIMD support enabled\n";
            std::cout << "  vector unit : " << vecname << '\n';
            std::cout << "  vector width: " << vecwidth <<'\n';
        #else
            std::cout << "SIMD support not enabled\n";
        #endif
        
        #ifdef _OPENMP
            std::cout << "OpenMP enabled\n";
            std::cout << "  # procs           : " << omp_get_num_procs() << '\n';
            std::cout << "  max_threads       : " << omp_get_max_threads() << '\n';
            #pragma omp parallel
            {
                if ( omp_get_thread_num() == 0 )
                    std::cout << "  default # threads : " << omp_get_num_threads() << '\n';
            }
        #else
            std::cout << "OpenMP support not enabled\n";
        #endif

    }
}

void cli_help( char * argv0 ) {
    std::cerr << "Usage: " << argv0 << " [-h] [-s] [-t name] [-n parameter]\n";

    std::cerr << '\n';
    std::cerr << "Options:\n";
    std::cerr << "  -h                  Display this message and exit\n";
    std::cerr << "  -s                  Silence information about MPI/OpenMP/SIMD parameters\n";
    std::cerr << "  -t <name>           Name of the test to run. Defaults to 'weibel'\n";
    std::cerr << "  -p <parameters>     Test parameters (string). Purpose will depend on the \n";
    std::cerr << "                      test chosen. Defaults to '2,2,16,16'\n";
    std::cerr << '\n';
}

int main( int argc, char *argv[] ) {

    // Initialize the MPI environment
    mpi::init( & argc, & argv );

    // Initialize SIMD support
    simd_init();

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

    // test_grid( );  
    // test_vec3grid( );
    // test_laser( );
    // test_inj( );
    // test_mov( );
    // test_current( );
    // test_weibel_debug();

    test_weibel_96();

    /*
    if ( test == "weibel" ) {
        test_weibel( param );
    } else {
        if ( mpi::world_root() ) 
            std::cerr << "Unknonw test '" << test << "', aborting...\n";
        mpi::abort(1);
    }
    */


    // Finalize the MPI environment
    mpi::finalize();

}
