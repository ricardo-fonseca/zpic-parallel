
// For getopt
#include <unistd.h>

#include <iostream>

#include "parallel.h"

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


#if 0
void test_laser( void ) {

    // Parallel partition
    uint2 partition = make_uint2( 2, 2 );

    // Global number of tiles
    uint2 ntiles = make_uint2( 16, 8 );
    uint2 nx = make_uint2( 64, 32 );

    float2 box = make_float2( 20.48, 25.6 );
    double dt = 0.014;

    Partition parallel( partition );

    EMF emf( ntiles, nx, box, dt, parallel );

    Laser::PlaneWave laser;
    laser.start = 10.2;
    laser.fwhm = 4.0;
    laser.a0 = 1.0;
    laser.omega0 = 10.0;

/*
    Laser::Gaussian laser;
    laser.start = 10.2;
    laser.fwhm = 4.0;
    laser.a0 = 1.0;
    laser.omega0 = 10.0;
    laser.W0 = 4.0;
    laser.focus = 20.48;
    laser.axis = 12.8;
*/

    laser.sin_pol = 1;
    laser.cos_pol = 0;

    laser.add( emf );

    auto save_emf = [& emf ]( ) {
        emf.save( emf::e, fcomp::x );
        emf.save( emf::e, fcomp::y );
        emf.save( emf::e, fcomp::z );

        emf.save( emf::b, fcomp::x );
        emf.save( emf::b, fcomp::y );
        emf.save( emf::b, fcomp::z );
    };

    save_emf();

    if ( parallel.root() ) 
        std::cout << "Starting test...\n";

    auto t0 = Timer("test");

    if ( parallel.root() ) 
        std::cout << "Clock resolution is " << t0.resolution() << " ns\n";

    t0.start();

    for( int i = 0; i < 1500; i ++) {
        if ( i == 500 || i == 1000 ) save_emf();
        emf.advance();
    }

    t0.stop();
    if ( mpi::world_rank() == 0 )  t0.report("1500 iterations:");

    save_emf( );
}

void test_partition() {
    uint2 dims = make_uint2( 2, 3 );
    Partition part( dims );
    part.info();
}

void test_grid() {

    uint2 dims = make_uint2( 3, 3 );
    Partition parallel( dims );

    uint2 global_ntiles {6, 6};
    uint2 nx {8, 8};

    bnd<unsigned int> gc;
    gc.x.lower = 1; gc.x.upper = 2;
    gc.y.lower = 1; gc.y.upper = 2;

    grid<float> data( global_ntiles, nx, gc, parallel );

//    data.set( float( parallel.get_rank() + 1 ) );
    data.set( 1 );

    data.add_from_gc();
    data.copy_to_gc();

    for( int i = 0; i < 5; i++ ) {
        data.x_shift_left_mk2( 1 );
    }

    data.save( "test/test.zdf" );

    parallel.barrier();
    if ( mpi::world_root() ) std::cout << "Done!\n";

}

void test_particles() {
    uint2 par_dims = make_uint2( 2, 2 );
    uint2 global_ntiles {16, 16};

//    uint2 dims = make_uint2( 2, 1 );
//    uint2 global_ntiles {2, 2};

    Partition parallel( par_dims );

    uint2 nx {8, 8};
    float2 box = make_float2( 12.8, 12.8 );

    uint2 gnx = nx * global_ntiles;
    float2 dx = { box.x / gnx.x, box.y / gnx.y };
    float dt = 1. / sqrt( 1./(dx.x*dx.x) + 1./(dx.y*dx.y) ); // max time step
    if ( mpi::world_root() ) {
        std::cout << ansi::bold;
        std::cout << "gnx = " << gnx << '\n';
        std::cout << "dx  = " << dx << '\n';
        std::cout << "dt  = " << dt;
        std::cout << ansi::reset << '\n';
    }

//    uint2 ppc = make_uint2( 5, 5 );
    uint2 ppc = make_uint2( 8, 8 );
//    uint2 ppc = make_uint2( 1, 1 );

    Species electrons("electrons", -1.0f, ppc );

//    electrons.set_density(Density::Step(coord::x, 1.0, 5.0));
//    electrons.set_density(Density::Slab(coord::x, 1.0, 5.0, 7.0));
    electrons.set_density(Density::Sphere( 1.0, make_float2( 6.4, 6.4 ), 2.8));

//    electrons.set_udist( UDistribution::Cold( make_float3( 1e6, 0., 0. ) ) );
//    electrons.set_udist( UDistribution::Cold( make_float3( 0, -1e6, 0. ) ) );
    electrons.set_udist( UDistribution::Cold( make_float3( -1e6, -1e6, 0. ) ) );
//    electrons.set_udist( UDistribution::Thermal( make_float3( 0.1, 0.2, 0.3 ) ) );

    electrons.initialize( box, global_ntiles, nx, dt, 0, parallel );

    electrons.save();

    electrons.save_charge();

#if 1
    for( int i = 0; i < 100; i ++ ) {
        electrons.advance();
        if ( ( electrons.get_iter() % 10 ) == 0 ) {
            if ( mpi::world_root() ) std::cout << "Now at iter = " << electrons.get_iter() << '\n';
            electrons.save();
            electrons.save_charge();
        }
    }
#else
    for( int i = 0; i < 1; i ++ ) {
        electrons.advance();
        electrons.save();
        electrons.save_charge();
    }
#endif

    if ( parallel.root() ) std::cout << ansi::bold << ansi::red << "Done!" << ansi::reset << "\n";
}

void test_current() {
    uint2 par_dims = make_uint2( 2, 2 );
    uint2 global_ntiles {4, 4};

//    uint2 dims = make_uint2( 2, 1 );
//    uint2 global_ntiles {2, 2};

    Partition parallel( par_dims );

    uint2 nx {8, 8};
    float2 box = make_float2( 12.8, 12.8 );

    uint2 gnx = nx * global_ntiles;
    float2 dx = { box.x / gnx.x, box.y / gnx.y };
    float dt = 1. / sqrt( 1./(dx.x*dx.x) + 1./(dx.y*dx.y) ) *.9; // max time step

    uint2 ppc = make_uint2( 8, 8 );

    Species electrons("electrons", -1.0f, ppc );

    electrons.set_density(Density::Sphere( 1.0, make_float2( 6.4, 6.4 ), 2.8));
    electrons.set_udist( UDistribution::Cold( make_float3( -1e6, +1e6, 0.1 ) ) );

    electrons.initialize( box, global_ntiles, nx, dt, 0, parallel );

    electrons.save_charge();

    Current current( global_ntiles, nx, box, dt, parallel );

    for( int i = 0; i < 10; i ++ ) {
        current.zero();
        electrons.advance( current );

        current.advance( );

        current.save( fcomp::x );
        current.save( fcomp::y );
        current.save( fcomp::z );

        electrons.save_charge();
    }

    if ( parallel.root() ) std::cout << ansi::bold << ansi::red << "Done!" << ansi::reset << "\n";
}

#endif

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

    int opt;
    int n = 1;
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

    if ( test == "weibel" ) {
        test_weibel( param );
    } else {
        if ( mpi::world_root() ) 
            std::cerr << "Unknonw test '" << test << "', aborting...\n";
        mpi::abort(1);
    }

//    info();
//    test_weibel_96();


    // Finalize the MPI environment
    mpi::finalize();

}
