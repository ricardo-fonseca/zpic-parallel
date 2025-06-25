// For getopt
#include <unistd.h>

#include <iostream>
#include <fstream>
#include <sstream>

#include "utils.h"
#include "grid.h"
#include "basic_grid.h"

#include "fft.h"
#include "simulation.h"

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


void set_charge( 
    float * const __restrict__ d_buffer,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx, 
    float2 dx, float2 x0, float r ) 
{

    #pragma omp parallel for
    for( unsigned tile_id = 0; tile_id < ntiles.y * ntiles.x; tile_id++ ) {

        const uint2  tile_idx { tile_id % ntiles.x, tile_id / ntiles.x };
        const size_t tile_off = tile_id * roundup4( ext_nx.x * ext_nx.y );
        auto * const __restrict__ tile_data = & d_buffer[ tile_off ];

        for( unsigned iy = 0; iy < nx.y; iy++ ) {
            for( unsigned ix = 0; ix < nx.x; ix++ ) {
                float x = ( tile_idx.x * nx.x + ix ) * dx.x;
                float y = ( tile_idx.y * nx.y + iy ) * dx.y;

                tile_data[ iy * ext_nx.x + ix ] = (x-x0.x)*(x-x0.x) + (y-x0.y) * (y-x0.y) <= r*r; 
            }
        }

    }
}

void poisson(
        std::complex<float> * const __restrict__ data, uint2 const dims, float2 const dk
    )
{
    constexpr std::complex<float> I{0,1};

    std::cout << "K-space dims: " << dims << '\n';

//    #pragma omp parallel for
    for( unsigned idx = 0; idx < dims.y * dims.x; idx ++ ) {
        const int ix = idx % dims.x;
        const int iy = idx / dims.x;

        const float ky = (( 2 * iy < int(dims.y) ) ? iy : ( iy - int(dims.y) ) ) * dk.y;
        //const float ky = iy * dk.y;
        
        const float kx = ix * dk.x;
        const float k2 = kx*kx + ky*ky;

        data[ idx ] *= ((k2 > 0)? 1.f / k2 : 0.);
    }
}

void test_grid( void ) {
    
    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;

    const float2 box{1.0, 1.0};
    const uint2 ntiles{ 16, 16 };
    const uint2 nx    { 16, 16 };

    uint2 in_dims{ ntiles.x * nx.x, ntiles.y * nx.y };
    float2 dx{ box.x / in_dims.x, box.y / in_dims.y };

    bnd<unsigned int> gc;
    gc.x = {0,1};
    gc.y = {0,1};

    std::cout << "Allocating arrays..." << '\n';

    grid<float> charge( ntiles, nx, gc );
    grid<float> potential( ntiles, nx, gc );

    uint2 out_dims = fft::fdims( ntiles * nx );
    basic_grid<std::complex<float>> fpotential( out_dims );

    set_charge (
        & charge.d_buffer[ charge.offset ], ntiles, nx, charge.ext_nx,
        dx, float2{ 0.25, 0.25 }, 0.1
    );

    std::cout << "Saving charge to disk..." << '\n';
    charge.save("charge.zdf");

    fft::plan plan_r2c( charge, fpotential );
    fft::plan plan_c2r( fpotential, charge );

    std::cout << "plan_r2c: " << plan_r2c << '\n';
    std::cout << "plan_c2r: " << plan_c2r << '\n';

    std::cout << "Transforming charge -> fpotential...\n";
    plan_r2c.transform( charge, fpotential );

    fpotential.name = "F(charge)";
    fpotential.save( "charge_k.zdf" );

    poisson (
        fpotential.d_buffer,
        out_dims, fft::dk( box )
    );

    fpotential.name = "F(potential)";
    fpotential.save( "potential_k.zdf" );

    std::cout << "Transforming fpotential -> charge...\n";
    plan_c2r.transform( fpotential, charge );

    std::cout << "Saving potential to disk..." << '\n';
    charge.save("potential.zdf");

    std::cout << ansi::bold;
    std::cout << "Done!\n";
    std::cout << ansi::reset;       
}

#include "emf.h"
#include "laser.h"

#include "timer.h"

void test_laser( ) {

    std::cout << ansi::bold
              << "Running " << __func__ << "()..."
              << ansi::reset << std::endl;

    uint2 ntiles{ 64, 16 };
    uint2 nx{ 16, 16 };

    float2 box{ 20.48, 25.6 };
    double dt{ 0.014 };

    std::cout << "Creating EMF object...\n";

    EMF emf( ntiles, nx, box, dt );

    auto save_emf = [ & emf ]( ) {
        emf.save( emf::e, fcomp::x );
        emf.save( emf::e, fcomp::y );
        emf.save( emf::e, fcomp::z );

        emf.save( emf::b, fcomp::x );
        emf.save( emf::b, fcomp::y );
        emf.save( emf::b, fcomp::z );

        emf.save( emf::fet, fcomp::x );
        emf.save( emf::fet, fcomp::y );
        emf.save( emf::fet, fcomp::z );

        emf.save( emf::fb, fcomp::x );
        emf.save( emf::fb, fcomp::y );
        emf.save( emf::fb, fcomp::z );

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

    std::cout << "Adding laser...\n";
    laser.add( emf );

    std::cout << "Saving initial fields...\n";
    save_emf();

    int niter = 20.48 / dt / 2;
    // int niter{ 10 };

    std::cout << "Starting test - " << niter << " iterations...\n";

    Timer t0("test");

    t0.start();

    for( int i = 0; i < niter; i ++) {
        emf.advance( );
    }

    t0.stop();
    
    save_emf();

    std::ostringstream buffer;
    buffer << niter << " iterations: ";

    t0.report( buffer.str() );

    std::cout << ansi::bold
              << "Done!\n"
              << ansi::reset;   

}

void test_mov( ) {

    std::cout << ansi::bold
              << "Running " << __func__ << "()..."
              << ansi::reset << std::endl;

    uint2 ntiles{ 4, 4 };
    uint2 nx{ 32, 32 };

    float2 box{ 12.8, 12.8 };

    auto dt = 0.99 * zpic::courant( ntiles, nx, box );

    uint2 ppc{ 8, 8 };
    Species electrons( "electrons", -1.0f, ppc );

    electrons.set_density( Density::Sphere( 1.0, float2{2.1, 2.1}, 2.0 ) );
    electrons.set_udist( UDistribution::Cold( float3{ -1, -2, -3 } ) );

    electrons.initialize( box, ntiles, nx, dt, 0 );

    electrons.save();
    electrons.save_charge();
    electrons.save_phasespace(
        phasespace::x, float2{0, 12.8}, 128,
        phasespace::y, float2{0, 12.8}, 128
    );

    // int niter = 1;
    int niter = 200;

    for( auto i = 0; i < niter; i ++ ) {
        electrons.advance();
    }

    electrons.save_charge();
    electrons.save();

    std::cout << ansi::bold
              << "Done!\n"
              << ansi::reset;   
}

void test_weibel( ) {

    std::cout << ansi::bold
              << "Running " << __func__ << "()..."
              << ansi::reset << std::endl;

    uint2 ntiles{ 16, 16 };
    uint2 nx{ 32, 32 };
    
    float2 box = 0.1f * make_float2( ntiles.x * nx.x, ntiles.y * nx.y );
    float dt = 0.07;

    Simulation sim( ntiles, nx, box, dt );

    uint2 ppc{ 8, 8 };

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

        sim.current.save(fcomp::x);
        sim.current.save(fcomp::y);
        sim.current.save(fcomp::z);

        sim.charge.save();

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

    std::cout << ansi::bold
              << "Done!\n"
              << ansi::reset;

    auto perf = sim.get_nmove() / timer.elapsed(timer::s) / 1.e9;

    std::cerr << "Elapsed time: " << timer.elapsed(timer::s) << " s"
              << ", Performance: " << perf << " GPart/s\n";
}

/**
 * @brief Print information about the environment
 * 
 */
void info( void ) {

    std::cout << ansi::bold;
    std::cout << "Environment\n";
    std::cout << ansi::reset;

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

void cli_help( char * argv0 ) {
    std::cerr << "Usage: " << argv0 << " [-h] [-s] [-t name] [-n parameter]\n";

    std::cerr << '\n';
    std::cerr << "Options:\n";
    std::cerr << "  -h                  Display this message and exit\n";
    std::cerr << "  -s                  Silence information about host/CUDA device\n";
    std::cerr << "  -t <name>           Name of the test to run. Defaults to 'weibel'\n";
    std::cerr << "  -p <parameters>     Test parameters (string). Purpose will depend on the \n";
    std::cerr << "                      test chosen. Defaults to '2,2,16,16'\n";
    std::cerr << '\n';
}

int main( int argc, char *argv[] ) {

    // Initialize SIMD support
    simd_init();

    // Initialize FFT library
    fft::init();

    // Process command line arguments
    int opt;
    int silent = 0;
    std::string test = "weibel";
    std::string param = "16,16";
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
            std::exit(0);
        default:
            cli_help( argv[0] );    
            std::exit(1);
        }
    }
    
    // Print information about the environment
    if ( ! silent ) info();    

    // test_grid();
    // test_laser();
    // test_mov();
    test_weibel();

    fft::cleanup();
}