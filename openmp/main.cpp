
/**
 * OMP_NUM_THREADS=16 GOMP_CPU_AFFINITY="0-15" ./zpic
 */

#include <iostream>

#include <stdint.h>

#include "utils.h"
#include "vec_types.h"
#include "grid.h"

#include "vec3grid.h"
#include "emf.h"
#include "laser.h"

#include "timer.h"
#include "simulation.h"
#include "cathode.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "simd/simd.h"

#include <sys/prctl.h>

void info( void ) {

#ifdef SIMD
    std::cout << "SIMD support enabled\n";
    std::cout << "  vector unit : " << vecname << '\n';
    std::cout << "  vector width: " << vecwidth <<'\n';
#else
    std::cout << "SIMD support not enabled\n";
#endif

#ifdef __ARM_FEATURE_SVE_BITS
#if __ARM_FEATURE_SVE_BITS > 0
    std::cout << "ARM SVE bits: " << __ARM_FEATURE_SVE_BITS << '\n';
    prctl(PR_SVE_SET_VL, __ARM_FEATURE_SVE_BITS / 8);
#endif
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

void test_laser( void ) {

    uint2 ntiles = make_uint2( 16, 8 );
    uint2 nx = make_uint2( 64, 32 );

    float2 box = make_float2( 20.48, 25.6 );
    double dt = 0.014;

    EMF emf( ntiles, nx, box, dt );

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
    laser.W0 = 4.0;
    laser.focus = 20.48;
    laser.axis = 12.8;

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

    std::cout << "Starting test...\n";

    auto t0 = Timer("test");

    std::cout << "Clock resolution is " << t0.resolution() << " ns\n";

    t0.start();

    for( int i = 0; i < 1500; i ++) {
        emf.advance();
    }

    t0.stop();
    t0.report("1500 iterations:");

    save_emf( );

}

void test_inj( void ) {

    uint2 ntiles = make_uint2( 2, 2 );
    uint2 nx = make_uint2( 64, 64 );

    float2 box = make_float2( 12.8, 12.8 );

    auto dt = 0.99 * zpic::courant( ntiles, nx, box );

    Simulation sim( ntiles, nx, box, dt );

    uint2 ppc = make_uint2( 8, 8 );
    Species electrons("electrons", -1.0f, ppc);

    // electrons.set_density(Density::Step(coord::x, 1.0, 5.0));
    // electrons.set_density(Density::Slab(coord::y, 1.0, 5.0, 7.0));

    electrons.set_density( Density::Sphere( 1.0, make_float2(5.0, 7.0), 2.0 ) );
    electrons.set_udist( UDistribution::Cold( make_float3( 100, 50, 25 ) ) );

    sim.add_species(electrons);

    auto diag = [ & ]() {
        electrons.save_charge();
        sim.current.save(fcomp::x);
        sim.current.save(fcomp::y);
        sim.current.save(fcomp::z);
    };

    for( int i = 0; i < 1000; ++i ) {
        if ( i % 100 == 0 ) diag();

        sim.current.zero();
        electrons.advance( sim.emf, sim.current );
        sim.current.advance();
    }

    diag();

    std::cout << __func__ << " complete.\n";
}

void test_weibel( void ) {
    uint2 gnx = make_uint2( 128, 128 );
    uint2 ntiles = make_uint2( 8, 8 );

//    uint2 gnx = make_uint2( 256, 256);
//    uint2 ntiles = make_uint2( 16, 16 );

    uint2 nx = make_uint2( gnx.x/ntiles.x, gnx.y/ntiles.y );
    float2 box = make_float2( gnx.x/10.0, gnx.y/10.0 );
    float dt = 0.07;

    std::cout << "Starting Weibel test\n"
              << "ntiles    : " << ntiles << '\n'
              << "tile size : " << nx << '\n'
              << "gnx       : " << make_uint2( ntiles.x * nx.x, ntiles.y * nx.y ) << '\n'
              << "box       : " << box << '\n';

    Simulation sim( ntiles, nx, box, dt );

    uint2 ppc = make_uint2( 4, 4 );

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

    // Lambda function for diagnostic output
    auto diag = [ & ]( ) {
        sim.emf.save(emf::b, fcomp::x);
        sim.emf.save(emf::b, fcomp::y);
        sim.emf.save(emf::b, fcomp::z);

        electrons.save();

        sim.energy_info();
    };


    Timer timer; timer.start();

    while ( sim.get_t() <= 35.0 ) {
        // if ( sim.get_iter() % 10 == 0 )
            // std::cout << "t = " << sim.get_t() << '\n';
        sim.advance();
    }

    timer.stop();

    diag();

    auto perf = sim.get_nmove() / timer.elapsed(timer::s) / 1.e9;

    std::cout << "[benchmark] " << perf << " GPart/s\n";

    std::cout << "Elapsed time: " << timer.elapsed(timer::s) << " s"
              << ", Performance: " << perf << " GPart/s\n";

    
}

void test_cathode()
{

    Simulation sim(
        make_uint2(    8,    8), // ntiles
        make_uint2(   16,   16), // nx
        make_float2(12.8, 12.8), // box
        0.07                // dt
    );

    // Create cathode
    Cathode cathode(
        "cathode",
        +1.0f,       // m_q
        make_uint2(4, 4), // ppc
        edge::lower, // edge
        1.0e3f       // ufl
    );

    // Set additional cathode parameters
    cathode.n0 = 1.0f;
    cathode.wall = edge::lower;
    cathode.start = -6.4;
    //cathode.uth = float3(0.1, 0.1, 0.1);

    cathode.uth = make_float3(0,0,0);

    auto bc = cathode.get_bc();
    bc.x = {
        .lower = species::bc::open,
        .upper = species::bc::open};

    cathode.set_bc(bc);
    sim.add_species(cathode);

    // Lambda function for diagnostic output
    auto diag = [ & ]( ) {
        cathode.save_phasespace( 
            phasespace::x, make_float2( 0, 12.8 ), 128,
            phasespace::ux, make_float2( -1, 2.0e3 ), 512
        );
        cathode.save_charge();
        cathode.save();
        sim.current.save(fcomp::x);
        sim.emf.save(emf::e, fcomp::x);
    };

    float const tmax = 12.8;

    printf("Running Cathode test up to t = %g...\n", tmax);

    Timer timer;
    timer.start();

    // diag();
    while (sim.get_t() < tmax) {
        std::cout << "Now at t = " << sim.get_t() << '\n';
        sim.advance();
    /*    if (sim.get_iter() % 50 == 0) {
            std::cout << "Writing data\n";
            diag();
        } */
    }

    printf("Simulation complete at t = %g\n", sim.get_t());

    timer.stop();

    diag();

    auto perf = sim.get_nmove() / timer.elapsed(timer::s) / 1.e9;

    std::cerr << "Elapsed time: " << timer.elapsed(timer::s) << " s"
              << ", Performance: " << perf << " GPart/s\n";

}

void benchmark( void ) {
    uint2 gnx = make_uint2( 128, 128 );
    uint2 ppc = make_uint2( 2, 2 );

    auto dt = 0.07;
    auto tmax = 35.0;

    auto bench_weibel = [ & ]( uint2 const ntiles ) {
        uint2 nx = make_uint2( gnx.x/ntiles.x, gnx.y/ntiles.y );
        float2 box = make_float2( gnx.x/10., gnx.y/10. );

        Simulation sim( ntiles, nx, box, dt );

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

        Timer timer; timer.start();
        while ( sim.get_t() <= tmax ) { sim.advance(); }
        timer.stop();

        auto perf = sim.get_nmove() / timer.elapsed(timer::s) / 1.e9;

        std::cout << ntiles << " : " <<  timer.elapsed(timer::s)
                  << " s, " << perf << " GPart/s\n";
    };


    std::vector<uint2> ntiles_list{ {1,1}, {2,2}, {4,4}, {8,8}, {16,16} };
    for( auto ntiles : ntiles_list ){
        bench_weibel( ntiles );
    }
}

void test_weibel_large( )
{
                            
    // Create simulation box
    uint2 ntiles {64, 64};
    uint2 nx {32, 32};
    uint2 ppc {8, 8};
                                                                                                    
    uint64_t vol = static_cast<uint64_t>(nx.x * nx.y) *  static_cast<uint64_t>(ntiles.x * ntiles.y);
                                             
    std::cout << "** Large Weibel test **\n";                                   
    std::cout << " # tiles          : " << ntiles.x << ", " << ntiles.y << "\n";
    std::cout << " tile size        : " << nx.x << ", " << nx.y << "\n";                      
    std::cout << " global size      : " << nx.x * ntiles.x << ", " << nx.y * ntiles.y << "\n";
    std::cout << " # part / species : " << vol * (ppc.x * ppc.y) / (1048576.0) << " M \n";
                                                                  
    float2 box {nx.x * ntiles.x * 0.1f, nx.y * ntiles.y * 0.1f};
                    
    float dt = 0.07;
                                        
    Simulation sim(ntiles, nx, box, dt);
                            
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
                                                          
    printf("Running Weibel test up to n = %d...\n", imax);
                
    Timer timer;
                  
    timer.start();
                                 
    while (sim.get_iter() < imax)
    {
        // std::cout << "n = " << sim.get_iter() << '\n';
        sim.advance();
    }
                 
    timer.stop();
                                                              
    std::cout << "Simulation complete at i = " << sim.get_iter() << '\n';
                      
    sim.energy_info();
                                    
    sim.emf.save(emf::b, fcomp::x);
    sim.emf.save(emf::b, fcomp::y);
    sim.emf.save(emf::b, fcomp::z);
                                                                  
    auto perf = sim.get_nmove() / timer.elapsed(timer::s) / 1.e9;

    std::cerr << "Elapsed time: " << timer.elapsed(timer::s) << " s"
              << ", Performance: " << perf << " GPart/s\n";

}

void test_weibel_96( )
{
                            
    // Create simulation box
    uint2 ntiles {12, 16};
    uint2 nx {32, 32};
    uint2 ppc {8, 8};
                                                                                                                                                                      
    float2 box {nx.x * ntiles.x * 0.1f, nx.y * ntiles.y * 0.1f};
                    
    float dt = 0.07;
                                        
    Simulation sim(ntiles, nx, box, dt);
                            
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
                                                          
    printf("Running Weibel(96) test up to n = %d...\n", imax);
                
    Timer timer;
                  
    timer.start();
                                 
    while (sim.get_iter() < imax)
    {
        // std::cout << "n = " << sim.get_iter() << '\n';
        sim.advance();
    }
                 
    timer.stop();
                                                              
    std::cout << "Simulation complete at i = " << sim.get_iter() << '\n';
                      
    sim.energy_info();
                                    
    sim.emf.save(emf::b, fcomp::x);
    sim.emf.save(emf::b, fcomp::y);
    sim.emf.save(emf::b, fcomp::z);
                                                                  
    auto perf = sim.get_nmove() / timer.elapsed(timer::s) / 1.e9;

    std::cerr << "Elapsed time: " << timer.elapsed(timer::s) << " s"
              << ", Performance: " << perf << " GPart/s\n";

}

int main( void ) {

    info();

    // test_random();

    // test_laser();

    // test_inj();

    // test_weibel();

    // test_cathode();

    // benchmark();

    // test_weibel_large();

    test_weibel_96();
}
