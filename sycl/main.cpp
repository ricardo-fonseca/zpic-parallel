
#include <iostream>

#include "utils.h"
#include "vec_types.h"
#include "grid.h"

#include "vec3grid.h"
#include "emf.h"
#include "laser.h"

#include "timer.h"
#include "simulation.h"

void test_grid( sycl::queue & q ) {
    std::cout << "Declaring grid<float> data...\n";
    
    const uint2 ntiles{8,8};
    const uint2 nx{16,16};
    
    bnd<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {1,2};

    grid<float> data( ntiles, nx, gc, q );

    data.zero( );

    std::cout << "Setting values...\n";

#if 0
    {
        const auto ntiles   = data.ntiles;
        const auto tile_vol = data.tile_vol;
        const auto nx       = data.nx;
        const auto ext_nx   = data.ext_nx;
        const auto offset   = data.offset;
        auto * __restrict__ d_buffer = data.d_buffer;

        q.submit([&](sycl::handler &h) {

            h.parallel_for( 
                sycl::range{ ntiles.y, ntiles.x},
                [=](sycl::id<2> idx) { 
                
                const uint2 tile_idx( idx[0], idx[1] );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const int  ystride  = ext_nx.x;

                auto * __restrict__ buffer = & d_buffer[ tid * tile_vol + offset ];

                for( unsigned iy = 0; iy < nx.y; iy ++ ) {
                    for( unsigned ix = 0; ix < nx.x; ix ++ ) {
                        buffer[ iy * ystride + ix ] = tid;
                    }
                }
            });
        });
    }
#endif

#if 1
    {
        q.submit([&](sycl::handler &h) {

            const auto ntiles   = data.ntiles;
            const auto tile_vol = data.tile_vol;
            const auto nx       = data.nx;
            const auto ext_nx   = data.ext_nx;
            const auto offset   = data.offset;
            auto * __restrict__ d_buffer = data.d_buffer;

            // 8×1 work items per group
            sycl::range<2> local{ 8, 1 };

            // ntiles.x × ntiles.y groups
            sycl::range<2> global{ ntiles.x, ntiles.y };

            h.parallel_for( 
                sycl::nd_range{global * local , local},
                [=](sycl::nd_item<2> it) { 

                const auto tile_idx = make_uint2( it.get_group(0), it.get_group(1) );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const int  ystride  = ext_nx.x;

                auto * __restrict__ buffer = & d_buffer[ tid * tile_vol + offset ];

                for( unsigned idx = it.get_local_id(0); idx < nx.y * nx.x; idx += it.get_local_range(0) ) {
                    const auto iy = idx / nx.x; 
                    const auto ix = idx % nx.x;

                    buffer[ iy * ystride + ix ] = tid;
                }
            });
        });
    }
#endif


#if 0
    data.set( 3.0 );
#endif

    data.copy_to_gc( );

    data.add_from_gc( );

    data.x_shift_left( 1 );

    data.kernel3_x( 1., 2., 1. );
    data.kernel3_y( 1., 2., 1. );

    std::cout << "Saving data...\n";

    data.save( "sycl" );

    std::cout << "Waiting for queue...\n";
    q.wait();

    std::cout << "Done!\n";
}

void test_vec3grid( sycl::queue & q ) {
    std::cout << "Declaring grid<float> data...\n";
    
    const uint2 ntiles{8,8};
    const uint2 nx{16,16};
    
    bnd<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {1,2};

    vec3grid<float3> data( ntiles, nx, gc, q );

    data.zero( );

    std::cout << "Setting values...\n";

#if 0
    data.set( float3(1.0, 2.0, 3.0), q );
#endif

#if 1
    {
        q.submit([&](sycl::handler &h) {

            const auto ntiles   = data.ntiles;
            const auto tile_vol = data.tile_vol;
            const auto nx       = data.nx;
            const auto ext_nx   = data.ext_nx;
            const auto offset   = data.offset;
            auto * __restrict__ d_buffer = data.d_buffer;

            // 8×1 work items per group
            sycl::range<2> local{ 8, 1 };

            // ntiles.x × ntiles.y groups
            sycl::range<2> global{ ntiles.x, ntiles.y };

            h.parallel_for( 
                sycl::nd_range{global * local , local},
                [=](sycl::nd_item<2> it) { 

                const auto tile_idx = make_uint2( it.get_group().get_group_id(0), it.get_group().get_group_id(1) );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const int  ystride  = ext_nx.x;

                auto * __restrict__ buffer = & d_buffer[ tid * tile_vol + offset ];

                for( unsigned idx = it.get_local_id(0); idx < nx.y * nx.x; idx += it.get_local_range(0) ) {
                    const auto iy = idx / nx.x; 
                    const auto ix = idx % nx.x;

                    buffer[ iy * ystride + ix ] = make_float3( 1 + tid, 2 + tid, 3 + tid );
                }
            });
        });
    }
#endif


    data.copy_to_gc( );

    data.add_from_gc( );

    data.x_shift_left( 1 );

    data.kernel3_x( 1., 2., 1. );
    data.kernel3_y( 1., 2., 1. );

    std::cout << "Saving data...\n";

    data.save( fcomp::x, "sycl" );
    data.save( fcomp::y, "sycl" );
    data.save( fcomp::z, "sycl" );

    std::cout << "Waiting for queue...\n";
    q.wait();

    std::cout << "Done!\n";
}

void test_laser( sycl::queue & q ) {

    uint2 ntiles{ 64, 16 };
    uint2 nx{ 16, 16 };

    float2 box{ 20.48, 25.6 };
    double dt = 0.014;

    EMF emf( ntiles, nx, box, dt, q );

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

    std::cout << "Starting test - " << niter << " iterations...\n";

    auto t0 = Timer("test");

    t0.start();

    for( int i = 0; i < niter; i ++) {
        emf.advance( );
    }

    t0.stop();

    char buffer[128];
    snprintf( buffer, 127, "%d iterations: ", niter );

    t0.report( buffer );

    save_emf();

}

void test_inj( sycl::queue & q ) {

    std::cout << "starting " << __func__ << "...\n";

    uint2 ntiles{ 2, 2 };
    uint2 nx{ 64, 64 };

    float2 box{ 12.8, 12.8 };

    auto dt = 0.99 * zpic::courant( ntiles, nx, box );

    uint2 ppc{ 8, 8 };
    Species electrons( "electrons", -1.0f, ppc );

    electrons.set_density(Density::Step(coord::x, 1.0, 5.0));
    // electrons.set_density(Density::Slab(coord::y, 1.0, 5.0, 7.0));
    // electrons.set_density( Density::Sphere( 1.0, float2(5.0, 7.0), 2.0 ) );
    electrons.set_udist( UDistribution::Thermal( make_float3( 0.1, 0.2, 0.3 ), make_float3(1,0,0) ) );

    electrons.initialize( box, ntiles, nx, dt, 0, q );

    electrons.save_charge();
    electrons.save();
    electrons.save_phasespace(
        phasespace::ux, make_float2(-1, 3), 256,
        phasespace::uz, make_float2(-1, 1), 128
    );

    std::cout << __func__ << " complete.\n";
}


void test_mov( sycl::queue & q ) {

    std::cout << "starting " << __func__ << "...\n";

    uint2 ntiles{ 4, 4 };
    uint2 nx{ 32, 32 };

    float2 box{ 12.8, 12.8 };

    auto dt = 0.99 * zpic::courant( ntiles, nx, box );

    uint2 ppc{ 8, 8 };
    Species electrons( "electrons", -1.0f, ppc );

    electrons.set_density( Density::Sphere( 1.0, make_float2(5.0, 7.0), 2.0 ) );
    electrons.set_udist( UDistribution::Cold( make_float3( 1, 2, 3 ) ) );

    electrons.initialize( box, ntiles, nx, dt, 0, q );

    electrons.save_charge();

    int niter = 100;
    for( auto i = 0; i < niter; i ++ ) {
        electrons.advance();
    }

    electrons.save_charge();
    electrons.save();

    std::cout << __func__ << " complete.\n";
}

void test_current( sycl::queue & q ) {

    std::cout << "starting " << __func__ << "...\n";

    uint2 ntiles{ 4, 4 };
    uint2 nx{ 32, 32 };

    float2 box{ 12.8, 12.8 };

    auto dt = 0.99 * zpic::courant( ntiles, nx, box );

    uint2 ppc{ 8, 8 };
    Species electrons( "electrons", -1.0f, ppc );

    electrons.set_density( Density::Sphere( 1.0, make_float2(5.0, 7.0), 2.0 ) );
    electrons.set_udist( UDistribution::Cold( make_float3( 1, 2, 3 ) ) );

    electrons.initialize( box, ntiles, nx, dt, 0, q );

    Current current( ntiles, nx, box, dt, q );

    electrons.save_charge();

    electrons.advance( current );
    current.advance( );
    
    current.save( fcomp::x );
    current.save( fcomp::y );
    current.save( fcomp::z );

    std::cout << __func__ << " complete.\n";
}

void test_weibel( sycl::queue & q  ) {
    uint2 gnx{ 256, 256 };
    uint2 ntiles{ 16, 16 };

    uint2 nx{ gnx.x/ntiles.x, gnx.y/ntiles.y };
    float2 box = make_float2( gnx.x/10.0, gnx.y/10.0 );
    float dt = 0.07;

    Simulation sim( ntiles, nx, box, dt, q );

    uint2 ppc{ 4, 4 };

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
        //if ( sim.get_iter() % 10 == 0 )
        //    std::cout << "t = " << sim.get_t() << '\n';
        sim.advance();
    }

    timer.stop();

    diag();

    auto perf = sim.get_nmove() / timer.elapsed(timer::s) / 1.e9;

    std::cout << "[benchmark] " << perf << " GPart/s\n";

//    std::cerr << "Elapsed time: " << timer.elapsed(timer::s) << " s"
//              << ", Performance: " << perf << " GPart/s\n";
}

void test_weibel_large( sycl::queue & q  )
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
                                        
    Simulation sim(ntiles, nx, box, dt, q);
                            
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

void test_warm( sycl::queue & q  ) 
{

    Simulation sim(         
        uint2{32, 32},        // ntiles
        uint2{32, 32},        // nx
        float2{102.4, 102.4}, // box
        0.07,                 // dt
        q
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


int main( void ) {

    // Run on cpu
    sycl::queue q{sycl::cpu_selector_v};;
    
    // Run on gpu
    // sycl::queue q{sycl::gpu_selector_v};;

    print_dev_info( q );

    // test_grid(q);
    // test_vec3grid(q);
    // test_laser( q );
    // test_inj( q );

    // test_mov( q );
    // test_current( q );

    test_weibel( q );

    // test_weibel_large( q );

    // test_warm( q );
}
