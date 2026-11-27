#include <iostream>
#include <string>

#include "em2ds.hpp"
#include "util/term.hpp"

#include "core/bounds.hpp"
#include "grid/grid.hpp"

#include "util/complex.hpp"

using complex_t = util::complex<float>;

void test_tiled_grid( ) {
    
    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;

    // Global number of tiles
    // const uint2 grid_ntiles = { 4, 4 };
    const uint2 grid_ntiles = { 3, 2 };


    // const uint2 tile_dims     = { 12, 12 };
    const uint2 tile_dims     = { 11, 17 };

    bounds_2d<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {1,2};

    // Periodic boundaries
    const int2 periodic = make_int2( 1, 1 );

    grid::tiled<float> data( grid_ntiles, tile_dims, gc, periodic );

    // Get local number of tiles
    const auto ntiles   = data.get_grid_ntiles();

    std::cout << "Setting values...\n";

    data.zero( );
    data.set( 1.0 );

    #pragma omp parallel for
    for( unsigned int tid = 0; tid < ntiles.y * ntiles.x; tid++ ) {
        const uint2 tile_idx = { tid % ntiles.x, tid / ntiles.x  };
        float * const __restrict__ tile_data = data.tile_data(tid);

        const auto   tile_val = tile_idx.y * grid_ntiles.x + tile_idx.x;

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

    data.save( "openmp/openmp.zdf" );

    std::cout << ansi::bold;
    std::cout << "Done!\n";
    std::cout << ansi::reset;      
}

void test_tiled_vec3_grid( ) {

    std::cout << ansi::bold;
    std::cout << "Running test_grid()\n";
    std::cout << ansi::reset;
    std::cout << "Declaring test_vec3grid<float> data...\n";

    const uint2 grid_ntiles = { 8, 8 };
    const uint2 tile_dims = { 16,16 };
    
    bounds_2d<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {1,2};

    // Periodic boundaries
    const int2 periodic = make_int2( 1, 1 );

    grid::tiled_vec3< float > data( grid_ntiles, tile_dims, gc, periodic );

    // Get local number of tiles
    const auto ntiles = data.get_grid_ntiles();

    // Set zero
    // data.zero( );

    // Set constant
    // data.set( float3{1.0, 2.0, 3.0} );

    // Set different value per tile
    #pragma omp parallel for
    for( unsigned int tid = 0; tid < ntiles.y * ntiles.x; tid++ ) {
        const uint2 tile_idx = { tid % ntiles.x, tid / ntiles.x  };
        auto * const __restrict__ tile_data = data.tile_data( tid );

        const auto   tile_val = tile_idx.y * grid_ntiles.x + tile_idx.x;

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

    data.save( fcomp::x, "openmp/openmp-vec3-x.zdf" );
    data.save( fcomp::y, "openmp/openmp-vec3-y.zdf" );
    data.save( fcomp::z, "openmp/openmp-vec3-z.zdf" );

    std::cout << ansi::bold;
    std::cout << "Done!\n";
    std::cout << ansi::reset;
}

void test_halo( ) {
    
    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;

    // Global grid dimensions
    const uint2 global_dims = { 1024, 512 };

    // Guard cells
    bounds_2d<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {3,4};

    // Periodic boundaries
    const int2 periodic = make_int2( 1, 1 );

    grid::halo<float> data( global_dims, gc, periodic );

    const auto grid_dims = data.get_grid_dims();
    std::cout << "grid dims: " << grid_dims << '\n';

    std::cout << "Setting values...\n";

    data.zero( );

    //    data.set( 1.0 );

    auto stridey = data.get_grid_ext_dims().x;
    auto * __restrict__ buffer = & data.data()[ data.get_offset() ];
    for( int iy = 0; iy < static_cast<int>(grid_dims.y); iy++ ) {
        for( int ix = 0; ix < static_cast<int>(grid_dims.x); ix++ ) {
            buffer[ iy * stridey + ix ] = iy + ix;
        }
    }
    data.copy_to_gc();

    data.add_from_gc();
    data.copy_to_gc();

    for( auto i = 0; i < 5; i++)
       data.x_shift_left( 1 );

    // data.kernel3_x( 1., 2., 1. );
    data.kernel3_y( 1., 2., 1. );

    std::cout << "Saving data...\n";

    data.save( "openmp/openmp.zdf" );

    std::cout << ansi::bold;
    std::cout << "Done!\n";
    std::cout << ansi::reset;      
}

void test_flat( ) {
    
    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;

    // Global grid dimensions
    const uint2 global_dims = { 1024, 512 };

    // Guard cells
    bounds_2d<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {3,4};

    grid::flat<float> data( global_dims );

    const auto grid_dims = data.get_grid_dims();
    std::cout << "grid dims: " << grid_dims << '\n';

    std::cout << "Setting values...\n";

    data.zero( );

    //    data.set( 1.0 );

    auto stridey = data.get_grid_dims().x;
    auto * __restrict__ buffer = & data.data()[ 0 ];
    for( int iy = 0; iy < static_cast<int>(grid_dims.y); iy++ ) {
        for( int ix = 0; ix < static_cast<int>(grid_dims.x); ix++ ) {
            buffer[ iy * stridey + ix ] = iy + ix;
        }
    }

    std::cout << "Saving data...\n";

    data.save( "openmp/openmp.zdf" );

    std::cout << ansi::bold;
    std::cout << "Done!\n";
    std::cout << ansi::reset;      
}

void test_fft_tile( ) {

    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;

    // Periodic boundaries
    const int2 periodic = make_int2( 1, 1 );
    const uint2 global_dims = { 1024, 512 };
    const uint2 tile_dims = {16, 16 };

    bounds_2d<unsigned int> gc;
    gc.x = { 1, 2 };
    gc.y = { 1, 2 };

    // Input grid
    grid::tiled<float> data( 
        global_dims / tile_dims, 
        tile_dims, 
        gc, 
        periodic 
    );
    data.name = "Data";

    auto grid_ntiles = data.get_grid_ntiles();
    auto stridey = data.tile_ext_dims.x;

    #pragma omp parallel for collapse(2)
    for( unsigned int ty = 0; ty < grid_ntiles.y; ty++ ) {
        for( unsigned int tx = 0; tx < grid_ntiles.x; tx++ ) {
            float * const __restrict__ tile_data = data.tile_data(tx,ty);

            unsigned int ix0 = tx * tile_dims.x;
            unsigned int iy0 = ty * tile_dims.y;
            
            for( int iy = 0; iy < static_cast<int>(tile_dims.y); iy++ ) {
                for( int ix = 0; ix < static_cast<int>(tile_dims.x); ix++ ) {
                    float x = ( ix0 + ix - 512.0f ) / 512.f;
                    float y = ( iy0 + iy - 256.0f ) / 256.f;
                    tile_data[ iy * stridey + ix ] = std::exp( - (x*x)/0.001 - (y*y)/0.006 );
                }
            }
        }
    }

    data.save( "openmp/data.zdf");

    // Create FFT plan
    auto r2c_plan = grid::fft::r2c_plan( data );

    // Output grid
    auto cdata = r2c_plan.kspace_grid();
    cdata.name = "transform";

    // Transform data
    r2c_plan.transform( cdata, data );

    // Save output
    cdata.save( "openmp/transform.zdf");

    data.set( 0 );

    auto c2r_plan = grid::fft::c2r_plan( data );

    c2r_plan.transform( data, cdata );

    data.name = "test";
    data.save( "openmp/test.zdf");

    std::cout << ansi::bold;
    std::cout << "Done!\n";
    std::cout << ansi::reset;
}

void set_charge( grid::tiled<float> & charge, const float2 dx, const float2 center, const float r )
{
    auto grid_ntiles = charge.get_grid_ntiles();
    auto tile_dims = charge.tile_dims;

    int stridey = charge.tile_ext_dims.x;
    
    #pragma omp parallel for collapse(2)
    for( unsigned int ty = 0; ty < grid_ntiles.y; ty++ ) {
        for( unsigned int tx = 0; tx < grid_ntiles.x; tx++ ) {
            float * const __restrict__ tile_data = charge.tile_data(tx,ty);

            unsigned int ix0 = tx * tile_dims.x;
            unsigned int iy0 = ty * tile_dims.y;
            
            for( int iy = 0; iy < static_cast<int>(tile_dims.y); iy++ ) {
                for( int ix = 0; ix < static_cast<int>(tile_dims.x); ix++ ) {
                    float x = ( ix0 + ix ) * dx.x;
                    float y = ( iy0 + iy ) * dx.y;
                    tile_data[ iy * stridey + ix ] = (x-center.x)*(x-center.x) + (y-center.y) * (y-center.y) <= r*r;
                }
            }
        }
    }
}

void poisson( grid::flat<complex_t> & potential, const float2 dk )
{
    auto dims   = potential.get_grid_dims();

    #pragma omp parallel for
    for( unsigned idx = 0; idx < dims.y * dims.x; idx ++ ) {
        const int ix = idx % dims.x;
        const int iy = idx / dims.x;

        const float2 kvec = grid::fft::k( make_int2( ix, iy ), dims, dk );
        const float kx = kvec.x;
        const float ky = kvec.y;

        const float k2 = kx*kx + ky*ky;

        potential.data()[ idx ] *= ((k2 > 0)? 1.f / k2 : 0.);
    }

}

void test_poisson(){
    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;

    const float2 box{1.0, 1.0};

    // Periodic boundaries
    const int2 periodic = make_int2( 1, 1 );
    const uint2 global_dims = { 1024, 1024 };
    const uint2 tile_dims = {16, 16 };

    const float2 dx{ box.x / global_dims.x, box.y / global_dims.y };

    bounds_2d<unsigned int> gc;
    gc.x = { 1, 2 };
    gc.y = { 1, 2 };

    // Input grid
    grid::tiled<float> charge( 
        global_dims / tile_dims, 
        tile_dims, 
        gc, 
        periodic 
    );
    charge.name = "charge";

    grid::tiled<float> potential( 
        global_dims / tile_dims, 
        tile_dims, 
        gc, 
        periodic 
    );
    charge.name = "potential";

    set_charge ( charge, dx, float2{ 0.25, 0.25 }, 0.1 );

    charge.save( "openmp/charge.zdf");

    // Create FFT plan
    auto r2c_plan = grid::fft::r2c_plan( charge );

    // Output grid
    auto fpotential = r2c_plan.kspace_grid();
    
    // Transform data
    r2c_plan.transform( fpotential, charge );

    // Save F(charge)
    fpotential.name = "F(charge)";
    fpotential.save( "openmp/charge_k.zdf");

    poisson( fpotential, grid::fft::dk( box ) );

    fpotential.name = "F(potential)";
    fpotential.save( "openmp/potential_k.zdf" );

    // Save output
    auto c2r_plan = grid::fft::c2r_plan( charge );

    c2r_plan.transform( potential, fpotential );

    potential.save( "openmp/potential.zdf");

    std::cout << ansi::bold;
    std::cout << "Done!\n";
    std::cout << ansi::reset;
}

#include "emf.hpp"
#include "laser.hpp"

#include "util/timer.hpp"

void test_laser( ) {

    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;

    // Periodic boundaries
    const int2 periodic = make_int2( 1, 1 );

    uint2 ntiles{ 64, 16 };
    uint2 nx{ 16, 16 };

    float2 box{ 20.48, 25.6 };
    double dt{ 0.014 };

    std::cout << "Creating EMF object...\n";

    em2ds::emf emf( ntiles, nx, box, dt, periodic );

    auto save_emf = [ & emf ]( ) {
        emf.save( em2ds::emf::quantity::e, fcomp::x );
        emf.save( em2ds::emf::quantity::e, fcomp::y );
        emf.save( em2ds::emf::quantity::e, fcomp::z );

        emf.save( em2ds::emf::quantity::b, fcomp::x );
        emf.save( em2ds::emf::quantity::b, fcomp::y );
        emf.save( em2ds::emf::quantity::b, fcomp::z );

        emf.save( em2ds::emf::quantity::fet, fcomp::x );
        emf.save( em2ds::emf::quantity::fet, fcomp::y );
        emf.save( em2ds::emf::quantity::fet, fcomp::z );

        emf.save( em2ds::emf::quantity::fb, fcomp::x );
        emf.save( em2ds::emf::quantity::fb, fcomp::y );
        emf.save( em2ds::emf::quantity::fb, fcomp::z );

    };

/*
    em2ds::laser::plane_wave laser;
    laser.start = 10.2;
    laser.fwhm = 4.0;
    laser.a0 = 1.0;
    laser.omega0 = 10.0;
*/

    em2ds::laser::gaussian laser;
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

    // Periodic boundaries
    const int2 periodic = make_int2( 1, 1 );

    uint2 ntiles{ 4, 4 };
    uint2 nx{ 32, 32 };

    float2 box{ 12.8, 12.8 };

    auto dt = 0.99 * em2ds::courant( ntiles, nx, box );

    uint2 ppc{ 8, 8 };
    em2ds::species electrons( "electrons", -1.0f, ppc );

    std::cout << "Created species\n";

    //electrons.set_density(em2ds::density::step(coord::x, 1.0, 5.0));
    // electrons.set_density(em2ds::density::slab(coord::y, 1.0, 5.0, 8.0));
    electrons.set_density( em2ds::density::sphere( 1.0, float2{5.0, 7.0}, 2.0 ) );

    std::cout << "Density set\n";

    electrons.set_udist( em2ds::udist::thermal( float3{ 0.1, 0.2, 0.3 }, float3{1,0,0} ) );

    electrons.initialize( box, ntiles, nx, dt, 0, periodic );

    electrons.save_charge();
    electrons.save();
    electrons.save_phasespace(
        em2ds::phasespace::quantity::ux, float2{-1, 3}, 256,
        em2ds::phasespace::quantity::uz, float2{-1, 1}, 128
    );

    std::cout << ansi::bold;
    std::cout << __func__ << "() complete!\n";
    std::cout << ansi::reset;
}

void test_mov( ) {

    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;

    // Periodic boundaries
    const int2 periodic = make_int2( 1, 1 );

    uint2 ntiles{ 4, 4 };
    uint2 nx{ 32, 32 };

    float2 box{ 12.8, 12.8 };

    auto dt = 0.99 * em2ds::courant( ntiles, nx, box );

    uint2 ppc{ 8, 8 };
    em2ds::species electrons( "electrons", -1.0f, ppc );

    electrons.set_density( em2ds::density::sphere( 1.0, float2{2.1, 2.1}, 2.0 ) );
    electrons.set_udist( em2ds::udist::cold( float3{ -1, -2, -3 } ) );
    electrons.initialize( box, ntiles, nx, dt, 0, periodic );

    electrons.save_charge();
    electrons.save();

    int niter = 200; //200
    for( auto i = 0; i < niter; i ++ ) {
        std::cout << "i = " << i << ", total particles: " << electrons.np() << '\n';
        electrons.advance();
    }

    electrons.save_charge();
    electrons.save();

    std::cout << ansi::bold;
    std::cout << __func__ << "() complete!\n";
    std::cout << ansi::reset;
}

void test_current_charge( ) {

    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;

    // Periodic boundaries
    const int2 periodic = make_int2( 1, 1 );

    uint2 ntiles{ 4, 4 };
    uint2 nx{ 32, 32 };

    float2 box{ 12.8, 12.8 };

    auto dt = 0.99 * em2ds::courant( ntiles, nx, box );

    uint2 ppc{ 8, 8 };
    em2ds::species electrons( "electrons", -1.0f, ppc );

    electrons.set_density( em2ds::density::sphere( 1.0, float2{6.4, 6.4}, 5.0 ) );
    electrons.set_udist( em2ds::udist::cold( float3{ 1, 2, 3 } ) );

    electrons.initialize( box, ntiles, nx, dt, 0, periodic );

    em2ds::current current( ntiles, nx, box, dt, periodic );
    em2ds::charge charge( ntiles, nx, box, dt, periodic );

    electrons.advance( current, charge );

    current.advance( );
    current.save( em2ds::current::quantity::j, fcomp::x );
    current.save( em2ds::current::quantity::j, fcomp::y );
    current.save( em2ds::current::quantity::j, fcomp::z );
    current.save( em2ds::current::quantity::fj, fcomp::x );
    current.save( em2ds::current::quantity::fj, fcomp::y );
    current.save( em2ds::current::quantity::fj, fcomp::z );

    charge.advance();
    charge.save( em2ds::charge::quantity::rho );
    charge.save( em2ds::charge::quantity::frho );

    std::cout << ansi::bold;
    std::cout << __func__ << "() complete!\n";
    std::cout << ansi::reset;
}

void test_weibel( )
{
   
    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;

    // Create simulation box
    uint2 ntiles{16, 16};
    uint2 tile_dims{32, 32};
    float2 box = {tile_dims.x * ntiles.x * 0.1f, tile_dims.y * ntiles.y * 0.1f};
    float dt = 0.07;
                                        
    em2ds::simulation sim( ntiles, tile_dims, box, dt );
                            
    uint2 ppc{4, 4};

    em2ds::species electrons("electrons", -1.0f, ppc);
    electrons.set_udist(
        em2ds::udist::thermal_corr( 
            float3{ 0.1, 0.1, 0.1 },
            float3{ 0, 0, 0.6 }
        )
    );
    electrons.push_type = em2ds::species::pusher::euler;

    sim.add_species( electrons );

    em2ds::species positrons("positrons", +1.0f, ppc);
    positrons.set_udist(
        em2ds::udist::thermal_corr( 
            float3{ 0.1, 0.1, 0.1 },
            float3{ 0, 0, -0.6 }
        )
    );
    positrons.push_type = em2ds::species::pusher::euler;

    sim.add_species( positrons );

#if 1
    // Lambda function for diagnostic output
    auto diag = [ & ]( ) {
        sim.emf.save(em2ds::emf::quantity::b, fcomp::x);
        sim.emf.save(em2ds::emf::quantity::b, fcomp::y);
        sim.emf.save(em2ds::emf::quantity::b, fcomp::z);

        sim.emf.save(em2ds::emf::quantity::fet, fcomp::x);
        sim.emf.save(em2ds::emf::quantity::fet, fcomp::y);
        sim.emf.save(em2ds::emf::quantity::fet, fcomp::z);


        electrons.save_charge();
        positrons.save_charge();

        electrons.save_phasespace(
            em2ds::phasespace::quantity::ux, float2{-2, 2}, 256,
            em2ds::phasespace::quantity::uy, float2{-2, 2}, 256
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

    // Initialize FFT library
    grid::fft::init();

    em2ds::sys_info();

    // test_tiled_grid();
    // test_tiled_vec3_grid();
    // test_halo();
    // test_flat();

    // test_fft_tile();
    // test_poisson();
    //test_laser();

    // test_inj();
    // test_mov();
    // test_current_charge();
    test_weibel();

    grid::fft::cleanup();

}