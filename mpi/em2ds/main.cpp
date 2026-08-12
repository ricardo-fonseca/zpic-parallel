#include <iostream>

#include "bounds.hpp"

#include "grid/grid.hpp"

// #include "transpose.h"

/**
 * MPI support 
 */
#include "parallel.hpp"

#include "grid/fft.hpp"

void test_tiled_grid( ) {
    
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


    // const uint2 nx     = { 12, 12 };
    const uint2 tile_dims     = { 11, 17 };

    bounds_2d<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {1,2};

    Partition parallel( partition );

    grid::tiled<float> data( global_ntiles, tile_dims, gc, parallel );

    // Get local number of tiles
    const auto ntiles   = data.get_local_ntiles();

    const uint2 local_tile_start = data.get_local_tile_start();

    if ( mpi::root() ) {
        std::cout << "Setting values...\n";
    }

    data.zero( );
    data.set( 1.0 );

    auto ext_dims = data.tile_ext_dims;

    for( unsigned int tid = 0; tid < ntiles.y * ntiles.x; tid++ ) {
        const uint2 tile_idx = { tid % ntiles.x, tid / ntiles.x  };
        const size_t tile_off = tid * roundup4( ext_dims.x * ext_dims.y );
        float * const __restrict__ tile_data = & data.data()[ tile_off + data.offset ];

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

void test_vec3_tiled_grid( ) {

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

    Partition parallel( partition );

    grid::vec3_tiled< float > data( global_ntiles, tile_dims, gc, parallel );

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
        const size_t tile_off = tid * roundup4( data.tile_ext_dims.x * data.tile_ext_dims.y );
        auto * const __restrict__ tile_data = & data.data()[ data.offset + tile_off ];

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

void test_ghosted( ) {
    
    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    // Parallel partition
    uint2 partition = make_uint2( 2, 2 );

    // Global grid dimensions
    const uint2 global_dims = { 1024, 512 };

    // Guard cells
    bounds_2d<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {3,4};

    Partition parallel( partition );

    grid::ghosted<float> data( global_dims, gc, parallel );

    const auto local_dims = data.get_local_dims();
    mpi::cout << "local dims: " << local_dims << '\n';
    parallel.barrier();

    const auto local_start  = data.get_local_start();
    mpi::cout << "local start: " << local_start << '\n';

    if ( mpi::root() ) {
        std::cout << "Setting values...\n";
    }

    data.zero( );

    //    data.set( 1.0 );

    auto stridey = data.get_local_ext_dims().x;
    auto * __restrict__ buffer = & data.data()[ data.get_offset() ];
    for( int iy = 0; iy < local_dims.y; iy++ ) {
        for( int ix = 0; ix < local_dims.x; ix++ ) {
            buffer[ iy * stridey + ix ] = (local_start.y + iy ) + (local_start.x + ix );
        }
    }
    data.copy_to_gc();

    data.add_from_gc();
    data.copy_to_gc();

    for( auto i = 0; i < 5; i++)
       data.x_shift_left( 1 );

    // data.kernel3_x( 1., 2., 1. );
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

void test_flat( ) {
    
    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    // Parallel partition
    uint2 partition = make_uint2( 2, 2 );

    // Global grid dimensions
    const uint2 global_dims = { 1024, 512 };

    // Guard cells
    bounds_2d<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {3,4};

    Partition parallel( partition );

    grid::flat<float> data( global_dims, parallel );

    const auto local_dims = data.get_local_dims();
    mpi::cout << "local dims: " << local_dims << '\n';
    parallel.barrier();

    const auto local_start  = data.get_local_start();
    mpi::cout << "local pos: " << local_start << '\n';

    if ( mpi::root() ) {
        std::cout << "Setting values...\n";
    }

    data.zero( );

    //    data.set( 1.0 );

    auto stridey = data.get_local_dims().x;
    auto * __restrict__ buffer = & data.data()[ 0 ];
    for( int iy = 0; iy < local_dims.y; iy++ ) {
        for( int ix = 0; ix < local_dims.x; ix++ ) {
            buffer[ iy * stridey + ix ] = (local_start.y + iy ) + (local_start.x + ix );
        }
    }

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

void test_fft_tile( ) {

    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    Partition parallel( uint2 { 1, 4 } );
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
        parallel 
    );
    data.name = "Data";

    auto local_ntiles = data.get_local_ntiles();
    auto stridey = data.tile_ext_dims.x;

    auto tile_start = data.get_local_tile_start();

    #pragma omp parallel for collapse(2)
    for( unsigned int ty = 0; ty < local_ntiles.y; ty++ ) {
        for( unsigned int tx = 0; tx < local_ntiles.x; tx++ ) {
            float * const __restrict__ tile_data = data.tile_data(tx,ty) + data.offset;

            unsigned int ix0 = ( tile_start.x + tx ) * tile_dims.x;
            unsigned int iy0 = ( tile_start.y + ty ) * tile_dims.y;
            
            for( int iy = 0; iy < tile_dims.y; iy++ ) {
                for( int ix = 0; ix < tile_dims.x; ix++ ) {
                    float x = ( ix0 + ix - 512.0f ) / 512.f;
                    float y = ( iy0 + iy - 256.0f ) / 256.f;
                    tile_data[ iy * stridey + ix ] = std::exp( - (x*x)/0.001 - (y*y)/0.006 );
                }
            }
        }
    }

    data.save( "mpi/data.zdf");

    // Output grid
    auto cdata = grid::fft::complex_grid( data );
    cdata.name = "transform";

    // Create FFT plan
    auto r2c_plan = grid::fft::r2c_plan( data );

    // Transform data
    r2c_plan.transform( cdata, data );

    // Save output
    cdata.save( "mpi/transform.zdf");

    data.set( 0 );

    auto c2r_plan = grid::fft::c2r_plan( data );

    c2r_plan.transform( data, cdata );

    data.name = "test";
    data.save( "mpi/test.zdf");

    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Done!\n";
        std::cout << ansi::reset;
    }
}

void set_charge( grid::tiled<float> & charge, const float2 dx, const float2 center, const float r )
{
    auto local_ntiles = charge.get_local_ntiles();
    auto tile_start = charge.get_local_tile_start();
    auto tile_dims = charge.tile_dims;

    int stridey = charge.tile_ext_dims.x;
    
    #pragma omp parallel for collapse(2)
    for( unsigned int ty = 0; ty < local_ntiles.y; ty++ ) {
        for( unsigned int tx = 0; tx < local_ntiles.x; tx++ ) {
            float * const __restrict__ tile_data = &charge.tile_data(tx,ty)[charge.offset];

            unsigned int ix0 = ( tile_start.x + tx ) * tile_dims.x;
            unsigned int iy0 = ( tile_start.y + ty ) * tile_dims.y;
            
            for( int iy = 0; iy < tile_dims.y; iy++ ) {
                for( int ix = 0; ix < tile_dims.x; ix++ ) {
                    float x = ( ix0 + ix ) * dx.x;
                    float y = ( iy0 + iy ) * dx.y;
                    tile_data[ iy * stridey + ix ] = (x-center.x)*(x-center.x) + (y-center.y) * (y-center.y) <= r*r;
                }
            }
        }
    }
}

void poisson( grid::flat<std::complex<float>> & potential, const float2 dk )
{
    auto start  = potential.get_local_start();
    auto dims   = potential.get_local_dims();
    auto global = potential.get_global_dims();

    #pragma omp parallel for
    for( unsigned idx = 0; idx < dims.y * dims.x; idx ++ ) {
        const int ix = start.x + idx % dims.x;
        const int iy = start.y + idx / dims.x;

        // Note that ky is along the x direction, and kx is along the y direction
        const float ky = (( 2 * ix < int(global.x) ) ? ix : ( ix - int(global.x) ) ) * dk.y;
        const float kx = iy * dk.x;

        const float k2 = kx*kx + ky*ky;

        potential.data()[ idx ] *= ((k2 > 0)? 1.f / k2 : 0.);
    }

}

void test_poisson(){
    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    const float2 box{1.0, 1.0};

    Partition parallel( uint2 { 1, 4 } );
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
        parallel 
    );
    charge.name = "charge";

    grid::tiled<float> potential( 
        global_dims / tile_dims, 
        tile_dims, 
        gc, 
        parallel 
    );
    charge.name = "potential";

    set_charge ( charge, dx, float2{ 0.25, 0.25 }, 0.1 );

    charge.save( "mpi/charge.zdf");

    // Output grid
    auto fpotential = grid::fft::complex_grid( charge );
    
    // Create FFT plan
    auto r2c_plan = grid::fft::r2c_plan( charge );

    // Transform data
    r2c_plan.transform( fpotential, charge );

    // Save F(charge)
    fpotential.name = "F(charge)";
    fpotential.save( "mpi/charge_k.zdf");

    poisson( fpotential, grid::fft::dk( box ) );

    fpotential.name = "F(potential)";
    fpotential.save( "mpi/potential_k.zdf" );

    // Save output
    auto c2r_plan = grid::fft::c2r_plan( charge );

    c2r_plan.transform( potential, fpotential );

    potential.save( "mpi/potential.zdf");

    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Done!\n";
        std::cout << ansi::reset;
    }
}

#include "emf.hpp"
#include "laser.hpp"

#include "timer.hpp"

void test_laser( ) {

    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    Partition parallel( uint2 { 1, 4 } );

    uint2 ntiles{ 64, 16 };
    uint2 nx{ 16, 16 };

    float2 box{ 20.48, 25.6 };
    double dt{ 0.014 };

    if ( mpi::root() ) std::cout << "Creating EMF object...\n";

    EMF emf( ntiles, nx, box, dt, parallel );

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

    std::ostringstream buffer;
    buffer << niter << " iterations: ";

    if ( mpi::root() ) t0.report( buffer.str() );

    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Done!\n";
        std::cout << ansi::reset;
    }

}


/**
 * @brief Print information about the environment
 * 
 */
void info( ) {

    if ( mpi::root() ) {

        std::cout << "MPI running on " << mpi::size() << " processes\n";

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

int main( int argc, char *argv[] ) {

    // Initialize the MPI environment
    mpi::init( & argc, & argv );

    grid::fft::init();

    info();

    // test_tiled_grid();
    // test_vec3_tiled_grid();
    // test_ghosted();
    // test_flat();

    //test_fft_tile();
    // test_poisson();
    test_laser();

    grid::fft::cleanup();

    // Finalize the MPI environment
    mpi::finalize();

}