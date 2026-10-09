#include <iostream>

#include "core/gpu.cuh"
#include "device_types.h"
#include "grid/tiled.cuh"
#include "grid/tiled_vec3.cuh"

#include "parallel/mpi.hpp"
#include "util/term.hpp"
#include "util/complex.hpp"

#include "core/bounds.hpp"
#include "core/vec_types.cuh"
#include "grid/grid.cuh"

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

}

void test_tiled_grid( ) {
    
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
//    data.set( 123.0 );

    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 64 );
    kernel::test_tiled <<< grid, block >>>
        (data.view(), global_ntiles, local_tile_start );

    data.add_from_gc();
    data.copy_to_gc();

    for( auto i = 0; i < 10; i++) {
        data.x_shift_left( 1 );
    }

    data.kernel3_x( 1., 2., 1. );
    data.kernel3_y( 1., 2., 1. );

    parallel.barrier();
    if ( mpi::root() )
        std::cout << "Saving data...\n";

    data.save( "cuda-mpi/cuda-mpi.zdf" );
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

/*
    // Set different value per tile
    dim3 grid( local_ntiles.x, local_ntiles.y );
    dim3 block( 64 );
    kernel::test_vec3 <<< grid, block >>>
        (data.view(), global_ntiles, local_tile_start );
*/
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

    data.save( fcomp::x, "cuda-mpi/mpi-vec3-x.zdf" );
    data.save( fcomp::y, "cuda-mpi/mpi-vec3-y.zdf" );
    data.save( fcomp::z, "cuda-mpi/mpi-vec3-z.zdf" );

    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Done!\n";
        std::cout << ansi::reset;
    }
}

#if 0

void test_halo( ) {
    
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

    mpi::cart2d parallel( partition );

    grid::halo<float> data( global_dims, gc, parallel );

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
    for( int iy = 0; iy < static_cast<int>(local_dims.y); iy++ ) {
        for( int ix = 0; ix < static_cast<int>(local_dims.x); ix++ ) {
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

#endif

namespace kernel {

/**
 * @brief Fills grid values for test_flat
 *
 * @note Must be called with <<< local_dims.y, threads per block >>>
 * 
 * @param buffer 
 * @param local_dims 
 * @param local_start 
 */
__global__
void test_flat( float * __restrict__ buffer, const uint2 local_dims, const uint2 local_start ) {

    // Kernel must be called with one block per line
    const int iy = blockIdx.x;
    const int stridey = local_dims.x;

    for( int ix = gpu::block::thread_rank(); ix < local_dims.x; ix += gpu::block::num_threads()) {
        buffer[ iy * stridey + ix ] = (local_start.y + iy ) + (local_start.x + ix );
    }
}


} // namespace kernel


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

    mpi::cart2d parallel( partition );

    grid::flat<float> data( global_dims, parallel );

    const auto local_dims = data.get_local_dims();
    mpi::cout << "local dims: " << local_dims << '\n';
    parallel.barrier();

    const auto local_start  = data.get_local_start();
    mpi::cout << "local start: " << local_start << '\n';

    if ( mpi::root() ) {
        std::cout << "Setting values...\n";
    }

    // data.zero( );
    // data.set( 1.0 );

    kernel::test_flat <<< local_dims.y, 64 >>> (
        & data.data()[ 0 ], local_dims, local_start
    );

    parallel.barrier();
    if ( mpi::root() )
        std::cout << "Saving data...\n";

    data.save( "cuda-mpi/cuda-mpi.zdf" );

    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Done!\n";
        std::cout << ansi::reset;
    }      
}

#include "grid/fft.cuh"

namespace kernel {
__global__
void test_fft_tile( grid::tiled_view<float> tiles, uint2 const tile_start ) {

    auto tx = blockIdx.x;
    auto ty = blockIdx.y;

    float * const __restrict__ tile_data = tiles.tile_data( tx, ty );
    
    unsigned int ix0 = ( tile_start.x + tx ) * tiles.tile_dims.x;
    unsigned int iy0 = ( tile_start.y + ty ) * tiles.tile_dims.y;
    
    for( auto idx = gpu::block::thread_rank(); 
        idx < tiles.tile_dims.y * tiles.tile_dims.x; 
        idx += gpu::block::num_threads() ) {
        const auto iy = idx / tiles.tile_dims.x; 
        const auto ix = idx % tiles.tile_dims.x;

        float x = ( ix0 + ix - 512.0f ) / 512.f;
        float y = ( iy0 + iy - 256.0f ) / 256.f;
        tile_data[ iy * tiles.tile_ystride() + ix ] = 
            exp( - (x*x)/0.001 - (y*y)/0.006 );
    }
}

} // namespace kernel

void test_fft_tile( ) {

    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    mpi::cart2d parallel( uint2 { 1, 4 } );
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
    auto tile_start = data.get_local_tile_start();

    dim3 grid( local_ntiles.x, local_ntiles.y );
    dim3 block( 64 );
    kernel::test_fft_tile <<< grid, block >>>
        (data.view(), tile_start );

    data.save( "cuda-mpi/data.zdf");

    // Create R2C plan
    auto r2c_plan = grid::fft::r2c_plan( data );

    // Output grid
    auto cdata = r2c_plan.kspace_grid( );
    cdata.name = "transform";

    // Transform data
    r2c_plan.transform( cdata, data );

    // Save output
    grid::fft::kspace_save( cdata, "cuda-mpi/transform.zdf");

    data.set( 0 );


    auto c2r_plan = grid::fft::c2r_plan( data );

    c2r_plan.transform( data, cdata );

    data.name = "test";
    data.save( "cuda-mpi/test.zdf");

    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Done!\n";
        std::cout << ansi::reset;
    }
}

namespace kernel {
__global__
void set_charge( grid::tiled_view<float> tiles, uint2 const local_tile_start,
    const float2 dx, const float2 center, const float r )
{
    auto tile_dims = tiles.tile_dims;
    int tile_ystride = tiles.tile_ystride();

    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    auto * const __restrict__ tile_data = tiles.tile_data( tile_idx );

    unsigned int ix0 = ( local_tile_start.x + tile_idx.x ) * tile_dims.x;
    unsigned int iy0 = ( local_tile_start.y + tile_idx.y ) * tile_dims.y;
    
    for( auto idx = gpu::block::thread_rank(); 
        idx < tiles.tile_dims.y * tiles.tile_dims.x; 
        idx += gpu::block::num_threads() ) {
        const auto iy = idx / tiles.tile_dims.x; 
        const auto ix = idx % tiles.tile_dims.x;

        float x = ( ix0 + ix ) * dx.x;
        float y = ( iy0 + iy ) * dx.y;
        tile_data[ iy * tile_ystride + ix ] = (x-center.x)*(x-center.x) + (y-center.y) * (y-center.y) <= r*r;
    }
}

/**
 * @brief 
 * 
 * @param potential 
 * @param global_dims 
 * @param local_start 
 * @param dk 
 */
__global__
void poisson( util::complex64 * potential, 
    const uint2 global_dims, 
    const uint2 local_dims,
    const uint2 local_start,
    const float2 dk )
{

    const int iy = blockIdx.x;

    for( int ix = gpu::block::thread_rank();
         ix < local_dims.x; 
         ix += gpu::block::num_threads() ) {

        const int kiy = local_start.y + iy;
        const float ky = (( 2 * kiy < int(global_dims.y) ) ? kiy : ( kiy - int(global_dims.y) ) ) * dk.y;
        const float kx = ( ix + local_start.x ) * dk.x;
        const float k2 = kx*kx + ky*ky;

        potential[ iy * local_dims.x + ix ] *= ((k2 > 0)? 1.f / k2 : 0.);

    }
}

} // namespace kernel

void test_poisson(){
    if ( mpi::root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    const float2 box{1.0, 1.0};

    mpi::cart2d parallel( uint2 { 1, 4 } );
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
    potential.name = "potential";

    auto local_ntiles = charge.get_local_ntiles();
    dim3 grid( local_ntiles.x, local_ntiles.y );
    dim3 block( 64 );
    kernel::set_charge <<< grid, block >>>
        ( charge.view(), charge.get_local_tile_start(), 
                 dx, float2{ 0.25, 0.25 }, 0.1 );

    charge.save( "cuda-mpi/charge.zdf");

    // Create FFT plan
    auto r2c_plan = grid::fft::r2c_plan( charge );

    // Output grid
    auto fpotential = r2c_plan.kspace_grid( );

    // Transform data
    r2c_plan.transform( fpotential, charge );

    // Save F(charge)
    fpotential.name = "F(charge)";
    grid::fft::kspace_save( fpotential, "cuda-mpi/charge_k.zdf");

    kernel::poisson<<< fpotential.get_local_dims().y, 64 >>> ( 
        reinterpret_cast< util::complex64 * > (fpotential.data()), 
        fpotential.get_global_dims(),
        fpotential.get_local_dims(), fpotential.get_local_start(),
        grid::fft::dk( box ) );

    fpotential.name = "F(potential)";
    grid::fft::kspace_save( fpotential, "cuda-mpi/potential_k.zdf" );

    // Save output
    auto c2r_plan = grid::fft::c2r_plan( charge );

    c2r_plan.transform( potential, fpotential );

    potential.save( "cuda-mpi/potential.zdf");

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

    mpi::cart2d parallel( uint2 { 1, 4 } );

    uint2 ntiles{ 64, 16 };
    uint2 nx{ 16, 16 };

    float2 box{ 20.48, 25.6 };
    double dt{ 0.014 };

    if ( mpi::root() ) std::cout << "Creating EMF object...\n";

    emf emf( ntiles, nx, box, dt, parallel );

    auto save_emf = [ & emf ]( ) {
        emf.save( emf::quantity::e, fcomp::x );
        emf.save( emf::quantity::e, fcomp::y );
        emf.save( emf::quantity::e, fcomp::z );

        emf.save( emf::quantity::b, fcomp::x );
        emf.save( emf::quantity::b, fcomp::y );
        emf.save( emf::quantity::b, fcomp::z );

        emf.save( emf::quantity::fet, fcomp::x );
        emf.save( emf::quantity::fet, fcomp::y );
        emf.save( emf::quantity::fet, fcomp::z );

        emf.save( emf::quantity::fb, fcomp::x );
        emf.save( emf::quantity::fb, fcomp::y );
        emf.save( emf::quantity::fb, fcomp::z );

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

#if 0

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

    auto dt = 0.99 * zpic::courant( ntiles, nx, box );

    uint2 ppc{ 8, 8 };
    species electrons( "electrons", -1.0f, ppc );

    parallel.barrier();
    if ( mpi::root() ) std::cout << "Created species\n";

    //electrons.set_density(density::step(coord::x, 1.0, 5.0));
    // electrons.set_density(density::slab(coord::y, 1.0, 5.0, 8.0));
    electrons.set_density( density::sphere( 1.0, float2{5.0, 7.0}, 2.0 ) );

    parallel.barrier();
    if ( mpi::root() ) std::cout << "Density set\n";

    electrons.set_udist( udist::thermal( float3{ 0.1, 0.2, 0.3 }, float3{1,0,0} ) );

    electrons.initialize( box, ntiles, nx, dt, 0, parallel );

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

    auto dt = 0.99 * zpic::courant( ntiles, nx, box );

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

void test_current_charge( ) {

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

    auto dt = 0.99 * zpic::courant( ntiles, nx, box );

    uint2 ppc{ 8, 8 };
    species electrons( "electrons", -1.0f, ppc );

    electrons.set_density( density::sphere( 1.0, float2{6.4, 6.4}, 5.0 ) );
    electrons.set_udist( udist::cold( float3{ 1, 2, 3 } ) );

    electrons.initialize( box, ntiles, nx, dt, 0, parallel );

    current current( ntiles, nx, box, dt, parallel );
    charge charge( ntiles, nx, box, dt, parallel );

    electrons.advance( current, charge );

    current.advance( );
    current.save( current::quantity::j, fcomp::x );
    current.save( current::quantity::j, fcomp::y );
    current.save( current::quantity::j, fcomp::z );
    current.save( current::quantity::fj, fcomp::x );
    current.save( current::quantity::fj, fcomp::y );
    current.save( current::quantity::fj, fcomp::z );

    charge.advance();
    charge.save( charge::quantity::rho );
    charge.save( charge::quantity::frho );

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

#endif

/**
 * @brief Print information about the environment
 * 
 * @note Only MPI root node prints this
 */
void info( void ) {

    if ( mpi::root() ) {

        std::cout << ansi::bold;
        std::cout << "Environment\n";
        std::cout << ansi::reset;

        char name[MPI_MAX_PROCESSOR_NAME];
        int len; MPI_Get_processor_name(name, &len);

        std::cout << "MPI running on " << mpi::size() << " processes\n";

        std::cout << "GPU devices on rank 0 (" << name << "):\n";
        gpu::print_info();
    }
}

int main( int argc, char *argv[] ) {

    // Initialize the MPI environment
    mpi::init( & argc, & argv );

    gpu::parallel_init( MPI_COMM_WORLD );

    info();

    grid::fft::init();

    // test_tiled_grid();
    // test_tiled_vec3_grid();
    // test_halo();
    // test_flat();

    // test_fft_tile();
    // test_poisson();
    test_laser();

    // test_inj();
    // test_mov();
    // test_current_charge();

    // test_weibel();

    grid::fft::cleanup();

    // Finalize the MPI environment
    mpi::finalize();

}