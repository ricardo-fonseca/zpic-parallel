#include <iostream>

#include "basic_grid.h"

/**
 * MPI support 
 */
#include "parallel.h"

void test_basic_grid( void ) {
    
    if ( mpi::world_root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    // Parallel partition
    uint2 partition = make_uint2( 2, 2 );

    // Global grid dimensions
    const uint2 global_dims = { 1024, 512 };

    // Guard cells
    bnd<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {3,4};

    Partition parallel( partition );

    basic_grid<float> data( global_dims, gc, parallel );

    const auto local_dims = data.get_local_dims();
    mpi::cout << "local dims: " << local_dims << '\n';
    parallel.barrier();

    const auto local_pos  = data.get_local_pos();
    mpi::cout << "local pos: " << local_pos << '\n';

    if ( mpi::world_root() ) {
        std::cout << "Setting values...\n";
    }

    data.zero( );

    //    data.set( 1.0 );

    auto stridey = data.get_local_ext_dims().x;
    auto * __restrict__ buffer = & data.d_buffer[ data.get_offset() ];
    for( int iy = 0; iy < local_dims.y; iy++ ) {
        for( int ix = 0; ix < local_dims.x; ix++ ) {
            buffer[ iy * stridey + ix ] = (local_pos.y + iy ) + (local_pos.x + ix );
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
    if ( mpi::world_root() )
        std::cout << "Saving data...\n";

    data.save( "mpi/mpi.zdf" );

    if ( mpi::world_root() ) {
        std::cout << ansi::bold;
        std::cout << "Done!\n";
        std::cout << ansi::reset;
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

int main( int argc, char *argv[] ) {

    // Initialize the MPI environment
    mpi::init( & argc, & argv );

    info();

    test_basic_grid( );  

    // Finalize the MPI environment
    mpi::finalize();

}