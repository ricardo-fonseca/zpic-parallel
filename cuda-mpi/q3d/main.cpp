#include <iostream>
#include <fstream>

/**
 * MPI support 
 */
#include "parallel.h"

#include "cylmodes.h"
#include "cyl3modes.h"

void test_cylgrid( void ) {

    if ( mpi::world_root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    // Parallel partition
    uint2 partition = make_uint2( 2, 2 );

    uint2 ntiles{ 8, 8 };
    uint2 nx{ 16, 16 };
    bnd<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {1,2};

    Partition parallel( partition, make_int2(1,0) );

    CylGrid<float> field( 4, ntiles, nx, gc, parallel );

    if ( mpi::world_root() ) {
        std::cout << "field: " << field << '\n';

        std::cout << "mode0: " << field.mode0() << '\n';
        std::cout << "mode1: " << field.mode(1) << '\n';
        std::cout << "mode2: " << field.mode(2) << '\n';
        std::cout << "mode3: " << field.mode(3) << '\n';

        std::cout << ansi::bold;
        std::cout << "Done!\n";
        std::cout << ansi::reset;
    }

}

void test_cyl3_cylgrid( void ) {

    if ( mpi::world_root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    // Parallel partition
    uint2 partition = make_uint2( 2, 2 );

    uint2 ntiles{ 8, 8 };
    uint2 nx{ 16, 16 };
    bnd<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {1,2};

    Partition parallel( partition, make_int2(1,0) );

    Cyl3CylGrid<float> field( 4, ntiles, nx, gc, parallel );
    field.zero();

    cyl3<float> val0;
/*
    val0.z =  1 +     mpi::world_rank();
    val0.r =  1 + 2 * mpi::world_rank();
    val0.th = 1 + 3 * mpi::world_rank();
*/
    val0 = cyl3<float>{ 1, 2, 3 };
    field.mode0().set(val0);
    cyl3<ops::complex<float>> val1{ .1, .2, .3 };
    field.mode(1).set(val1);

    field.add_from_gc();

    field.copy_to_gc();
    for( auto i = 0; i < 5; i++)
       field.x_shift_left( 1 );

    field.kernel3_x( 1., 2., 1. );
    field.kernel3_y( 1., 2., 1. );

    if ( mpi::world_root() )
        std::cout << "Saving data...\n";

    field.mode0().save( fcomp::z, "test/cylgrid0_z.zdf");
    field.mode0().save( fcomp::r, "test/cylgrid0_r.zdf");
    field.mode0().save( fcomp::th, "test/cylgrid0_th.zdf");

    field.mode(1).save( fcomp::z, "test/cylgrid1_z.zdf");
    field.mode(1).save( fcomp::r, "test/cylgrid1_r.zdf");
    field.mode(1).save( fcomp::th, "test/cylgrid1_th.zdf");

    if ( mpi::world_root() ) {
       std::cout << ansi::bold;
        std::cout << "Done!\n";
        std::cout << ansi::reset;
    }
}

#include "emf.h"

void test_emf( void ) {

    if ( mpi::world_root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }
    // Parallel partition
    uint2 partition = make_uint2( 2, 2 );
    Partition parallel( partition, make_int2(1,0) );

    uint2 ntiles{ 8, 8 };
    uint2 nx{ 16, 16 };

    float2 box{ 12.8, 12.8 };
    double dt{ 0.04 };

    EMF emf( 4, ntiles, nx, box, dt, parallel );

    if ( mpi::world_root() ) {
        std::cout << "field: " << emf << '\n';

        std::cout << "mode0: " << emf.E->mode0() << '\n';
        std::cout << "mode1: " << emf.B->mode(1) << '\n';
        std::cout << "mode2: " << emf.E->mode(2) << '\n';
        std::cout << "mode3: " << emf.B->mode(3) << '\n';

        std::cout << ansi::bold
                << "Completed " << __func__ << "()\n"
                << ansi::reset;
    }
}

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

#include "simulation.h"
#include "laser.h"

void test_laser( void ) {

    if ( mpi::world_root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }
    // Parallel partition
    uint2 partition = make_uint2( 1, 4 );

    uint2 ntiles{ 32, 8 };
    uint2 nx{ 32, 32 };
    float2 box{ 20.48, 25.6 };

    // double dt = 0.01;
    auto dt = 0.9 * zpic::courant( 2, ntiles, nx, box );

    if ( mpi::world_root() )
        std::cout << "Using dt = " << dt << '\n';

    // Create simulation
    Simulation sim( 2, ntiles, nx, box, dt, partition, false );

//    Laser::PlaneWave laser;

    Laser::Gaussian laser;
    laser.W0 = 8.0;
    laser.focus = 20.48;

    // Common laser parameters
    laser.start = 15.36;
    laser.fwhm = 4.0;
    laser.a0 = 1.0;
    laser.omega0 = 10.0;

//    laser.sin_pol = 1; laser.cos_pol = 0;
//    laser.sin_pol = 0; laser.cos_pol = 1;
    laser.sin_pol = sqrt(2.)/2; laser.cos_pol = sqrt(2.)/2;

    laser.add( sim.emf );

    sim.set_moving_window();

    auto diag = [& sim ]( ) {
        sim.emf.save( emf::e, fcomp::r, 1 );
        sim.emf.save( emf::e, fcomp::th, 1 );
        sim.emf.save( emf::e, fcomp::z, 1 );

        sim.emf.save( emf::b, fcomp::r, 1 );
        sim.emf.save( emf::b, fcomp::th, 1 );
        sim.emf.save( emf::b, fcomp::z, 1 );
    };

    diag();

    auto niter = 0;
//    auto niter = 1000;
    for( int i = 0; i <= niter; i ++) {
        if ( sim.get_iter() % 10 == 0 ) diag();
        sim.advance_mov_window();
    }

    if ( mpi::world_root() ) {
        std::cout << ansi::bold
                << "Completed " << __func__ << "()\n"
                << ansi::reset;
    }

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

int main( int argc, char *argv[] ) {

    // Initialize the MPI environment
    mpi::init( & argc, & argv );

    // Initialize the gpu device
    gpu_init( MPI_COMM_WORLD );

    info();

    // test_cylgrid();
    // test_cyl3_cylgrid();
    // test_emf();
    test_laser();

    // Finalize the MPI environment
    mpi::finalize();
}