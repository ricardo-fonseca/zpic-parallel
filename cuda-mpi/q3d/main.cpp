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
    laser.W0 = 4.0;
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

    auto niter = 1000;
    for( int i = 0; i <= niter; i ++) {
        if ( sim.get_iter() % 100 == 0 ) diag();
        sim.advance_mov_window();
    }

    if ( mpi::world_root() ) {
        std::cout << ansi::bold
                << "Completed " << __func__ << "()\n"
                << ansi::reset;
    }

}

#include "species.h"

void test_inj( void ) {

    if ( mpi::world_root() ) {
        std::cout << ansi::bold
                << "Running " << __func__ << "()...\n"
                << ansi::reset;
    }

    int nmodes = 2;
    uint2 ntiles{ 4, 4 };
    uint2 nx{ 16, 16 };
    float2 box{ 4, 4 };
    auto dt = 0.99 * zpic::courant( nmodes, ntiles, nx, box );

    // Parallel partition
    uint2 partition = make_uint2( 2, 2 );
    Partition parallel( partition, make_int2(1,0) );

    uint3 ppc{ 2, 2, 8 };
    Species electrons( "electrons", -1.0f, ppc );

//    electrons.set_density(Density::Step(coord::z, 1.0, 2.0));
//    electrons.set_density( Density::Slab(coord::z, 1.0, 2.0, 3.0)); 
//    electrons.set_density( Density::Sphere(1.0, float2{2,2}, 1.0)); 

    electrons.initialize( nmodes, box, ntiles, nx, dt, 0, parallel );
    electrons.save( );
    electrons.save_charge( 0 );

    if ( mpi::world_root() ) {
        std::cout << ansi::bold
                << "Completed " << __func__ << "()\n"
                << ansi::reset;
    }

}

void test_mov( void ) {

    if ( mpi::world_root() ) {
        std::cout << ansi::bold
                << "Running " << __func__ << "()...\n"
                << ansi::reset;
    }

    int nmodes = 2;
    uint2 ntiles{ 4, 4 };
    uint2 nx{ 32, 32 };
    float2 box{ 12.8, 12.8 };

    // Parallel partition
    uint2 partition = make_uint2( 2, 2 );
    Partition parallel( partition, make_int2(1,0) );

    auto dt = 0.99 * zpic::courant( 2, ntiles, nx, box );

    uint3 ppc{ 2, 2, 8 };
    Species electrons( "electrons", -1.0f, ppc );

    electrons.set_density( Density::Sphere(1.0, float2{2,2}, 1.0)); 

    //    electrons.set_udist( UDistribution::Cold( float3{ 0, 0, 1.e6 } ) );
    electrons.set_udist( UDistribution::Cold( float3{ 1e6, 0, 0 } ) );

    electrons.initialize( nmodes, box, ntiles, nx, dt, 0, parallel );

    electrons.save_charge(0);

    int niter = 100;
    for( auto i = 0; i < niter; i ++ ) {
        electrons.advance();
    }

    electrons.save_charge(0);
    electrons.save();

    if ( mpi::world_root() ) {
        std::cout << ansi::bold
                << "Completed " << __func__ << "()\n"
                << ansi::reset;
    }

}

void test_current( void ) {

    if ( mpi::world_root() ) {
        std::cout << ansi::bold
                << "Running " << __func__ << "()...\n"
                << ansi::reset;
    }

    int nmodes = 2;
    uint2 ntiles{ 8, 8 };
    uint2 nx{ 16, 16 };
    float2 box{ 12.8, 12.8 };

    // Parallel partition
    uint2 partition = make_uint2( 2, 2 );

    // auto dt = 0.99 * zpic::courant( ntiles, nx, box );
    
    auto dt = 0.06;

    // Create simulation
    Simulation sim( nmodes, ntiles, nx, box, dt, partition );

    uint3 ppc{ 2, 2, 8 };
    Species electrons( "electrons", -1.0f, ppc );
    electrons.set_density( Density::Sphere(1.0, float2{6.4,6.4}, 3.2)); 
//    electrons.set_density( Density::Sphere(1.0, float2{6.4,0.0}, 3.2)); 

//    electrons.set_udist( UDistribution::Cold( float3{ 1e6, 1e6, 1.e6 } ) );
//    electrons.set_udist( UDistribution::Cold( float3{ 0, 0, 1e6 } ) );

    // ux, uy, uz
    // electrons.set_udist( UDistribution::Cold( float3{ 1e6, 0, 0 } ) );
    // electrons.set_udist( UDistribution::Cold( float3{ 0, 1e6, 0 } ) );
    electrons.set_udist( UDistribution::Cold( float3{ -1e6, 1e6, -1.e6 } ) );

    electrons.set_udist( UDistribution::Cold( float3{ -1e-1, 1e-1, -1e-1 } ) );

    sim.add_species( electrons );

    auto diag = [& sim, & electrons ]( ) {
        // Save mode 0
        electrons.save_charge(0);

        // Save mode 0
        sim.current.save( fcomp::z, 0 );
        sim.current.save( fcomp::r, 0 );
        sim.current.save( fcomp::th, 0 );

        // Save mode 1
        sim.current.save( fcomp::z, 1 );
        sim.current.save( fcomp::r, 1 );
        sim.current.save( fcomp::th, 1 );

        sim.emf.save(emf::e, fcomp::z, 0);
        sim.emf.save(emf::e, fcomp::r, 0);
        sim.emf.save(emf::e, fcomp::th, 0);

        sim.emf.save(emf::b, fcomp::z, 0);
        sim.emf.save(emf::b, fcomp::r, 0);
        sim.emf.save(emf::b, fcomp::th, 0);

        sim.emf.save(emf::e, fcomp::z, 1);
        sim.emf.save(emf::e, fcomp::r, 1);
        sim.emf.save(emf::e, fcomp::th, 1);

        sim.emf.save(emf::b, fcomp::z, 1);
        sim.emf.save(emf::b, fcomp::r, 1);
        sim.emf.save(emf::b, fcomp::th, 1);
    };

    diag();
    for (int i = 0; i < 50 ; i++ ) {
        sim.advance();
        diag();
    }

    if ( mpi::world_root() ) {
        std::cout << ansi::bold
                << "Completed " << __func__ << "()\n"
                << ansi::reset;
    }
}

void test_beam( void ) {

    if ( mpi::world_root() ) {
        std::cout << ansi::bold
                << "Running " << __func__ << "()...\n"
                << ansi::reset;
    }

    int nmodes = 2;
    uint2 dims { 256, 256 };
    float2 box { 25.6, 25.6 };
    uint2 partition { 2, 2 };

    // auto dt = 0.99 * zpic::courant( ntiles, nx, box );

    uint2 nx { 32, 32 };
    uint2 ntiles { dims.x / nx.x, dims.y / nx.y };

    auto dt = zpic::courant( nmodes, dims, box ) * 0.5f;

    if ( mpi::world_root() ) {
        std::cout << "dt     = " << dt << '\n';
        std::cout << "ntiles = " << ntiles << '\n';
    }

    // Create simulation
    Simulation sim( nmodes, ntiles, nx, box, dt, partition );


    uint3 ppc{ 2, 2, 1 };
    Species electrons( "electrons", -1.0f, ppc );

    electrons.set_density( Density::Sphere(1.0, float2{20.0,0.0}, 1.6)); 
    electrons.set_udist( UDistribution::Cold( float3{ 0, 0, 1e6 } ) );

    sim.add_species( electrons );
    sim.set_moving_window();

    electrons.save();

    auto diag = [& sim, & electrons ]( ) {
        // Save mode 0
        electrons.save_charge(0);

        sim.emf.save(emf::e, fcomp::z, 0);
        sim.emf.save(emf::e, fcomp::r, 0);
        sim.emf.save(emf::e, fcomp::th, 0);

        sim.emf.save(emf::b, fcomp::z, 0);
        sim.emf.save(emf::b, fcomp::r, 0);
        sim.emf.save(emf::b, fcomp::th, 0);

        sim.current.save( fcomp::z, 0 );
        sim.current.save( fcomp::r, 0 );
        sim.current.save( fcomp::th, 0 );
    };

    for( int i = 0; i < 1000; i++ ) {
        if ( i % 10 == 0 ) diag();
        sim.advance_mov_window();
    }
    diag();


    if ( mpi::world_root() ) {
        std::cout << ansi::bold
                << "Completed " << __func__ << "()\n"
                << ansi::reset;
    }

}

void test_pwfa( void ) {

    if ( mpi::world_root() ) {
        std::cout << ansi::bold
                << "Running " << __func__ << "()...\n"
                << ansi::reset;
    }

    int nmodes = 1;
    uint2 dims { 256, 256 };
    float2 box { 25.6, 25.6 };
    uint2 partition { 2, 2 };

    // auto dt = 0.99 * zpic::courant( ntiles, nx, box );

    uint2 nx { 32, 32 };
    uint2 ntiles { dims.x / nx.x, dims.y / nx.y };

    auto dt = zpic::courant( nmodes, dims, box ) * 0.5f;

    if ( mpi::world_root() ) {
        std::cout << "dt     = " << dt << '\n';
        std::cout << "ntiles = " << ntiles << '\n';
    }

    // Create simulation
    Simulation sim( nmodes, ntiles, nx, box, dt, partition);

    uint3 ppc{ 2, 2, 1 };

    Species beam( "beam", -1.0f, ppc );
    beam.set_density( Density::Sphere(1.0, float2{23.0,0.0}, 1.6)); 
    beam.set_udist( UDistribution::Cold( float3{ 0, 0, 1e6 } ) );
    sim.add_species( beam );

    Species plasma( "plasma", -1.0f, ppc );
    plasma.set_density( Density::Step( coord::z, 1.0, 40.96 ) ); 
    sim.add_species( plasma );

    sim.set_moving_window();

    auto diag = [& sim, & beam, & plasma ]( ) {
        // Save mode 0
        beam.save_charge(0);
        plasma.save_charge(0);
        plasma.save();

        sim.emf.save(emf::e, fcomp::z, 0);
        sim.emf.save(emf::e, fcomp::r, 0);
        sim.emf.save(emf::e, fcomp::th, 0);

        sim.emf.save(emf::b, fcomp::z, 0);
        sim.emf.save(emf::b, fcomp::r, 0);
        sim.emf.save(emf::b, fcomp::th, 0);

        sim.current.save( fcomp::z, 0 );
        sim.current.save( fcomp::r, 0 );
        sim.current.save( fcomp::th, 0 );
    };

    while ( sim.get_t() <= 61.5 ) {
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

#include "timer.h"

void test_lwfa() {

    if ( mpi::world_root() ) {
        std::cout << ansi::bold;
        std::cout << "Running " << __func__ << "()...";
        std::cout << ansi::reset << std::endl;
    }

    int nmodes = 2;
    uint2 dims { 1024, 128 };
    float2 box { 20.48, 12.8 };
    uint2 partition { 2, 2 };

    uint2 nx { 32, 32 };
    uint2 ntiles { dims.x / nx.x, dims.y / nx.y };

    auto dt = zpic::courant( nmodes, dims, box ) * 0.9f;

    // Create simulation
    Simulation sim( nmodes, ntiles, nx, box, dt, partition );

    // Add electrons
    Species electrons("electrons", -1.0f, make_uint3( 2, 2, 8 ));
    electrons.set_density( Density::Step( coord::z, 1.0, 20.48 ) );
//    electrons.set_density( Density::Step( coord::z, 1.0, 10.24 ) );
    sim.add_species( electrons );
    
    // Add Laser
    Laser::Gaussian laser;
    laser.start   = 20.0;
    laser.fwhm    = 2.0;
    laser.a0      = 2.0;
    laser.omega0  = 10.0;
    laser.W0      = 4.0;
    laser.focus   = 20.48;
    laser.sin_pol = 1;

    laser.add( sim.emf );

    // Set moving window and current filtering
    sim.set_moving_window();
    sim.current.set_filter( Filter::Compensated( coord::z, 4 ) );

    auto diag = [& sim, & electrons ]( ) {
        sim.emf.save(emf::e, fcomp::z, 0);
        sim.emf.save(emf::e, fcomp::r, 0);
        sim.emf.save(emf::e, fcomp::th, 0);

        sim.emf.save(emf::e, fcomp::z, 1);
        sim.emf.save(emf::e, fcomp::r, 1);
        sim.emf.save(emf::e, fcomp::th, 1);

        sim.current.save( fcomp::z, 0);
        sim.current.save( fcomp::r, 0);
        sim.current.save( fcomp::th, 0);

        sim.current.save( fcomp::z, 1 );
        sim.current.save( fcomp::r, 1 );
        sim.current.save( fcomp::th, 1 );

    
        electrons.save_charge( 0 );
        electrons.save_charge( 1 );
    };

    if ( mpi::world_root() )
        std::cout << "Starting simulation, dt = " << dt << '\n';

    Timer timer;
    timer.start();

    while( sim.get_t() <= 1.05 * box.x ) {
        if ( sim.get_iter() % 100 == 0 ) {
            if ( mpi::world_root() )
                std::cout << "i = " << sim.get_iter() << '\n';
            diag();
        }
        sim.advance_mov_window();
    }

    timer.stop();

    if ( mpi::world_root() ) {
        std::cout << "Simulation run up to t = " << sim.get_t()
                << " in " << timer.elapsed(timer::s) << " s\n";

        std::cout << ansi::bold;
        std::cout << "Done!\n";
        std::cout << ansi::reset; 
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
    // test_laser();
    
    // test_inj();
    // test_mov();
    // test_current();

    // test_beam();
    // test_pwfa();
    test_lwfa();

    // Finalize the MPI environment
    mpi::finalize();
}