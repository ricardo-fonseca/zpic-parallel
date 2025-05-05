
// For getopt
#include <unistd.h>

#include <iostream>
#include <fstream>

/**
 * MPI support 
 */
#include "parallel.h"

#include "utils.h"

#include "grid.h"

__global__
void test_grid_kernel( 
    float * const __restrict__ d_buffer,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx )
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const size_t tile_off = tile_id * roundup4( ext_nx.x * ext_nx.y );

    auto * const __restrict__ tile_data = & d_buffer[ tile_off ];

    for( auto idx = block_thread_rank(); idx < nx.x * nx.y; idx += block_num_threads() ) {
        const auto iy = idx / nx.x; 
        const auto ix = idx % nx.x;
        tile_data[iy * ext_nx.x + ix] = tile_id;
    }
}

void test_grid( void ) {
    
    if ( mpi::world_root() ) {
        std::cout << ansi::bold;
        std::cout << "Running test_grid()\n";
        std::cout << ansi::reset;
        std::cout << "Declaring grid<float> data...\n";
    }

    // Parallel partition
    uint2 partition = make_uint2( 2, 2 );

    // Global number of tiles
    const uint2 ntiles = { 8, 8 };
    const uint2 nx = { 32, 32 };

    bnd<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {1,2};

    Partition parallel( partition );

    grid<float> data( ntiles, nx, gc, parallel );

    data.zero( );

    if ( mpi::world_root() )
        std::cout << "Setting values...\n";

    dim3 block( 64 );
    dim3 grid( ntiles.x, ntiles.y );

    data.set( 32.0 * ( 1 + parallel.get_rank() ) );

    test_grid_kernel <<< block, grid >>> ( data.d_buffer + data.offset, ntiles, nx, data.ext_nx );

    data.add_from_gc();
    data.copy_to_gc();

/*
    for( auto i = 0; i < 5; i++)
       data.x_shift_left( 1 );

    data.kernel3_x( 1., 2., 1. );
    data.kernel3_y( 1., 2., 1. );
*/

    if ( mpi::world_root() )
        std::cout << "Saving data...\n";

    data.save( "cuda" );

    if ( mpi::world_root() ) {
        std::cout << ansi::bold;
        std::cout << "Done!\n";
        std::cout << ansi::reset;
    }
        
}

#if 0

#include "vec_types.h"
#include "vec3grid.h"

__global__
void test_vec3grid_kernel( 
    float3 * const __restrict__ d_buffer,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx )
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const size_t tile_off = tile_id * roundup4( ext_nx.x * ext_nx.y );

    auto * const __restrict__ tile_data = & d_buffer[ tile_off ];

    for( auto idx = block_thread_rank(); idx < nx.x * nx.y; idx += block_num_threads() ) {
        const auto iy = idx / nx.x; 
        const auto ix = idx % nx.x;
        tile_data[iy * ext_nx.x + ix] = make_float3( 1 + tile_id, 2 + tile_id, 3 + tile_id );
    }
}

void test_vec3grid( ) {
    std::cout << "Declaring vec3grid<float3> data...\n";
    
    const uint2 ntiles = { 8, 8 };
    const uint2 nx = { 16,16 };
    
    bnd<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {1,2};

    vec3grid<float3> data( ntiles, nx, gc );

    // Set zero
    // data.zero( );

    // Set constant
    // data.set( float3{1.0, 2.0, 3.0} );

    // Set different value per tile
    dim3 block( 64 );
    dim3 grid( ntiles.x, ntiles.y );
    test_vec3grid_kernel <<< block, grid >>> ( & data.d_buffer[ data.offset ], ntiles, nx, data.ext_nx );


    data.copy_to_gc( );

    data.add_from_gc( );

    data.copy_to_gc( );
    for( int i = 0; i < 5; i++ ) {
        data.x_shift_left( 1 );
    }

    data.kernel3_x( 1., 2., 1. );
    data.kernel3_y( 1., 2., 1. );

    std::cout << "Saving data...\n";

    data.save( fcomp::x, "cuda" );
    data.save( fcomp::y, "cuda" );
    data.save( fcomp::z, "cuda" );

    std::cout << "Done!\n";
}

#include "emf.h"
#include "laser.h"

#include "timer.h"

void test_laser( ) {

    uint2 ntiles = { 64, 16 };
    uint2 nx = { 16, 16 };

    float2 box = { 20.48, 25.6 };
    double dt = 0.014;

    EMF emf( ntiles, nx, box, dt );

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

#include "simulation.h"

void test_inj( ) {

    std::cout << "starting " << __func__ << "...\n";

    uint2 ntiles{ 2, 2 };
    uint2 nx{ 64, 64 };

    float2 box{ 12.8, 12.8 };

    auto dt = 0.99 * zpic::courant( ntiles, nx, box );

    uint2 ppc{ 8, 8 };
    Species electrons( "electrons", -1.0f, ppc );

    // electrons.set_density(Density::Step(coord::x, 1.0, 5.0));
    // electrons.set_density(Density::Slab(coord::y, 1.0, 5.0, 7.0));
    electrons.set_density( Density::Sphere( 1.0, float2{5.0, 7.0}, 2.0 ) );
    electrons.set_udist( UDistribution::Thermal( float3{ 0.1, 0.2, 0.3 }, float3{1,0,0} ) );

    electrons.initialize( box, ntiles, nx, dt, 0 );

    electrons.save_charge();
    electrons.save();
    electrons.save_phasespace(
        phasespace::ux, float2{-1, 3}, 256,
        phasespace::uz, float2{-1, 1}, 128
    );

    std::cout << __func__ << " complete.\n";
}

void test_mov( ) {

    std::cout << "Starting " << __func__ << "...\n";

    uint2 ntiles{ 4, 4 };
    uint2 nx{ 32, 32 };

    float2 box{ 12.8, 12.8 };

    auto dt = 0.99 * zpic::courant( ntiles, nx, box );

    uint2 ppc{ 8, 8 };
    Species electrons( "electrons", -1.0f, ppc );

    electrons.set_density( Density::Sphere( 1.0, float2{2.1, 2.1}, 2.0 ) );
    electrons.set_udist( UDistribution::Cold( float3{ -1, -2, -3 } ) );

    electrons.initialize( box, ntiles, nx, dt, 0 );

    electrons.save_charge();

    int niter = 200;
    for( auto i = 0; i < niter; i ++ ) {
        electrons.advance();
    }

    electrons.save_charge();
    electrons.save();

    std::cout << __func__ << " complete.\n";
}

void test_current( ) {

    std::cout << "Starting " << __func__ << "...\n";

    uint2 ntiles{ 4, 4 };
    uint2 nx{ 32, 32 };

    float2 box{ 12.8, 12.8 };

    auto dt = 0.99 * zpic::courant( ntiles, nx, box );

    uint2 ppc{ 8, 8 };
    Species electrons( "electrons", -1.0f, ppc );

    electrons.set_density( Density::Sphere( 1.0, float2{5.0, 7.0}, 2.0 ) );
    electrons.set_udist( UDistribution::Cold( float3{ 1, 2, 3 } ) );

    electrons.initialize( box, ntiles, nx, dt, 0 );

    Current current( ntiles, nx, box, dt );

    electrons.save_charge();

    electrons.advance( current );
    current.advance( );
    
    current.save( fcomp::x );
    current.save( fcomp::y );
    current.save( fcomp::z );

    std::cout << __func__ << " complete.\n";
}


void test_mov_sim( ) {

    std::cout << "Starting " << __func__ << "...\n";

    uint2 ntiles{ 4, 4 };
    uint2 nx{ 32, 32 };
    float2 box{ 12.8, 12.8 };

    auto dt = 0.99 * zpic::courant( ntiles, nx, box );

    Simulation sim( ntiles, nx, box, dt );


    uint2 ppc{ 8, 8 };
    Species electrons( "electrons", -1.0f, ppc );
    electrons.set_density( Density::Sphere( 1.0, float2{2.1, 2.1}, 2.0 ) );
    electrons.set_udist( UDistribution::Cold( float3{ -1, -2, -3 } ) );

    sim.add_species( electrons );

    electrons.save_charge();

    int niter = 200;
    for( auto i = 0; i < niter; i ++ ) {
        sim.advance();
    }

    electrons.save_charge();
    electrons.save();

    std::cout << __func__ << " complete.\n";
}

void test_weibel( ) {
    uint2 gnx{ 128, 128 };
    uint2 ntiles{ 2, 2 };

    uint2 nx{ gnx.x/ntiles.x, gnx.y/ntiles.y };
    float2 box = make_float2( gnx.x/10.0, gnx.y/10.0 );
    float dt = 0.07;

    Simulation sim( ntiles, nx, box, dt );

    uint2 ppc{ 4, 4 };

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

    auto perf = sim.get_nmove() / timer.elapsed(timer::s) / 1.e9;

    std::cerr << "Elapsed time: " << timer.elapsed(timer::s) << " s"
              << ", Performance: " << perf << " GPart/s\n";
}


void test_weibel_large( )
{
                            
    // Create simulation box
    uint2 ntiles{64, 64};
    uint2 nx{32, 32};
    uint2 ppc{8, 8};
                                                                                                    
    uint64_t vol = static_cast<uint64_t>(nx.x * nx.y) *  static_cast<uint64_t>(ntiles.x * ntiles.y);
                                             
    std::cout << "** Large Weibel test **\n";                                   
    std::cout << " # tiles          : " << ntiles.x << ", " << ntiles.y << "\n";
    std::cout << " tile size        : " << nx.x << ", " << nx.y << "\n";                      
    std::cout << " global size      : " << nx.x * ntiles.x << ", " << nx.y * ntiles.y << "\n";
    std::cout << " # part / species : " << vol * (ppc.x * ppc.y) / (1048576.0) << " M \n";
                                                                  
    float2 box = {nx.x * ntiles.x * 0.1f, nx.y * ntiles.y * 0.1f};
                    
    float dt = 0.07;
                                        
    Simulation sim( ntiles, nx, box, dt );
                            
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
                     
    // Run simulation    
    int const imax = 500;
                                                          
    printf("Running large Weibel test up to n = %d...\n", imax);
                
    Timer timer;
                  
    timer.start();
                                 
    while (sim.get_iter() < imax)
    {                 
        sim.advance();
    }
                 
    timer.stop();
                                                              
    std::cout << "Simulation complete at i = " << sim.get_iter() << '\n';
                      
    sim.energy_info();
                                    
    sim.emf.save(emf::b, fcomp::x);
    sim.emf.save(emf::b, fcomp::y);
    sim.emf.save(emf::b, fcomp::z);
                                                                  
    auto perf = sim.get_nmove() / timer.elapsed(timer::s) / 1.e9;

    // std::cerr << "Elapsed time: " << timer.elapsed(timer::s) << " s"
    //          << ", Performance: " << perf << " GPart/s\n";

    std::cout << "[benchmark] " << perf << " GPart/s\n";
}

void test_warm( ) 
{

    Simulation sim(         
        uint2{32, 32},        // ntiles
        uint2{32, 32},        // nx
        float2{102.4, 102.4}, // box
        0.07                  // dt
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

void benchmark_weibel()
{

    auto bench_weibel_sim = [ ]( std::ostream &output, uint2 ppc, uint2 nx, uint2 ntiles ) {
        Simulation sim(
            ntiles,                                                      // ntiles
            nx,                                                          // nx
            make_float2(ntiles.x * nx.x * 0.1f, ntiles.y * nx.y * 0.1f), // box
            0.07                                                         // dt
        );

        float3 ufl = float3{0., 0., 0.6};
        float3 uth = float3{0.1, 0.1, 0.1};

        Species electrons("electrons", -1.0f, ppc);
        electrons.set_udist( UDistribution::Thermal (uth, ufl) );

        Species positrons("positrons", +1.0f, ppc);
        ufl.z = - ufl.z;
        positrons.set_udist( UDistribution::Thermal (uth, ufl) );

        sim.add_species(electrons);
        sim.add_species(positrons);

        float const imax = 500;

        Timer timer;

        timer.start();

        while (sim.get_iter() < imax)
        {
            sim.advance();
        }

        timer.stop();

        output         << ppc.x << ", " << ppc.y 
               << ", " << nx.x<< ", " << nx.y
               << ", " << ntiles.x << ", " << ntiles.y
               << ", " << timer.elapsed(timer::s)
               << ", " << sim.get_nmove() / timer.elapsed(timer::s) / 1.e9
               << ", " << std::endl;
    };

    std::vector<uint2> ppc_list{
        {1, 1}, {2, 1}, {2, 2}, {4, 2}, {4, 4}, {8, 4}, {8, 8}, {16, 8}, {16, 16}, {32, 16}, {32, 32}};

    std::vector<uint2> ntiles_list{
        {8, 8}, {16, 8}, {16, 16}, {32, 16}, {32, 32}, {64, 32}, {64, 64}, {128, 64}, {128, 128}, {256, 128}, {256, 256}, {512, 256}, {512, 512}, {1024, 512}, {1024, 1024}};

    std::vector<uint2> nx_list{
        {4, 4}, {8, 4}, {8, 8}, {16, 8}, {16, 16}, {32, 16}, {32, 32}, {64, 32}, {64, 64}, { 80, 64 }, { 80, 80 } };

/* 
    std::vector<uint2> ppc_list{ {4, 4} };
    std::vector<uint2> ntiles_list{ { 8, 8 } };
    std::vector<uint2> nx_list{ {80, 80} };
 */

    std::ofstream output;
    output.open("benchmark_weibel.csv");
    output << "ppc.x, ppc.y, nx.x, nx.y, ntiles.x, ntiles.y, time, perf" << std::endl;

    int ntests = ppc_list.size() *
                nx_list.size() *
                ntiles_list.size();

    int i = 0;
    for (uint2 ppc : ppc_list)
        for (uint2 nx : nx_list)
            for (uint2 ntiles : ntiles_list)
            {
                // The casts avoid integer overflows (because 64^2 * 1024^2 = 2^32)
                int64_t vol = static_cast<int64_t>(nx.x * nx.y) * 
                              static_cast<int64_t>(ntiles.x * ntiles.y);
                int64_t npart = vol * static_cast<int64_t>(ppc.x * ppc.y);

                printf("[%3d/%3d] %d, %d, %d, %d, %d, %d (%ld)\n", i++, ntests,
                       ppc.x, ppc.y, nx.x, nx.y, ntiles.x, ntiles.y, npart);

                // Limit the test to 512 M part / species
                if ( npart < 512 * 1024 * 1024 )
                {
                    deviceReset();
                    bench_weibel_sim(output, ppc, nx, ntiles);
                }
            }

    output.close();
}

#endif

/**
 * @brief Print information about the environment
 * 
 */
void info( void ) {

    if ( mpi::world_root() ) {

        std::cout << "MPI running on " << mpi::world_size() << " processes\n";

        std::cout << "GPU devices on rank 0:\n";
        print_gpu_info();

    }
}

void cli_help( char * argv0 ) {
    std::cerr << "Usage: " << argv0 << " [-h] [-s] [-t name] [-n parameter]\n";

    std::cerr << '\n';
    std::cerr << "Options:\n";
    std::cerr << "  -h                  Display this message and exit\n";
    std::cerr << "  -s                  Silence information about MPI/CUDA device\n";
    std::cerr << "  -t <name>           Name of the test to run. Defaults to 'weibel'\n";
    std::cerr << "  -p <parameters>     Test parameters (string). Purpose will depend on the \n";
    std::cerr << "                      test chosen. Defaults to '2,2,16,16'\n";
    std::cerr << '\n';
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

    // Reset current device
    deviceReset();
}

int main( int argc, char *argv[] ) {

    // Initialize the MPI environment
    mpi::init( & argc, & argv );

    // Initialize the gpu device
    gpu_init( MPI_COMM_WORLD );

    // Process command line arguments
    int opt;
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

    test_grid( );
    // test_laser( );


    // test_vec3grid( );
    // test_inj( );

    // test_mov( );
    // test_current( );

    // test_mov_sim( );

    // test_weibel( );

    // test_weibel_large( );

    // test_warm( );

    // benchmark_weibel();

    // Finalize the MPI environment
    mpi::finalize();
}
