#include <iostream>
#include "complex.h"

#include "zpic.h"

#include "bnd.h"

#include "vec_types.h"
#include "complex.h"
#include "cylmodes.h"
#include "cyl3modes.h"

#include "emf.h"
#include "laser.h"

void test_cylgrid( void ) {

    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;

    uint2 ntiles{ 8, 8 };
    uint2 nx{ 16, 16 };
    bnd<unsigned int> gc{ 0 };

    CylGrid<float> field( 4, ntiles, nx, gc );

    std::cout << "field: " << field << '\n';

    std::cout << "mode0: " << field.mode0() << '\n';
    std::cout << "mode1: " << field.mode(1) << '\n';
    std::cout << "mode2: " << field.mode(2) << '\n';
    std::cout << "mode3: " << field.mode(3) << '\n';

    std::cout << ansi::bold;
    std::cout << "Done!\n";
    std::cout << ansi::reset;

}

void test_cyl3_cylgrid( void ) {

    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;

    uint2 ntiles{ 8, 8 };
    uint2 nx{ 16, 16 };
    bnd<unsigned int> gc{ 0 };

    Cyl3CylGrid<float> field( 4, ntiles, nx, gc );

    std::cout << "field: " << field << '\n';

    std::cout << "mode0: " << field.mode0() << '\n';
    std::cout << "mode1: " << field.mode(1) << '\n';
    std::cout << "mode2: " << field.mode(2) << '\n';
    std::cout << "mode3: " << field.mode(3) << '\n';

    std::cout << ansi::bold;
    std::cout << "Done!\n";
    std::cout << ansi::reset;

}

void test_pvec3_cylgrid( void ) {

    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;

    uint2 ntiles{ 8, 8 };
    uint2 nx{ 16, 16 };
    bnd<unsigned int> gc{ 0 };

    auto A = new Cyl3CylGrid<float>( 4, ntiles, nx, gc );
    auto B = new Cyl3CylGrid<float>( 4, ntiles, nx, gc );

    A -> set_name( "field A" );
    B -> set_name( "field B" );

    A -> zero();
    B -> zero();

    std::cout << "A: " << * A << '\n';

    std::cout << "mode0: " << A->mode0() << '\n';
    std::cout << "mode1: " << B->mode(1) << '\n';
    std::cout << "mode2: " << A->mode(2) << '\n';
    std::cout << "mode3: " << B->mode(3) << '\n';

    delete( A );
    delete( B );

    std::cout << ansi::bold;
    std::cout << "Done!\n";
    std::cout << ansi::reset;

}


void test_emf( void ) {

    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;

    uint2 ntiles{ 8, 8 };
    uint2 nx{ 16, 16 };

    float2 box{ 12.8, 12.8 };
    double dt{ 0.04 };

    EMF emf( 4, ntiles, nx, box, dt );

    std::cout << "field: " << emf << ", nmodes: " << emf.nmodes << '\n';

    std::cout << "mode0: " << emf.E->mode0() << '\n';
    std::cout << "mode1: " << emf.B->mode(1) << '\n';
    std::cout << "mode2: " << emf.E->mode(2) << '\n';
    std::cout << "mode3: " << emf.B->mode(3) << '\n';

    std::cout << ansi::bold
              << "Completed " << __func__ << "()\n"
              << ansi::reset;
}

void test_laser( void ) {

    std::cout << ansi::bold
              << "Running " << __func__ << "()...\n"
              << ansi::reset;

    uint2 ntiles{ 32, 8 };
    uint2 nx{ 32, 32 };
    float2 box{ 20.48, 25.6 };

    // 1024 * 256

    // double dt = 0.01;
    auto dt = 0.9 * zpic::courant( 2, ntiles, nx, box );

    std::cout << "Using dt = " << dt << '\n';

    EMF emf( 2, ntiles, nx, box, dt );

//    Laser::PlaneWave laser;

    Laser::Gaussian laser;
    laser.W0 = 2.0;
    laser.focus = 20.48;

    // Common laser parameters
    laser.start = 15.36;
    laser.fwhm = 4.0;
    laser.a0 = 1.0;
    laser.omega0 = 10.0;

//    laser.sin_pol = 1; laser.cos_pol = 0;
//    laser.sin_pol = 0; laser.cos_pol = 1;
    laser.sin_pol = sqrt(2.)/2; laser.cos_pol = sqrt(2.)/2;

    laser.add( emf );

    emf.set_moving_window();

    auto diag = [& emf ]( ) {
        emf.save( emf::e, fcomp::r, 1 );
        emf.save( emf::e, fcomp::θ, 1 );
        emf.save( emf::e, fcomp::z, 1 );

        emf.save( emf::b, fcomp::r, 1 );
        emf.save( emf::b, fcomp::θ, 1 );
        emf.save( emf::b, fcomp::z, 1 );
    };

    diag();

    auto niter = 1000;
    for( int i = 0; i <= niter; i ++) {
        if ( emf.get_iter() % 10 == 0 ) diag();
        emf.advance();
    }
 
    std::cout << ansi::bold
              << "Completed " << __func__ << "()\n"
              << ansi::reset;

}

#if 0

/**
 * @brief Print command line help
 * 
 * @param argv0     Program name (from argv[0])
 */
void cli_help( char * argv0 ) {
    std::cerr << "Usage: " << argv0 << " [-h] [-s] [-t name] [-n parameter]\n";

    std::cerr << '\n';
    std::cerr << "Options:\n";
    std::cerr << "  -h                  Display this message and exit\n";
    std::cerr << "  -s                  Silence information about MPI/CUDA device\n";
    std::cerr << "  -t <name>           Name of the test to run. Defaults to 'weibel'\n";
    std::cerr << "  -p <parameters>     Test parameters (string). Purpose will depend on the \n";
    std::cerr << "                      test chosen. Defaults to '8,8,16,16'\n";
    std::cerr << '\n';
}


int main( int argc, char *argv[] ) {

    // Initialize the gpu device
    gpu_init();

    // Process command line arguments
    int opt;
    int silent = 0;
    std::string test = "weibel";
    std::string param = "8,8,16,16";
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
            exit(0);
        default:
            cli_help( argv[0] );    
            std::exit(1);
        }
    }
    
    // Print information about the environment
    if ( ! silent ) info();

    // test_grid( );
    // test_vec3grid( );
    // test_laser( );
    // test_inj( );
    // test_mov( );
    // test_current( );
    // test_mov_sim( );
    // test_weibel( );
    // test_weibel_large( );
    // test_warm( );
    // benchmark_weibel();

    if ( test == "weibel" ) {
        test_weibel( param );
    } else {
        std::cerr << "Unknonw test '" << test << "', aborting...\n";
        std::exit(1);
    }

    deviceReset();
}

#endif

/**
 * @brief Initialize GPU device
 *
 */
void gpu_init() {
    deviceReset();
}

int main( int argc, char *argv[] ) {

    std::cout << ansi::bold 
              << "CUDA quasi-3D tests\n"
              << ansi::reset ;

    zpic::sys_info();

    // Initialize the gpu device
    gpu_init();

    // test_cylgrid();
    // test_cyl3_cylgrid();
    // test_pvec3_cylgrid();

    // test_emf();
    test_laser();

   // test_inj();
   // test_mov();
   // test_current();

   // test_beam();
   // test_pwfa();
   // test_lwfa();

    deviceReset();
}