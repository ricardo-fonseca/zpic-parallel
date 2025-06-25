#ifndef CHARGE_H_
#define CHARGE_H_

#include "zpic.h"
#include "vec3grid.h"
#include "fft.h"

#include "filter.h"

namespace charge {
    namespace bc {
        enum type { none = 0, periodic, reflecting };
    }
    typedef bnd<bc::type> bc_type;
}

class Charge {

    private:

    /// @brief Simulation box size
    float2 box;

    /// @brief cell size
    float2 dx;
    
    /// @brief time step
    float dt;

    /// @brief Boundary condition
    charge::bc_type bc;

    /// @brief Iteration number
    int iter;

    /// @brief FFT plan
    fft::plan  * fft_forward;

    /**
     * @brief Process boundary conditions
     * 
     */
    void process_bc();

    public:

    /// @brief Charge density
    grid<float> * rho;

    /// @brief Charge density k-space
    basic_grid<std::complex<float>> * frho;

    /// @brief Neutralizing background
    grid<float> * neutral;

    /// @brief Filtering parameters
    Filter::Digital *filter;

    Charge( uint2 const ntiles, uint2 const nx, float2 const box, float const dt ) :
        box(box), 
        dx( { box.x / ( nx.x * ntiles.x ), box.y / ( nx.y * ntiles.y ) } ),
        dt(dt)
    {
        // Guard cells (1 below, 2 above)
        bnd<unsigned int> gc;
        gc.x = {1,2};
        gc.y = {1,2};

        rho = new grid<float> ( ntiles, nx, gc );
        rho -> name = "Charge";

        uint2 dims{ nx.x * ntiles.x, nx.y * ntiles.y };
        auto fdims = fft::fdims( dims );
        frho = new basic_grid< std::complex<float> >( fdims );

        fft_forward = new fft::plan( *rho, *frho );

        // Zero initial charge
        // This is only relevant for diagnostics, current should always zeroed before deposition
        rho -> zero();

        // Set default boundary conditions to periodic
        bc = charge::bc_type (charge::bc::periodic);

        // Set default filtering
        filter = new Filter::Lowpass( make_float2( 0.5, 0.5 ) );

        // Reset iteration number
        iter = 0;

        // Default is not to have a neutralizing background
        neutral = nullptr;
    };
    
    ~Charge() {
        delete (filter);
        
        delete (rho);
        delete (frho);
        delete (neutral);

        delete( fft_forward );
    }

    charge::bc_type get_bc( ) { return bc; }

    void set_bc( charge::bc_type new_bc ) {

        // Validate parameters
        if ( (new_bc.x.lower == charge::bc::periodic) || (new_bc.x.upper == charge::bc::periodic) ) {
            if ( new_bc.x.lower != new_bc.x.upper ) {
                std::cerr << "(*error*) Chrarge boundary type mismatch along x.\n";
                std::cerr << "(*error*) When choosing periodic boundaries both lower and upper types must be set to current::bc::periodic.\n";
                exit(1);
            }
        }

        if ( (new_bc.y.lower == charge::bc::periodic) || (new_bc.y.upper == charge::bc::periodic) ) {
            if ( new_bc.y.lower != new_bc.y.upper ) {
                std::cerr << "(*error*) Charge boundary type mismatch along y.\n";
                std::cerr << "(*error*) When choosing periodic boundaries both lower and upper types must be set to emf::bc::periodic.\n";
                exit(1);
            }
        }

        // Store new values
        bc = new_bc;

        std::string bc_name[] = {"none", "periodic", "reflecting"};
        std::cout << "(*info*) Charge boundary conditions\n";
        std::cout << "(*info*) x : [ " << bc_name[ bc.x.lower ] << ", " << bc_name[ bc.x.upper ] << " ]\n";
        std::cout << "(*info*) y : [ " << bc_name[ bc.y.lower ] << ", " << bc_name[ bc.y.upper ] << " ]\n";

        // Set periodic flags on tile grids
        rho -> periodic.x = ( bc.x.lower == charge::bc::periodic );
        rho -> periodic.y = ( bc.y.lower == charge::bc::periodic );
    }

    /**
     * @brief Advance charge
     * 
     * @note This will i) update tile edge values, ii) add neutral background,
     *       iii) Fourier transform and iv) filter 
     */
    void advance();

    /**
     * @brief Zero charge density values
     * 
     */
    void zero() {
        rho -> zero();
    }

    /**
     * @brief Save charge density to disk
     * 
     */
    void save( );
};


#endif