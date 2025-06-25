#ifndef CURRENT_H_
#define CURRENT_H_

#include "zpic.h"
#include "vec3grid.h"
#include "fft.h"

#include "filter.h"

namespace current {
    namespace bc {
        enum type { none = 0, periodic, reflecting };
    }
    typedef bnd<bc::type> bc_type;
}

class Current {

    private:

    /// @brief Simulation box size
    float2 box;

    /// @brief cell size
    float2 dx;
    
    /// @brief time step
    float dt;

    /// @brief Boundary condition
    current::bc_type bc;

    /// @brief Iteration number
    int iter;

    /// @brief FFT plan
    fft::plan * fft_forward;

    /**
     * @brief Process boundary conditions
     * 
     */
    void process_bc();

    public:

    /// @brief Current density
    vec3grid<float3> * J;

    /// @brief Charge density k-space
    basic_grid3<std::complex<float>> * fJ;

    /// @brief Filtering parameters
    Filter::Digital *filter;

    /**
     * @brief Construct a new Current object
     * 
     * @param ntiles 
     * @param nx 
     * @param box 
     * @param dt 
     */
    Current( uint2 const ntiles, uint2 const nx, float2 const box, float const dt ):
        box(box), 
        dx( { box.x / ( nx.x * ntiles.x ), box.y / ( nx.y * ntiles.y ) } ),
        dt(dt)
    {
        // Guard cells (1 below, 2 above)
        bnd<unsigned int> gc;
        gc.x = {1,2};
        gc.y = {1,2};

        J = new vec3grid<float3> ( ntiles, nx, gc );
        J -> name = "Current";

        uint2 dims{ nx.x * ntiles.x, nx.y * ntiles.y };
        fft_forward = new fft::plan( dims, fft::type::r2c_v3 );

        fJ = new basic_grid3< std::complex<float> >( fft::fdims( dims ) );

        // Zero initial charge
        // This is only relevant for diagnostics, current is always zeroed before deposition
        J -> zero();

        // Set default boundary conditions to periodic
        bc = current::bc_type (current::bc::periodic);

        // Set default filtering
        filter = new Filter::Lowpass( make_float2( 0.5, 0.5 ) );

        // Reset iteration number
        iter = 0;
    };
    
    ~Current() {
        delete (filter);
        
        delete (J);
        delete (fJ);

        delete( fft_forward );
    }

    current::bc_type get_bc( ) { return bc; }

    void set_bc( current::bc_type new_bc ) {

        // Validate parameters
        if ( (new_bc.x.lower == current::bc::periodic) || (new_bc.x.upper == current::bc::periodic) ) {
            if ( new_bc.x.lower != new_bc.x.upper ) {
                std::cerr << "(*error*) Current boundary type mismatch along x.\n";
                std::cerr << "(*error*) When choosing periodic boundaries both lower and upper types must be set to current::bc::periodic.\n";
                exit(1);
            }
        }

        if ( (new_bc.y.lower == current::bc::periodic) || (new_bc.y.upper == current::bc::periodic) ) {
            if ( new_bc.y.lower != new_bc.y.upper ) {
                std::cerr << "(*error*) Current boundary type mismatch along y.\n";
                std::cerr << "(*error*) When choosing periodic boundaries both lower and upper types must be set to emf::bc::periodic.\n";
                exit(1);
            }
        }

        // Store new values
        bc = new_bc;

        std::string bc_name[] = {"none", "periodic", "reflecting"};
        std::cout << "(*info*) Current boundary conditions\n";
        std::cout << "(*info*) x : [ " << bc_name[ bc.x.lower ] << ", " << bc_name[ bc.x.upper ] << " ]\n";
        std::cout << "(*info*) y : [ " << bc_name[ bc.y.lower ] << ", " << bc_name[ bc.y.upper ] << " ]\n";

        // Set periodic flags on tile grids
        J->periodic.x = ( bc.x.lower == current::bc::periodic );
        J->periodic.y = ( bc.y.lower == current::bc::periodic );
    }

    /**
     * @brief Advances electric current density 1 time step
     * 
     * The routine will:
     * 1. Update the guard cells
     * 2. Get the Fourier transform of the current
     * 3. Apply spectral filtering
     * 
     */
    void advance();

    /**
     * @brief Zero electric current values
     * 
     */
    void zero() {
        J -> zero();
    }

    void save( fcomp::cart const jc );
};


#endif