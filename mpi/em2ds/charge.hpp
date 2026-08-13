#pragma once

#include "grid/tiled.hpp"
#include "grid/fft.hpp"

#include "filter.hpp"

namespace charge {
    enum field  { rho, frho };

    namespace bc {
        enum type { none = 0, periodic, reflecting };
    }
    typedef bounds_2d<bc::type> bc_type;
}

class Charge {

    private:

    /// @brief Simulation box size
    float2 box;

    /// @brief cell size
    // float2 dx;
    
    /// @brief time step
    float dt;

    /// @brief Boundary condition
    charge::bc_type bc;

    /// @brief Iteration number
    int iter;

    /// @brief FFT plan
    grid::fft::r2c_plan  * fft_forward;

    /**
     * @brief Process boundary conditions
     * 
     */
    void process_bc();

    public:

    /// @brief Charge density
    grid::tiled<float> * rho;

    /// @brief Charge density k-space
    grid::flat<std::complex<float>> * frho;

    /// @brief Neutralizing background
    grid::tiled<float> * neutral;

    /// @brief Filtering parameters
    filter::digital *filter;

    /**
     * @brief Construct a new Charge object
     * 
     * @param global_ntiles     Global number of tiles
     * @param tile_dims         Individual tile size
     * @param box               Global simulation box size
     * @param dt                Time step
     * @param parallel          Parallel partition 
     */
    Charge( uint2 const global_ntiles, uint2 const tile_dims, float2 const box, float const dt, mpi::cart2d & parallel ):
        box(box), 
        // dx( { box.x / ( nx.x * ntiles.x ), box.y / ( nx.y * ntiles.y ) } ),
        dt(dt)
    {
        // Guard cells (1 below, 2 above)
        bounds_2d<unsigned int> gc;
        gc.x = {1,2};
        gc.y = {1,2};

        rho = new grid::tiled<float> ( global_ntiles, tile_dims, gc, parallel );
        rho -> name = "Charge";

        fft_forward = new grid::fft::r2c_plan( *rho );
        frho = grid::fft::new_complex_grid( *rho );

        // Zero initial charge
        // This is only relevant for diagnostics, current should always zeroed before deposition
        rho -> zero();

        // Set default boundary conditions to periodic
        bc = charge::bc_type (charge::bc::periodic);

        // Set default filtering
        filter = new filter::lowpass( make_float2( 0.5, 0.5 ) );

        // Reset iteration number
        iter = 0;

        // Default is not to have a neutralizing background
        neutral = nullptr;
    };
    
    /**
     * @brief Destroy the Charge object
     * 
     */
    ~Charge() {
        delete (filter);
        
        delete (rho);
        delete (frho);
        delete (neutral);

        delete( fft_forward );
    }

    /**
     * @brief Get the type of boundary conditions
     * 
     * @return current::bc_type
     */
    charge::bc_type get_bc( ) { return bc; }

    /**
     * @brief Set the boundary conditions
     * 
     * @param new_bc    New boundary condition values
     */
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

        if ( rho -> get_part().periodic.x && new_bc.x.lower != charge::bc::periodic ) {
            std::cerr << "(*error*) Only periodic x boundaries are supported with periodic x parallel partitions.\n";
            mpi::abort(1);
        }

        if ( rho -> get_part().periodic.y && new_bc.y.lower != charge::bc::periodic ) {
            std::cerr << "(*error*) Only periodic y boundaries are supported with periodic y parallel partitions.\n";
            mpi::abort(1);
        }

        // Store new values
        bc = new_bc;

        if ( mpi::root() ) {
            std::string bc_name[] = {"none", "periodic", "reflecting"};
            std::cout << "(*info*) Charge boundary conditions\n";
            std::cout << "(*info*) x : [ " << bc_name[ bc.x.lower ] << ", " << bc_name[ bc.x.upper ] << " ]\n";
            std::cout << "(*info*) y : [ " << bc_name[ bc.y.lower ] << ", " << bc_name[ bc.y.upper ] << " ]\n";
        }
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
     * @param field     Which field to save (rho or frho)
     */
    void save( const charge::field field );
};
