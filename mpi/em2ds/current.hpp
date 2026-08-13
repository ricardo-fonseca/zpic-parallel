#pragma once


#include "grid/vec3_tiled.hpp"
#include "grid/fft.hpp"

#include "filter.hpp"

namespace current {
    enum field  { j, fj };

    namespace bc {
        enum type { none = 0, periodic, reflecting };
    }
    typedef bounds_2d<bc::type> bc_type;
}

class Current {

    private:

    /// @brief Simulation box size
    const float2 box;

    /// @brief cell size
    // const float2 dx;
    
    /// @brief time step
    float dt;

    /// @brief Boundary condition
    current::bc_type bc;

    /// @brief Iteration number
    int iter;

    /// @brief FFT plan
    grid::fft::r2c_plan * fft_forward;

    /**
     * @brief Process boundary conditions
     * 
     */
    void process_bc();

    public:

    /// @brief Current density
    grid::vec3_tiled<float> * J = nullptr;

    /// @brief Charge density k-space
    grid::flat3<std::complex<float>> * fJ = nullptr;

    /// @brief Filtering parameters
    filter::digital *filter = nullptr;

    /**
     * @brief Construct a new Current object
     * 
     * @param global_ntiles     Global number of tiles
     * @param tile_dims         Individual tile size
     * @param box               Global simulation box size
     * @param dt                Time step
     * @param parallel          Parallel partition 
     */
    Current( uint2 const global_ntiles, uint2 const tile_dims, float2 const box, float const dt, Partition & parallel ):
        box(box), 
        // dx( make_float2( box.x / ( nx.x * global_ntiles.x ), box.y / ( nx.y * global_ntiles.y ) ) ),
        dt(dt)
    {
        // Guard cells (1 below, 2 above)
        bounds_2d<unsigned int> gc;
        gc.x = {1,2};
        gc.y = {1,2};

        J = new grid::vec3_tiled<float> ( global_ntiles, tile_dims, gc, parallel );
        J -> name = "Current density";

        fft_forward = new grid::fft::r2c_plan( *J );

        fJ = grid::fft::new_complex_grid(*J);

        // Zero initial charge
        // This is only relevant for diagnostics, current is always zeroed before deposition
        J -> zero();

        // Set default boundary conditions to be none or periodic
        // according to parallel partition
        bc = current::bc_type (current::bc::periodic);
        if ( ! parallel.periodic.x ) bc.x.lower = bc.x.upper = current::bc::none;
        if ( ! parallel.periodic.y ) bc.y.lower = bc.y.upper = current::bc::none;

        // Set default filtering
        filter = new filter::lowpass( make_float2( 0.5, 0.5 ) );

        // Reset iteration number
        iter = 0;
    };
    
    /**
     * @brief Destroy the Current object
     * 
     */
    ~Current() {
        delete (filter);
        
        delete (J);
        delete (fJ);

        delete( fft_forward );
    }

    /**
     * @brief Get the type of boundary conditions
     * 
     * @return current::bc_type
     */
    current::bc_type get_bc( ) const noexcept { return bc; }

    /**
     * @brief Set the boundary conditions
     * 
     * @param new_bc    New boundary condition values
     */
    void set_bc( current::bc_type new_bc ) {

        // Validate parameters
        if ( (new_bc.x.lower == current::bc::periodic) || (new_bc.x.upper == current::bc::periodic) ) {
            if ( new_bc.x.lower != new_bc.x.upper ) {
                std::cerr << "(*error*) Current boundary type mismatch along x.\n";
                std::cerr << "(*error*) When choosing periodic boundaries both lower and upper types must be set to current::bc::periodic.\n";
                mpi::abort(1);
            }
        }

        if ( (new_bc.y.lower == current::bc::periodic) || (new_bc.y.upper == current::bc::periodic) ) {
            if ( new_bc.y.lower != new_bc.y.upper ) {
                std::cerr << "(*error*) Current boundary type mismatch along y.\n";
                std::cerr << "(*error*) When choosing periodic boundaries both lower and upper types must be set to emf::bc::periodic.\n";
                mpi::abort(1);
            }
        }

        if ( J -> part.periodic.x && new_bc.x.lower != current::bc::periodic ) {
            std::cerr << "(*error*) Only periodic x boundaries are supported with periodic x parallel partitions.\n";
            mpi::abort(1);
        }

        if ( J -> part.periodic.y && new_bc.y.lower != current::bc::periodic ) {
            std::cerr << "(*error*) Only periodic y boundaries are supported with periodic y parallel partitions.\n";
            mpi::abort(1);
        }

        // Store new values
        bc = new_bc;

        if ( mpi::root() ) {
            std::string bc_name[] = {"none", "periodic", "reflecting"};
            std::cout << "(*info*) Current boundary conditions\n";
            std::cout << "(*info*) x : [ " << bc_name[ bc.x.lower ] << ", " << bc_name[ bc.x.upper ] << " ]\n";
            std::cout << "(*info*) y : [ " << bc_name[ bc.y.lower ] << ", " << bc_name[ bc.y.upper ] << " ]\n";
        }
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

    /**
     * @brief Save electric current data to diagnostic file
     * 
     * @param field     Which field to save (J or fJ)
     * @param jc        Current component to save
     */
    void save( const current::field field, const fcomp::cart jc );
};
