#pragma once

#include "core/vec_types.cuh"
#include "core/bounds.hpp"

#include "grid/tiled_vec3.cuh"
#include "moving_window.hpp"
#include "parallel/partition.hpp"
#include "current.hpp"

#include <cmath>
#include <ostream>
#include <sstream>
#include <string>

/**
 * @brief Electro-magnetic fields (FDTD based solver)
 * 
 */
class emf {
    public:

    enum class quantity {e = 0, b};
    /**
     * @brief Boundary condition types
     * 
     */
    struct bc { 
        enum type { none = 0, periodic, pec, pmc };
    };
    using bc_type = bounds_2d<bc::type>;

    private:

    /// @brief Boundary condition (global values)
    emf::bc_type global_bc;

    /// @brief Boundary condition (local rank values)
    emf::bc_type local_bc;

    /// @brief cell size
    const float2 dx;

    /// @brief time step
    const double dt;

    /// @brief Iteration number
    int iter = 0;

    /// @brief Device buffer for field energy calculations (6 values)
    double * d_energy = nullptr;

    /// @brief Moving window information
    moving_window mov_window;

    /**
     * @brief Process boundary conditions
     *
     */
    void process_bc( );

    /// @brief Check if a boundary type is a conducting wall (pec or pmc)
    static bool is_conducting( emf::bc::type t ) noexcept {
        return t == emf::bc::pec || t == emf::bc::pmc;
    }

    /**
     * @brief Validate boundary conditions against the parallel partition,
     *        moving window and tile size
     * 
     * @note Calls mpi::fatal() on invalid combinations
     * 
     * @param test_bc   Boundary conditions to validate
     */
    void validate_bc( const emf::bc_type & test_bc ) const {

        // Periodic boundaries must be set on both sides
        if ( (test_bc.x.lower == emf::bc::periodic) || (test_bc.x.upper == emf::bc::periodic) ) {
            if ( test_bc.x.lower != test_bc.x.upper ) {
                mpi::fatal( "EMF boundary type mismatch along x."
                            " When choosing periodic boundaries both lower and upper types"
                            " must be set to emf::bc::periodic." );
            }
        }

        if ( (test_bc.y.lower == emf::bc::periodic) || (test_bc.y.upper == emf::bc::periodic) ) {
            if ( test_bc.y.lower != test_bc.y.upper ) {
                mpi::fatal( "EMF boundary type mismatch along y."
                            " When choosing periodic boundaries both lower and upper types"
                            " must be set to emf::bc::periodic." );
            }
        }

        // Boundary periodicity must match partition periodicity (both ways)
        const auto & part = E -> get_part();

        if ( part.periodic.x && test_bc.x.lower != emf::bc::periodic ) {
            mpi::fatal( "Only periodic x boundaries are supported with periodic x parallel partitions.");
        }

        if ( part.periodic.y && test_bc.y.lower != emf::bc::periodic ) {
            mpi::fatal( "Only periodic y boundaries are supported with periodic y parallel partitions.");
        }

        if ( ! part.periodic.x && test_bc.x.lower == emf::bc::periodic ) {
            mpi::fatal( "Periodic x boundaries require a periodic x parallel partition.");
        }

        if ( ! part.periodic.y && test_bc.y.lower == emf::bc::periodic ) {
            mpi::fatal( "Periodic y boundaries require a periodic y parallel partition.");
        }

        // The moving window requires open x boundaries
        if ( mov_window.active() &&
             ( test_bc.x.lower != emf::bc::none || test_bc.x.upper != emf::bc::none ) ) {
            mpi::fatal( "With a moving window both x boundaries must be emf::bc::none." );
        }

        // Wall boundaries touch cells -1..1 at the lower edge and nx-2..nx+1
        // at the upper edge; with tiles narrower than 3 cells the lower and
        // upper edge blocks of the bc kernels would overlap when local_ntiles == 1
        if ( ( is_conducting( test_bc.x.lower ) || is_conducting( test_bc.x.upper ) ) && E -> tile_dims.x < 3 ) {
            mpi::fatal( "PEC / PMC x boundaries require tile_dims.x >= 3." );
        }

        if ( ( is_conducting( test_bc.y.lower ) || is_conducting( test_bc.y.upper ) ) && E -> tile_dims.y < 3 ) {
            mpi::fatal( "PEC / PMC y boundaries require tile_dims.y >= 3." );
        }
    }

    /**
     * @brief Move simulation window if needed
     * 
     */
    void move_window( );

    public:

    /// @brief Electric field
    grid::tiled_vec3<float> * E = nullptr;
    /// @brief Magnetic field
    grid::tiled_vec3<float> * B = nullptr;
    /// @brief Simulation box size
    const float2 box;

    /**
     * @brief Construct a new EMF object
     * 
     * @param global_ntiles     Global number of tiles
     * @param tile_dims         Individual tile size
     * @param box               Global simulation box size
     * @param dt                Time step
     * @param parallel          Parallel partition 
     */
    emf( uint2 const global_ntiles, uint2 const tile_dims, float2 const box,
        double const dt, mpi::cart2d & parallel ) : 
        dx( make_float2( box.x / ( tile_dims.x * global_ntiles.x ), box.y / ( tile_dims.y * global_ntiles.y ) ) ),
        dt( dt ), box(box)
    {
        // Verify Courant condition
        auto cour = std::sqrt( 1.0f/( 1.0f/(dx.x*dx.x) + 1.0f/(dx.y*dx.y) ) );
        if ( dt >= cour ){
            std::ostringstream msg;
            msg << "Invalid timestep, courant condition violation."
                << " For the current resolution " << dx
                << " the maximum timestep is dt = " << cour;
            mpi::fatal( msg.str() );
        }

        // Guard cells (1 below, 2 above)
        // These are required for the Yee solver AND for field interpolation
        bounds_2d<unsigned int> gc;
        gc.x = {1,2};
        gc.y = {1,2};

        E = new grid::tiled_vec3<float> ( global_ntiles, tile_dims, gc, parallel );
        E -> name = "Electric field";

        B = new grid::tiled_vec3<float> ( global_ntiles, tile_dims, gc, parallel );
        B -> name = "Magnetic field";

        // Check shared memory buffer
        // E and B tiles (including guard cells) are loaded into shared memory for
        // field interpolation
        auto local_mem_size = gpu::block::shared_mem_size();
        if ( local_mem_size < 2 * E->tile_vol * sizeof( float3 ) ) {
            std::ostringstream msg;
            msg << "Tile size too large " << tile_dims << " (plus guard cells),"
                << " insufficient local memory (" << local_mem_size << " B) for EMF object.";
            mpi::fatal( msg.str() );
        }

        // Zero fields
        E -> zero();
        B -> zero();

        // Default boundary conditions follow the partition periodicity:
        // periodic where the partition is periodic, none otherwise
        emf::bc_type default_bc = emf::bc_type (emf::bc::periodic);
        if ( ! parallel.periodic.x ) default_bc.x.lower = default_bc.x.upper = emf::bc::none;
        if ( ! parallel.periodic.y ) default_bc.y.lower = default_bc.y.upper = emf::bc::none;
        set_bc( default_bc );

        // Reserve device memory for energy diagnostic
        d_energy = gpu::device::malloc<double>( 6 );
    }
    
    emf( const emf& ) = delete;
    emf& operator=( const emf& ) = delete;

    /**
     * @brief Destroy the EMF object
     * 
     */
    ~emf() {
        delete (E);
        delete (B);
        gpu::device::free( d_energy );
    }

    /**
     * @brief Stream extraction
     * 
     * @param os 
     * @param obj 
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, const emf & obj) {
        return os << "EMF object";
    }

    /**
     * @brief Get the iteration number
     * 
     * @return int
     */
    int get_iter() const noexcept { return iter; }

    /**
     * @brief Get the type of global boundary conditions
     * 
     * @return emf::bc_type 
     */
    emf::bc_type get_bc( ) const noexcept { return global_bc; }

    /**
     * @brief Set the boundary conditions
     * 
     * @param new_bc    New boundary condition values
     */
    void set_bc( emf::bc_type new_bc ) {

        // Validate parameters
        validate_bc( new_bc );

        // Store new values
        local_bc = global_bc = new_bc;

        // Correct local rank values
        const auto & part = E -> get_part();
        if ( ! part.on_edge( coord::x, edge::lower ) ) local_bc.x.lower = emf::bc::none;
        if ( ! part.on_edge( coord::x, edge::upper ) ) local_bc.x.upper = emf::bc::none;
        if ( ! part.on_edge( coord::y, edge::lower ) ) local_bc.y.lower = emf::bc::none;
        if ( ! part.on_edge( coord::y, edge::upper ) ) local_bc.y.upper = emf::bc::none;
    }

    /**
     * @brief Turns on the moving window algorithm
     * 
     * Sets both x boundaries to emf::bc::none. Can only be called before the
     * simulation has started (iter = 0); calls mpi::fatal() otherwise or if
     * the x boundaries are periodic.
     */
    void set_moving_window() { 
        if ( iter != 0 ) {
            mpi::fatal( "set_moving_window() called with iter != 0" );
        }

        // Use the global values, local values may have been reset on internal ranks
        if ( global_bc.x.lower == emf::bc::periodic ) {
            mpi::fatal( "Unable to set_moving_window() with periodic x boundaries" );
        }

        mov_window.init( dx.x );

        // Correct boundary conditions
        auto new_bc = global_bc;
        new_bc.x.lower = new_bc.x.upper = emf::bc::none;
        set_bc( new_bc );
    }

    /**
     * @brief Advance EM field 1 iteration assuming no current
     * 
     */
    void advance( );
    
    /**
     * @brief Advance EM field 1 iteration
     * 
     * @note `current` must already have been advanced (i.e. boundary conditions and filtering)
     * 
     * @param current   Current density
     */    
    void advance( current & current );

    /**
     * @brief Save EM field component to file
     * 
     * @param quant     Which field to save (E, or B)
     * @param fc        Which field component to save (x, y or z)
     */
    void save( const emf::quantity quant, const fcomp::cart fc ) const;
    
    /**
     * @brief Get EM field energy
     * 
     * @note The energy is recalculated each time this routine is called
     * @warning Collective: values are summed over all parallel nodes
     * 
     * @param ene_E     Electric field energy (per component)
     * @param ene_B     Magnetic field energy (per component)
     */
    void get_energy( double3 & ene_E, double3 & ene_B ) const;
};
