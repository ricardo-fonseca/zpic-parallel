#pragma once

#include <memory>

#include "grid/tiled_vec3.cuh"

#include "filter.hpp"
#include "moving_window.hpp"


class current {

    public:

    /**
     * @brief Boundary condition types
     * 
     * @note Values match emf::bc, and the current boundary conditions must
     *       match the field boundary conditions.
     * 
     * - none:      open boundary, no processing
     * - periodic:  periodic boundary (handled by the grid guard cell updates)
     * - pec, pmc:  conducting wall with reflecting particles. The current
     *              deposited beyond the wall is folded back into the domain
     *              (particle reflection), and the guard cells / wall node are
     *              set from the image currents of the wall type:
     *              pec - normal J even, tangential J odd, tangential J = 0 on the wall node
     *              pmc - normal J odd, tangential J even, tangential J doubled on the wall node
     */
    struct bc {
        enum type { none = 0, periodic, pec, pmc };
    };
    using bc_type = bounds_2d<bc::type>;

    private:

    /// @brief Simulation box size
    const float2 box;
    
    /// @brief Time step

    const double dt;

    /// @brief Moving window information
    moving_window mov_window;

    /// @brief Boundary condition (global values)
    current::bc_type global_bc;

    /// @brief Boundary condition (local rank values)
    current::bc_type local_bc;

    /// @brief Iteration number
    int iter = 0;

    /**
     * @brief Digital filter
     * 
     * @note Owned by this object, use set_filter() to replace it
     */
    std::unique_ptr<filter::digital> filter;

    /**
     * @brief Check if boundary condition is PEC / PMC
     * 
     * @param t     Boundary condition tyoe
     * @return      True if t is PEC or PMC, false otherwise
     */
    static bool is_conducting( current::bc::type t ) noexcept {
        return t == current::bc::pec || t == current::bc::pmc;
    }


    /**
     * @brief Apply wall (pec / pmc) boundary conditions on local physical edges
     * 
     * @param fold  `true`: fold the current deposited in the guard cells beyond
     *              the wall back into the domain, apply the wall node image
     *              correction and set the guard cells from the image currents.
     *              Use exactly once per time step, after add_from_gc() /
     *              copy_to_gc().
     *              `false`: only set the guard cells from the image currents
     *              (idempotent). Use after a filter pass.
     */
    void process_bc( const bool fold );

    /**
     * @brief Validate boundary conditions against the parallel partition,
     *        moving window and tile size
     * 
     * @note Calls mpi::fatal() on invalid combinations
     * 
     * @param test_bc   Boundary conditions to validate
     */
    void validate_bc( const current::bc_type & test_bc ) const {

        // Periodic boundaries must be set on both sides
        if ( (test_bc.x.lower == current::bc::periodic) || (test_bc.x.upper == current::bc::periodic) ) {
            if ( test_bc.x.lower != test_bc.x.upper ) {
                mpi::fatal( "Current boundary type mismatch along x."
                            " When choosing periodic boundaries both lower and upper types"
                            " must be set to current::bc::periodic." );
            }
        }

        if ( (test_bc.y.lower == current::bc::periodic) || (test_bc.y.upper == current::bc::periodic) ) {
            if ( test_bc.y.lower != test_bc.y.upper ) {
                mpi::fatal( "Current boundary type mismatch along y."
                            " When choosing periodic boundaries both lower and upper types"
                            " must be set to current::bc::periodic." );
            }
        }

        // Boundary periodicity must match partition periodicity (both ways)
        const auto & part = J -> get_part();

        if ( part.periodic.x && test_bc.x.lower != current::bc::periodic ) {
            mpi::fatal( "Only periodic x boundaries are supported with periodic x parallel partitions.");
        }

        if ( part.periodic.y && test_bc.y.lower != current::bc::periodic ) {
            mpi::fatal( "Only periodic y boundaries are supported with periodic y parallel partitions.");
        }

        if ( ! part.periodic.x && test_bc.x.lower == current::bc::periodic ) {
            mpi::fatal( "Periodic x boundaries require a periodic x parallel partition.");
        }

        if ( ! part.periodic.y && test_bc.y.lower == current::bc::periodic ) {
            mpi::fatal( "Periodic y boundaries require a periodic y parallel partition.");
        }

        // The moving window requires open x boundaries
        if ( mov_window.active() &&
             ( test_bc.x.lower != current::bc::none || test_bc.x.upper != current::bc::none ) ) {
            mpi::fatal( "With a moving window both x boundaries must be current::bc::none." );
        }

        // Wall boundaries touch cells -1..1 at the lower edge and nx-2..nx+1
        // at the upper edge; with tiles narrower than 4 cells the lower and upper
        // edge blocks of the bc kernels would overlap when local_ntiles == 1
        if ( ( is_conducting( test_bc.x.lower ) || is_conducting( test_bc.x.upper ) ) && J -> tile_dims.x < 4 ) {
            mpi::fatal( "PEC / PMC x boundaries require tile_dims.x >= 4." );
        }

        if ( ( is_conducting( test_bc.y.lower ) || is_conducting( test_bc.y.upper ) ) && J -> tile_dims.y < 4 ) {
            mpi::fatal( "PEC / PMC y boundaries require tile_dims.y >= 4." );
        }

        // The upper wall node is on the first upper guard cell, and the image
        // currents use the second one
        if ( ( is_conducting( test_bc.x.upper ) && J -> gc.x.upper < 2 ) ||
             ( is_conducting( test_bc.y.upper ) && J -> gc.y.upper < 2 ) ) {
            mpi::fatal( "PEC / PMC upper boundaries require at least 2 upper guard cells." );
        }
    }

    public:

    /**
     * @brief Current density
     * 
     * @note Owned by this object
     */
    grid::tiled_vec3<float> * J = nullptr;

    /**
     * @brief Construct a new Current object
     * 
     * @param global_ntiles     Global number of tiles
     * @param tile_dims         Individual tile size
     * @param box               Global simulation box size
     * @param dt                Time step
     * @param parallel          Parallel partition 
     */
    current( uint2 const global_ntiles, uint2 const tile_dims, float2 const box, double const dt, mpi::cart2d & parallel ):
        box(box), 
        dt(dt)
    {
        // Guard cells (1 below, 2 above)
        bounds_2d<unsigned int> gc;
        gc.x = {1,2};
        gc.y = {1,2};

        J = new grid::tiled_vec3<float> ( global_ntiles, tile_dims, gc, parallel );
        J -> name = "Current density";

        // Zero initial current
        // This is only relevant for diagnostics, current is always zeroed before deposition
        J -> zero();

        // Default boundary conditions follow the partition periodicity:
        // periodic where the partition is periodic, none otherwise
        current::bc_type default_bc = current::bc_type (current::bc::periodic);
        if ( ! parallel.periodic.x ) default_bc.x.lower = default_bc.x.upper = current::bc::none;
        if ( ! parallel.periodic.y ) default_bc.y.lower = default_bc.y.upper = current::bc::none;
        set_bc( default_bc );

        // Disable filtering by default
        filter = std::make_unique<filter::none>( );
    };
    
    current( const current& ) = delete;
    current& operator=( const current& ) = delete;

    /**
     * @brief Destroy the Current object
     * 
     */
    ~current() {
        delete (J);
    }

    /**
     * @brief Get the type of global boundary conditions
     * 
     * @return current::bc_type
     */
    current::bc_type get_bc( ) const noexcept { return global_bc; }

    /**
     * @brief Set the boundary conditions
     * 
     * @note Must match the field (emf) boundary conditions
     * 
     * @param new_bc    New boundary condition values
     */
    void set_bc( current::bc_type new_bc ) {

        // Validate parameters
        validate_bc( new_bc );

        // Store new values
        local_bc = global_bc = new_bc;

        // Correct local rank values
        const auto & part = J -> get_part();
        if ( ! part.on_edge( coord::x, edge::lower ) ) local_bc.x.lower = current::bc::none;
        if ( ! part.on_edge( coord::x, edge::upper ) ) local_bc.x.upper = current::bc::none;
        if ( ! part.on_edge( coord::y, edge::lower ) ) local_bc.y.lower = current::bc::none;
        if ( ! part.on_edge( coord::y, edge::upper ) ) local_bc.y.upper = current::bc::none;
    }

    /**
     * @brief Turns on the moving window algorithm
     * 
     * Sets both x boundaries to current::bc::none. Can only be called before
     * the simulation has started (iter = 0); calls mpi::fatal() otherwise or
     * if the x boundaries are periodic.
     */
    void set_moving_window() { 
        if ( iter != 0 ) {
            mpi::fatal( "set_moving_window() called with iter != 0" );
        }

        // Use the global values, local values may have been reset on internal ranks
        if ( global_bc.x.lower == current::bc::periodic ) {
            mpi::fatal( "Unable to set_moving_window() with periodic x boundaries" );
        }

        // Turn moving window on
        mov_window.init( box.x, J -> get_global_dims().x );

        // Correct boundary conditions
        auto new_bc = global_bc;
        new_bc.x.lower = new_bc.x.upper = current::bc::none;
        set_bc( new_bc );
    }

    /**
     * @brief Replace the digital filter
     * 
     * @note Stores a copy (via clone()); the caller keeps ownership of
     *       `new_filter`
     * 
     * @param new_filter    New filter object
     */
    void set_filter( const filter::digital & new_filter ) {
        filter = new_filter.clone();
    }

    /**
     * @brief Get the current digital filter
     * 
     * @return const filter::digital& 
     */
    const filter::digital & get_filter() const noexcept { return *filter; }

    /**
     * @brief Advances electric current density 1 time step
     * 
     * The routine will:
     * 1. Add up current deposited on guard cells
     * 2. Apply physical boundary conditions
     * 3. Apply filtering
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
     * @note With pmc walls the tangential current on the wall nodes is the
     *       image-corrected (doubled) value seen by the field solver
     * 
     * @param jc    Current component to save
     */
    void save( fcomp::cart const jc ) const;
};
