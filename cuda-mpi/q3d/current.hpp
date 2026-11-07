#pragma once

#include <memory>
#include <utility>

#include "moving_window.hpp"
#include "filter.hpp"

#include "cyl_modes/tiled_cyl3.cuh"


class current {

    public:

    /**
     * @brief Boundary condition types
     * 
     * @note Values match emf::bc, and the current boundary conditions must
     *       match the field boundary conditions.
     * 
     * - none:      open boundary, no processing
     * - axial:     axial boundary
     * - periodic:  periodic boundary (handled by the grid guard cell updates)
     * - pec, pmc:  conducting wall with reflecting particles. The current
     *              deposited beyond the wall is folded back into the domain
     *              (particle reflection), and the guard cells / wall node are
     *              set from the image currents of the wall type:
     *              pec - normal J even, tangential J odd, tangential J = 0 on the wall node
     *              pmc - normal J odd, tangential J even, tangential J doubled on the wall node
     */
    struct bc {
        enum type { none = 0, axial, periodic, pec, pmc };
    };
    using bc_type = bounds_2d<bc::type>;

    private:

    /// @brief Number of cylindrical modes (including fundamental mode)
    const int nmodes;

    /// @brief Simulation box size
    const float2 box;

    /// @brief cell size
    const float2 dx;
    
    /// @brief time step
    float dt;

    /// @brief Moving window information
    moving_window mov_window;

    /// @brief Boundary condition (global values)
    current::bc_type global_bc;

    /// @brief Boundary condition (local rank values)
    current::bc_type local_bc;

    /// @brief Iteration number
    int iter;

    /// @brief Digital filter
    filter::digital filter;

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
     * @brief Fold the current deposited beyond the walls / axis back into the
     *        domain and normalize for ring particles (all modes)
     * 
     * @note Use exactly once per time step, on the raw deposit: after
     *       add_from_gc() and before copy_to_gc()
     */
    void fold_and_normalize();

    /**
     * @brief Set the wall / axis image currents in the guard cells (all modes)
     * 
     * @note Idempotent. Use after copy_to_gc() on the normalized current, and
     *       after every filter pass.
     */
    void set_images();

    /**
     * @brief Validate boundary conditions against the parallel partition,
     *        moving window and tile size
     * 
     * @note Calls mpi::fatal() on invalid combinations
     * 
     * @param test_bc   Global boundary conditions to validate
     */
    void validate_bc( const current::bc_type & test_bc ) const {

        if ( (test_bc.y.lower == current::bc::periodic) || (test_bc.y.upper == current::bc::periodic) ) {
            mpi::fatal( "Periodic boundaries along radial direction are not allowed" );
        }

        if ( test_bc.y.lower != current::bc::axial) {
            mpi::fatal( "Lower radial boundary must be set to current::bc::axial" );
        }

        if ( test_bc.y.upper == current::bc::axial ||
             test_bc.x.lower == current::bc::axial || test_bc.x.upper == current::bc::axial ) {
            mpi::fatal( "current::bc::axial is only valid for the lower radial boundary" );
        }

        // Periodic boundaries must be set on both sides
        if ( (test_bc.x.lower == current::bc::periodic) || (test_bc.x.upper == current::bc::periodic) ) {
            if ( test_bc.x.lower != test_bc.x.upper ) {
                mpi::fatal( "Current boundary type mismatch along x."
                            " When choosing periodic boundaries both lower and upper types"
                            " must be set to current::bc::periodic." );
            }
        }

        // Boundary periodicity must match partition periodicity (both ways)
        const auto & part = J -> get_part();

        if ( part.periodic.x && test_bc.x.lower != current::bc::periodic ) {
            mpi::fatal( "Only periodic x boundaries are supported with periodic x parallel partitions.");
        }

        if ( ! part.periodic.x && test_bc.x.lower == current::bc::periodic ) {
            mpi::fatal( "Periodic x boundaries require a periodic x parallel partition.");
        }

        // The moving window requires open x boundaries
        if ( mov_window.active() &&
             ( test_bc.x.lower != current::bc::none || test_bc.x.upper != current::bc::none ) ) {
            mpi::fatal( "With a moving window both x boundaries must be current::bc::none." );
        }

        // Tile size / guard cells required by the boundary kernels (see current.cu)
        const auto tile_dims = J -> get_tile_dims();
        const auto gc        = J -> mode0().gc;

        if ( tile_dims.y < 3 || gc.y.lower < 1 ) {
            mpi::fatal( "The axial boundary requires tile_dims.y >= 3 and at least 1 lower r guard cell" );
        }

        if ( ( is_conducting( test_bc.x.lower ) || is_conducting( test_bc.x.upper ) ) &&
             ( tile_dims.x < 4 || gc.x.upper < 2 ) ) {
            mpi::fatal( "pec / pmc z boundaries require tile_dims.x >= 4 and at least 2 upper z guard cells" );
        }

        if ( is_conducting( test_bc.y.upper ) && ( tile_dims.y < 4 || gc.y.upper < 2 ) ) {
            mpi::fatal( "pec / pmc r boundaries require tile_dims.y >= 4 and at least 2 upper r guard cells" );
        }
    }

    public:

    /**
     * @brief Current density
     * 
     * @note Owned by this object
     */
    std::unique_ptr< cyl_modes::tiled_cyl3<float> > J;

    /**
     * @brief Construct a new Current object
     * 
     * @param nmodes        Number of cylindrical modes (>= 1)
     * @param global_ntiles Global number of tiles in z,r direction
     * @param tile_dims     Tile size (#cells)
     * @param box           Simulation box size (sim. units)
     * @param dt            Time step
     * @param parallel      Parallel partition
     */
    current( const unsigned nmodes, uint2 const global_ntiles, uint2 const tile_dims, float2 const box, double const dt, const mpi::cart2d & parallel ):
        nmodes( nmodes ),
        box(box),
        dx( make_float2( box.x / ( tile_dims.x * global_ntiles.x ), box.y / ( tile_dims.y * global_ntiles.y ) ) ),
        dt(dt),
        iter(0)
    {
        // Guard cells (1 below, 2 above)
        bounds_2d<unsigned int> gc;
        gc.x = {1,2};
        gc.y = {1,2};

        J = std::make_unique< cyl_modes::tiled_cyl3<float> >( nmodes, global_ntiles, tile_dims, gc, parallel );
        J -> set_name( "Current" );

        // Zero initial current
        // This is only relevant for diagnostics, current is always zeroed before deposition
        J -> zero();

        // Set default boundary conditions
        current::bc_type default_bc = current::bc_type (current::bc::none);
        default_bc.y.lower = current::bc::axial;
        if ( parallel.periodic.x ) default_bc.x.lower = default_bc.x.upper = current::bc::periodic;
        set_bc( default_bc );

        // Filtering is disabled by default (filter is default constructed empty)
    };

    /**
     * @brief Get the number of modes (including mode 0)
     * 
     * @return auto 
     */
    int get_nmodes() const { return J -> get_nmodes(); };

    /**
     * @brief Get mode 0 cyl. grid (real)
     * 
     * @return auto& 
     */
    auto & mode0() {
        return J -> mode0();
    }

    const auto & mode0() const {
        return J -> mode0();
    }

    /**
     * @brief Get mode m > 0 cyl. grid (complex)
     * 
     * @param m 
     * @return auto& 
     */
    auto & mode( int m ) {
        return J -> mode( m );
    }

    const auto & mode( int m ) const {
        return std::as_const( *J ).mode( m );
    }

    /**
     * @brief Get the boundary conditions
     * 
     * @return current::bc_type 
     */
    current::bc_type get_bc( ) const { return global_bc; }

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
     * @brief Advances electric current density 1 time step
     * 
     * The routine will:
     * 1. Update the guard cells
     * 2. Fold the current deposited beyond walls / axis (raw deposit) and
     *    normalize grid values for ring particles
     * 3. Update guard cells and set the wall / axis images
     * 4. Apply digital filtering
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
     * @brief Save current density to file
     * 
     * @param jc        Which current component to save (r, θ or z)
     * @param m         Mode
     */
    void save( const fcomp::cyl jc, unsigned m );

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
     * @brief Set the current filter to apply
     * 
     * @param new_filter 
     */
    void set_filter( filter::digital const & new_filter ) {
        filter = new_filter;
    }

    /**
     * @brief Apply digital filter to all modes
     * 
     * @note Every pass also processes the first upper guard column / row
     *       (upper wall nodes), and the wall / axis images are refreshed
     *       after every pass. Guard cells and images must be valid on entry
     *       (see advance()).
     */
    void apply_filter() {
        filter.apply( *J, true, [this]{ set_images(); } );
    }
};
