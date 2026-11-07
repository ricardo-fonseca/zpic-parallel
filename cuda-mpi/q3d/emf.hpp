#pragma once

#include "core/bounds.hpp"
#include "current.hpp"
#include "moving_window.hpp"

#include "cyl_modes/tiled_cyl3.cuh"

#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>


class emf {

    public:

    enum class quantity {e = 0, b};
    /**
     * @brief Boundary condition types
     * 
     */
    struct bc { 
        enum type { none = 0, axial, periodic, pec, pmc };
    };
    using bc_type = bounds_2d<bc::type>;

    /**
    * @brief CFL time limit from cell size
    * 
    * The limit is
    * 
    *      1 / Δt² = 1/Δz² + 1/Δr² + max( m_max², 0.3 ) / Δr²
    * 
    * with m_max = nmodes - 1 the highest mode.
    * 
    * @note This is an empirical fit: the m_max term was obtained from the
    *       stable time step of the Yee solver (including the axial cell
    *       treatment) for nmodes = 1 to 6 and Δz/Δr = 1/2, 1 and 2, measured
    *       over 3000 steps with random fields in the highest mode. The
    *       measured stability limit is 1.008 to 1.12 times this value, so a
    *       small additional safety factor is advisable.
    * 
    * @param m         Number of cylindrical modes (including mode 0)
    * @param dx        Cell size
    * @return float    CFL time limit
    */
    static float courant( const unsigned m, const float2 dx ) {
        const double dz = dx.x;
        const double dr = dx.y;

        // Highest mode
        const double mmax = ( m > 0 ) ? m - 1.0 : 0.0;

        const double cour = 1.0 / ( dz * dz ) + 1.0 / ( dr * dr ) 
                          + std::max( mmax * mmax, 0.3 ) / ( dr * dr );

        return std::sqrt( 1.0 / cour );
    }

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

        if ( (test_bc.y.lower == emf::bc::periodic) || (test_bc.y.upper == emf::bc::periodic) ) {
            mpi::fatal( "Periodic boundaries along radial direction are not allowed" );
        }

        if ( test_bc.y.lower != emf::bc::axial) {
            mpi::fatal( "Lower radial boundary must be set to emf::bc::axial" );
        }

        if ( test_bc.y.upper == emf::bc::axial ||
             test_bc.x.lower == emf::bc::axial || test_bc.x.upper == emf::bc::axial ) {
            mpi::fatal( "emf::bc::axial is only valid for the lower radial boundary" );
        }

        // Periodic boundaries must be set on both sides
        if ( (test_bc.x.lower == emf::bc::periodic) || (test_bc.x.upper == emf::bc::periodic) ) {
            if ( test_bc.x.lower != test_bc.x.upper ) {
                mpi::fatal( "EMF boundary type mismatch along x."
                            " When choosing periodic boundaries both lower and upper types"
                            " must be set to emf::bc::periodic." );
            }
        }

        // Boundary periodicity must match partition periodicity (both ways)
        const auto & part = E -> get_part();

        if ( part.periodic.x && test_bc.x.lower != emf::bc::periodic ) {
            mpi::fatal( "Only periodic x boundaries are supported with periodic x parallel partitions.");
        }

        if ( ! part.periodic.x && test_bc.x.lower == emf::bc::periodic ) {
            mpi::fatal( "Periodic x boundaries require a periodic x parallel partition.");
        }

        // The moving window requires open x boundaries
        if ( mov_window.active() &&
             ( test_bc.x.lower != emf::bc::none || test_bc.x.upper != emf::bc::none ) ) {
            mpi::fatal( "With a moving window both x boundaries must be emf::bc::none." );
        }

        // Tile size / guard cells required by the axis and wall boundary code (see emf.cu)
        const auto tile_dims = E -> get_tile_dims();
        const auto gc        = E -> mode0().gc;

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

    /**
     * @brief Move simulation window if needed
     * 
     */
    void move_window( );

    public:

    /// @brief Number of cylindrical modes (including fundamental mode)
    const int nmodes;
    /// @brief Simulation box size
    const float2 box;
    /// @brief Electric field
    std::unique_ptr< cyl_modes::tiled_cyl3<float> > E;
    /// @brief Magnetic field
    std::unique_ptr< cyl_modes::tiled_cyl3<float> > B;

    /**
     * @brief Construct a new EMF object
     * 
     * @param nmodes        Number of cylindrical modes (>= 1)
     * @param global_ntiles Global number of tiles in z,r direction
     * @param tile_dims     Tile size (#cells)
     * @param box           Simulation box size (sim. units)
     * @param dt            Time step
     * @param parallel      Parallel partition 
     */
    emf( unsigned int nmodes, uint2 const global_ntiles, uint2 const tile_dims, float2 const box, 
        double const dt, const mpi::cart2d & parallel ) :
        dx( make_float2( box.x / ( tile_dims.x * global_ntiles.x ), box.y / ( tile_dims.y * global_ntiles.y ) ) ),
        dt( dt ),
        iter(0),
        nmodes( nmodes ),
        box(box)
    {
        // Verify Courant condition
        auto cour = courant(nmodes, dx );
        if ( dt >= cour ){
            std::ostringstream msg;

            msg << "Invalid timestep, courant condition violation."
                << " For the no. modes / current resolution " << nmodes << " / " << dx
                << " the maximum timestep is dt = " << cour;
            mpi::fatal( msg.str() );
        }

        // Guard cells (1 below, 2 above)
        // These are required for the Yee solver AND for field interpolation
        bounds_2d<unsigned int> gc;
        gc.x = {1,2};
        gc.y = {1,2};

        E = std::make_unique< cyl_modes::tiled_cyl3<float> > ( nmodes, global_ntiles, tile_dims, gc, parallel );
        E -> set_name( "E" );

        B = std::make_unique< cyl_modes::tiled_cyl3<float> >( nmodes, global_ntiles, tile_dims, gc, parallel );
        B -> set_name( "B" );

        // Check that shared memory can hold the E and B fields of 1 tile for
        // the largest mode type (see the Yee kernels in emf.cu)
        auto & E0 = E -> mode0();
        auto req_mem_size = 2 * E0.tile_vol * 
            (( nmodes == 1 ) ? sizeof( cyl_float3 ) : sizeof( cyl_cfloat3 ));
        auto local_mem_size = gpu::block::shared_mem_size();
        
        if ( local_mem_size < req_mem_size ) {
            std::ostringstream msg;
            msg << "Tile size too large " << tile_dims << " (plus guard cells),"
                << " insufficient local memory (" << local_mem_size << " B) for EMF object.";
            mpi::fatal( msg.str() );
        }

        // Zero fields
        E -> zero();
        B -> zero();

        // Reserve device memory for energy diagnostic
        d_energy = gpu::device::malloc<double>( 6 );

        // Set default boundary conditions
        emf::bc_type default_bc = emf::bc_type (emf::bc::none);
        default_bc.y.lower = emf::bc::axial;
        if ( parallel.periodic.x ) default_bc.x.lower = default_bc.x.upper = emf::bc::periodic;
        set_bc( default_bc );
    };

    emf( const emf& ) = delete;
    emf& operator=( const emf& ) = delete;

    /**
     * @brief Destroy the EMF object
     * 
     */
    ~emf() {
        gpu::device::free( d_energy );
    }

    friend std::ostream& operator<<(std::ostream& os, const emf & obj) {
        os << "EMF object";
        return os;
    }

    /**
     * @brief Get the iteration number
     * 
     * @return auto 
     */
    int get_iter() const noexcept { return iter; }

    /**
     * @brief Get the time step (dt) value
     * 
     * @return auto 
     */
    double get_dt() const noexcept { return dt; }

    /**
     * @brief Get the boundary condition type
     * 
     * @return emf::bc_type 
     */
    emf::bc_type get_bc( ) const noexcept { return global_bc; }

    /**
     * @brief Set the boundary condition type
     * 
     * @param new_bc    New boundary condition types (all directions)
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
     * @brief Sets moving window algorithm
     * 
     * This method can only be called before the simulation has started (iter = 0)
     * 
     * @return int  0 on success, -1 on error
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
     * @param current   Current density
     */
    void advance( current & current );

    /**
     * @brief Save EM field component to file
     * 
     * @param field     Which field to save (E or B)
     * @param fc        Which field component to save (r, θ or z)
     * @param m         Mode
     */
    void save( emf::quantity const field, const fcomp::cyl fc, const int m ) const;
    
    /**
     * @brief Get EM field energy
     * 
     * 
     * @note The energy will be recalculated each time this routine is called.
     *       For modes m > 0 the energy includes the factor 1/2 from the θ
     *       average of |Re(F_m exp(i m θ))|^2.
     * 
     * @param ene_E     Electric field energy
     * @param ene_b     Magnetic field energy
     * @param m         Mode
     */
    void get_energy( cyl_double3 & ene_E, cyl_double3 & ene_B, const int m ) const;

};

