#pragma once

#include "core/vec_types.hpp"
#include "core/bounds.hpp"

#include "grid/tiled_vec3.hpp"
#include "util/error.hpp"
#include "moving_window.hpp"
#include "current.hpp"

#include <cmath>
#include <ostream>
#include <sstream>
#include <string>

namespace em2d {

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

        /**
         * @brief Name of boundary condition type
         * 
         * @param t                 Boundary condition type
         * @return std::string      Name of boundary condition type
         */
        static std::string name( const type t ) {
            switch( t ) {
                case none:      return "none";
                case periodic:  return "periodic";
                case pec:       return "pec";
                case pmc:       return "pmc";
            }
            return "-";
        }
    };
    using bc_type = bounds_2d<bc::type>;

    /**
    * @brief CFL time limit from cell size
    * 
    * @param dx        Cell size
    * @return float    CFL time limit
    */
    static float courant( const float2 dx ) {
        return std::sqrt( 1.0f/( 1.0f/(dx.x*dx.x) + 1.0f/(dx.y*dx.y) ) );
    }

    private:

    /// @brief Boundary condition
    emf::bc_type bc;

    /// @brief cell size
    const float2 dx;

    /// @brief time step
    const double dt;

    /// @brief Iteration number
    int iter = 0;

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
     * @brief Validate boundary conditions against the grid periodicity,
     *        moving window and tile size
     * 
     * @note Calls util::fatal() on invalid combinations
     * 
     * @param test_bc   Boundary conditions to validate
     */
    void validate_bc( const emf::bc_type & test_bc ) const {

        // Periodic boundaries must be set on both sides
        if ( (test_bc.x.lower == emf::bc::periodic) || (test_bc.x.upper == emf::bc::periodic) ) {
            if ( test_bc.x.lower != test_bc.x.upper ) {
                util::fatal( "EMF boundary type mismatch along x."
                            " When choosing periodic boundaries both lower and upper types"
                            " must be set to emf::bc::periodic." );
            }
        }

        if ( (test_bc.y.lower == emf::bc::periodic) || (test_bc.y.upper == emf::bc::periodic) ) {
            if ( test_bc.y.lower != test_bc.y.upper ) {
                util::fatal( "EMF boundary type mismatch along y."
                            " When choosing periodic boundaries both lower and upper types"
                            " must be set to emf::bc::periodic." );
            }
        }

        // Boundary periodicity must match grid periodicity (both ways)
        const auto periodic = E -> get_grid_periodic();

        if ( periodic.x && test_bc.x.lower != emf::bc::periodic ) {
            util::fatal( "Only periodic x boundaries are supported with periodic x grids.");
        }

        if ( periodic.y && test_bc.y.lower != emf::bc::periodic ) {
            util::fatal( "Only periodic y boundaries are supported with periodic y grids.");
        }

        if ( ! periodic.x && test_bc.x.lower == emf::bc::periodic ) {
            util::fatal( "Periodic x boundaries require a periodic x grid.");
        }

        if ( ! periodic.y && test_bc.y.lower == emf::bc::periodic ) {
            util::fatal( "Periodic y boundaries require a periodic y grid.");
        }

        // The moving window requires open x boundaries
        if ( mov_window.active() &&
             ( test_bc.x.lower != emf::bc::none || test_bc.x.upper != emf::bc::none ) ) {
            util::fatal( "With a moving window both x boundaries must be emf::bc::none." );
        }

        // Wall boundaries touch cells -1..1 at the lower edge and nx-2..nx+1
        // at the upper edge; with tiles narrower than 3 cells the lower and
        // upper edge blocks of the bc kernels would overlap when ntiles == 1
        if ( ( is_conducting( test_bc.x.lower ) || is_conducting( test_bc.x.upper ) ) && E -> tile_dims.x < 3 ) {
            util::fatal( "PEC / PMC x boundaries require tile_dims.x >= 3." );
        }

        if ( ( is_conducting( test_bc.y.lower ) || is_conducting( test_bc.y.upper ) ) && E -> tile_dims.y < 3 ) {
            util::fatal( "PEC / PMC y boundaries require tile_dims.y >= 3." );
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
     * @param ntiles            Number of tiles
     * @param tile_dims         Individual tile size
     * @param box               Simulation box size
     * @param dt                Time step
     * @param periodic          Periodic boundaries (x,y)
     */
    emf( uint2 const ntiles, uint2 const tile_dims, float2 const box,
        double const dt, int2 const periodic ) : 
        dx( make_float2( box.x / ( tile_dims.x * ntiles.x ), box.y / ( tile_dims.y * ntiles.y ) ) ),
        dt( dt ), box(box)
    {
        // Verify Courant condition
        auto cour = courant(dx);
        if ( dt >=  cour){
            std::ostringstream msg;
            msg << "Invalid timestep, courant condition violation."
                << " For the current resolution " << dx
                << " the maximum timestep is dt = " << cour;
            util::fatal( msg.str() );
        }

        // Guard cells (1 below, 2 above)
        // These are required for the Yee solver AND for field interpolation
        bounds_2d<unsigned int> gc;
        gc.x = {1,2};
        gc.y = {1,2};

        E = new grid::tiled_vec3<float> ( ntiles, tile_dims, gc, periodic );
        E -> name = "Electric field";

        B = new grid::tiled_vec3<float> ( ntiles, tile_dims, gc, periodic );
        B -> name = "Magnetic field";

         // Zero fields
        E -> zero();
        B -> zero();

        // Default boundary conditions follow the grid periodicity:
        // periodic where the grid is periodic, none otherwise
        emf::bc_type default_bc = emf::bc_type (emf::bc::periodic);
        if ( ! periodic.x ) default_bc.x.lower = default_bc.x.upper = emf::bc::none;
        if ( ! periodic.y ) default_bc.y.lower = default_bc.y.upper = emf::bc::none;
        set_bc( default_bc );
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
     * @brief Get the type of boundary conditions
     * 
     * @return emf::bc_type 
     */
    emf::bc_type get_bc( ) const noexcept { return bc; }

    /**
     * @brief Set the boundary conditions
     * 
     * @param new_bc    New boundary condition values
     */
    void set_bc( emf::bc_type new_bc ) {

        // Validate parameters
        validate_bc( new_bc );

        // Store new values
        bc = new_bc;
    }

    /**
     * @brief Turns on the moving window algorithm
     * 
     * Sets both x boundaries to emf::bc::none. Can only be called before the
     * simulation has started (iter = 0); calls util::fatal() otherwise or if
     * the x boundaries are periodic.
     */
    void set_moving_window() { 
        if ( iter != 0 ) {
            util::fatal( "set_moving_window() called with iter != 0" );
        }

        if ( bc.x.lower == emf::bc::periodic ) {
            util::fatal( "Unable to set_moving_window() with periodic x boundaries" );
        }

        mov_window.init( dx.x );

        // Correct boundary conditions
        auto new_bc = bc;
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
     * @brief Get local EM field energy
     * 
     * @note The energy is recalculated each time this routine is called
     * 
     * @param ene_E     Electric field energy (per component)
     * @param ene_B     Magnetic field energy (per component)
     */
    void get_energy( double3 & ene_E, double3 & ene_B ) const;
};

} // namespace em2d
