#ifndef EMF_H_
#define EMF_H_

#include "zpic.h"
#include "utils.h"

#include "bnd.h"
#include "current.h"
#include "moving_window.h"

#include "cyl3modes.h"

#include <string>

namespace emf {
    enum field  { e, b };

    namespace bc {
        enum type { none = 0, axial, periodic, pec, pmc };
    }

    typedef bnd<bc::type> bc_type;
}

class EMF {

    /// @brief Boundary condition
    emf::bc_type bc;

    /// @brief cell size
    const float2 dx;

    /// @brief time step
    const double dt;

    /// @brief Iteration number
    int iter;

    /// @brief Moving window information
    MovingWindow moving_window;

    /// @brief Device buffer for field energy calculations
    double * d_energy;

    /**
     * @brief Move simulation window if needed
     * 
     */
    void move_window( );

    /**
     * @brief Process boundary conditions
     * 
     */
    // void process_bc( );

    public:

    /// @brief Number of cylindrical modes (including fundamental mode)
    const int nmodes;
    /// @brief Simulation box size
    const float2 box;
    /// @brief Electric field
    Cyl3CylGrid<float> * E;
    /// @brief Magnetic field
    Cyl3CylGrid<float> * B;

    /**
     * @brief Construct a new EMF object
     * 
     * @param nmodes    Number of cylindrical modes (>= 1)
     * @param ntiles    Number of tiles in x,y direction
     * @param nx        Tile size (#cells)
     * @param box       Simulation box size (sim. units)
     * @param dt        Time step
     */
    EMF( int nmodes, uint2 const ntiles, uint2 const nx, float2 const box, double const dt );
    
    /**
     * @brief Destroy the EMF object
     * 
     */
    ~EMF() {
        device::free( d_energy );
        delete (E);
        delete (B);
    }

    friend std::ostream& operator<<(std::ostream& os, const EMF & obj) {
        os << "EMF object";
        return os;
    }

    /**
     * @brief Get the iteration number
     * 
     * @return auto 
     */
    auto get_iter() { return iter; }

    /**
     * @brief Get the time step (dt) value
     * 
     * @return auto 
     */
    auto get_dt() { return dt; }

    /**
     * @brief Get the boundary condition type
     * 
     * @return emf::bc_type 
     */
    auto get_bc( ) { return bc; }

    /**
     * @brief Set the boundary condition type
     * 
     * @param new_bc    New boundary condition types (all directions)
     */
    void set_bc( emf::bc_type new_bc ) {

        if ( new_bc.y.lower != emf::bc::axial ) {
            std::cerr <<  "(*warning*) Invalid EMF axial boundary, overriding value\n";
            new_bc.y.lower = emf::bc::axial;
        }

        if ( new_bc.y.upper == emf::bc::periodic ) {
            std::cerr << "(*error*) Invalid EMF boundary along r.\n";
            std::cerr << "(*error*) Periodic boundaries are not allowed in the radial direction.\n";
            std::exit(1);
        }

        if ( (new_bc.x.lower == emf::bc::periodic) || (new_bc.x.upper == emf::bc::periodic) ) {
            if ( new_bc.x.lower != new_bc.x.upper ) {
                std::cerr << "(*error*) EMF boundary type mismatch along x.\n";
                std::cerr << "(*error*) When choosing periodic boundaries both lower and upper types must be set to emf::bc::periodic.\n";
                std::exit(1);
            }
        }

        // Store new values
        bc = new_bc;

        std::string bc_name[] = { "none", "axial", "periodic", "pec", "pmc"};
        std::cout << "(*info*) EMF boundary conditions\n";
        std::cout << "(*info*) x : [ " << bc_name[ bc.x.lower ] << ", " << bc_name[ bc.x.upper ] << " ]\n";
        std::cout << "(*info*) y : [ " << bc_name[ bc.y.lower ] << ", " << bc_name[ bc.y.upper ] << " ]\n";

        // Set periodic flags on tile grids
        E -> set_periodic( bc.x.lower == emf::bc::periodic );
        B -> set_periodic( bc.x.lower == emf::bc::periodic );
    }

    /**
     * @brief Sets moving window algorithm
     * 
     * This method can only be called before the simulation has started (iter = 0)
     * 
     * @return int  0 on success, -1 on error
     */
    int set_moving_window() { 
        if ( iter == 0 ) {
            moving_window.init( dx.x );

            bc.x.lower = bc.x.upper = emf::bc::none;
 
            E -> set_periodic( false );
            B -> set_periodic( false );

            return 0;
        } else {
            std::cerr << "(*error*) set_moving_window() called with iter != 0\n";
            return -1; 
        }
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
    void advance( Current & current );

    /**
     * @brief Save EM field component to file
     * 
     * @param field     Which field to save (E or B)
     * @param fc        Which field component to save (r, θ or z)
     * @param m         Mode
     */
    void save( emf::field const field, const fcomp::cyl fc, const int m );
    
    /**
     * @brief Get EM field energy
     * 
     * 
     * @note The energy will be recalculated each time this routine is called
     * 
     * @param ene_E     Electric field energy
     * @param ene_b     Magnetic field energy
     * @param m         Mode
     */
    void get_energy( cyl_double3 & ene_E, cyl_double3 & ene_B, const int m );
};

#endif
