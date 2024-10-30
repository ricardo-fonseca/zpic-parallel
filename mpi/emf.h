#ifndef EMF_H_
#define EMF_H_

#include "zpic.h"
#include "utils.h"

#include "bnd.h"
#include "vec3grid.h"
#include "current.h"
#include "moving_window.h"

#include <string>

namespace emf {
    enum field  { e, b };

    namespace bc {
        enum type { none = 0, periodic, pec, pmc };
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

    /**
     * @brief Move simulation window if needed
     * 
     */
    void move_window();

    /**
     * @brief Process boundary conditions
     * 
     */
    void process_bc();

    public:

    /// @brief Electric field
    vec3grid<float3> * E;
    /// @brief Magnetic field
    vec3grid<float3> * B;
    /// @brief Simulation box size
    const float2 box;

    /**
     * @brief Construct a new EMF object
     * 
     * @param ntiles    Number of tiles in x,y direction
     * @param nx        Tile size (#cells)
     * @param box       Simulation box size (sim. units)
     * @param dt        Time step
     */
    EMF( uint2 const ntiles, uint2 const nx, float2 const box, double const dt, Partition & part );
    
    /**
     * @brief Destroy the EMF object
     * 
     */
    ~EMF() {
        delete (E);
        delete (B);
    }

    friend std::ostream& operator<<(std::ostream& os, const EMF obj) {
        os << "EMF object\n";
        return os;
    }

    /**
     * @brief Get the iteration number
     * 
     * @return auto 
     */
    auto get_iter() { return iter; }

    /**
     * @brief Get the boundary condition values
     * 
     * @return emf::bc_type 
     */
    emf::bc_type get_bc( ) { return bc; }

    void set_bc( emf::bc_type new_bc ) {

        // Validate parameters
        if ( (new_bc.x.lower == emf::bc::periodic) || (new_bc.x.upper == emf::bc::periodic) ) {
            if ( new_bc.x.lower != new_bc.x.upper ) {
                std::cerr << "(*error*) EMF boundary type mismatch along x.\n";
                std::cerr << "(*error*) When choosing periodic boundaries both lower and upper types must be set to emf::bc::periodic.\n";
                exit(1);
            }
        }

        if ( (new_bc.y.lower == emf::bc::periodic) || (new_bc.y.upper == emf::bc::periodic) ) {
            if ( new_bc.y.lower != new_bc.y.upper ) {
                std::cerr << "(*error*) EMF boundary type mismatch along y.\n";
                std::cerr << "(*error*) When choosing periodic boundaries both lower and upper types must be set to emf::bc::periodic.\n";
                exit(1);
            }
        }

        if ( E -> part.periodic.x && new_bc.x.lower != emf::bc::periodic ) {
            std::cerr << "(*error*) Only periodic x boundaries are supported with periodic x parallel partitions.\n";
            exit(1);
        }

        if ( E -> part.periodic.y && new_bc.y.lower != emf::bc::periodic ) {
            std::cerr << "(*error*) Only periodic y boundaries are supported with periodic y parallel partitions.\n";
            exit(1);
        }

        // Store new values
        bc = new_bc;

        std::string bc_name[] = {"none", "periodic", "pec", "pmc"};
        std::cout << "(*info*) EMF boundary conditions\n";
        std::cout << "(*info*) x : [ " << bc_name[ bc.x.lower ] << ", " << bc_name[ bc.x.upper ] << " ]\n";
        std::cout << "(*info*) y : [ " << bc_name[ bc.y.lower ] << ", " << bc_name[ bc.y.upper ] << " ]\n";
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
            if ( E -> part.periodic.x ) {
                std::cerr << "(*error*) Unable to set_moving_window() with periodic x partition\n";
                return -1; 
            }

            moving_window.init( dx.x );
            bc.x.lower = bc.x.upper = emf::bc::none;
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
     * @param fc        Which field component to save (x, y or z)
     */
    void save( emf::field const field, const fcomp::cart fc );
    
    /**
     * @brief Get EM field energy
     * 
     * @note The energy will be recalculated each time this routine is called
     * 
     * @param ene_E     Electric field energy
     * @param ene_b     Magnetic field energy
     */
    void get_energy( double3 & ene_E, double3 & ene_b );

};

#endif
