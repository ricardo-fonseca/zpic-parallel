#ifndef CURRENT_H_
#define CURRENT_H_

#include "zpic.h"

#include "vec3grid.h"
#include "moving_window.h"
#include "filter.h"

namespace current {
    namespace bc {
        enum type { none = 0, periodic, reflecting };
    }
    typedef bnd<bc::type> bc_type;
}

class Current {

    /// @brief Global simulation box size
    float2 box;

    /// @brief cell size
    float2 dx;
    
    /// @brief time step
    float dt;

    /// @brief Moving window information
    MovingWindow moving_window;

    /// @brief Boundary condition
    current::bc_type bc;

    /// @brief Iteration number
    int iter;

    /**
     * @brief Process boundary conditions
     * 
     */
    void process_bc();

    public:

    /// @brief Current density
    vec3grid<float3> * J = nullptr;

    /// @brief Filtering parameters
    Filter::Digital *filter = nullptr;

    /**
     * @brief Construct a new Current object
     * 
     * @param global_ntiles     Global number of tiles
     * @param nx                Individual tile size
     * @param box               Global simulation box size
     * @param dt                Time step
     * @param parallel          Parallel partition 
     */
    Current( uint2 const global_ntiles, uint2 const nx, float2 const box, float const dt, Partition & parallel );
    
    /**
     * @brief Destroy the Current object
     * 
     */
    ~Current() {
        delete (J);
        delete (filter);
    }

    friend std::ostream& operator<<(std::ostream& os, const Current obj) {
        os << "Current object\n";
        return os;
    }

    /**
     * @brief Get the type of boundary conditions
     * 
     * @return current::bc_type
     */
    current::bc_type get_bc( ) { return bc; }

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
                std::cerr << "(*error*) When choosing periodic boundaries both lower and upper types must be set to current::bc::periodic.\n";
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

        if ( mpi::world_root() ) {
            std::string bc_name[] = {"none", "periodic", "reflecting"};
            std::cout << "(*info*) Current boundary conditions\n";
            std::cout << "(*info*) x : [ " << bc_name[ bc.x.lower ] << ", " << bc_name[ bc.x.upper ] << " ]\n";
            std::cout << "(*info*) y : [ " << bc_name[ bc.y.lower ] << ", " << bc_name[ bc.y.upper ] << " ]\n";
        }
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
            if ( J -> part.periodic.x ) {
                std::cerr << "(*error*) Unable to set_moving_window() with periodic x partition\n";
                return -1; 
            }
            
            moving_window.init( dx.x );
            bc.x.lower = bc.x.upper = current::bc::none;
            return 0;
        } else {
            std::cerr << "(*error*) set_moving_window() called with iter != 0\n";
            return -1; 
        }
    }

    /**
     * @brief Sets the digital filter
     *
     * @param new_filter    New filter to be used
     */
    void set_filter( Filter::Digital const & new_filter ) {
        delete filter;
        filter = new_filter.clone();
    }

    /**
     * @brief Advance electric current 1 iteration
     * 
     * @note Adds up current deposited on guard cells and (optionally) applies digital filtering
     * 
     */
    void advance();

    /**
     * @brief Zero electric current values
     * 
     */
    void zero();

    /**
     * @brief Save electric current data to diagnostic file
     * 
     * @param jc    Current component to save
     */
    void save( fcomp::cart const jc );
};


#endif