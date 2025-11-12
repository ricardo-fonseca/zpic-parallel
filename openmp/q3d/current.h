#ifndef CURRENT_H_
#define CURRENT_H_

#include "zpic.h"
#include "cyl3modes.h"
#include "moving_window.h"
#include "filter.h"

namespace current {
    namespace bc {
        enum type { none = 0, axial, periodic, reflecting };
    }
    typedef bnd<bc::type> bc_type;
}

class Current {

    private:

    /// @brief cell size
    const float2 dx;
    
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

    /// @brief Number of cylindrical modes (including fundamental mode)
    const int nmodes;
    /// @brief Simulation box size
    const float2 box;

    /// @brief Current density
    Cyl3CylGrid<float> * J;

    /// @brief Filtering parameters
    Filter::Digital *filter;

    /**
     * @brief Construct a new Current object
     * 
     * @param nmodes    Number of cylindrical modes (>= 1)
     * @param ntiles    Number of tiles in z,r direction
     * @param nx        Tile size (#cells)
     * @param box       Simulation box size (sim. units)
     * @param dt        Time step
     */
    Current( unsigned int nmodes, uint2 const ntiles, uint2 const nx, float2 const box, float const dt ):
        dx( make_float2( box.x / ( nx.x * ntiles.x ), box.y / ( nx.y * ntiles.y ) ) ),
        dt(dt),
        nmodes( nmodes ),
        box(box)
    {
        // Guard cells (1 below, 2 above)
        bnd<unsigned int> gc;
        gc.x = {1,2};
        gc.y = {1,2};

        J = new Cyl3CylGrid<float> ( nmodes, ntiles, nx, gc );
        J -> set_name( "Current" );

        // Zero initial charge
        // This is only relevant for diagnostics, current is always zeroed before deposition
        J -> zero();

        // Set default boundary conditions
        bc.x.lower = bc.x.upper = current::bc::periodic;
        bc.y.lower = current::bc::axial;
        bc.y.upper = current::bc::none;

        // Set default filtering
        filter = nullptr;

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
    }

    /**
     * @brief Get the number of modes (including mode 0)
     * 
     * @return auto 
     */
    auto get_nmodes() { return J -> get_nmodes(); };

    /**
     * @brief Get mode 0 cyl. grid (real)
     * 
     * @return auto& 
     */
    auto & mode0() {
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

    /**
     * @brief Get the boundary conditions
     * 
     * @return current::bc_type 
     */
    current::bc_type get_bc( ) { return bc; }

    /**
     * @brief Set the boundary conditions
     * 
     * @param new_bc    New boundary condition types (all directions)
     */
    void set_bc( current::bc_type new_bc ) {

        if ( new_bc.y.lower != current::bc::axial ) {
            std::cerr <<  "(*warning*) Invalid Current axial boundary, overriding value\n";
            new_bc.y.lower = current::bc::axial;
        }

        if ( new_bc.y.upper == current::bc::periodic ) {
            std::cerr << "(*error*) Invalid Current boundary along r.\n";
            std::cerr << "(*error*) Periodic boundaries are not allowed in the radial direction.\n";
            std::exit(1);
        }

        // Validate parameters
        if ( (new_bc.x.lower == current::bc::periodic) || (new_bc.x.upper == current::bc::periodic) ) {
            if ( new_bc.x.lower != new_bc.x.upper ) {
                std::cerr << "(*error*) Current boundary type mismatch along x.\n";
                std::cerr << "(*error*) When choosing periodic boundaries both lower and upper types must be set to current::bc::periodic.\n";
                std::exit(1);
            }
        }

        // Store new values
        bc = new_bc;

        std::string bc_name[] = {"none", "axial", "periodic", "reflecting"};
        std::cout << "(*info*) Current boundary conditions\n";
        std::cout << "(*info*) x : [ " << bc_name[ bc.x.lower ] << ", " << bc_name[ bc.x.upper ] << " ]\n";
        std::cout << "(*info*) y : [ " << bc_name[ bc.y.lower ] << ", " << bc_name[ bc.y.upper ] << " ]\n";

        // Set periodic flags on tile grids
        J -> set_periodic( bc.x.lower == current::bc::periodic );
    }

    /**
     * @brief Normalize grid values for ring particles
     * 
     */
    void normalize();


    /**
     * @brief Advances electric current density 1 time step
     * 
     * The routine will:
     * 1. Update the guard cells
     * 2. Process "physical" boundary conditions
     * 3. Normalize for grid values for ring particles
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
     * @brief Sets moving window algorithm
     * 
     * @note This method can only be called before the simulation has started (iter = 0)
     * 
     * @return int  0 on success, -1 on error
     */
    int set_moving_window() { 
        if ( iter == 0 ) {
            moving_window.init( dx.x );

            bc.x.lower = bc.x.upper = current::bc::none;
            J-> set_periodic( false );

            return 0;
        } else {
            std::cerr << "(*error*) set_moving_window() called with iter != 0\n";
            return -1; 
        }
    }

    /**
     * @brief Set the current filter to apply
     * 
     * @param new_filter 
     */
    void set_filter( Filter::Digital const & new_filter ) {
        delete filter;
        filter = new_filter.clone();
    }

    void apply_filter() {
        if ( filter != nullptr ) {
            auto & J0 = J -> mode0();
            filter -> apply( J0 );
            for( auto m = 1; m < nmodes; m++ ) {
                auto & Jm = J -> mode(m);
                filter -> apply( Jm );
            }
        }
    }
};


#endif