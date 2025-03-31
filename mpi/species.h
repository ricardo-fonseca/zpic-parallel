#ifndef SPECIES_H_
#define SPECIES_H_

#include <string>

#include "zpic.h"

#include "particles.h"

#include "emf.h"
#include "current.h"

#include "density.h"
#include "moving_window.h"
#include "udist.h"

namespace phasespace {
    enum quant { x, y, ux, uy, uz };

    static inline void qinfo( quant q, std::string & name, std::string & label, std::string & units ) {
        switch(q) {
        case x :
            name = "x"; label = "x"; units = "c/\\omega_n";
            break;
        case y :
            name = "y"; label = "y"; units = "c/\\omega_n";
            break;
        case ux :
            name = "ux"; label = "u_x"; units = "c";
            break;
        case uy :
            name = "uy"; label = "u_y"; units = "c";
            break;
        case uz :
            name = "uz"; label = "u_y"; units = "c";
            break;
        }
    }
}

namespace species {
    enum pusher { boris, euler };
    namespace bc {
        enum type { open = 0, periodic, reflecting };
    }
    typedef bnd<bc::type> bc_type;

}

/**
 * @brief Charged particles class
 * 
 */
class Species {

protected:

     /// @brief Unique species identifier
    int id;

    /// @brief Nunber of particles per cell
    uint2 ppc;

    /// @brief reference particle charge
    float q;

    /// @brief Cell dize
    float2 dx;

    /// @brief Simulation box size
    float2 box;

    /// @brief Time step
    float dt;

     /// @brief Iteration
    int iter;

     /// @brief Particle data buffer
    Particles *particles;

    /// @brief Secondary data buffer to speed up some calculations
    Particles *tmp;

    /// @brief Particle tile sort aux. data
    ParticleSort *sort;

    /// @brief Initial density profile
    Density::Profile * density;

    /// @brief Number of particles being injected
    int * np_inj;

    /**
     * @brief Process (physical) boundary conditions
     * 
     */
    void process_bc();

private:

    /// @brief Boundary condition
    species::bc_type bc;

     /// @brief Moving window information
    MovingWindow moving_window;

    /// @brief Initial velocity distribution
    UDistribution::Type * udist;

    /// @brief Total species energy on device
    double d_energy;

    /// @brief Total number of particles moved
    uint64_t d_nmove;

    /**
     * @brief Shift particle positions due to moving window motion
     * 
     */
    void move_window_shift();

    /**
     * @brief Inject new particles due to moving window motion
     * 
     */
    void move_window_inject();

    /**
     * @brief Deposit 1D phasespace density
     * 
     * @param d_data    Data buffer
     * @param q         Quantity for axis
     * @param range     Value range
     * @param size      Number of grid points
     */
    void dep_phasespace( float * const d_data, 
        phasespace::quant q, float2 const range, unsigned const size ) const;

    /**
     * @brief Deposit 2D phasespace density
     * 
     * @param d_data    Data buffer
     * @param quant0    axis 0 quantity
     * @param range0    axis 0 value range
     * @param size0     axis 0 number of points
     * @param quant1    axis 1 quantity
     * @param range1    axis 1 value range
     * @param size1     axis 1 number of points
     */
    void dep_phasespace( float * const d_data,
        phasespace::quant quant0, float2 range0, unsigned const size0,
        phasespace::quant quant1, float2 range1, unsigned const size1 ) const;

public:

     /// @brief Species name
    const std::string name;

     /// @brief  Mass over charge ratio
    const float m_q;

    /// @brief Type of particle pusher to use
    species::pusher push_type;

    /**
     * @brief Construct a new Species object
     * 
     * @param name  Name for the species object (used for diagnostics)
     * @param m_q   Mass over charge ratio
     * @param ppc   Number of particles per cell
     */
    Species( std::string const name, float const m_q, uint2 const ppc );

    /**
     * @brief Initialize data structures
     * 
     * @param box       Simulation global box size
     * @param ntiles    Number of tiles
     * @param nx        Tile grid dimension
     * @param dt        
     * @param id 
     */
    virtual void initialize( float2 const box, uint2 const ntiles, uint2 const nx,
        float const dt, int const id_, Partition & parallel );

    /**
     * @brief Destroy the Species object
     * 
     */
    ~Species();

    /**
     * @brief Set the density profile object
     * 
     * @param new_density   New density object to be cloned
     */
    virtual void set_density( Density::Profile const & new_density ) {
        delete density;
        density = new_density.clone();
    }

    /**
     * @brief Get the density object
     * 
     * @return Density::Profile& 
     */
    Density::Profile & get_density() {
        return * density;
    }

    /**
     * @brief Set the velocity distribution object
     * 
     * @param new_udist     New udist object to be cloned
     */
    virtual void set_udist( UDistribution::Type const & new_udist ) {
        delete udist;
        udist = new_udist.clone();
    }

    /**
     * @brief Get the udist object
     * 
     * @return UDistribution::Type& 
     */
    UDistribution::Type & get_udist() {
        return *udist;
    } 

    /**
     * @brief Sets the boundary condition type
     * 
     * @param new_bc 
     */
    void set_bc( species::bc_type new_bc ) {

        // Validate parameters
        if ( (new_bc.x.lower == species::bc::periodic) || (new_bc.x.upper == species::bc::periodic) ) {
            if ( new_bc.x.lower != new_bc.x.upper ) {
                std::cerr << "(*error*) Species boundary type mismatch along x.\n";
                std::cerr << "(*error*) When choosing periodic boundaries both lower and upper types must be set to species::bc::periodic.\n";
                exit(1);
            }
        }

        if ( (new_bc.y.lower == species::bc::periodic) || (new_bc.y.upper == species::bc::periodic) ) {
            if ( new_bc.y.lower != new_bc.y.upper ) {
                std::cerr << "(*error*) Species boundary type mismatch along y.\n";
                std::cerr << "(*error*) When choosing periodic boundaries both lower and upper types must be set to species::bc::periodic.\n";
                exit(1);
            }
        }

        // Store new values
        bc = new_bc;

/*
        std::string bc_name[] = {"open", "periodic", "reflecting"};
        std::cout << "(*info*) Species " << name << " boundary conditions\n";
        std::cout << "(*info*) x : [ " << bc_name[ bc.x.lower ] << ", " << bc_name[ bc.x.upper ] << " ]\n";
        std::cout << "(*info*) y : [ " << bc_name[ bc.y.lower ] << ", " << bc_name[ bc.y.upper ] << " ]\n";
*/

        // Set periodic flags on tile grids
        if ( particles ) {
            particles -> set_periodic( make_int2( 
                bc.x.lower == species::bc::periodic,
                bc.y.lower == species::bc::periodic
            ));
        }
    }

    /**
     * @brief Get the current boundary condition types
     * 
     * @return species::bc_type 
     */
    species::bc_type get_bc( ) { return bc; }


    /**
     * @brief Sets moving window algorithm
     * 
     * This method can only be called before the simulation has started (iter = 0)
     * 
     * @return int  0 on success, -1 on error
     */
    virtual int set_moving_window() { 
        if ( iter == 0 ) {
            moving_window.init( dx.x );

            // Set global open boundary conditions
            bc.x.lower = bc.x.upper = species::bc::open;

            // Disable periodic.x boundaries for particles object
            auto periodic = particles -> get_periodic();
            periodic.x = false;
            particles -> set_periodic( periodic );

            return 0;
        } else {
            std::cerr << "(*error*) set_moving_window() called with iter != 0\n";
            return -1; 
        }
    }

    /**
     * @brief Inject particles in the simulation box
     * 
     */
    virtual void inject();

    /**
     * @brief Inject particles in the specified range of the simulation
     * 
     * @param range     Range in which to inject particles
     */
    virtual void inject( bnd<unsigned int> range );


    /**
     * @brief Gets number of particles that will be injected per tile
     * 
     * @param range     Range in which to inject particles
     * @param np        Device pointer to number of particles to be injected per tile
     */
    virtual void np_inject( bnd<unsigned int> range, int * np );

    /**
     * @brief Advance particle velocities
     * 
     * @param E     Electric field
     * @param B     Magnetic field
     */
    void push( vec3grid<float3> * const E, vec3grid<float3> * const B );

    /**
     * @brief Move particles (advance positions) and deposit current
     * 
     * @param current   Electric current density
     */
    void move( vec3grid<float3> * const current );

    /**
     * @brief Move particles (advance positions), deposit current and shift positions
     * 
     * @param current   Electric current density
     */
    void move( vec3grid<float3> * const current, int2 const shift );

    /**
     * @brief Move particles (advance positions) without depositing current
     * 
     */
    void move( );

    /**
     * @brief Free stream particles 1 timestep
     * 
     * Particles are free-streamed (no momentum update), no current is deposited.
     * Used mostly for debug purposes.
     * 
     */
    virtual void advance();

    /**
     * @brief Free stream particles 1 timestep
     * 
     * Particles are free-streamed (no momentum update) and current is deposited
     * 
     * @param current   Electric current density
     */
    virtual void advance( Current &current );

    /**
     * @brief Advance particles 1 timestep
     * 
     * Momentum is advanced from EMF fields and current is deposited
     * 
     * @param emf       EM fields
     * @param current   Electric current density
     */
    virtual void advance( EMF const &emf, Current &current );

    /**
     * @brief Advance particles 1 timestep and update moving window
     * 
     * @param emf       EM fields
     * @param current   Electric current density
     */
    virtual void advance_mov_window( EMF const &emf, Current &current );

    /**
     * @brief Deposit species charge
     * 
     * @param charge    Charge density grid
     */
    void deposit_charge( grid<float> &charge ) const;

    /**
     * @brief Returns total time centered kinetic energy
     * 
     * @return double 
     */
    double get_energy() const {
        // Normalize and return
        return d_energy * q * m_q * dx.x * dx.y;
    }

    /**
     * @brief Returns total number of particles moved
     * 
     * @return uint64_t
     */
    auto get_nmove() const {

        return d_nmove;
    }

    /**
     * @brief Gets the number of iterations
     * 
     * @return auto 
     */
    auto get_iter() const {
        return iter;
    }

    /**
     * @brief Returns the maximum number of particles per tile
     * 
     * @return auto 
     */
    uint32_t np_max_tile() const {
        return particles -> np_max_tile();
    }

    /**
     * @brief Returns the (node) local number of particles
     * 
     * @return uint64_t     Local number of particles
     */
    uint64_t np_local() const {
        return particles -> np_local();
    }

    /**
     * @brief Gets global number of particles
     * @note By default, the correct result is only returned on root node
     * 
     * @param all           Return result on all parallel nodes (defaults to false)
     * @return uint64_t     Global number of particles
     */
    uint64_t np_global( bool all = false ) {
        return particles -> np_global( all );
    }

    /**
     * @brief Save particle data to file
     * 
     * Saves positions and velocities for all particles. Positions are currently
     * normalized to cell size
     */
    void save() const;

    /**
     * @brief Save charge density for species to file
     * 
     */
    void save_charge() const;

    /**
     * @brief Save 1D phasespace density to file
     * 
     * @param quant     Phasespace quantity
     * @param range     Value range
     * @param size      Number of grid points
     */
    void save_phasespace ( 
        phasespace::quant quant, float2 const range, int const size ) const;

    /**
     * @brief Save 2D phasespace density to file
     * 
     * @param quant0    axis 0 quantity
     * @param range0    axis 0 value range
     * @param size0     axis 0 number of points
     * @param quant1    axis 1 quantity
     * @param range1    axis 1 value range
     * @param size1     axis 1 number of points
     */
    void save_phasespace ( 
        phasespace::quant quant0, float2 const range0, int const size0,
        phasespace::quant quant1, float2 const range1, int const size1 ) const;


    void info_np() {
        particles->info_np();
    }
};


#endif