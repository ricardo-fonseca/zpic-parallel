#pragma once

#include <string>

#include "part/particles.hpp"

#include "emf.hpp"
#include "current.hpp"
#include "charge.hpp"

#include "density.hpp"
#include "udist.hpp"

namespace phasespace {
    enum class quantity { x, y, ux, uy, uz };

    static inline void qinfo( quantity q, std::string & name, std::string & label, std::string & units ) {
        switch(q) {
        case quantity::x :
            name = "x"; label = "x"; units = "c/\\omega_n";
            break;
        case quantity::y :
            name = "y"; label = "y"; units = "c/\\omega_n";
            break;
        case quantity::ux :
            name = "ux"; label = "u_x"; units = "c";
            break;
        case quantity::uy :
            name = "uy"; label = "u_y"; units = "c";
            break;
        case quantity::uz :
            name = "uz"; label = "u_z"; units = "c";
            break;
        }
    }
}

/**
 * @brief Charged particles class
 * 
 */
class species {

public:

    enum class pusher{ boris = 0, euler };
    struct bc { enum type { open = 0, periodic, reflecting }; };
    using bc_type = bounds_2d<bc::type>;

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
    part::particles *particles;

    /// @brief Secondary data buffer to speed up some calculations
    part::particles *tmp;

    /// @brief Particle tile sort aux. data
    part::particle_sort *sort;

    /// @brief Initial density profile
    density::profile * density;

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

    /// @brief Initial velocity distribution
    udist::type * udist;

    /// @brief Total species energy on device
    double d_energy;

    /// @brief Total number of particles moved
    uint64_t d_nmove;

    /**
     * @brief Deposit 1D phasespace density
     * 
     * @param d_data    Data buffer
     * @param q         Quantity for axis
     * @param range     Value range
     * @param size      Number of grid points
     */
    void dep_phasespace( float * const d_data, 
        const phasespace::quantity q, float2 range, const int size ) const;

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
        const phasespace::quantity quant0, float2 range0, const int size0,
        const phasespace::quantity quant1, float2 range1, const int size1 ) const;

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
    species( std::string const name, float const m_q, uint2 const ppc );

    /**
     * @brief Copy constructor
     * 
     */
    species( const species & ) = delete;
    
    /**
     * @brief Move constructor
     * 
     */
    species( species && ) = delete;

    /**
     * @brief Initialize data structures and inject initial particle distribution
     * 
     * @param box               Global simulation box size
     * @param global_ntiles     Global number of tiles
     * @param tile_dims         Individutal tile grid size
     * @param dt                Time step
     * @param id                Species unique identifier
     * @param parallel          Parallel configuration
     */
    virtual void initialize( float2 const box_, uint2 const global_ntiles, uint2 const tile_dims,
    float const dt_, int const id_, mpi::cart2d & parallel );

    /**
     * @brief Destroy the Species object
     * 
     */
    virtual ~species();

    /**
     * @brief Set the density profile object
     * 
     * @param new_density   New density object to be cloned
     */
    virtual void set_density( density::profile const & new_density ) {
        delete density;
        density = new_density.clone();

        // Recompute charge normalization factor in case initialize()
        // has already been called
        q = copysign( density->n0 , m_q ) / (ppc.x * ppc.y);
    }

    /**
     * @brief Get the density object
     * 
     * @return density::profile& 
     */
    density::profile & get_density() {
        return * density;
    }

    /**
     * @brief Set the velocity distribution object
     * 
     * @param new_udist     New udist object to be cloned
     */
    virtual void set_udist( udist::type const & new_udist ) {
        delete udist;
        udist = new_udist.clone();
    }

    /**
     * @brief Get the udist object
     * 
     * @return UDistribution::Type& 
     */
    udist::type & get_udist() {
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
                mpi::fatal( "Species boundary type mismatch along x."
                            " When choosing periodic boundaries both lower and upper types"
                            " must be set to species::bc::periodic." );
            }
        }

        if ( (new_bc.y.lower == species::bc::periodic) || (new_bc.y.upper == species::bc::periodic) ) {
            if ( new_bc.y.lower != new_bc.y.upper ) {
                mpi::fatal( "Species boundary type mismatch along y."
                            " When choosing periodic boundaries both lower and upper types"
                            " must be set to species::bc::periodic." );
            }
        }

        // Only periodic and open boundaries are currently implemented
        if (( new_bc.x.lower == species::bc::reflecting ) ||
            ( new_bc.x.upper == species::bc::reflecting ) ||
            ( new_bc.y.lower == species::bc::reflecting ) ||
            ( new_bc.y.upper == species::bc::reflecting ) ) {
            
            mpi::fatal( "Reflecting boundaries for species is not yet implemented");
        }

        // Store new values
        bc = new_bc;

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
     * @brief Inject particles in the simulation box
     * 
     */
    virtual void inject();

    /**
     * @brief Inject particles in the specified range of the simulation
     * 
     * @param range     Range in which to inject particles
     */
    virtual void inject( bounds_2d<unsigned int> range );


    /**
     * @brief Gets number of particles that will be injected per tile
     * 
     * @param range     Range in which to inject particles
     * @param np        Device pointer to number of particles to be injected per tile
     */
    virtual void np_inject( bounds_2d<unsigned int> range, int * np );

    /**
     * @brief Advance particle velocities
     * 
     * @param E     Electric field
     * @param B     Magnetic field
     */
    void push( grid::tiled_vec3<float> * const E, grid::tiled_vec3<float> * const B );

    /**
     * @brief Move particles (advance positions) and deposit current/charge
     * 
     * @param current   Electric current density
     */
    void move( grid::tiled_vec3<float> * const current, grid::tiled<float> * charge );

    /**
     * @brief Move particles (advance positions) without depositing current
     * 
     */
    void move();

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
    virtual void advance( current &current, charge &charge );

    /**
     * @brief Advance particles 1 timestep
     * 
     * Momentum is advanced from EMF fields and current is deposited
     * 
     * @param emf       EM fields
     * @param current   Electric current density
     */
    virtual void advance( emf const &emf, current &current, charge &charge );

    /**
     * @brief Deposit species charge
     * 
     * @param charge    Charge density grid
     */
    void deposit_charge( grid::tiled<float> &charge ) const;

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
    uint32_t tile_np_max() const {
        return particles -> tile_np_max();
    }

    /**
     * @brief Returns the (node) local number of particles
     * 
     * @return uint64_t     Local number of particles
     */
    uint64_t local_np() const {
        return particles -> local_np();
    }

    /**
     * @brief Gets global number of particles
     * @note By default, the correct result is only returned on root node
     * 
     * @param all           Return result on all parallel nodes (defaults to false)
     * @return uint64_t     Global number of particles
     */
    uint64_t global_np( bool all = false ) {
        return particles -> global_np( all );
    }

    /**
     * @brief Save particle data to file
     * 
     * @note Saves positions and velocities for all particles in simulation units
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
        phasespace::quantity quant, float2 const range, int const size ) const;

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
        phasespace::quantity quant0, float2 const range0, int const size0,
        phasespace::quantity quant1, float2 const range1, int const size1 ) const;

    /**
     * @brief Print information on the number of particles per tile
     * 
     * @warning Used for debug purposes only
     * 
     * @param msg   (optional) Message to print before printing particle information
     */
    void info_np() {
        particles->info_np();
    }
};
