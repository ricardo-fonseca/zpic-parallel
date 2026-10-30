#pragma once

#include <string>
#include <memory>
#include <cmath>

#include "parallel/partition.hpp"
#include "part/particles.hpp"

#include "emf.hpp"
#include "current.hpp"
#include "moving_window.hpp"

#include "density.hpp"
#include "udist.hpp"

namespace phasespace {
    enum class quantity { x = 0, y, ux, uy, uz };

    inline void qinfo( quantity q, std::string & name, std::string & label, std::string & units ) {
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

    /**
     * @brief Boundary condition types
     * 
     * @note Reflecting boundaries must be used together with emf / current
     *       pec or pmc walls (the current fold assumes particle reflection),
     *       and open boundaries with emf / current `none` boundaries. This is
     *       not checked here, the simulation class must ensure it.
     */
    struct bc { enum type { open = 0, comm, periodic, reflecting }; };
    using bc_type = bounds_2d<bc::type>;

protected:

    /// @brief Unique species identifier
    int id;

    /// @brief Number of particles per cell
    uint2 ppc;

    /// @brief reference particle charge
    float q;

    /// @brief Cell size
    float2 dx;

    /// @brief Simulation box size
    float2 box;

    /// @brief Time step
    double dt;

    /// @brief Iteration
    int iter;

     /// @brief Particle data buffer
    part::particles *particles;

    /// @brief Secondary data buffer to speed up some calculations
    part::particles *tmp;

    /// @brief Particle tile sort aux. data
    part::particle_sort *sort;

    /// @brief Initial density profile
    std::unique_ptr<density::profile> density;

    /// @brief Initial velocity distribution
    std::unique_ptr<udist::type> udist;

    /// @brief Number of particles being injected
    int * np_inj;

    /**
     * @brief Process (physical) boundary conditions
     * 
     */
    void process_bc();

private:

    /// @brief Boundary condition (global values)
    species::bc_type global_bc;

    /// @brief Boundary condition (local rank values)
    species::bc_type local_bc;

    /// @brief Moving window information
    moving_window mov_window;

    /// @brief Total species energy on device
    double * d_energy;

    /// @brief Total number of particles moved
    unsigned long long * d_nmove;


    /**
     * @brief Periodic flags (x,y) corresponding to the current global_bc
     * 
     * @return int2 
     */
    int2 global_periodic() const {
        return make_int2(
            global_bc.x.lower == species::bc::periodic,
            global_bc.y.lower == species::bc::periodic
        );
    }

    /**
     * @brief Validate boundary conditions against the parallel partition
     * 
     * @note Calls mpi::fatal() on invalid combinations
     * 
     * @param test_bc   Global boundary conditions to validate
     */
    void validate_bc( const species::bc_type & test_bc, const mpi::cart2d & parallel ) const {

        if ( test_bc.x.lower == species::bc::comm || test_bc.x.upper == species::bc::comm ||
             test_bc.y.lower == species::bc::comm || test_bc.y.upper == species::bc::comm  ) {
            mpi::fatal( "invalid species boundary type, bc::comm is for internal use only");
        }

        if ( (test_bc.x.lower == species::bc::periodic) || (test_bc.x.upper == species::bc::periodic) ) {
            if ( test_bc.x.lower != test_bc.x.upper ) {
                mpi::fatal( "species boundary type mismatch along x."
                            " When choosing periodic boundaries both lower and upper types"
                            " must be set to species::bc::periodic." );
            }
        }

        if ( (test_bc.y.lower == species::bc::periodic) || (test_bc.y.upper == species::bc::periodic) ) {
            if ( test_bc.y.lower != test_bc.y.upper ) {
                mpi::fatal( "species boundary type mismatch along y."
                            " When choosing periodic boundaries both lower and upper types"
                            " must be set to species::bc::periodic." );
            }
        }

        // Test against parallel partition (both ways)
        if ( ! parallel.periodic.x && test_bc.x.lower == species::bc::periodic ) {
            mpi::fatal( "Periodic x boundaries require a periodic x parallel partition.");
        }

        if ( ! parallel.periodic.y && test_bc.y.lower == species::bc::periodic ) {
            mpi::fatal( "Periodic y boundaries require a periodic y parallel partition.");
        }

        if ( parallel.periodic.x && test_bc.x.lower != species::bc::periodic ) {
            mpi::fatal( "Only periodic x boundaries are supported with periodic x parallel partitions.");
        }

        if ( parallel.periodic.y && test_bc.y.lower != species::bc::periodic ) {
            mpi::fatal( "Only periodic y boundaries are supported with periodic y parallel partitions.");
        }

        // The moving window requires open x boundaries
        if ( mov_window.active() &&
             ( test_bc.x.lower != species::bc::open || test_bc.x.upper != species::bc::open ) ) {
            mpi::fatal( "With a moving window both x boundaries must be species::bc::open." );
        }
    }

    /**
     * @brief Update values of local_bc and other objects from global_bc
     *
     * @note Assumes global_bc have been validated
     */
    void update_bc( const mpi::cart2d & parallel ) {
        // update local rank boundary conditions
        local_bc = global_bc;
        if ( ! parallel.on_edge( coord::x, edge::lower ) ) local_bc.x.lower = species::bc::comm;
        if ( ! parallel.on_edge( coord::x, edge::upper ) ) local_bc.x.upper = species::bc::comm;
        if ( ! parallel.on_edge( coord::y, edge::lower ) ) local_bc.y.lower = species::bc::comm;
        if ( ! parallel.on_edge( coord::y, edge::upper ) ) local_bc.y.upper = species::bc::comm;

        // Update particles, tmp and sort objects, if initialized
        int2 periodic = global_periodic();
        if ( particles ) particles -> set_periodic( periodic );
        if ( tmp )  tmp -> set_periodic( periodic );
        if ( sort ) sort -> update_neighbors( parallel, periodic );
    };

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
    species( std::string const name, float const m_q, uint2 const ppc ):
    ppc(ppc), name(name), m_q(m_q)
    {

        // Validate parameters
        if ( m_q == 0 ) {
            mpi::fatal( "Invalid m_q value (0.0) must be not 0" );
        }

        if ( ppc.x < 1 || ppc.y < 1 ) {
            mpi::fatal( "Invalid ppc value (" + to_string(ppc) + 
                "), must be >= 1 in all directions" );
        }

        // Set default parameters
        density   = std::make_unique<density::uniform>( 1.0 );
        udist     = std::make_unique<udist::none>();
        // Upon initialization, open boundaries on periodic domains will be replaced
        global_bc = species::bc_type (species::bc::open);
        push_type = species::pusher::boris;

        // Nullify pointers to data structures
        particles = nullptr;
        tmp = nullptr;
        sort = nullptr;
        np_inj = nullptr;

        // Invalidate parameters that will be set on initialize
        id = 0;
        q = 0;
        dx = {0, 0};
        box = { 0, 0 };
        iter = -1;

        d_energy = nullptr;
        d_nmove = nullptr;
    }

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
     * @param tile_dims         Individual tile grid size
     * @param dt                Time step
     * @param id                Species unique identifier
     * @param parallel          Parallel configuration
     */
    virtual void initialize( float2 const box_, uint2 const global_ntiles, uint2 const tile_dims,
    double const dt_, int const id_, mpi::cart2d & parallel );

    /**
     * @brief Destroy the Species object
     * 
     */
    virtual ~species() {
        gpu::device::free( np_inj );
        gpu::device::free( d_energy );
        gpu::device::free( d_nmove );

        delete( tmp );
        delete( sort );
        delete( particles );
    };

    /**
     * @brief Set the density profile object
     * 
     * @param new_density   New density object to be cloned
     */
    virtual void set_density( density::profile const & new_density ) {
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
     * @brief Get the density object (read only)
     * 
     * @return const density::profile& 
     */
    const density::profile & get_density() const {
        return * density;
    }

    /**
     * @brief Set the velocity distribution object
     * 
     * @param new_udist     New udist object to be cloned
     */
    virtual void set_udist( udist::type const & new_udist ) {
        udist = new_udist.clone();
    }

    /**
     * @brief Get the udist object
     * 
     * @return udist::type& 
     */
    udist::type & get_udist() {
        return *udist;
    }

    /**
     * @brief Get the udist object (read only)
     * 
     * @return const udist::type& 
     */
    const udist::type & get_udist() const {
        return *udist;
    }

    /**
     * @brief Sets the boundary condition type
     * 
     * @param new_bc 
     */
    void set_bc( species::bc_type new_bc ) {

        // Store new values
        global_bc = new_bc;

        // If initialized, update local_bc and other objects
        if ( particles ) {
            const auto & parallel = particles -> get_parallel();
            validate_bc( new_bc, parallel );
            update_bc( parallel );
        }
    }

    /**
     * @brief Get the current boundary condition types
     * 
     * @return species::bc_type 
     */
    species::bc_type get_bc( ) const noexcept { return global_bc; }

    /**
     * @brief Turns on the moving window algorithm
     * 
     * Sets both x boundaries to species::bc::open.
     * 
     * @note Must be called after initialize() (which sets the cell size) and
     *       before the simulation starts (iter = 0); calls mpi::fatal()
     *       otherwise or if the x boundaries are periodic.
     */
    void set_moving_window() { 
        if ( iter != 0 ) {
            mpi::fatal( "set_moving_window() must be called after initialize() and before the first iteration" );
        }

        // Use the global values, local values are bc::comm on internal ranks
        if ( global_bc.x.lower == species::bc::periodic ) {
            mpi::fatal( "Unable to set_moving_window() with periodic x boundaries" );
        }

        mov_window.init( dx.x );

        // Correct boundary conditions
        auto new_bc = global_bc;
        new_bc.x.lower = new_bc.x.upper = species::bc::open;
        set_bc( new_bc );
    }

    /**
     * @brief Inject particles in the complete simulation box
     * 
     */
    virtual void inject() {

        /// @brief position of lower corner of global grid in simulation units
        float2 global_ref{static_cast<float>(mov_window.motion()),0};

        density -> inject( *particles, ppc, dx, global_ref, particles -> local_range() );
    }

    /**
     * @brief Inject particles in the specified range of the simulation
     * 
     * @param range     Range in which to inject particles
     */
    virtual void inject( bounds_2d<unsigned int> range ){

        /// @brief position of lower corner of global grid in simulation units
        float2 global_ref{static_cast<float>(mov_window.motion()),0};

        density -> inject( *particles, ppc, dx, global_ref, range );
    }


    /**
     * @brief Gets number of particles that will be injected per tile
     * 
     * Although the routine only considers injection in a specific range, the
     * number of particles to be injected is calculated on all tiles (returning
     * zero on those, as expected)
     *
     * @param range     Range in which to inject particles
     * @param np        Device pointer to number of particles to be injected per tile
     */
    virtual void np_inject( bounds_2d<unsigned int> range, int * np ) {

        /// @brief position of lower corner of global grid in simulation units
        float2 global_ref{ static_cast<float>(mov_window.motion()),0};

        density -> np_inject( *particles, ppc, dx, global_ref, range, np );
    }

    /**
     * @brief Advance particle velocities
     * 
     * @param E     Electric field
     * @param B     Magnetic field
     */
    void push( grid::tiled_vec3<float> const & E, grid::tiled_vec3<float> const & B );

    /**
     * @brief Move particles (advance positions) and deposit current/charge
     * 
     * @param current   Electric current density
     * @param shift     (optional) Cell index shift applied after the move
     */
    void move( grid::tiled_vec3<float> & current, int2 const shift = make_int2(0,0) );

    /**
     * @brief Move particles (advance positions) without depositing current
     * 
     */
    void move();

    /**
     * @brief Free stream particles 1 timestep
     * 
     * @note Particles are free-streamed (no momentum update), no current is deposited.
     *       Used mostly for debug purposes.
     * 
     */
    virtual void advance() {

        // Advance positions
        move();

        // Process physical boundary conditions
        process_bc();
        
        // Sort particles according to tile
        particles -> tile_sort( *tmp, *sort );

        // Increase internal iteration number
        iter++;
    }

    /**
     * @brief Advance particle positions 1 timestep (no momentum update)
     * 
     * This routine will:
     * 1. Advance positions and deposit current
     * 2. Process boundary conditions
     * 3. Sort particles according to tiles
     * 
     * @param current   Electric current density
     */
    virtual void advance( current &current ) {

        // Advance positions and deposit current
        move( * current.J );

        // Process physical boundary conditions
        process_bc();

        // Sort particles according to tile
        particles -> tile_sort( *tmp, *sort );

        // Increase internal iteration number
        iter++;
    }

    /**
    * @brief Advance particles 1 iteration
    * 
    * This routine will:
    * 1. Advance momenta
    * 2. Advance positions and deposit current
    * 3. Process boundary conditions
    * 4. Sort particles according to tiles
    * 
    * @param emf       EM fields
    * @param current   Electric current density
    */
    virtual void advance( emf const &emf, current &current ){

        // Advance momenta
        push( * emf.E, * emf.B );

        // Advance positions and deposit current
        move( * current.J );

        // Process physical boundary conditions
        process_bc();

        // Sort particles according to tile
        particles -> tile_sort( *tmp, *sort );

        // Increase internal iteration number
        iter++;
    }

    /**
    * @brief Advance particles 1 iteration
    * 
    * This routine will:
    * 1. Advance momenta
    * 2. Advance positions and deposit current
    * 3. Process boundary conditions
    * 4. Handle moving window algorithm
    * 5. Sort particles according to tiles
    * 
    * @note The window move is checked at time (iter+1)*dt, i.e. the same
    *       time used by emf::advance() and current::advance() after their
    *       iteration counter is incremented
    * 
    * @param emf       EM fields
    * @param current   Electric current density
    */
    virtual void advance_mov_window( emf const &emf, current &current ){

        // Advance momenta
        push( * emf.E, * emf.B );

        if ( mov_window.needs_move((iter+1) * dt ) ) {
            // Advance positions, deposit current and shift particles
            move( * current.J, make_int2(-1,0) );

            // Process boundary conditions
            process_bc();

            // Advance moving window. This must be done before injecting, so
            // that the new particles are injected at the shifted position
            mov_window.advance();

            // Upper x edge node needs to inject new particles
            if ( particles -> parallel.on_edge( coord::x, edge::upper ) ) {
                // Find range where new particles need to be injected
                auto local_dims = particles -> get_local_dims();
                bounds_2d<unsigned int> range;
                range.x = { local_dims.x - 1, local_dims.x - 1 };
                range.y = { 0, local_dims.y - 1 };
            
                // Count new particles to be injected
                np_inject( range, np_inj );

                // Sort particles over tiles, leaving room for new particles to be injected
                particles -> tile_sort( *tmp, *sort, np_inj );

                // Inject new particles
                inject( range );
            } else {
                particles -> tile_sort( *tmp, *sort );
            }

        } else {
            // Advance positions and deposit current
            move( * current.J );

            // Process physical boundary conditions
            process_bc();

            // Sort particles according to tile
            particles -> tile_sort( *tmp, *sort );
        }

        // Increase internal iteration number
        iter++;
    }

    /**
     * @brief Deposit species charge
     * 
     * @param charge    Charge density grid
     */
    void deposit_charge( grid::tiled<float> &charge ) const;

    /**
     * @brief Returns total time centered kinetic energy
     * 
     * @note The energy is computed during push(); the value refers to the
     *       last push and is not updated by advance( current & )
     * 
     * @return double 
     */
    double get_energy() const {
        double energy;
        gpu::device::memcpy_tohost( &energy, d_energy, 1 );
        
        // Normalize and return
        return energy * q * m_q * dx.x * dx.y;
    }

    /**
     * @brief Returns total number of particles moved
     * 
     * @return uint64_t
     */
    unsigned long long get_nmove() const {
        unsigned long long nmove;
        gpu::device::memcpy_tohost( &nmove, d_nmove, 1 );
        return nmove;
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
