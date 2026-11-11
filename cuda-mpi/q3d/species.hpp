#pragma once

#include <string>
#include <memory>
#include <cmath>

#include "parallel/partition.hpp"
#include "cyl_part/particles.hpp"

#include "emf.hpp"
#include "current.hpp"
#include "moving_window.hpp"

#include "density.hpp"
#include "udist.hpp"

namespace phasespace {
    enum class quantity { z = 0, r, ux, uy, uz };

    static inline void qinfo( quantity q, std::string & name, std::string & label, std::string & units ) {
        switch(q) {
        case quantity::z :
            name = "z"; label = "z"; units = "c/\\omega_n";
            break;
        case quantity::r :
            name = "r"; label = "r"; units = "c/\\omega_n";
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

    using complex_t = util::complex64;

    enum class pusher{ boris = 0, euler };

    /**
     * @brief Boundary condition types
     * 
     * @note Reflecting boundaries must be used together with emf / current
     *       pec or pmc walls (the current fold assumes particle reflection),
     *       and open boundaries with emf / current `none` boundaries. This is
     *       not checked here, the simulation class must ensure it.
     */
    struct bc { enum type { open = 0, comm, axial, periodic, reflecting }; };
    using bc_type = bounds_2d<bc::type>;

protected:

    /// @brief Number of cylindrical modes (including fundamental mode)
    int nmodes;

    /// @brief Unique species identifier
    int id;

    /// @brief Number of particles per cell(z,r,θ)
    uint3 ppc;

    /// @brief reference particle charge
    float q_ref;

    /// @brief Cell size
    float2 dx;

    /// @brief Simulation box size
    float2 box;

    /// @brief Time step
    double dt;

     /// @brief Iteration
    int iter;

     /// @brief Particle data buffer
    cyl_part::particles *particles;

    /// @brief Secondary data buffer to speed up some calculations
    cyl_part::particles *tmp;

    /// @brief Particle tile sort aux. data
    cyl_part::particle_sort *sort;

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
            mpi::fatal( "Periodic boundaries along radial direction are not allowed" );
        }

        if ( test_bc.y.lower != species::bc::axial) {
            mpi::fatal( "Lower radial boundary must be set to species::bc::axial" );
        }

        // Test against parallel partition (both ways)
        if ( ! parallel.periodic.x && test_bc.x.lower == species::bc::periodic ) {
            mpi::fatal( "Periodic x boundaries require a periodic x parallel partition.");
        }

        if ( parallel.periodic.x && test_bc.x.lower != species::bc::periodic ) {
            mpi::fatal( "Only periodic x boundaries are supported with periodic x parallel partitions.");
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
        phasespace::quantity q, float2 range, unsigned const size ) const;

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
        phasespace::quantity quant0, float2 range0, unsigned const size0,
        phasespace::quantity quant1, float2 range1, unsigned const size1 ) const;

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
    species( std::string const name, float const m_q, uint3 const ppc ):
    ppc(ppc), name(name), m_q(m_q)
    {
        // Validate parameters
        if ( m_q == 0 ) {
            mpi::fatal( "Invalid m_q value (0.0) must be not 0" );
        }

        if ( ppc.x < 1 || ppc.y < 1 || ppc.z < 1 ) {
            mpi::fatal( "Invalid ppc value (" + to_string(ppc) + 
                "), must be >= 1 in all directions" );
        }

        // Set default parameters
        density   = std::make_unique<density::uniform>( 1.0 );
        udist     = std::make_unique<udist::none>();
        
        // Upon initialization, open boundaries on periodic domains will be replaced
        global_bc = species::bc_type (species::bc::open);
        global_bc.y.lower = species::bc::axial; 
        
        push_type = species::pusher::boris;

        // Nullify pointers to data structures
        particles = nullptr;
        tmp = nullptr;
        sort = nullptr;
        np_inj = nullptr;

        // Invalidate parameters that will be set on initialize
        nmodes = 0;
        id = 0;
        q_ref = 0;
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
     * @param nmodes            Number of cylindrical modes (including fundamental mode)
     * @param box               Global simulation box size
     * @param global_ntiles     Global number of tiles
     * @param tile_dims         Individual tile grid size
     * @param dt                Time step
     * @param id                Species unique identifier
     * @param parallel          Parallel configuration
     */
    virtual void initialize( const int nmodes, float2 const box_, uint2 const global_ntiles, uint2 const tile_dims,
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
     * @brief Get the number of cylindrical modes (including fundamental mode )
     * 
     * @return int 
     */
    int get_nmodes() const { return nmodes; }

    /**
     * @brief Set the density profile object
     * 
     * @param new_density   New density object to be cloned
     */
    virtual void set_density( density::profile const & new_density ) {
        density = new_density.clone();

        // Recompute charge normalization factor in case initialize()
        // has already been called
        q_ref = copysign( density->n0 , m_q ) / (ppc.x * ppc.y * ppc.z);
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
     * @brief Inject particles in the specified range of the simulation
     * 
     * @param range     Range in which to inject particles
     */
    virtual void inject( bounds_2d<unsigned int> range ){

        /// @brief position of lower corner of global grid in simulation units
        float2 global_ref{static_cast<float>(mov_window.motion()),0};

        density -> inject( *particles, copysign( 1.0f, m_q ), ppc, dx, global_ref, range );
    }

    /**
     * @brief Inject particles in the complete simulation box
     * 
     */
    void inject() {
        inject( particles -> local_range() );
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
     * @param emf   EM Field
     */
    void push( cyl_modes::tiled_cyl3<float> & E, cyl_modes::tiled_cyl3<float> & B );

    /**
     * @brief Move particles (advance positions) and deposit current/charge
     * 
     * @param current   Electric current density
     * @param shift     (optional) Cell index shift applied after the move
     */
    void move( cyl_modes::tiled_cyl3<float> & current, int2 const shift = make_int2(0,0) );

    /**
     * @brief Move particles (advance positions) without depositing current
     * 
     */
    void move( int2 const shift = make_int2(0,0) );

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
     * @brief Deposit charge density (mode 0)
     * 
     * @param charge0   Charge density grid
     */
    void deposit_charge_0( grid::tiled<float> &charge ) const;

    /**
     * @brief Deposit charge density (high order mode)
     * 
     * @param m         Cylindrical mode to deposit (1 to 4)
     * @param charge    Charge density grid (complex)
     */
    void deposit_charge_m( const int m, grid::tiled<complex_t> &charge ) const;

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
        return energy * m_q * dx.x * dx.y * 2 * M_PI;
    }


    /**
     * @brief Returns total number of particles moved on local node
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
     * @brief Save charge density mode for species to file
     * 
     * @param m     Cylindrical mode to deposit (0 to 4)
     */
    void save_charge(const unsigned m) const;

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
    void info_np( std::string msg = "" ) {
        particles->info_np( msg );
    }

    /**
     * @brief Validates particle data
     * 
     * @details In case of invalid particle data prints out an error message and aborts
     *          the program
     * 
     * @param msg   Message to print in case of error
     * @param over  Amount of extra cells indices beyond limit allowed. Used
     *              when checking the buffer before tile_sort(). Defaults to 0
     */
    void validate( std::string msg, int const over = 0 ) {
        particles -> validate( msg, over );
    }
};
