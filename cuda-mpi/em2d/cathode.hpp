#pragma once

#include <cstdint>
#include <limits>

#include "species.hpp"

/**
 * @brief Cathode: species continuously injected from an x wall
 * 
 * Particles are injected from the lower or upper x wall with fluid
 * (generalized) velocity `ufl` pointing into the box, and an optional
 * thermal spread `uth`. Injection is active for start <= t < end. If
 * start < 0 the region the flow would have filled by t = 0 is loaded with
 * particles at initialization.
 * 
 * The injection wall must use an open (absorbing) species boundary, particles
 * returning to the cathode are removed. The other boundaries are unrestricted.
 */
class cathode : public species {

private:

    /// @brief Velocity (beta) of cathode flow (always > 0)
    double vel;

    /// @brief Number of stream particles (per ppc.y and per cell along y)
    ///        injected so far
    uint64_t inj_count;

    /**
     * @brief Total number of stream particles that have crossed the wall
     *        up to time t
     * 
     * @param t             Simulation time
     * @return uint64_t     Number of particles (per ppc.y and per cell along y)
     */
    uint64_t n_crossed( double const t ) const;

    /**
     * @brief Sort particles over tiles, injecting new cathode particles
     *        if required
     * 
     * @note Calls particles -> tile_sort() exactly once on every rank
     * 
     * @param J     Current density to deposit the current of the injected
     *              particles to. Use nullptr to skip deposition.
     */
    void sort_inject( grid::tiled_vec3<float> * const J );

    /**
     * @brief Checks that the injection wall uses an open boundary
     * 
     * @note Calls mpi::fatal() otherwise
     * 
     * @param test_bc   Boundary conditions to check
     */
    void validate_wall_bc( species::bc_type const & test_bc ) const {
        auto const wall_bc = ( wall == edge::lower ) ? test_bc.x.lower : test_bc.x.upper;
        if ( wall_bc != species::bc::open ) {
            mpi::fatal( std::string( "cathode ") + name + ": the injection wall (x " +
                ( ( wall == edge::lower ) ? "lower" : "upper" ) +
                ") must use a species::bc::open boundary" );
        }
    }

public:

    /// @brief Generalized velocity of cathode flow (always > 0, direction
    ///        is set by the wall)
    const float ufl;

    /// @brief Wall to inject particles from
    edge::pos wall;

    /// @brief Time to start cathode injection
    double start;

    /// @brief Time to end cathode injection
    double end;

    /// @brief Temperature of injected particles
    float3 uth;

    /// @brief Reference density for cathode
    float n0;

    /**
     * @brief Deposit the current of injected particles (defaults to true)
     * 
     * @note When true, the current of the segment each new particle travels
     *       inside the box (from the wall to its injection position) is
     *       deposited, so that the continuity equation holds for injected
     *       particles. When false particles appear with no associated current
     *       (previous behavior).
     */
    bool inj_current;

    /**
     * @brief Construct a new cathode object
     * 
     * @param name      Cathode name
     * @param m_q       Mass over charge ratio
     * @param ppc       Number of particles per cell
     * @param wall      Wall to inject particles from (edge::lower or edge::upper)
     * @param ufl       Fluid generalized velocity (sign is ignored)
     */
    cathode( std::string const name, float const m_q, uint2 const ppc,
        edge::pos const wall, float const ufl );

    /**
     * @brief Initialize data structures and inject initial particles (if start < 0)
     * 
     * @param box_              Global simulation box size
     * @param global_ntiles     Global number of tiles
     * @param tile_dims         Individual tile grid size
     * @param dt_               Time step
     * @param id_               Species unique identifier
     * @param parallel          Parallel configuration
     */
    void initialize( float2 const box_, uint2 const global_ntiles, uint2 const tile_dims,
        double const dt_, int const id_, mpi::cart2d & parallel ) override;

    /**
     * @brief Free stream particles 1 timestep and inject new particles
     * 
     */
    void advance() override;

    /**
     * @brief Advance positions 1 timestep (no momentum update), deposit
     *        current and inject new particles
     * 
     * @param current   Electric current density
     */
    void advance( current & current ) override;

    /**
     * @brief Advance particles 1 iteration, deposit current and inject new
     *        particles
     * 
     * @param emf       EM fields
     * @param current   Electric current density
     */
    void advance( emf const & emf, current & current ) override;

    void advance_mov_window( emf const &, current & ) override {
        mpi::fatal( "cathodes cannot be used with moving windows" );
    }

    /**
     * @brief Sets the boundary condition type
     * 
     * @note The injection wall must use species::bc::open boundaries
     * 
     * @param new_bc    New boundary conditions
     */
    void set_bc( species::bc_type new_bc ) override {
        validate_wall_bc( new_bc );
        species::set_bc( new_bc );
    }

    void set_density( density::profile const & ) override {
        mpi::fatal( "cathodes do not support the set_density() method,"
                    " use the n0 parameter instead" );
    }

    void set_udist( udist::type const & ) override {
        mpi::fatal( "cathodes do not support the set_udist() method,"
                    " use the uth parameter instead" );
    }
};
