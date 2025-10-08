#ifndef SPECIES_H_
#define SPECIES_H_

#include <string>

#include "zpic.h"

#include "particles.h"

#include "emf.h"
#include "current.h"

#include "density.h"
#include "udist.h"

namespace phasespace {
    enum quant { z, r, ux, uy, uz };

    static inline void qinfo( quant q, std::string & name, std::string & label, std::string & units ) {
        switch(q) {
        case z :
            name = "z"; label = "z"; units = "c/\\omega_n";
            break;
        case r :
            name = "r"; label = "r"; units = "c/\\omega_n";
            break;
        case ux :
            name = "ux"; label = "u_x"; units = "c";
            break;
        case uy :
            name = "uy"; label = "u_y"; units = "c";
            break;
        case uz :
            name = "uz"; label = "u_z"; units = "c";
            break;
        }
    }
}

namespace species {
    enum pusher { boris, euler };
    namespace bc {
        enum type { open = 0, axial, periodic, reflecting };
    }
    typedef bnd<bc::type> bc_type;

}

/**
 * @brief Charged particles class
 * 
 */
class Species {

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

    /// @brief Initial velocity distribution
    UDistribution::Type * udist;

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
    Species( std::string const name, float const m_q, uint3 const ppc );

    /**
     * @brief Initialize data structures and inject initial particle distribution
     * 
     * @param nmodes            Number of cylindrical modes (including fundamental mode)
     * @param box               Global simulation box size
     * @param global_ntiles     Global number of tiles
     * @param nx                Individutal tile grid size
     * @param dt                Time step
     * @param id                Species unique identifier
     * @param parallel          Parallel configuration
     */
    virtual void initialize( int nmodes, float2 const box, uint2 const global_ntiles, uint2 const nx,
        float const dt, int const id );

    /**
     * @brief Destroy the Species object
     * 
     */
    ~Species();

    /**
     * @brief Get the number of cylindrical modes (including fundamental mode )
     * 
     * @return int 
     */
    int get_nmodes() { return nmodes; }

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

        if ( new_bc.y.lower != species::bc::axial ) {
            std::cerr <<  "(*warning*) Invalid species axial boundary, overriding value\n";
            new_bc.y.lower = species::bc::axial;
        }

        if ( new_bc.y.upper == species::bc::periodic  ) {
            std::cerr << "(*error*) Invalid species boundary type along (r).\n"
                      << "(*error*) Periodic boundaries are not allowed in this direction";
            exit(1);
        }

        // Store new values
        bc = new_bc;

        // Set periodic flags on tile grids
        if ( particles ) {
            particles -> periodic_z = (bc.x.lower == species::bc::periodic);
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
    void push( Cyl3CylGrid<float> & E, Cyl3CylGrid<float> & B );

    /**
     * @brief Move particles (advance positions) and deposit current
     * 
     * @param current   Electric current density
     */
    void move( Current & current );

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
     * @brief Deposit charge density (mode 0)
     * 
     * @param charge0   Charge density grid
     */
    void deposit_charge0( grid<float> &charge0 ) const;

    /**
     * @brief Deposit charge density (high order mode)
     * 
     * @param m         Cylindrical mode to deposit (1 to 4)
     * @param charge    Charge density grid (complex)
     */
    void deposit_charge( const unsigned m, grid<std::complex<float>> &charge ) const;

    /**
     * @brief Returns total time centered kinetic energy
     * 
     * @return double 
     */
    double get_energy() const {
        // Normalize and return
        return d_energy * m_q * dx.x * dx.y * 2 * M_PI;
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
    uint64_t np_total() const {
        return particles -> np_total();
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

template< int mode >
constexpr auto expimθ( float2 θ ) { 
    static_assert( 0, "expimθ not supported for this m value");
    return std::complex<float>{0};
}

template <> 
constexpr auto expimθ<0> ( float2 θ ) { 
    return std::complex<float>{1,0};
}

template <> 
constexpr auto expimθ<1> ( float2 θ ) { 
    auto cosθ = θ.x;
    auto sinθ = θ.y;

    return std::complex<float>{
        cosθ,
        sinθ
    };
}

template <>
constexpr auto expimθ<2> ( float2 θ ) { 
    auto cosθ = θ.x;
    auto sinθ = θ.y;

    return std::complex<float>{
        (cosθ-sinθ)*(cosθ+sinθ),    // cos2θ = cosθ^2 - sinθ^2
        2 * sinθ * cosθ             // sin2θ = 2 sinθ cosθ
    };
}

template <>
constexpr auto expimθ<3> ( float2 θ ) { 
    auto cosθ = θ.x;
    auto sinθ = θ.y;

    return std::complex<float>{
         4 * cosθ * ops::fma( cosθ, cosθ, -0.75f ),  // cos3θ = 4 cosθ (cosθ^2 - 3/4)
        -4 * sinθ * ops::fma( sinθ, sinθ, -0.75f )   // sin3θ = 4 sinθ (3/4 - sinθ^2)
    };
}

template <>
constexpr auto expimθ<4> ( float2 θ ) { 
    auto cosθ = θ.x;
    auto sinθ = θ.y;

    auto cos2 = cosθ * cosθ;

    return std::complex<float>{
        ops::fma( ops::fma( 8.f, cos2, -1.f ), cos2, 1.f ),
        -8 * sinθ * cosθ * ops::fma( sinθ, sinθ, -0.5f )
    };
}


#endif