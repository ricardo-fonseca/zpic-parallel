#pragma once

#include <cmath>
#include <vector>
#include <algorithm>

#include "core/vec_types.hpp"
#include "parallel/omp.hpp"
#include "simd/simd.hpp"

#include "emf.hpp"
#include "current.hpp"
#include "species.hpp"

namespace q3d {

/**
 * @brief Print out system information
 * 
 */
inline void sys_info() {
    simd::info();
    omp::info();
}

/**
 * @brief Q3D simulation
 * 
 */
class simulation {

    public:

    /**
    * @brief CFL time limit from cell size
    *
    * @note Returns the CFL limit from the emf solver
    * 
    * @param m         Number of cylindrical modes (including mode 0)
    * @param dx        Cell size
    * @return float    CFL time limit
    */
    static float courant( const unsigned m, const float2 dx ) {
        return emf::courant( m, dx );
    }

    /**
    * @brief CFL time limit number cells and box size 
    * 
    * @param global_dims    Global number of cells
    * @param box            Simulation box size (global)
    * @return float         CFL time limit
    */
    static float courant( const unsigned m, const uint2 global_dims, const float2 box  ) {
        float2 dx = make_float2( box.x/global_dims.x, box.y/global_dims.y);
        return courant(m, dx);
    }

    /**
    * @brief CFL time limit from number of tiles, tile size and simulation box size
    * 
    * @param ntiles    Number of tiles
    * @param nx        Number of cells per tile
    * @param box       Simulation box size (global)
    * @return float    CFL time limit
    */
    static float courant( const unsigned m, const uint2 ntiles, const uint2 tile_dims, const float2 box ) {
        float2 dx = make_float2( box.x / ( tile_dims.x * ntiles.x ), box.y / ( tile_dims.y * ntiles.y ) );
        return courant(m, dx);
    }

    private:

    /// @brief number of iterations
    unsigned int iter = 0;

    public:

    /// @brief Number of cylindrical modes
    const int nmodes;
    /// @brief Number of tiles
    const uint2 grid_ntiles;
    /// @brief Tile grid dimensions
    const uint2 tile_dims;
    /// @brief Simulation box size
    const float2 box;
    /// @brief Time step
    const float dt;

    /// @brief Periodic boundaries (z,r)
    const int2 periodic;

    /// @brief EM fields
    q3d::emf emf;
    /// @brief Electric current density
    q3d::current current;

    /// @brief Vector of particle species
    std::vector< q3d::species * > species;

    /**
     * @brief Construct a new simulation object
     * 
     * @param nmodes            Number of cylindrical modes
     * @param ntiles            Number of tiles
     * @param tile_dims         Individual tile grid size
     * @param box               Simulation box size
     * @param dt                Time step
     * @param periodic_z        Use periodic boundaries along z
     */
    simulation( int const nmodes, const uint2 ntiles, const uint2 tile_dims, const float2 box, const float dt, 
        int periodic_z = 0 ):
        nmodes(nmodes), 
        grid_ntiles( ntiles ), tile_dims( tile_dims ), box( box ), dt( dt ), 
        periodic( make_int2( periodic_z, 0 ) ),
        emf( nmodes, ntiles, tile_dims, box, dt, periodic ),
        current( nmodes, ntiles, tile_dims, box, dt, periodic ) {
    }

    /**
     * @brief Copy constructor
     * 
     */
    simulation( const simulation & ) = delete;
    
    /**
     * @brief Move constructor
     * 
     */
    simulation( simulation && ) = delete;

    /**
     * @brief Destroy the simulation object
     * 
     */
    ~simulation() {
    };

    /**
     * @brief Turns on the moving window algorithm
     * 
     */
    void set_moving_window() {
        emf.set_moving_window();
        current.set_moving_window();
        for ( auto sp : species )
            sp->set_moving_window();
    }

    /**
     * @brief Adds particle species to the simulation
     *
     * @param s     Particle species 
     */
    void add_species( class species & s ) {
        species.push_back( &s );
        s.initialize( nmodes, box, grid_ntiles, tile_dims, dt, species.size(), periodic );
    }

    /**
     * @brief Gets a pointer to a specific species object
     * 
     * @param name                species name
     * @return species* (nullptr if not found)
     */
    class species * get_species( const std::string & name ) {
        auto it = std::find_if( species.begin(), species.end(),
            [&name]( const auto & s ) { return s->name == name; } );
        return ( it != species.end() ) ? *it : nullptr;
    }

    const class species * get_species( const std::string & name ) const {
        auto it = std::find_if( species.begin(), species.end(),
            [&name]( const auto & s ) { return s->name == name; } );
        return ( it != species.end() ) ? *it : nullptr;
    }

    /**
     * @brief Advance simulation 1 iteration
     * 
     */
    void advance() {

        // Zero global current
        current.zero();

        // Advance all species
        for ( auto sp : species ) {
            sp -> advance( emf, current );
        }

        // Update current edge values and guard cells
        current.advance();

        // Advance EM fields
        emf.advance( current );

        iter++;
    }

    /**
     * @brief Advance simulation 1 iteration using a moving window
     * 
     */
    void advance_mov_window() {

        // Zero global current
        current.zero( );

        // Advance all species
        for ( auto sp : species ) {
            sp -> advance_mov_window( emf, current );
        }

        // Update current edge values and guard cells
        current.advance();
        
        // Advance EM fields
        emf.advance( current );

        iter++;
    }

    /**
     * @brief Get current iteration value
     * 
     * @return unsigned int     Iteration
     */
    unsigned int get_iter() const { return iter; };

    /**
     * @brief Get current simulation time
     * 
     * @return double   Simulation time
     */
    double get_t() const { return iter * double(dt); };

    /**
     * @brief Print energy diagnostic
     */
    void energy_info() const {
        std::cout << "(*info*) Energy at n = " << iter << ", t = " << iter * double(dt)  << '\n';
        
        double part_ene = 0;
        for ( auto sp : species ) {
            double kin = sp->get_energy();
            
            std::cout << "(*info*) " << sp->name << " = " << kin << '\n';

            part_ene += kin;
        }

        if ( species.size() > 1 ) 
            std::cout << "(*info*) Total particle energy = " << part_ene << '\n';

        std::cout << "(*info*) EMF energy\n";
        
        cyl_double3 ene_E, ene_B, total_E{}, total_B{};
        for( int i = 0; i < nmodes; i++ ) {
            emf.get_energy( ene_E, ene_B, i );
            
            double ene_fld[6] = {ene_E.z, ene_E.r, ene_E.th, ene_B.z, ene_B.r, ene_B.th};

            std::cout << "(*info*) Mode " << i << '\n'
                      << "(*info*) Electric field("<< i << ") = " << ene_fld[0] + ene_fld[1] + ene_fld[2] << '\n'
                      << "(*info*) Magnetic field("<< i << ") = " << ene_fld[3] + ene_fld[4] + ene_fld[5] << '\n';

            total_E.z += ene_fld[0];
            total_E.r += ene_fld[1];
            total_E.th += ene_fld[2];

            total_B.z += ene_fld[3];
            total_B.r += ene_fld[4];
            total_B.th += ene_fld[5];
        }

        if ( nmodes > 1 ) {
            std::cout << "(*info*) Total (all modes)\n"
                      << "(*info*) Electric field " << total_E.z + total_E.r + total_E.th << '\n'
                      << "(*info*) Magnetic field " << total_B.z + total_B.r + total_B.th << '\n';
        }

        double total = part_ene + 
                    total_E.z + total_E.r + total_E.th +
                    total_B.z + total_B.r + total_B.th;
        std::cout << "(*info*) total = " << total << '\n';
    }

    /**
     * @brief Returns total number of particles moved
     * 
     * @return uint64_t 
     */
    uint64_t get_nmove() {
        uint64_t nmove = 0;
        for ( auto sp : species ) 
            nmove += sp -> get_nmove();
        return nmove;
    }
};

} // namespace q3d
