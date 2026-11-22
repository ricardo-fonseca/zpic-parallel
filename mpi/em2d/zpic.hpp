#pragma once

#include <cmath>
#include <vector>
#include <algorithm>

#include "core/vec_types.hpp"
#include "parallel/mpi.hpp"
#include "parallel/partition.hpp"
#include "parallel/omp.hpp"
#include "simd/simd.hpp"

#include "emf.hpp"
#include "current.hpp"
#include "species.hpp"

namespace zpic {

/**
 * @brief Print out system information
 *
 * @note Only root node prints any information
 * 
 */
inline void sys_info() {
    if ( mpi::root() ) {
        std::cout << "MPI running on " << mpi::size() << " processes\n";
        simd::info();
        omp::info();
    }
}

/**
 * @brief EM2D simulation
 * 
 */
class simulation {

    public:

    /**
    * @brief CFL time limit from cell size
    *
    * @note This uses the stability limit from field solver
    * 
    * @param dx        Cell size
    * @return float    CFL time limit
    */
    static float courant( const float2 dx ) {
        return emf::courant(dx);
    }

    /**
    * @brief CFL time limit number cells and box size 
    * 
    * @param gnx       Global number of cells
    * @param box       Simulation box size (global)
    * @return float    CFL time limit
    */
    static float courant( const uint2 gnx, const float2 box  ) {
        float2 dx = make_float2( box.x/gnx.x, box.y/gnx.y);
        return courant(dx);
    }

    /**
    * @brief CFL time limit from number of tiles, tile size and simulation box size
    * 
    * @param ntiles    Number of tiles
    * @param nx        Number of cells per tile
    * @param box       Simulation box size (global)
    * @return float    CFL time limit
    */
    static float courant( const uint2 ntiles, const uint2 nx, const float2 box ) {
        float2 dx = make_float2( box.x / ( nx.x * ntiles.x ), box.y / ( nx.y * ntiles.y ) );
        return courant(dx);
    }

    private:

    /// @brief number of iterations
    unsigned int iter;

    public:

    /// @brief Global number of tiles
    const uint2 global_ntiles;
    /// @brief Tile grid dimensions
    const uint2 tile_dims;
    /// @brief Global simulation box size
    const float2 box;
    /// @brief Time step
    const float dt;

    /// @brief MPI parallel partition
    mpi::cart2d parallel;

    /// @brief EM fields
    ::emf emf;
    /// @brief Electric current density
    ::current current;

    /// @brief Vector of particle species
    std::vector< ::species * > species;

    /**
     * @brief Construct a new simulation object
     * 
     * @note Global periodic boundaries are set to true
     * 
     * @param global_ntiles     Global number of tiles
     * @param nx                Individual tile grid size
     * @param box               Global simulation box size
     * @param dt                Time step
     * @param partition         Parallel partition (number of parallel nodes in each direction)
     */
    simulation( const uint2 global_ntiles, const uint2 tile_dims, const float2 box, const float dt, 
        const uint2 partition ):
        iter(0), 
        global_ntiles( global_ntiles ), tile_dims( tile_dims ), box( box ), dt( dt ), 
        parallel( partition ),
        emf( global_ntiles, tile_dims, box, dt, parallel ),
        current( global_ntiles, tile_dims, box, dt, parallel ) {
    }

    /**
     * @brief Construct a new simulation object
     * 
     * @param global_ntiles     Global number of tiles
     * @param nx                Individual tile grid size
     * @param box               Global simulation box size
     * @param dt                Time step
     * @param partition         Parallel partition (number of parallel nodes in each direction)
     * @param periodic          Global periodic boundaries
     */
    simulation( const uint2 global_ntiles, const uint2 tile_dims, const float2 box, const float dt, 
        const uint2 partition, const int2 periodic ):
        iter(0), 
        global_ntiles( global_ntiles ), tile_dims( tile_dims ), box( box ), dt( dt ), 
        parallel( partition, periodic ),
        emf( global_ntiles, tile_dims, box, dt, parallel ),
        current( global_ntiles, tile_dims, box, dt, parallel ) {
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
        for ( auto & sp : species )
            sp->set_moving_window();
    }

    /**
     * @brief Adds particle species to the simulation
     *
     * @param s     Particle species 
     */
    void add_species( class species & s ) {
        species.push_back( &s );
        s.initialize( box, global_ntiles, tile_dims, dt, species.size(), parallel );
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
     * @brief Checks that the species, field and current boundary conditions
     *        are consistent
     * 
     * On every global edge:
     * | current          | emf      | species    |
     * |------------------|----------|------------|
     * | none             | none     | open       |
     * | periodic         | periodic | periodic   |
     * | pec              | pec      | reflecting |
     * | pmc              | pmc      | reflecting |
     * | pec_absorbing    | pec      | open       |
     * 
     * Periodic boundaries are validated against the parallel partition by
     * each object.
     * 
     * @note Calls mpi::fatal() on invalid combinations. This is a cheap host
     *       side check, that could be called at every iteration so that 
     *.      boundary conditions may be changed at any time.
     */
    void validate_bc() const {

        // Access boundary values by direction / edge
        auto at = []( auto const & b, coord::cart const dir, edge::pos const pos ) {
            auto const & bd = ( dir == coord::x ) ? b.x : b.y;
            return ( pos == edge::lower ) ? bd.lower : bd.upper;
        };

        auto const emf_bc = emf.get_bc();
        auto const current_bc = current.get_bc();

        for( auto dir : coord::all ) {
            for( auto pos : edge::all ) {
                std::string const loc = coord::name( dir ) + " " + edge::name( pos );

                // Get current bc
                auto const j = at( current_bc, dir, pos );

                // Get required emf bc
                emf::bc::type emf_req;
                switch( at( current_bc, dir, pos ) ) {
                case current::bc::none:
                    emf_req = emf::bc::none; break;
                case current::bc::pmc:
                    emf_req = emf::bc::pmc; break;
                case current::bc::pec_absorbing:
                case current::bc::pec:
                    emf_req = emf::bc::pec; break;
                default:
                    continue;   // periodic
                } 

                auto const e = at( emf_bc, dir, pos );
                if ( e != emf_req ) {
                    mpi::fatal( "emf (" + emf::bc::name( e ) + ") and current (" + 
                        current::bc::name( j ) + ") boundary types do not match on the " +
                        loc + " edge" );
                }

                // Get required species bc
                species::bc::type sp_req;
                switch( at( current_bc, dir, pos ) ) {
                case current::bc::none:
                case current::bc::pec_absorbing:
                    sp_req = species::bc::open; break;
                case current::bc::pec:
                case current::bc::pmc:
                    sp_req = species::bc::reflecting; break;
                default:
                    continue;   // periodic
                }

                for( auto const * sp : species ) {
                    auto const s = at( sp -> get_bc(), dir, pos );
                    if ( s != sp_req ) {
                        mpi::fatal( "species " + sp -> name + ": the " + loc + " edge is " +
                            species::bc::name( s ) + ", but the current boundary (" +
                            current::bc::name( j ) + ") requires " + species::bc::name( sp_req ) );
                    }
                }
            }
        }
    }

    /**
     * @brief Advance simulation 1 iteration
     * 
     */
    void advance() {

        // Zero global current
        current.zero();

        // Advance all species
        for ( auto & sp : species ) {
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
        for ( auto & sp : species ) {
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
    unsigned int get_iter() { return iter; };

    /**
     * @brief Get current simulation time
     * 
     * @return double   Simulation time
     */
    double get_t() { return iter * double(dt); };

    /**
     * @brief Print global energy diagnostic
     * @note must be called by all MPI nodes
     */
    void energy_info() {
        if ( parallel.root() ) {
            std::cout << "(*info*) Energy at n = " << iter << ", t = " << iter * double(dt)  << '\n';
        }
        
        double part_ene = 0;
         for ( auto & sp : species ) {
            double kin = sp -> get_energy();
            parallel.reduce( &kin, 1, mpi::sum );
            
            if ( parallel.root() )
                std::cout << "(*info*) " << sp->name << " = " << kin << '\n';

            part_ene += kin;
        }

        if ( species.size() > 1 && parallel.root() ) 
            std::cout << "(*info*) Total particle energy = " << part_ene << '\n';

        double3 ene_E, ene_B;
        emf.get_energy( ene_E, ene_B );

        // MPI does not natively support MPI_SUM for double3
        // We could implement it, but this is simpler
        double ene_fld[6] = {ene_E.x,ene_E.y,ene_E.z,ene_B.x, ene_B.y, ene_B.z};

        parallel.reduce( ene_fld, 6, mpi::sum );

        if ( parallel.root() ) {
            std::cout << "(*info*) Electric field = " << ene_fld[0] + ene_fld[1] + ene_fld[2] << '\n';
            std::cout << "(*info*) Magnetic field = " << ene_fld[3] + ene_fld[4] + ene_fld[5] << '\n';

            double total = part_ene;
            for( int i = 0; i < 6; i++ ) total += ene_fld[i];
            std::cout << "(*info*) total = " << total << '\n';
        }
    }

    /**
     * @brief Returns total number of particles moved
     * 
     * @note The default behavior is to only return the global result on the
     *       root MPI node.
     * 
     * @param all           (optional) Set to true to return value on all MPI
     *                      nodes, defaults to false
     * @return uint64_t 
     */
    uint64_t get_nmove( bool all = false ) {
        uint64_t nmove = 0;
        for ( auto & sp : species ) 
            nmove += sp -> get_nmove();
        
        if ( all ) {
            parallel.allreduce( &nmove, 1, mpi::sum );
        } else {
            parallel.reduce( &nmove, 1, mpi::sum );
        }

        return nmove;
    }
};

} // namespace zpic