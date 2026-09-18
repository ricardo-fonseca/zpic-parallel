#pragma once

#include <cmath>
#include <vector>

#include "core/vec_types.hpp"
#include "parallel/mpi.hpp"
#include "parallel/partition.hpp"
#include "parallel/omp.hpp"
#include "simd/simd.hpp"

#include "emf.hpp"
#include "charge.hpp"
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

        #ifdef SIMD
            std::cout << "SIMD support enabled\n";
            std::cout << "  vector unit : " << vecname << '\n';
            std::cout << "  vector width: " << vecwidth <<'\n';
        #else
            std::cout << "SIMD support not enabled\n";
        #endif
        
        #ifdef _OPENMP
            std::cout << "OpenMP enabled\n";
            std::cout << "  # procs           : " << omp_get_num_procs() << '\n';
            std::cout << "  max_threads       : " << omp_get_max_threads() << '\n';
            #pragma omp parallel
            {
                if ( omp_get_thread_num() == 0 )
                    std::cout << "  default # threads : " << omp_get_num_threads() << '\n';
            }
        #else
            std::cout << "OpenMP support not enabled\n";
        #endif

    }
}

/**
 * @brief CFL time limit from cell size
 * 
 * @param dx        Cell size
 * @return float    CFL time limit
 */
inline float courant( const float2 dx ) {
    return std::sqrt( 1.0f/( 1.0f/(dx.x*dx.x) + 1.0f/(dx.y*dx.y) ) );
}

/**
 * @brief CFL time limit number cells and box size 
 * 
 * @param gnx       Global number of cells
 * @param box       Simulation box size (global)
 * @return float    CFL time limit
 */
inline float courant( const uint2 gnx, const float2 box  ) {
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
inline float courant( const uint2 ntiles, const uint2 nx, const float2 box ) {
    float2 dx = make_float2( box.x / ( nx.x * ntiles.x ), box.y / ( nx.y * ntiles.y ) );
    return courant(dx);
}


/**
 * @brief EM2Ds simulation
 * 
 */
class simulation {

    private:

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
    emf emf;
    /// @brief Electric current density
    current current;
    /// @brief Charge density
    charge charge;

    /// @brief Vector of particle species
    std::vector <species*> species;

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
        current( global_ntiles, tile_dims, box, dt, parallel ),
        charge( global_ntiles, tile_dims, box, dt, parallel ) {
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
        current( global_ntiles, tile_dims, box, dt, parallel ),
        charge( global_ntiles, tile_dims, box, dt, parallel ) {
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
     * @brief Adds particle species to the simulation
     *
     * @param s     Particle species 
     */
    inline void add_species( class species & s ) {
        species.push_back( &s );
        s.initialize( box, global_ntiles, tile_dims, dt, species.size(), parallel );
    }

    /**
     * @brief Gets a pointer to a specific species object
     * 
     * @param name                species name
     * @return species const* 
     */
    inline class species * get_species( const std::string & name ) {
        unsigned id = 0;
        for( id = 0; id < species.size(); id++ )
            if ( (species[id])->name == name ) break;
        return ( id < species.size() ) ? species[id] : nullptr;
    }

    /**
     * @brief Advance simulation 1 iteration
     * 
     */
    inline void advance( ) {

        // Zero global current and charge
        current.zero( );
        charge.zero( );

        // Advance all species
        for ( auto & sp : species ) {
            sp -> advance( emf, current, charge );
        }

        // Update current edge values and guard cells
        current.advance( );
        charge.advance();

        // Advance EM fields
        emf.advance( current, charge );

        iter++;
    }

    /**
     * @brief Get current iteration value
     * 
     * @return unsigned int     Iteration
     */
    inline unsigned int get_iter() const { return iter; };

    /**
     * @brief Get current simulation time
     * 
     * @return double   simulation time
     */
    inline double get_t() const { return iter * double(dt); };

    /**
     * @brief Print global energy diagnostic
     * @note must be called by all MPI nodes
     */
    inline void energy_info() {
        if ( parallel.root() ) {
            std::cout << "(*info*) Energy at n = " << iter << ", t = " << iter * double(dt)  << '\n';
        }
        
        double part_ene = 0;
        for (unsigned i = 0; i < species.size(); i++) {
            double kin = species[i]->get_energy();
            parallel.reduce( &kin, 1, mpi::sum );
            
            if ( parallel.root() )
                std::cout << "(*info*) " << species[i]->name << " = " << kin << '\n';

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
     * @return unsigned long long 
     */
    inline uint64_t get_nmove() const {
        uint64_t nmove = 0;
        for ( auto & sp : species ) 
            nmove += sp -> get_nmove();
        return nmove;
    }
};


}