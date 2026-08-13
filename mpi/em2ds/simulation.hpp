#pragma once

#include "emf.hpp"
#include "current.hpp"
#include "species.hpp"

#include <vector>

class Simulation {

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
    Partition parallel;

    /// @brief EM fields
    EMF emf;
    /// @brief Electric current density
    Current current;
    /// @brief Charge density
    Charge charge;

    /// @brief Vector of particle species
    std::vector <Species*> species;

    /**
     * @brief Construct a new Simulation object
     * 
     * @note Global periodic boundaries are set to true
     * 
     * @param global_ntiles     Global number of tiles
     * @param nx                Individual tile grid size
     * @param box               Global simulation box size
     * @param dt                Time step
     * @param partition         Parallel partition (number of parallel nodes in each direction)
     */
    Simulation( const uint2 global_ntiles, const uint2 tile_dims, const float2 box, const float dt, 
        const uint2 partition ):
        iter(0), 
        global_ntiles( global_ntiles ), tile_dims( tile_dims ), box( box ), dt( dt ), 
        parallel( partition ),
        emf( global_ntiles, tile_dims, box, dt, parallel ),
        current( global_ntiles, tile_dims, box, dt, parallel ),
        charge( global_ntiles, tile_dims, box, dt, parallel ) {
    }

    /**
     * @brief Construct a new Simulation object
     * 
     * @param global_ntiles     Global number of tiles
     * @param nx                Individual tile grid size
     * @param box               Global simulation box size
     * @param dt                Time step
     * @param partition         Parallel partition (number of parallel nodes in each direction)
     * @param periodic          Global periodic boundaries
     */
    Simulation( const uint2 global_ntiles, const uint2 tile_dims, const float2 box, const float dt, 
        const uint2 partition, const int2 periodic ):
        iter(0), 
        global_ntiles( global_ntiles ), tile_dims( tile_dims ), box( box ), dt( dt ), 
        parallel( partition, periodic ),
        emf( global_ntiles, tile_dims, box, dt, parallel ),
        current( global_ntiles, tile_dims, box, dt, parallel ),
        charge( global_ntiles, tile_dims, box, dt, parallel ) {
    }

    /**
     * @brief Destroy the Simulation object
     * 
     */
    ~Simulation() {
    };

    /**
     * @brief Adds particle species to the simulation
     *
     * @param s     Particle species 
     */
    void add_species( Species & s ) {
        species.push_back( &s );
        s.initialize( box, global_ntiles, tile_dims, dt, species.size(), parallel );
    }

    /**
     * @brief Gets a pointer to a specific species object
     * 
     * @param name                Species name
     * @return Species const* 
     */
    Species * get_species( const std::string & name ) {
        unsigned id = 0;
        for( id = 0; id < species.size(); id++ )
            if ( (species[id])->name == name ) break;
        return ( id < species.size() ) ? species[id] : nullptr;
    }

    /**
     * @brief Advance simulation 1 iteration
     * 
     */
    void advance( ) {

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
    unsigned int get_iter() const { return iter; };

    /**
     * @brief Get current simulation time
     * 
     * @return double   Simulation time
     */
    double get_t() const { return iter * double(dt); };

    /**
     * @brief Print global energy diagnostic
     * @note must be called by all MPI nodes
     */
    void energy_info() {
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
    uint64_t get_nmove() const {
        uint64_t nmove = 0;
        for ( auto & sp : species ) 
            nmove += sp -> get_nmove();
        return nmove;
    }
};

