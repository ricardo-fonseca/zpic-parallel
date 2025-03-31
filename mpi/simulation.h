#ifndef SIMULATION_H_
#define SIMULATION_H_

#include "zpic.h"

#include "parallel.h"
#include "emf.h"
#include "current.h"
#include "species.h"

#include "timer.h"

#include <vector>

class Simulation {

    private:

    unsigned int iter;

    public:

    const uint2 ntiles;
    const uint2 nx;
    const float2 box;
    const float dt;

    Partition parallel;
    EMF emf;
    Current current;
    std::vector <Species*> species;

    /**
     * @brief Construct a new Simulation object
     * 
     * @param ntiles    Global number of tiles
     * @param nx        Tile grid size
     * @param box       Simulation box size
     * @param dt        Time step
     */
    Simulation( uint2 const ntiles, uint2 const nx, float2 const box, float const dt, uint2 partition ):
        iter(0), 
        ntiles( ntiles ), nx( nx ), box( box ), dt( dt ), 
        parallel( partition ),
        emf( ntiles, nx, box, dt, parallel ),
        current( ntiles, nx, box, dt, parallel ) {
    }

    Simulation( uint2 const ntiles, uint2 const nx, float2 const box, float const dt, 
        uint2 const partition, int2 const periodic ):
        iter(0), 
        ntiles( ntiles ), nx( nx ), box( box ), dt( dt ), 
        parallel( partition, periodic ),
        emf( ntiles, nx, box, dt, parallel ),
        current( ntiles, nx, box, dt, parallel ) {
    }

    /**
     * @brief Destroy the Simulation object
     * 
     */
    ~Simulation() {
    };

    /**
     * @brief Turns on the moving window algorithm
     * 
     */
    void set_moving_window() {
        emf.set_moving_window();
        current.set_moving_window();
        for (unsigned i = 0; i < species.size(); i++)
            species[i]->set_moving_window();
    }

    /**
     * @brief Adds particle species to the simulation
     *
     * @param s     Particle species 
     */
    void add_species( Species & s ) {
        species.push_back( &s );
        s.initialize( box, ntiles, nx, dt, species.size(), parallel );
    }

    /**
     * @brief Gets a pointer to a specific species object
     * 
     * @param name                Species name
     * @return Species const* 
     */
    Species * get_species( std::string name ) {
        unsigned id = 0;
        for( id = 0; id < species.size(); id++ )
            if ( (species[id])->name == name ) break;
        return ( id < species.size() ) ? species[id] : nullptr;
    }

    /**
     * @brief Advance simulation 1 iteration
     * 
     */
    void advance() {

        // Zero global current
        current.zero();

        // Advance all species
        // for (int i = 0; i < species.size(); i++) {
        //     species[i] -> advance( *emf, *current );
        // }

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
        current.zero();

        // Advance all species
        // for (int i = 0; i < species.size(); i++) {
        //     species[i] -> advance_mov_window( emf, current );
        // }

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
     * @note Only root MPI node gets the proper result
     * 
     * @return unsigned long long 
     */

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
        for (unsigned i = 0; i < species.size(); i++) nmove += species[i] -> get_nmove();
        
        if ( all ) {
            parallel.allreduce( &nmove, 1, mpi::sum );
        } else {
            parallel.reduce( &nmove, 1, mpi::sum );
        }

        return nmove;
    }
};


#endif
