#ifndef SIMULATION_H_
#define SIMULATION_H_

#include "zpic.h"

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
    const double dt;

    EMF emf;
    Current current;
    std::vector <Species*> species;

    /**
     * @brief Construct a new Simulation object
     * 
     * @param ntiles    Number of tiles
     * @param nx        Tile grid size
     * @param box       Simulation box size
     * @param dt        Time step
     */
    Simulation( uint2 const ntiles, uint2 const nx, float2 const box, double dt ):
        iter(0), ntiles( ntiles ), nx( nx ), box( box ), dt( dt ), 
        emf( ntiles, nx, box, dt ),
        current( ntiles, nx, box, dt ) {
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
        s.initialize( box, ntiles, nx, dt, species.size() );
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
     * 
     */
    void energy_info() {
        std::cout << "(*info*) Energy at n = " << iter << ", t = " << iter * double(dt)  << '\n';
        double part_ene = 0;
        for (unsigned i = 0; i < species.size(); i++) {
            double kin = species[i]->get_energy();
            std::cout << "(*info*) " << species[i]->name << " = " << kin << '\n';
            part_ene += kin;
        }

        if ( species.size() > 1 )
            std::cout << "(*info*) Total particle energy = " << part_ene << '\n';

        double3 ene_E, ene_B;
        emf.get_energy( ene_E, ene_B );
        std::cout << "(*info*) Electric field = " << ene_E.x + ene_E.y + ene_E.z << '\n';
        std::cout << "(*info*) Magnetic field = " << ene_B.x + ene_B.y + ene_B.z << '\n';

        double total = part_ene + ene_E.x + ene_E.y + ene_E.z + ene_B.x + ene_B.y + ene_B.z;
        std::cout << "(*info*) total = " << total << '\n';
    }

    /**
     * @brief Returns total number of particles moved
     * 
     * @return unsigned long long 
     */
    uint64_t get_nmove() {
        uint64_t nmove = 0;
        for (unsigned i = 0; i < species.size(); i++) nmove += species[i] -> get_nmove();
        return nmove;
    }
};


#endif
