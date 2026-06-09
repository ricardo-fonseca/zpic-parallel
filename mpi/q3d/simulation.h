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

    /// @brief Number of cylindrical modes
    const int nmodes;
   /// @brief Global number of tiles
    const uint2 global_ntiles;
    /// @brief Tile grid size
    const uint2 nx;
    /// @brief Global simulation box size
    const float2 box;
    /// @brief Time step
    const float dt;
    /// @brief MPI parallel partition
    Partition parallel;
    /// @brief EM fields
    EMF emf;
    /// @brief Current density
    Current current;
    /// @brief Vector of particle species
    std::vector <Species*> species;

    /**
     * @brief Construct a new Simulation object
     *
     * @note Global periodic boundaries along z are set to true
     * 
     * @param nmodes            Number of cylindrical modes
     * @param global_ntiles     Global number of tiles
     * @param nx                Individual tile grid size
     * @param box               Simulation box size
     * @param dt                Time step
     * @param partition         Parallel partition  (number of parallel nodes in each direction)
     * @param periodic_z        Use periodic boundaries along z
     */
    Simulation( int const nmodes, uint2 const global_ntiles, uint2 const nx,
                float2 const box, float const dt, uint2 partition,
                int periodic_z = 0 ):
        iter(0), nmodes(nmodes),
        global_ntiles( global_ntiles ), nx( nx ), box( box ), dt( dt ), 
        parallel( partition, make_int2( periodic_z, 0 ) ),
        emf( nmodes, global_ntiles, nx, box, dt, parallel ),
        current( nmodes, global_ntiles, nx, box, dt, parallel ) {
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

        if ( parallel.periodic.x ) {
            if ( parallel.root() )
                std::cerr << "(*error*) Simulation::set_moving_window() - Unable to set_moving_window() with periodic x partition\n";
            mpi::abort(1); 
        }

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
        s.initialize( nmodes, box, global_ntiles, nx, dt, species.size(), parallel );
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
        if ( parallel.root() )
            std::cout << "(*info*) Energy at n = " << iter << ", t = " << iter * double(dt)  << '\n';
        double part_ene = 0;
        for (unsigned i = 0; i < species.size(); i++) {
            double kin = species[i]->get_energy();
            parallel.reduce( &kin, 1, mpi::sum );

            if ( parallel.root() ) {
                std::cout << "(*info*) " << species[i]->name << " = " << kin << '\n';
                part_ene += kin;
            }
        }

        if ( species.size() > 1 && parallel.root() ) 
            std::cout << "(*info*) Total particle energy = " << part_ene << '\n';

        if ( parallel.root() )
            std::cout << "(*info*) EMF energy\n";
        
        cyl_double3 ene_E, ene_B, total_E{0}, total_B{0};
        for( int i = 0; i < nmodes; i++ ) {
            emf.get_energy( ene_E, ene_B, i );
            
            double ene_fld[6] = {ene_E.z, ene_E.r, ene_E.θ, ene_B.z, ene_B.r, ene_B.θ};
            parallel.reduce( ene_fld, 6, mpi::sum );

            if ( parallel.root() ) {
                std::cout << "(*info*) Mode " << i << '\n';
                std::cout << "(*info*) Electric field("<< i << ") = " << ene_fld[0] + ene_fld[1] + ene_fld[2] << '\n';
                std::cout << "(*info*) Magnetic field("<< i << ") = " << ene_fld[3] + ene_fld[4] + ene_fld[5] << '\n';

                total_E.z += ene_fld[0];
                total_E.r += ene_fld[1];
                total_E.θ += ene_fld[2];

                total_B.z += ene_fld[3];
                total_B.r += ene_fld[4];
                total_B.θ += ene_fld[5];
            }
        }

        if ( nmodes > 1 && parallel.root() ) {
            std::cout << "(*info*) Total (all modes)\n";
            std::cout << "(*info*) Electric field " << total_E.z + total_E.r + total_E.θ << '\n';
            std::cout << "(*info*) Magnetic field " << total_B.z + total_B.r + total_B.θ << '\n';
        }

        if ( parallel.root() ) {
            double total = part_ene + 
                        total_E.z + total_E.r + total_E.θ +
                        total_B.z + total_B.r + total_B.θ;
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
