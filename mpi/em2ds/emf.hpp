#pragma once

#include "vec_types.hpp"

#include "bounds.hpp"
#include "grid/tiled_vec3.hpp"
#include "grid/flat3.hpp"
#include "grid/fft.hpp"

#include "current.hpp"
#include "charge.hpp"

#include <string>

class emf {

    public:

    enum class quantity {e = 0, b, fe, fet, fb};
    struct bc { enum type { none = 0, periodic, pec, pmc }; };
    using bc_type = bounds_2d<bc::type>;

    private:

    /// @brief Boundary condition
    emf::bc_type bc;

    /// @brief cell size
    const float2 dx;

    /// @brief time step
    const double dt;

    /// @brief Iteration number
    int iter;

    /// @brief Device buffer for field energy calculations
    double * d_energy;

    /**
     * @brief Move simulation window if needed
     * 
     */
    void move_window( );

    /**
     * @brief Process boundary conditions
     * 
     */
    void process_bc( );

    public:

    /// @brief Electric field
    grid::tiled_vec3<float> * E;
    /// @brief Magnetic field
    grid::tiled_vec3<float> * B;
    /// @brief Simulation box size
    const float2 box;

    /// @brief Fourier transform of Electric field
    grid::flat3<std::complex<float>> * fE;
    /// @brief Fourier transform of transverse Electric field
    grid::flat3<std::complex<float>> * fEt;
    /// @brief Fourier transform of Magnetic field
    grid::flat3<std::complex<float>> * fB;

    grid::fft::c2r_plan * fft_backward;

    /**
     * @brief Construct a new EMF object
     * 
     * @param global_ntiles     Global number of tiles
     * @param tile_dims         Individual tile size
     * @param box               Global simulation box size
     * @param dt                Time step
     * @param parallel          Parallel partition 
     */
    emf( uint2 const global_ntiles, uint2 const tile_dims, float2 const box, double const dt, mpi::cart2d & parallel );
    
    /**
     * @brief Destroy the EMF object
     * 
     */
    ~emf() {
        delete (E);
        delete (B);

        delete (fE);
        delete (fEt);
        delete (fB);

        delete( fft_backward );
    }

    /**
     * @brief Stream extraction
     * 
     * @param os 
     * @param obj 
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, const emf & obj) {
        return os << "EMF object";
    }

    /**
     * @brief Get the iter value
     * 
     * @return auto 
     */
    int get_iter() const noexcept { return iter; }

    /**
     * @brief Get the boundary conditions
     * 
     * @return emf::bc_type 
     */
    emf::bc_type get_bc( ) const noexcept { return bc; }

    /**
     * @brief Set the boundary conditions
     * 
     * @param new_bc 
     */
    void set_bc( emf::bc_type new_bc ) {

        // Validate parameters
        if ( (new_bc.x.lower == emf::bc::periodic) || (new_bc.x.upper == emf::bc::periodic) ) {
            if ( new_bc.x.lower != new_bc.x.upper ) {
                mpi::fatal( "EMF boundary type mismatch along x."
                            " When choosing periodic boundaries both lower and upper types"
                            " must be set to emf::bc::periodic." );
            }
        }

        if ( (new_bc.y.lower == emf::bc::periodic) || (new_bc.y.upper == emf::bc::periodic) ) {
            if ( new_bc.y.lower != new_bc.y.upper ) {
                mpi::fatal( "EMF boundary type mismatch along y."
                            " When choosing periodic boundaries both lower and upper types"
                            " must be set to emf::bc::periodic." );
            }
        }

        if ( E -> part.periodic.x && new_bc.x.lower != emf::bc::periodic ) {
            mpi::fatal( "Only periodic x boundaries are supported with periodic x parallel partitions." );
        }

        if ( E -> part.periodic.y && new_bc.y.lower != emf::bc::periodic ) {
            mpi::fatal( "Only periodic y boundaries are supported with periodic y parallel partitions." );
        }

        // Store new values
        bc = new_bc;

        // Debug, remove from production code
        if ( mpi::root() ) {
            std::string bc_name[] = {"none", "periodic", "pec", "pmc"};
            std::cout << "(*info*) EMF boundary conditions\n";
            std::cout << "(*info*) x : [ " << bc_name[ bc.x.lower ] << ", " << bc_name[ bc.x.upper ] << " ]\n";
            std::cout << "(*info*) y : [ " << bc_name[ bc.y.lower ] << ", " << bc_name[ bc.y.upper ] << " ]\n";
        }
    }

    /**
     * @brief Advance EM field 1 iteration assuming no current or charge
     * 
     */
    void advance( );
    
    /**
     * @brief Advance EM field 1 iteration
     * 
     * @param current   Electric current density
     * @param charge    Electric charge densisty
     */
    void advance( current & current, charge & charge );

    /**
     * @brief Save EM field component to file
     * 
     * @param quant     Which field to save (E, B, fEt, etc.)
     * @param fc        Which field component to save (x, y or z)
     */
    void save( quantity const quant, const fcomp::cart fc ) const;
    
    /**
     * @brief Get EM field energy
     * 
     * @note The energy will be recalculated each time this routine is called
     * 
     * @param ene_E     Electric field energy (per component)
     * @param ene_b     Magnetic field energy (per component)
     */
    void get_energy( double3 & ene_E, double3 & ene_b ) const;
};

