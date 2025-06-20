#ifndef EMF_H_
#define EMF_H_

#include "zpic.h"

#include "utils.h"

#include "bnd.h"
#include "vec3grid.h"
#include "current.h"
#include "charge.h"


#include <string>

namespace emf {
    enum field  { e, b };

    namespace bc {
        enum type { none = 0, periodic, pec, pmc };
    }

    typedef bnd<bc::type> bc_type;
}

class EMF {

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


    fft::plan * fft_backward;

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
    vec3grid<float3> * E;
    /// @brief Magnetic field
    vec3grid<float3> * B;
    /// @brief Simulation box size
    const float2 box;

    /// @brief Fourier transform of Electric field
    basic_grid3<std::complex<float>> * fE;
    /// @brief Fourier transform of trasnverse Electric field
    basic_grid3<std::complex<float>> * fEt;
    /// @brief Fourier transform of Magnetic field
    basic_grid3<std::complex<float>> * fB;

    /**
     * @brief Construct a new EMF object
     * 
     * @param ntiles    Number of tiles in x,y direction
     * @param nx        Tile size (#cells)
     * @param box       Simulation box size (sim. units)
     * @param dt        Time step
     */
    EMF( uint2 const ntiles, uint2 const nx, float2 const box, double const dt );
    
    /**
     * @brief Destroy the EMF object
     * 
     */
    ~EMF() {
        device::free( d_energy );
        delete (E);
        delete (B);

        delete (fE);
        delete (fEt);
        delete (fB);

        delete( fft_backward );
    }

    friend std::ostream& operator<<(std::ostream& os, const EMF obj) {
        os << "EMF object\n";
        return os;
    }

    /**
     * @brief Get the iter value
     * 
     * @return auto 
     */
    int get_iter() { return iter; }

    /**
     * @brief Get the boundary conditions
     * 
     * @return emf::bc_type 
     */
    emf::bc_type get_bc( ) { return bc; }

    /**
     * @brief Set the boundary conditions
     * 
     * @param new_bc 
     */
    void set_bc( emf::bc_type new_bc ) {

        // Validate parameters
        if ( (new_bc.x.lower == emf::bc::periodic) || (new_bc.x.upper == emf::bc::periodic) ) {
            if ( new_bc.x.lower != new_bc.x.upper ) {
                std::cerr << "(*error*) EMF boundary type mismatch along x.\n";
                std::cerr << "(*error*) When choosing periodic boundaries both lower and upper types must be set to emf::bc::periodic.\n";
                exit(1);
            }
        }

        if ( (new_bc.y.lower == emf::bc::periodic) || (new_bc.y.upper == emf::bc::periodic) ) {
            if ( new_bc.y.lower != new_bc.y.upper ) {
                std::cerr << "(*error*) EMF boundary type mismatch along y.\n";
                std::cerr << "(*error*) When choosing periodic boundaries both lower and upper types must be set to emf::bc::periodic.\n";
                exit(1);
            }
        }

        // Store new values
        bc = new_bc;


        std::string bc_name[] = {"none", "periodic", "pec", "pmc"};
        std::cout << "(*info*) EMF boundary conditions\n";
        std::cout << "(*info*) x : [ " << bc_name[ bc.x.lower ] << ", " << bc_name[ bc.x.upper ] << " ]\n";
        std::cout << "(*info*) y : [ " << bc_name[ bc.y.lower ] << ", " << bc_name[ bc.y.upper ] << " ]\n";

        // Set periodic flags on tile grids
        E -> periodic.x = B->periodic.x = ( bc.x.lower == emf::bc::periodic );
        E -> periodic.y = B->periodic.y = ( bc.y.lower == emf::bc::periodic );
    }

    void advance( );
    void advance( Current & current, Charge & charge );

    void save( emf::field const field, const fcomp::cart fc );
    
    /**
     * @brief Get total field energy per field component
     * 
     * @warning This function will always recalculate the energy each time it is
     *          called.
     * 
     * @param ene_E     Total E-field energy (per component)
     * @param ene_B     Total B-field energy (per component)
     */
    void get_energy( double3 & ene_E, double3 & ene_b );
};

#endif
