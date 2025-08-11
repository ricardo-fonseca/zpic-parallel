#ifndef LASER_H_
#define LASER_H_

#include "emf.h"
#include "cyl3grid.h"

namespace Laser {

class Pulse {
    public:

    float start;    // Front edge of the laser pulse, in simulation units
    float fwhm;     // FWHM of the laser pulse duration, in simulation units
    float rise, flat, fall; // Rise, flat and fall time of the laser pulse, in simulation units 

    float a0;       // Normalized peak vector potential of the pulse
    float omega0;   // Laser frequency, normalized to the simulation frequency

    float polarization;

    float cos_pol;
    float sin_pol;

    unsigned int filter;

    Pulse() : start(0), fwhm(0), rise(0), flat(0), fall(0),
        a0(0), omega0(0),
        polarization(0), cos_pol(0), sin_pol(0), filter(1) {};

    /**
     * @brief Gets longitudinal laser envelope a given position
     * 
     * @param laser     Laser parameters
     * @param z         position
     * @return          laser envelope
     */
    float lon_env( const float z ) {

        if ( z > start ) {
            // Ahead of laser
            return 0.0;
        } else if ( z > start - rise ) {
            // Laser rise
            float csi = z - start;
            float e = std::sin( M_PI_2 * csi / rise );
            return e*e;
        } else if ( z > start - (rise + flat) ) {
            // Flat-top
            return 1.0;
        } else if ( z > start - (rise + flat + fall) ) {
            // Laser fall
            float csi = z - (start - rise - flat - fall);
            float e = std::sin( M_PI_2 * csi / fall );
            return e*e;
        }

        // Before laser
        return 0.0;
    }

    /**
     * @brief Validate laser parameters
     * 
     * @return  1 if parameters are ok, 0 otherwise 
     */
    virtual int validate();

    /**
     * @brief Launch a laser pulse
     * @note Sets E and B fields to the laser field
     * 
     * @param E     Electric field
     * @param B     Magnetic field
     * @param box   Box size (simulation units)
     * @return      Returns 0 on success, -1 on error (invalid laser parameters)
     */
    virtual int launch( cyl3grid<std::complex<float>>& E, cyl3grid<std::complex<float>>& B, float2 box ) = 0;

    /**
     * @brief Adds a new laser pulse onto an EMF object
     * 
     * @param emf   EMF object
     * @return      Returns 0 on success, -1 on error (invalid laser parameters)
     */
    virtual int add( EMF & emf ) {
        if ( emf.nmodes < 2 ) {
            std::cerr << "Laser pulses require at least 2 cylindrical modes (m = 0,1)"
                      << ", aborting...\n";
            std::exit(1);
        }
        
        // Get mode m = 1 fields
        auto & E1 = emf.E -> mode(1);
        auto & B1 = emf.B -> mode(1);

        cyl3grid<std::complex<float>> tmp_E( E1.ntiles, E1.nx, E1.gc );
        cyl3grid<std::complex<float>> tmp_B( B1.ntiles, B1.nx, B1.gc );

        // Get mode m = 1 laser fields
        int ierr = launch( tmp_E, tmp_B, emf.box );

        // Add laser to simulation
        if ( ! ierr ) {
            E1.add( tmp_E );
            B1.add( tmp_B );
        }

        std::cout << "Added laser to the simulation\n";

        return ierr;
    };
};

class PlaneWave : public Pulse {

    public:
    PlaneWave() : Pulse() {};

    int validate() { return Pulse::validate(); };
    int launch( cyl3grid<std::complex<float>>& E, cyl3grid<std::complex<float>>& B, float2 box );

    int add( EMF & emf ) { return Pulse::add(emf); }
};

class Gaussian : public Pulse {

    public:

    float W0;
    float focus;

    Gaussian() : Pulse(), W0(0), focus(0) {};

    int validate();
    int launch( cyl3grid<std::complex<float>>& E, cyl3grid<std::complex<float>>& B, float2 box );

    int add( EMF & emf ) { return Pulse::add(emf); }
};

}

#endif
