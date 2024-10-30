#ifndef LASER_H_
#define LASER_H_

#include "emf.h"
#include "vec3grid.h"

namespace Laser {

class Pulse {
    public:

    float start;    // Front edge of the laser pulse, in simulation units
    float fwhm;     // FWHM of the laser pulse duration, in simulation units
    float rise, flat, fall; // Rise, flat and fall time of the laser pulse, in simulation units 

    float a0;       // Normalized peak vector potential of the pulse
    float omega0;   // Laser frequency, normalized to the plasma frequency

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
    virtual int launch( vec3grid<float3>& E, vec3grid<float3>& B, float2 box ) = 0;

    /**
     * @brief Adds a new laser pulse onto an EMF object
     * 
     * @param emf   EMF object
     * @return      Returns 0 on success, -1 on error (invalid laser parameters)
     */
    int add( EMF & emf ) {
        vec3grid<float3> tmp_E( emf.E -> global_ntiles, emf.E-> nx, emf.E -> gc, emf.E -> part );
        vec3grid<float3> tmp_B( emf.E -> global_ntiles, emf.B-> nx, emf.B -> gc, emf.E -> part );

        // Get laser fields
        int ierr = launch( tmp_E, tmp_B, emf.box );

        // Add laser to simulation
        if ( ! ierr ) {
            emf.E->add( tmp_E );
            emf.B->add( tmp_B );
        }

        return ierr;
    };
};

class PlaneWave : public Pulse {

    public:
    PlaneWave() : Pulse() {};

    int validate() { return Pulse::validate(); };
    int launch( vec3grid<float3>& E, vec3grid<float3>& B, float2 box );

};

class Gaussian : public Pulse {

    public:

    float W0;
    float focus;
    float axis;

    Gaussian() : Pulse(), W0(0), focus(0), axis(0) {};

    int validate();
    int launch( vec3grid<float3>& E, vec3grid<float3>& B, float2 box );
};

}

#endif