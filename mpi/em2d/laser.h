#ifndef LASER_H_
#define LASER_H_

#include "emf.h"
#include "vec3grid.h"

namespace Laser {

 /**
 * @brief Base class for laser pulses
 * 
 */
class Pulse {

    public:

    /// @brief Front edge of the laser pulse (simulation units)
    float start;
    /**
     * @brief FWHM of the laser pulse duration (simulation units)
     * @note When this value is set it overrides rise, flat and fall parameters
     */
    float fwhm;
    /// @brief Rise time of the laser pulse (simulation units)
    float rise;
    /// @brief Flat time of the laser pulse (simulation units)
    float flat;
    /// @brief Fall time of the laser pulse (simulation units)
    float fall; 
    /// @brief Normalized peak vector potential of the pulse
    float a0;
    /// @brief Laser frequency, normalized to the plasma frequency
    float omega0;

    /**
     * @brief Laser polarization angle (radians)
     * @note polarization = 0 corresponds to an E field polarization along y.
     *       This will only be used if both cos_pol and sin_pol are 0.
     */
    float polarization;

    /// @brief cosine of the polarization angle
    float cos_pol;
    /// @brief sine of the polarization angle
    float sin_pol;

    /**
     * @brief Filter level to apply to the laser fields
     * @note Defaults to 1, set to 0 to disable filtering
     */
    unsigned int filter;

    /**
     * @brief Construct a new Pulse object
     * 
     */
    Pulse() : start(0), fwhm(0), rise(0), flat(0), fall(0),
        a0(0), omega0(0),
        polarization(0), cos_pol(0), sin_pol(0), filter(1) {};

    /**
     * @brief Validate laser parameters
     * 
     * @return  1 if parameters are ok, 0 otherwise 
     */
    virtual int validate();

    /**
     * @brief Stream extraction
     * 
     * @param os 
     * @param obj 
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, const Pulse& obj) {
        os << "[Laser pulse base class]";
        return os;
    };

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
     * @brief Adds a new laser pulse onto an EMF object
     * 
     * @param emf   EMF object
     * @return      Returns 0 on success, -1 on error (invalid laser parameters)
     */
    int add( EMF & emf ) {
        vec3grid<float3> tmp_E( emf.E -> global_ntiles, emf.E-> nx, emf.E -> gc, emf.E -> part );
        vec3grid<float3> tmp_B( emf.E -> global_ntiles, emf.B-> nx, emf.B -> gc, emf.B -> part );

        tmp_E.zero();
        tmp_B.zero();

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

    /**
     * @brief Stream extraction
     * 
     * @param os 
     * @param obj 
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, const PlaneWave& obj) {
        os << "Plane wave"
           << ", start: " << obj.start
           << ", fwhm: " << obj.fwhm
           << ", omega0: " << obj.omega0
           << ", a0: " << obj.a0;
        return os;
    };

};

class Gaussian : public Pulse {

    public:

    float W0;
    float focus;
    float axis;

    Gaussian() : Pulse(), W0(0), focus(0), axis(0) {};

    int validate();
    int launch( vec3grid<float3>& E, vec3grid<float3>& B, float2 box );

    /**
     * @brief Stream extraction
     * 
     * @param os 
     * @param obj 
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, const Gaussian& obj) {
        os << "Gaussian beam"
           << ", start: " << obj.start
           << ", fwhm: " << obj.fwhm
           << ", omega0: " << obj.omega0
           << ", a0: " << obj.a0
           << ", W0: " << obj.W0
           << ", focus: " << obj.focus
           << ", axis: " << obj.axis;

        return os;
    };

};

}

#endif