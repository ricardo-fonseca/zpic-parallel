#pragma once

#include "emf.hpp"

namespace laser {

using complex_t = util::complex64;

/**
 * @brief Longitudinal laser envelope (resolved parameters)
 * 
 * @note Plain-data type, so it can be passed directly to CUDA kernels
 */
struct envelope {
    /// @brief Front edge of the laser pulse
    float start;
    /// @brief Rise length
    float rise;
    /// @brief Flat length
    float flat;
    /// @brief Fall length
    float fall;
};

/**
 * @brief Base class for laser pulses
 * 
 * @note The user-set parameters are never modified by this class. Derived
 *       values (envelope, polarization direction) are obtained from
 *       get_envelope() and get_polarization().
 */
class pulse {

    public:

    /// @brief Front edge of the laser pulse (simulation units)
    float start;
    /**
     * @brief FWHM of the laser pulse duration (simulation units)
     * @note When this value is > 0 it overrides the rise, flat and fall
     *       parameters (see get_envelope())
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
     * @brief Low-pass filter to apply to the laser fields (cutoff)
     * @note Defaults to [0.5,0.5], set to [1.0, 1.0] to disable filtering
     */
    float2 filter;

    /**
     * @brief Construct a new Pulse object
     * 
     */
    pulse() : start(0), fwhm(0), rise(0), flat(0), fall(0),
        a0(0), omega0(0),
        polarization(0), cos_pol(0), sin_pol(0), filter( make_float2(0.5f, 0.5f) ) {};

    /**
     * @brief Destroy the Pulse object
     * 
     */
    virtual ~pulse() = default;

    /**
     * @brief Validate laser parameters
     * @note Invalid parameters cause the simulation to stop (mpi::fatal)
     */
    virtual void validate() const;

    /**
     * @brief Get the longitudinal envelope
     * @note If fwhm > 0 this returns rise = fall = fwhm and flat = 0,
     *       otherwise the rise, flat and fall parameters are used
     * 
     * @return envelope 
     */
    envelope get_envelope() const;

    /**
     * @brief Get the polarization direction
     * @note Uses cos_pol / sin_pol if either is set, otherwise these are
     *       calculated from the polarization angle
     * 
     * @return float2   { cos(polarization), sin(polarization) }
     */
    float2 get_polarization() const;

    /**
     * @brief Stream extraction
     * 
     * @param os 
     * @param obj 
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, const pulse & obj) {
        os << "[Laser pulse base class]";
        return os;
    };

    /**
     * @brief Launch a laser pulse
     * @note Sets the transverse components of the E and B laser fields (real
     *       space, interior cells only). The longitudinal components are set
     *       in k-space by lon_x(), called from add().
     * 
     * @param E     Electric field
     * @param B     Magnetic field
     * @param box   Box size (simulation units)
     * @return      Returns 0 on success
     */
    virtual int launch( grid::tiled_vec3<float>& E, grid::tiled_vec3<float>& B, const float2 box ) const = 0;

    /**
     * @brief Sets longitudinal component
     * @note Enforces div.fld = 0 by setting x component
     * 
     * @param fld   Fourier transform of E or B field
     * @param dk    Cell size in k space
     * @return int  Returns 0 on success
     */
    virtual int lon_x( grid::flat3<complex_t> & fld, const float2 dk ) const = 0;


    /**
     * @brief Adds a new laser pulse onto an EMF object
     * 
     * @param emf   EMF object
     * @return      Returns 0 on success
     */
    int add( emf & emf ) const;
};

/**
 * @brief Plane wave laser pulse
 * 
 */
class plane_wave : public pulse {

    public:
    plane_wave() : pulse() {};

    int launch( grid::tiled_vec3<float> & E, grid::tiled_vec3<float>& B, const float2 box ) const override;

    // No need to set longitudinal field on plane wave
    int lon_x( grid::flat3<complex_t> & fld, const float2 dk ) const override { return 0; }; 

    /**
     * @brief Stream extraction
     * 
     * @param os 
     * @param obj 
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, const plane_wave& obj) {
        os << "Plane wave"
           << ", start: " << obj.start
           << ", fwhm: " << obj.fwhm
           << ", omega0: " << obj.omega0
           << ", a0: " << obj.a0;
        return os;
    };
};

/**
 * @brief Gaussian laser pulse
 * 
 */
class gaussian : public pulse {

    public:

    float W0;
    float focus;
    float axis;

    gaussian() : pulse(), W0(0), focus(0), axis(0) {};

    void validate() const override;
    int launch( grid::tiled_vec3<float> & E, grid::tiled_vec3<float> & B, const float2 box ) const override;
    int lon_x( grid::flat3<complex_t> & fld, const float2 dk ) const override;

    /**
     * @brief Stream extraction
     * 
     * @param os 
     * @param obj 
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, const gaussian & obj) {
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
