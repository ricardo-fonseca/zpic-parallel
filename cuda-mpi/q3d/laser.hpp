#pragma once

#include "emf.hpp"
#include "util/complex.hpp"

namespace laser {

using complex_t        = util::complex64;
using complex_double_t = util::complex128;

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
     * @note The E field is polarized along (cos φ, sin φ) in the transverse
     *       plane, so polarization = 0 corresponds to an E field along x
     *       (θ = 0). This will only be used if both cos_pol and sin_pol are 0.
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
    pulse() : start(0), fwhm(0), rise(0), flat(0), fall(0),
        a0(0), omega0(0),
        polarization(0), cos_pol(0), sin_pol(0), filter(1) {};

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
     * @note Sets the transverse components of the mode m = 1 E and B laser
     *       fields. The longitudinal components are set by lon_z(), called
     *       from add().
     * 
     * @param E     Electric field
     * @param B     Magnetic field
     * @param box   Box size (simulation units)
     * @return      Returns 0 on success
     */
    virtual int launch( grid::tiled_cyl3<complex_t>& E, grid::tiled_cyl3<complex_t>& B, const float2 box ) const = 0;

    /**
     * @brief Sets longitudinal component
     * @note Enforces div.E = div.B = 0 by setting E.z and B.z components.
     *       Guard cells must be valid on entry, and are updated on exit.
     * 
     * @param E     E field
     * @param B     B field
     * @param dx    Cell size in real space
     * @return int  Returns 0 on success
     */
    virtual int lon_z( grid::tiled_cyl3<complex_t> & E, grid::tiled_cyl3<complex_t> & B, float2 dx ) const = 0;

    /**
     * @brief Adds a new laser pulse onto an EMF object
     *
     * @note This calls lon_z() to compute longitudinal components before
     *       adding the fields to mode m = 1
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

    int launch( grid::tiled_cyl3<complex_t> & E, grid::tiled_cyl3<complex_t>& B, const float2 box ) const override;

    // No need to set longitudinal field on plane wave
    int lon_z( grid::tiled_cyl3<complex_t> &, grid::tiled_cyl3<complex_t> &, float2 ) const override { return 0; };

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

    /// @brief Beam waist (simulation units)
    float W0;
    /// @brief Focal plane position (simulation units)
    float focus;

    gaussian() : pulse(), W0(0), focus(0) {};

    void validate() const override;
    int launch( grid::tiled_cyl3<complex_t> & E, grid::tiled_cyl3<complex_t> & B, const float2 box ) const override;
    int lon_z( grid::tiled_cyl3<complex_t> & E, grid::tiled_cyl3<complex_t> & B, float2 dx ) const override;

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
           << ", focus: " << obj.focus;

        return os;
    };
};

}
