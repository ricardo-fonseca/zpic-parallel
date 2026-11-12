#pragma once

/**
 * @file particles.hpp
 * @brief Particles (cylindrical modes)
 * 
 * @note See common/particles.hpp for details. The cell index (int2 ix) is
 *       always included and must not be listed.
 */

#define PARTICLES_NAMESPACE cyl_part

#define PARTICLES_QUANTITIES( X ) \
    X( float2, x,  "Position inside cell normalized to cell size [-0.5,0.5)" ) \
    X( float3, u,  "Generalized velocity" ) \
    X( float,  q,  "Individual particle charge" ) \
    X( float2, th, "Angular position stored as { cos(θ), sin(θ) }" )

#include "../common/part/particles.hpp"
