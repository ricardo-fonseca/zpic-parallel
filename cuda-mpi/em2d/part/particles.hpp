#pragma once

/**
 * @file particles.hpp
 * @brief Particles (2D cartesian)
 * 
 * @note See common/part/particles.hpp for details. The cell index (int2 ix) is
 *       always included and must not be listed.
 */

#define PARTICLES_NAMESPACE part

#define PARTICLES_QUANTITIES( X ) \
    X( float2, x, "Position inside cell normalized to cell size [-0.5,0.5)" ) \
    X( float3, u, "Generalized velocity" )

#include "../common/part/particles.hpp"
