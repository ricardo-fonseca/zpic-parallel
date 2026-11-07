#pragma once

#include <cmath>
#include <memory>

#include "util/memory.cuh"
#include "cyl_part/particles.hpp"

#include "cyl3/cyl3.cuh"

namespace density {

    /**
     * @brief Base class for plasma density profiles
     * 
     * All injection methods share the same arguments:
     * 
     * @param part      Particle data object
     * @param norm      Charge normalization constant (inject() only)
     * @param ppc       Number of particles per cell
     * @param dx        Cell size
     * @param ref       Position of local grid on global simulation box in
     *                  simulation units
     * @param range     Cell range in which to inject, in local cell coordinates,
     *                  inclusive on both ends
     * 
     * `np_inject()` writes the number of particles that `inject()` would add to
     * each tile into `np` (one int per local tile, device memory). The two
     * methods are guaranteed to use exactly the same injection test.
     */
    class profile {
        public:

        ///@brief Reference density
        const float n0;

        profile(float const n0) : n0(std::abs(n0)) {};

        virtual std::unique_ptr<profile> clone() const = 0;
        
        virtual void inject( cyl_part::particles & part, float const norm, uint3 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range ) const = 0;
    
        virtual void np_inject( cyl_part::particles & part, uint3 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range, int * np ) const = 0;

        virtual ~ profile() = default;
    };

    /**
     * @brief Zero density (no particles), used to disable injection
     * 
     */
    class none : public profile {

        public:

        none( float const n0) : profile( n0 ) { };

        std::unique_ptr<profile> clone() const override {
            return std::make_unique<none>( n0 );
        };
        void inject( cyl_part::particles &, float const, uint3 const, float2 const, float2 const, bounds_2d<unsigned int> ) const override {
            // no injection
        };
        /// @brief Sets np to 0 on all local tiles (no injection)
        void np_inject( cyl_part::particles & part, uint3 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range, int * np ) const override;
    };


    /**
     * @brief Uniform plasma density
     * 
     */
    class uniform : public profile {

        public:

        uniform( float const n0 ) : profile(n0) { };

        std::unique_ptr<profile> clone() const override {
            return std::make_unique<uniform>( *this );
        };
        void inject( cyl_part::particles & part, float const norm, uint3 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range ) const override;
        void np_inject( cyl_part::particles & part, uint3 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range, int * np ) const override;
    };

    /**
     * @brief step (Heaviside) plasma density
     * 
     * Uniform plasma density for positions >= `pos`. Can be set in either z or
     * r coordinates
     */
    class step : public profile {
        public:

        ///@brief step position
        const float pos;
        ///@brief step direction
        const coord::cyl dir;

        step( coord::cyl dir, float const n0, float const pos )  : profile(n0), pos(pos), dir(dir) {
            if ( dir == coord::cyl::r &&  pos < 0 ) {
                mpi::fatal( "Invalid radial position, must be >= 0");
            }
        };

        std::unique_ptr<profile> clone() const override {
            return std::make_unique<step>( *this );
        };

        void inject( cyl_part::particles & part, float const norm, uint3 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range ) const override;
        void np_inject( cyl_part::particles & part, uint3 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range, int * np ) const override;
    };

    /**
     * @brief slab plasma density
     * 
     * Uniform plasma density for positions in [`begin`, `end`). Can be set in
     * either z or r coordinates
     * 
     */
    class slab : public profile {
        public:

        ///@brief slab begin position
        const float begin;
        ///@brief slab end position
        const float end;
        ///@brief slab direction
        const coord::cyl dir;

        slab( coord::cyl dir, float const n0, float begin, float end ) : 
            profile(n0), begin(begin), end(end), dir(dir) {
            if (( dir == coord::cyl::r ) && ( begin < 0 || end < 0 )) {
                mpi::fatal("Invalid radial position, must be >= 0");
            }
        };
        
        std::unique_ptr<profile> clone() const override {
            return std::make_unique<slab>( *this );
        };

        void inject( cyl_part::particles & part, float const norm, uint3 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range ) const override;
        void np_inject( cyl_part::particles & part, uint3 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range, int * np ) const override;
    };

    /**
     * @brief Sphere plasma density
     * 
     * Uniform plasma density for positions strictly inside a circle of radius
     * `radius` centered about `center`
     *
     * @note In this geometry this actually corresponds to a 3D torus
     * 
     */
    class sphere : public profile {
        public:

        ///@brief sphere center position
        const float2 center;
        ///@brief sphere radius
        const float radius;

        sphere( float const n0, float2 center, float radius ) : profile(n0), center(center), radius(radius) {
            if ( radius < 0 ) {
                mpi::fatal( "Invalid sphere radius, must be >= 0" );
            }
        };

        std::unique_ptr<profile> clone() const override { 
            return std::make_unique<sphere>( n0, center, radius );
        };
        void inject( cyl_part::particles & part, float const norm, uint3 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range ) const override;
        void np_inject( cyl_part::particles & part, uint3 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range, int * np ) const override;
    };
}
