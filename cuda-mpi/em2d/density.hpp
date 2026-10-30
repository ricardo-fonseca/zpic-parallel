#pragma once

#include <cmath>
#include <memory>

#include "util/memory.cuh"
#include "part/particles.hpp"

namespace density {

    /**
     * @brief Base class for plasma density profiles
     * 
     * All injection methods share the same arguments:
     * 
     * @param part      Particle data object
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
        
        virtual void inject( part::particles & part, uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range ) const = 0;
    
        virtual void np_inject( part::particles & part, uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range, int * np ) const = 0;

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
        void inject( part::particles & part, uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range ) const override {
            // no injection
        };
        void np_inject( part::particles & part, uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range, int * np ) const override {
            // no injection
            auto ntiles = part.get_local_ntiles();
            gpu::device::zero(np, static_cast<size_t>( ntiles.x ) * ntiles.y );
        };
    };


    /**
     * @brief Uniform plasma density
     * 
     */
    class uniform : public profile {

        public:

        uniform( float const n0 ) : profile(n0) { };

        std::unique_ptr<profile> clone() const override {
            return std::make_unique<uniform>( n0 );
        };
        void inject( part::particles & part, uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range ) const override;
        void np_inject( part::particles & part, uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range, int * np ) const override;
    };

    /**
     * @brief step (Heaviside) plasma density
     * 
     * Uniform plasma density for positions >= `pos`. Can be set in either x or
     * y coordinates
     */
    class step : public profile {
        public:

        ///@brief step position
        const float pos;
        ///@brief step direction
        const coord::cart dir;

        step( coord::cart dir, float const n0, float const pos ) : profile(n0), pos(pos), dir(dir) {};

        std::unique_ptr<profile> clone() const override {
            return std::make_unique<step>( dir, n0, pos );
        };

        void inject( part::particles & part, uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range ) const override;
        void np_inject( part::particles & part, uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range, int * np ) const override;
    };

    /**
     * @brief slab plasma density
     * 
     * Uniform plasma density for positions in [`begin`, `end`). Can be set in
     * either x or y coordinates
     * 
     */
    class slab : public profile {
        public:

        ///@brief slab begin position
        const float begin;
        ///@brief slab end position
        const float end;
        ///@brief slab direction
        const coord::cart dir;

        slab( coord::cart dir, float const n0, float begin, float end ) : profile(n0), begin(begin), end(end), dir(dir) {};
        
        std::unique_ptr<profile> clone() const override {
            return std::make_unique<slab>( dir, n0, begin, end );
        };

        void inject( part::particles & part, uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range ) const override;
        void np_inject( part::particles & part, uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range, int * np ) const override;
    };

    /**
     * @brief sphere plasma density
     * 
     * Uniform plasma density for positions strictly inside a circle of radius
     * `radius` centered about `center`
     * 
     */
    class sphere : public profile {
        public:

        ///@brief sphere center position
        const float2 center;
        ///@brief sphere radius
        const float radius;

        sphere( float const n0, float2 center, float radius ) : profile(n0), center(center), radius(radius) {};

        std::unique_ptr<profile> clone() const override { 
            return std::make_unique<sphere>( n0, center, radius );
        };
        void inject( part::particles & part, uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range ) const override;
        void np_inject( part::particles & part, uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range, int * np ) const override;
    };

}
