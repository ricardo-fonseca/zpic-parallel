#pragma once

#include "part/particles.hpp"

namespace density {

    class profile {
        public:

        ///@brief Reference density
        const float n0;

        profile(float const n0) : n0(std::abs(n0)) {};

        virtual profile * clone() const = 0;
        
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

        none * clone() const override {
            return new none( n0 );
        };
        void inject( part::particles & part, uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range ) const override {
            // no injection
        };
        void np_inject( part::particles & part, uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range, int * np ) const override {
            // no injection
            part.zero_np();
        };
    };


    /**
     * @brief Uniform plasma density
     * 
     */
    class uniform : public profile {

        public:

        uniform( float const n0 ) : profile(n0) { };

        uniform * clone() const override {
            return new uniform(n0);
        };
        void inject( part::particles & part, uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range ) const override;
        void np_inject( part::particles & part, uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range, int * np ) const override;
    };

    /**
     * @brief step (Heavyside) plasma density
     * 
     * Uniform plasma density after a given position. Can be set in either x or y coordinates
     */
    class step : public profile {
        public:

        ///@brief step position
        const float pos;
        ///@brief step direction
        const coord::cart dir;

        step( coord::cart dir, float const n0, float const pos ) : profile(n0), pos(pos), dir(dir) {};

        step * clone() const override {
            return new step( dir, n0, pos );
        };

        void inject( part::particles & part, uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range ) const override;
        void np_inject( part::particles & part, uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range, int * np ) const override;
    };

    /**
     * @brief slab plasma density
     * 
     * Uniform plasma density inside given 1D range. Can be set in either x or y coordinates
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
        
        slab * clone() const override {
            return new slab( dir, n0, begin, end );
        };

        void inject( part::particles & part, uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range ) const override;
        void np_inject( part::particles & part, uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range, int * np ) const override;
    };

    /**
     * @brief sphere plasma density
     * 
     * Uniform plasma density centered about a given position
     * 
     */
    class sphere : public profile {
        public:

        ///@brief sphere center position
        const float2 center;
        ///@brief sphere radius
        const float radius;

        sphere( float const n0, float2 center, float radius ) : profile(n0), center(center), radius(radius) {};

        sphere * clone() const override { 
            return new sphere(n0, center, radius);
        };
        void inject( part::particles & part, uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range ) const override;
        void np_inject( part::particles & part, uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range, int * np ) const override;
    };

}
