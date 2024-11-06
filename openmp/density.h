#ifndef DENSITY_H_
#define DENSITY_H_

#include "zpic.h"
#include "particles.h"

#include <iostream>


namespace Density {

    class Profile {
        public:

        const float n0;

        Profile(float const n0) : n0(std::abs(n0)) {};

        virtual Profile * clone() const = 0;
        
        virtual void inject( Particles & part, uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const = 0;
    
        virtual void np_inject( Particles & part, uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range, int * np ) const = 0;

        virtual ~ Profile() = default;
    };

    /**
     * @brief Zero density (no particles), used to disable injection
     * 
     */
    class None : public Profile {

        public:

        None( float const n0) : Profile( n0 ) { };

        None * clone() const override {
            return new None( n0 );
        };
        void inject( Particles & part, uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const override {
            // no injection
        };
        void np_inject( Particles & part, uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range, int * np ) const override {
            // no injection
            part.zero_np();
        };
    };


    /**
     * @brief Uniform plasma density
     * 
     */
    class Uniform : public Profile {

        public:

        Uniform( float const n0 ) : Profile(n0) { };

        Uniform * clone() const override {
            return new Uniform(n0);
        };
        void inject( Particles & part, uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const override;
        void np_inject( Particles & part, uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range, int * np ) const override;
    };

    /**
     * @brief Step (Heavyside) plasma density
     * 
     * Uniform plasma density after a given position. Can be set in either x or y coordinates
     */
    class Step : public Profile {
        public:

        const float pos;
        const coord::cart dir;

        Step( coord::cart dir, float const n0, float const pos ) : Profile(n0), pos(pos), dir(dir) {};

        Step * clone() const override {
            return new Step( dir, n0, pos );
        };

        void inject( Particles & part, uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const override;
        void np_inject( Particles & part, uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range, int * np ) const override;
    };

    /**
     * @brief Slab plasma density
     * 
     * Uniform plasma density inside given 1D range. Can be set in either x or y coordinates
     * 
     */
    class Slab : public Profile {
        public:

        const float begin, end;
        const coord::cart dir;

        Slab( coord::cart dir, float const n0, float begin, float end ) : Profile(n0), begin(begin), end(end), dir(dir) {};
        
        Slab * clone() const override {
            return new Slab( dir, n0, begin, end );
        };

        void inject( Particles & part, uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const override;
        void np_inject( Particles & part, uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range, int * np ) const override;
    };

    /**
     * @brief Sphere plasma density
     * 
     * Uniform plasma density centered about a given position
     * 
     */
    class Sphere : public Profile {
        public:

        const float2 center;
        const float radius;

        Sphere( float const n0, float2 center, float radius ) : Profile(n0), center(center), radius(radius) {};

        Sphere * clone() const override { 
            return new Sphere(n0, center, radius);
        };
        void inject( Particles & part, uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const override;
        void np_inject( Particles & part, uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range, int * np ) const override;
    };

}

#endif
