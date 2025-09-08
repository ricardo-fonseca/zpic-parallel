#ifndef UDIST_H_
#define UDIST_H_

#include "zpic.h"

#include <iostream>
#include "particles.h"

namespace UDistribution {

    class Type {
        public:
        virtual Type * clone() const = 0;
        virtual void set( Particles & part, unsigned int seed ) const = 0;
        virtual ~Type() = default;
    };
    
    /**
     * @brief No momentum distribution, all particles are set to u = 0
     * 
     */
    class None : public Type {
        public:
        None * clone() const override { return new None(); };
        void set( Particles & part, unsigned int seed ) const override ;
    };

    /**
     * @brief Cold momentum distribution, all particles are set to u = ufl
     * 
     */
    class Cold : public Type {
        public:
        const float3 ufl;
        Cold( float3 const ufl ) : ufl(ufl) {};
        Cold * clone() const override { return new Cold(ufl); };
        void set( Particles & part, unsigned int seed ) const override ;
    };

    class Thermal : public Type {
        public:
        const float3 uth;
        const float3 ufl;
        Thermal( float3 const uth, float3 const ufl ) : uth(uth), ufl(ufl) {};
        Thermal( float3 const uth ) : uth(uth), ufl( make_float3(0,0,0) ) {};

        Thermal * clone() const override { return new Thermal(uth, ufl); };
        void set( Particles & part, unsigned int seed ) const override ;
    };

    class ThermalCorr : public Type {
        
        public:
        const float3 uth;
        const float3 ufl;
        const int npmin;
        ThermalCorr( float3 const uth, float3 const ufl, int const npmin = 2 ) : uth(uth), ufl(ufl), npmin(npmin) {
            if ( npmin <= 1 ) {
                std::cout << "(*error*) invalid npmin parameter, must be > 1\n";
                exit(1);
            }
        };

        ThermalCorr * clone() const override { return new ThermalCorr(uth, ufl,npmin); };
        void set( Particles & part, unsigned int seed ) const override ;
    };
}

#endif