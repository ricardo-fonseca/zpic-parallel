#pragma once

#include "part/particles.hpp"

namespace udist {

    class type {
        public:
        virtual type * clone() const = 0;
        virtual void set( part::particles & part, unsigned int seed ) const = 0;
        virtual ~type() = default;
    };
    
    /**
     * @brief No momentum distribution, all particles are set to u = 0
     * 
     */
    class none : public type {
        public:
        none * clone() const override { return new none(); };
        void set( part::particles & part, unsigned int seed ) const override ;
    };

    /**
     * @brief Cold momentum distribution, all particles are set to u = ufl
     * 
     */
    class cold : public type {
        public:
        const float3 ufl;
        cold( float3 const ufl ) : ufl(ufl) {};
        cold * clone() const override { return new cold(ufl); };
        void set( part::particles & part, unsigned int seed ) const override ;
    };

    class thermal : public type {
        public:
        const float3 uth;
        const float3 ufl;
        thermal( float3 const uth, float3 const ufl ) : uth(uth), ufl(ufl) {};
        thermal( float3 const uth ) : uth(uth), ufl( make_float3(0,0,0) ) {};

        thermal * clone() const override { return new thermal(uth, ufl); };
        void set( part::particles & part, unsigned int seed ) const override ;
    };

    class thermal_corr : public type {
        
        public:
        const float3 uth;
        const float3 ufl;
        const int npmin;
        thermal_corr( float3 const uth, float3 const ufl, int const npmin = 2 ) : uth(uth), ufl(ufl), npmin(npmin) {
            if ( npmin <= 1 ) {
                mpi::fatal( "Invalid npmin (" + std::to_string(npmin) + " parameter, must be > 1" );
            }
        };

        thermal_corr * clone() const override { return new thermal_corr(uth, ufl,npmin); };
        void set( part::particles & part, unsigned int seed ) const override ;
    };
}
