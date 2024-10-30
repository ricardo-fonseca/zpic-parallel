#ifndef CATHODE_H_
#define CATHODE_H_

#include "species.h"

class Cathode : public Species {

private:

    /// @brief velocity (beta) of cathode flow
    float vel;

    /// @brief Position of injection particles
    float *d_inj_pos;

    void cathode_np_inject( int * np );

    void cathode_inject( );

    void update_inj_pos();

public:

    /// @brief generalized velocity of cathode flow
    const float ufl;

    /// @brief wall to inject particles from
    edge::pos wall;

    /// @brief Time to start cathode injection
    float start;

    /// @brief Time to end cathode injection
    float end;

    /// @brief temperature of injected particles
    float3 uth;

    /// @brief Reference density for cathode
    float n0;

    Cathode( std::string const name, float const m_q, uint2 const ppc, 
        edge::pos wall, float ufl );

    void initialize( float2 const box, uint2 const ntiles, uint2 const nx,
        float const dt, int const id_, Partition & parallel ) override;
    
    ~Cathode();

    void advance( EMF const &emf, Current &current ) override;

    void advance( Current &current ) override;

    int set_moving_window() override {
        std::cerr << "(*error*) Cathodes cannot be used with moving windows, aborting...\n";
        exit(1);
    }

    void set_udist( UDistribution::Type const & new_udist ) override {
        std::cerr << "(*error*) Cathodes do not support the set_udist() method, use the uth parameter instead\n";
        exit(1);
    }

    void set_density( Density::Profile const & new_density ) override {
        std::cerr << "(*error*) Cathodes do not support the set_density() method, use the n0 parameter instead\n";
        exit(1);
    }

    virtual void inject() override;

    virtual void inject( bnd<unsigned int> range )  override;

    virtual void np_inject( bnd<unsigned int> range, int * np ) override;
};

#endif