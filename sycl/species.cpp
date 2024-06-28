/**
 * @file species.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-08-06
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "species.h"

#include <iostream>

/**
 * This is required for using bnd<species::bc::type> inside a Sycl kernel
 */
template<>
struct sycl::is_device_copyable<bnd<species::bc::type>> : std::true_type {};

/**
 * @brief Construct a new Species object
 * 
 * @param name  Name for the species object (used for diagnostics)
 * @param m_q   Mass over charge ratio
 * @param ppc   Number of particles per cell
 */
Species::Species( std::string const name, float const m_q, uint2 const ppc ):
    ppc(ppc), name(name), m_q(m_q)
{

    // Validate parameters
    if ( m_q == 0 ) {
        std::cerr << "(*error*) Invalid m_q value, must be not 0, aborting...\n";
        exit(1);
    }

    if ( ppc.x < 1 || ppc.y < 1 ) {
        std::cerr << "(*error*) Invalid ppc value, must be >= 1 in all directions\n";
        exit(1);
    }

    // Set default parameters
    density   = new Density::Uniform( 1.0 );
    udist     = new UDistribution::None();
    bc        = species::bc_type (species::bc::periodic);
    push_type = species::boris;

    // Nullify pointers to data structures
    particles = nullptr;
    tmp = nullptr;
    sort = nullptr;

    d_energy = nullptr;
    d_nmove = nullptr;

    queue = nullptr;
}


/**
 * @brief Initialize data structures and inject particles
 * 
 * @param box_      Simulation global box size
 * @param ntiles    Number of tiles
 * @param nx        Grid size per tile
 * @param dt_       Time step
 * @param id_       Species unique id
 */
void Species::initialize( float2 const box_, uint2 const ntiles, uint2 const nx,
    float const dt_, int const id_, sycl::queue & queue_ ) {
    
    queue = & queue_;

    // Store simulation box size
    box = box_;

    // Store simulation time step
    dt = dt_;

    // Store species id (used by RNG)
    id = id_;

    // Set charge normalization factor
    q = copysign( density->n0 , m_q ) / (ppc.x * ppc.y);
    
    float2 gnx = make_float2( nx.x * ntiles.x, nx.y * ntiles.y );

    // Set cell size
    dx.x = box.x / (gnx.x);
    dx.y = box.y / (gnx.y);

    // Reference number maximum number of particles
    unsigned int max_part = 1.2 * gnx.x * gnx.y * ppc.x * ppc.y;

    particles = new Particles( ntiles, nx, max_part, *queue );
    particles->periodic.x = ( bc.x.lower == species::bc::periodic );
    particles->periodic.y = ( bc.y.lower == species::bc::periodic );

    tmp = new Particles( ntiles, nx, max_part, *queue );
    sort = new ParticleSort( ntiles, max_part, *queue );
    np_inj = device::malloc<int>( ntiles.x * ntiles.y, *queue );

    // Initialize energy diagnostic
    d_energy = device::malloc<double>( 1, *queue );
    device::zero( d_energy, 1, *queue );

    d_nmove = device::malloc<uint64_t>( 1, *queue );
    device::zero( d_nmove, 1, *queue );

    // Reset iteration numbers
    iter = 0;

    // Inject initial distribution

    // Count particles to inject and store in particles -> offset
    np_inject( particles -> g_range(), particles -> offset );

    // Do an exclusive scan to get the required offsets
    device::exscan_add( particles -> offset, ntiles.x * ntiles.y, *queue );

    // Inject the particles
    inject( particles -> g_range() );

    // Set inital velocity distribution
    udist -> set( *particles, id );

    // Test initial particle set, remove from production code
    // particles -> validate( name + ": after initial injection");
}

/**
 * @brief Destroy the Species object
 * 
 */
Species::~Species() {
    device::free( np_inj, *queue );
    device::free( d_energy, *queue );
    device::free( d_nmove, *queue );

    delete( tmp );
    delete( sort );
    delete( particles );
    delete( density );
    delete( udist );
};


/**
 * @brief Returns reciprocal Lorentz gamma factor
 * 
 * $ \frac{1}{\sqrt{u_x^2 + u_y^2 + u_z^2 + 1 }} $
 * 
 * @param u         Generalized momentum in units of c
 * @return float    Reciprocal Lorentz gamma factor
 */
inline
float rgamma( const float3 u ) {
    return 1.0f/sycl::sqrt( sycl::fma( u.z, u.z, sycl::fma( u.y, u.y, sycl::fma( u.x, u.x, 1.0f ) ) ) );
}


/**
 * @brief Inject particles in the complete simulation box
 * 
 */
void Species::inject( ) {

    float2 ref = make_float2( moving_window.motion(), 0 );

    density -> inject( *particles, ppc, dx, ref, particles -> g_range() );
}

/**
 * @brief Inject particles in a specific cell range
 * 
 */
void Species::inject( bnd<unsigned int> range ) {

    float2 ref = make_float2( moving_window.motion(), 0 );

    density -> inject( *particles, ppc, dx, ref, range );
}

/**
 * @brief Gets the number of particles that would be injected in a specific cell range
 * 
 * Although the routine only considers injection in a specific range, the
 * number of particles to be injected is calculated on all tiles (returning
 * zero on those, as expected)
 * 
 * @param range 
 * @param np        (device pointer) Number of particles to inject in each tile
 */
void Species::np_inject( bnd<unsigned int> range, int * np ) {

    float2 ref = make_float2( moving_window.motion(), 0 );

    density -> np_inject( *particles, ppc, dx, ref, range, np );
}

/**
 * @brief Physical boundary conditions for the x direction 
 * 
 * @param ntiles    Number of tiles
 * @param tile_idx  Tile index
 * @param tiles     Particle tile information
 * @param data      Particle data
 * @param nx        Tile grid size
 * @param bc        Boundary condition
 */
void species_bcx(
    sycl::nd_item<2> & it,
    ParticleData const part,
    species::bc_type const bc ) 
{
    const uint2 ntiles  = part.ntiles;
        const auto tile_idx = make_uint2( 
        it.get_group(0) * (ntiles.x-1),
        it.get_group(1)
    );

    const int nx = part.nx.x;
    
    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    const int part_offset    = part.offset[ tid ];
    const int np             = part.np[ tid ];
    int2   * __restrict__ ix = &part.ix[ part_offset ];
    float2 * __restrict__ x  = &part.x[ part_offset ];
    float3 * __restrict__ u  = &part.u[ part_offset ];

    if ( tile_idx.x == 0 ) {
        // Lower boundary
        switch( bc.x.lower ) {
        case( species::bc::reflecting ) :
            for( int i = it.get_local_id(0); i < np; i+= it.get_local_range(0) ) {
                if( ix[i].x < 0 ) {
                    ix[i].x += 1;
                    x[i].x = -x[i].x;
                    u[i].x = -u[i].x;
                }
            }
            break;
        default:
            break;
        }
    } else {
        // Upper boundary
        switch( bc.x.upper ) {
        case( species::bc::reflecting ) :
            for( int i = it.get_local_id(0); i < np; i+= it.get_local_range(0) ) {
                if( ix[i].x >=  nx ) {
                    ix[i].x -= 1;
                    x[i].x = -x[i].x;
                    u[i].x = -u[i].x;
                }
            }
            break;
        default:
            break;
        }
    }

}

/**
 * @brief Physical boundary conditions for the y direction 
 * 
 * @param ntiles    Number of tiles
 * @param tile_idx  Tile index
 * @param tiles     Particle tile information
 * @param data      Particle data
 * @param nx        Tile grid size
 * @param bc        Boundary condition
 */
void species_bcy(
    sycl::nd_item<2> & it,
    ParticleData const part,
    species::bc_type const bc ) 
{
    const uint2 ntiles  = part.ntiles;

    const auto tile_idx = make_uint2( 
        it.get_group(0) ,
        it.get_group(1) * (ntiles.y-1)
    );

    const int ny = part.nx.y;

    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    const int part_offset    = part.offset[ tid ];
    const int np             = part.np[ tid ];
    int2   * __restrict__ ix = &part.ix[ part_offset ];
    float2 * __restrict__ x  = &part.x[ part_offset ];
    float3 * __restrict__ u  = &part.u[ part_offset ];

    if ( tile_idx.y == 0 ) {
        // Lower boundary
        switch( bc.y.lower ) {
        case( species::bc::reflecting ) :
            for( int i = it.get_local_id(0); i < np; i+= it.get_local_range(0) ) {
                if( ix[i].y < 0 ) {
                    ix[i].y += 1;
                    x[i].y = -x[i].y;
                    u[i].y = -u[i].y;
                }
            }
            break;
        default:
            break;
        }
    } else {
        // Upper boundary
        switch( bc.y.upper ) {
        case( species::bc::reflecting ) :
            for( int i = it.get_local_id(0); i < np; i+= it.get_local_range(0) ) {
                if( ix[i].y >=  ny ) {
                    ix[i].y -= 1;
                    x[i].y = -x[i].y;
                    u[i].y = -u[i].y;
                }
            }
            break;
        default:
            break;
        }
    }
}

/**
 * @brief Processes "physical" boundary conditions
 * 
 */
void Species::process_bc() {

    // x boundaries
    if ( bc.x.lower > species::bc::periodic || bc.x.upper > species::bc::periodic ) {
        
        queue->submit([&](sycl::handler &h) {

            const ParticleData part = *particles;
            auto bc = this -> bc;

            auto ntiles = particles -> ntiles;

            // 8×1 work items per group
            sycl::range<2> local{ 8, 1 };

            // 2 × ntiles.y groups
            sycl::range<2> global{ 2, ntiles.y };

            h.parallel_for( 
                sycl::nd_range{ global * local, local },
                [=](sycl::nd_item<2> it) { 
                species_bcx ( it, part, bc );
            });
        });
        queue->wait();
    }

    // y boundaries
    if ( bc.y.lower > species::bc::periodic || bc.y.upper > species::bc::periodic ) {
        queue->submit([&](sycl::handler &h) {

            const ParticleData part = *particles;
            auto bc = this -> bc;

            auto ntiles = particles -> ntiles;

            // 8×1 work items per group
            sycl::range<2> local{ 8, 1 };

            // ntiles.x × 2 groups
            sycl::range<2> global{ ntiles.x, 2 };

            h.parallel_for( 
                sycl::nd_range{ global * local, local },
                [=](sycl::nd_item<2> it) { 

                species_bcy ( it, part, bc );
            });
        });
        queue->wait();
    }
}

/**
 * @brief Free stream particles 1 iteration
 * 
 * No acceleration or current deposition is performed. Used for debug purposes.
 * 
 */
void Species::advance( ) {

    // Advance positions
    move();
    
    // Process physical boundary conditions
    process_bc();
    
    // Sort particles according to tile
    particles -> tile_sort( *tmp, *sort );

    // Increase internal iteration number
    iter++;
}

/**
 * @brief Free-stream particles 1 iteration
 * 
 * This routine will:
 * 1. Advance positions and deposit current
 * 2. Process boundary conditions
 * 3. Sort particles according to tiles
 * 
 * @param emf       EM fields
 * @param current   Electric durrent density
 */
void Species::advance( Current &current ) {

    // Advance positions and deposit current
    move( current.J );

    // Process physical boundary conditions
    process_bc();
    
    // Sort particles according to tile
    particles -> tile_sort( *tmp, *sort );

    // Increase internal iteration number
    iter++;
}

/**
 * @brief Advance particles 1 iteration
 * 
 * This routine will:
 * 1. Advance momenta
 * 2. Advance positions and deposit current
 * 3. Process boundary conditions
 * 4. Sort particles according to tiles
 * 
 * @param emf       EM fields
 * @param current   Electric durrent density
 */
void Species::advance( EMF const &emf, Current &current ) {

    // Advance momenta
    push( emf.E, emf.B );

    // Advance positions and deposit current
    move( current.J );

    // Process physical boundary conditions
    process_bc();
    
    // Sort particles according to tile
    particles -> tile_sort( *tmp, *sort );

    // Increase internal iteration number
    iter++;
}

/**
 * @brief Advance particles 1 iteration
 * 
 * This routine will:
 * 1. Advance momenta
 * 2. Advance positions and deposit current
 * 3. Process boundary conditions
 * 4. Handle moving window algorith,
 * 5. Sort particles according to tiles
 * 
 * @param emf       EM fields
 * @param current   Electric durrent density
 */
void Species::advance_mov_window( EMF const &emf, Current &current ) {

    // Advance momenta
    push( emf.E, emf.B );

    if ( moving_window.needs_move( (iter+1) * dt ) ) {

        // Advance positions, deposit current and shift particles
        move( current.J, make_int2(-1,0) );

        // Process boundary conditions
        process_bc();

        // Find range where new particles need to be injected
        uint2 g_nx = particles -> gnx;
        bnd<unsigned int> range;
        range.x = { .lower = g_nx.x - 1, .upper = g_nx.x - 1 };
        range.y = { .lower = 0, .upper = g_nx.y - 1 };

        // Count new particles to be injected
        np_inject( range, np_inj );

        // Sort particles over tiles, leaving room for new particles to be injected
        particles -> tile_sort( *tmp, *sort, np_inj );

        // Inject new particles
        inject( range );

        // Advance moving window
        moving_window.advance();

    } else {
        
        // Advance positions and deposit current
        move( current.J );

        // Process boundary conditions
        process_bc();

        // Sort particles over tiles
        particles -> tile_sort( *tmp, *sort );
    }

    // Increase internal iteration number
    iter++;


}


/**
 * @brief Deposit (charge conserving) current for 1 segment inside a cell
 * 
 * @param ix        Particle cell
 * @param x0        Initial particle position
 * @param x1        Final particle position
 * @param qnx       Normalization values for in plane current deposition
 * @param qvz       Out of plane current
 * @param J         current(J) grid (should be in shared memory)
 * @param stride    current(J) grid stride
 */
inline void dep_current_seg(
    const int2 ix, const float2 x0, const float2 x1,
    const float2 qnx, const float qvz,
    float3 * __restrict__ J, const int stride )
{
    const float S0x0 = 0.5f - x0.x;
    const float S0x1 = 0.5f + x0.x;

    const float S1x0 = 0.5f - x1.x;
    const float S1x1 = 0.5f + x1.x;

    const float S0y0 = 0.5f - x0.y;
    const float S0y1 = 0.5f + x0.y;

    const float S1y0 = 0.5f - x1.y;
    const float S1y1 = 0.5f + x1.y;

    const float wl1 = qnx.x * (x1.x - x0.x);
    const float wl2 = qnx.y * (x1.y - x0.y);
    
    const float wp10 = 0.5f*(S0y0 + S1y0);
    const float wp11 = 0.5f*(S0y1 + S1y1);
    
    const float wp20 = 0.5f*(S0x0 + S1x0);
    const float wp21 = 0.5f*(S0x1 + S1x1);

/*
    float * __restrict__ const Js = (float *) (&J[ix.x   + stride* ix.y]);
    int const stride3 = 3 * stride;

    // Reorder for linear access
    //                   y    x  fc

    // When using more than 1 thread per tile all of these need to be atomic
    device::local::atomicAdd(&Js[       0 + 0 + 0 ], wl1 * wp10 );
    device::local::atomicAdd(&Js[       0 + 0 + 1 ], wl2 * wp20 );
    device::local::atomicAdd(&Js[       0 + 0 + 2 ], qvz * ( S0x0 * S0y0 + S1x0 * S1y0 + (S0x0 * S1y0 - S1x0 * S0y0)/2.0f ));

    device::local::atomicAdd(&Js[       0 + 3 + 1 ], wl2 * wp21 );
    device::local::atomicAdd(&Js[       0 + 3 + 2 ], qvz * ( S0x1 * S0y0 + S1x1 * S1y0 + (S0x1 * S1y0 - S1x1 * S0y0)/2.0f ));

    device::local::atomicAdd(&Js[ stride3 + 0 + 0 ], wl1 * wp11 );
    device::local::atomicAdd(&Js[ stride3 + 0 + 2 ], qvz * ( S0x0 * S0y1 + S1x0 * S1y1 + (S0x0 * S1y1 - S1x0 * S0y1)/2.0f ));
    device::local::atomicAdd(&Js[ stride3 + 3 + 2 ], qvz * ( S0x1 * S0y1 + S1x1 * S1y1 + (S0x1 * S1y1 - S1x1 * S0y1)/2.0f ));
*/

    float3 * __restrict__ const Js = &J[ stride* ix.y + ix.x ];
    device::local::atomicAdd(&Js[      0 + 0 ].x, wl1 * wp10 );
    device::local::atomicAdd(&Js[      0 + 0 ].y, wl2 * wp20 );
    device::local::atomicAdd(&Js[      0 + 0 ].z, qvz * ( S0x0 * S0y0 + S1x0 * S1y0 + (S0x0 * S1y0 - S1x0 * S0y0)/2.0f ));

    device::local::atomicAdd(&Js[      0 + 1 ].y, wl2 * wp21 );
    device::local::atomicAdd(&Js[      0 + 1 ].z, qvz * ( S0x1 * S0y0 + S1x1 * S1y0 + (S0x1 * S1y0 - S1x1 * S0y0)/2.0f ));

    device::local::atomicAdd(&Js[ stride + 0 ].x, wl1 * wp11 );
    device::local::atomicAdd(&Js[ stride + 0 ].z, qvz * ( S0x0 * S0y1 + S1x0 * S1y1 + (S0x0 * S1y1 - S1x0 * S0y1)/2.0f ));
    device::local::atomicAdd(&Js[ stride + 1 ].z, qvz * ( S0x1 * S0y1 + S1x1 * S1y1 + (S0x1 * S1y1 - S1x1 * S0y1)/2.0f ));

}

/**
 * @brief Moves particles and deposit current
 * 
 * Current will be accumulated on existing data
 * 
 * @param current   Current grid
 */
void Species::move( vec3grid<float3> * J )
{
    const float2 dt_dx = make_float2(
        dt / dx.x,
        dt / dx.y
    );

    const float2 qnx = make_float2(
        q * dx.x / dt,
        q * dx.y / dt
    );


    const ParticleData part = *particles;

    const int2 ntiles = make_int2( part.ntiles.x, part.ntiles.y );
    const int tile_vol = J -> tile_vol;
    const auto current_offset = J -> offset;
    const auto d_current = J -> d_buffer;
    const int ystride = J -> ext_nx.x;

    const auto q = this -> q;
    auto d_nmove = this -> d_nmove;

    // 512×1 work items per group
    sycl::range<2> local( 512, 1 );

    // ntiles.x × ntiles.y groups
    sycl::range<2> grid( ntiles.x, ntiles.y );


    queue->submit([&](sycl::handler &h) {

        /// @brief [shared] Local copy of current density
        auto J_local = sycl::local_accessor< float3, 1 > ( tile_vol, h );

        h.parallel_for( 
            sycl::nd_range{ grid * local, local },
            [=](sycl::nd_item<2> it) { 
            
            const int2 tile_idx = make_int2( it.get_group(0), it.get_group(1) );

            // Zero local current buffer
            for( auto i = it.get_local_id(0); i < tile_vol; i+= it.get_local_range(0) ) 
                J_local[i] = make_float3(0,0,0);

            float3 * J = & J_local[ current_offset ];
            it.barrier();

            // Move particles and deposit current
            const int tile_id   = tile_idx.y * ntiles.x + tile_idx.x;

            const auto tile_off        = part.offset[ tile_id ];
            const auto tile_np         = part.np[ tile_id ];
            int2   * __restrict__ ix  = &part.ix[ tile_off ];
            float2 * __restrict__ x   = &part.x[ tile_off ];
            float3 * __restrict__ u   = &part.u[ tile_off ];

            for( int i = it.get_local_id(0); i < tile_np; i+= it.get_local_range(0) ) {
                float3 pu = u[i];
                float2 const x0 = x[i];
                int2   const ix0 =ix[i];

                // Get 1 / Lorentz gamma
                float const rg = rgamma( pu );

                // Get particle motion
                float2 const delta = make_float2(
                    dt_dx.x * rg * pu.x,
                    dt_dx.y * rg * pu.y
                );

                // Advance position
                float2 x1 = make_float2(
                    x0.x + delta.x,
                    x0.y + delta.y
                );

                // Check for cell crossings
                int2 const deltai = make_int2(
                    ((x1.x >= 0.5f) - (x1.x < -0.5f)),
                    ((x1.y >= 0.5f) - (x1.y < -0.5f))
                );

                // Split trajectories:
                int nvp = 1;
                int2 v0_ix; float2 v0_x0, v0_x1; float v0_qvz;
                int2 v1_ix; float2 v1_x0, v1_x1; float v1_qvz;
                int2 v2_ix; float2 v2_x0, v2_x1; float v2_qvz;

                float eps, xint, yint;
                float qvz = q * pu.z * rg * 0.5f;

                // Initial position is the same on all cases
                v0_ix = ix0; v0_x0 = x0;

                switch( 2*(deltai.x != 0) + (deltai.y != 0) )
                {
                case(0): // no splits
                    v0_x1 = x1; v0_qvz = qvz;
                    break;

                case(1): // only y crossing
                    nvp++;

                    yint = 0.5f * deltai.y;
                    eps  = ( yint - x0.y ) / delta.y;
                    xint = x0.x + delta.x * eps;

                    v0_x1  = make_float2(xint,yint);
                    v0_qvz = qvz * eps;

                    v1_ix = make_int2( ix0.x, ix0.y  + deltai.y );
                    v1_x0 = make_float2(xint,-yint);
                    v1_x1 = make_float2( x1.x, x1.y  - deltai.y );
                    v1_qvz = qvz * (1-eps);

                    break;

                case(2): // only x crossing
                case(3): // x-y crossing
                    
                    // handle x cross
                    nvp++;
                    xint = 0.5f * deltai.x;
                    eps  = ( xint - x0.x ) / delta.x;
                    yint = x0.y + delta.y * eps;

                    v0_x1 = make_float2(xint,yint);
                    v0_qvz = qvz * eps;

                    v1_ix = make_int2( ix0.x + deltai.x, ix0.y);
                    v1_x0 = make_float2(-xint,yint);
                    v1_x1 = make_float2( x1.x - deltai.x, x1.y );
                    v1_qvz = qvz * (1-eps);

                    // handle additional y-cross, if need be
                    if ( deltai.y ) {
                        float yint2 = 0.5f * deltai.y;
                        nvp++;

                        if ( yint >= -0.5f && yint < 0.5f ) {
                            // y crosssing on 2nd vp
                            eps   = (yint2 - yint) / (x1.y - yint );
                            float xint2 = -xint + (x1.x - xint ) * eps;
                            
                            v2_ix = make_int2( v1_ix.x, v1_ix.y + deltai.y );
                            v2_x0 = make_float2(xint2,-yint2);
                            v2_x1 = make_float2( v1_x1.x, v1_x1.y - deltai.y );
                            v2_qvz = v1_qvz * (1-eps);

                            // Correct other particle
                            v1_x1 = make_float2(xint2,yint2);
                            v1_qvz *= eps;
                        } else {
                            // y crossing on 1st vp
                            eps   = (yint2 - x0.y) / ( yint - x0.y );
                            float xint2 = x0.x + ( xint - x0.x ) * eps;

                            v2_ix = make_int2( v0_ix.x, v0_ix.y + deltai.y );
                            v2_x0 = make_float2( xint2,-yint2);
                            v2_x1 = make_float2( v0_x1.x, v0_x1.y - deltai.y );
                            v2_qvz = v0_qvz * (1-eps);

                            // Correct other particles
                            v0_x1 = make_float2(xint2,yint2);
                            v0_qvz *= eps;

                            v1_ix.y += deltai.y;
                            v1_x0.y -= deltai.y;
                            v1_x1.y -= deltai.y;
                        }
                    }
                    break;
                }

                // Deposit vp current
                               dep_current_seg( v0_ix, v0_x0, v0_x1, qnx, v0_qvz, J, ystride );
                if ( nvp > 1 ) dep_current_seg( v1_ix, v1_x0, v1_x1, qnx, v1_qvz, J, ystride );
                if ( nvp > 2 ) dep_current_seg( v2_ix, v2_x0, v2_x1, qnx, v2_qvz, J, ystride );

                // Correct position and store
                x1.x -= deltai.x;
                x1.y -= deltai.y;
                        
                x[i] = x1;

                // Modify cell and store
                int2 ix1 = make_int2(
                    ix0.x + deltai.x,
                    ix0.y + deltai.y
                );
                ix[i] = ix1;

            }

            it.barrier();

            // Add current to global buffer
            const int current_tile_off = tile_id * tile_vol;

            for( auto i =  it.get_local_id(0); i < tile_vol; i+= it.get_local_range(0) ) 
                d_current[current_tile_off + i] += J_local[i];


            if ( it.get_local_id(0) == 0 ) {
                // Update total particle pushes counter (for performance metrics)
                uint64_t np64 = tile_np;
                device::global::atomicAdd( d_nmove, np64 );
            }
        });
    });
    queue->wait();
}

/**
 * @brief Moves particles and deposit current
 * 
 * Current will be accumulated on existing data
 * 
 * @param current   Current grid
 */
void Species::move( vec3grid<float3> * J, const int2 shift )
{
    const float2 dt_dx = make_float2(
        dt / dx.x,
        dt / dx.y
    );

    const float2 qnx = make_float2(
        q * dx.x / dt,
        q * dx.y / dt
    );

    queue->submit([&](sycl::handler &h) {

        const ParticleData part = *particles;

        auto ntiles = part.ntiles;
        const auto tile_vol = roundup4( J -> tile_vol );
        const auto current_offset = J -> offset;
        const auto d_current = J -> d_buffer;
        const int ystride = J -> ext_nx.x;

        const auto q = this -> q;
        const auto d_nmove = this -> d_nmove;

        /// @brief [shared] Local copy of current density
        auto J_local = sycl::local_accessor< float3, 1 > ( tile_vol, h );

        // 8×1 work items per group
        sycl::range<2> local{ 8, 1 };

        // ntiles.x × ntiles.y groups
        sycl::range<2> global{ ntiles.x, ntiles.y };

        h.parallel_for( 
            sycl::nd_range{ global * local, local },
            [=](sycl::nd_item<2> it) { 

            // Zero local current buffer
            for( auto i = it.get_local_id(0); i < tile_vol; i+= it.get_local_range(0) ) 
                J_local[i] = make_float3(0,0,0);

            float3 * __restrict__ J = & J_local[ current_offset ];

            it.barrier();

            // Move particles and deposit current
            const int2 tile_idx = make_int2( it.get_group(0), it.get_group(1) );
            const int tile_id   = tile_idx.y * ntiles.x + tile_idx.x;

            const int offset          = part.offset[ tile_id ];
            const int np              = part.np[ tile_id ];
            int2   * __restrict__ ix  = &part.ix[ offset ];
            float2 * __restrict__ x   = &part.x[ offset ];
            float3 * __restrict__ u   = &part.u[ offset ];

            for( auto i = it.get_local_id(0); i < np; i+= it.get_local_range(0) ) {
                float3 pu = u[i];
                float2 const x0 = x[i];
                int2   const ix0 =ix[i];

                // Get 1 / Lorentz gamma
                float const rg = rgamma( pu );

                // Get particle motion
                float2 const delta = make_float2(
                    dt_dx.x * rg * pu.x,
                    dt_dx.y * rg * pu.y
                );

                // Advance position
                float2 x1 = make_float2(
                    x0.x + delta.x,
                    x0.y + delta.y
                );

                // Check for cell crossings
                int2 const deltai = make_int2(
                    ((x1.x >= 0.5f) - (x1.x < -0.5f)),
                    ((x1.y >= 0.5f) - (x1.y < -0.5f))
                );

                // Split trajectories:
                int nvp = 1;
                int2 v0_ix; float2 v0_x0, v0_x1; float v0_qvz;
                int2 v1_ix; float2 v1_x0, v1_x1; float v1_qvz;
                int2 v2_ix; float2 v2_x0, v2_x1; float v2_qvz;

                float eps, xint, yint;
                float qvz = q * pu.z * rg * 0.5f;

                // Initial position is the same on all cases
                v0_ix = ix0; v0_x0 = x0;

                switch( 2*(deltai.x != 0) + (deltai.y != 0) )
                {
                case(0): // no splits
                    v0_x1 = x1; v0_qvz = qvz;
                    break;

                case(1): // only y crossing
                    nvp++;

                    yint = 0.5f * deltai.y;
                    eps  = ( yint - x0.y ) / delta.y;
                    xint = x0.x + delta.x * eps;

                    v0_x1  = make_float2(xint,yint);
                    v0_qvz = qvz * eps;

                    v1_ix = make_int2( ix0.x, ix0.y  + deltai.y );
                    v1_x0 = make_float2(xint,-yint);
                    v1_x1 = make_float2( x1.x, x1.y  - deltai.y );
                    v1_qvz = qvz * (1-eps);

                    break;

                case(2): // only x crossing
                case(3): // x-y crossing
                    
                    // handle x cross
                    nvp++;
                    xint = 0.5f * deltai.x;
                    eps  = ( xint - x0.x ) / delta.x;
                    yint = x0.y + delta.y * eps;

                    v0_x1 = make_float2(xint,yint);
                    v0_qvz = qvz * eps;

                    v1_ix = make_int2( ix0.x + deltai.x, ix0.y);
                    v1_x0 = make_float2(-xint,yint);
                    v1_x1 = make_float2( x1.x - deltai.x, x1.y );
                    v1_qvz = qvz * (1-eps);

                    // handle additional y-cross, if need be
                    if ( deltai.y ) {
                        float yint2 = 0.5f * deltai.y;
                        nvp++;

                        if ( yint >= -0.5f && yint < 0.5f ) {
                            // y crosssing on 2nd vp
                            eps   = (yint2 - yint) / (x1.y - yint );
                            float xint2 = -xint + (x1.x - xint ) * eps;
                            
                            v2_ix = make_int2( v1_ix.x, v1_ix.y + deltai.y );
                            v2_x0 = make_float2(xint2,-yint2);
                            v2_x1 = make_float2( v1_x1.x, v1_x1.y - deltai.y );
                            v2_qvz = v1_qvz * (1-eps);

                            // Correct other particle
                            v1_x1 = make_float2(xint2,yint2);
                            v1_qvz *= eps;
                        } else {
                            // y crossing on 1st vp
                            eps   = (yint2 - x0.y) / ( yint - x0.y );
                            float xint2 = x0.x + ( xint - x0.x ) * eps;

                            v2_ix = make_int2( v0_ix.x, v0_ix.y + deltai.y );
                            v2_x0 = make_float2( xint2,-yint2);
                            v2_x1 = make_float2( v0_x1.x, v0_x1.y - deltai.y );
                            v2_qvz = v0_qvz * (1-eps);

                            // Correct other particles
                            v0_x1 = make_float2(xint2,yint2);
                            v0_qvz *= eps;

                            v1_ix.y += deltai.y;
                            v1_x0.y -= deltai.y;
                            v1_x1.y -= deltai.y;
                        }
                    }
                    break;
                }

                // Deposit vp current
                               dep_current_seg( v0_ix, v0_x0, v0_x1, qnx, v0_qvz, J, ystride );
                if ( nvp > 1 ) dep_current_seg( v1_ix, v1_x0, v1_x1, qnx, v1_qvz, J, ystride );
                if ( nvp > 2 ) dep_current_seg( v2_ix, v2_x0, v2_x1, qnx, v2_qvz, J, ystride );

                // Correct position and store
                x1.x -= deltai.x;
                x1.y -= deltai.y;
                        
                x[i] = x1;

                // Modify cell and store
                int2 ix1 = make_int2(
                    ix0.x + deltai.x + shift.x,
                    ix0.y + deltai.y + shift.y
                );
                ix[i] = ix1;

            }

            it.barrier();

            // Add current to global buffer
            const int tile_off = tile_id * tile_vol;

            for( auto i =  it.get_local_id(0); i < tile_vol; i+= it.get_local_range(0) ) 
                d_current[tile_off + i] += J_local[i];


            if ( it.get_local_id(0) == 0 ) {
                // Update total particle pushes counter (for performance metrics)
                uint64_t np64 = np;
                device::global::atomicAdd( d_nmove, np64 );
            }
        });
    });
    queue->wait();
}

/**
 * @brief Moves particles (no current deposition)
 * 
 * This is usually used for test species: species that do not self-consistently
 * influence the simulation
 * 
 * @param current   Current grid
 */
void Species::move( )
{
    const float2 dt_dx = make_float2(
        dt / dx.x,
        dt / dx.y
    );

    queue->submit([&](sycl::handler &h) {

        ParticleData part = *particles;
        const auto d_nmove = this -> d_nmove;

        // 8×1 work items per group
        sycl::range<2> local{ 8, 1 };

        // ntiles.x × 2 groups
        sycl::range<2> global{ part.ntiles.x, part.ntiles.y };

        h.parallel_for( 
            sycl::nd_range{ global * local, local },
            [=](sycl::nd_item<2> it) { 

                const int2 tile_idx = make_int2( it.get_group(0), it.get_group(1) );
                const int tile_id  = tile_idx.y * part.ntiles.x + tile_idx.x;

                const int offset         = part.offset[ tile_id ];
                const int np             = part.np[ tile_id ];
                int2   * __restrict__ ix = &part.ix[ offset ];
                float2 * __restrict__ x  = &part.x[ offset ];
                float3 * __restrict__ u  = &part.u[ offset ];

                for( int i = it.get_local_id(0); i < np; i+= it.get_local_range(0) ) {
                    float3 pu = u[i];
                    float2 x0 = x[i];
                    int2 ix0 =ix[i];

                    // Get 1 / Lorentz gamma
                    float rg = rgamma( pu );

                    // Get particle motion
                    float2 delta = make_float2(
                        dt_dx.x * rg * pu.x,
                        dt_dx.y * rg * pu.y
                    );

                    // Advance position
                    float2 x1 = make_float2(
                        x0.x + delta.x,
                        x0.y + delta.y
                    );

                    // Check for cell crossings
                    int2 deltai = make_int2(
                        ((x1.x >= 0.5f) - (x1.x < -0.5f)),
                        ((x1.y >= 0.5f) - (x1.y < -0.5f))
                    );

                    // Correct position and store
                    x1.x -= deltai.x;
                    x1.y -= deltai.y;
                            
                    x[i] = x1;

                    // Modify cell and store
                    int2 ix1 = make_int2(
                        ix0.x + deltai.x,
                        ix0.y + deltai.y
                    );
                    ix[i] = ix1;
                }

                if ( it.get_local_id(0) == 0 ) { 
                    // Update total particle pushes counter (for performance metrics)
                    uint64_t np64 = np;
                    device::global::atomicAdd( d_nmove, np64 );
                }
        });
    });
    queue->wait();
}

/**
 * @brief Advance momentum using a relativistic Boris pusher.
 * 
 * The momemtum advance in this method is split into 3 parts:
 * 1. Perform half of E-field acceleration
 * 2. Perform full B-field rotation
 * 3. Perform half of E-field acceleration
 * 
 * Note that this implementation (as it is usual in textbooks) uses a
 * linearization of a tangent calculation in the rotation, which may lead
 * to issues for high magnetic fields.
 * 
 * For the future, other, more accurate, rotation algorithms should be used
 * instead, such as employing the full Euler-Rodrigues formula.
 * 
 * Note: uses CUDA intrinsic fma() and rsqrtf() functions
 * 
 * @param tem 
 * @param e 
 * @param b 
 * @param u 
 * @return float3 
 */
inline float3 dudt_boris( const float alpha, float3 e, float3 b, float3 u, double & energy )
{

    // First half of acceleration
    e.x *= alpha;
    e.y *= alpha;
    e.z *= alpha;

    float3 ut = make_float3( 
        u.x + e.x,
        u.y + e.y,
        u.z + e.z
    );

    {
        const float utsq = sycl::fma( ut.z, ut.z, sycl::fma( ut.y, ut.y, ut.x * ut.x ) );
        const float gamma = sycl::sqrt( 1.0f + utsq );
        
        // Get time centered energy
        energy += utsq / (gamma + 1.0f);

        // Time centered \alpha / \gamma
        const float alpha_gamma = alpha / gamma;

        // Rotation
        b.x *= alpha_gamma;
        b.y *= alpha_gamma;
        b.z *= alpha_gamma;
    }

    u.x = sycl::fma( b.z, ut.y, ut.x );
    u.y = sycl::fma( b.x, ut.z, ut.y );
    u.z = sycl::fma( b.y, ut.x, ut.z );

    u.x = sycl::fma( -b.y, ut.z, u.x );
    u.y = sycl::fma( -b.z, ut.x, u.y );
    u.z = sycl::fma( -b.x, ut.y, u.z );

    {
        const float otsq = 2.0f / 
            sycl::fma( b.z, b.z, sycl::fma( b.y, b.y, sycl::fma( b.x, b.x, 1.0f ) ) );
        
        b.x *= otsq;
        b.y *= otsq;
        b.z *= otsq;
    }

    ut.x = sycl::fma( b.z, u.y, ut.x );
    ut.y = sycl::fma( b.x, u.z, ut.y );
    ut.z = sycl::fma( b.y, u.x, ut.z );

    ut.x = sycl::fma( -b.y, u.z, ut.x );
    ut.y = sycl::fma( -b.z, u.x, ut.y );
    ut.z = sycl::fma( -b.x, u.y, ut.z );

    // Second half of acceleration
    ut.x += e.x;
    ut.y += e.y;
    ut.z += e.z;

    return ut;
}


/**
 * @brief Advance memntum using a relativistic Boris pusher for high magnetic fields
 * 
 * This is similar to the dudt_boris method above, but the rotation is done using
 * using an exact Euler-Rodriguez method.2
 * 
 * @param tem 
 * @param e 
 * @param b 
 * @param u 
 * @return float3 
 */
inline float3 dudt_boris_euler( const float alpha, float3 e, float3 b, float3 u, double & energy )
{

    // First half of acceleration
    e.x *= alpha;
    e.y *= alpha;
    e.z *= alpha;

    float3 ut = make_float3( 
        u.x + e.x,
        u.y + e.y,
        u.z + e.z
    );

    {
        const float utsq = sycl::fma( ut.z, ut.z, sycl::fma( ut.y, ut.y, ut.x * ut.x ) );
        const float gamma = sycl::sqrt( 1.0f + utsq );
        
        // Get time centered energy
        energy += utsq / (gamma + 1.0f);
        
        // Time centered 2 * \alpha / \gamma
        float const alpha2_gamma = ( alpha * 2 ) / gamma ;

        b.x *= alpha2_gamma;
        b.y *= alpha2_gamma;
        b.z *= alpha2_gamma;
    }

    {
        float const bnorm = sycl::sqrt(sycl::fma( b.x, b.x, sycl::fma( b.y, b.y, b.z * b.z ) ));
        float const s = -(( bnorm > 0 ) ? sycl::sin( bnorm / 2 ) / bnorm : 1 );

        float const ra = sycl::cos( bnorm / 2 );
        float const rb = b.x * s;
        float const rc = b.y * s;
        float const rd = b.z * s;

        float const r11 =   sycl::fma(ra,ra,rb*rb)-sycl::fma(rc,rc,rd*rd);
        float const r12 = 2*sycl::fma(rb,rc,ra*rd);
        float const r13 = 2*sycl::fma(rb,rd,-ra*rc);

        float const r21 = 2*sycl::fma(rb,rc,-ra*rd);
        float const r22 =   sycl::fma(ra,ra,rc*rc)-sycl::fma(rb,rb,rd*rd);
        float const r23 = 2*sycl::fma(rc,rd,ra*rb);

        float const r31 = 2*sycl::fma(rb,rd,ra*rc);
        float const r32 = 2*sycl::fma(rc,rd,-ra*rb);
        float const r33 =   sycl::fma(ra,ra,rd*rd)-sycl::fma(rb,rb,-rc*rc);

        u.x = sycl::fma( r11, ut.x, sycl::fma( r21, ut.y , r31 * ut.z ));
        u.y = sycl::fma( r12, ut.x, sycl::fma( r22, ut.y , r32 * ut.z ));
        u.z = sycl::fma( r13, ut.x, sycl::fma( r23, ut.y , r33 * ut.z ));
    }


    // Second half of acceleration
    u.x += e.x;
    u.y += e.y;
    u.z += e.z;

    return u;
}


/**
 * @brief Interpolate EM field values at particle position using linear 
 * (1st order) interpolation.
 * 
 * The EM fields are assumed to be organized according to the Yee scheme with
 * the charge defined at lower left corner of the cell
 * 
 * @param E         Pointer to position (0,0) of E field grid
 * @param B         Pointer to position (0,0) of B field grid
 * @param ystride   E and B grids y stride (must be signed)
 * @param ix        Particle cell index
 * @param x         Particle postion inside cell
 * @param e[out]    E field at particle position
 * @param b[out]    B field at particleposition
 */
inline void interpolate_fld( 
    float3 const * const __restrict__ E, 
    float3 const * const __restrict__ B, 
    const int ystride,
    const int2 ix, const float2 x, float3 & e, float3 & b)
{
    const int i = ix.x;
    const int j = ix.y;

    const float s0x = 0.5f - x.x;
    const float s1x = 0.5f + x.x;

    const float s0y = 0.5f - x.y;
    const float s1y = 0.5f + x.y;

    const int hx = x.x < 0;
    const int hy = x.y < 0;

    const int ih = i - hx;
    const int jh = j - hy;

    const float s0xh = (1-hx) - x.x;
    const float s1xh = (  hx) + x.x;

    const float s0yh = (1-hy) - x.y;
    const float s1yh = (  hy) + x.y;


    // Interpolate E field

    e.x = ( E[ih +     j *ystride].x * s0xh + E[ih+1 +     j*ystride].x * s1xh ) * s0y +
          ( E[ih + (j +1)*ystride].x * s0xh + E[ih+1 + (j+1)*ystride].x * s1xh ) * s1y;

    e.y = ( E[i  +     jh*ystride].y * s0x  + E[i+1  +     jh*ystride].y * s1x ) * s0yh +
          ( E[i  + (jh+1)*ystride].y * s0x  + E[i+1  + (jh+1)*ystride].y * s1x ) * s1yh;

    e.z = ( E[i  +     j *ystride].z * s0x  + E[i+1  +     j*ystride].z * s1x ) * s0y +
          ( E[i  + (j +1)*ystride].z * s0x  + E[i+1  + (j+1)*ystride].z * s1x ) * s1y;

    // Interpolate B field
    b.x = ( B[i  +     jh*ystride].x * s0x + B[i+1  +     jh*ystride].x * s1x ) * s0yh +
          ( B[i  + (jh+1)*ystride].x * s0x + B[i+1  + (jh+1)*ystride].x * s1x ) * s1yh;

    b.y = ( B[ih +      j*ystride].y * s0xh + B[ih+1 +      j*ystride].y * s1xh ) * s0y +
          ( B[ih + (j +1)*ystride].y * s0xh + B[ih+1 +  (j+1)*ystride].y * s1xh ) * s1y;

    b.z = ( B[ih +     jh*ystride].z * s0xh + B[ih+1 +     jh*ystride].z * s1xh ) * s0yh +
          ( B[ih + (jh+1)*ystride].z * s0xh + B[ih+1 + (jh+1)*ystride].z * s1xh ) * s1yh;

}

/**
 * @brief CUDA kernel for pushing particles
 * 
 * This kernel will interpolate fields and advance particle momentum using a 
 * relativistic Boris pusher
 * 
 * @param d_tiles       Particle tile information
 * @param d_ix          Particle data (cells)
 * @param d_x           Particle data (positions)
 * @param d_u           Particle data (momenta)
 * @param d_E           E field grid
 * @param d_B           B field grid
 * @param field_offset  Tile offset to field position (0,0)
 * @param ext_nx        E,B tile grid external size
 * @param alpha         Force normalization ( 0.5 * q / m * dt )
 */
template < species::pusher type >
void push_kernel ( 
    sycl::queue & queue,
    ParticleData const part,
    vec3grid<float3> * const E, vec3grid<float3> * const B, 
    float const alpha, double * const __restrict__ d_energy )
{
    const auto ntiles    = part.ntiles;
    const auto field_vol = E->tile_vol; 
    const auto d_E = E -> d_buffer;
    const auto d_B = B -> d_buffer;
    const auto field_offset = E -> offset;
    const int ystride = E -> ext_nx.x;

    // 8×1 work items per group
    sycl::range<2> local{ 64, 1 };

    // ntiles.x × ntiles.y groups
    sycl::range<2> global{ ntiles.x, ntiles.y };

    queue.submit([&](sycl::handler &h) {

        /// @brief [shared] Local copy of E-field
        auto E_local = sycl::local_accessor< float3, 1 > ( field_vol, h );
        /// @brief [shared] Local copy of B-field
        auto B_local = sycl::local_accessor< float3, 1 > ( field_vol, h );

        h.parallel_for( 
            sycl::nd_range{ global * local, local },
            [=](sycl::nd_item<2> it) {
            
            const int2 tile_idx = make_int2( it.get_group(0), it.get_group(1) );
            const int tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
            const auto tile_off = tile_id * field_vol;

            for( auto i = it.get_local_id(0); i < field_vol; i+= it.get_local_range(0)) {
                E_local[i] = d_E[tile_off + i];
                B_local[i] = d_B[tile_off + i];
            }

            float3 const * const __restrict__ E = & E_local[ field_offset ];
            float3 const * const __restrict__ B = & B_local[ field_offset ];

            it.barrier();

            // Push particles
            const int part_offset = part.offset[ tile_id ];
            const int np          = part.np[ tile_id ];
            int2   * __restrict__ ix = &part.ix[ part_offset ];
            float2 * __restrict__ x  = &part.x[ part_offset ];
            float3 * __restrict__ u  = &part.u[ part_offset ];

            double energy = 0;

            for( int i = it.get_local_id(0); i < np; i+= it.get_local_range(0) ) {

                // Interpolate field
                float3 e, b;
                interpolate_fld( E, B, ystride, ix[i], x[i], e, b );
                
                // Advance momentum
                float3 pu = u[i];
                
                if ( type == species::boris ) u[i] = dudt_boris( alpha, e, b, pu, energy );
                if ( type == species::euler ) u[i] = dudt_boris_euler( alpha, e, b, pu, energy );
            }

            // Add up energy from all threads
            auto sg = it.get_sub_group();
            energy = device::subgroup::reduce_add( sg, energy );
            if ( sg.get_local_id() == 0 ) { 
                device::global::atomicAdd( d_energy, energy );
            }
        });
    });
    queue.wait();
}


/**
 * @brief       Accelerates particles using a Boris pusher
 * 
 * @param E     Electric field
 * @param B     Magnetic field
 */
void Species::push( vec3grid<float3> * const E, vec3grid<float3> * const B )
{
    const float alpha = 0.5 * dt / m_q;
    
    device::zero( d_energy, 1, *queue );

    switch( push_type ) {
    case( species :: euler ):
        push_kernel<species::euler>( *queue, *particles, E, B, alpha, d_energy );
        break;
    case( species :: boris ):
        push_kernel<species::boris>( *queue, *particles, E, B, alpha, d_energy );
        break;
    }
}

/**
 * @brief Deposit charge density
 * 
 * @param charge    Charge density grid
 */
void Species::deposit_charge( grid<float> &charge ) const {

    const ParticleData part = *particles;

    const auto q = this -> q;

    const auto ntiles = particles -> ntiles;
    const auto tile_vol = charge.tile_vol;
    const auto charge_offset = charge.offset;
    const auto ystride = charge.ext_nx.x;
    const auto charge_global = charge.d_buffer;

    // Check that local memory can hold a full charge tile
    auto local_mem_size = queue->get_device().get_info<sycl::info::device::local_mem_size>();
    if ( local_mem_size < tile_vol * sizeof( float ) ) {
        std::cerr << "(*error*) Tile size too large " << charge.nx << " (plus guard cells)\n";
        std::cerr << "(*error*) Insufficient local memory (" << local_mem_size << " B) for depositing charge.\n";
        abort();
    }


    queue->submit([&](sycl::handler &h) {

        /// @brief [shared] Local copy of current density
        auto charge_local = sycl::local_accessor< float, 1 > ( tile_vol, h );

        // 8×1 work items per group
        sycl::range<2> local{ 8, 1 };

        // ntiles.x × ntiles.y groups
        sycl::range<2> global{ ntiles.x, ntiles.y };

        sycl::stream out(8192, 1024, h);

        h.parallel_for( 
            sycl::nd_range{ global * local, local },
            [=](sycl::nd_item<2> it) { 
 
            // Zero shared memory and sync.
            for( auto i = it.get_local_id(0); i < tile_vol; i += it.get_local_range(0) ) {
                charge_local[i] = 0;
            }

            float *charge = &charge_local[ charge_offset ];

            it.barrier();

            const int2 tile_idx = make_int2( it.get_group(0), it.get_group(1) );
            const int tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
            const int offset   = part.offset[ tile_id ];
            const int np       = part.np[ tile_id ];
            int2   const * __restrict__ const ix = &part.ix[ offset ];
            float2 const * __restrict__ const x  = &part.x[ offset ];

            for( int i = it.get_local_id(0); i < np; i += it.get_local_range(0) ) {
                const int idx = ix[i].y * ystride + ix[i].x;
                const float s0x = 0.5f - x[i].x;
                const float s1x = 0.5f + x[i].x;
                const float s0y = 0.5f - x[i].y;
                const float s1y = 0.5f + x[i].y;

                device::local::atomicAdd( & charge[ idx               ], s0y * s0x * q );
                device::local::atomicAdd( & charge[ idx + 1           ], s0y * s1x * q );
                device::local::atomicAdd( & charge[ idx     + ystride ], s1y * s0x * q );
                device::local::atomicAdd( & charge[ idx + 1 + ystride ], s1y * s1x * q );
            }

            it.barrier();

            // Copy data to global memory
            const int tile_off = tile_id * tile_vol;
            for( auto i = it.get_local_id(0); i < tile_vol; i += it.get_local_range(0) ) {
                charge_global[tile_off + i] += charge_local[i];
            } 
        });
    });
    queue->wait();
}


/**
 * @brief Save particle data to file
 * 
 */
void Species::save() const {

    const char * quants[] = {
        "x","y",
        "ux","uy","uz"
    };

    const char * qlabels[] = {
        "x","y",
        "u_x","u_y","u_z"
    };

    const char * qunits[] = {
        "c/\\omega_n", "c/\\omega_n",
        "c","c","c"
    };

    zdf::iteration iter_info = {
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_n"
    };

    // Omit number of particles, this will be filled in later
    zdf::part_info info = {
        .name = (char *) name.c_str(),
        .label = (char *) name.c_str(),
        .nquants = 5,
        .quants = (char **) quants,
        .qlabels = (char **) qlabels,
        .qunits = (char **) qunits,
    };

    particles -> save( info, iter_info, "PARTICLES" );
}

/**
 * @brief Saves charge density to file
 * 
 * The routine will create a new charge grid, deposit the charge and save the grid
 * 
 */
void Species::save_charge() const {

    // For linear interpolation we only require 1 guard cell at the upper boundary
    bnd<unsigned int> gc;
    gc.x = {0,1};
    gc.y = {0,1};

    // Deposit charge on device
    grid<float> charge( particles -> ntiles, particles -> nx, gc, *queue );

    charge.zero();

    deposit_charge( charge );

    charge.add_from_gc();

    // Prepare file info
    zdf::grid_axis axis[2];
    axis[0] = (zdf::grid_axis) {
        .name = (char *) "x",
        .min = 0. + moving_window.motion(),
        .max = box.x + moving_window.motion(),
        .label = (char *) "x",
        .units = (char *) "c/\\omega_n"
    };

    axis[1] = (zdf::grid_axis) {
        .name = (char *) "y",
        .min = 0.,
        .max = box.y,
        .label = (char *) "y",
        .units = (char *) "c/\\omega_n"
    };

    std::string grid_name = name + "-charge";
    std::string grid_label = name + " \\rho";

    zdf::grid_info info = {
        .name = (char *) grid_name.c_str(),
        .label = (char *) grid_label.c_str(),
        .units = (char *) "n_e",
        .axis  = axis
    };

    zdf::iteration iter_info = {
        .name = (char *) "ITERATION",
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_n"
    };

    std::string path = "CHARGE/";
    path += name;
    
    charge.save( info, iter_info, path.c_str() );
}

/**
 * @brief kernel for depositing 1d phasespace
 * 
 * @tparam q        Phasespace quantity
 * @param d_data    Output data
 * @param range     Phasespace value range
 * @param size      Phasespace grid size
 * @param tile_nx   Size of tile grid
 * @param norm      Normalization factor
 * @param d_tiles   Particle tile information
 * @param d_ix      Particle data (cell)
 * @param d_x       Particle data (pos)
 * @param d_u       Particle data (generalized momenta)
 */
template < phasespace::quant quant >
void dep_pha1_kernel(
    sycl::nd_item<2> it, 
    float * const __restrict__ d_data, float2 const range, int const size,
    float const norm, 
    ParticleData const part )
{
    uint2 const tile_nx = part.nx;

    const int2 tile_idx = make_int2( it.get_group(0), it.get_group(1) );
    const int tile_id  = tile_idx.y * part.ntiles.x + tile_idx.x;

    const int part_offset = part.offset[ tile_id ];
    const int np          = part.np[ tile_id ];
    int2   * __restrict__ ix = &part.ix[ part_offset ];
    float2 * __restrict__ x  = &part.x[ part_offset ];
    float3 * __restrict__ u  = &part.u[ part_offset ];

    float const pha_rdx = size / (range.y - range.x);

    for( int i = it.get_local_id(0); i < np; i += it.get_local_range(0) ) {
        float d;
        switch( quant ) {
        case( phasespace:: x ): d = ( tile_idx.x * tile_nx.x + ix[i].x) + (x[i].x + 0.5f); break;
        case( phasespace:: y ): d = ( tile_idx.y * tile_nx.y + ix[i].y) + (x[i].y + 0.5f); break;
        case( phasespace:: ux ): d = u[i].x; break;
        case( phasespace:: uy ): d = u[i].y; break;
        case( phasespace:: uz ): d = u[i].z; break;
        }

        float n =  (d - range.x ) * pha_rdx - 0.5f;
        int   k = int( n + 1 ) - 1;
        float w = n - k;

        if ((k   >= 0) && (k   < size-1)) device::global::atomicAdd( &d_data[k  ], (1-w) * norm );
        if ((k+1 >= 0) && (k+1 < size-1)) device::global::atomicAdd( &d_data[k+1],    w  * norm );
    }
}


/**
 * @brief Deposit 1D phasespace
 * 
 * @note Output data will be zeroed before deposition
 * 
 * @param d_data    Output (device) data
 * @param quant     Phasespace quantity
 * @param range     Phasespace value range
 * @param size      Phasespace grid size
 */
void Species::dep_phasespace( float * const d_data, phasespace::quant quant, 
    float2 range, unsigned const size ) const
{
    // Zero device memory
    device::zero( d_data, size, *queue );
    
    // In OSIRIS we don't take the absolute value of q
    float norm = sycl::fabs(q) * ( dx.x * dx.y ) *
                 size / (range.y - range.x) ;

    queue->submit([&](sycl::handler &h) {

        const ParticleData part = *particles;
        const auto dx = this -> dx;

        auto ntiles = part.ntiles;

        // 8×1 work items per group
        sycl::range<2> local{ 8, 1 };

        // ntiles.x × ntiles.y groups
        sycl::range<2> global{ ntiles.x, ntiles.y };

        switch(quant) {
        case( phasespace::x ):
            h.parallel_for( 
                sycl::nd_range{ global * local, local },
                [=](sycl::nd_item<2> it) { 
                auto lrange = range;
                lrange.y /= dx.x;
                lrange.x /= dx.x;
                dep_pha1_kernel<phasespace::x> ( it, d_data, lrange, size, norm, part );
            });
            break;
        case( phasespace:: y ):
            h.parallel_for( 
                sycl::nd_range{ global * local, local },
                [=](sycl::nd_item<2> it) { 
                auto lrange = range;
                lrange.y /= dx.y;
                lrange.x /= dx.y;
                dep_pha1_kernel<phasespace::y> ( it, d_data, lrange, size, norm, part );
            });
            break;
        case( phasespace:: ux ):
            h.parallel_for( 
                sycl::nd_range{ global * local, local },
                [=](sycl::nd_item<2> it) { 
                dep_pha1_kernel<phasespace::ux> ( it, d_data, range, size, norm, part );
            });
            break;
        case( phasespace:: uy ):
            h.parallel_for( 
                sycl::nd_range{ global * local, local },
                [=](sycl::nd_item<2> it) { 
                dep_pha1_kernel<phasespace::uy> ( it, d_data, range, size, norm, part );
            });
            break;
        case( phasespace:: uz ):
            h.parallel_for( 
                sycl::nd_range{ global * local, local },
                [=](sycl::nd_item<2> it) { 
                dep_pha1_kernel<phasespace::uz> ( it, d_data, range, size, norm, part );
            });
            break;
        };
    });
    queue->wait();
}

/**
 * @brief Save 1D phasespace
 * 
 * @param q         Phasespace quantity
 * @param range     Phasespace range
 * @param size      Phasespace grid size
 */
void Species::save_phasespace( phasespace::quant quant, float2 const range, 
    int const size ) const
{
    std::string qname, qlabel, qunits;

    phasespace::qinfo( quant, qname, qlabel, qunits );
    
    // Prepare file info
    zdf::grid_axis axis = {
        .name = (char *) qname.c_str(),
        .min = range.x,
        .max = range.y,
        .label = (char *) qlabel.c_str(),
        .units = (char *) qunits.c_str()
    };

    if ( quant == phasespace::x ) {
        axis.min += moving_window.motion();
        axis.max += moving_window.motion();
    }

    std::string pha_name  = name + "-" + qname;
    std::string pha_label = name + "\\,(" + qlabel+")";

    zdf::grid_info info = {
        .name = (char *) pha_name.c_str(),
        .ndims = 1,
        .label = (char *) pha_label.c_str(),
        .units = (char *) "n_e",
        .axis  = &axis
    };

    info.count[0] = size;

    zdf::iteration iter_info = {
        .name = (char *) "ITERATION",
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_n"
    };

    // Deposit 1D phasespace
    float * d_data = device::malloc<float>( size, *queue );
    float * h_data = host::malloc<float>( size, *queue );
    
    dep_phasespace( d_data, quant, range, size );
    device::memcpy_tohost(  h_data, d_data, size, *queue );

    // Save file
    zdf::save_grid( h_data, info, iter_info, "PHASESPACE/" + name );

    host::free( h_data, *queue );
    device::free( d_data, *queue );
}

/**
 * @brief CUDA kernel for depositing 2D phasespace
 * 
 * @tparam q0       Quantity 0
 * @tparam q1       Quantity 1
 * @param d_data    Ouput data
 * @param range0    Range of values of quantity 0
 * @param size0     Phasespace grid size for quantity 0
 * @param range1    Range of values of quantity 1
 * @param size1     Range of values of quantity 1
 * @param tile_nx   Size of tile grid
 * @param norm      Normalization factor
 * @param d_tiles   Particle tile information
 * @param d_ix      Particle data (cell)
 * @param d_x       Particle data (pos)
 * @param d_u       Particle data (generalized momenta)
 */
template < phasespace::quant quant0, phasespace::quant quant1 >
void dep_pha2_kernel(
   sycl::nd_item<2> it, 
    float * const __restrict__ d_data, 
    float2 const range0, int const size0,
    float2 const range1, int const size1,
    float const norm, 
    ParticleData const part )
{
    static_assert( quant1 > quant0, "quant1 must be > quant0" );
    
    const auto tile_nx  = part.nx;

    const int2 tile_idx = make_int2( it.get_group(0), it.get_group(1) );
    const int tile_id  = tile_idx.y * part.ntiles.x + tile_idx.x;

    const int offset = part.offset[ tile_id ];
    const int np     = part.np[ tile_id ];
    int2   * __restrict__ ix  = &part.ix[ offset ];
    float2 * __restrict__ x   = &part.x [ offset ];
    float3 * __restrict__ u   = &part.u [ offset ];

    float const pha_rdx0 = size0 / (range0.y - range0.x);
    float const pha_rdx1 = size1 / (range1.y - range1.x);

    for( int i = it.get_local_id(0); i < np; i += it.get_local_range(0) ) {
        float d0;
        switch( quant0 ) {
        case( phasespace:: x ):  d0 = ( tile_idx.x * tile_nx.x + ix[i].x) + (x[i].x + 0.5f); break;
        case( phasespace:: y ):  d0 = ( tile_idx.y * tile_nx.y + ix[i].y) + (x[i].y + 0.5f); break;
        case( phasespace:: ux ): d0 = u[i].x; break;
        case( phasespace:: uy ): d0 = u[i].y; break;
        case( phasespace:: uz ): d0 = u[i].z; break;
        }

        float n0 =  (d0 - range0.x ) * pha_rdx0 - 0.5f;
        int   k0 = int( n0 + 1 ) - 1;
        float w0 = n0 - k0;

        float d1;
        switch( quant1 ) {
        //case( phasespace:: x ):  d1 = ( tile_idx.x * tile_nx.x + ix[i].x) + (x[i].x + 0.5f); break;
        case( phasespace:: y ):  d1 = ( tile_idx.y * tile_nx.y + ix[i].y) + (x[i].y + 0.5f); break;
        case( phasespace:: ux ): d1 = u[i].x; break;
        case( phasespace:: uy ): d1 = u[i].y; break;
        case( phasespace:: uz ): d1 = u[i].z; break;
        }

        float n1 =  (d1 - range1.x ) * pha_rdx1 - 0.5f;
        int   k1 = int( n1 + 1 ) - 1;
        float w1 = n1 - k1;

        if ((k0   >= 0) && (k0   < size0-1) && (k1   >= 0) && (k1   < size1-1))
            device::global::atomicAdd( &d_data[(k1  )*size0 + k0  ] , (1-w0) * (1-w1) * norm );
        if ((k0+1 >= 0) && (k0+1 < size0-1) && (k1   >= 0) && (k1   < size1-1))
            device::global::atomicAdd( &d_data[(k1  )*size0 + k0+1] ,    w0  * (1-w1) * norm );
        if ((k0   >= 0) && (k0   < size0-1) && (k1+1 >= 0) && (k1+1 < size1-1))
            device::global::atomicAdd( &d_data[(k1+1)*size0 + k0  ] , (1-w0) *    w1  * norm );
        if ((k0+1 >= 0) && (k0+1 < size0-1) && (k1+1 >= 0) && (k1+1 < size1-1))
            device::global::atomicAdd( &d_data[(k1+1)*size0 + k0+1] ,    w0  *    w1  * norm );
    }
}


/**
 * @brief Deposits a 2D phasespace in a device buffer
 * 
 * @note Output data will be zeroed before deposition
 *
 * @param d_data    Pointer to device buffer
 * @param quant0    Quantity 0
 * @param range0    Range of values of quantity 0
 * @param size0     Phasespace grid size for quantity 0
 * @param quant0    Quantity 1
 * @param range1    Range of values of quantity 1
 * @param size1     Phasespace grid size for quantity 1
 */
void Species::dep_phasespace( 
    float * const d_data,
    phasespace::quant quant0, float2 range0, unsigned const size0,
    phasespace::quant quant1, float2 range1, unsigned const size1 ) const
{
    // Zero device memory
    device::zero( d_data, size0 * size1, *queue );

    // In OSIRIS we don't take the absolute value of q
    float norm = sycl::fabs(q) * ( dx.x * dx.y ) *
                          ( size0 / (range0.y - range0.x) ) *
                          ( size1 / (range1.y - range1.x) );

    const ParticleData part = *particles;
    auto ntiles = part.ntiles;

    queue->submit([&](sycl::handler &h) {

        // 8×1 work items per group
        sycl::range<2> local{ 8, 1 };

        // ntiles.x × ntiles.y groups
        sycl::range<2> global{ ntiles.x, ntiles.y };

        auto lrange0 = range0;
        auto lrange1 = range1;

        switch(quant0) {
        case( phasespace::x ):
            lrange0.y /= dx.x;
            lrange0.x /= dx.x;
            switch(quant1) {
            case( phasespace::y ):
                lrange1.y /= dx.y;
                lrange1.x /= dx.y;
                h.parallel_for( 
                    sycl::nd_range{ global * local, local },
                    [=](sycl::nd_item<2> it) {
                    dep_pha2_kernel<phasespace::x,phasespace::y> (
                        it, d_data, lrange0, size0, lrange1, size1, norm, part
                    );
                });
                break;
            case( phasespace::ux ):
                h.parallel_for( 
                    sycl::nd_range{ global * local, local },
                    [=](sycl::nd_item<2> it) {
                    dep_pha2_kernel<phasespace::x,phasespace::ux> (
                        it, d_data, lrange0, size0, lrange1, size1, norm, part
                    );
                });
                break;
            case( phasespace::uy ):
                h.parallel_for( 
                    sycl::nd_range{ global * local, local },
                    [=](sycl::nd_item<2> it) {
                    dep_pha2_kernel<phasespace::x,phasespace::uy> (
                        it, d_data, lrange0, size0, lrange1, size1, norm, part
                    );
                });
                break;
            case( phasespace::uz ):
                h.parallel_for( 
                    sycl::nd_range{ global * local, local },
                    [=](sycl::nd_item<2> it) {
                    dep_pha2_kernel<phasespace::x,phasespace::uz> (
                        it, d_data, lrange0, size0, lrange1, size1, norm, part
                    );
                });
                break;
            default:
                break;
            }
            break;
        case( phasespace:: y ):
            lrange0.y /= dx.y;
            lrange0.x /= dx.y;
            switch(quant1) {
                h.parallel_for( 
                    sycl::nd_range{ global * local, local },
                    [=](sycl::nd_item<2> it) {
                    dep_pha2_kernel<phasespace::y,phasespace::ux> (
                        it, d_data, lrange0, size0, lrange1, size1, norm, part
                    );
                });
                break;
            case( phasespace::uy ):
                h.parallel_for( 
                    sycl::nd_range{ global * local, local },
                    [=](sycl::nd_item<2> it) {
                    dep_pha2_kernel<phasespace::y,phasespace::uy> (
                        it, d_data, lrange0, size0, lrange1, size1, norm, part
                    );
                });
                break;
            case( phasespace::uz ):
                h.parallel_for( 
                    sycl::nd_range{ global * local, local },
                    [=](sycl::nd_item<2> it) {
                    dep_pha2_kernel<phasespace::y,phasespace::uz> (
                        it, d_data, lrange0, size0, lrange1, size1, norm, part
                    );
                });
                break;
            default:
                break;
            }
            break;
        case( phasespace:: ux ):
            switch(quant1) {
            case( phasespace::uy ):
                h.parallel_for( 
                    sycl::nd_range{ global * local, local },
                    [=](sycl::nd_item<2> it) {
                    dep_pha2_kernel<phasespace::ux,phasespace::uy> (
                        it, d_data, lrange0, size0, lrange1, size1, norm, part
                    );
                });
                break;
            case( phasespace::uz ):
                h.parallel_for( 
                    sycl::nd_range{ global * local, local },
                    [=](sycl::nd_item<2> it) {
                    dep_pha2_kernel<phasespace::ux,phasespace::uz> (
                        it, d_data, lrange0, size0, lrange1, size1, norm, part
                    );
                });
                break;
            default:
                break;
            }
            break;
        case( phasespace:: uy ):
            h.parallel_for( 
                sycl::nd_range{ global * local, local },
                [=](sycl::nd_item<2> it) {
                dep_pha2_kernel<phasespace::uy,phasespace::uz> (
                        it, d_data, lrange0, size0, lrange1, size1, norm, part
                    );
                });
                break;
        default:
            break;
        };

    });
    queue->wait();
}


/**
 * @brief Save 2D phasespace
 * 
 * @param quant0    Quantity 0
 * @param range0    Range of values of quantity 0
 * @param size0     Phasespace grid size for quantity 0
 * @param quant1    Quantity 1
 * @param range1    Range of values of quantity 1
 * @param size1     Phasespace grid size for quantity 0
 */
void Species::save_phasespace( 
    phasespace::quant quant0, float2 const range0, int const size0,
    phasespace::quant quant1, float2 const range1, int const size1 )
    const
{

    if ( quant0 >= quant1 ) {
        std::cerr << "(*error*) for 2D phasespaces, the 2nd quantity must be indexed higher than the first one\n";
        return;
    }

    std::string qname0, qlabel0, qunits0;
    std::string qname1, qlabel1, qunits1;

    phasespace::qinfo( quant0, qname0, qlabel0, qunits0 );
    phasespace::qinfo( quant1, qname1, qlabel1, qunits1 );
    
    // Prepare file info
    zdf::grid_axis axis[2] = {
        zdf::grid_axis {
            .name = (char *) qname0.c_str(),
            .min = range0.x,
            .max = range0.y,
            .label = (char *) qlabel0.c_str(),
            .units = (char *) qunits0.c_str()
        },
        zdf::grid_axis {
            .name = (char *) qname1.c_str(),
            .min = range1.x,
            .max = range1.y,
            .label = (char *) qlabel1.c_str(),
            .units = (char *) qunits1.c_str()
        }
    };

    if ( quant0 == phasespace::x ) {
        axis[0].min += moving_window.motion();
        axis[0].max += moving_window.motion();
    }


    std::string pha_name  = name + "-" + qname0 + qname1;
    std::string pha_label = name + " \\,(" + qlabel0 + "\\rm{-}" + qlabel1+")";

    zdf::grid_info info = {
        .name = (char *) pha_name.c_str(),
        .ndims = 2,
        .count = { static_cast<unsigned>(size0), static_cast<unsigned>(size1), 0 },
        .label = (char *) pha_label.c_str(),
        .units = (char *) "n_e",
        .axis  = axis
    };

    zdf::iteration iter_info = {
        .name = (char *) "ITERATION",
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_n"
    };

    std::cout << "phasespace size: " << size0 << ',' << size1 << '\n';

    float * d_data = device::malloc<float>( size0 * size1, *queue );
    float * h_data = host::malloc<float>( size0 * size1, *queue );

    dep_phasespace( d_data, quant0, range0, size0, quant1, range1, size1 );
    device::memcpy_tohost(  h_data, d_data, size0 * size1, *queue );

    zdf::save_grid( h_data, info, iter_info, "PHASESPACE/" + name );

    host::free( h_data, *queue );
    device::free( d_data, *queue );
}