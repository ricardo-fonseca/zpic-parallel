#include "cathode.h"
#include "random.h"

/**
 * ## Cathode algorithm
 * 
 * To minimize differences from a free streaming species with the same (fluid)
 * velocity, this implementation of the cathode stores `ppc.x` longitudinal
 * positions:
 * 1. These positions are initialized as if injecting a uniform density profile
 * 2. At each time-step they are advanced as if they are a free-streaming 
 *    particle species, i.e., positions are moved by v * dt / dx
 * 4. The complete set of positions is copied onto device memory; all positions
 *    >= 0.5 (for a lower edge cathode) correspond to particles that must be
 *    injected.
 * 5. Reference positions are then trimmed 
 * 
 * While this represents an overhead compared to the previous algorithm, it
 * ensures a much steadier flow of particles. Currently, the only limitation of
 * this implementation is that the algorithm requires a shared memory array of
 * `ppc.x` floats.
 */


/**
 * @brief Construct a new Cathode:: Cathode object
 * 
 * @param name      Cathode name
 * @param m_q       Mass over charge ratio
 * @param ppc       Number of particles per cell
 * @param ufl       Fluid velocity
 */
Cathode::Cathode( std::string const name, float const m_q, uint2 const ppc, edge::pos wall, float ufl ):
    Species( name, m_q, ppc ), ufl( sycl::fabs(ufl) ), wall( wall )
{ 
    if ( ufl == 0 ) {
        std::cerr << "(*error*) Cathodes cannot have ufl = 0, aborting...\n";
        exit(1);
    }

    // Default values
    start = 0;
    end   = __FLT_MAX__;
    uth   = make_float3( 0,0,0 );
    n0    = 1.0;
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
void Cathode::initialize( float2 const box_, uint2 const ntiles, uint2 const nx,
    float const dt_, int const id_, sycl::queue & queue_ ) {

    // Cathode velocity (always > 0)
    vel = (ufl / sqrtf( ops::fma( ufl , ufl , 1.0f ) ));

    // Initialize position of cathode particles in the cell outside the box
    d_inj_pos = device::malloc<float>( ppc.x, queue_ );

    double dpcx = 1.0  / ppc.x;
    for( unsigned i = 0; i < ppc.x; i ++ ) {
        d_inj_pos[i] = dpcx * ( i + 0.5 ) - 0.5;
    }

    // Set initial density, if any
    // This allows injecting an initial density / temperature distribution if required
    if ( start < 0 ) {
        float x0 = 0, x1 = 0, u = 0;

        switch (wall)
        {
        case edge::lower:
            x0 = 0;
            x1 = -vel * start;
            u = ufl;
            break;
        
        case edge::upper:
            x0 = box.x + vel * start;
            x1 = box.x;
            u = -ufl;
            break;
        }
        Species::set_density( Density::Slab( coord::x, n0, x0, x1 ) );
        Species::set_udist( UDistribution::Thermal( uth, make_float3( u, 0, 0 ) ) );
    } else {
        Species::set_density( Density::None( n0 ) );
        // No need to set udist because no particles will be injected
    }

    // Complete species initialization (will inject initial particles, if any)
    Species::initialize( box_, ntiles, nx, dt_, id_, queue_ );
}

/**
 * @brief Destroy the Cathode
 * 
 */
Cathode::~Cathode() {
    device::free( d_inj_pos, * queue );
}

/**
 * @brief Inject particles inside the simulation box
 * 
 * This will only happen if iter == 0 and start < 0
 * This also sets the velocity of the injected particles
 */
void Cathode::inject() {
    
    if ( iter == 0 && start < 0 ) {

        uint2 g_nx = particles -> gnx;
        bnd<unsigned int> range;
        range.x = { .lower = 0, .upper = g_nx.x - 1 };
        range.y = { .lower = 0, .upper = g_nx.y - 1 };

        Species::inject( range );
    }
}

/**
 * @brief Inject particles inside the specified range
 *
 * This will only happen if iter == 0 and start < 0
 * 
 * @param range     Cell range in which to inject
 */
void Cathode::inject( bnd<unsigned int> range ) {

    if ( iter == 0 && start < 0 ) {
        Species::inject( range );
    }
}

/**
 * @brief Get number of particles to inject in the specified range
 * 
 * This will only happen if iter == 0 and start < 0, otherwise the
 * output array will just be set to 0
 * 
 * @param range 
 * @param np 
 */
void Cathode::np_inject( bnd<unsigned int> range, int * np ) {
    
    if ( iter == 0 && start < 0 ) {
        Species::np_inject( range, np );
    } else {
        device::zero( np, particles -> ntiles.x * particles -> ntiles.y, *queue );
    }
}


/**
 * @brief Kernel for injecting cathode particles from lower wall
 * 
 * @param tile_idx      Tile index
 * @param d_inj_pos     Pointer to injection position
 * @param ufl           Cathode flow generalized velocity
 * @param uth           Temperature for injected particles
 * @param seed          RNG seed
 * @param ppc           Number of particles per cell
 * @param tiles         Particle tile information
 * @param data          Particle data
 */
void inject_cathode_lower( 
    sycl::nd_item<1> it, 
    float * __restrict__ const d_inj_pos, float const ufl,
    float3 uth, uint2 seed,  uint2 const ppc,
    ParticleData const part,
    int * __restrict__ inj_pos, int * __restrict__ inj_np_local )
{
    const auto ntiles  = part.ntiles;
    const auto nx      = part.nx;

    const uint2 tile_idx = make_uint2( 0, it.get_local_id(0) );
    const int tile_id = tile_idx.y * ntiles.x + tile_idx.x;

    // Initialize random state variables
    uint2 state;
    double norm;
    zrandom::rand_init( it.get_global_linear_id(), seed, state, norm );

    const int offset =  part.offset[ tile_id ];
    int2   * __restrict__ const ix = &part.ix[ offset ];
    float2 * __restrict__ const x  = &part.x[ offset ];
    float3 * __restrict__ const u  = &part.u[ offset ];

    int np = part.np[ tile_id ];

    // Advance injection positions and count number of particles to inject

    *inj_np_local = 0;
    it.barrier();

    int inj_np = 0;
    for( auto idx = it.get_local_id(0); idx < ppc.x; idx += it.get_local_range(0) ) {
        inj_pos[idx] = d_inj_pos[idx];
        if ( inj_pos[idx] >= 0.5f ) inj_np++;
    }

    auto sg = it.get_sub_group();
    inj_np = device::subgroup::reduce_add( sg, inj_np );
    if ( sg.get_local_id() == 0 ) {
        device::local::atomicAdd( inj_np_local, inj_np );
    }

    it.barrier();

    inj_np = *inj_np_local;

    // Inject particles
    double dpcy = 1.0 / ppc.y;

    // 1 thread per cell
    for( auto idx = it.get_local_id(0); idx < nx.y; idx += it.get_local_range(0) ) {
        int2 cell = make_int2( 0, idx );

        int part_idx = np + idx * ( inj_np * ppc.y );

        for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
            for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                if ( inj_pos[i0] >= 0.5f ) {
                    float2 pos = make_float2(
                        inj_pos[i0] - 1.0f,
                        dpcy * ( i1 + 0.5 ) - 0.5
                    );
                    ix[ part_idx ] = cell;
                    x[ part_idx ] = pos;
                    u[ part_idx ] = make_float3( 
                        ufl + uth.x * zrandom::rand_norm( state, norm ),
                              uth.y * zrandom::rand_norm( state, norm ),
                              uth.z * zrandom::rand_norm( state, norm )
                    );
                    part_idx++;
                }
            }
        }
    }

    // Update global number of particles in tile
    if ( it.get_local_id(0) == 0 )
        part.np[ tile_id ] += nx.y * inj_np * ppc.y;
}


/**
 * @brief Kernel for injecting cathode particles from upper wall
 * 
 * @param tile_idx      Tile index
 * @param d_inj_pos     Pointer to injection position
 * @param ufl           Cathode flow generalized velocity
 * @param uth           Temperature for injected particles
 * @param seed          RNG seed
 * @param ppc           Number of particles per cell
 * @param tiles         Particle tile information
 * @param data          Particle data
 */
void inject_cathode_upper( 
    sycl::nd_item<1> it, 
    float * const d_inj_pos, float const ufl,
    float3 uth, uint2 seed,  uint2 const ppc,
    ParticleData const part,
    int * __restrict__ inj_pos, int * __restrict__ inj_np_local )
{
    const auto ntiles  = part.ntiles;
    const auto nx      = part.nx;

    const uint2 tile_idx = make_uint2( ntiles.x-1, it.get_local_id(0) );
    const int tile_id = tile_idx.y * ntiles.x + tile_idx.x;

    // Initialize random state variables
    uint2 state;
    double norm;
    zrandom::rand_init( it.get_global_linear_id(), seed, state, norm );

    const int offset =  part.offset[ tile_id ];
    int2   * __restrict__ const ix = &part.ix[ offset ];
    float2 * __restrict__ const x  = &part.x[ offset ];
    float3 * __restrict__ const u  = &part.u[ offset ];

    int np = part.np[ tile_id ];

    // Advance injection positions and count number of particles to inject

    *inj_np_local = 0;
    it.barrier();

    int inj_np = 0;
    for( auto idx = it.get_local_id(0); idx < ppc.x; idx += it.get_local_range(0) ) {
        inj_pos[idx] = d_inj_pos[idx];
        if ( inj_pos[idx] < -0.5f ) inj_np ++;
    }

    auto sg = it.get_sub_group();
    inj_np = device::subgroup::reduce_add( sg, inj_np );
    if ( sg.get_local_id() == 0 ) {
        device::local::atomicAdd( inj_np_local, inj_np );
    }

    it.barrier();

    inj_np = *inj_np_local;

    // Inject particles
    double dpcy = 1.0 / ppc.y;

    // 1 thread per cell
    for( auto idx = it.get_local_id(0); idx < nx.y; idx += it.get_local_range(0) ) {
        int2 const cell = make_int2( nx.x-1, idx );

        int part_idx = np + idx * ( inj_np * ppc.y );

        for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
            for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                if ( inj_pos[i0] < -0.5f ) {
                    float2 const pos = make_float2(
                        inj_pos[i0] + 1.0f,
                        dpcy * ( i1 + 0.5 ) - 0.5
                    );
                    ix[ part_idx ] = cell;
                    x[ part_idx ] = pos;
                    u[ part_idx ] = make_float3( 
                        -ufl + uth.x * zrandom::rand_norm( state, norm ),
                               uth.y * zrandom::rand_norm( state, norm ),
                               uth.z * zrandom::rand_norm( state, norm )
                    );
                    part_idx++;
                }
            }
        }
    }

    // Update global number of particles in tile
    if ( it.get_local_id(0) == 0 )
        part.np[ tile_id ] += nx.y * inj_np * ppc.y;
}

/**
 * @brief Kernel for counting particles to inject from lower wall
 * 
 * @param tile_idx      Tile index
 * @param d_inj_pos     Injection positions
 * @param ppc           Number of particles per cell
 * @param tiles         Tile information
 * @param np            (out) Number of particles per tile to inject
 */
void np_inject_cathode_lower( 
    sycl::nd_item<1> it,
    float * __restrict__ const d_inj_pos,
    uint2 const ppc,
    ParticleData const part,
    int * __restrict__ inj_pos, int * __restrict__ inj_np_local,
    int * __restrict__ np )
{
    const auto ntiles  = part.ntiles;

    const uint2 tile_idx = make_uint2( 0, it.get_local_id(0) );
    const int tile_id = tile_idx.y * ntiles.x + tile_idx.x;

    *inj_np_local = 0;
    it.barrier();

    // Number of particles to inject per cell
    int inj_np = 0;
    for( auto idx = it.get_local_id(0); idx < ppc.x; idx += it.get_local_range(0) ) {
        inj_pos[idx] = d_inj_pos[idx];
        if ( inj_pos[idx] >= 0.5f ) inj_np++;
    }

    auto sg = it.get_sub_group();
    inj_np = device::subgroup::reduce_add( sg, inj_np );
    if ( sg.get_local_id() == 0 ) {
        device::local::atomicAdd( inj_np_local, inj_np );
    }

    it.barrier();

    // Get total number of particle to inject
    if ( it.get_local_id(0) == 0 )
        np[ tile_id ] = *inj_np_local * ppc.y * part.nx.y;
}

/**
 * @brief Kernel for counting particles to inject from upper wall
 * 
 * @param tile_idx      Tile index
 * @param d_inj_pos     Injection positions
 * @param ppc           Number of particles per cell
 * @param tiles         Tile information
 * @param np            (out) Number of particles per tile to inject
 */
void np_inject_cathode_upper( 
    sycl::nd_item<1> it,
    float * __restrict__ const d_inj_pos,
    uint2 const ppc,
    ParticleData const part,
    int * __restrict__ inj_pos, int * __restrict__ inj_np_local,
    int * __restrict__ np )
{
    const auto ntiles  = part.ntiles;

    const uint2 tile_idx = make_uint2( ntiles.x - 1, it.get_local_id(0) );
    const int tile_id = tile_idx.y * ntiles.x + tile_idx.x;

    *inj_np_local = 0;
    it.barrier();

    // Number of particles to inject per cell
    int inj_np = 0;
    for( auto idx = it.get_local_id(0); idx < ppc.x; idx += it.get_local_range(0) ) {
        inj_pos[idx] = d_inj_pos[idx];
        if ( inj_pos[idx] < -0.5f ) inj_np ++;
    }

    auto sg = it.get_sub_group();
    inj_np = device::subgroup::reduce_add( sg, inj_np );
    if ( sg.get_local_id() == 0 ) {
        device::local::atomicAdd( inj_np_local, inj_np );
    }

    it.barrier();

    // Get total number of particle to inject
    if ( it.get_local_id(0) == 0 )
        np[ tile_id ] = *inj_np_local * ppc.y * part.nx.y;
}

/**
 * @brief Count how many particles will be injected into each tile
 * 
 * The routine also updates d_inj_pos (position of injection particles)
 * 
 */
void Cathode::cathode_np_inject( int * np )
{

    // Injection will be zero on most tiles
    // We could also change the kernel so that they run on all tiles and do it there
    device::zero( np, particles -> ntiles.y * particles -> ntiles.x, *queue );

    const ParticleData part = *particles;
    const auto d_inj_pos = this -> d_inj_pos;
    const auto ppc = this -> ppc;

    switch (wall)
    {
    case edge::lower:
        queue->submit([&](sycl::handler &h) {
            auto inj_pos      = sycl::local_accessor< int, 1 > ( ppc.x, h );
            auto inj_np_local = sycl::local_accessor< int, 1 > ( 1, h );
            sycl::range<1> local{ 8 };
            sycl::range<1> global{ particles -> ntiles.y };
            h.parallel_for( 
                sycl::nd_range{ global * local, local },
                [=](sycl::nd_item<1> it) {

                np_inject_cathode_lower( it, d_inj_pos, ppc, part, &inj_pos[0], &inj_np_local[0], np );
            });
        });
        break;
    
    case edge::upper:
        queue->submit([&](sycl::handler &h) {
            auto inj_pos      = sycl::local_accessor< int, 1 > ( ppc.x, h );
            auto inj_np_local = sycl::local_accessor< int, 1 > ( 1, h );
            sycl::range<1> local{ 8 };
            sycl::range<1> global{ particles -> ntiles.y };
            h.parallel_for( 
                sycl::nd_range{ global * local, local },
                [=](sycl::nd_item<1> it) {

                np_inject_cathode_upper( it, d_inj_pos, ppc, part, &inj_pos[0], &inj_np_local[0], np );
            });
        });
        break;
    }
    queue->wait();

}

/**
 * @brief Inject new cathode particles
 * 
 */
void Cathode::cathode_inject( )
{
    uint2 rnd_seed = {12345 + (unsigned int) iter, 67890 + (unsigned int ) id };

    const ParticleData part = *particles;
    const auto d_inj_pos = this -> d_inj_pos;
    const auto ufl = this -> ufl;
    const auto uth = this -> uth;
    const auto ppc = this -> ppc;

    switch (wall)
    {
    case edge::lower:

        queue->submit([&](sycl::handler &h) {
            auto inj_pos      = sycl::local_accessor< int, 1 > ( ppc.x, h );
            auto inj_np_local = sycl::local_accessor< int, 1 > ( 1, h );
            sycl::range<1> local{ 8 };
            sycl::range<1> global{ particles -> ntiles.y };
            h.parallel_for( 
                sycl::nd_range{ global * local, local },
                [=](sycl::nd_item<1> it) {
                
                inject_cathode_lower ( it, d_inj_pos, ufl, uth, rnd_seed, ppc, part, &inj_pos[0], &inj_np_local[0] );
            }); 
        });
        break;
    
    case edge::upper:
        queue->submit([&](sycl::handler &h) {
            auto inj_pos      = sycl::local_accessor< int, 1 > ( ppc.x, h );
            auto inj_np_local = sycl::local_accessor< int, 1 > ( 1, h );
            sycl::range<1> local{ 8 };
            sycl::range<1> global{ particles -> ntiles.y };
            h.parallel_for( 
                sycl::nd_range{ global * local, local },
                [=](sycl::nd_item<1> it) {
                
                inject_cathode_upper ( it, d_inj_pos, ufl, uth, rnd_seed, ppc, part, &inj_pos[0], &inj_np_local[0] );
            }); 
        });
        break;
    }
    queue->wait();
}


/**
 * @brief Updates injection position of cathode particles
 * 
 */
void Cathode::update_inj_pos() {
    float motion = vel * dt / dx.x;

    const auto d_inj_pos = this -> d_inj_pos;

    // Update d_inj_pos
    switch (wall)
    {
    case edge::lower:
        queue->submit([&](sycl::handler &h) {
            h.parallel_for( sycl::range{ppc.x}, [=](sycl::id<1> idx) {
                float x = d_inj_pos[idx];
                if ( x >= 0.5f ) x -= 1.0f;
                d_inj_pos[idx] = x + motion;
            });
        });
        break;
    
    case edge::upper:
        queue->submit([&](sycl::handler &h) {
            h.parallel_for( sycl::range{ppc.x}, [=](sycl::id<1> idx) {
                float x = d_inj_pos[idx];
                if ( x < -0.5f ) x += 1.0f;
                d_inj_pos[idx] = x - motion;
            });
        });
        break;
    }

    queue->wait();
}

/**
 * @brief Advance cathode species.
 * 
 * Advances existing particles and injects new particles if needed.
 * 
 * @param emf 
 * @param current 
 */
void Cathode::advance( EMF const &emf, Current &current ) 
{

    // Advance momenta
    push( emf.E, emf.B );

    // Advance positions and deposit current
    move( current.J );

    // Process physical boundary conditions
    process_bc();

    double t = ( iter - 1 ) * dt;
    if (( t >= start ) && ( t < end ) ) { 

        // Update injection positions of cathode particles
        update_inj_pos();

        // Count how many particles are being injected
        cathode_np_inject( np_inj );

        // Sort particles according to tile, leaving room for new particles to be injected
        particles -> tile_sort( *tmp, *sort, np_inj );

        // Inject new cathode particles
        cathode_inject() ;

    } else {
        // Cathode is not active, just sort particles normally
        particles -> tile_sort( *tmp, *sort );
    } 

    // Increase internal iteration number
    iter++;
}

/**
 * @brief Free stream cathode species.
 * 
 * Free streams existing particles and injects new particles if needed.
 * 
 * @param current 
 */
void Cathode::advance( Current &current ) 
{
    // Advance positions and deposit current
    move( current.J );

    // Process physical boundary conditions
    process_bc();

    double t = ( iter - 1 ) * dt;
    if (( t >= start ) && ( t < end ) ) { 

        // Update injection positions of cathode particles
        update_inj_pos();

        // Count how many particles are being injected
        cathode_np_inject( np_inj );

        // Sort particles according to tile, leaving room for new particles to be injected
        particles -> tile_sort( *tmp, *sort, np_inj );

        // Inject new cathode particles
        cathode_inject() ;

    } else {
        // Just sort particles over tiles
        particles -> tile_sort( *tmp, *sort );
    } 

    // Increase internal iteration number
    iter++;
}

