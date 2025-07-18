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
    Species( name, m_q, ppc ), ufl( fabs(ufl) ), wall( wall )
{ 
    if ( ufl == 0 ) {
        std::cerr << "(*error*) Cathodes cannot have ufl = 0, aborting...\n";
        exit(1);
    }

    // Default values
    start = 0;
    end   = std::numeric_limits<float>::infinity();
    uth   = make_float3( 0, 0, 0 );
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
    double const dt_, int const id_ ) {

    // Cathode velocity (always > 0)
    vel = (ufl / std::sqrt( ops::fma( ufl , ufl , 1.0f ) ));

    // Initialize position of cathode particles in the cell outside the box
    d_inj_pos = memory::malloc<float>( ppc.x );

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
    Species::initialize( box_, ntiles, nx, dt_, id_ );
}

/**
 * @brief Destroy the Cathode
 * 
 */
Cathode::~Cathode() {
    memory::free( d_inj_pos );
}

/**
 * @brief Inject particles inside the simulation box
 * 
 * This will only happen if iter == 0 and start < 0
 * This also sets the velocity of the injected particles
 */
void Cathode::inject() {
    
    if ( iter == 0 && start < 0 ) {

        uint2 dims = particles -> dims;
        bnd<unsigned int> range;
        range.x = { .lower = 0, .upper = dims.x - 1 };
        range.y = { .lower = 0, .upper = dims.y - 1 };

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
        memory::zero( np, particles -> ntiles.x * particles -> ntiles.y );
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
    uint2 const tile_idx,
    float * const d_inj_pos, float const ufl,
    float3 uth, uint2 seed,  uint2 const ppc,
    ParticleData const part )
{
    const auto ntiles  = part.ntiles;
    const auto nx      = part.nx;
    float inj_pos[ ppc.x ];

    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    // Initialize random state variables
    uint2 state;
    double norm;
    zrandom::rand_init( tid, seed, state, norm );

    const int offset =  part.offset[ tid ];
    int2   * __restrict__ const ix = &part.ix[ offset ];
    float2 * __restrict__ const x  = &part.x[ offset ];
    float3 * __restrict__ const u  = &part.u[ offset ];

    int np = part.np[ tid ];

    // Advance injection positions and count number of particles to inject

    unsigned int _inj_np;
    _inj_np = 0;

    // sync

    unsigned int inj_np = 0;
    for( unsigned idx = 0; idx < ppc.x; idx ++ ) {
        inj_pos[idx] = d_inj_pos[idx];
        if ( inj_pos[idx] >= 0.5f ) inj_np++;
    }

    // Not needed with 1 thread / tile
    // inj_np = device::warp_reduce_add( inj_np );
    {   // Only one thread per block does this
        // atomicAdd( &_inj_np, inj_np);
        _inj_np += inj_np;
    }

    //sync

    inj_np = _inj_np;

    // Inject particles
    double dpcy = 1.0 / ppc.y;

    // 1 thread per cell
    for( unsigned idx = 0; idx < nx.y; idx ++ ) {
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
    {   // Only 1 thread per tile does this
        part.np[ tid ] += nx.y * inj_np * ppc.y;
    }
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
    uint2 const tile_idx,
    float * const d_inj_pos, float const ufl,
    float3 uth, uint2 seed,  uint2 const ppc,
    ParticleData const part )
{
    const auto ntiles  = part.ntiles;
    const auto nx      = part.nx;
    float inj_pos[ ppc.x ];

    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    // Initialize random state variables
    uint2 state;
    double norm;
    zrandom::rand_init( tid, seed, state, norm );

    const int offset =  part.offset[ tid ];
    int2   * __restrict__ const ix = &part.ix[ offset ];
    float2 * __restrict__ const x  = &part.x[ offset ];
    float3 * __restrict__ const u  = &part.u[ offset ];

    int np = part.np[ tid ];

    // Advance injection positions and count number of particles to inject

    unsigned int _inj_np;
    _inj_np = 0;

    // sync

    unsigned int inj_np = 0;
    for( unsigned idx = 0; idx < ppc.x; idx ++ ) {
        inj_pos[idx] = d_inj_pos[idx];
        if ( inj_pos[idx] < -0.5f ) inj_np ++;
    }

    // Not needed with 1 thread / tile
    // inj_np = device::warp_reduce_add( inj_np );
    {   // Only one thread per warp does this
        // atomicAdd( &_inj_np, inj_np);
        _inj_np += inj_np;
    }

    //sync

    inj_np = _inj_np;

    // Inject particles
    double dpcy = 1.0 / ppc.y;

    // 1 thread per cell
    for( unsigned idx = 0; idx < nx.y; idx ++ ) {
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
    {   // Only 1 thread per tile does this
        part.np[ tid ] += nx.y * inj_np * ppc.y;
    }
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
    uint2 const tile_idx,  float * const d_inj_pos, uint2 const ppc,
    ParticleData const part, int * __restrict__ np )
{
    const auto ntiles  = part.ntiles;
    float inj_pos[ ppc.x ];

    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    unsigned _inj_np;
    _inj_np = 0;

    // sync

    // Number of particles to inject per cell
    unsigned inj_np = 0;
    for( unsigned idx = 0; idx < ppc.x; idx ++ ) {
        inj_pos[idx] = d_inj_pos[idx];
        if ( inj_pos[idx] >= 0.5f ) inj_np++;
    }

    // Not needed with 1 thread / tile
    // inj_np = device::warp_reduce_add( inj_np );
    {   // Only one thread per warp does this
        // atomicAdd( &_inj_np, inj_np);
        _inj_np += inj_np;
    }

    //sync

    // Get total number of particle to inject
    np[ tid ] = _inj_np * ppc.y * part.nx.y;
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
    uint2 const tile_idx,  float * const d_inj_pos, uint2 const ppc,
    ParticleData const part, int * __restrict__ np )
{
    const auto ntiles  = part.ntiles;
    float inj_pos[ ppc.x ];

    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    unsigned int _inj_np;
    _inj_np = 0;

    // sync

    // Number of particles to inject per cell
    unsigned int inj_np = 0;
    for( unsigned idx = 0; idx < ppc.x; idx ++ ) {
        inj_pos[idx] = d_inj_pos[idx];
        if ( inj_pos[idx] < -0.5f ) inj_np ++;
    }

    // Not needed with 1 thread / tile
    // inj_np = device::warp_reduce_add( inj_np );
    {   // Only one thread per warp does this
        // atomicAdd( &_inj_np, inj_np);
        _inj_np += inj_np;
    }

    //sync

    // Get total number of particle to inject
    np[ tid ] = _inj_np * ppc.y * part.nx.y;
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
    memory::zero( np, particles -> ntiles.y * particles -> ntiles.x );

    switch (wall)
    {
    case edge::lower:
        for( unsigned ty = 0; ty < particles -> ntiles.y; ty++ ) {
            uint2 tile_idx = make_uint2( 0, ty );
            np_inject_cathode_lower( tile_idx, d_inj_pos, ppc, *particles, np );
        }
        break;
    
    case edge::upper:
        for( unsigned ty = 0; ty < particles -> ntiles.y; ty++ ) {
            uint2 tile_idx = make_uint2( particles -> ntiles.x - 1, ty );
            np_inject_cathode_upper( tile_idx, d_inj_pos, ppc, *particles, np );
        }
        break;
    }

}

/**
 * @brief Inject new cathode particles
 * 
 */
void Cathode::cathode_inject( )
{
    uint2 rnd_seed = {12345 + (unsigned int) iter, 67890 + (unsigned int ) id };

    switch (wall)
    {
    case edge::lower:
        for( unsigned ty = 0; ty < particles -> ntiles.y; ty++ ) {
            uint2 tile_idx = make_uint2( 0, ty );
            inject_cathode_lower (
                tile_idx,
                d_inj_pos, ufl, uth, rnd_seed, ppc, 
                *particles
            );
        }
        break;
    
    case edge::upper:
        for( unsigned ty = 0; ty < particles -> ntiles.y; ty++ ) {
            uint2 tile_idx = make_uint2( particles-> ntiles.x - 1, ty );
            inject_cathode_upper (
                tile_idx,
                d_inj_pos, ufl, uth, rnd_seed, ppc, 
                *particles
            );
        }
        break;
    }
}


/**
 * @brief Updates injection position of cathode particles
 * 
 */
void Cathode::update_inj_pos() {
    float motion = vel * dt / dx.x;

    // Update d_inj_pos
    switch (wall)
    {
    case edge::lower:
        for( unsigned i = 0; i < ppc.x; i++ ) {
            float x = d_inj_pos[i];
            if ( x >= 0.5f ) x -= 1.0f;
            d_inj_pos[i] = x + motion;
        }
        break;
    
    case edge::upper:
        for( unsigned i = 0; i < ppc.x; i++ ) {
            float x = d_inj_pos[i];
            if ( x < -0.5f ) x += 1.0f;
            d_inj_pos[i] = x - motion;
        }
        break;
    }
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

/**
 * @brief Free stream cathode species.
 * 
 * Free streams existing particles and injects new particles if needed.
 * No acceleration or current deposition is performed. Used for debug purposes.
 * 
 * @param current 
 */
void Cathode::advance( ) 
{
    // Advance positions without depositing current
    move( );

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
