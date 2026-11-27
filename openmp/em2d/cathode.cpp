#include "cathode.hpp"

#include <algorithm>
#include <cmath>

#include "util/math.hpp"
#include "util/random.hpp"
#include "util/hash.hpp"

namespace em2d {

/**
 * ## Cathode algorithm
 * 
 * To minimize differences from a free streaming species with the same (fluid)
 * velocity, the cathode is modeled as a uniform stream of particles located
 * behind the injection wall, `ppc.x` particles per cell, moving with velocity
 * `vel` into the box:
 * 
 * 1. Stream particle k (k = 0, 1, 2, ...) sits at distance (k + 0.5)/ppc.x
 *    cells behind the wall when injection is turned on (at t_on = max(start,0))
 * 2. At time t the stream has moved D(t) = vel * (t - t_on) / dx cells, so
 *    the number of particles that have crossed the wall is
 *    N(t) = floor( D(t) * ppc.x + 0.5 )
 * 3. Particle k is at distance D(t) - (k + 0.5)/ppc.x from the wall, inside
 *    the box.
 * 
 * At each time step the new particles are those with N(t_n) <= k < N(t_{n+1});
 * each of these is replicated `ppc.y` times along y in every cell next to the
 * wall. The injection is computed from the iteration number alone, so there
 * is no accumulation of round-off in the stream position over long runs.
 * 
 * Injection end times that fall inside a time step are handled exactly:
 * only particles that crossed the wall before `end` are injected.
 * 
 * ## Current of injected particles
 * 
 * New particles have spent part of the time step inside the box, travelling
 * from the wall to their injection position. If `inj_current` is set (default)
 * the current of this segment is deposited using the same charge conserving
 * scheme as species::move(), so the continuity equation also holds for
 * injected particles. Otherwise the particles appear with no current, which
 * leads to a spurious (static) E_x field building up near the wall.
 * 
 * Injected particles are not pushed in the step they are injected in.
 * 
 * @note This is the CPU version of cuda-mpi/em2d/cathode.cu. The particle
 *       generation uses the same counter based PRNG and counters, so both
 *       versions inject the same particles.
 */

/**
 * @brief Construct a new cathode object
 * 
 * @param name      Cathode name
 * @param m_q       Mass over charge ratio
 * @param ppc       Number of particles per cell
 * @param wall      Wall to inject particles from (edge::lower or edge::upper)
 * @param ufl       Fluid generalized velocity (sign is ignored)
 */
cathode::cathode( std::string const name, float const m_q, uint2 const ppc,
    edge::pos const wall, float const ufl ) :
    species( name, m_q, ppc ), vel( 0 ), inj_count( 0 ),
    ufl( std::fabs( ufl ) ), wall( wall )
{ 
    if ( this -> ufl == 0 ) {
        util::fatal( "cathodes cannot have ufl = 0" );
    }

    // Default values
    start = 0;
    end   = std::numeric_limits<double>::max();
    uth   = make_float3( 0, 0, 0 );
    n0    = 1.0;
    inj_current = true;
}

/**
 * @brief Initialize data structures and inject initial particles (if start < 0)
 * 
 * @param box_              Global simulation box size
 * @param grid_ntiles       Number of tiles
 * @param tile_dims         Individual tile grid size
 * @param dt_               Time step
 * @param id_               Species unique identifier
 * @param periodic          Periodic boundaries (x,y)
 */
void cathode::initialize( float2 const box_, uint2 const grid_ntiles, uint2 const tile_dims,
    double const dt_, int const id_, int2 const periodic ) {

    // species::initialize() would turn open x boundaries into periodic ones
    if ( periodic.x ) {
        util::fatal( "cathodes require a non-periodic x grid" );
    }

    // Injection wall must be open (set_bc() also checks this)
    validate_wall_bc( get_bc() );

    // Cathode velocity (always > 0)
    vel = ufl / std::sqrt( 1.0 + static_cast<double>( ufl ) * ufl );

    // If start < 0, load the particles injected between start and min(end,0)
    double const t_stop = std::min( end, 0.0 );

    if ( start < t_stop ) {
        // Distance from wall of particles injected at t_stop and start
        float const d_near = vel * ( - t_stop );
        float const d_far  = vel * ( - start );

        float x0, x1, u;
        if ( wall == edge::lower ) {
            x0 = d_near;
            x1 = d_far;
            u  = ufl;
        } else {
            x0 = box_.x - d_far;
            x1 = box_.x - d_near;
            u  = -ufl;
        }

        species::set_density( density::slab( coord::x, n0, x0, x1 ) );
        species::set_udist( udist::thermal( uth, make_float3( u, 0, 0 ) ) );
    } else {
        // No initial particles, but n0 sets the charge normalization
        species::set_density( density::none( n0 ) );
    }

    // Complete species initialization (will inject initial particles, if any)
    species::initialize( box_, grid_ntiles, tile_dims, dt_, id_, periodic );

    // dx is only known after species::initialize()
    if ( vel * dt / dx.x >= 1.0 ) {
        util::fatal( "cathode flow moves 1 cell or more per time step, reduce dt" );
    }

    // Reset injection counter
    inj_count = 0;
}

namespace {

/// @brief Random number tag for cathode injection
constexpr uint32_t cathode_rnd_tag = "cathode rnd tag"_hash;

/**
 * @brief Injects new cathode particles in one tile at the injection wall
 * 
 * @note Particles are added after the existing tile particles, the tile
 *       must have room for them (see tile_sort() `extra` parameter)
 * 
 * @note Contiguous particles are placed in different cells (same ordering as
 *       density::uniform), to minimize collisions when depositing current
 * 
 * @tparam wall         Injection wall
 * @param tile_idx_y    Tile index along y (the x index is set by the wall)
 * @param part          Particle data
 * @param ppc           Number of particles per cell
 * @param n_new         Number of new stream particles (per ppc.y and per cell)
 * @param d0            Distance from the wall (in cells) of the first new particle
 * @param ufl           Cathode flow generalized velocity (> 0)
 * @param uth           Temperature of injected particles
 * @param rnd_key       PRNG key
 * @param iter          Iteration number (used in the PRNG counter)
 */
template< edge::pos wall >
void cathode_inject(
    int const tile_idx_y,
    part::particles_view const part,
    uint2 const ppc, int const n_new, float const d0,
    float const ufl, float3 const uth, uint2 const rnd_key, uint32_t const iter )
{
    const uint2 ntiles     = part.grid_ntiles;
    const int   tile_idx_x = ( wall == edge::lower ) ? 0 : ntiles.x - 1;
    const int   tid        = tile_idx_y * ntiles.x + tile_idx_x;

    // Number of particles before injection
    const int np     = part.tile_np[ tid ];
    const int offset = part.tile_offset[ tid ];

    int2   * __restrict__ const ix = &part.ix[ offset ];
    float2 * __restrict__ const x  = &part.x [ offset ];
    float3 * __restrict__ const u  = &part.u [ offset ];

    // Global tile id, used in the PRNG counter
    const int global_tile_id = 
        tile_idx_y * part.grid_ntiles.x + tile_idx_x;

    const int   ny      = part.tile_dims.y;
    const int   total   = ny * ppc.y * n_new;
    const int   cell_x  = ( wall == edge::lower ) ? 0 : part.tile_dims.x - 1;
    const float ux      = ( wall == edge::lower ) ? ufl : -ufl;
    const float dpcx    = 1.0f / ppc.x;
    const float dpcy    = 1.0f / ppc.y;

    // Flat index n = ppc_idx * ny + cy, i.e. the cell is the fastest index
    for( int n = 0; n < total; n++ ) {
        const int ppc_idx = n / ny;
        const int cy      = n - ppc_idx * ny;
        const int i1      = ppc_idx / n_new;
        const int j       = ppc_idx - i1 * n_new;

        // Distance from wall, in [0, vel * dt / dx)
        const float d = d0 - j * dpcx;

        float px = ( wall == edge::lower ) ? d - 0.5f : 0.5f - d;
        px = std::fmin( std::fmax( px, -0.5f ), std::nextafter( 0.5f, 0.f ) );

        // Counter is unique per particle, tile and iteration
        const uint4  rnd_ctr = make_uint4( n, global_tile_id, iter, cathode_rnd_tag );
        const float4 norm    = zrandom::rand_norm4_f( rnd_ctr, rnd_key );

        ix[ np + n ] = make_int2( cell_x, cy );
        x [ np + n ] = make_float2( px, dpcy * ( i1 + 0.5f ) - 0.5f );
        u [ np + n ] = make_float3(
            ux + uth.x * norm.x,
                 uth.y * norm.y,
                 uth.z * norm.z
        );
    }

    part.tile_np[ tid ] = np + total;
}

/**
 * @brief Deposits the current of a straight segment inside a single cell
 * 
 * @note Same scheme as dep_current_seg() in species.cpp
 * 
 * @param ix        Particle cell
 * @param x0        Initial particle position
 * @param x1        Final particle position
 * @param qnx       Normalization values for in plane current deposition
 * @param qvz       Out of plane current, q vz / 3 (times the fraction of
 *                  the time step spent on this segment)
 * @param J         Current density grid (tile buffer, at cell [0,0])
 * @param stride    Current density grid y stride
 */
inline void cathode_dep_seg(
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

    float * __restrict__ const Js = (float *) (&J[ix.x + stride * ix.y]);
    int const stride3 = 3 * stride;

    // Jz weights are the time average of Sx(t) Sy(t) along the segment:
    //   (S0x S0y + S1x S1y)/3 + (S0x S1y + S1x S0y)/6
    // The factor 1/3 is included in qvz

    Js[       0 + 0 + 0 ] += wl1 * wp10;
    Js[       0 + 0 + 1 ] += wl2 * wp20;
    Js[       0 + 0 + 2 ] += qvz * ( S0x0 * S0y0 + S1x0 * S1y0 + (S0x0 * S1y0 + S1x0 * S0y0)/2.0f );

    Js[       0 + 3 + 1 ] += wl2 * wp21;
    Js[       0 + 3 + 2 ] += qvz * ( S0x1 * S0y0 + S1x1 * S1y0 + (S0x1 * S1y0 + S1x1 * S0y0)/2.0f );

    Js[ stride3 + 0 + 0 ] += wl1 * wp11;
    Js[ stride3 + 0 + 2 ] += qvz * ( S0x0 * S0y1 + S1x0 * S1y1 + (S0x0 * S1y1 + S1x0 * S0y1)/2.0f );
    Js[ stride3 + 3 + 2 ] += qvz * ( S0x1 * S0y1 + S1x1 * S1y1 + (S0x1 * S1y1 + S1x1 * S0y1)/2.0f );
}

/**
 * @brief Deposits the current of newly injected cathode particles in one
 *        tile at the injection wall
 * 
 * Each new particle is assumed to have entered the box through the wall
 * (at the same y position) during this time step, and moved along x to its
 * injection position.
 * 
 * @note Call after cathode_inject(); the new particles are the last
 *       `np_tile` particles in the tile
 * 
 * @tparam wall         Injection wall
 * @param tile_idx_y    Tile index along y (the x index is set by the wall)
 * @param part          Particle data
 * @param current       Electric current density
 * @param np_tile       Number of particles injected in each wall tile
 * @param q             Particle charge
 * @param qnx           Normalization values for in plane current deposition
 * @param rmotion       1 / (cells moved by the cathode flow in 1 time step)
 */
template< edge::pos wall >
void cathode_deposit(
    int const tile_idx_y,
    part::particles_view const part,
    grid::tiled_view<float3> current,
    int const np_tile, float const q, float2 const qnx, float const rmotion )
{
    const uint2 ntiles     = part.grid_ntiles;
    const int   tile_idx_x = ( wall == edge::lower ) ? 0 : ntiles.x - 1;
    const int   tid        = tile_idx_y * ntiles.x + tile_idx_x;

    const int np     = part.tile_np[ tid ];
    const int offset = part.tile_offset[ tid ];

    int2   const * __restrict__ const ix = &part.ix[ offset ];
    float2 const * __restrict__ const x  = &part.x [ offset ];
    float3 const * __restrict__ const u  = &part.u [ offset ];

    float3 * __restrict__ const J = current.tile_buffer( tid ) + current.inner_offset;
    const int ystride = current.tile_ystride();

    for( int i = np - np_tile; i < np; i++ ) {
        const float2 x1 = x[i];
        const float3 pu = u[i];

        // Entry point on the wall
        const float2 x0 = make_float2( ( wall == edge::lower ) ? -0.5f : 0.5f, x1.y );

        // Distance travelled inside the box (cells) and fraction of the time
        // step spent inside the box (at the cathode flow velocity)
        const float d    = ( wall == edge::lower ) ? x1.x + 0.5f : 0.5f - x1.x;
        const float frac = d * rmotion;

        const float rg  = 1.0f / std::sqrt( ops::fma( pu.z, pu.z, ops::fma( pu.y, pu.y, ops::fma( pu.x, pu.x, 1.0f ) ) ) );
        const float qvz = q * pu.z * rg * ( 1.0f / 3.0f ) * frac;

        cathode_dep_seg( ix[i], x0, x1, qnx, qvz, J, ystride );
    }
}

} // anonymous namespace

/**
 * @brief Checks that the field / current boundaries at the injection wall
 *        are compatible with an open (absorbing) species boundary
 * 
 * The injection wall may be either open (emf / current bc::none) or a
 * conducting cathode (emf::bc::pec / current::bc::pec_absorbing).
 * 
 * @note Calls util::fatal() otherwise.
 * 
 * @param fld   EM fields (may be nullptr)
 * @param J     Electric current density
 */
void cathode::validate_fld_bc( emf const * const fld, current const & J ) const {

    // Check current bc
    auto const current_bc  = ( wall == edge::lower ) ? J.get_bc().x.lower : J.get_bc().x.upper;

    if ( current_bc != current::bc::none && current_bc != current::bc::pec_absorbing ) {
        util::fatal( "cathode " + name + ": the injection wall (x " + edge::name( wall ) + ")"
                         " requires a current::bc::none or current::bc::pec_absorbing boundary" );
    }

    // Check emf bc - must match current bc
    if ( fld != nullptr ) {
        auto const emf_bc  = ( wall == edge::lower ) ? fld -> get_bc().x.lower : fld -> get_bc().x.upper;
        auto const emf_req = ( current_bc == current::bc::none ) ? emf::bc::none : emf::bc::pec;
        if ( emf_bc != emf_req ) {
            util::fatal( "cathode " + name + ": the injection wall (x " + edge::name( wall ) + ")"
                        ": emf boundary does not match the current boundary" );
        }
    }
}

/**
 * @brief Total number of stream particles that have crossed the wall
 *        up to time t
 * 
 * @param t             Simulation time
 * @return uint64_t     Number of particles (per ppc.y and per cell along y)
 */
uint64_t cathode::n_crossed( double const t ) const {
    double const t_on = std::max( start, 0.0 );
    double const D = vel * ( std::min( t, end ) - t_on ) / dx.x;
    return ( D > 0 ) ? static_cast<uint64_t>( std::floor( D * ppc.x + 0.5 ) ) : 0;
}

/**
 * @brief Sort particles over tiles, injecting new cathode particles
 *        if required
 * 
 * @note Calls particles -> tile_sort() exactly once
 * 
 * @param J     Current density to deposit the current of the injected
 *              particles to. Use nullptr to skip deposition.
 */
void cathode::sort_inject( grid::tiled_vec3<float> * const J ) {

    // Particles are being advanced to t = (iter+1) * dt
    double const t1 = ( iter + 1 ) * dt;

    uint64_t const n_total = n_crossed( t1 );
    int const n_new = n_total - inj_count;

    // Distance from the wall of the first new particle (computed in double
    // precision, D grows with time)
    float d0 = 0;
    if ( n_new > 0 ) {
        double const D = vel * ( t1 - std::max( start, 0.0 ) ) / dx.x;
        d0 = D - ( inj_count + 0.5 ) / ppc.x;
    }

    // Update counter
    inj_count = n_total;

    if ( n_new > 0 ) {

        auto const ntiles    = particles -> get_grid_ntiles();
        auto const tile_dims = particles -> get_tile_dims();

        // Number of particles to inject per tile
        unsigned int const wall_tile_x = ( wall == edge::lower ) ? 0 : ntiles.x - 1;
        int const np_tile = tile_dims.y * ppc.y * n_new;

        for( unsigned tid = 0; tid < ntiles.x * ntiles.y; tid++ ) {
            np_inj[ tid ] = ( tid % ntiles.x == wall_tile_x ) ? np_tile : 0;
        }

        // Sort particles according to tile, leaving room for new particles
        // (grows the particle buffers if required)
        particles -> tile_sort( *tmp, *sort, np_inj );

        // Inject new particles
        uint2 const rnd_key = make_uint2( 12345 + id, 67890 );
        auto const view = particles -> view();

        switch( wall ) {
        case edge::lower:
            #pragma omp parallel for
            for( unsigned ty = 0; ty < ntiles.y; ty++ )
                cathode_inject< edge::lower >( ty, view, ppc, n_new, d0, ufl, uth, rnd_key, iter );
            break;
        case edge::upper:
            #pragma omp parallel for
            for( unsigned ty = 0; ty < ntiles.y; ty++ )
                cathode_inject< edge::upper >( ty, view, ppc, n_new, d0, ufl, uth, rnd_key, iter );
            break;
        }

        // Deposit current of injected particles
        if ( J != nullptr && inj_current ) {
            float2 const qnx = make_float2(
                q * dx.x / dt,
                q * dx.y / dt
            );
            float const rmotion = dx.x / ( vel * dt );

            // Each tile deposits only to its own buffer
            switch( wall ) {
            case edge::lower:
                #pragma omp parallel for
                for( unsigned ty = 0; ty < ntiles.y; ty++ )
                    cathode_deposit< edge::lower >( ty, view, J -> view(), np_tile, q, qnx, rmotion );
                break;
            case edge::upper:
                #pragma omp parallel for
                for( unsigned ty = 0; ty < ntiles.y; ty++ )
                    cathode_deposit< edge::upper >( ty, view, J -> view(), np_tile, q, qnx, rmotion );
                break;
            }
        }

    } else {
        // Nothing to inject, just sort particles over tiles
        particles -> tile_sort( *tmp, *sort );
    }
}

/**
 * @brief Free stream particles 1 timestep and inject new particles
 * 
 */
void cathode::advance() {

    // Advance positions
    move();

    // Process physical boundary conditions
    process_bc();

    // Sort particles according to tile and inject new particles (no current)
    sort_inject( nullptr );

    // Increase internal iteration number
    iter++;
}

/**
 * @brief Advance positions 1 timestep (no momentum update), deposit current
 *        and inject new particles
 * 
 * @param current   Electric current density
 */
void cathode::advance( current & current ) {

    // Check current boundaries at the injection wall
    validate_fld_bc( nullptr, current );

    // Advance positions and deposit current
    move( * current.J );

    // Process physical boundary conditions
    process_bc();

    // Sort particles according to tile and inject new particles
    sort_inject( current.J );

    // Increase internal iteration number
    iter++;
}

/**
 * @brief Advance particles 1 iteration, deposit current and inject new
 *        particles
 * 
 * @param emf       EM fields
 * @param current   Electric current density
 */
void cathode::advance( emf const & emf, current & current ) {

    // Check field / current boundaries at the injection wall
    validate_fld_bc( &emf, current );

    // Advance momenta
    push( * emf.E, * emf.B );

    // Advance positions and deposit current
    move( * current.J );

    // Process physical boundary conditions
    process_bc();

    // Sort particles according to tile and inject new particles
    sort_inject( current.J );

    // Increase internal iteration number
    iter++;
}

} // namespace em2d
