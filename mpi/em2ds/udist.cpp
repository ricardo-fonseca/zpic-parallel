#include "udist.hpp"
#include "util/random.hpp"

#include "util/hash.hpp"

#include <vector>

namespace em2ds {

/**
 * @note The kernel:: functions process a single tile, mirroring the CUDA
 *       kernels (one thread block per tile) in cuda-mpi/em2ds/udist.cu. They
 *       have internal linkage (static) to avoid clashes with other kernel::
 *       functions in other translation units.
 */
namespace kernel {

constexpr uint32_t udist_rnd_tag = "udist rnd tag"_hash;

static void none( const uint2 tile_idx, part::particles_view const part ) {

    const int   tile_id  = tile_idx.y * part.local_ntiles.x + tile_idx.x;

    const auto  tile_off = part.tile_offset[ tile_id ];
    const auto  tile_np  = part.tile_np[ tile_id ];

    float3 * __restrict__ const u  = &part.u[ tile_off ];

    for( int i = 0; i < tile_np; i++ )
        u[i] = make_float3( 0, 0, 0 );
}

} // namespace kernel

/**
 * @brief Sets none(0 temperature, 0 fluid) u distribution
 * 
 * @param part  Particle data
 * @param seed  Random number seed (unused)
 */
void udist::none::set( part::particles & part, [[maybe_unused]] unsigned int seed ) const {

    const auto ntiles = part.get_local_ntiles();
    const auto view   = part.view();

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for( unsigned ty = 0; ty < ntiles.y; ty++ ) {
        for( unsigned tx = 0; tx < ntiles.x; tx++ ) {
            kernel::none( make_uint2( tx, ty ), view );
        }
    }
}

namespace kernel {

static void cold( const uint2 tile_idx, part::particles_view const part, const float3 ufl ) {

    const int   tile_id  = tile_idx.y * part.local_ntiles.x + tile_idx.x;
    
    const auto  tile_off = part.tile_offset[ tile_id ];
    const auto  tile_np  = part.tile_np[ tile_id ];

    float3 * __restrict__ const u  = &part.u[ tile_off ];

    for( int i = 0; i < tile_np; i++ )
        u[i] = ufl;
}

} // namespace kernel

/**
 * @brief Sets cold(0 temperatures) u distribution
 * 
 * @param part  Particle data
 * @param seed  Random number seed (unused)
 */
void udist::cold::set( part::particles & part, [[maybe_unused]] unsigned int seed ) const {

    const auto ntiles = part.get_local_ntiles();
    const auto view   = part.view();

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for( unsigned ty = 0; ty < ntiles.y; ty++ ) {
        for( unsigned tx = 0; tx < ntiles.x; tx++ ) {
            kernel::cold( make_uint2( tx, ty ), view, ufl );
        }
    }
}

namespace kernel {

static void thermal( const uint2 tile_idx, part::particles_view const part,
    const uint2 rnd_key, const float3 uth, const float3 ufl ) {

    const int   tile_id  = tile_idx.y * part.local_ntiles.x + tile_idx.x;
    
    const auto  tile_off = part.tile_offset[ tile_id ];
    const auto  tile_np  = part.tile_np[ tile_id ];

    float3 * __restrict__ const u  = &part.u[ tile_off ];

    // Initialize random state variables
    uint2 global_tile = make_uint2( 
        part.local_tile_start.x + tile_idx.x,
        part.local_tile_start.y + tile_idx.y
    );
    int global_tile_id = global_tile.y * part.global_ntiles.x + global_tile.x;

    for( int i = 0; i < tile_np; i++ ) {
        // Use particle index in the random counter, next particle will use a different counter
        const uint4 rnd_ctr = make_uint4( i, global_tile_id, 0, udist_rnd_tag );
        // Generate 4 random deviates from 1 prng call
        const float4 norm = zrandom::rand_norm4_f( rnd_ctr, rnd_key );
        u[i] = make_float3(
            ufl.x + uth.x * norm.x,
            ufl.y + uth.y * norm.y,
            ufl.z + uth.z * norm.z
        );
    }
}

} // namespace kernel

/**
 * @brief Sets momentum of all particles in object using uth / ufl
 * 
 * @param part  Particle data
 * @param seed  Random number seed
 */
void udist::thermal::set( part::particles & part, unsigned int seed ) const {

    uint2 rnd_key = make_uint2( 12345 + seed, 67890 );

    const auto ntiles = part.get_local_ntiles();
    const auto view   = part.view();

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for( unsigned ty = 0; ty < ntiles.y; ty++ ) {
        for( unsigned tx = 0; tx < ntiles.x; tx++ ) {
            kernel::thermal( make_uint2( tx, ty ), view, rnd_key, uth, ufl );
        }
    }
}

namespace kernel {

/**
 * @brief Sets thermal momentum for a single tile, correcting local fluid
 *        fluctuations
 * 
 * @param tile_idx  Tile index
 * @param part      Particle data
 * @param rnd_key   PRNG key
 * @param uth       Thermal momentum
 * @param ufl       Fluid momentum
 * @param npmin     Minimum number of particles in cell to apply correction
 * @param npcell    (scratch) Number of particles per cell, size tile_dims.x * tile_dims.y
 * @param fluid     (scratch) Fluid momentum per cell, size tile_dims.x * tile_dims.y
 */
static void thermal_corr( const uint2 tile_idx, part::particles_view const part,
    const uint2 rnd_key, const float3 uth, const float3 ufl, int const npmin,
    int * const __restrict__ npcell, float3 * const __restrict__ fluid )
{
    const int   tile_id  = tile_idx.y * part.local_ntiles.x + tile_idx.x;
    
    const auto  tile_off = part.tile_offset[ tile_id ];
    const auto  tile_np  = part.tile_np[ tile_id ];

    const int ystride = part.tile_dims.x;

    float3 * __restrict__ const u  = &part.u [ tile_off ];
    int2   * __restrict__ const ix = &part.ix[ tile_off ];

    // Initialize random state variables
    uint2 global_tile = make_uint2( 
        part.local_tile_start.x + tile_idx.x,
        part.local_tile_start.y + tile_idx.y
    );
    int global_tile_id = global_tile.y * part.global_ntiles.x + global_tile.x;

    ///@brief Buffer size (number of cells)
    const unsigned bsize = part.tile_dims.x * part.tile_dims.y;

    for( unsigned idx = 0; idx < bsize; idx++ ) {
        fluid[idx] = make_float3( 0, 0, 0 );
        npcell[idx] = 0;
    }

    for( int i = 0; i < tile_np; i++ ) {
        // Use particle index in the random counter, next particle will use a different counter
        const uint4 rnd_ctr = make_uint4( i, global_tile_id, 0, udist_rnd_tag );
        // Generate 4 random deviates from 1 prng call
        const float4 norm = zrandom::rand_norm4_f( rnd_ctr, rnd_key );

        // Thermal part only; ufl is added after the correction
        float3 upart = make_float3(
            uth.x * norm.x,
            uth.y * norm.y,
            uth.z * norm.z
        );

        u[i] = upart;

        int const idx = ix[i].x + ystride * ix[i].y;

        npcell[idx] += 1;
        fluid[idx]  += upart;
    }

    for( unsigned idx = 0; idx < bsize; idx++ ) {
        if ( npcell[idx] > npmin ) {
            fluid[idx].x /= npcell[idx];
            fluid[idx].y /= npcell[idx];
            fluid[idx].z /= npcell[idx];
        } else {
            fluid[idx] = make_float3( 0, 0, 0 );
        }
    }

    // Remove the local fluid fluctuation and restore the temperature:
    // u_i - <u> ~ N( 0, uth^2 (n-1)/n ), so scaling by sqrt( n / (n-1) )
    // gives an exactly N( 0, uth^2 ) marginal again. Cells with npcell <= npmin
    // are left uncorrected (fluid = 0, scale = 1).
    for( int i = 0; i < tile_np; i++ ) {
        int const idx = ix[i].x + ystride * ix[i].y;
        int const n   = npcell[idx];
        float const scale = ( n > npmin ) ? std::sqrt( float(n) / float(n - 1) ) : 1.0f;
        u[i] = ufl + scale * ( u[i] - fluid[idx] );
    }
}

} // namespace kernel

/**
 * @brief Sets particle momentum correcting local ufl fluctuations
 *
 * @note In each cell with more than npmin particles the sample mean of the
 *       thermal momenta is removed and the remainder rescaled by
 *       sqrt( n / (n-1) ), so that the cell has fluid momentum exactly ufl and
 *       the particle momenta keep the requested uth.
 * 
 * @param part  Particle data
 * @param seed  Random number seed
 */
void udist::thermal_corr::set( part::particles & part, unsigned int seed ) const {

    uint2 rnd_key = make_uint2( 12345 + seed, 67890 );

    const auto ntiles = part.get_local_ntiles();
    const auto view   = part.view();

    const auto bsize = part.get_tile_dims().x * part.get_tile_dims().y;

    #pragma omp parallel
    {
        // Per-thread scratch buffers (shared memory in the CUDA version)
        std::vector< int >    npcell( bsize );
        std::vector< float3 > fluid( bsize );

        #pragma omp for collapse(2) schedule(dynamic)
        for( unsigned ty = 0; ty < ntiles.y; ty++ ) {
            for( unsigned tx = 0; tx < ntiles.x; tx++ ) {
                kernel::thermal_corr( make_uint2( tx, ty ), view, rnd_key, uth, ufl, npmin,
                    npcell.data(), fluid.data() );
            }
        }
    }
}

} // namespace em2ds
