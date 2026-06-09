#include "udist.h"
#include "random.h"

// For debug purposes only
#include "parallel.h"

/**
 * @brief Sets none(0 temperature, 0 fluid) u distribution
 * 
 * @param part  Particle data
 */
void UDistribution::None::set( Particles & part, unsigned int seed ) const {

    #pragma omp parallel for schedule(dynamic)
    for( unsigned tid = 0; tid < part.ntiles.x * part.ntiles.y; tid ++ ) {

        const int offset = part.offset[tid];
        const int np     = part.np[tid];
        float3 * __restrict__ const u  = &part.u[ offset ];

        for( auto i = 0; i < np; i++ ) u[i] = make_float3(0,0,0);
    }
}

/**
 * @brief Sets cold(0 temperatures) u distribution
 * 
 * @param part  Particle data
 */
void UDistribution::Cold::set( Particles & part, unsigned int seed ) const {

    #pragma omp parallel for schedule(dynamic)
    for( unsigned tid = 0; tid < part.ntiles.x * part.ntiles.y; tid ++ ) {

        const int offset = part.offset[tid];
        const int np     = part.np[tid];
        float3 * __restrict__ const u  = &part.u[ offset ];

        for( auto i = 0; i < np; i++ ) u[i] = ufl;
    }
}

/**
 * @brief Sets momentum of all particles in object using uth / ufl
 * 
 */
void UDistribution::Thermal::set( Particles & part, unsigned int seed ) const {

    uint2 rnd_seed = make_uint2( 12345 + seed, 67890 );

    #pragma omp parallel for schedule(dynamic)
    for( unsigned tid = 0; tid < part.ntiles.x * part.ntiles.y; tid ++ ) {

        // Get global tile index
        uint2 global_tile = make_uint2( 
            part.tile_off.x + tid % part.ntiles.x,
            part.tile_off.y + tid / part.ntiles.x
        );
        int global_tid = global_tile.y * part.global_ntiles.x + global_tile.x;

        // Initialize random state variables
        uint2 state;
        double norm;
        zrandom::rand_init( global_tid, rnd_seed, state, norm );

        const int offset = part.offset[tid];
        const int np     = part.np[tid];
        float3 * __restrict__ const u  = &part.u[ offset ];

        for( int i = 0; i < np; i++ ) {
            u[i] = make_float3(
                ufl.x + uth.x * zrandom::rand_norm( state, norm ),
                ufl.y + uth.y * zrandom::rand_norm( state, norm ),
                ufl.z + uth.z * zrandom::rand_norm( state, norm )
            );
        }
    }
}

/**
 * @brief Sets particle momentum correcting local ufl fluctuations
 * 
 */
void UDistribution::ThermalCorr::set( Particles & part, unsigned int seed ) const {

    uint2 rnd_seed = make_uint2( 12345 + seed, 67890 );

    const auto bsize = part.nx.x * part.nx.y;
    const int ystride = part.nx.x;

    #pragma omp parallel for schedule(dynamic)   
    for( unsigned tid = 0; tid < part.ntiles.x * part.ntiles.y; tid ++ ) {

        // fluid[] and npcell[] should be in block shared memory
        float3 fluid[ bsize ];
        int npcell[ bsize ];

        for( unsigned idx = 0; idx < bsize; idx ++ ) {
            fluid[idx] = make_float3(0,0,0);
            npcell[idx] = 0;
        }

        //sync - not required with 1 thread per tile

        // Get global tile index
        uint2 global_tile = make_uint2( 
            part.tile_off.x + tid % part.ntiles.x,
            part.tile_off.y + tid / part.ntiles.x
        );
        int global_tid = global_tile.y * part.global_ntiles.x + global_tile.x;

        // Initialize random state variables
        uint2 state;
        double norm;
        zrandom::rand_init( global_tid, rnd_seed, state, norm );

        const int offset = part.offset[tid];
        const int np     = part.np[tid];
        float3 * __restrict__ const u  = &part.u[ offset ];
        int2 const * const __restrict__ ix = &part.ix[offset];

        for( auto i = 0; i < np; i++ ) {
            float3 upart = make_float3(
                uth.x * zrandom::rand_norm( state, norm ),
                uth.y * zrandom::rand_norm( state, norm ),
                uth.z * zrandom::rand_norm( state, norm )
            );
    
            u[i] = upart;

            int const idx = ix[i].x + ystride * ix[i].y;

            // The following should be atomic ops
            npcell[idx] += 1;
            fluid[ idx ] += upart;
        }

        //sync - not required with 1 thread per tile

        for( unsigned idx = 0; idx < bsize; idx++ ) {
            if ( npcell[idx] > npmin ) {
                fluid[idx].x /= npcell[idx];
                fluid[idx].y /= npcell[idx];
                fluid[idx].z /= npcell[idx];
            } else {
                fluid[idx] = make_float3(0,0,0);
            }
        }

        //sync - not required with 1 thread per tile

        for( int i = 0; i < np; i++ ) {
            int const idx = ix[i].x + ystride * ix[i].y;
            u[i] += ufl - fluid[idx];
        }
    }
}