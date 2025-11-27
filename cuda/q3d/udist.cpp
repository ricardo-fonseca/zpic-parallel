#include "udist.h"
#include "random.h"

namespace kernel {

__global__
void none( ParticleData const part ) {

    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int   tile_id  = tile_idx.y * part.ntiles.x + tile_idx.x;
    
    const auto  tile_off = part.offset[ tile_id ];
    const auto  tile_np  = part.np[ tile_id ];

    float3 * __restrict__ const u  = &part.u[ tile_off ];

    for( auto i = block_thread_rank(); i < tile_np; i += block_num_threads() )
        u[i] = make_float3( 0, 0, 0 );
}

}

/**
 * @brief Sets none(0 temperature, 0 fluid) u distribution
 * 
 * @param part  Particle data
 */
void UDistribution::None::set( Particles & part, unsigned int seed ) const {

    dim3 grid( part.ntiles.x, part.ntiles.y );
    dim3 block( 64 );

    kernel::none <<< grid, block >>> ( part );
}

namespace kernel {

__global__
void cold( ParticleData const part, const float3 ufl ) {

    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int   tile_id  = tile_idx.y * part.ntiles.x + tile_idx.x;
    
    const auto  tile_off = part.offset[ tile_id ];
    const auto  tile_np  = part.np[ tile_id ];

    float3 * __restrict__ const u  = &part.u[ tile_off ];

    for( auto i = block_thread_rank(); i < tile_np; i += block_num_threads() )
        u[i] = ufl;
}

}

/**
 * @brief Sets cold(0 temperatures) u distribution
 * 
 * @param part  Particle data
 */
void UDistribution::Cold::set( Particles & part, unsigned int seed ) const {

    dim3 grid( part.ntiles.x, part.ntiles.y );
    dim3 block( 64 );

    kernel::cold <<< grid, block >>> ( part, ufl );
}

namespace kernel {

__global__
void thermal( ParticleData const part, uint2 rnd_seed, const float3 uth, const float3 ufl ) {

    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int   tile_id  = tile_idx.y * part.ntiles.x + tile_idx.x;
    
    const auto  tile_off = part.offset[ tile_id ];
    const auto  tile_np  = part.np[ tile_id ];

    float3 * __restrict__ const u  = &part.u[ tile_off ];

    // Initialize random state variables
    uint2 state; double norm;
    zrandom::rand_init( 
        tile_id * block_num_threads() +  block_thread_rank(),
        rnd_seed, state, norm
    );

    for( auto i = block_thread_rank(); i < tile_np; i += block_num_threads() ) {
        u[i] = make_float3(
            ufl.x + uth.x * zrandom::rand_norm( state, norm ),
            ufl.y + uth.y * zrandom::rand_norm( state, norm ),
            ufl.z + uth.z * zrandom::rand_norm( state, norm )
        );
    }
}

}

/**
 * @brief Sets momentum of all particles in object using uth / ufl
 * 
 */
void UDistribution::Thermal::set( Particles & part, unsigned int seed ) const
{
    uint2 rnd_seed{ 12345 + seed, 67890 };
    dim3 grid( part.ntiles.x, part.ntiles.y );
    dim3 block( 64 );

    kernel::thermal <<< grid, block >>> ( part, rnd_seed, uth, ufl );
}

namespace kernel {

__global__
void thermal_corr( ParticleData const part, uint2 rnd_seed, const float3 uth, const float3 ufl, int const npmin )
{
    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int   tile_id  = tile_idx.y * part.ntiles.x + tile_idx.x;
    
    const auto  tile_off = part.offset[ tile_id ];
    const auto  tile_np  = part.np[ tile_id ];

    const auto ystride = part.nx.x;

    float3 * __restrict__ const u  = &part.u [ tile_off ];
    int2   * __restrict__ const ix = &part.ix[ tile_off ];

    // Initialize random state variables
    uint2 state; double norm;
    zrandom::rand_init( 
        tile_id * block_num_threads() +  block_thread_rank(),
        rnd_seed, state, norm
    );

    // Get shared memory addresses
    extern __shared__ char block_shm[];

    ///@brief Buffer size (number of cells)
    auto const bsize = part.nx.x * part.nx.y;
    int * const __restrict__ npcell   = reinterpret_cast<int*>    ( & block_shm[0] );
    float3 * const __restrict__ fluid = reinterpret_cast<float3*> ( & npcell[ bsize ] );

    for( auto idx = block_thread_rank(); idx < bsize; idx += block_num_threads() ) {
        fluid[idx] = make_float3( 0, 0, 0 );
        npcell[idx] = 0;
    }

    block_sync();

    for( auto i = block_thread_rank(); i < tile_np; i += block_num_threads() ) {
        float3 upart = make_float3(
            uth.x * zrandom::rand_norm( state, norm ),
            uth.y * zrandom::rand_norm( state, norm ),
            uth.z * zrandom::rand_norm( state, norm )
        );

        u[i] = upart;

        int const idx = ix[i].x + ystride * ix[i].y;

        block::atomic_fetch_add( & npcell[idx], 1 );
        block::atomic_fetch_add( & fluid[idx].x, upart.x );
        block::atomic_fetch_add( & fluid[idx].y, upart.y );
        block::atomic_fetch_add( & fluid[idx].z, upart.z );
    }

    block_sync();

    for( auto idx = block_thread_rank(); idx < bsize; idx += block_num_threads() ) {
        if ( npcell[idx] > npmin ) {
            fluid[idx].x /= npcell[idx];
            fluid[idx].y /= npcell[idx];
            fluid[idx].z /= npcell[idx];
        } else {
            fluid[idx] = make_float3( 0, 0, 0 );
        }
    }

    block_sync();

    for( auto i = block_thread_rank(); i < tile_np; i += block_num_threads() ) {
        int const idx = ix[i].x + ystride * ix[i].y;
        u[i] += ufl - fluid[idx];
    }
}

}

/**
 * @brief Sets particle momentum correcting local ufl fluctuations
 * 
 */
void UDistribution::ThermalCorr::set( Particles & part, unsigned int seed ) const {

    uint2 rnd_seed{ 12345 + seed, 67890 };
    dim3 grid( part.ntiles.x, part.ntiles.y );
    dim3 block( 64 );

    const auto bsize = part.nx.x * part.nx.y;
    const size_t shm_size = bsize * ( sizeof( float3 ) + sizeof( int ) );
    block::set_shmem_size( kernel::thermal_corr, shm_size );
    kernel::thermal_corr <<< grid, block, shm_size >>> ( part, rnd_seed, uth, ufl, npmin );
}