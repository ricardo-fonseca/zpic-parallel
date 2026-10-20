#include "part/particles.hpp"
#include "udist.hpp"
#include "util/random.cuh"


namespace kernel {

__global__
void none( part::particles_view const part ) {

    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int   tile_id  = tile_idx.y * part.local_ntiles.x + tile_idx.x;

    const auto  tile_off = part.tile_offset[ tile_id ];
    const auto  tile_np  = part.tile_np[ tile_id ];

    float3 * __restrict__ const u  = &part.u[ tile_off ];

    for( auto i = gpu::block::thread_rank(); i < tile_np; i += gpu::block::num_threads() )
        u[i] = make_float3( 0, 0, 0 );
}

} // namespace kernel

/**
 * @brief Sets none(0 temperature, 0 fluid) u distribution
 * 
 * @param part  Particle data
 */
void udist::none::set( part::particles & part, unsigned int seed ) const {

    auto ntiles = part.get_local_ntiles();
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 64 );

    kernel::none <<< grid, block >>> ( part.view() );
}

namespace kernel {

__global__
void cold( part::particles_view const part, const float3 ufl ) {

    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int   tile_id  = tile_idx.y * part.local_ntiles.x + tile_idx.x;
    
    const auto  tile_off = part.tile_offset[ tile_id ];
    const auto  tile_np  = part.tile_np[ tile_id ];

    float3 * __restrict__ const u  = &part.u[ tile_off ];

    for( auto i = gpu::block::thread_rank(); i < tile_np; i += gpu::block::num_threads() )
        u[i] = ufl;
}

} // namespace kernel

/**
 * @brief Sets cold(0 temperatures) u distribution
 * 
 * @param part  Particle data
 */
void udist::cold::set( part::particles & part, unsigned int seed ) const {

    auto ntiles = part.get_local_ntiles();
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 64 );

    kernel::cold <<< grid, block >>> ( part.view(), ufl );
}

namespace kernel {

__global__
void thermal( part::particles_view const part, uint2 rnd_seed, const float3 uth, const float3 ufl ) {

    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int   tile_id  = tile_idx.y * part.local_ntiles.x + tile_idx.x;
    
    const auto  tile_off = part.tile_offset[ tile_id ];
    const auto  tile_np  = part.tile_np[ tile_id ];

    float3 * __restrict__ const u  = &part.u[ tile_off ];

    // Initialize random state variables
    uint2 global_tile = make_uint2( 
        part.local_tile_start.x + tile_idx.x,
        part.local_tile_start.y + tile_idx.y
    );
    int global_tid = global_tile.y * part.global_ntiles.x + global_tile.x;

    uint2 state; double norm;
    zrandom::rand_init( 
        global_tid * gpu::block::num_threads() +  gpu::block::thread_rank(),
        rnd_seed, state, norm
    );

    for( auto i = gpu::block::thread_rank(); i < tile_np; i += gpu::block::num_threads() ) {
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
void udist::thermal::set( part::particles & part, unsigned int seed ) const {

    uint2 rnd_seed = make_uint2( 12345 + seed, 67890 );

    auto ntiles = part.get_local_ntiles();
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 64 );
    kernel::thermal <<< grid, block >>> ( part.view(), rnd_seed, uth, ufl );
}

namespace kernel {

__global__
void thermal_corr( part::particles_view const part, uint2 rnd_seed, const float3 uth, const float3 ufl, int const npmin )
{
    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int   tile_id  = tile_idx.y * part.local_ntiles.x + tile_idx.x;
    
    const auto  tile_off = part.tile_offset[ tile_id ];
    const auto  tile_np  = part.tile_np[ tile_id ];

    const auto ystride = part.tile_dims.x;

    float3 * __restrict__ const u  = &part.u [ tile_off ];
    int2   * __restrict__ const ix = &part.ix[ tile_off ];

    // Initialize random state variables
    uint2 global_tile = make_uint2( 
        part.local_tile_start.x + tile_idx.x,
        part.local_tile_start.y + tile_idx.y
    );
    int global_tid = global_tile.y * part.global_ntiles.x + global_tile.x;

    uint2 state; double norm;
    zrandom::rand_init( 
        global_tid * gpu::block::num_threads() +  gpu::block::thread_rank(),
        rnd_seed, state, norm
    );

    // Get shared memory addresses
    extern __shared__ char block_shm[];

    ///@brief Buffer size (number of cells)
    auto const bsize = part.tile_dims.x * part.tile_dims.y;
    int * const __restrict__ npcell   = reinterpret_cast<int*>    ( & block_shm[0] );
    float3 * const __restrict__ fluid = reinterpret_cast<float3*> ( & npcell[ bsize ] );

    for( auto idx = gpu::block::thread_rank(); idx < bsize; idx += gpu::block::num_threads() ) {
        fluid[idx] = make_float3( 0, 0, 0 );
        npcell[idx] = 0;
    }

    gpu::block::sync();

    for( auto i = gpu::block::thread_rank(); i < tile_np; i += gpu::block::num_threads() ) {
        float3 upart = make_float3(
            uth.x * zrandom::rand_norm( state, norm ),
            uth.y * zrandom::rand_norm( state, norm ),
            uth.z * zrandom::rand_norm( state, norm )
        );

        u[i] = upart;

        int const idx = ix[i].x + ystride * ix[i].y;

        gpu::block::atomic_fetch_add( & npcell[idx], 1 );
        gpu::block::atomic_fetch_add( & fluid[idx].x, upart.x );
        gpu::block::atomic_fetch_add( & fluid[idx].y, upart.y );
        gpu::block::atomic_fetch_add( & fluid[idx].z, upart.z );
    }

    gpu::block::sync();

    for( auto idx = gpu::block::thread_rank(); idx < bsize; idx += gpu::block::num_threads() ) {
        if ( npcell[idx] > npmin ) {
            fluid[idx].x /= npcell[idx];
            fluid[idx].y /= npcell[idx];
            fluid[idx].z /= npcell[idx];
        } else {
            fluid[idx] = make_float3( 0, 0, 0 );
        }
    }

    gpu::block::sync();

    for( auto i = gpu::block::thread_rank(); i < tile_np; i += gpu::block::num_threads() ) {
        int const idx = ix[i].x + ystride * ix[i].y;
        u[i] += ufl - fluid[idx];
    }
}

} // namespace kernel

/**
 * @brief Sets particle momentum correcting local ufl fluctuations
 * 
 */
void udist::thermal_corr::set( part::particles & part, unsigned int seed ) const {

    uint2 rnd_seed = make_uint2( 12345 + seed, 67890 );

    auto ntiles = part.get_local_ntiles();
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 64 );

    const auto bsize = part.get_tile_dims().x * part.get_tile_dims().y;
    const size_t shm_size = bsize * ( sizeof( float3 ) + sizeof( int ) );
    gpu::block::set_shmem_size( kernel::thermal_corr, shm_size );
    kernel::thermal_corr <<< grid, block, shm_size >>> ( part.view(), rnd_seed, uth, ufl, npmin );
}