#include "udist.h"
#include "random.h"

/**
 * @brief Sets none(0 temperature, 0 fluid) u distribution
 * 
 * @param part  Particle data
 */
void UDistribution::None::set( Particles & part, unsigned int seed ) const {

    sycl::range<1> local{ 8 };
    sycl::range<1> global{ part.ntiles.x * part.ntiles.y };

    part.queue.submit([&](sycl::handler &h) {

        auto part_offset = part.offset;
        auto part_np     = part.np;
        auto part_u      = part.u;

        h.parallel_for( 
            sycl::nd_range{ global * local, local },
            [=](sycl::nd_item<1> it) {

            const auto tile_id  = it.get_group_linear_id();

            const int offset               = part_offset[tile_id];
            const int np                   = part_np[tile_id];
            float3 * __restrict__ const u  = &part_u[ offset ];

            for( auto i = it.get_local_id(0); i < np; i += it.get_local_range(0) )
                u[i] = make_float3(0,0,0);
        });
    });
}

/**
 * @brief Sets cold(0 temperatures) u distribution
 * 
 * @param part  Particle data
 */
void UDistribution::Cold::set( Particles & part, unsigned int seed ) const {

    sycl::range<1> local{ 8 };
    sycl::range<1> global{ part.ntiles.x * part.ntiles.y };

    part.queue.submit([&](sycl::handler &h) {

        auto part_offset = part.offset;
        auto part_np     = part.np;
        auto part_u      = part.u;
        auto ufl         = this -> ufl;

        h.parallel_for( 
            sycl::nd_range{ global * local, local },
            [=](sycl::nd_item<1> it) {

            const auto tile_id  = it.get_group_linear_id();

            const int offset = part_offset[tile_id];
            const int np     = part_np[tile_id];
            float3 * __restrict__ const u  = &part_u[ offset ];

            for( auto i = it.get_local_id(0); i < np; i += it.get_local_range(0) )
                u[i] = ufl;
        });
    });
}

/**
 * @brief Sets momentum of all particles in object using uth / ufl
 * 
 */
void UDistribution::Thermal::set( Particles & part, unsigned int seed ) const {

    uint2 rnd_seed { 12345 + seed, 67890 };

    sycl::range<1> local{ 8 };
    sycl::range<1> global{ part.ntiles.x * part.ntiles.y };

    part.queue.submit([&](sycl::handler &h) {

        auto part_offset = part.offset;
        auto part_np     = part.np;
        auto part_u      = part.u;
        auto ufl         = this -> ufl;
        auto uth         = this -> uth;

        h.parallel_for( 
            sycl::nd_range{ global * local, local },
            [=](sycl::nd_item<1> it) {

            const auto tile_id  = it.get_group_linear_id();

            // Initialize random state variables
            uint2 state;
            double norm;
            zrandom::rand_init( it.get_global_linear_id(), rnd_seed, state, norm );

            const int offset = part_offset[tile_id];
            const int np     = part_np[tile_id];
            float3 * __restrict__ const u  = &part_u[ offset ];

            for( int i = 0; i < np; i++ ) {
                u[i] = make_float3(
                    ufl.x + uth.x * zrandom::rand_norm( state, norm ),
                    ufl.y + uth.y * zrandom::rand_norm( state, norm ),
                    ufl.z + uth.z * zrandom::rand_norm( state, norm )
                );
            }
        });
    });
}

/**
 * @brief Sets particle momentum correcting local ufl fluctuations
 * 
 */
void UDistribution::ThermalCorr::set( Particles & part, unsigned int seed ) const {

    uint2 rnd_seed{ 12345 + seed, 67890 };

    const auto bsize = part.nx.x * part.nx.y;
    const int ystride = part.nx.x;

    sycl::range<1> local{ 8 };
    sycl::range<1> global{ part.ntiles.x * part.ntiles.y };

    part.queue.submit([&](sycl::handler &h) {

        auto part_offset = part.offset;
        auto part_np     = part.np;
        auto part_u      = part.u;
        auto part_ix     = part.ix;
        auto ufl         = this -> ufl;
        auto uth         = this -> uth;
        auto npmin       = this -> npmin;

        /// @brief [shared] Local copy of E-field
        auto fluid = sycl::local_accessor< float3, 1 > ( bsize, h );
        /// @brief [shared] Local copy of B-field
        auto npcell = sycl::local_accessor< int, 1 > ( bsize, h );

        h.parallel_for( 
            sycl::nd_range{ global * local, local },
            [=](sycl::nd_item<1> it) {

            const auto tile_id  = it.get_group_linear_id();
            const int offset = part_offset[tile_id];
            const int np     = part_np[tile_id];
            float3 * __restrict__ const u  = &part_u[ offset ];
            int2 const * const __restrict__ ix = &part_ix[offset];

            // Initialize random state variables
            uint2 state;
            double norm;
            zrandom::rand_init( it.get_global_linear_id(), rnd_seed, state, norm );

            for( auto idx = it.get_local_id(0); idx < bsize; idx += it.get_local_range(0) ) {
                fluid[idx] = make_float3(0,0,0);
                npcell[idx] = 0;
            }

            it.barrier();

            for( auto i = it.get_local_id(0); i < np; i += it.get_local_range(0) ) {
                float3 upart = make_float3 (
                    uth.x * zrandom::rand_norm( state, norm ),
                    uth.y * zrandom::rand_norm( state, norm ),
                    uth.z * zrandom::rand_norm( state, norm )
                );
        
                u[i] = upart;

                int const idx = ix[i].x + ystride * ix[i].y;

                device::local::atomicAdd( & npcell[idx], 1 );
                device::local::atomicAdd( & fluid[idx].x, upart.x );
                device::local::atomicAdd( & fluid[idx].y, upart.y );
                device::local::atomicAdd( & fluid[idx].z, upart.z );
            }

            it.barrier();

            for( auto idx = it.get_local_id(0); idx < bsize; idx += it.get_local_range(0) ) {
                if ( npcell[idx] > npmin ) {
                    fluid[idx].x /= npcell[idx];
                    fluid[idx].y /= npcell[idx];
                    fluid[idx].z /= npcell[idx];
                } else {
                    fluid[idx] = make_float3(0,0,0);
                }
            }

            it.barrier();

            for( auto i = it.get_local_id(0); i < np; i += it.get_local_range(0) ) {
                int const idx = ix[i].x + ystride * ix[i].y;
                u[i] += ufl - fluid[idx];
            }
        });
    });
}