#include "density.h"

/**
 * This is required for using bnd<unsigned int> inside a Sycl kernel
 */
template<>
struct sycl::is_device_copyable<bnd<unsigned int>> : std::true_type {};

/**
 * @brief kernel for injecting a uniform plasma density (mk1)
 * 
 * Particles in the same cell are injected contiguously
 * 
 * @param range     Cell range (global) to inject particles in 
 * @param ppc       Number of particles per cell 
 * @param nx        Number of cells in tile 
 * @param tiles     Particle tile information 
 * @param data      Particle data 
 */
inline void inject_uniform_kernel_mk1( 
    sycl::nd_item<2> it, 
    bnd<unsigned int> range,
    uint2 const ppc,
    ParticleData const part )
{
    const int2 nx = make_int2( part.nx.x, part.nx.y );
    
    // Tile ID
    const int2 tile_idx = make_int2( it.get_group(0), it.get_group(1));
    const int tile_id = tile_idx.y * part.ntiles.x + tile_idx.x;

    // Store number of particles before injection
    const int np = part.np[ tile_id ];

    it.barrier();

    // Find injection range in tile coordinates
    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;
    
    // If range overlaps with tile
    if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
        ( rj0 < nx.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nx.x ) ri1 = nx.x-1;
        if (rj1 >= nx.y ) rj1 = nx.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        const int offset =  part.offset[ tile_id ];
        int2   * __restrict__ const ix = &part.ix[ offset ];
        float2 * __restrict__ const x  = &part.x[ offset ];
        float3 * __restrict__ const u  = &part.u[ offset ];

        const int np_cell = ppc.x * ppc.y;

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        // Each thread takes 1 cell
        for( int idx = it.get_local_id(0); idx < vol; idx += it.get_local_range(0) ) {
            int2 const cell = make_int2(
                idx % row + ri0,
                idx / row + rj0
            );

            int part_idx = np + idx * np_cell;

            for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
                for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                    float2 const pos = make_float2(
                        dpcx * ( i0 + 0.5 ) - 0.5,
                        dpcy * ( i1 + 0.5 ) - 0.5
                    );
                    ix[ part_idx ] = cell;
                    x[ part_idx ] = pos;
                    u[ part_idx ] = make_float3(0,0,0);
                    part_idx++;
                }
            }
        }

        // Update global number of particles in tile
        if ( it.get_local_id(0) == 0 ) {
            part.np[ tile_id ] = np + vol * np_cell ;
        }
    }
}

/**
 * @brief kernel for injecting a uniform plasma density (mk2)
 * 
 * Places contiguous particles in different cells. This minimizes memory collisions
 * when depositing current, especially for very low temperatures.
 * 
 * @param range     Cell range (global) to inject particles in 
 * @param ppc       Number of particles per cell 
 * @param nx        Number of cells in tile 
 * @param data      Particle data 
 */
inline void inject_uniform_kernel( 
    sycl::nd_item<2> it, 
    bnd<unsigned int> range,
    uint2 const ppc,
    ParticleData const part )
{
    // This must be signed
    const int2 nx = make_int2( part.nx.x, part.nx.y );
    
    // Tile ID
    const int2 tile_idx = make_int2( it.get_group(0), it.get_group(1));
    const int tile_id = tile_idx.y * part.ntiles.x + tile_idx.x;

    // Store number of particles before injection
    const int np = part.np[ tile_id ];

    it.barrier();

    // Find injection range in tile coordinates
    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;

    
    // If range overlaps with tile
    if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
        ( rj0 < nx.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nx.x ) ri1 = nx.x-1;
        if (rj1 >= nx.y ) rj1 = nx.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        const int offset =  part.offset[ tile_id ];

        int2   * __restrict__ const ix = &part.ix[ offset ];
        float2 * __restrict__ const x  = &part.x[ offset ];
        float3 * __restrict__ const u  = &part.u[ offset ];

        const int np_cell = ppc.x * ppc.y;

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
            for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                float2 const pos = make_float2(
                    dpcx * ( i0 + 0.5 ) - 0.5,
                    dpcy * ( i1 + 0.5 ) - 0.5
                );

                int ppc_idx = i1 * ppc.x + i0;

                // Each thread takes 1 cell
                for( int idx = it.get_local_id(0); idx < vol; idx += it.get_local_range(0) ) {
                    int2 const cell = make_int2( 
                        idx % row + ri0,
                        idx / row + rj0
                    );

                    int part_idx = np + vol * ppc_idx + idx;

                    ix[ part_idx ] = cell;
                    x[ part_idx ] = pos;
                    u[ part_idx ] = make_float3(0,0,0);
                }
                ppc_idx ++;
            }
        }

        // Update global number of particles in tile
        if ( it.get_local_id(0) == 0 ) {
            part.np[ tile_id ] = np + vol * np_cell ;
        }
    }
}

void Density::Uniform::inject( Particles & particles, 
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const
{
    ParticleData part = particles;

    // 8×1 work items per group
    sycl::range<2> local{ 8, 1 };

    // ntiles.x × ntiles.y groups
    sycl::range<2> global{ part.ntiles.x, part.ntiles.y };

    particles.queue.submit([&](sycl::handler &h) {

        h.parallel_for( 
            sycl::nd_range{ global * local, local },
            [=](sycl::nd_item<2> it) { 

#if 0
            // Use only for benchmarking
            inject_uniform_kernel_mk1( it, range, ppc, part );
#else
            inject_uniform_kernel( it, range, ppc, part );
#endif

        });
    });
    particles.queue.wait();
}

void Density::Uniform::np_inject( Particles & particles, 
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range,
    int * np ) const
{
    ParticleData part = particles;

    // Run serial inside group
    sycl::range<2> local{ 1, 1 };

    // ntiles.x × ntiles.y groups
    sycl::range<2> global{ part.ntiles.x, part.ntiles.y };

    particles.queue.submit([&](sycl::handler &h) {

        const int2 nx = make_int2( part.nx.x, part.nx.y );

        h.parallel_for( 
            sycl::nd_range{ global * local, local },
            [=](sycl::nd_item<2> it) { 

            const int2 tile_idx = make_int2( it.get_group(0), it.get_group(1));
            const int tile_id = tile_idx.y * part.ntiles.x + tile_idx.x;

            // Find injection range in tile coordinates
            int ri0 = range.x.lower - tile_idx.x * nx.x;
            int ri1 = range.x.upper - tile_idx.x * nx.x;

            int rj0 = range.y.lower - tile_idx.y * nx.y;
            int rj1 = range.y.upper - tile_idx.y * nx.y;

            int local_np;

            // If range overlaps with tile
            if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
                ( rj0 < nx.y ) && ( rj1 >= 0 )) {
                
                // Limit to range inside this tile
                if (ri0 < 0) ri0 = 0;
                if (rj0 < 0) rj0 = 0;
                if (ri1 >= nx.x ) ri1 = nx.x-1;
                if (rj1 >= nx.y ) rj1 = nx.y-1;

                int const row = (ri1-ri0+1);
                int const vol = (rj1-rj0+1) * row;

                local_np = vol * ppc.x * ppc.y;

            } else {
                local_np = 0;
            }

            np[ tile_id ] = local_np;        
        });
    });
    particles.queue.wait();
}

/**
 * @brief kernel for injecting step profile
 * 
 * @note This kernel must be launched using a 2D grid with 1 block per tile
 * 
 * @param range     Cell range (global) to inject particles in
 * @param step      Step position normalized to cell size
 * @param ppc       Number of particles per cell
 * @param nx        Tile size
 * @param d_tiles    Tile information
 * @param d_ix      Particle buffer (cells)
 * @param d_x       Particle buffer (positions)
 * @param d_u       Particle buffer (momenta)
 */

template < coord::cart dir >
void inject_step_kernel( 
    sycl::nd_item<2> it, 
    bnd<unsigned int> range,
    const float step, const uint2 ppc,
    ParticleData const part, int * np_local, int * tmp )
{
    const int2 nx = make_int2( part.nx.x, part.nx.y );
    
    // Tile ID
    const int2 tile_idx = make_int2( it.get_group(0), it.get_group(1));
    const int tile_id = tile_idx.y * part.ntiles.x + tile_idx.x;

    // Store number of particles before injection
    *np_local = part.np[ tile_id ];

    it.barrier();

    // Find injection range in tile coordinates
    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;
    
    // If range overlaps with tile
    if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
        ( rj0 < nx.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nx.x ) ri1 = nx.x-1;
        if (rj1 >= nx.y ) rj1 = nx.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        const int offset =  part.offset[ tile_id ];
        int2   * __restrict__ ix = &part.ix[ offset ];
        float2 * __restrict__ x  = &part.x[ offset ];
        float3 * __restrict__ u  = &part.u[ offset ];

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = tile_idx.x * nx.x;
        const int shifty = tile_idx.y * nx.y;

        for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
            for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                float2 const pos = make_float2(
                    dpcx * ( i0 + 0.5 ) - 0.5,
                    dpcy * ( i1 + 0.5 ) - 0.5
                );
                for( int idx = it.get_local_id(0); idx < vol; idx += it.get_local_range(0)) {
                    int2 const cell = make_int2(
                        idx % row + ri0,
                        idx / row + rj0
                    );

                    float t;
                    if ( dir == coord::x ) t = (shiftx + cell.x) + (pos.x + 0.5);
                    if ( dir == coord::y ) t = (shifty + cell.y) + (pos.y + 0.5);

                    int inj = t > step;
                    int off = device::group::exscan_add( it, tmp, inj );

                    if ( inj ) {
                        const int k = np_local[0] + off;
                        ix[ k ] = cell;
                        x[ k ]  = pos;
                        u[ k ]  = make_float3(0,0,0);
                    }

                    auto sg = it.get_sub_group();
                    inj = device::subgroup::reduce_add( sg, inj );
                    if ( sg.get_local_id() == 0 ) {
                        device::local::atomicAdd( np_local, inj );
                    }
                    it.barrier();
                }
            }
        }

        if ( it.get_local_id(0) == 0 ) {
            part.np[ tile_id ] = np_local[0];
        }
    }
}

/**
 * @brief Inject a step density profile
 * 
 * @param particles     Particle data
 * @param ppc           Number of particles per cell
 * @param dx            Cell size
 * @param ref           Reference for step position
 * @param range         Global cell range for injection
 */
void Density::Step::inject( Particles & particles,
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const
{

    ParticleData part = particles;

    // 8×1 work items per group
    sycl::range<2> local{ 8, 1 };

    // ntiles.x × ntiles.y groups
    sycl::range<2> global{ part.ntiles.x, part.ntiles.y };

    particles.queue.submit([&](sycl::handler &h) {

        /// @brief [shared] Local number of particles
        auto np_local = sycl::local_accessor< int, 1 > ( 1, h );

        const int max_num_sub_groups = particles.queue.get_device().get_info<sycl::info::device::max_num_sub_groups>();
        
        /// @brief [shared] Temporary memory for exscan calculations
        auto tmp = sycl::local_accessor< int, 1 > ( max_num_sub_groups, h );

        float step_pos;

        switch( dir ) {
        case( coord::x ):
            step_pos = (pos - ref.x) / dx.x;
            h.parallel_for( 
                sycl::nd_range{ global * local, local },
                [=](sycl::nd_item<2> it) {
                inject_step_kernel <coord::x> ( it, range, step_pos, ppc, part, &np_local[0], &tmp[0] );
            });
            break;
        case( coord::y ):
            step_pos = (pos - ref.y) / dx.y;
            h.parallel_for( 
                sycl::nd_range{ global * local, local },
                [=](sycl::nd_item<2> it) {
                inject_step_kernel <coord::y> ( it, range, step_pos, ppc, part, &np_local[0], &tmp[0] );
            });
            break;
        }
    });
    particles.queue.wait();

}

template < coord::cart dir >
void np_inject_step_kernel( 
    sycl::nd_item<2> it, 
    bnd<unsigned int> range,
    const float step, const uint2 ppc,
    ParticleData const part, int * np_local, int * np )
{
    // Tile ID
    const int2 tile_idx = make_int2( it.get_group(0), it.get_group(1));
    const int tile_id = tile_idx.y * part.ntiles.x + tile_idx.x;

    // Store number of particles before injection
    *np_local = 0;
    it.barrier();

    // Find injection range in tile coordinates
    const int2 nx = make_int2( part.nx.x, part.nx.y );
    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;
    
    int inj_np = 0;

    // If range overlaps with tile
    if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
        ( rj0 < nx.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nx.x ) ri1 = nx.x-1;
        if (rj1 >= nx.y ) rj1 = nx.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = tile_idx.x * nx.x;
        const int shifty = tile_idx.y * nx.y;

        for( int idx = it.get_local_id(0); idx < vol; idx += it.get_local_range(0) ) {
            int2 const cell = make_int2(
                idx % row + ri0,
                idx / row + rj0
            );
            for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
                for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                    float2 const pos = make_float2(
                        dpcx * ( i0 + 0.5 ) - 0.5,
                        dpcy * ( i1 + 0.5 ) - 0.5
                    );
                    float t;
                    if ( dir == coord::x ) t = (shiftx + cell.x) + (pos.x + 0.5);
                    if ( dir == coord::y ) t = (shifty + cell.y) + (pos.y + 0.5);
                    
                    int inj = t > step;
                    inj_np += inj;
                }
            }
        }
    }

    auto sg = it.get_sub_group();
    inj_np = device::subgroup::reduce_add( sg, inj_np );
    if ( sg.get_local_id() == 0 ) {
        device::local::atomicAdd( np_local, inj_np );
    } 

    it.barrier();

    if ( it.get_local_id(0) == 0 ) {
        np[ tile_id ] = *np_local;
    }
}

void Density::Step::np_inject( Particles & particles, 
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range,
    int * np ) const
{
    ParticleData part = particles;

    // 8×1 work items per group
    sycl::range<2> local{ 8, 1 };

    // ntiles.x × ntiles.y groups
    sycl::range<2> global{ part.ntiles.x, part.ntiles.y };

    particles.queue.submit([&](sycl::handler &h) {

        /// @brief [shared] Local number of particles
        auto np_local = sycl::local_accessor< int, 1 > ( 1, h );
        float step_pos;

        switch( dir ) {
        case( coord::x ):
            step_pos = (pos - ref.x) / dx.x;
            h.parallel_for( 
                sycl::nd_range{ global * local, local },
                [=](sycl::nd_item<2> it) {

                np_inject_step_kernel <coord::x> ( it, range, step_pos, ppc, part, &np_local[0], np );
            });
            break;
        case( coord::y ):
            step_pos = (pos - ref.y) / dx.y;
            h.parallel_for( 
                sycl::nd_range{ global * local, local },
                [=](sycl::nd_item<2> it) {

                np_inject_step_kernel <coord::y> ( it, range, step_pos, ppc, part, &np_local[0], np );
            });
            break;
        }
    });
    particles.queue.wait();

    // device::print( np, part.ntiles.x*part.ntiles.y, "Density::Step::np_inject() - np", particles.queue );
}

template < coord::cart dir >
void inject_slab_kernel( 
    sycl::nd_item<2> it,
    bnd<unsigned int> range,
    const float start, const float finish, const uint2 ppc,
    ParticleData const part, int * np_local, int * tmp )
{
    // Tile ID
    const int2 tile_idx = make_int2( it.get_group(0), it.get_group(1));
    const int tile_id = tile_idx.y * part.ntiles.x + tile_idx.x;

    // Store number of particles before injection
    *np_local = part.np[ tile_id ];

    it.barrier();

    // Find injection range in tile coordinates
    const int2 nx = make_int2( part.nx.x, part.nx.y );
    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;

    // If range overlaps with tile
    if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
        ( rj0 < nx.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nx.x ) ri1 = nx.x-1;
        if (rj1 >= nx.y ) rj1 = nx.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        const int offset =  part.offset[ tile_id ];
        int2   * __restrict__ ix = &part.ix[ offset ];
        float2 * __restrict__ x  = &part.x[ offset ];
        float3 * __restrict__ u  = &part.u[ offset ];

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = tile_idx.x * nx.x;
        const int shifty = tile_idx.y * nx.y;

        auto sg    = it.get_sub_group();

        for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
            for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                float2 const pos = make_float2(
                    dpcx * ( i0 + 0.5 ) - 0.5,
                    dpcy * ( i1 + 0.5 ) - 0.5
                );
                for( int idx = it.get_local_id(0); idx < vol; idx += it.get_local_range(0)) {
                    int2 const cell = make_int2(
                        idx % row + ri0,
                        idx / row + rj0
                    );

                    float t;
                    if ( dir == coord::x ) t = (shiftx + cell.x) + (pos.x + 0.5);
                    if ( dir == coord::y ) t = (shifty + cell.y) + (pos.y + 0.5);
                    
                    int inj = (t >= start) && (t<finish );

                    int off = device::group::exscan_add( it, tmp, inj );
                    
                    if (inj) {
                        const int k = *np_local + off;
                        ix[ k ] = cell;
                        x[ k ] = pos;
                        u[ k ] = make_float3(0,0,0);
                    }

                    inj = device::subgroup::reduce_add( sg, inj );
                    if ( sg.get_local_id() == 0 ) {
                        device::local::atomicAdd( np_local, inj );
                    }
                    it.barrier();
                }
            }
        }

        if ( it.get_local_id(0) == 0 ) {
            part.np[ tile_id ] = *np_local;
        }
    }
}

void Density::Slab::inject( Particles & particles,
    uint2 const ppc,float2 const dx, float2 const ref, bnd<unsigned int> range ) const
{
    ParticleData part = particles;

    // 8×1 work items per group
    sycl::range<2> local{ 8, 1 };

    // ntiles.x × ntiles.y groups
    sycl::range<2> global{ part.ntiles.x, part.ntiles.y };

    const int max_num_sub_groups = particles.queue.get_device().get_info<sycl::info::device::max_num_sub_groups>();

    particles.queue.submit([&](sycl::handler &h) {

        /// @brief [shared] Local number of particles
        auto np_local = sycl::local_accessor< int, 1 > ( 1, h );

        /// @brief [shared] Temporary memory for exscan calculations
        auto tmp = sycl::local_accessor< int, 1 > ( max_num_sub_groups, h );

        float slab_begin, slab_end;

        switch( dir ) {
        case( coord::x ):
            slab_begin = (begin - ref.x)/ dx.x;
            slab_end = (end - ref.x)/ dx.x;
            h.parallel_for( 
                sycl::nd_range{ global * local, local },
                [=](sycl::nd_item<2> it) {
                inject_slab_kernel <coord::x> ( it, range, slab_begin, slab_end, ppc, part, &np_local[0], &tmp[0] );
            });
            break;
        case( coord::y ):
            slab_begin = (begin - ref.y)/ dx.y;
            slab_end = (end - ref.y)/ dx.y;
            h.parallel_for( 
                sycl::nd_range{ global * local, local },
                [=](sycl::nd_item<2> it) {
                inject_slab_kernel <coord::y> ( it, range, slab_begin, slab_end, ppc, part, &np_local[0], &tmp[0] );
            });
            break;
        }
    });
    particles.queue.wait();

}

template < coord::cart dir >
void np_inject_slab_kernel( 
    sycl::nd_item<2> it,
    bnd<unsigned int> range,
    const float start, const float finish, uint2 ppc,
    ParticleData const part, int * np_local, int * np )
{
    // Tile ID
    const int2 tile_idx = make_int2( it.get_group(0), it.get_group(1));
    const int tile_id = tile_idx.y * part.ntiles.x + tile_idx.x;

    // Store number of particles before injection
    *np_local = 0;
    it.barrier();

    // Find injection range in tile coordinates
    const int2 nx = make_int2( part.nx.x, part.nx.y );
    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;
    
    int inj_np = 0;

    // If range overlaps with tile
    if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
        ( rj0 < nx.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nx.x ) ri1 = nx.x-1;
        if (rj1 >= nx.y ) rj1 = nx.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = tile_idx.x * nx.x;
        const int shifty = tile_idx.y * nx.y;

        for( int idx = it.get_local_id(0); idx < vol; idx += it.get_local_range(0) ) {
            int2 const cell = make_int2(
                idx % row + ri0,
                idx / row + rj0
            );
            for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
                for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                    float2 const pos = make_float2(
                        dpcx * ( i0 + 0.5 ) - 0.5,
                        dpcy * ( i1 + 0.5 ) - 0.5
                    );
                    float t;
                    if ( dir == coord::x ) t = (shiftx + cell.x) + (pos.x + 0.5);
                    if ( dir == coord::y ) t = (shifty + cell.y) + (pos.y + 0.5);
                    
                    int inj = (t >= start) && (t<finish );
                    inj_np += inj;
                }
            }
        }
    }

    auto sg = it.get_sub_group();
    inj_np = device::subgroup::reduce_add( sg, inj_np );
    if ( sg.get_local_id() == 0 ) {
        device::local::atomicAdd( np_local, inj_np );
    } 

    it.barrier();

    if ( it.get_local_id(0) == 0 ) {
        np[ tile_id ] = *np_local;
    }
}

void Density::Slab::np_inject( Particles & particles, 
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range,
    int * np ) const
{
    ParticleData part = particles;

    // 8×1 work items per group
    sycl::range<2> local{ 8, 1 };

    // ntiles.x × ntiles.y groups
    sycl::range<2> global{ part.ntiles.x, part.ntiles.y };

    particles.queue.submit([&](sycl::handler &h) {

        /// @brief [shared] Local number of particles
        auto np_local = sycl::local_accessor< int, 1 > ( 1, h );

        float slab_begin, slab_end;

        switch( dir ) {
        case( coord::x ):
            slab_begin = (begin - ref.x)/ dx.x;
            slab_end = (end - ref.x)/ dx.x;
            h.parallel_for( 
                sycl::nd_range{ global * local, local },
                [=](sycl::nd_item<2> it) {
                np_inject_slab_kernel <coord::x> ( it, range, slab_begin, slab_end, ppc, part, &np_local[0], np );
            });
            break;
        case( coord::y ):
            slab_begin = (begin - ref.y)/ dx.y;
            slab_end = (end - ref.y)/ dx.y;
            h.parallel_for( 
                sycl::nd_range{ global * local, local },
                [=](sycl::nd_item<2> it) {
                np_inject_slab_kernel <coord::y> ( it, range, slab_begin, slab_end, ppc, part, &np_local[0], np );
            });
            break;
        }
    });
    particles.queue.wait();
}


inline void inject_sphere_kernel( 
    sycl::nd_item<2> it,
    bnd<unsigned int> range,
    float2 center, float radius, float2 dx, uint2 ppc,
    ParticleData const part, int * np_local, int * tmp )
{
    // Tile ID
    const int2 tile_idx = make_int2( it.get_group(0), it.get_group(1));
    const int tile_id = tile_idx.y * part.ntiles.x + tile_idx.x;

    // Store number of particles before injection
    *np_local = part.np[ tile_id ];

    it.barrier();

    // Find injection range in tile coordinates
    const int2 nx = make_int2( part.nx.x, part.nx.y );
    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;

    // If range overlaps with tile
    if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
        ( rj0 < nx.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nx.x ) ri1 = nx.x-1;
        if (rj1 >= nx.y ) rj1 = nx.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        const int offset =  part.offset[ tile_id ];
        int2   * __restrict__ ix = &part.ix[ offset ];
        float2 * __restrict__ x  = &part.x[ offset ];
        float3 * __restrict__ u  = &part.u[ offset ];

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = tile_idx.x * nx.x;
        const int shifty = tile_idx.y * nx.y;
        const float r2 = radius*radius;

        auto sg    = it.get_sub_group();

        for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
            for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                float2 const pos = make_float2(
                    dpcx * ( i0 + 0.5 ) - 0.5,
                    dpcy * ( i1 + 0.5 ) - 0.5
                );
                for( int idx = it.get_local_id(0); idx < vol; idx += it.get_local_range(0) ) {
                    int2 const cell = make_int2( 
                        idx % row + ri0,
                        idx / row + rj0
                    );
                    float gx = ((shiftx + cell.x) + (pos.x+0.5)) * dx.x;
                    float gy = ((shifty + cell.y) + (pos.y+0.5)) * dx.y;
                    
                    int inj = ((gx - center.x)*(gx - center.x) + (gy - center.y)*(gy - center.y)) < r2;
                    int off = device::group::exscan_add( it, tmp, inj );

                    if ( inj ) {
                        const int k = *np_local + off;
                        ix[ k ] = cell;
                        x[ k ] = pos;
                        u[ k ] = make_float3(0,0,0);
                    }
                    
                    inj = device::subgroup::reduce_add( sg, inj );
                    if ( sg.get_local_id() == 0 ) {
                        device::local::atomicAdd( np_local, inj );
                    }
                    it.barrier();
                }
            }
        }

        if ( it.get_local_id(0) == 0 ) {
            part.np[ tile_id ] = *np_local;
        }
    }
}

void Density::Sphere::inject( Particles & particles,
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const
{

    float2 sphere_center = center;
    sphere_center.x -= ref.x;
    sphere_center.y -= ref.y;

    ParticleData part = particles;

    // 8×1 work items per group
    sycl::range<2> local{ 8, 1 };

    // ntiles.x × ntiles.y groups
    sycl::range<2> global{ part.ntiles.x, part.ntiles.y };

    particles.queue.submit([&](sycl::handler &h) {

        /// @brief [shared] Local number of particles
        auto np_local = sycl::local_accessor< int, 1 > ( 1, h );

        const int max_num_sub_groups = particles.queue.get_device().get_info<sycl::info::device::max_num_sub_groups>();

        /// @brief [shared] Temporary memory for exscan calculations
        auto tmp = sycl::local_accessor< int, 1 > ( max_num_sub_groups, h );

        auto radius = this -> radius;

        h.parallel_for( 
            sycl::nd_range{ global * local, local },
            [=](sycl::nd_item<2> it) {
            inject_sphere_kernel( it, range, sphere_center, radius, dx, ppc, part, &np_local[0], &tmp[0] );
        });
    });
    particles.queue.wait();
}


void np_inject_sphere_kernel(
    sycl::nd_item<2> it,
    bnd<unsigned int> range,
    float2 center, float radius, float2 dx, uint2 ppc,
    ParticleData const part, int * np_local, int * np )
{
    // Tile ID
    const int2 tile_idx = make_int2( it.get_group(0), it.get_group(1));
    const int tile_id = tile_idx.y * part.ntiles.x + tile_idx.x;

    // Store number of particles before injection
    *np_local = 0;
    it.barrier();

    // Find injection range in tile coordinates
    const int2 nx = make_int2( part.nx.x, part.nx.y );
    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;
    
    int inj_np = 0;

    // If range overlaps with tile
    if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
        ( rj0 < nx.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nx.x ) ri1 = nx.x-1;
        if (rj1 >= nx.y ) rj1 = nx.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = tile_idx.x * nx.x;
        const int shifty = tile_idx.y * nx.y;
        const float r2 = radius*radius;

        for( int idx = it.get_local_id(0); idx < vol; idx += it.get_local_range(0) ) {
            const int2 cell = make_int2( 
                idx % row + ri0,
                idx / row + rj0
            );
            for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
                for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                    float2 const pos = make_float2(
                        dpcx * ( i0 + 0.5 ) - 0.5,
                        dpcy * ( i1 + 0.5 ) - 0.5
                    );
                    float gx = ((shiftx + cell.x) + (pos.x+0.5)) * dx.x;
                    float gy = ((shifty + cell.y) + (pos.y+0.5)) * dx.y;
                    
                    int inj = (gx - center.x)*(gx - center.x) + (gy - center.y)*(gy - center.y) < r2;
                    inj_np += inj;
                }
            }
        }
    }

    auto sg = it.get_sub_group();
    inj_np = device::subgroup::reduce_add( sg, inj_np );
    if ( sg.get_local_id() == 0 ) {
        device::local::atomicAdd( np_local, inj_np );
    } 

    it.barrier();

    if ( it.get_local_id(0) == 0 ) {
        np[ tile_id ] = *np_local;
    }


}

void Density::Sphere::np_inject( Particles & particles, 
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range,
    int * np ) const
{
    float2 sphere_center = center;
    sphere_center.x -= ref.x;
    sphere_center.y -= ref.y;

    ParticleData part = particles;
    
    // 8×1 work items per group
    sycl::range<2> local{ 8, 1 };

    // ntiles.x × ntiles.y groups
    sycl::range<2> global{ part.ntiles.x, part.ntiles.y };

    particles.queue.submit([&](sycl::handler &h) {

        /// @brief [shared] Local number of particles
        auto np_local = sycl::local_accessor< int, 1 > ( 1, h );

        auto radius = this -> radius;

        h.parallel_for( 
            sycl::nd_range{ global * local, local },
            [=](sycl::nd_item<2> it) {
            np_inject_sphere_kernel ( it, range, sphere_center, radius, dx, ppc, part, &np_local[0], np );
        });
    });
    particles.queue.wait();
}
