#include "density.h"

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
    uint2 const tile_idx, 
    bnd<unsigned int> range,
    uint2 const ppc,
    ParticleData const part )
{
    const uint2 ntiles  = part.ntiles;
    const int2 nx = make_int2( part.nx.x, part.nx.y );
    
    // Tile ID
    int const tid = tile_idx.y * ntiles.x + tile_idx.x;

    // Store number of particles before injection
    const int np = part.np[ tid ];

    // sync;

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

        const int offset =  part.offset[ tid ];
        int2   * __restrict__ const ix = &part.ix[ offset ];
        float2 * __restrict__ const x  = &part.x[ offset ];
        float3 * __restrict__ const u  = &part.u[ offset ];

        const int np_cell = ppc.x * ppc.y;

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        // Each thread takes 1 cell
        for( int idx = 0; idx < vol; ++idx ) {
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
        { // only one thread per tile does this
            part.np[ tid ] = np + vol * np_cell ;
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
 * @param tiles     Particle tile information 
 * @param data      Particle data 
 */
inline void inject_uniform_kernel( 
    uint2 const tile_idx, 
    bnd<unsigned int> range,
    uint2 const ppc,
    ParticleData const part )
{

    const uint2 ntiles  = part.ntiles;
    const uint2 nx      = part.nx;
    
    // Tile ID
    int const tid = tile_idx.y * ntiles.x + tile_idx.x;

    // Store number of particles before injection
    const int np = part.np[ tid ];

    // sync;

    // Find injection range in tile coordinates
    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;

    // Use signed integers for the calculations below
    int const nxx = nx.x;
    int const nxy = nx.y;
    
    // If range overlaps with tile
    if (( ri0 < nxx ) && ( ri1 >= 0 ) &&
        ( rj0 < nxy ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nxx ) ri1 = nxx-1;
        if (rj1 >= nxy ) rj1 = nxy-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        const int offset =  part.offset[ tid ];

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
                for( int idx = 0; idx < vol; idx ++ ) {
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
        { // only one thread per tile does this
            part.np[ tid ] = np + vol * np_cell ;
        }
    }
}

/**
 * @brief Injects a uniform density profile
 * 
 * @param part      Particle data object
 * @param ppc       Number of particles per cell
 * @param dx        Cell size
 * @param ref       Position of local grid on global simulation box in simulation
 *                  units
 * @param range     Cell range in which to inject
 */
void Density::Uniform::inject( Particles & part, 
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const
{

#if 0

    // Use only for benchmarking
    std::cout << "(*info*) Injecting uniform density using algorithm mk I\n";
    for( auto ty = 0; ty < part.tiles.ntiles.y; ++ty ) {
        for( auto tx = 0; tx < part.tiles.ntiles.x; ++tx ) {
            const uint2 tile_idx( tx, ty );
            inject_uniform_kernel_mk1(
                tile_idx, range, ppc, 
                part.tiles, part.data
            );
        }
    }

#else



    // const int2 ntiles = make_int2( part.ntiles.x, part.ntiles.y );

    auto ntiles = part.ntiles;

    #pragma omp parallel for
    for( unsigned tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {
        auto tx = tid % ntiles.x;
        auto ty = tid / ntiles.x;
        const uint2 tile_idx = make_uint2( tx, ty );
        inject_uniform_kernel( tile_idx, range, ppc, part );
    }
#endif

}

/**
 * @brief Kernel for counting how many particles will be injected
 * 
 * @note Uses only 1 thread per tile
 * 
 * @param range     Cell range to inject particles in
 * @param ppc       Number of particles per cell
 * @param nx        Number of cells in tile
 * @param d_tiles   Particle tile information
 * @param np        Number of particles to inject (out)
 */
inline void np_inject_uniform_kernel( 
    uint2 const tile_idx, 
    bnd<unsigned int> range,
    uint2 const ppc,
    ParticleData const part,
    int * np )
{

    const uint2 ntiles  = part.ntiles;
    const uint2 nx = part.nx;

    int const tid = tile_idx.y * ntiles.x + tile_idx.x;

        // Find injection range in tile coordinates
    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;

    // Comparing signed and unsigned integers does not work here
    int const nxx = nx.x;
    int const nxy = nx.y;

    int _np;

    // If range overlaps with tile
    if (( ri0 < nxx ) && ( ri1 >= 0 ) &&
        ( rj0 < nxy ) && ( rj1 >= 0 )) {
        
        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nxx ) ri1 = nxx-1;
        if (rj1 >= nxy ) rj1 = nxy-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        _np = vol * ppc.x * ppc.y;

    } else {
        _np = 0;
    }

    np[ tid ] = _np;

    // std::cout << __func__ << " np[" << tid << "] = " << np[tid] << '\n';

}

/**
 * @brief Returns number of particles per tile that a uniform profile would inject
 * 
 * @param part      Particle data object
 * @param ppc       Number of particles per cell
 * @param dx        Cell size
 * @param ref       Position of local grid on global simulation box in simulation
 *                  units
 * @param range     Cell range in which to inject
 * @param np        (out) Number of particles to inject per tile
 */
void Density::Uniform::np_inject( Particles & part, 
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range,
    int * np ) const
{
    const int2 ntiles = make_int2( part.ntiles.x, part.ntiles.y );

    #pragma omp parallel for
    for( auto tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {
        auto tx = tid % ntiles.x;
        auto ty = tid / ntiles.x;
        const auto tile_idx = make_uint2( tx, ty );
        np_inject_uniform_kernel(
            tile_idx, range, ppc,
            part, np
        );
    }
}

/**
 * @brief Kernel for injecting step profile
 * 
 * @tparam dir      Step direction ( coord::x | coord::y )
 * @param tile_idx  Tile index (x,y)
 * @param range     Cell range to inject particles in
 * @param step      Step position (normalized to node grid coordinates)
 * @param ppc       Number of particles per cell
 * @param part      Particle data
 */
template < coord::cart dir >
void inject_step_kernel( 
    uint2 const tile_idx, 
    bnd<unsigned int> range,
    const float step, const uint2 ppc,
    ParticleData const part )
{
    const uint2 ntiles  = part.ntiles;
    const int2 nx = make_int2( part.nx.x, part.nx.y );

    // Tile ID
    int const tid = tile_idx.y * ntiles.x + tile_idx.x;

    // Store number of particles before injection
    int np_tile = part.np[ tid ];

    // sync

    // Find injection range in tile coordinates
    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;
   
    // If range overlaps with tile
    if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
        ( rj0 < nx.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if ( ri0 < 0 ) ri0 = 0;
        if ( rj0 < 0 ) rj0 = 0;
        if ( ri1 >= nx.x ) ri1 = nx.x-1;
        if ( rj1 >= nx.y ) rj1 = nx.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        const int offset =  part.offset[ tid ];
        int2   * __restrict__ ix = &part.ix[ offset ];
        float2 * __restrict__ x  = &part.x[ offset ];
        float3 * __restrict__ u  = &part.u[ offset ];

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = (part.tile_off.x + tile_idx.x) * nx.x;
        const int shifty = (part.tile_off.y + tile_idx.y) * nx.y;

        for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
            for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                float2 const pos = make_float2(
                    dpcx * ( i0 + 0.5 ) - 0.5,
                    dpcy * ( i1 + 0.5 ) - 0.5
                );
                for( int idx = 0; idx < vol; idx++) {
                    int2 const cell = make_int2(
                        idx % row + ri0,
                        idx / row + rj0
                    );

                    float t;
                    if constexpr ( dir == coord::x ) t = (shiftx + cell.x) + (pos.x + 0.5);
                    if constexpr ( dir == coord::y ) t = (shifty + cell.y) + (pos.y + 0.5);

                    int inj = t > step;
                    
                    // int off = device::block_exscan_add( inj );
                    int off = 0; // always 0 with 1 thread

                    if ( inj ) {
                        const int k = np_tile + off;
                        ix[ k ] = cell;
                        x[ k ]  = pos;
                        u[ k ]  = make_float3(0,0,0);
                    }

                    // Not needed with 1 thread / tile
                    // inj = device::warp_reduce_add( inj );
                    
                    {   // only 1 thread / tile does this
                        // atomicAdd( &_np, inj );
                        np_tile += inj;
                    }
                }
            }
        }

        // sync;

        {   // Only 1 thread per tile does this
            part.np[ tid ] = np_tile;
        }
    }
}

/**
 * @brief Injects a step density profile
 * 
 * @param part      Particle data object
 * @param ppc       Number of particles per cell
 * @param dx        Cell size
 * @param ref       Position of local grid on global simulation box in simulation
 *                  units
 * @param range     Cell range in which to inject
 */
void Density::Step::inject( Particles & part,
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const
{    
    /// @brief Step position (normalized to node grid coordinates)
    float step_pos;

    switch( dir ) {
    case( coord::x ):
        step_pos = (pos - ref.x) / dx.x;
        for( unsigned ty = 0; ty < part.ntiles.y; ++ty ) {
            for( unsigned tx = 0; tx < part.ntiles.x; ++tx ) {
                const auto tile_idx = make_uint2( tx, ty );
                inject_step_kernel <coord::x> (
                    tile_idx, range, step_pos, ppc, 
                    part );
            }
        }
        break;
    case( coord::y ):
        step_pos = (pos - ref.y) / dx.y;
        for( unsigned ty = 0; ty < part.ntiles.y; ++ty ) {
            for( unsigned tx = 0; tx < part.ntiles.y; ++tx ) {
                const auto tile_idx = make_uint2( tx, ty );
                inject_step_kernel <coord::y> (
                    tile_idx, range, step_pos, ppc,
                    part );
            }
        }
        break;
    break;
    }
}

/**
 * @brief Kernel for counting how many particles will be injected by a step
 *        profile
 * 
 * @note Uses only 1 thread per tile
 * 
 * @tparam dir          Step direction ( coord::x | coord::y )
 * @param tile_idx      Tile index (x,y)
 * @param range         Cell range to inject particles in
 * @param step          Step position (normalized to node grid coordinates)
 * @param ppc           Number of particles per cell
 * @param part          Particle data
 * @param np            (out) Number of particles per tile to inject
 */
template < coord::cart dir >
void np_inject_step_kernel( 
    uint2 const tile_idx, 
    bnd<unsigned int> range,
    const float step, const uint2 ppc,
    ParticleData const part, int * np )
{
    int const tid = tile_idx.y * part.ntiles.x + tile_idx.x;

    int np_tile; np_tile = 0;
    // sync;

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

        const int shiftx = (part.tile_off.x + tile_idx.x) * nx.x;
        const int shifty = (part.tile_off.y + tile_idx.y) * nx.y;

        for( int idx = 0; idx < vol; idx++) {
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
                    if constexpr ( dir == coord::x ) t = (shiftx + cell.x) + (pos.x + 0.5);
                    if constexpr ( dir == coord::y ) t = (shifty + cell.y) + (pos.y + 0.5);
                    
                    int inj = t > step;
                    inj_np += inj;
                }
            }
        }
    }    

    // Not needed with 1 thread / tile
    // inj_np = device::warp_reduce_add( inj_np );
    {   // only 1 thread / tile does this
        // atomicAdd( &np_tile, inj_np );
        np_tile += inj_np;
    } 

    // sync

    {   // Only 1 thread per tile does this
        np[ tid ] = np_tile;
    }
}

/**
 * @brief Returns number of particles per tile that a step profile would inject
 * 
 * @param part      Particle data object
 * @param ppc       Number of particles per cell
 * @param dx        Cell size
 * @param ref       Position of local grid on global simulation box in simulation
 *                  units
 * @param range     Cell range in which to inject
 * @param np        (out) Number of particles to inject per tile
 */
void Density::Step::np_inject( Particles & part, 
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range,
    int * np ) const
{
    /// @brief Step position (normalized to node grid coordinates)
    float step_pos;

    switch( dir ) {
    case( coord::x ):
        step_pos = (pos - ref.x) / dx.x;
        for( unsigned ty = 0; ty < part.ntiles.y; ++ty ) {
            for( unsigned tx = 0; tx < part.ntiles.x; ++tx ) {
                const auto tile_idx = make_uint2( tx, ty );
                np_inject_step_kernel <coord::x> (
                    tile_idx, range, step_pos, ppc,
                    part, np );
            }
        }
        break;
    case( coord::y ):
        step_pos = (pos - ref.y) / dx.y;
        for( unsigned ty = 0; ty < part.ntiles.y; ++ty ) {
            for( unsigned tx = 0; tx < part.ntiles.y; ++tx ) {
                const auto tile_idx = make_uint2( tx, ty );
                np_inject_step_kernel <coord::y> (
                    tile_idx, range, step_pos, ppc,
                    part, np );
            }
        }
        break;
    break;
    }

}

/**
 * @brief Kernel for injecting slab profile
 * 
 * @tparam dir          Slab direction ( coord::x | coord::y )
 * @param tile_idx      Tile index (x,y)
 * @param range         Cell range to inject particles in
 * @param start         Slab start position (normalized to node grid coordinates)
 * @param finish        Slab end position (normalized to node grid coordinates)
 * @param ppc           Number of particles per cell
 * @param part          Particle data
 */
template < coord::cart dir >
void inject_slab_kernel( 
    uint2 const tile_idx,
    bnd<unsigned int> range,
    const float start, const float finish, uint2 ppc,
    ParticleData const part )
{
    const uint2 ntiles  = part.ntiles;
    const uint2 nx = part.nx;

    // Tile ID
    int const tid = tile_idx.y * ntiles.x + tile_idx.x;

    int _np = part.np[ tid ];

    // Find injection range in tile coordinates
    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;

    // Comparing signed and unsigned integers does not work
    int const nxx = nx.x;
    int const nxy = nx.y;

    // If range overlaps with tile
    if (( ri0 < nxx ) && ( ri1 >= 0 ) &&
        ( rj0 < nxy ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nxx ) ri1 = nxx-1;
        if (rj1 >= nxy ) rj1 = nxy-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        const int offset =  part.offset[ tid ];
        int2   * __restrict__ ix = &part.ix[ offset ];
        float2 * __restrict__ x  = &part.x[ offset ];
        float3 * __restrict__ u  = &part.u[ offset ];

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = (part.tile_off.x + tile_idx.x) * nx.x;
        const int shifty = (part.tile_off.y + tile_idx.y) * nx.y;

        for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
            for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                float2 const pos = make_float2(
                    dpcx * ( i0 + 0.5 ) - 0.5,
                    dpcy * ( i1 + 0.5 ) - 0.5
                );


                for( int idx = 0; idx < vol; idx++ ) {
                    int2 const cell = make_int2(
                        idx % row + ri0,
                        idx / row + rj0
                    );

                    float t;
                    if constexpr ( dir == coord::x ) t = (shiftx + cell.x) + (pos.x + 0.5);
                    if constexpr ( dir == coord::y ) t = (shifty + cell.y) + (pos.y + 0.5);
                    
                    int inj = (t >= start) && (t<finish );

                    // int off = device::block_exscan_add( inj );
                    int off = 0; // always 0 with 1 thread
                    
                    if (inj) {
                        const int k = _np + off;
                        ix[ k ] = cell;
                        x[ k ] = pos;
                        u[ k ] = make_float3(0,0,0);
                    }

                    // Not needed with 1 thread / tile
                    // inj = device::warp_reduce_add( inj );
                    
                    {   // only 1 thread / tile does this
                        // atomicAdd( &_np, inj );
                        _np += inj;
                    }
                }
            }
        }

        // sync;

        {   // Only 1 thread per tile does this
            part.np[ tid ] = _np;
        }
    }
}

/**
 * @brief Injects a slab density profile
 * 
 * @param part      Particle data object
 * @param ppc       Number of particles per cell
 * @param dx        Cell size
 * @param ref       Position of local grid on global simulation box in simulation
 *                  units
 * @param range     Cell range in which to inject
 */
void Density::Slab::inject( Particles & part,
    uint2 const ppc,float2 const dx, float2 const ref, bnd<unsigned int> range ) const
{
    /// @brief Slab start position (normalized to node grid coordinates) 
    float slab_begin;
    /// @brief Slab end position (normalized to node grid coordinates) 
    float slab_end;

    switch( dir ) {
    case( coord::x ):
        slab_begin = (begin - ref.x)/ dx.x;
        slab_end   = (end - ref.x)/ dx.x;
        for( unsigned ty = 0; ty < part.ntiles.y; ++ty ) {
            for( unsigned tx = 0; tx < part.ntiles.x; ++tx ) {
                const auto tile_idx = make_uint2( tx, ty );
                inject_slab_kernel < coord::x > (
                    tile_idx, range, slab_begin, slab_end, ppc,
                    part );
            }
        }
        break;
    case( coord::y ):
        slab_begin = (begin - ref.y)/ dx.y;
        slab_end   = (end - ref.y)/ dx.y;
        for( unsigned ty = 0; ty < part.ntiles.y; ++ty ) {
            for( unsigned tx = 0; tx < part.ntiles.x; ++tx ) {
                const auto tile_idx = make_uint2( tx, ty );
                inject_slab_kernel < coord::y > (
                    tile_idx, range, slab_begin, slab_end, ppc,
                    part );
            }
        }
        break;
    }

}

/**
 * @brief Kernel for counting how many particles will be injected by slab profie
 * 
 * @tparam dir          Slab direction ( coord::x | coord::y )
 * @param tile_idx      Tile index (x,y)
 * @param range         Cell range to inject particles in
 * @param start         Slab start position (normalized to node grid coordinates)
 * @param finish        Slab end position (normalized to node grid coordinates)
 * @param ppc           Number of particles per cell
 * @param part          Particle data
 * @param np            (out) Number of particles to inject per tile
 */
template < coord::cart dir >
void np_inject_slab_kernel( 
    uint2 const tile_idx,
    bnd<unsigned int> range,
    const float start, const float finish, uint2 ppc,
    ParticleData const part, int * np )
{
    const uint2 ntiles  = part.ntiles;
    const uint2 nx = part.nx;

    int const tid = tile_idx.y * ntiles.x + tile_idx.x;

    int _np;
    _np = 0;
    
    // sync

    // Find injection range in tile coordinates
    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;

    // Comparing signed and unsigned integers does not work
    int const nxx = nx.x;
    int const nxy = nx.y;

    unsigned int inj_np = 0;

    // If range overlaps with tile
    if (( ri0 < nxx ) && ( ri1 >= 0 ) &&
        ( rj0 < nxy ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nxx ) ri1 = nxx-1;
        if (rj1 >= nxy ) rj1 = nxy-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = (part.tile_off.x + tile_idx.x) * nx.x;
        const int shifty = (part.tile_off.y + tile_idx.y) * nx.y;

        for( int idx = 0; idx < vol; idx++) {
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
                    if constexpr ( dir == coord::x ) t = (shiftx + cell.x) + (pos.x + 0.5);
                    if constexpr ( dir == coord::y ) t = (shifty + cell.y) + (pos.y + 0.5);
                    
                    int inj = (t >= start) && (t<finish );
                    inj_np += inj;
                }
            }
        }
    }

    // Not needed with 1 thread / tile
    // inj_np = device::warp_reduce_add( inj_np );
    {   // only 1 thread / tile does this
        // atomicAdd( &_np, inj_np );
        _np += inj_np;
    } 

    // sync


    {   // Only 1 thread per tile does this
        np[ tid ] = _np;
    }

}

/**
 * Returns number of particles per tile that a slab profile would inject
 * 
 * @param part      Particle data object
 * @param ppc       Number of particles per cell
 * @param dx        Cell size
 * @param ref       Position of local grid on global simulation box in simulation
 *                  units
 * @param range     Cell range in which to inject
 * @param np        (out) Number of particles to inject per tile
 */
void Density::Slab::np_inject( Particles & part, 
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range,
    int * np ) const
{
    /// @brief Slab start position (normalized to node grid coordinates) 
    float slab_begin;
    /// @brief Slab end position (normalized to node grid coordinates) 
    float slab_end;

    switch( dir ) {
    case( coord::x ):
        slab_begin = (begin - ref.x)/ dx.x;
        slab_end   = (end - ref.x)/ dx.x;
        for( unsigned ty = 0; ty < part.ntiles.y; ++ty ) {
            for( unsigned tx = 0; tx < part.ntiles.x; ++tx ) {
                const auto tile_idx = make_uint2( tx, ty );
                np_inject_slab_kernel < coord::x > (
                    tile_idx,
                    range, slab_begin, slab_end, ppc,
                    part, np );
            }
        }
        break;
    case( coord::y ):
        slab_begin = (begin - ref.y)/ dx.y;
        slab_end   = (end - ref.y)/ dx.y;
        for( unsigned ty = 0; ty < part.ntiles.y; ++ty ) {
            for( unsigned tx = 0; tx < part.ntiles.x; ++tx ) {
                const auto tile_idx = make_uint2( tx, ty );
                np_inject_slab_kernel < coord::y > (
                    tile_idx, range, slab_begin, slab_end, ppc,
                    part, np );
            }
        }
        break;
    }
}

/**
 * @brief Kernel for injecting sphere profile
 * 
 * @param tile_idx      Tile index (x,y)
 * @param range         Cell range to inject particles in
 * @param center        Sphere center in simulation units
 * @param radius        Sphere radius in simulation units
 * @param dx            Cell size
 * @param ppc           Number of particles per cell
 * @param part          Particle data
 */
inline void inject_sphere_kernel( 
    uint2 const tile_idx,
    bnd<unsigned int> range,
    float2 center, float radius, float2 dx, uint2 ppc,
    ParticleData const part )
{

    // Tile ID
    const int tile_id = tile_idx.y * part.ntiles.x + tile_idx.x;

    int np_local; np_local = part.np[ tile_id ];

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

        const int shiftx = (part.tile_off.x + tile_idx.x) * nx.x;
        const int shifty = (part.tile_off.y + tile_idx.y) * nx.y;
        const float r2 = radius*radius;

        for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
            for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                float2 const pos = make_float2(
                    dpcx * ( i0 + 0.5 ) - 0.5,
                    dpcy * ( i1 + 0.5 ) - 0.5
                );
                for( int idx = 0; idx < vol; idx++) {
                    int2 const cell = make_int2( 
                        idx % row + ri0,
                        idx / row + rj0
                    );
                    float gx = ((shiftx + cell.x) + (pos.x+0.5)) * dx.x;
                    float gy = ((shifty + cell.y) + (pos.y+0.5)) * dx.y;
                    
                    int inj = ((gx - center.x)*(gx - center.x) + (gy - center.y)*(gy - center.y)) < r2;
                    // int off = device::block_exscan_add( inj );
                    int off = 0; // always 0 with 1 thread

                    if ( inj ) {
                        const int k = np_local + off;
                        ix[ k ] = cell;
                        x[ k ] = pos;
                        u[ k ] = make_float3(0,0,0);
                    }
                    
                    // inj = warp::reduce_add( inj );
                    // if ( warp::thread_rank() == 0 ) {
                    //     block::atomic_fetch_add( &np_local, inj );
                    // }
                    // block_sync();

                    np_local += inj;
                }
            }
        }

        // if ( block_thread_rank() == 0 ) {
        //    part.np[ tile_id ] = np_local;
        // }
        part.np[ tile_id ] = np_local;
    }
}

/**
 * @brief Injects a sphere density profile
 * 
 * @param part      Particle data object
 * @param ppc       Number of particles per cell
 * @param dx        Cell size
 * @param ref       Position of local grid on global simulation box in simulation
 *                  units
 * @param range     Cell range in which to inject
 */
void Density::Sphere::inject( Particles & part,
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const
{

    float2 sphere_center = center;
    sphere_center.x -= ref.x;
    sphere_center.y -= ref.y;

    for( unsigned ty = 0; ty < part.ntiles.y; ty++ ) {
        for( unsigned tx = 0; tx < part.ntiles.x; tx++ ) {
            const auto tile_idx = make_uint2( tx, ty );
            inject_sphere_kernel (
                tile_idx, range, sphere_center, radius, dx, ppc,
                part );
        }
    }
}

/**
 * @brief Kernel for counting how many particles will be injected by a sphere
 *        profile
 * 
 * @param tile_idx  Tile index (x,y)
 * @param range     Cell range to inject particles in
 * @param center    Sphere center in simulation units
 * @param radius    Sphere radius in simulation units
 * @param dx        Cell size
 * @param ppc       Number of particles per cell
 * @param part      Particle data
 * @param np        (out) Number of particles per tile to inject
 */
void np_inject_sphere_kernel(
    uint2 const tile_idx,
    bnd<unsigned int> range,
    float2 center, float radius, float2 dx, uint2 ppc,
    ParticleData const part, int * np )
{
    // Tile ID
    int const tile_id = tile_idx.y * part.ntiles.x + tile_idx.x;

    int np_local; np_local = 0;
    // sync

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

        const int shiftx = (part.tile_off.x + tile_idx.x) * nx.x;
        const int shifty = (part.tile_off.y + tile_idx.y) * nx.y;
        const float r2 = radius*radius;

        for( int idx = 0; idx < vol; idx++) {
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
                    float gx = ((shiftx + cell.x) + (pos.x+0.5)) * dx.x;
                    float gy = ((shifty + cell.y) + (pos.y+0.5)) * dx.y;
                    
                    int inj = ((gx - center.x)*(gx - center.x) + (gy - center.y)*(gy - center.y)) < r2;
                    inj_np += inj;
                }
            }
        }
    }

    // Not needed with 1 thread / tile
    // inj_np = warp::reduce_add( inj_np );
    // if ( warp::thread_rank() == 0 ) {
    //    block::atomic_fetch_add( &np_local, inj_np );
    //}
    
    np_local += inj_np;

    // block_sync();  
    //if ( block_thread_rank() == 0 ) {
    //    np[ tile_id ] = np_local;
    //}

    // Only 1 thread per tile does this
    np[ tile_id ] = np_local;
}

void Density::Sphere::np_inject( Particles & part, 
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range,
    int * np ) const
{
    float2 sphere_center = center;
    sphere_center.x -= ref.x;
    sphere_center.y -= ref.y;

    for( unsigned ty = 0; ty < part.ntiles.y; ty++ ) {
        for( unsigned tx = 0; tx < part.ntiles.x; tx++ ) {
            const auto tile_idx =  make_uint2( tx, ty );
            np_inject_sphere_kernel (
                tile_idx,
                range, sphere_center, radius, dx, ppc,
                part, np );
        }
    }
}
