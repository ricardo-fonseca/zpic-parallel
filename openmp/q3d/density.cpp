#include "density.h"

/**
 * @brief kernel for injecting a uniform plasma density 
 * 
 * @param tile_idx      Tile id
 * @param range         Cell range (global) to inject particles in 
 * @param ppc           Number of particles per cell (z,r,θ)
 * @param dr            Radial cell size
 * @param part          Particle data
 */
inline void inject_uniform_kernel( 
    uint2 const tile_idx, 
    bnd<unsigned int> range,
    uint3 const ppc,
    double const dr,
    ParticleData const part )
{

    const uint2 ntiles = part.ntiles;
    const uint2 nx     = part.nx;
    
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

        const int offset =  part.offset[ tid ];

        auto * __restrict__ const ix = &part.ix[ offset ];
        auto * __restrict__ const x  = &part.x[ offset ];
        auto * __restrict__ const u  = &part.u[ offset ];
        auto * __restrict__ const q  = &part.q[ offset ];
        auto * __restrict__ const θ  = &part.θ[ offset ];

        double dpcz = 1.0 / ppc.x;
        double dpcr = 1.0 / ppc.y;

        /// @brief radial position shift for this tile
        const int shiftr = tile_idx.y * nx.y;

        /// @brief angular positions of particles
        float2 posθ[ ppc.z ];
        posθ[0] = { 1, 0 };
        if ( ppc.z > 1 ) {
            const float Δθ = ( 2 * M_PI ) / ppc.z;
            posθ[1] = { cos( Δθ ), sin( Δθ ) };
            // Use recurrence formulas for remaining angles
            for( unsigned i = 2; i < ppc.z; i++ ) {
                posθ[i] = { 
                    2 * posθ[i-1].x * posθ[1].x - posθ[i-2].y,
                    2 * posθ[i-1].y * posθ[1].x - posθ[i-2].x
                };
            }
        }

        // Check if axial cell is part of injection domain
        int axis = ( rj0 == 0 && shiftr == 0 );
        if ( axis ) rj0 = 1;

        /// @brief grid volume not including axial cell
        int const grid_vol = (rj1 - rj0 + 1 ) * row;

        // Inject particles outside axial cell
        for( unsigned iθ = 0; iθ < ppc.z; iθ++ ) {
            for( unsigned ir = 0; ir < ppc.y; ir++ ) {
                for( unsigned iz = 0; iz < ppc.x; iz++) {
                    float2 const pos = make_float2(
                        dpcz * ( iz + 0.5 ) - 0.5,
                        dpcr * ( ir + 0.5 ) - 0.5
                    );

                    int ppc_idx = ( iθ * ppc.y + ir ) * ppc.x + iz;

                    // On GPU each thread takes 1 cell
                    for( int grid_idx = 0; grid_idx < grid_vol; grid_idx ++ ) {
                        int2 const cell = make_int2( 
                            grid_idx % row + ri0,
                            grid_idx / row + rj0
                        );

                        double r = ( shiftr + cell.y ) + double( pos.y );
                        float qnorm = r * dr / ( ppc.z * ppc.y * ppc.x );

                        int part_idx = np + grid_vol * ppc_idx + grid_idx;

                        ix[ part_idx ] = cell;
                        x [ part_idx ] = pos;
                        u [ part_idx ] = make_float3(0,0,0);
                        q [ part_idx ] = qnorm;
                        θ [ part_idx ] = posθ[ iθ ];
                    }
                }
            }
        }

        ///@brief particles already injected in tile
        int tile_np = grid_vol * ppc.z * ppc.y * ppc.x ;

        // Axial cells get special treatment
        if ( axis ) {
            
            // Additional charge correction for particles on axial cell
            // see notes
            double α, roff;
            if ( ppc.y % 2 == 0) {
                α = ( 2 * ( ppc.y*ppc.y - 1.) ) / ( 3 * ( ppc.y * ppc.y ) );
                roff = 0.5;
            } else {
                α    = (2. / 3.);
                roff = 1;
            }

            for( unsigned iθ = 0; iθ < ppc.z; iθ++ ) {
                for( unsigned ir = 0; ir < ppc.y/2; ir++ ) {
                    for( unsigned iz = 0; iz < ppc.x; iz++) {
                        float2 const pos = make_float2(
                            dpcz * ( iz + 0.5 ) - 0.5,
                            dpcr * ( ir + roff )
                        );

                        int ppc_idx = ( iθ * (ppc.y/2) + ir ) * ppc.x + iz;

                        // On GPU each thread takes 1 cell
                        for( int grid_idx = 0; grid_idx < row; grid_idx ++ ) {
                            int2 const cell = make_int2( grid_idx, 0 );

                            int part_idx = np + tile_np + row * ppc_idx + grid_idx;

                            float qnorm = α * pos.y * dr / ( ppc.z * ppc.y * ppc.x );

                            ix[ part_idx ] = cell;
                            x [ part_idx ] = pos;
                            u [ part_idx ] = make_float3(0,0,0);
                            q [ part_idx ] = qnorm;
                            θ [ part_idx ] = posθ[ iθ ];
                        }
                    }
                }
            }
            tile_np += row * ppc.z * ( ppc.y / 2 ) * ppc.x;
        }

        // Update global number of particles in tile
        { // only one thread per tile does this
            part.np[ tid ] = np + tile_np;
        }
    }
}

/**
 * @brief Injects a uniform density profile
 * 
 * @param part      Particle data object
 * @param ppc       Number of particles per cell (z,r,θ)
 * @param dx        Cell size (z,r)
 * @param ref       Position of local grid on global simulation box in simulation
 *                  units (z,r)
 * @param range     Cell range in which to inject
 */
void Density::Uniform::inject( Particles & part, 
    uint3 const ppc, float2 const dx, float2 const ref,
    bnd<unsigned int> range ) const
{
    auto ntiles = part.ntiles;

    #pragma omp parallel for
    for( unsigned tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {
        auto tx = tid % ntiles.x;
        auto ty = tid / ntiles.x;
        const uint2 tile_idx = make_uint2( tx, ty );
        inject_uniform_kernel( tile_idx, range, ppc, dx.y, part );
    }
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
    uint3 const ppc,
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

    const auto ppc_zθ = ppc.x * ppc.z;
    const auto ppc_r  = ppc.y;

    // If range overlaps with tile
    if (( ri0 < nxx ) && ( ri1 >= 0 ) &&
        ( rj0 < nxy ) && ( rj1 >= 0 )) {
        
        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nxx ) ri1 = nxx-1;
        if (rj1 >= nxy ) rj1 = nxy-1;

        int const row = (ri1-ri0+1);

        if ( tile_idx.y * nx.y + rj0 > 0 ) {
            // Tile does not include axial boundary
            _np = (rj1-rj0+1) * row * ppc_zθ * ppc_r;
        } else {
            // Tile includes axial boundary
            // Since no particles will be injected for r <= 0, we only inject
            // half the particles in the axial cell
            _np = row * ppc_zθ * ( (rj1-rj0) * ppc_r + ppc_r / 2 );
        }
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
    uint3 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range,
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

#if 0

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
    const float step, const uint3 ppc,
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

        const int shiftx = tile_idx.x * nx.x;
        const int shifty = tile_idx.y * nx.y;

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
    uint3 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const
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
    float step, const uint3 ppc,
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

        const int shiftx = tile_idx.x * nx.x;
        const int shifty = tile_idx.y * nx.y;

        // Don't allow injection for r < 0
        if constexpr ( dir == coord::y ) { step = ( step >= 0 ) ? step : 0; }

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
                    if constexpr ( dir == coord::y ) t = (shifty + cell.y) + pos.y;
                    
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
    uint3 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range,
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

        const int shiftx = tile_idx.x * nx.x;
        const int shifty = tile_idx.y * nx.y;

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
    uint3 const ppc,float2 const dx, float2 const ref,
    bnd<unsigned int> range ) const
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
    const float start, const float finish, uint3 ppc,
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

        const int shiftx = tile_idx.x * nx.x;
        const int shifty = tile_idx.y * nx.y;

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
    uint3 const ppc, float2 const dx, float2 const ref,
    bnd<unsigned int> range,
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

        const int shiftx = tile_idx.x * nx.x;
        const int shifty = tile_idx.y * nx.y;
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
    uint3 const ppc, float2 const dx, float2 const ref,
    bnd<unsigned int> range ) const
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
    float2 center, float radius, float2 dx, uint3 ppc,
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

        const int shiftx = tile_idx.x * nx.x;
        const int shifty = tile_idx.y * nx.y;
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
    uint3 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range,
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

#endif