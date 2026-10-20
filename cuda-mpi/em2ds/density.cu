#include "core/gpu.cuh"
#include "density.hpp"


namespace kernel {

/**
 * @brief kernel for injecting a uniform plasma density (mk1)
 * 
 * part::particles in the same cell are injected contiguously
 * 
 * @param range     Cell range (global) to inject particles in 
 * @param ppc       Number of particles per cell 
 * @param nx        Number of cells in tile 
 * @param tiles     Particle tile information 
 * @param data      Particle data 
 */
__global__
inline void inject_uniform_mk1( 
    bounds_2d<unsigned int> range,
    uint2 const ppc,
    part::particles_view const part )
{
    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const uint2 ntiles  = part.local_ntiles;
    const int2 tile_dims = make_int2( part.tile_dims.x, part.tile_dims.y );
    
    // Tile ID
    int const tile_id = tile_idx.y * ntiles.x + tile_idx.x;

    // Store number of particles before injection
    const int np = part.tile_np[ tile_id ];

    gpu::block::sync();

    // Find injection range in tile coordinates
    int ri0 = range.x.lower - tile_idx.x * tile_dims.x;
    int ri1 = range.x.upper - tile_idx.x * tile_dims.x;

    int rj0 = range.y.lower - tile_idx.y * tile_dims.y;
    int rj1 = range.y.upper - tile_idx.y * tile_dims.y;
    
    // If range overlaps with tile
    if (( ri0 < tile_dims.x ) && ( ri1 >= 0 ) &&
        ( rj0 < tile_dims.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= tile_dims.x ) ri1 = tile_dims.x-1;
        if (rj1 >= tile_dims.y ) rj1 = tile_dims.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        const int offset =  part.tile_offset[ tile_id ];
        int2   * __restrict__ const ix = &part.ix[ offset ];
        float2 * __restrict__ const x  = &part.x[ offset ];
        float3 * __restrict__ const u  = &part.u[ offset ];

        const int np_cell = ppc.x * ppc.y;

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        // Each thread takes 1 cell
        for( int idx = gpu::block::thread_rank(); idx < vol; idx += gpu::block::num_threads() ) {
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
        if ( gpu::block::thread_rank() == 0 ) { // only one thread per tile does this
            part.tile_np[ tile_id ] = np + vol * np_cell ;
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
__global__
inline void inject_uniform( 
    bounds_2d<unsigned int> range,
    uint2 const ppc,
    part::particles_view const part )
{
    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const uint2 ntiles    = part.local_ntiles;
    const uint2 tile_dims = part.tile_dims;
    
    // Tile ID
    int const tile_id = tile_idx.y * ntiles.x + tile_idx.x;

    // Store number of particles before injection
    const int np = part.tile_np[ tile_id ];

    gpu::block::sync();

    // Find injection range in tile coordinates
    int ri0 = range.x.lower - tile_idx.x * tile_dims.x;
    int ri1 = range.x.upper - tile_idx.x * tile_dims.x;

    int rj0 = range.y.lower - tile_idx.y * tile_dims.y;
    int rj1 = range.y.upper - tile_idx.y * tile_dims.y;

    // Use signed integers for the calculations below
    int const nxx = tile_dims.x;
    int const nxy = tile_dims.y;
    
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

        const int offset =  part.tile_offset[ tile_id ];

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
                for( int idx = gpu::block::thread_rank(); idx < vol; idx += gpu::block::num_threads() ) {
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
        if ( gpu::block::thread_rank() == 0 ) { // only one thread per tile does this
            part.tile_np[ tile_id ] = np + vol * np_cell ;
        }
    }
}

} // namespace kernel

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
void density::uniform::inject( part::particles & part, 
    uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range ) const
{

    auto ntiles = part.get_local_ntiles();
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 1024 );

#if 0
    std::cout << "(*info*) Injecting uniform density using algorithm mk I\n";
    kernel::uniform_mk1 <<< grid, block >>> (
        range, ppc, part.view()
    );
#else
    kernel::inject_uniform <<< grid, block >>> (
        range, ppc, part.view()
    );
#endif

}

namespace kernel {

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
__global__
inline void np_uniform( 
    bounds_2d<unsigned int> range,
    uint2 const ppc,
    part::particles_view const part,
    int * np )
{
    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const uint2 ntiles    = part.local_ntiles;
    const uint2 tile_dims = part.tile_dims;

    int const tile_id = tile_idx.y * ntiles.x + tile_idx.x;

        // Find injection range in tile coordinates
    int ri0 = range.x.lower - tile_idx.x * tile_dims.x;
    int ri1 = range.x.upper - tile_idx.x * tile_dims.x;

    int rj0 = range.y.lower - tile_idx.y * tile_dims.y;
    int rj1 = range.y.upper - tile_idx.y * tile_dims.y;

    // Comparing signed and unsigned integers does not work here
    int const nxx = tile_dims.x;
    int const nxy = tile_dims.y;

    int np_tile;

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

        np_tile = vol * ppc.x * ppc.y;

    } else {
        np_tile = 0;
    }

    np[ tile_id ] = np_tile;
}

} // namespace kernel

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
void density::uniform::np_inject( part::particles & part, 
    uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range,
    int * np ) const
{
    auto ntiles = part.get_local_ntiles();
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 1 ); // Use only 1 thread per tile

    kernel::np_uniform<<< grid, block >>> ( range, ppc, part.view(), np );
}

namespace kernel {
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
__global__
void inject_step( 
    bounds_2d<unsigned int> range,
    const float step, const uint2 ppc,
    part::particles_view const part )
{
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const uint2 ntiles  = part.local_ntiles;

    // Tile ID
    int const tile_id = tile_idx.y * ntiles.x + tile_idx.x;

    // Store number of particles before injection
    __shared__ int np_tile;
    np_tile = part.tile_np[ tile_id ];

    gpu::block::sync();

    // Find injection range in tile coordinates
    const int2 tile_dims = make_int2( part.tile_dims.x, part.tile_dims.y );

    int ri0 = range.x.lower - tile_idx.x * tile_dims.x;
    int ri1 = range.x.upper - tile_idx.x * tile_dims.x;

    int rj0 = range.y.lower - tile_idx.y * tile_dims.y;
    int rj1 = range.y.upper - tile_idx.y * tile_dims.y;
   
    // If range overlaps with tile
    if (( ri0 < tile_dims.x ) && ( ri1 >= 0 ) &&
        ( rj0 < tile_dims.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if ( ri0 < 0 ) ri0 = 0;
        if ( rj0 < 0 ) rj0 = 0;
        if ( ri1 >= tile_dims.x ) ri1 = tile_dims.x-1;
        if ( rj1 >= tile_dims.y ) rj1 = tile_dims.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        const int offset =  part.tile_offset[ tile_id ];
        int2   * __restrict__ ix = &part.ix[ offset ];
        float2 * __restrict__ x  = &part.x[ offset ];
        float3 * __restrict__ u  = &part.u[ offset ];

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = (part.local_tile_start.x + tile_idx.x) * tile_dims.x;
        const int shifty = (part.local_tile_start.y + tile_idx.y) * tile_dims.y;

        for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
            for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                float2 const pos = make_float2(
                    dpcx * ( i0 + 0.5 ) - 0.5,
                    dpcy * ( i1 + 0.5 ) - 0.5
                );
                for( int idx = gpu::block::thread_rank(); idx < vol; idx += gpu::block::num_threads() ) {
                    int2 const cell = make_int2(
                        idx % row + ri0,
                        idx / row + rj0
                    );

                    float t;
                    if constexpr ( dir == coord::x ) t = (shiftx + cell.x) + (pos.x + 0.5);
                    if constexpr ( dir == coord::y ) t = (shifty + cell.y) + (pos.y + 0.5);

                    int inj = t > step;
                    int off = gpu::block::exscan_add( inj );

                    if ( inj ) {
                        const int k = np_tile + off;
                        ix[ k ] = cell;
                        x[ k ]  = pos;
                        u[ k ]  = make_float3(0,0,0);
                    }

                    inj = gpu::warp::reduce_add( inj );
                    if ( gpu::warp::thread_rank() == 0 ) {
                        gpu::block::atomic_fetch_add( &np_tile, inj );
                    }
                    gpu::block::sync();
                }
            }
        }

        if ( gpu::block::thread_rank() == 0 ) {   // Only 1 thread per tile does this
            part.tile_np[ tile_id ] = np_tile;
        }
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
void density::step::inject( part::particles & part,
    uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range ) const
{    
    /// @brief Step position (normalized to node grid coordinates)
    float step_pos;

    auto ntiles = part.get_local_ntiles();
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 1024 );

    switch( dir ) {
    case( coord::x ):
        step_pos = (pos - ref.x) / dx.x;
        kernel::inject_step <coord::x> <<< grid, block >>> ( range, step_pos, ppc, part.view() );
        gpu::device::check();
        break;
    case( coord::y ):
        step_pos = (pos - ref.y) / dx.y;
        kernel::inject_step <coord::y> <<< grid, block >>> ( range, step_pos, ppc, part.view() );
        break;
    break;
    }
}

namespace kernel {

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
__global__
void np_step( 
    bounds_2d<unsigned int> range,
    const float step, const uint2 ppc,
    part::particles_view const part, int * np )
{
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    int const tile_id = tile_idx.y * part.local_ntiles.x + tile_idx.x;

    __shared__ int np_tile;
    np_tile = 0;
    gpu::block::sync();

    // Find injection range in tile coordinates
    const int2 nx = make_int2( part.tile_dims.x, part.tile_dims.y );

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

        const int shiftx = (part.local_tile_start.x + tile_idx.x) * nx.x;
        const int shifty = (part.local_tile_start.y + tile_idx.y) * nx.y;

        for( int idx = gpu::block::thread_rank(); idx < vol; idx += gpu::block::num_threads()) {
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

    inj_np = gpu::warp::reduce_add( inj_np );
    if ( gpu::warp::thread_rank() == 0 ) {
        gpu::block::atomic_fetch_add( &np_tile, inj_np );
    } 

    gpu::block::sync();

    if ( gpu::block::thread_rank() == 0 ) {   // Only 1 thread per tile does this
        np[ tile_id ] = np_tile;
    }
}

} // namespace kernel

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
void density::step::np_inject( part::particles & part, 
    uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range,
    int * np ) const
{

    auto ntiles = part.get_local_ntiles();
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 1024 );

    /// @brief Step position (normalized to node grid coordinates)
    float step_pos;

    switch( dir ) {
    case( coord::x ):
        step_pos = (pos - ref.x) / dx.x;
        kernel::np_step <coord::x> <<< grid, block >>> ( range, step_pos, ppc, part.view(), np );
        break;
    case( coord::y ):
        step_pos = (pos - ref.y) / dx.y;
        kernel::np_step <coord::y> <<< grid, block >>> ( range, step_pos, ppc, part.view(), np );
        break;
    break;
    }
}

namespace kernel {

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
__global__
void inject_slab( 
    bounds_2d<unsigned int> range,
    const float start, const float finish, uint2 ppc,
    part::particles_view const part )
{
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int2 tile_dims = make_int2( part.tile_dims.x, part.tile_dims.y );

    // Tile ID
    int const tile_id = tile_idx.y * part.local_ntiles.x + tile_idx.x;

    __shared__ int np_tile;
    np_tile = part.tile_np[ tile_id ];
    gpu::block::sync();

    // Find injection range in tile coordinates
    int ri0 = range.x.lower - tile_idx.x * tile_dims.x;
    int ri1 = range.x.upper - tile_idx.x * tile_dims.x;

    int rj0 = range.y.lower - tile_idx.y * tile_dims.y;
    int rj1 = range.y.upper - tile_idx.y * tile_dims.y;

    // Comparing signed and unsigned integers does not work

    // If range overlaps with tile
    if (( ri0 < tile_dims.x ) && ( ri1 >= 0 ) &&
        ( rj0 < tile_dims.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= tile_dims.x ) ri1 = tile_dims.x-1;
        if (rj1 >= tile_dims.y ) rj1 = tile_dims.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        const int offset =  part.tile_offset[ tile_id ];
        int2   * __restrict__ ix = &part.ix[ offset ];
        float2 * __restrict__ x  = &part.x[ offset ];
        float3 * __restrict__ u  = &part.u[ offset ];

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = (part.local_tile_start.x + tile_idx.x) * tile_dims.x;
        const int shifty = (part.local_tile_start.y + tile_idx.y) * tile_dims.y;

        for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
            for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                float2 const pos = make_float2(
                    dpcx * ( i0 + 0.5 ) - 0.5,
                    dpcy * ( i1 + 0.5 ) - 0.5
                );


                for( int idx = gpu::block::thread_rank(); idx < vol; idx+= gpu::block::num_threads() ) {
                    int2 const cell = make_int2(
                        idx % row + ri0,
                        idx / row + rj0
                    );

                    float t;
                    if constexpr ( dir == coord::x ) t = (shiftx + cell.x) + (pos.x + 0.5);
                    if constexpr ( dir == coord::y ) t = (shifty + cell.y) + (pos.y + 0.5);
                    
                    int inj = (t >= start) && (t<finish );

                    int off = gpu::block::exscan_add( inj );
                    
                    if (inj) {
                        const int k = np_tile + off;
                        ix[ k ] = cell;
                        x[ k ] = pos;
                        u[ k ] = make_float3(0,0,0);
                    }

                    inj = gpu::warp::reduce_add( inj );
                    if ( gpu::warp::thread_rank() == 0 ) {
                        gpu::block::atomic_fetch_add( & np_tile, inj );
                    }
                    gpu::block::sync();
                }
            }
        }

        if ( gpu::block::thread_rank() == 0 ) {   // Only 1 thread per tile does this
            part.tile_np[ tile_id ] = np_tile;
        }
    }
}

} // namespace kernel

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
void density::slab::inject( part::particles & part,
    uint2 const ppc,float2 const dx, float2 const ref, bounds_2d<unsigned int> range ) const
{

    auto ntiles = part.get_local_ntiles();
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 1024 );

    /// @brief Slab start position (normalized to node grid coordinates) 
    float slab_begin;
    /// @brief Slab end position (normalized to node grid coordinates) 
    float slab_end;

    switch( dir ) {
    case( coord::x ):
        slab_begin = (begin - ref.x)/ dx.x;
        slab_end   = (end - ref.x)/ dx.x;
        kernel::inject_slab <coord::x> <<< grid, block >>> ( range, slab_begin, slab_end, ppc, part.view() );
        break;
    case( coord::y ):
        slab_begin = (begin - ref.y)/ dx.y;
        slab_end   = (end - ref.y)/ dx.y;
        kernel::inject_slab <coord::y> <<< grid, block >>> ( range, slab_begin, slab_end, ppc, part.view() );
        break;
    }

}

namespace kernel {

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
__global__
void np_slab( 
    bounds_2d<unsigned int> range,
    const float start, const float finish, uint2 ppc,
    part::particles_view const part, int * np )
{
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id = tile_idx.y * part.local_ntiles.x + tile_idx.x;
    const int2 tile_dims = make_int2(part.tile_dims.x, part.tile_dims.y);

    __shared__ int np_tile;
    np_tile = 0;
    gpu::block::sync();


    // Find injection range in tile coordinates
    int ri0 = range.x.lower - tile_idx.x * tile_dims.x;
    int ri1 = range.x.upper - tile_idx.x * tile_dims.x;

    int rj0 = range.y.lower - tile_idx.y * tile_dims.y;
    int rj1 = range.y.upper - tile_idx.y * tile_dims.y;

    int inj_np = 0;

    // If range overlaps with tile
    if (( ri0 < tile_dims.x ) && ( ri1 >= 0 ) &&
        ( rj0 < tile_dims.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= tile_dims.x ) ri1 = tile_dims.x-1;
        if (rj1 >= tile_dims.y ) rj1 = tile_dims.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = (part.local_tile_start.x + tile_idx.x) * tile_dims.x;
        const int shifty = (part.local_tile_start.y + tile_idx.y) * tile_dims.y;

        for( int idx = gpu::block::thread_rank(); idx < vol; idx+= gpu::block::num_threads() ) {
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

    inj_np = gpu::warp::reduce_add( inj_np );
    if ( gpu::warp::thread_rank() == 0 ) {
        gpu::block::atomic_fetch_add( & np_tile, inj_np );
    }
    gpu::block::sync();


    if ( gpu::block::thread_rank() == 0 ) {   // Only 1 thread per tile does this
        np[ tile_id ] = np_tile;
    }
}

} // namespace kernel

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
void density::slab::np_inject( part::particles & part, 
    uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range,
    int * np ) const
{

    auto ntiles = part.get_local_ntiles();
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 1024 );

    /// @brief Slab start position (normalized to node grid coordinates) 
    float slab_begin;
    /// @brief Slab end position (normalized to node grid coordinates) 
    float slab_end;

    switch( dir ) {
    case( coord::x ):
        slab_begin = (begin - ref.x)/ dx.x;
        slab_end   = (end - ref.x)/ dx.x;
        kernel::np_slab <coord::x> <<< grid, block >>> ( 
            range, slab_begin, slab_end, ppc, part.view(), np
        );
        break;
    case( coord::y ):
        slab_begin = (begin - ref.y)/ dx.y;
        slab_end   = (end - ref.y)/ dx.y;
        kernel::np_slab <coord::y> <<< grid, block >>> (
            range, slab_begin, slab_end, ppc, part.view(), np
        );
        break;
    }
}

namespace kernel {

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
__global__
inline void inject_sphere( 
    bounds_2d<unsigned int> range,
    float2 center, float radius, float2 dx, uint2 ppc,
    part::particles_view const part )
{

    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id = tile_idx.y * part.local_ntiles.x + tile_idx.x;
    const int2 tile_dims = make_int2(part.tile_dims.x, part.tile_dims.y);


    __shared__ int np_tile;
    np_tile = part.tile_np[ tile_id ];
    gpu::block::sync();

    // Find injection range in tile coordinates
    int ri0 = range.x.lower - tile_idx.x * tile_dims.x;
    int ri1 = range.x.upper - tile_idx.x * tile_dims.x;

    int rj0 = range.y.lower - tile_idx.y * tile_dims.y;
    int rj1 = range.y.upper - tile_idx.y * tile_dims.y;

    // If range overlaps with tile
    if (( ri0 < tile_dims.x ) && ( ri1 >= 0 ) &&
        ( rj0 < tile_dims.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= tile_dims.x ) ri1 = tile_dims.x-1;
        if (rj1 >= tile_dims.y ) rj1 = tile_dims.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        const int offset =  part.tile_offset[ tile_id ];
        int2   * __restrict__ ix = &part.ix[ offset ];
        float2 * __restrict__ x  = &part.x[ offset ];
        float3 * __restrict__ u  = &part.u[ offset ];

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = (part.local_tile_start.x + tile_idx.x) * tile_dims.x;
        const int shifty = (part.local_tile_start.y + tile_idx.y) * tile_dims.y;
        const float r2 = radius*radius;

        for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
            for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                float2 const pos = make_float2(
                    dpcx * ( i0 + 0.5 ) - 0.5,
                    dpcy * ( i1 + 0.5 ) - 0.5
                );
                for( int idx = gpu::block::thread_rank(); idx < vol; idx+= gpu::block::num_threads()) {
                    int2 const cell = make_int2( 
                        idx % row + ri0,
                        idx / row + rj0
                    );
                    float gx = ((shiftx + cell.x) + (pos.x+0.5)) * dx.x;
                    float gy = ((shifty + cell.y) + (pos.y+0.5)) * dx.y;
                    
                    int inj = ((gx - center.x)*(gx - center.x) + (gy - center.y)*(gy - center.y)) < r2;
                    int off = gpu::block::exscan_add( inj );

                    if ( inj ) {
                        const int k = np_tile + off;
                        ix[ k ] = cell;
                        x[ k ] = pos;
                        u[ k ] = make_float3(0,0,0);
                    }
                    
                    inj = gpu::warp::reduce_add( inj );
                    if ( gpu::warp::thread_rank() == 0 ) {
                        gpu::block::atomic_fetch_add( &np_tile, inj );
                    }
                    gpu::block::sync();
                }
            }
        }

        if ( gpu::block::thread_rank() == 0 ) {
            part.tile_np[ tile_id ] = np_tile;
        }
    }
}

} // namespace kernel

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
void density::sphere::inject( part::particles & part,
    uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range ) const
{

    auto ntiles = part.get_local_ntiles();
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 1024 );

    float2 sphere_center = center;
    sphere_center.x -= ref.x;
    sphere_center.y -= ref.y;

    kernel::inject_sphere <<< grid, block >>> ( 
        range, sphere_center, radius, dx, ppc, part.view()
    );
}

namespace kernel {

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
__global__
void np_sphere(
    bounds_2d<unsigned int> range,
    float2 center, float radius, float2 dx, uint2 ppc,
    part::particles_view const part, int * np )
{
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id = tile_idx.y * part.local_ntiles.x + tile_idx.x;
    const int2 tile_dims = make_int2(part.tile_dims.x, part.tile_dims.y);

    __shared__ int np_tile; 
    np_tile = 0;
    gpu::block::sync();

    // Find injection range in tile coordinates
    int ri0 = range.x.lower - tile_idx.x * tile_dims.x;
    int ri1 = range.x.upper - tile_idx.x * tile_dims.x;

    int rj0 = range.y.lower - tile_idx.y * tile_dims.y;
    int rj1 = range.y.upper - tile_idx.y * tile_dims.y;
    
    int inj_np = 0;

    // If range overlaps with tile
    if (( ri0 < tile_dims.x ) && ( ri1 >= 0 ) &&
        ( rj0 < tile_dims.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= tile_dims.x ) ri1 = tile_dims.x-1;
        if (rj1 >= tile_dims.y ) rj1 = tile_dims.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = (part.local_tile_start.x + tile_idx.x) * tile_dims.x;
        const int shifty = (part.local_tile_start.y + tile_idx.y) * tile_dims.y;
        const float r2 = radius*radius;

        for( int idx = gpu::block::thread_rank(); idx < vol; idx+= gpu::block::num_threads()) {
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

    inj_np = gpu::warp::reduce_add( inj_np );
    if ( gpu::warp::thread_rank() == 0 ) {
        gpu::block::atomic_fetch_add( &np_tile, inj_np );
    }
    
    gpu::block::sync();  
    if ( gpu::block::thread_rank() == 0 ) {
        np[ tile_id ] = np_tile;
    }
}

} // namespace kernel

void density::sphere::np_inject( part::particles & part, 
    uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range,
    int * np ) const
{
    auto ntiles = part.get_local_ntiles();
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 1024 );

    float2 sphere_center = center;
    sphere_center.x -= ref.x;
    sphere_center.y -= ref.y;

    kernel::np_sphere <<< grid, block >>> ( 
        range, sphere_center, radius, dx, ppc, part.view(), np
    );

}
