#include "core/gpu.cuh"
#include "density.hpp"
#include "util/math.hpp"

namespace {

/**
 * @brief Thread block size for the injection / counting kernels
 * 
 * Enough threads to cover one tile (1 thread per cell), rounded up to a full
 * warp, at most 1024.
 * 
 * @param part      Particle data object
 * @return dim3     Block size
 */
dim3 inject_block( part::particles & part ) {
    auto const tile_dims = part.get_tile_dims();
    unsigned int nt = roundup<gpu::warp_size>( tile_dims.x * tile_dims.y );
    if ( nt > 1024 ) nt = 1024;
    return dim3( nt );
}

}

namespace kernel {

/**
 * @brief Injection range clipped to a single tile (tile cell coordinates)
 * 
 */
struct tile_range {
    /// @brief Lower / upper (inclusive) x cell
    int ri0, ri1;
    /// @brief Lower / upper (inclusive) y cell
    int rj0, rj1;
    /// @brief Number of cells along x
    int row;
    /// @brief Total number of cells
    int vol;
};

/**
 * @brief Clips the (local) injection range to the current tile
 * 
 * @param range         Cell range (local, inclusive)
 * @param tile_idx      Tile index (x,y)
 * @param tile_dims     Tile dimensions
 * @param r             (out) Clipped range in tile coordinates
 * @return true         Range overlaps the tile
 * @return false        Range does not overlap the tile
 */
__device__ 
inline bool clip_range( 
    bounds_2d<unsigned int> const range, int2 const tile_idx, int2 const tile_dims,
    tile_range & r )
{
    r.ri0 = static_cast<int>( range.x.lower ) - tile_idx.x * tile_dims.x;
    r.ri1 = static_cast<int>( range.x.upper ) - tile_idx.x * tile_dims.x;
    r.rj0 = static_cast<int>( range.y.lower ) - tile_idx.y * tile_dims.y;
    r.rj1 = static_cast<int>( range.y.upper ) - tile_idx.y * tile_dims.y;

    if (( r.ri0 >= tile_dims.x ) || ( r.ri1 < 0 ) ||
        ( r.rj0 >= tile_dims.y ) || ( r.rj1 < 0 )) return false;

    if ( r.ri0 < 0 ) r.ri0 = 0;
    if ( r.rj0 < 0 ) r.rj0 = 0;
    if ( r.ri1 >= tile_dims.x ) r.ri1 = tile_dims.x - 1;
    if ( r.rj1 >= tile_dims.y ) r.rj1 = tile_dims.y - 1;

    r.row = r.ri1 - r.ri0 + 1;
    int const col = r.rj1 - r.rj0 + 1;

    // Empty (inverted) range
    if (( r.row <= 0 ) || ( col <= 0 )) return false;

    r.vol = r.row * col;
    return true;
}

/**
 * @brief Cell (tile coordinates) corresponding to linear index inside the clipped range
 * 
 */
__device__ 
inline int2 range_cell( tile_range const & r, int const idx ) {
    return make_int2( idx % r.row + r.ri0, idx / r.row + r.rj0 );
}

/**
 * @brief Position of particle (i0,i1) inside the cell, in [-0.5, 0.5)
 * 
 * @note Uses explicitly rounded operations so that the result is identical in
 *       every kernel (no FMA contraction)
 */
__device__ 
inline float2 cell_pos( unsigned const i0, unsigned const i1, double const dpcx, double const dpcy ) {

/*
    return make_float2(
        __dadd_rn( __dmul_rn( dpcx, i0 + 0.5 ), -0.5 ),
        __dadd_rn( __dmul_rn( dpcy, i1 + 0.5 ), -0.5 )
    );
*/
    // Force fma contraction
    return make_float2(
        fma( dpcx, i0 + 0.5, -0.5 ),
        fma( dpcy, i1 + 0.5, -0.5 )
    );
}

/**
 * @brief Global cell offset of the current tile
 * 
 */
__device__ 
inline int2 tile_shift( part::particles_view const & part, int2 const tile_idx ) {
    return make_int2(
        ( static_cast<int>( part.local_tile_start.x ) + tile_idx.x ) * static_cast<int>( part.tile_dims.x ),
        ( static_cast<int>( part.local_tile_start.y ) + tile_idx.y ) * static_cast<int>( part.tile_dims.y )
    );
}

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
    int2 const tile_idx  = make_int2( blockIdx.x, blockIdx.y );
    int2 const tile_dims = make_int2( part.tile_dims.x, part.tile_dims.y );
    int const tile_id = tile_idx.y * static_cast<int>( part.local_ntiles.x ) + tile_idx.x;

    // Find injection range overlap with tile; if no overlap return
    tile_range r;
    if ( ! clip_range( range, tile_idx, tile_dims, r ) ) return;

    // Number of particles before injection
    int const np = part.tile_np[ tile_id ];

    int const offset = part.tile_offset[ tile_id ];
    int2   * __restrict__ const ix = &part.ix[ offset ];
    float2 * __restrict__ const x  = &part.x[ offset ];
    float3 * __restrict__ const u  = &part.u[ offset ];

    int const np_cell = ppc.x * ppc.y;

    double const dpcx = 1.0 / ppc.x;
    double const dpcy = 1.0 / ppc.y;

    // Each thread takes 1 cell
    for( int idx = gpu::block::thread_rank(); idx < r.vol; idx += gpu::block::num_threads() ) {
        int2 const cell = range_cell( r, idx );

        int part_idx = np + idx * np_cell;

        for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
            for( unsigned i0 = 0; i0 < ppc.x; i0++ ) {
                ix[ part_idx ] = cell;
                x[ part_idx ]  = cell_pos( i0, i1, dpcx, dpcy );
                u[ part_idx ]  = make_float3( 0, 0, 0 );
                part_idx++;
            }
        }
    }

    // All threads have read tile_np[ tile_id ] before it is updated
    gpu::block::sync();

    if ( gpu::block::thread_rank() == 0 ) {
        part.tile_np[ tile_id ] = np + r.vol * np_cell;
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
    int2 const tile_idx  = make_int2( blockIdx.x, blockIdx.y );
    int2 const tile_dims = make_int2( part.tile_dims.x, part.tile_dims.y );
    int const tile_id = tile_idx.y * static_cast<int>( part.local_ntiles.x ) + tile_idx.x;

    tile_range r;
    if ( ! clip_range( range, tile_idx, tile_dims, r ) ) return;

    // Number of particles before injection
    int const np = part.tile_np[ tile_id ];

    int const offset = part.tile_offset[ tile_id ];
    int2   * __restrict__ const ix = &part.ix[ offset ];
    float2 * __restrict__ const x  = &part.x[ offset ];
    float3 * __restrict__ const u  = &part.u[ offset ];

    int const np_cell = ppc.x * ppc.y;

    double const dpcx = 1.0 / ppc.x;
    double const dpcy = 1.0 / ppc.y;

    for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
        for( unsigned i0 = 0; i0 < ppc.x; i0++ ) {
            float2 const pos = cell_pos( i0, i1, dpcx, dpcy );
            int const ppc_idx = i1 * ppc.x + i0;

            // Each thread takes 1 cell
            for( int idx = gpu::block::thread_rank(); idx < r.vol; idx += gpu::block::num_threads() ) {
                int const part_idx = np + r.vol * ppc_idx + idx;
                ix[ part_idx ] = range_cell( r, idx );
                x[ part_idx ]  = pos;
                u[ part_idx ]  = make_float3( 0, 0, 0 );
            }
        }
    }

    // All threads have read tile_np[ tile_id ] before it is updated
    gpu::block::sync();

    if ( gpu::block::thread_rank() == 0 ) {
        part.tile_np[ tile_id ] = np + r.vol * np_cell;
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
    dim3 block= inject_block( part );

#if 0
    std::cout << "(*info*) Injecting uniform density using algorithm mk I\n";
    kernel::inject_uniform_mk1 <<< grid, block >>> (
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
    int2 const tile_idx  = make_int2( blockIdx.x, blockIdx.y );
    int2 const tile_dims = make_int2( part.tile_dims.x, part.tile_dims.y );
    int const tile_id = tile_idx.y * static_cast<int>( part.local_ntiles.x ) + tile_idx.x;

    tile_range r;
    np[ tile_id ] = clip_range( range, tile_idx, tile_dims, r ) ? 
                    r.vol * static_cast<int>( ppc.x * ppc.y ) : 0;
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

template < coord::cart dir >
__device__
inline int step_mask( const float step, const int2 shift, const int2 cell, const float2 pos ) {
    float t;
    if constexpr ( dir == coord::x ) t = (shift.x + cell.x) + (pos.x + 0.5);
    if constexpr ( dir == coord::y ) t = (shift.y + cell.y) + (pos.y + 0.5);

    return (t > step);
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
__global__
void inject_step( 
    bounds_2d<unsigned int> range,
    const float step, const uint2 ppc,
    part::particles_view const part )
{
    int2 const tile_idx  = make_int2( blockIdx.x, blockIdx.y );
    int2 const tile_dims = make_int2( part.tile_dims.x, part.tile_dims.y );
    int const tile_id = tile_idx.y * static_cast<int>( part.local_ntiles.x ) + tile_idx.x;

    // Block-uniform early exit
    tile_range r;
    if ( ! clip_range( range, tile_idx, tile_dims, r ) ) return;

    int const thread_rank = gpu::block::thread_rank();
    int const num_threads = gpu::block::num_threads();

    // Store number of particles before injection
    __shared__ int np_tile;
    if ( thread_rank == 0 ) np_tile = part.tile_np[ tile_id ];
    gpu::block::sync();


    int const offset =  part.tile_offset[ tile_id ];
    int2   * __restrict__ ix = &part.ix[ offset ];
    float2 * __restrict__ x  = &part.x[ offset ];
    float3 * __restrict__ u  = &part.u[ offset ];

    double dpcx = 1.0 / ppc.x;
    double dpcy = 1.0 / ppc.y;

    // Position in global tile array
    int2 const shift = tile_shift( part, tile_idx );

    // Ensure uniform trip count for all threads
    int const niter = ( r.vol + num_threads - 1 ) / num_threads;

    for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
        for( unsigned i0 = 0; i0 < ppc.x; i0++) {
            float2 const pos = cell_pos( i0, i1, dpcx, dpcy );

            for( int it = 0; it < niter; it++ ) {
                int const idx = it * num_threads + thread_rank;

                int2 cell = make_int2( 0, 0 );
                int inj = 0;

                if ( idx < r.vol ) {
                    cell = range_cell( r, idx );
                    inj = step_mask<dir>( step, shift, cell, pos );
                }

                // Called by every thread in the block
                int const off = gpu::block::exscan_add( inj );

                if ( inj ) {
                    const int k = np_tile + off;
                    ix[ k ] = cell;
                    x[ k ]  = pos;
                    u[ k ]  = make_float3(0,0,0);
                }

                // The last thread's exclusive scan + own value is the block total
                if ( thread_rank == num_threads - 1 ) np_tile += off + inj;
                gpu::block::sync();
            }
        }
    }

    if ( thread_rank == 0 ) 
        part.tile_np[ tile_id ] = np_tile;
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
__global__
void np_step( 
    bounds_2d<unsigned int> range,
    const float step, const uint2 ppc,
    part::particles_view const part, int * np )
{
    int2 const tile_idx  = make_int2( blockIdx.x, blockIdx.y );
    int2 const tile_dims = make_int2( part.tile_dims.x, part.tile_dims.y );
    int const tile_id = tile_idx.y * static_cast<int>( part.local_ntiles.x ) + tile_idx.x;

    int const thread_rank = gpu::block::thread_rank();
    int const num_threads = gpu::block::num_threads();

    __shared__ int np_tile;
    if ( thread_rank == 0 ) np_tile = 0;
    gpu::block::sync();

    int inj_np = 0;

    tile_range r;
    if ( clip_range( range, tile_idx, tile_dims, r ) ) {

        double const dpcx = 1.0 / ppc.x;
        double const dpcy = 1.0 / ppc.y;

        int2 const shift = tile_shift( part, tile_idx );

        // No collectives inside this loop, so a non-uniform trip count is fine
        for( int idx = thread_rank; idx < r.vol; idx += num_threads ) {
            int2 const cell = range_cell( r, idx );
            for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
                for( unsigned i0 = 0; i0 < ppc.x; i0++ ) {
                    float2 const pos = cell_pos( i0, i1, dpcx, dpcy );
                    inj_np += step_mask<dir>( step, shift, cell, pos );
                }
            }
        }
    }

    // Called by every thread in the block
    inj_np = gpu::warp::reduce_add( inj_np );
    if ( gpu::warp::thread_rank() == 0 ) {
        gpu::block::atomic_fetch_add( &np_tile, inj_np );
    }
    gpu::block::sync();

    if ( thread_rank == 0 )
        np[ tile_id ] = np_tile;
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
    dim3 block = inject_block( part );

    switch( dir ) {
    case( coord::x ):
        step_pos = (pos - ref.x) / dx.x;
        kernel::inject_step <coord::x> <<< grid, block >>> ( range, step_pos, ppc, part.view() );
        break;
    case( coord::y ):
        step_pos = (pos - ref.y) / dx.y;
        kernel::inject_step <coord::y> <<< grid, block >>> ( range, step_pos, ppc, part.view() );
        break;
    default:
        mpi::fatal( "Invalid direction for step density profile");
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
void density::step::np_inject( part::particles & part, 
    uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range,
    int * np ) const
{

    auto ntiles = part.get_local_ntiles();
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block = inject_block( part );

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

template < coord::cart dir >
__device__
inline int slab_mask( const float start, const float finish, const int2 shift, const int2 cell, const float2 pos ) {
    float t;
    if constexpr ( dir == coord::x ) t = (shift.x + cell.x) + (pos.x + 0.5);
    if constexpr ( dir == coord::y ) t = (shift.y + cell.y) + (pos.y + 0.5);

    return (t >= start) && (t<finish );
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
__global__
void inject_slab( 
    bounds_2d<unsigned int> range,
    const float start, const float finish, uint2 ppc,
    part::particles_view const part )
{
    int2 const tile_idx  = make_int2( blockIdx.x, blockIdx.y );
    int2 const tile_dims = make_int2( part.tile_dims.x, part.tile_dims.y );
    int const tile_id = tile_idx.y * static_cast<int>( part.local_ntiles.x ) + tile_idx.x;

    // Block-uniform early exit
    tile_range r;
    if ( ! clip_range( range, tile_idx, tile_dims, r ) ) return;

    int const thread_rank = gpu::block::thread_rank();
    int const num_threads = gpu::block::num_threads();

    // Store number of particles before injection
    __shared__ int np_tile;
    if ( thread_rank == 0 ) np_tile = part.tile_np[ tile_id ];
    gpu::block::sync();


    int const offset =  part.tile_offset[ tile_id ];
    int2   * __restrict__ ix = &part.ix[ offset ];
    float2 * __restrict__ x  = &part.x[ offset ];
    float3 * __restrict__ u  = &part.u[ offset ];

    double dpcx = 1.0 / ppc.x;
    double dpcy = 1.0 / ppc.y;

    // Position in global tile array
    int2 const shift = tile_shift( part, tile_idx );

    // Ensure uniform trip count for all threads
    int const niter = ( r.vol + num_threads - 1 ) / num_threads;

    for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
        for( unsigned i0 = 0; i0 < ppc.x; i0++) {
            float2 const pos = cell_pos( i0, i1, dpcx, dpcy );

            for( int it = 0; it < niter; it++ ) {
                int const idx = it * num_threads + thread_rank;

                int2 cell = make_int2( 0, 0 );
                int inj = 0;

                if ( idx < r.vol ) {
                    cell = range_cell( r, idx );
                    inj = slab_mask<dir>( start, finish, shift, cell, pos );
                }

                // Called by every thread in the block
                int const off = gpu::block::exscan_add( inj );

                if ( inj ) {
                    const int k = np_tile + off;
                    ix[ k ] = cell;
                    x[ k ]  = pos;
                    u[ k ]  = make_float3(0,0,0);
                }

                // The last thread's exclusive scan + own value is the block total
                if ( thread_rank == num_threads - 1 ) np_tile += off + inj;
                gpu::block::sync();
            }
        }
    }

    if ( thread_rank == 0 ) 
        part.tile_np[ tile_id ] = np_tile;
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
__global__
void np_slab( 
    bounds_2d<unsigned int> range,
    const float start, const float finish, uint2 ppc,
    part::particles_view const part, int * np )
{
    int2 const tile_idx  = make_int2( blockIdx.x, blockIdx.y );
    int2 const tile_dims = make_int2( part.tile_dims.x, part.tile_dims.y );
    int const tile_id = tile_idx.y * static_cast<int>( part.local_ntiles.x ) + tile_idx.x;

    int const thread_rank = gpu::block::thread_rank();
    int const num_threads = gpu::block::num_threads();

    __shared__ int np_tile;
    if ( thread_rank == 0 ) np_tile = 0;
    gpu::block::sync();

    int inj_np = 0;

    tile_range r;
    if ( clip_range( range, tile_idx, tile_dims, r ) ) {

        double const dpcx = 1.0 / ppc.x;
        double const dpcy = 1.0 / ppc.y;

        int2 const shift = tile_shift( part, tile_idx );

        // No collectives inside this loop, so a non-uniform trip count is fine
        for( int idx = thread_rank; idx < r.vol; idx += num_threads ) {
            int2 const cell = range_cell( r, idx );
            for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
                for( unsigned i0 = 0; i0 < ppc.x; i0++ ) {
                    float2 const pos = cell_pos( i0, i1, dpcx, dpcy );
                    inj_np += slab_mask<dir>( start, finish, shift, cell, pos );
                }
            }
        }
    }

    // Called by every thread in the block
    inj_np = gpu::warp::reduce_add( inj_np );
    if ( gpu::warp::thread_rank() == 0 ) {
        gpu::block::atomic_fetch_add( &np_tile, inj_np );
    }
    gpu::block::sync();

    if ( thread_rank == 0 )
        np[ tile_id ] = np_tile;
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
    dim3 block = inject_block( part );

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
    dim3 block = inject_block( part );

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

__device__
inline int sphere_mask( const float r2, const float2 center, const float2 dx, const int2 shift, const int2 cell, const float2 pos ) {
    float gx = ((shift.x + cell.x) + (pos.x+0.5)) * dx.x;
    float gy = ((shift.y + cell.y) + (pos.y+0.5)) * dx.y;
    return ((gx - center.x)*(gx - center.x) + (gy - center.y)*(gy - center.y)) < r2;
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
__global__
inline void inject_sphere( 
    bounds_2d<unsigned int> range,
    float2 center, float radius, float2 dx, uint2 ppc,
    part::particles_view const part )
{
    int2 const tile_idx  = make_int2( blockIdx.x, blockIdx.y );
    int2 const tile_dims = make_int2( part.tile_dims.x, part.tile_dims.y );
    int const tile_id = tile_idx.y * static_cast<int>( part.local_ntiles.x ) + tile_idx.x;

    // Block-uniform early exit
    tile_range r;
    if ( ! clip_range( range, tile_idx, tile_dims, r ) ) return;

    int const thread_rank = gpu::block::thread_rank();
    int const num_threads = gpu::block::num_threads();

    // Store number of particles before injection
    __shared__ int np_tile;
    if ( thread_rank == 0 ) np_tile = part.tile_np[ tile_id ];
    gpu::block::sync();


    int const offset =  part.tile_offset[ tile_id ];
    int2   * __restrict__ ix = &part.ix[ offset ];
    float2 * __restrict__ x  = &part.x[ offset ];
    float3 * __restrict__ u  = &part.u[ offset ];

    double dpcx = 1.0 / ppc.x;
    double dpcy = 1.0 / ppc.y;

    // Position in global tile array
    int2 const shift = tile_shift( part, tile_idx );

    const float r2 = radius*radius;

    // Ensure uniform trip count for all threads
    int const niter = ( r.vol + num_threads - 1 ) / num_threads;

    for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
        for( unsigned i0 = 0; i0 < ppc.x; i0++) {
            float2 const pos = cell_pos( i0, i1, dpcx, dpcy );

            for( int it = 0; it < niter; it++ ) {
                int const idx = it * num_threads + thread_rank;

                int2 cell = make_int2( 0, 0 );
                int inj = 0;

                if ( idx < r.vol ) {
                    cell = range_cell( r, idx );
                    inj = sphere_mask( r2, center, dx, shift, cell, pos );
                }

                // Called by every thread in the block
                int const off = gpu::block::exscan_add( inj );

                if ( inj ) {
                    const int k = np_tile + off;
                    ix[ k ] = cell;
                    x[ k ]  = pos;
                    u[ k ]  = make_float3(0,0,0);
                }

                // The last thread's exclusive scan + own value is the block total
                if ( thread_rank == num_threads - 1 ) np_tile += off + inj;
                gpu::block::sync();
            }
        }
    }

    if ( thread_rank == 0 ) 
        part.tile_np[ tile_id ] = np_tile;
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
__global__
void np_sphere(
    bounds_2d<unsigned int> range,
    float2 center, float radius, float2 dx, uint2 ppc,
    part::particles_view const part, int * np )
{
    int2 const tile_idx  = make_int2( blockIdx.x, blockIdx.y );
    int2 const tile_dims = make_int2( part.tile_dims.x, part.tile_dims.y );
    int const tile_id = tile_idx.y * static_cast<int>( part.local_ntiles.x ) + tile_idx.x;

    int const thread_rank = gpu::block::thread_rank();
    int const num_threads = gpu::block::num_threads();

    __shared__ int np_tile;
    if ( thread_rank == 0 ) np_tile = 0;
    gpu::block::sync();

    int inj_np = 0;

    tile_range r;
    if ( clip_range( range, tile_idx, tile_dims, r ) ) {

        double const dpcx = 1.0 / ppc.x;
        double const dpcy = 1.0 / ppc.y;

        int2 const shift = tile_shift( part, tile_idx );

        const float r2 = radius*radius;

        // No collectives inside this loop, so a non-uniform trip count is fine
        for( int idx = thread_rank; idx < r.vol; idx += num_threads ) {
            int2 const cell = range_cell( r, idx );
            for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
                for( unsigned i0 = 0; i0 < ppc.x; i0++ ) {
                    float2 const pos = cell_pos( i0, i1, dpcx, dpcy );
                    inj_np += sphere_mask( r2, center, dx, shift, cell, pos );
                }
            }
        }
    }

    // Called by every thread in the block
    inj_np = gpu::warp::reduce_add( inj_np );
    if ( gpu::warp::thread_rank() == 0 ) {
        gpu::block::atomic_fetch_add( &np_tile, inj_np );
    }
    gpu::block::sync();

    if ( thread_rank == 0 )
        np[ tile_id ] = np_tile;
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
    dim3 block = inject_block( part );

    float2 sphere_center = center;
    sphere_center.x -= ref.x;
    sphere_center.y -= ref.y;

    kernel::inject_sphere <<< grid, block >>> ( 
        range, sphere_center, radius, dx, ppc, part.view()
    );
}


void density::sphere::np_inject( part::particles & part, 
    uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range,
    int * np ) const
{
    auto ntiles = part.get_local_ntiles();
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block = inject_block( part );

    float2 sphere_center = center;
    sphere_center.x -= ref.x;
    sphere_center.y -= ref.y;

    kernel::np_sphere <<< grid, block >>> ( 
        range, sphere_center, radius, dx, ppc, part.view(), np
    );
}
