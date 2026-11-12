#include "core/gpu.cuh"
#include "density.hpp"
#include "util/math.hpp"
#include <cmath>
#include <iostream>

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
 */
__device__ 
inline float2 cell_pos( unsigned const i0, unsigned const i1, double const dpcx, double const dpcy ) {

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
 * @note particles in the same cell are injected contiguously
 * 
 * @param range     Cell range (local) to inject particles in
 * @param ppc       Number of particles per cell 
 * @param part      Particle data 
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
 * @note Places contiguous particles in different cells. This minimizes memory
 *       collisions when depositing current, especially for very low 
 *       temperatures.
 * 
 * @param range     Cell range (local) to inject particles in
 * @param ppc       Number of particles per cell 
 * @param part      Particle data 
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

    // Total number of particles to inject in tile
    int const total = r.vol * np_cell;

    // All particles are processed in a single flat loop, so that all threads
    // are kept busy even when the injection range is small (e.g. a single
    // cell column when a moving window advances).
    //
    // The flat index is n = ppc_idx * r.vol + cell_idx, i.e. the
    // particle-in-cell index is the slowest, so that contiguous particles are
    // in different cells (same ordering as the original nested loop version)
    for( int n = gpu::block::thread_rank(); n < total; n += gpu::block::num_threads() ) {
        int const ppc_idx  = n / r.vol;
        int const cell_idx = n - ppc_idx * r.vol;

        unsigned const i1 = ppc_idx / ppc.x;
        unsigned const i0 = ppc_idx - i1 * ppc.x;

        int const part_idx = np + n;
        ix[ part_idx ] = range_cell( r, cell_idx );
        x[ part_idx ]  = cell_pos( i0, i1, dpcx, dpcy );
        u[ part_idx ]  = make_float3( 0, 0, 0 );
    }

    // All threads have read tile_np[ tile_id ] before it is updated
    gpu::block::sync();

    if ( gpu::block::thread_rank() == 0 ) {
        part.tile_np[ tile_id ] = np + total;
    }
}

/**
 * @brief Kernel for counting how many particles will be injected
 *
 * @note Launch with grid(ntiles.x, ntiles.y) and block(1)
 * 
 * @param range     Cell range (local) to inject particles in
 * @param ppc       Number of particles per cell 
 * @param part      Particle data
 * @param tile_np   (out) Number of particles to inject per tile
 */
__global__
inline void np_uniform( 
    bounds_2d<unsigned int> range,
    uint2 const ppc,
    part::particles_view const part,
    int * tile_np )
{
    int2 const tile_idx  = make_int2( blockIdx.x, blockIdx.y );
    int2 const tile_dims = make_int2( part.tile_dims.x, part.tile_dims.y );
    int const tile_id = tile_idx.y * static_cast<int>( part.local_ntiles.x ) + tile_idx.x;

    tile_range r;
    tile_np[ tile_id ] = clip_range( range, tile_idx, tile_dims, r ) ? 
                    r.vol * static_cast<int>( ppc.x * ppc.y ) : 0;
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
struct step_mask {
    /// @brief Step position (normalized to node grid coordinates)
    float step;

    __device__ 
    int operator()( const int2 shift, const int2 cell, const float2 pos ) const {
        float t;
        if constexpr ( dir == coord::x ) t = (shift.x + cell.x) + (pos.x + 0.5);
        if constexpr ( dir == coord::y ) t = (shift.y + cell.y) + (pos.y + 0.5);

        return (t >= step);
    }
};

template < coord::cart dir >
struct slab_mask {
    /// @brief Slab start position (normalized to node grid coordinates)
    float start;
    /// @brief Slab end position (normalized to node grid coordinates)
    float finish;

    __device__ 
    int operator()( const int2 shift, const int2 cell, const float2 pos ) const {
        float s;
        if constexpr ( dir == coord::x ) s = (shift.x + cell.x) + (pos.x + 0.5);
        if constexpr ( dir == coord::y ) s = (shift.y + cell.y) + (pos.y + 0.5);
        return ( s >= start ) && ( s < finish );
    }
};

struct sphere_mask {
    /// @brief Sphere center in simulation units (relative to ref)
    float2 center;
    /// @brief Square of sphere radius in simulation units
    double r2;
    /// @brief Cell size
    float2 dx;

    __device__ 
    int operator()( const int2 shift, const int2 cell, const float2 pos ) const {
        double gx = fma((shift.x + cell.x) + (pos.x+0.5), 
            static_cast<double>(dx.x),
            -static_cast<double>(center.x));
        double gy = fma((shift.y + cell.y) + (pos.y+0.5), 
            static_cast<double>(dx.y),
            -static_cast<double>(center.y));
        return fma( gx, gx, gy*gy) < r2;
    }
};

/**
 * @brief Kernel for injecting particles using a mask profile
 * 
 * @details All candidate particles in the tile (cells in range x particles per
 *          cell) are processed in a single flat loop, so that all threads are
 *          kept busy even when the injection range is small (e.g. a single
 *          cell column when a moving window advances).
 * 
 *          The flat candidate index is `n = ppc_idx * r.vol + cell_idx`, i.e.
 *          the particle-in-cell index is the slowest, which reproduces the
 *          particle ordering of the original (nested loop) version.
 * 
 *          Particle positions in the buffer are deterministic: each injected
 *          particle is stored at the position given by the number of injected
 *          candidates with a lower flat index. This is computed using a
 *          ballot-based block scan: each warp counts its injected particles
 *          with a ballot, warp totals are exchanged through shared memory,
 *          and each warp combines them with 2 warp reductions. Warp totals
 *          are double-buffered, so only 1 block sync per iteration is required.
 * 
 * @note    Requires a 1D block with a whole number of warps.
 * 
 * @tparam mask_t       Mask type
 * @param mask          Injection mask
 * @param range         Cell range (local) to inject particles in
 * @param ppc           Number of particles per cell
 * @param part          Particle data
 */
template < class mask_t >
__global__
void inject_mask( 
    mask_t mask,
    bounds_2d<unsigned int> range,
    const uint2 ppc,
    part::particles_view const part )
{
    int2 const tile_idx  = make_int2( blockIdx.x, blockIdx.y );
    int2 const tile_dims = make_int2( part.tile_dims.x, part.tile_dims.y );
    int const tile_id = tile_idx.y * static_cast<int>( part.local_ntiles.x ) + tile_idx.x;

    // Block-uniform early exit
    tile_range r;
    if ( ! clip_range( range, tile_idx, tile_dims, r ) ) return;

    // Total number of candidate particles in tile
    int const np_cell = ppc.x * ppc.y;
    int const total   = r.vol * np_cell;

    // Block-uniform early exit (e.g. ppc = 0)
    if ( total == 0 ) return;

    // Number of particles before injection (kept in a register by every thread,
    // all threads update it identically)
    int np = part.tile_np[ tile_id ];

    int const offset =  part.tile_offset[ tile_id ];
    int2   * __restrict__ ix = &part.ix[ offset ];
    float2 * __restrict__ x  = &part.x[ offset ];
    float3 * __restrict__ u  = &part.u[ offset ];

    double const dpcx = 1.0 / ppc.x;
    double const dpcy = 1.0 / ppc.y;

    // Position in global tile array
    int2 const shift = tile_shift( part, tile_idx );

    int const thread_rank = gpu::block::thread_rank();
    int const num_threads = gpu::block::num_threads();

    int const      lane        = gpu::warp::thread_rank();
    int const      warp_id     = gpu::warp::group_rank();
    int const      nwarps      = gpu::block::num_warps();
    unsigned const lanemask_lt = ( 1u << lane ) - 1;

    /// @brief [shared] Number of injected particles per warp (double buffered)
    __shared__ int _warp_np[ 2 ][ gpu::max_warps ];

    // Ensure uniform trip count for all threads
    int const niter = ( total + num_threads - 1 ) / num_threads;

    for( int it = 0; it < niter; it++ ) {
        int const n = it * num_threads + thread_rank;

        int2   cell = make_int2( 0, 0 );
        float2 pos  = make_float2( 0, 0 );
        int    inj  = 0;

        if ( n < total ) {
            // Particle-in-cell index (slowest) and cell index (fastest)
            int const ppc_idx  = n / r.vol;
            int const cell_idx = n - ppc_idx * r.vol;

            unsigned const i1 = ppc_idx / ppc.x;
            unsigned const i0 = ppc_idx - i1 * ppc.x;

            pos  = cell_pos( i0, i1, dpcx, dpcy );
            cell = range_cell( r, cell_idx );
            inj  = mask( shift, cell, pos );
        }

        // Number of injected particles in this warp
        unsigned const ballot = gpu::warp::ballot( inj );

        // Exchange warp totals. Double buffering makes this safe with a single
        // sync: a warp can only write this buffer again 2 iterations later,
        // after the next iteration's sync, i.e., after all warps are done
        // reading it
        int * const warp_np = _warp_np[ it & 1 ];
        if ( lane == 0 ) warp_np[ warp_id ] = __popc( ballot );
        gpu::block::sync();

        // Lane l holds the total of warp l
        int const v = ( lane < nwarps ) ? warp_np[ lane ] : 0;

        // Injected particles in previous warps, and in the whole block
        int const prev = gpu::warp::reduce_add( ( lane < warp_id ) ? v : 0 );
        int const sum  = gpu::warp::reduce_add( v );

        if ( inj ) {
            int const k = np + prev + __popc( ballot & lanemask_lt );
            ix[ k ] = cell;
            x[ k ]  = pos;
            u[ k ]  = make_float3(0,0,0);
        }

        np += sum;
    }

    // All threads read tile_np[ tile_id ] before the first sync in the loop
    // (niter >= 1), so it is safe to update it here without an additional sync
    if ( thread_rank == 0 ) 
        part.tile_np[ tile_id ] = np;
}

/**
 * @brief Kernel for counting how many particles will be injected by a mask profile
 * 
 * @tparam mask_t       Mask type
 * @param mask          Injection mask
 * @param range         Cell range (local) to inject particles in
 * @param ppc           Number of particles per cell
 * @param part          Particle data
 * @param tile_np       (out) Number of particles to inject per tile
 */
template < class mask_t >
__global__
void np_mask( 
    mask_t mask,
    bounds_2d<unsigned int> range,
    const uint2 ppc,
    part::particles_view const part, int * tile_np )
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
                    inj_np += mask( shift, cell, pos );
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
        tile_np[ tile_id ] = np_tile;
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
    auto ntiles = part.get_local_ntiles();
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block = inject_block( part );

    switch( dir ) {
    case( coord::x ): {
        kernel::step_mask<coord::x> const mask { (pos - ref.x) / dx.x };
        kernel::inject_mask <<< grid, block >>> ( mask, range, ppc, part.view() );
        break; }
    case( coord::y ): {
        kernel::step_mask<coord::y> const mask { (pos - ref.y) / dx.y };
        kernel::inject_mask <<< grid, block >>> ( mask, range, ppc, part.view() );
        break; }
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

    switch( dir ) {
    case( coord::x ): {
        kernel::step_mask<coord::x> const mask { (pos - ref.x) / dx.x };
        kernel::np_mask <<< grid, block >>> ( mask, range, ppc, part.view(), np );
        break; }
    case( coord::y ): {
        kernel::step_mask<coord::y> const mask { (pos - ref.y) / dx.y };
        kernel::np_mask <<< grid, block >>> ( mask, range, ppc, part.view(), np );
        break; }
    default:
        mpi::fatal( "Invalid direction for step density profile");
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
void density::slab::inject( part::particles & part,
    uint2 const ppc,float2 const dx, float2 const ref, bounds_2d<unsigned int> range ) const
{
    auto ntiles = part.get_local_ntiles();
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block = inject_block( part );

    switch( dir ) {
    case( coord::x ): {
        kernel::slab_mask<coord::x> const mask { 
            (begin - ref.x)/ dx.x,
            (end - ref.x)/ dx.x };
        kernel::inject_mask <<< grid, block >>> ( mask, range, ppc, part.view() );
        break; }
    case( coord::y ): {
        kernel::slab_mask<coord::y> const mask { 
            (begin - ref.y)/ dx.y,
            (end - ref.y)/ dx.y };
        kernel::inject_mask <<< grid, block >>> ( mask, range, ppc, part.view() );
        break; }
    default:
        mpi::fatal( "Invalid direction for slab density profile");
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

    switch( dir ) {
    case( coord::x ): {
        kernel::slab_mask<coord::x> const mask { 
            (begin - ref.x)/ dx.x,
            (end - ref.x)/ dx.x };
        kernel::np_mask <<< grid, block >>> ( mask, range, ppc, part.view(), np
        );
        break; }
    case( coord::y ): {
        kernel::slab_mask<coord::y> const mask { 
            (begin - ref.y)/ dx.y,
            (end - ref.y)/ dx.y };
        kernel::np_mask <<< grid, block >>> ( mask, range, ppc, part.view(), np
        );
        break; }
    default:
        mpi::fatal( "Invalid direction for slab density profile");
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
void density::sphere::inject( part::particles & part,
    uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range ) const
{
    auto ntiles = part.get_local_ntiles();
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block = inject_block( part );

    kernel::sphere_mask mask {
        center - ref,
        radius * radius,
        dx
    };

    kernel::inject_mask <<< grid, block >>> ( 
        mask, range, ppc, part.view()
    );
}


void density::sphere::np_inject( part::particles & part, 
    uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range,
    int * np ) const
{
    auto ntiles = part.get_local_ntiles();
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block = inject_block( part );

    kernel::sphere_mask mask {
        center - ref,
        radius * radius,
        dx
    };

    kernel::np_mask <<< grid, block >>> ( 
        mask, range, ppc, part.view(), np
    );
}
