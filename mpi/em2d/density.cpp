#include "density.hpp"
#include "util/math.hpp"
#include <cmath>
#include <iostream>

/**
 * @note The kernel:: functions process a single tile, mirroring the CUDA
 *       kernels (one thread block per tile) in cuda-mpi/em2ds/density.cu. The
 *       host methods loop over all local tiles using OpenMP.
 * 
 *       On the CPU each tile is processed serially, so particles are simply
 *       stored in the order they are generated. This is the same order the
 *       GPU version obtains through its block scan: candidate particles are
 *       ordered with the particle-in-cell index (i1, i0) as the slowest
 *       index and the cell index as the fastest one, so that contiguous
 *       particles are in different cells.
 * 
 *       Everything is in an anonymous namespace (internal linkage) to avoid
 *       clashes with kernel:: symbols in other translation units.
 */
namespace {

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
inline int2 range_cell( tile_range const & r, int const idx ) {
    return make_int2( idx % r.row + r.ri0, idx / r.row + r.rj0 );
}

/**
 * @brief Position of particle (i0,i1) inside the cell, in [-0.5, 0.5)
 *
 */
inline float2 cell_pos( unsigned const i0, unsigned const i1, double const dpcx, double const dpcy ) {

    return make_float2(
        std::fma( dpcx, i0 + 0.5, -0.5 ),
        std::fma( dpcy, i1 + 0.5, -0.5 )
    );
}

/**
 * @brief Global cell offset of the current tile
 * 
 */
inline int2 tile_shift( part::particles_view const & part, int2 const tile_idx ) {
    return make_int2(
        ( static_cast<int>( part.local_tile_start.x ) + tile_idx.x ) * static_cast<int>( part.tile_dims.x ),
        ( static_cast<int>( part.local_tile_start.y ) + tile_idx.y ) * static_cast<int>( part.tile_dims.y )
    );
}

/**
 * @brief Loops over all local tiles (OpenMP parallel), calling f( tile_idx )
 * 
 * @param part  Particle data object
 * @param f     Function to call for each tile
 */
template < class F >
inline void for_each_tile( part::particles const & part, F && f ) {
    auto const ntiles = part.get_local_ntiles();
    int const nx = ntiles.x;
    int const ny = ntiles.y;

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for( int ty = 0; ty < ny; ty++ ) {
        for( int tx = 0; tx < nx; tx++ ) {
            f( make_int2( tx, ty ) );
        }
    }
}

/**
 * @brief kernel for injecting a uniform plasma density (mk1)
 *
 * @note particles in the same cell are injected contiguously
 * 
 * @param tile_idx  Tile index
 * @param range     Cell range (local) to inject particles in
 * @param ppc       Number of particles per cell 
 * @param part      Particle data 
 */
[[maybe_unused]]
void inject_uniform_mk1( 
    int2 const tile_idx,
    bounds_2d<unsigned int> range,
    uint2 const ppc,
    part::particles_view const part )
{
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

    for( int idx = 0; idx < r.vol; idx++ ) {
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

    part.tile_np[ tile_id ] = np + r.vol * np_cell;
}

/**
 * @brief kernel for injecting a uniform plasma density (mk2)
 *
 * @note Places contiguous particles in different cells. This minimizes memory
 *       collisions when depositing current, especially for very low 
 *       temperatures.
 * 
 * @param tile_idx  Tile index
 * @param range     Cell range (local) to inject particles in
 * @param ppc       Number of particles per cell 
 * @param part      Particle data 
 */
void inject_uniform( 
    int2 const tile_idx,
    bounds_2d<unsigned int> range,
    uint2 const ppc,
    part::particles_view const part )
{
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

    double const dpcx = 1.0 / ppc.x;
    double const dpcy = 1.0 / ppc.y;

    // Particle-in-cell index is the slowest, cell index the fastest, i.e.
    // flat index n = ppc_idx * r.vol + cell_idx (same as the GPU version)
    int part_idx = np;
    for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
        for( unsigned i0 = 0; i0 < ppc.x; i0++ ) {
            float2 const pos = cell_pos( i0, i1, dpcx, dpcy );
            for( int idx = 0; idx < r.vol; idx++ ) {
                ix[ part_idx ] = range_cell( r, idx );
                x[ part_idx ]  = pos;
                u[ part_idx ]  = make_float3( 0, 0, 0 );
                part_idx++;
            }
        }
    }

    part.tile_np[ tile_id ] = part_idx;
}

/**
 * @brief Kernel for counting how many particles will be injected
 * 
 * @param tile_idx  Tile index
 * @param range     Cell range (local) to inject particles in
 * @param ppc       Number of particles per cell 
 * @param part      Particle data
 * @param tile_np   (out) Number of particles to inject per tile
 */
void np_uniform( 
    int2 const tile_idx,
    bounds_2d<unsigned int> range,
    uint2 const ppc,
    part::particles_view const part,
    int * tile_np )
{
    int2 const tile_dims = make_int2( part.tile_dims.x, part.tile_dims.y );
    int const tile_id = tile_idx.y * static_cast<int>( part.local_ntiles.x ) + tile_idx.x;

    tile_range r;
    tile_np[ tile_id ] = clip_range( range, tile_idx, tile_dims, r ) ? 
                    r.vol * static_cast<int>( ppc.x * ppc.y ) : 0;
}

/**
 * @brief Step mask: inject for positions >= step
 * 
 * @tparam dir  Step direction
 */
template < coord::cart dir >
struct step_mask {
    /// @brief Step position (normalized to node grid coordinates)
    float step;

    int operator()( const int2 shift, const int2 cell, const float2 pos ) const {
        float t;
        if constexpr ( dir == coord::x ) t = (shift.x + cell.x) + (pos.x + 0.5);
        if constexpr ( dir == coord::y ) t = (shift.y + cell.y) + (pos.y + 0.5);

        return (t >= step);
    }
};

/**
 * @brief Slab mask: inject for positions in [start, finish)
 * 
 * @tparam dir  Slab direction
 */
template < coord::cart dir >
struct slab_mask {
    /// @brief Slab start position (normalized to node grid coordinates)
    float start;
    /// @brief Slab end position (normalized to node grid coordinates)
    float finish;

    int operator()( const int2 shift, const int2 cell, const float2 pos ) const {
        float s;
        if constexpr ( dir == coord::x ) s = (shift.x + cell.x) + (pos.x + 0.5);
        if constexpr ( dir == coord::y ) s = (shift.y + cell.y) + (pos.y + 0.5);
        return ( s >= start ) && ( s < finish );
    }
};

/**
 * @brief Sphere mask: inject for positions strictly inside the circle
 * 
 */
struct sphere_mask {
    /// @brief Sphere center in simulation units (relative to ref)
    float2 center;
    /// @brief Square of sphere radius in simulation units
    double r2;
    /// @brief Cell size
    float2 dx;

    int operator()( const int2 shift, const int2 cell, const float2 pos ) const {
        double gx = std::fma((shift.x + cell.x) + (pos.x+0.5), 
            static_cast<double>(dx.x),
            -static_cast<double>(center.x));
        double gy = std::fma((shift.y + cell.y) + (pos.y+0.5), 
            static_cast<double>(dx.y),
            -static_cast<double>(center.y));
        return std::fma( gx, gx, gy*gy) < r2;
    }
};

/**
 * @brief Kernel for injecting particles using a mask profile
 * 
 * @details Candidate particles (cells in range x particles per cell) are
 *          tested against the mask in the order given by the flat index
 *          `n = ppc_idx * r.vol + cell_idx`, i.e. the particle-in-cell index
 *          is the slowest. Accepted particles are stored contiguously in this
 *          order, which matches the (block scan) ordering of the GPU version.
 * 
 * @tparam mask_t       Mask type
 * @param tile_idx      Tile index
 * @param mask          Injection mask
 * @param range         Cell range (local) to inject particles in
 * @param ppc           Number of particles per cell
 * @param part          Particle data
 */
template < class mask_t >
void inject_mask( 
    int2 const tile_idx,
    mask_t const & mask,
    bounds_2d<unsigned int> range,
    const uint2 ppc,
    part::particles_view const part )
{
    int2 const tile_dims = make_int2( part.tile_dims.x, part.tile_dims.y );
    int const tile_id = tile_idx.y * static_cast<int>( part.local_ntiles.x ) + tile_idx.x;

    tile_range r;
    if ( ! clip_range( range, tile_idx, tile_dims, r ) ) return;

    // Number of particles before injection
    int np = part.tile_np[ tile_id ];

    int const offset =  part.tile_offset[ tile_id ];
    int2   * __restrict__ ix = &part.ix[ offset ];
    float2 * __restrict__ x  = &part.x[ offset ];
    float3 * __restrict__ u  = &part.u[ offset ];

    double const dpcx = 1.0 / ppc.x;
    double const dpcy = 1.0 / ppc.y;

    // Position in global tile array
    int2 const shift = tile_shift( part, tile_idx );

    for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
        for( unsigned i0 = 0; i0 < ppc.x; i0++ ) {
            float2 const pos = cell_pos( i0, i1, dpcx, dpcy );
            for( int idx = 0; idx < r.vol; idx++ ) {
                int2 const cell = range_cell( r, idx );
                if ( mask( shift, cell, pos ) ) {
                    ix[ np ] = cell;
                    x[ np ]  = pos;
                    u[ np ]  = make_float3( 0, 0, 0 );
                    np++;
                }
            }
        }
    }

    part.tile_np[ tile_id ] = np;
}

/**
 * @brief Kernel for counting how many particles will be injected by a mask profile
 * 
 * @tparam mask_t       Mask type
 * @param tile_idx      Tile index
 * @param mask          Injection mask
 * @param range         Cell range (local) to inject particles in
 * @param ppc           Number of particles per cell
 * @param part          Particle data
 * @param tile_np       (out) Number of particles to inject per tile
 */
template < class mask_t >
void np_mask( 
    int2 const tile_idx,
    mask_t const & mask,
    bounds_2d<unsigned int> range,
    const uint2 ppc,
    part::particles_view const part, int * tile_np )
{
    int2 const tile_dims = make_int2( part.tile_dims.x, part.tile_dims.y );
    int const tile_id = tile_idx.y * static_cast<int>( part.local_ntiles.x ) + tile_idx.x;

    int inj_np = 0;

    tile_range r;
    if ( clip_range( range, tile_idx, tile_dims, r ) ) {

        double const dpcx = 1.0 / ppc.x;
        double const dpcy = 1.0 / ppc.y;

        int2 const shift = tile_shift( part, tile_idx );

        for( int idx = 0; idx < r.vol; idx++ ) {
            int2 const cell = range_cell( r, idx );
            for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
                for( unsigned i0 = 0; i0 < ppc.x; i0++ ) {
                    float2 const pos = cell_pos( i0, i1, dpcx, dpcy );
                    inj_np += mask( shift, cell, pos );
                }
            }
        }
    }

    tile_np[ tile_id ] = inj_np;
}

/**
 * @brief Injects particles in all local tiles using a mask profile
 * 
 */
template < class mask_t >
void inject_mask_all( part::particles & part, mask_t const & mask,
    bounds_2d<unsigned int> const range, uint2 const ppc )
{
    auto const view = part.view();
    for_each_tile( part, [&]( int2 const tile_idx ) {
        inject_mask( tile_idx, mask, range, ppc, view );
    });
}

/**
 * @brief Gets number of particles to inject in all local tiles using a mask profile
 * 
 */
template < class mask_t >
void np_mask_all( part::particles & part, mask_t const & mask,
    bounds_2d<unsigned int> const range, uint2 const ppc, int * np )
{
    auto const view = part.view();
    for_each_tile( part, [&]( int2 const tile_idx ) {
        np_mask( tile_idx, mask, range, ppc, view, np );
    });
}

} // namespace kernel

} // namespace

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
    uint2 const ppc, [[maybe_unused]] float2 const dx, [[maybe_unused]] float2 const ref,
    bounds_2d<unsigned int> range ) const
{
    auto const view = part.view();

#if 0
    std::cout << "(*info*) Injecting uniform density using algorithm mk I\n";
    kernel::for_each_tile( part, [&]( int2 const tile_idx ) {
        kernel::inject_uniform_mk1( tile_idx, range, ppc, view );
    });
#else
    kernel::for_each_tile( part, [&]( int2 const tile_idx ) {
        kernel::inject_uniform( tile_idx, range, ppc, view );
    });
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
    uint2 const ppc, [[maybe_unused]] float2 const dx, [[maybe_unused]] float2 const ref,
    bounds_2d<unsigned int> range, int * np ) const
{
    auto const view = part.view();
    kernel::for_each_tile( part, [&]( int2 const tile_idx ) {
        kernel::np_uniform( tile_idx, range, ppc, view, np );
    });
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
    switch( dir ) {
    case( coord::x ): {
        kernel::step_mask<coord::x> const mask { (pos - ref.x) / dx.x };
        kernel::inject_mask_all( part, mask, range, ppc );
        break; }
    case( coord::y ): {
        kernel::step_mask<coord::y> const mask { (pos - ref.y) / dx.y };
        kernel::inject_mask_all( part, mask, range, ppc );
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
    switch( dir ) {
    case( coord::x ): {
        kernel::step_mask<coord::x> const mask { (pos - ref.x) / dx.x };
        kernel::np_mask_all( part, mask, range, ppc, np );
        break; }
    case( coord::y ): {
        kernel::step_mask<coord::y> const mask { (pos - ref.y) / dx.y };
        kernel::np_mask_all( part, mask, range, ppc, np );
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
    switch( dir ) {
    case( coord::x ): {
        kernel::slab_mask<coord::x> const mask { 
            (begin - ref.x)/ dx.x,
            (end - ref.x)/ dx.x };
        kernel::inject_mask_all( part, mask, range, ppc );
        break; }
    case( coord::y ): {
        kernel::slab_mask<coord::y> const mask { 
            (begin - ref.y)/ dx.y,
            (end - ref.y)/ dx.y };
        kernel::inject_mask_all( part, mask, range, ppc );
        break; }
    default:
        mpi::fatal( "Invalid direction for slab density profile");
    }
}

/**
 * @brief Returns number of particles per tile that a slab profile would inject
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
    switch( dir ) {
    case( coord::x ): {
        kernel::slab_mask<coord::x> const mask { 
            (begin - ref.x)/ dx.x,
            (end - ref.x)/ dx.x };
        kernel::np_mask_all( part, mask, range, ppc, np );
        break; }
    case( coord::y ): {
        kernel::slab_mask<coord::y> const mask { 
            (begin - ref.y)/ dx.y,
            (end - ref.y)/ dx.y };
        kernel::np_mask_all( part, mask, range, ppc, np );
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
    kernel::sphere_mask const mask {
        center - ref,
        radius * radius,
        dx
    };

    kernel::inject_mask_all( part, mask, range, ppc );
}

/**
 * @brief Returns number of particles per tile that a sphere profile would inject
 * 
 * @param part      Particle data object
 * @param ppc       Number of particles per cell
 * @param dx        Cell size
 * @param ref       Position of local grid on global simulation box in simulation
 *                  units
 * @param range     Cell range in which to inject
 * @param np        (out) Number of particles to inject per tile
 */
void density::sphere::np_inject( part::particles & part, 
    uint2 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range,
    int * np ) const
{
    kernel::sphere_mask const mask {
        center - ref,
        radius * radius,
        dx
    };

    kernel::np_mask_all( part, mask, range, ppc, np );
}
