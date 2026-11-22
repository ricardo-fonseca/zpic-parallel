#include "density.hpp"
#include "util/math.hpp"
#include <cmath>
#include <iostream>

/**
 * @note The kernel:: functions process a single tile, mirroring the CUDA
 *       kernels (one thread block per tile) in cuda-mpi/q3d/density.cu. The
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

/**
 * @brief Loops over all local tiles (OpenMP parallel), calling f( tile_idx )
 * 
 * @param part  Particle data object
 * @param f     Function to call for each tile
 */
template < class F >
inline void for_each_tile( cyl_part::particles_view const & view, F && f ) {
    int const nx = view.local_ntiles.x;
    int const ny = view.local_ntiles.y;

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for( int ty = 0; ty < ny; ty++ ) {
        for( int tx = 0; tx < nx; tx++ ) {
            f( make_int2( tx, ty ) );
        }
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
        ops::fma( dpcx, i0 + 0.5, -0.5 ),
        ops::fma( dpcy, i1 + 0.5, -0.5 )
    );
}

/**
 * @brief Global cell offset of the current tile
 * 
 */ 
inline int2 tile_shift( cyl_part::particles_view const & part, int2 const tile_idx ) {
    return make_int2(
        ( static_cast<int>( part.local_tile_start.x ) + tile_idx.x ) * static_cast<int>( part.tile_dims.x ),
        ( static_cast<int>( part.local_tile_start.y ) + tile_idx.y ) * static_cast<int>( part.tile_dims.y )
    );
}

/**
 * @brief Radial position of a particle, in cell units (r / Δr)
 * 
 * Cell j is centred at r = j Δr, so the axial cell (j = 0) spans
 * [-Δr/2, Δr/2].
 * 
 * @param shift     Global cell offset of the tile (see tile_shift())
 * @param cell      Cell (tile coordinates)
 * @param pos       Position inside the cell, in [-0.5, 0.5)
 * @return double   r / Δr
 */
inline double part_r( int2 const shift, int2 const cell, float2 const pos ) {
    return ( shift.y + cell.y ) + double( pos.y );
}

/**
 * @brief Injection test for mask profiles
 * 
 * A particle is injected if it lies strictly off axis (r > 0) and inside the
 * mask. Used by both inject_mask() and np_mask(), so that the particle count
 * always matches the particles injected.
 * 
 * @tparam mask_t   Mask type
 * @param mask      Injection mask
 * @param shift     Global cell offset of the tile (see tile_shift())
 * @param cell      Cell (tile coordinates)
 * @param pos       Position inside the cell
 * @return          true if the particle is to be injected
 */
template < class mask_t >
inline bool inject_test( mask_t const & mask, int2 const shift, int2 const cell, float2 const pos ) {
    return ( part_r( shift, cell, pos ) > 0 ) && mask( shift, cell, pos );
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
    uint3 const ppc,
    const float q0,
    double const dr,
    cyl_part::particles_view const part )
{
    int2 const tile_dims = make_int2( part.tile_dims.x, part.tile_dims.y );
    int const tile_id = tile_idx.y * static_cast<int>( part.local_ntiles.x ) + tile_idx.x;

    tile_range r;
    if ( ! clip_range( range, tile_idx, tile_dims, r ) ) return;

    // Number of particles before injection
    int np = part.tile_np[ tile_id ];

    int const offset = part.tile_offset[ tile_id ];
    int2   * __restrict__ const ix = &part.ix[ offset ];
    float2 * __restrict__ const x  = &part.x[ offset ];
    float3 * __restrict__ const u  = &part.u[ offset ];
    float  * __restrict__ const q  = &part.q[ offset ];
    float2 * __restrict__ const th = &part.th[ offset ];

    double const dpcz = 1.0 / ppc.x;
    double const dpcr = 1.0 / ppc.y;

    /// @brief angular positions of particles
    float2 pos_th[ ppc.z ];
    pos_th[0] = { 1, 0 };
    
    if ( ppc.z > 1 ) {
        const float Δt = ( 2 * M_PI ) / ppc.z;
        for( unsigned i = 1; i < ppc.z; i ++) {
            pos_th[i] = { std::cos( i * Δt ), std::sin( i * Δt ) };
        }
    }

    // Charge normalization
    double const qnorm =  q0 * dr;

    /// @brief radial position shift for this tile
    const int shiftr = ( part.local_tile_start.y + tile_idx.y ) * tile_dims.y;

    // If range include radial axis handle it separately
    bool axis = ( r.rj0 == 0 && shiftr == 0 );
    if (axis) {
        r.rj0 = 1;
        r.vol -= r.row;
    }

    // Particles are processed in flat loops (same as the CUDA version, where
    // this keeps all threads busy even when the injection range is small).
    // The flat index is n = ppc_idx * ncells + cell_idx,
    // i.e., the particle-in-cell index is the slowest, so that contiguous
    // particles are in different cells (same ordering as the original nested
    // loop version), with ppc_idx = ( ith * nr + ir ) * ppc.x + iz

    // Inject particles outside axial cell
    {
        int const total = r.vol * static_cast<int>( ppc.x * ppc.y * ppc.z );

        for( int n = 0; n < total; n ++ ) {
            int const ppc_idx = n / r.vol;
            int const idx     = n - ppc_idx * r.vol;

            unsigned const iz  = ppc_idx % ppc.x;
            unsigned const t   = ppc_idx / ppc.x;
            unsigned const ir  = t % ppc.y;
            unsigned const ith = t / ppc.y;

            float2 const pos = cell_pos( iz, ir, dpcz, dpcr );
            int2 const cell  = range_cell( r, idx );

            // Radial position (cell units), the axial cell is handled below
            double const rpos = ( shiftr + cell.y ) + double( pos.y );

            int const part_idx = np + n;
            ix[ part_idx ] = cell;
            x[ part_idx ]  = pos;
            u[ part_idx ]  = make_float3( 0, 0, 0 );
            q [ part_idx ] = rpos * qnorm;
            th[ part_idx ] = pos_th[ ith ];
        }

        // add particles already injected in tile
        np += total;
    }

    // Axial cells get special treatment
    if ( axis ) {
        
        // Additional charge correction for particles on axial cell
        // see notes
        double α, roff;
        if ( ppc.y % 2 == 0) {
            α = q0 * ( 2. * ( ppc.y*ppc.y - 1) ) / ( 3 * ( ppc.y * ppc.y ) );
            roff = 0.5;
        } else {
            α  = q0 * (2. / 3.);
            roff = 1;
        }

        // Only particles with r > 0 are injected in the axial cell
        unsigned const nr_axis = ppc.y / 2;
        int const total = r.row * static_cast<int>( ppc.x * nr_axis * ppc.z );

        for( int n = 0; n < total; n ++ ) {
            int const ppc_idx  = n / r.row;
            int const grid_idx = n - ppc_idx * r.row;

            unsigned const iz  = ppc_idx % ppc.x;
            unsigned const t   = ppc_idx / ppc.x;
            unsigned const ir  = t % nr_axis;
            unsigned const ith = t / nr_axis;

            float2 const pos = make_float2(
                dpcz * ( iz + 0.5 ) - 0.5,
                dpcr * ( ir + roff )
            );

            int2 const cell = make_int2( r.ri0 + grid_idx, 0 );

            int const part_idx = np + n;
            ix[ part_idx ] = cell;
            x [ part_idx ] = pos;
            u [ part_idx ] = make_float3(0,0,0);
            q [ part_idx ] = α * pos.y * dr;
            th[ part_idx ] = pos_th[ ith ];
        }
        np += total;
    }

    // Update global number of particles in tile
    part.tile_np[ tile_id ] = np;
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
    uint3 const ppc,
    cyl_part::particles_view const part,
    int * tile_np )
{
    int2 const tile_dims = make_int2( part.tile_dims.x, part.tile_dims.y );
    int const tile_id = tile_idx.y * static_cast<int>( part.local_ntiles.x ) + tile_idx.x;

    tile_range r;
    if ( clip_range( range, tile_idx, tile_dims, r  ) ) {
        bool axis = ( r.rj0 == 0 && ( part.local_tile_start.y + tile_idx.y ) == 0 );
        if ( ! axis ) {
            tile_np[ tile_id ] = r.vol * static_cast<int>( ppc.x * ppc.y * ppc.z );
        } else {
            tile_np[ tile_id ] = r.row * static_cast<int>( ppc.x * ppc.z * ( (r.rj1-r.rj0) * ppc.y + ppc.y / 2 ) );
        }
    } else {
        tile_np[ tile_id ] = 0;
    }
}

} // namespace kernel

}

/**
 * @brief Injects a uniform density profile
 * 
 * @param part      Particle data object
 * @param norm      Charge normalization constant
 * @param ppc       Number of particles per cell (z,r,θ)
 * @param dx        Cell size (z,r)
 * @param ref       Position of local grid on global simulation box in simulation
 *                  units (z,r)
 * @param range     Cell range in which to inject
 */
void density::uniform::inject( cyl_part::particles & part, float const norm, 
    uint3 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range ) const
{
    /// @brief single particle charge
    float q0 = norm * n0 / (ppc.x*ppc.y*ppc.z);

    auto const view = part.view();
    for_each_tile( view, [&]( int2 const tile_idx ) {
        kernel::inject_uniform( tile_idx, 
            range, ppc, q0, dx.y, view );
    });
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
void density::uniform::np_inject( cyl_part::particles & part, 
    uint3 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range,
    int * np ) const
{
    auto const view = part.view();
    for_each_tile( view, [&]( int2 const tile_idx ) {
        kernel::np_uniform( tile_idx, range, ppc, part.view(), np );
    });
}

namespace {
namespace kernel {

template < coord::cyl dir >
struct step_mask {
    /// @brief Step position (normalized to node grid coordinates)
    float step;

    int operator()( const int2 shift, const int2 cell, const float2 pos ) const {
        float t;
        if constexpr ( dir == coord::cyl::z ) t = (shift.x + cell.x) + (pos.x + 0.5);
        if constexpr ( dir == coord::cyl::r ) t = (shift.y + cell.y) + pos.y;

        return (t >= step);
    }
};

template < coord::cyl dir >
struct slab_mask {
    /// @brief Slab start position (normalized to node grid coordinates)
    float start;
    /// @brief Slab end position (normalized to node grid coordinates)
    float finish;

    int operator()( const int2 shift, const int2 cell, const float2 pos ) const {
        float s;
        if constexpr ( dir == coord::cyl::z ) s = (shift.x + cell.x) + (pos.x + 0.5);
        if constexpr ( dir == coord::cyl::r ) s = (shift.y + cell.y) + pos.y;
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
        double gz = ops::fma((shift.x + cell.x) + (pos.x+0.5), 
            static_cast<double>(dx.x),
            -static_cast<double>(center.x));
        double gr = ops::fma((shift.y + cell.y) + static_cast<double>(pos.y), 
            static_cast<double>(dx.y),
            -static_cast<double>(center.y));
        return ops::fma( gz, gz, gr*gr) < r2;
    }
};

/**
 * @brief Kernel for injecting particles using a mask profile
 * 
 * @details All candidate particles in the tile (cells in range x particles per
 *          cell) are processed in a single flat loop (in the CUDA version this
 *          keeps all threads busy even when the injection range is small,
 *          e.g. a single cell column when a moving window advances).
 * 
 *          The flat candidate index is `n = ppc_idx * vol + cell_idx`, with
 *          `ppc_idx = ( ith * ppc.y + ir ) * ppc.x + iz`, i.e. the
 *          particle-in-cell index is the slowest, which reproduces the
 *          particle ordering of the original (nested loop) version.
 * 
 *          Particle positions in the buffer are deterministic: each injected
 *          particle is stored at the position given by the number of injected
 *          candidates with a lower flat index. Here this is simply the
 *          generation order; the CUDA version obtains the same order with a
 *          ballot-based block scan.
 * 
 * @tparam mask_t       Mask type
 * @param tile_idx      Tile index
 * @param mask          Injection mask
 * @param range         Cell range (local) to inject particles in
 * @param ppc           Number of particles per cell
 * @param q0            Reference charge
 * @param dr            Radial cell size
 * @param part          Particle data
 */
template < class mask_t >
void inject_mask( 
    int2 const tile_idx,
    mask_t mask,
    bounds_2d<unsigned int> range,
    const uint3 ppc,
    const float q0,
    double const dr,
    cyl_part::particles_view const part )
{
    int2 const tile_dims = make_int2( part.tile_dims.x, part.tile_dims.y );
    int const tile_id = tile_idx.y * static_cast<int>( part.local_ntiles.x ) + tile_idx.x;

    // Block-uniform early exit
    tile_range local_range;
    if ( ! clip_range( range, tile_idx, tile_dims, local_range ) ) return;

    // Total number of candidate particles in tile
    int const total = local_range.vol * static_cast<int>( ppc.x * ppc.y * ppc.z );

    // Block-uniform early exit (e.g. ppc = 0)
    if ( total == 0 ) return;

    // Number of particles before injection
    int np = part.tile_np[ tile_id ];

    /// @brief angular positions of particles
    float2 pos_th[ ppc.z ];
    pos_th[0] = { 1, 0 };

    if ( ppc.z > 1 ) {
        const float Δt = ( 2 * M_PI ) / ppc.z;
        for( unsigned i = 1; i < ppc.z; i ++ ) {
            pos_th[i] = { std::cos( i * Δt ), std::sin( i * Δt ) };
        }
    }

    int const offset =  part.tile_offset[ tile_id ];
    int2   * __restrict__ ix = &part.ix[ offset ];
    float2 * __restrict__ x  = &part.x[ offset ];
    float3 * __restrict__ u  = &part.u[ offset ];
    float  * __restrict__ const q  = &part.q[ offset ];
    float2 * __restrict__ const th = &part.th[ offset ];

    double const dpcz = 1.0 / ppc.x;
    double const dpcr = 1.0 / ppc.y;

    // Position in global tile array
    int2 const shift = tile_shift( part, tile_idx );

    // Charge normalization
    double const qnorm =  q0 * dr;
    /// @brief Axial cell charge correction (same as inject_uniform)
    double const α = ( ppc.y % 2 == 0) ? 
        ( 2 * ( ppc.y*ppc.y - 1.) ) / ( 3 * ( ppc.y * ppc.y ) ) :
        (2. / 3.);

    /// @brief radial position shift for this tile
    const int shiftr = shift.y;

    for( int n = 0; n < total; n++ ) {

        // Particle-in-cell index (slowest) and cell index (fastest)
        int const ppc_idx  = n / local_range.vol;
        int const cell_idx = n - ppc_idx * local_range.vol;

        unsigned const iz = ppc_idx % ppc.x;
        unsigned const t  = ppc_idx / ppc.x;
        unsigned const ir = t % ppc.y;
        unsigned ith = t / ppc.y;

        float2 pos  = cell_pos( iz, ir, dpcz, dpcr );
        int2 cell = range_cell( local_range, cell_idx );
        int inj  = inject_test( mask, shift, cell, pos );


        if ( inj ) {
            // Radial position (cell units)
            double const rpos = part_r( shift, cell, pos );
            ix[ np ] = cell;
            x [ np ]  = pos;
            u [ np ]  = make_float3(0,0,0);
            q [ np ] = rpos * ( (shiftr + cell.y == 0) ? α * qnorm : qnorm );
            th[ np ] = pos_th[ ith ];
            np ++;
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
    mask_t mask,
    bounds_2d<unsigned int> range,
    const uint3 ppc,
    cyl_part::particles_view const part, int * tile_np )
{
    int2 const tile_dims = make_int2( part.tile_dims.x, part.tile_dims.y );
    int const tile_id = tile_idx.y * static_cast<int>( part.local_ntiles.x ) + tile_idx.x;

    int inj_np = 0;

    tile_range r;
    if ( clip_range( range, tile_idx, tile_dims, r ) ) {

        double const dpcz = 1.0 / ppc.x;
        double const dpcr = 1.0 / ppc.y;

        int2 const shift = tile_shift( part, tile_idx );

        for( int idx = 0; idx < r.vol; idx++ ) {
            int2 const cell = range_cell( r, idx );
            for( unsigned ith = 0; ith < ppc.z; ith++ ) {
                for( unsigned ir = 0; ir < ppc.y; ir++ ) {
                    for( unsigned iz = 0; iz < ppc.x; iz++ ) {
                        float2 const pos = cell_pos( iz, ir, dpcz, dpcr );
                        inj_np += inject_test( mask, shift, cell, pos );
                    }
                }
            }
        }
    }

    tile_np[ tile_id ] = inj_np;
}

} // namespace kernel

}

/**
 * @brief Injects a step density profile
 * 
 * @param part      Particle data object
 * @param norm      Charge normalization constant
 * @param ppc       Number of particles per cell
 * @param dx        Cell size
 * @param ref       Position of local grid on global simulation box in simulation
 *                  units
 * @param range     Cell range in which to inject
 */
void density::step::inject( cyl_part::particles & part, float const norm,
    uint3 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range ) const
{    
    /// @brief single particle charge
    auto q0 = norm * n0 / (ppc.x*ppc.y*ppc.z);

    auto view = part.view();

    switch( dir ) {
    case( coord::cyl::z ): {
        kernel::step_mask<coord::cyl::z> const mask { (pos - ref.x) / dx.x };
        for_each_tile( view, [&]( int2 const tile_idx ) {
            kernel::inject_mask( tile_idx, mask, range, ppc, q0, dx.y, view );
        });
        break; }
    case( coord::cyl::r ): {
        kernel::step_mask<coord::cyl::r> const mask { (pos - ref.y) / dx.y };
        for_each_tile( view, [&]( int2 const tile_idx ) {
            kernel::inject_mask( tile_idx, mask, range, ppc, q0, dx.y, view );
        });
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
void density::step::np_inject( cyl_part::particles & part, 
    uint3 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range,
    int * np ) const
{
    auto view = part.view();
    switch( dir ) {
    case( coord::cyl::z ): {
        kernel::step_mask<coord::cyl::z> const mask { (pos - ref.x) / dx.x };
        for_each_tile( view, [&]( int2 const tile_idx ) {
            kernel::np_mask( tile_idx, mask, range, ppc, view, np );
        });
        break; }
    case( coord::cyl::r ): {
        kernel::step_mask<coord::cyl::r> const mask { (pos - ref.y) / dx.y };
        for_each_tile( view, [&]( int2 const tile_idx ) {
            kernel::np_mask( tile_idx, mask, range, ppc, view, np );
        });
        break; }
    default:
        mpi::fatal( "Invalid direction for step density profile");
    }
}



/**
 * @brief Injects a slab density profile
 * 
 * @param part      Particle data object
 * @param norm      Charge normalization constant
 * @param ppc       Number of particles per cell
 * @param dx        Cell size
 * @param ref       Position of local grid on global simulation box in simulation
 *                  units
 * @param range     Cell range in which to inject
 */
void density::slab::inject( cyl_part::particles & part, float const norm,
    uint3 const ppc,float2 const dx, float2 const ref, bounds_2d<unsigned int> range ) const
{
    /// @brief single particle charge
    auto q0 = norm * n0 / (ppc.x*ppc.y*ppc.z);

    auto view = part.view();

    switch( dir ) {
    case( coord::cyl::z ): {
        kernel::slab_mask<coord::cyl::z> const mask { 
            (begin - ref.x)/ dx.x,
            (end - ref.x)/ dx.x };
        for_each_tile( view, [&]( int2 const tile_idx ) {
            kernel::inject_mask( tile_idx, mask, range, ppc, q0, dx.y, view );
        });
        break; }
    case( coord::cyl::r ): {
        kernel::slab_mask<coord::cyl::r> const mask { 
            (begin - ref.y)/ dx.y,
            (end - ref.y)/ dx.y };
        for_each_tile( view, [&]( int2 const tile_idx ) {
            kernel::inject_mask( tile_idx, mask, range, ppc, q0, dx.y, view );
        });
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
void density::slab::np_inject( cyl_part::particles & part, 
    uint3 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range,
    int * np ) const
{
    auto view = part.view();

    switch( dir ) {
    case( coord::cyl::z ): {
        kernel::slab_mask<coord::cyl::z> const mask { 
            (begin - ref.x)/ dx.x,
            (end - ref.x)/ dx.x };
        for_each_tile( view, [&]( int2 const tile_idx ) {
            kernel::np_mask( tile_idx, mask, range, ppc, view, np );
        });
        break; }
    case( coord::cyl::r ): {
        kernel::slab_mask<coord::cyl::r> const mask { 
            (begin - ref.y)/ dx.y,
            (end - ref.y)/ dx.y };
        for_each_tile( view, [&]( int2 const tile_idx ) {
            kernel::np_mask( tile_idx, mask, range, ppc, view, np );
        });
        break; }
    default:
        mpi::fatal( "Invalid direction for slab density profile");
    }
}

/**
 * @brief Injects a sphere density profile
 * 
 * @param part      Particle data object
 * @param norm      Charge normalization constant
 * @param ppc       Number of particles per cell
 * @param dx        Cell size
 * @param ref       Position of local grid on global simulation box in simulation
 *                  units
 * @param range     Cell range in which to inject
 */
void density::sphere::inject( cyl_part::particles & part, float const norm,
    uint3 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range ) const
{
    /// @brief single particle charge
    auto q0 = norm * n0 / (ppc.x*ppc.y*ppc.z);

    kernel::sphere_mask const mask {
        center - ref,
        radius * radius,
        dx
    };

    auto view = part.view();
    for_each_tile( view, [&]( int2 const tile_idx ) {
        kernel::inject_mask( tile_idx, mask, range, ppc, q0, dx.y, view );
    });
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
void density::sphere::np_inject( cyl_part::particles & part, 
    uint3 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range,
    int * np ) const
{
    kernel::sphere_mask const mask {
        center - ref,
        radius * radius,
        dx
    };

    auto view = part.view();
    for_each_tile( view, [&]( int2 const tile_idx ) {
        kernel::np_mask( tile_idx, mask, range, ppc, view, np );
    });
}

/**
 * @brief Returns number of particles per tile that the none profile would
 *        inject (always 0)
 * 
 * @param part      Particle data object
 * @param ppc       Number of particles per cell
 * @param dx        Cell size
 * @param ref       Position of local grid on global simulation box in simulation
 *                  units
 * @param range     Cell range in which to inject
 * @param np        (out) Number of particles to inject per tile
 */
void density::none::np_inject( cyl_part::particles & part, 
    uint3 const, float2 const, float2 const, bounds_2d<unsigned int>,
    int * np ) const
{
    auto ntiles = part.get_local_ntiles();
    memory::zero( np, ntiles.x * ntiles.y );
}
