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
dim3 inject_block( cyl_part::particles & part ) {
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
__device__ 
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
__device__ 
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
 * @param range     Cell range (local) to inject particles in
 * @param ppc       Number of particles per cell 
 * @param q0        Reference charge
 * @param dr        Radial cell size
 * @param part      Particle data 
 */
__global__
inline void inject_uniform( 
    bounds_2d<unsigned int> range,
    uint3 const ppc,
    const float q0,
    double const dr,
    cyl_part::particles_view const part )
{
    int2 const tile_idx  = make_int2( blockIdx.x, blockIdx.y );
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
    float2 * pos_th = gpu::block::shared_mem<float2>();
    if ( gpu::block::thread_rank() == 0 )
        pos_th[0] = { 1, 0 };
    
    if ( ppc.z > 1 ) {
        const float Δt = ( 2 * M_PI ) / ppc.z;
        for( int i = gpu::block::thread_rank() + 1; i < ppc.z; i += gpu::block::num_threads()) {
            pos_th[i] = { std::cos( i * Δt ), std::sin( i * Δt ) };
        }
    }
    gpu::block::sync();

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

    // Inject particles outside axial cell
    for( unsigned ith = 0; ith < ppc.z; ith++ ) {
        for( unsigned ir = 0; ir < ppc.y; ir++ ) {
            for( unsigned iz = 0; iz < ppc.x; iz++ ) {
                float2 const pos = cell_pos( iz, ir, dpcz, dpcr );
                int const ppc_idx = ( ith * ppc.y + ir ) * ppc.x + iz;

                // Each thread takes 1 cell
                for( int idx = gpu::block::thread_rank(); idx < r.vol; idx += gpu::block::num_threads() ) {
                    int const part_idx = np + r.vol * ppc_idx + idx;
                    
                    int2 const cell = range_cell( r, idx );

                    // Radial position (cell units), the axial cell is handled below
                    double const rpos = ( shiftr + cell.y ) + double( pos.y );

                    ix[ part_idx ] = cell;
                    x[ part_idx ]  = pos;
                    u[ part_idx ]  = make_float3( 0, 0, 0 );
                    q [ part_idx ] = rpos * qnorm;
                    th[ part_idx ] = pos_th[ ith ];
                }
            }
        }
    }
    
    // add particles already injected in tile
    np += r.vol * ppc.z * ppc.y * ppc.x ;

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

        for( unsigned ith = 0; ith < ppc.z; ith++ ) {
            for( unsigned ir = 0; ir < ppc.y/2; ir++ ) {
                for( unsigned iz = 0; iz < ppc.x; iz++) {
                    float2 const pos = make_float2(
                        dpcz * ( iz + 0.5 ) - 0.5,
                        dpcr * ( ir + roff )
                    );

                    int ppc_idx = ( ith * (ppc.y/2) + ir ) * ppc.x + iz;

                    for( int grid_idx = gpu::block::thread_rank(); grid_idx < r.row; grid_idx += gpu::block::num_threads() ) {
                        int2 const cell = make_int2( grid_idx, 0 );

                        int part_idx = np + r.row * ppc_idx + grid_idx;

                        ix[ part_idx ] = cell;
                        x [ part_idx ] = pos;
                        u [ part_idx ] = make_float3(0,0,0);
                        q [ part_idx ] = α * pos.y * dr;
                        th[ part_idx ] = pos_th[ ith ];
                    }
                }
            }
        }
        np += r.row * ppc.z * ( ppc.y / 2 ) * ppc.x;
    }

    // All threads have read tile_np[ tile_id ] before it is updated
    gpu::block::sync();

    // Update global number of particles in tile
    if ( gpu::block::thread_rank() == 0 ) {
        part.tile_np[ tile_id ] = np;
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
    uint3 const ppc,
    cyl_part::particles_view const part,
    int * tile_np )
{
    int2 const tile_idx  = make_int2( blockIdx.x, blockIdx.y );
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

    auto ntiles = part.get_local_ntiles();
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block= inject_block( part );
    auto shm_size = ppc.z * sizeof( float2 );
    kernel::inject_uniform <<< grid, block, shm_size >>> (
        range, ppc, q0, dx.y, part.view()
    );
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
    auto ntiles = part.get_local_ntiles();
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 1 ); // Use only 1 thread per tile

    kernel::np_uniform<<< grid, block >>> ( range, ppc, part.view(), np );
}

namespace kernel {

template < coord::cyl dir >
struct step_mask {
    /// @brief Step position (normalized to node grid coordinates)
    float step;

    __device__ 
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

    __device__ 
    int operator()( const int2 shift, const int2 cell, const float2 pos ) const {
        float s;
        if constexpr ( dir == coord::cyl::z ) s = (shift.x + cell.x) + (pos.x + 0.5);
        if constexpr ( dir == coord::cyl::r ) s = (shift.y + cell.y) + pos.y;
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
        double gz = fma((shift.x + cell.x) + (pos.x+0.5), 
            static_cast<double>(dx.x),
            -static_cast<double>(center.x));
        double gr = fma((shift.y + cell.y) + static_cast<double>(pos.y), 
            static_cast<double>(dx.y),
            -static_cast<double>(center.y));
        return fma( gz, gz, gr*gr) < r2;
    }
};

/**
 * @brief Kernel for injecting a masked (step / slab / sphere) uniform density
 * 
 * @note Requires ppc.z * sizeof(float2) bytes of dynamic shared memory
 * 
 * @tparam mask_t       Mask type
 * @param mask          Injection mask
 * @param range         Cell range (local) to inject particles in
 * @param ppc           Number of particles per cell
 * @param q0            Reference charge
 * @param dr            Radial cell size
 * @param part          Particle data
 */
template < class mask_t >
__global__
void inject_mask( 
    mask_t mask,
    bounds_2d<unsigned int> range,
    const uint3 ppc,
    const float q0,
    double const dr,
    cyl_part::particles_view const part )
{
    int2 const tile_idx  = make_int2( blockIdx.x, blockIdx.y );
    int2 const tile_dims = make_int2( part.tile_dims.x, part.tile_dims.y );
    int const tile_id = tile_idx.y * static_cast<int>( part.local_ntiles.x ) + tile_idx.x;

    // Block-uniform early exit
    tile_range local_range;
    if ( ! clip_range( range, tile_idx, tile_dims, local_range ) ) return;

    int const thread_rank = gpu::block::thread_rank();
    int const num_threads = gpu::block::num_threads();

    // Store number of particles before injection
    __shared__ int np_tile;
    /// @brief angular positions of particles
    float2 * pos_th = gpu::block::shared_mem<float2>();

    if ( thread_rank == 0 ) {
        np_tile = part.tile_np[ tile_id ];
        pos_th[0] = { 1, 0 };
    }

    if ( ppc.z > 1 ) {
        const float Δt = ( 2 * M_PI ) / ppc.z;
        for( int i = gpu::block::thread_rank() + 1; i < ppc.z; i += gpu::block::num_threads()) {
            pos_th[i] = { cos( i * Δt ), sin( i * Δt ) };
        }
    }
    gpu::block::sync();

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

    // Ensure uniform trip count for all threads
    int const niter = ( local_range.vol + num_threads - 1 ) / num_threads;

    // Charge normalization
    double const qnorm =  q0 * dr;
    /// @brief Axial cell charge correction (same as inject_uniform)
    double const α = ( ppc.y % 2 == 0) ? 
        ( 2 * ( ppc.y*ppc.y - 1.) ) / ( 3 * ( ppc.y * ppc.y ) ) :
        (2. / 3.);

    /// @brief radial position shift for this tile
    const int shiftr = shift.y;

    for( unsigned ith = 0; ith < ppc.z; ith++ ) {
        for( unsigned ir = 0; ir < ppc.y; ir++ ) {
            for( unsigned iz = 0; iz < ppc.x; iz++) {
                float2 const pos = cell_pos( iz, ir, dpcz, dpcr );

                for( int it = 0; it < niter; it++ ) {
                    int const idx = it * num_threads + thread_rank;

                    int2 cell = make_int2( 0, 0 );
                    int inj = 0;

                    if ( idx < local_range.vol ) {
                        cell = range_cell( local_range, idx );
                        inj = inject_test( mask, shift, cell, pos );
                    }

                    // All threads must have read np_tile before it is updated
                    int base = np_tile;

                    // Called by every thread in the block (implicit sync)
                    int const off = gpu::block::exscan_add( inj );

                    if ( inj ) {
                        const int k = base + off;
                        // Radial position (cell units)
                        double const rpos = part_r( shift, cell, pos );
                        ix[ k ] = cell;
                        x[ k ]  = pos;
                        u[ k ]  = make_float3(0,0,0);
                        q [ k ] = rpos * ( (shiftr + cell.y == 0) ? α * qnorm : qnorm );
                        th[ k ] = pos_th[ ith ];
                    }

                    // The last thread's exclusive scan + own value is the block total
                    if ( thread_rank == num_threads - 1 ) np_tile += off + inj;
                    gpu::block::sync();
                }
            }
        }
    }

    if ( thread_rank == 0 ) 
        part.tile_np[ tile_id ] = np_tile;
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
    const uint3 ppc,
    cyl_part::particles_view const part, int * tile_np )
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

        double const dpcz = 1.0 / ppc.x;
        double const dpcr = 1.0 / ppc.y;

        int2 const shift = tile_shift( part, tile_idx );

        // No collectives inside this loop, so a non-uniform trip count is fine
        for( int idx = thread_rank; idx < r.vol; idx += num_threads ) {
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

    // Called by every thread in the block
    inj_np = gpu::warp::reduce_add( inj_np );
    if ( gpu::warp::thread_rank() == 0 ) {
        gpu::block::atomic_fetch_add( &np_tile, inj_np );
    }
    gpu::block::sync();

    if ( thread_rank == 0 )
        tile_np[ tile_id ] = np_tile;
}

} // namespace kernel


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

    auto ntiles = part.get_local_ntiles();
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block = inject_block( part );
    // Angular positions of particles (see inject_mask)
    auto const shm_size = ppc.z * sizeof( float2 );

    switch( dir ) {
    case( coord::cyl::z ): {
        kernel::step_mask<coord::cyl::z> const mask { (pos - ref.x) / dx.x };
        kernel::inject_mask <<< grid, block, shm_size >>> ( mask, range, ppc, q0, dx.y, part.view() );
        break; }
    case( coord::cyl::r ): {
        kernel::step_mask<coord::cyl::r> const mask { (pos - ref.y) / dx.y };
        kernel::inject_mask <<< grid, block, shm_size >>> ( mask, range, ppc, q0, dx.y, part.view() );
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
    auto ntiles = part.get_local_ntiles();
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block = inject_block( part );

    switch( dir ) {
    case( coord::cyl::z ): {
        kernel::step_mask<coord::cyl::z> const mask { (pos - ref.x) / dx.x };
        kernel::np_mask <<< grid, block >>> ( mask, range, ppc, part.view(), np );
        break; }
    case( coord::cyl::r ): {
        kernel::step_mask<coord::cyl::r> const mask { (pos - ref.y) / dx.y };
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

    auto ntiles = part.get_local_ntiles();
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block = inject_block( part );
    // Angular positions of particles (see inject_mask)
    auto const shm_size = ppc.z * sizeof( float2 );

    switch( dir ) {
    case( coord::cyl::z ): {
        kernel::slab_mask<coord::cyl::z> const mask { 
            (begin - ref.x)/ dx.x,
            (end - ref.x)/ dx.x };
        kernel::inject_mask <<< grid, block, shm_size >>> ( mask, range, ppc, q0, dx.y, part.view() );
        break; }
    case( coord::cyl::r ): {
        kernel::slab_mask<coord::cyl::r> const mask { 
            (begin - ref.y)/ dx.y,
            (end - ref.y)/ dx.y };
        kernel::inject_mask <<< grid, block, shm_size >>> ( mask, range, ppc, q0, dx.y, part.view() );
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
void density::slab::np_inject( cyl_part::particles & part, 
    uint3 const ppc, float2 const dx, float2 const ref, bounds_2d<unsigned int> range,
    int * np ) const
{

    auto ntiles = part.get_local_ntiles();
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block = inject_block( part );

    switch( dir ) {
    case( coord::cyl::z ): {
        kernel::slab_mask<coord::cyl::z> const mask { 
            (begin - ref.x)/ dx.x,
            (end - ref.x)/ dx.x };
        kernel::np_mask <<< grid, block >>> ( mask, range, ppc, part.view(), np
        );
        break; }
    case( coord::cyl::r ): {
        kernel::slab_mask<coord::cyl::r> const mask { 
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

    auto ntiles = part.get_local_ntiles();
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block = inject_block( part );
    // Angular positions of particles (see inject_mask)
    auto const shm_size = ppc.z * sizeof( float2 );

    kernel::sphere_mask mask {
        center - ref,
        radius * radius,
        dx
    };

    kernel::inject_mask <<< grid, block, shm_size >>> ( 
        mask, range, ppc, q0, dx.y, part.view()
    );
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
    gpu::device::zero( np, ntiles.x * ntiles.y );
}
