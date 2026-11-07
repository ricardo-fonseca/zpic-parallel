#include "current.hpp"
#include "util/complex.hpp"

#include <string>
#include <type_traits>


namespace kernel {

namespace current_bc {

/**
 * @brief Normal / tangential current components for walls normal to z or r
 *
 * @note The accessors are templates so the same wall code works for real
 *       (mode 0) and complex (modes m > 0) grids
 */
template< coord::cyl dir > struct comp;

template<> struct comp< coord::cyl::z > {
    template< typename T > __device__ static T & n ( cyl3<T> & v ) { return v.z; }
    template< typename T > __device__ static T & t1( cyl3<T> & v ) { return v.r; }
    template< typename T > __device__ static T & t2( cyl3<T> & v ) { return v.th; }
};

template<> struct comp< coord::cyl::r > {
    template< typename T > __device__ static T & n ( cyl3<T> & v ) { return v.r; }
    template< typename T > __device__ static T & t1( cyl3<T> & v ) { return v.z; }
    template< typename T > __device__ static T & t2( cyl3<T> & v ) { return v.th; }
};

/**
 * @brief Lower wall (pec / pmc) boundary condition for one line of cells
 *
 * The wall sits on node 0. Layout along the line (stride s):
 *
 *  p[-s]   normal: -1/2  tangential: -1 (guard)
 *  p[ 0]   normal: +1/2  tangential: wall node
 *  p[+s]   normal: +3/2  tangential: +1
 *
 * Fold (particle reflection, same for pec and pmc): the current deposited
 * beyond the wall is mirrored into the domain, with the normal component
 * reversed. The deposit on the wall node is its own mirror and is unchanged.
 *
 * Images (wall type dependent):
 *  pec - opposite sign image charges: normal J even, tangential J odd, so
 *        the tangential J on the wall node is 0
 *  pmc - same sign image charges: normal J odd, tangential J even, so the
 *        tangential J on the wall node is doubled (fold step only)
 *
 * @note Walls do not involve θ, so the same rules apply to every mode
 *
 * @param p     Pointer to the wall node
 * @param s     Stride along the line normal to the wall (1 or ystride)
 * @param type  Wall type (pec or pmc)
 * @param fold  Fold guard cell current and correct wall node (once per step)
 */
template< coord::cyl dir, typename T >
__device__ void lower_conducting( cyl3<T> * const p, int const s, current::bc::type const type, bool const fold ) {
    using C = comp<dir>;
    const bool pec = ( type == current::bc::pec );

    if ( fold ) {
        C::n ( p[ 0 ] ) -= C::n ( p[ -s ] );
        C::t1( p[ s ] ) += C::t1( p[ -s ] );
        C::t2( p[ s ] ) += C::t2( p[ -s ] );

        if ( ! pec ) {
            C::t1( p[ 0 ] ) *= 2.0f;
            C::t2( p[ 0 ] ) *= 2.0f;
        }
    }

    if ( pec ) {
        C::t1( p[ 0 ] ) = T{};
        C::t2( p[ 0 ] ) = T{};
    }

    const float pn = pec ? +1.0f : -1.0f;   // normal component parity
    const float pt = -pn;                   // tangential component parity

    C::n ( p[ -s ] ) = pn * C::n ( p[ 0 ] );
    C::t1( p[ -s ] ) = pt * C::t1( p[ s ] );
    C::t2( p[ -s ] ) = pt * C::t2( p[ s ] );
}

/**
 * @brief Upper wall (pec / pmc) boundary condition for one line of cells
 *
 * The wall sits on node n (first upper guard cell). Layout along the line
 * (stride s):
 *
 *  p[-2s]  normal: n-3/2         tangential: n-2
 *  p[ -s]  normal: n-1/2         tangential: n-1
 *  p[  0]  normal: n+1/2 (guard) tangential: wall node (guard)
 *  p[ +s]  normal: n+3/2 (guard) tangential: n+1 (guard)
 *
 * See lower_conducting() for the fold / image rules.
 *
 * @param p     Pointer to the wall node
 * @param s     Stride along the line normal to the wall (1 or ystride)
 * @param type  Wall type (pec or pmc)
 * @param fold  Fold guard cell current and correct wall node (once per step)
 */
template< coord::cyl dir, typename T >
__device__ void upper_conducting( cyl3<T> * const p, int const s, current::bc::type const type, bool const fold ) {
    using C = comp<dir>;
    const bool pec = ( type == current::bc::pec );

    if ( fold ) {
        C::n ( p[  -s ] ) -= C::n ( p[ 0 ] );
        C::n ( p[ -2*s] ) -= C::n ( p[ s ] );
        C::t1( p[  -s ] ) += C::t1( p[ s ] );
        C::t2( p[  -s ] ) += C::t2( p[ s ] );

        if ( ! pec ) {
            C::t1( p[ 0 ] ) *= 2.0f;
            C::t2( p[ 0 ] ) *= 2.0f;
        }
    }

    if ( pec ) {
        C::t1( p[ 0 ] ) = T{};
        C::t2( p[ 0 ] ) = T{};
    }

    const float pn = pec ? +1.0f : -1.0f;   // normal component parity
    const float pt = -pn;                   // tangential component parity

    C::n ( p[ 0 ] ) = pn * C::n ( p[  -s  ] );
    C::n ( p[ s ] ) = pn * C::n ( p[ -2*s ] );
    C::t1( p[ s ] ) = pt * C::t1( p[  -s  ] );
    C::t2( p[ s ] ) = pt * C::t2( p[  -s  ] );
}

/**
 * @brief Axial (r = 0) boundary condition for one radial line of cells
 *
 * Radial layout (row j, stride s), cell j centred at r = j Δr:
 *
 *  p[ -s]  z, θ: r = -3/2 Δr (guard)   r: r = -Δr (guard)
 *  p[  0]  z, θ: r = -1/2 Δr           r: r = 0 (axis)
 *  p[ +s]  z, θ: r = +1/2 Δr           r: r = +Δr
 *  p[+2s]  z, θ: r = +3/2 Δr
 *
 * Mode m parity across the axis: (-1)^m for z, -(-1)^m for r and θ (the r̂
 * and θ̂ unit vectors flip). The r component on the axis is set to 0 when
 * its parity is odd (even m); for odd m it is left as computed.
 *
 * @param p     Pointer to row 0
 * @param s     Row stride (ystride)
 * @param m     Mode number
 * @param fold  `true`: fold the current deposited at r < 0 back into the
 *              domain, then set the images. `false`: only set the images.
 */
template< typename T >
__device__ void axial( cyl3<T> * const p, int const s, int const m, bool const fold ) {

    const float sz  = ( m & 1 ) ? -1.0f : +1.0f;    // z parity
    const float srt = -sz;                          // r, θ parity

    if ( fold ) {
        p[   s ].z  += sz  * p[  0 ].z;
        p[ 2*s ].z  += sz  * p[ -s ].z;

        p[   s ].th += srt * p[  0 ].th;
        p[ 2*s ].th += srt * p[ -s ].th;

        p[   s ].r  += srt * p[ -s ].r;
    }

    // Images below the axis
    p[  0 ].z  = sz  * p[   s ].z;
    p[ -s ].z  = sz  * p[ 2*s ].z;

    p[  0 ].th = srt * p[   s ].th;
    p[ -s ].th = srt * p[ 2*s ].th;

    p[ -s ].r  = srt * p[   s ].r;
    if ( srt < 0 ) p[ 0 ].r = T{};
}

/**
 * @brief True if t is a conducting wall (pec / pmc)
 */
__device__ inline bool conducting( const current::bc::type t ) {
    return t == current::bc::pec || t == current::bc::pmc;
}

/**
 * @brief Physical boundaries touching tile (tx, ty) on this rank
 */
struct tile_edges {
    current::bc::type z_lower, z_upper, r_upper;
    bool axis;

    __device__ tile_edges( const current::bc_type & local_bc,
        const unsigned tx, const unsigned ty, const uint2 ntiles ) :
        z_lower( ( tx == 0            && conducting( local_bc.x.lower ) ) ? local_bc.x.lower : current::bc::none ),
        z_upper( ( tx == ntiles.x - 1 && conducting( local_bc.x.upper ) ) ? local_bc.x.upper : current::bc::none ),
        r_upper( ( ty == ntiles.y - 1 && conducting( local_bc.y.upper ) ) ? local_bc.y.upper : current::bc::none ),
        axis   (   ty == 0            && local_bc.y.lower == current::bc::axial ) { }

    __device__ bool z_walls() const { return z_lower != current::bc::none || z_upper != current::bc::none; }
    __device__ bool r_edges() const { return r_upper != current::bc::none || axis; }
};

/**
 * @brief Apply the wall / axis boundary conditions to one tile
 *
 * Two stages, separated by a block synchronization:
 *
 *  1. z walls, on every extended row of the tile
 *  2. upper r wall and axis, on every extended column of the tile
 *
 * All operations are linear and act identically on every row / column, so
 * running the z stage first and then the r stage over the full extended
 * tile gives the correct corner values. Each tile's extended region
 * (including guard cells) is private to the tile, so different blocks never
 * touch the same memory.
 *
 * @note Within a stage, the lower and upper operations on a line are done
 *       by the same thread, so they never race even when a single tile
 *       holds both boundaries.
 *
 * @warning Must be called by all threads in the block
 *
 * @param tile      Pointer to the start of the tile buffer (extended region)
 * @param view      Grid view
 * @param edges     Physical boundaries touching this tile
 * @param m         Mode number
 * @param fold      See lower_conducting() / axial()
 */
template< typename T >
__device__ void tile_bc( cyl3<T> * const tile, const grid::tiled_cyl3_view<T> & view,
    const tile_edges & edges, const int m, const bool fold ) {

    const int ystride = view.tile_ystride();

    // Stage 1: z walls
    if ( edges.z_walls() ) {
        // ix = 0 (lower wall) and ix = nx (upper wall, first upper guard cell)
        cyl3<T> * const Jl = & tile[ view.gc.x.lower ];
        cyl3<T> * const Ju = & tile[ view.gc.x.lower + view.tile_dims.x ];

        for( unsigned idx = gpu::block::thread_rank(); idx < view.tile_ext_dims.y; idx += gpu::block::num_threads() ) {
            if ( edges.z_lower != current::bc::none )
                lower_conducting< coord::cyl::z >( & Jl[ idx * ystride ], 1, edges.z_lower, fold );
            if ( edges.z_upper != current::bc::none )
                upper_conducting< coord::cyl::z >( & Ju[ idx * ystride ], 1, edges.z_upper, fold );
        }
    }

    gpu::block::sync();

    // Stage 2: upper r wall and axis
    if ( edges.r_edges() ) {
        // iy = 0 (axis) and iy = ny (upper wall, first upper guard row)
        cyl3<T> * const Ja = & tile[ view.gc.y.lower * ystride ];
        cyl3<T> * const Ju = & tile[ ( view.gc.y.lower + view.tile_dims.y ) * ystride ];

        for( unsigned idx = gpu::block::thread_rank(); idx < view.tile_ext_dims.x; idx += gpu::block::num_threads() ) {
            if ( edges.r_upper != current::bc::none )
                upper_conducting< coord::cyl::r >( & Ju[ idx ], ystride, edges.r_upper, fold );
            if ( edges.axis )
                axial( & Ja[ idx ], ystride, m, fold );
        }
    }
}

} // namespace current_bc

/**
 * @brief Fold the current deposited beyond walls / axis and normalize for
 *        ring particles, mode m
 *
 * One block per tile:
 *
 *  1. Tiles on a physical boundary fold the raw deposit beyond the wall /
 *     axis back into the domain (current_bc::tile_bc() with fold = true).
 *     This is done before normalization so that each value is normalized at
 *     the position it is folded into.
 *  2. The full extended tile (including guard cells) is normalized. This
 *     covers the upper wall nodes, which sit on the first upper guard cell
 *     and are filtered like interior nodes; guard cells on internal
 *     boundaries are overwritten by the following copy_to_gc().
 *
 * Normalization (Jz, Jθ at the lower cell edge r = (j - 1/2) Δr, Jr at the
 * cell centre r = j Δr):
 *
 *  m = 0:  Jz *= (Δz/Δt) / r,      Jr *= (Δr/Δt) / r,      Jθ *= (Δr/Δt) / r
 *  m > 0:  Jz *= 2 (Δz/Δt) / r,    Jr *= 2 (Δr/Δt) / r,    Jθ *= 2 i / (m Δt)
 *
 * with Jr on the axis set to 0.
 *
 * @note Launch with grid( local_ntiles.x, local_ntiles.y )
 *
 * @tparam T            float (m = 0) or util::complex64 (m > 0)
 * @param J_grid        View of the mode m current grid
 * @param local_bc      Local rank boundary conditions
 * @param m             Mode number
 * @param dx            Cell size
 * @param dt            Time step
 */
template< typename T >
__global__
void current_fold_norm(
    const grid::tiled_cyl3_view<T> J_grid,
    const current::bc_type local_bc,
    const int m,
    float2 const dx, double const dt ) {

    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    cyl3<T> * const __restrict__ current = J_grid.tile_buffer( tile_idx );
    const int jstride = J_grid.tile_ystride();

    // Fold
    const current_bc::tile_edges edges( local_bc, tile_idx.x, tile_idx.y, J_grid.local_ntiles );
    current_bc::tile_bc( current, J_grid, edges, m, true );

    gpu::block::sync();

    // Normalize
    const float dr = dx.y;
    const float dz_dt = dx.x / dt;
    const float dr_dt = dx.y / dt;

    // Mode 0 is real, modes m > 0 carry an extra factor 2
    constexpr bool real_mode = std::is_floating_point_v< T >;
    const float fm = real_mode ? 1.0f : 2.0f;

    const auto ext_dims = J_grid.tile_ext_dims;
    const int j0 = ( J_grid.local_tile_start.y + tile_idx.y ) * J_grid.tile_dims.y
                   - static_cast<int>( J_grid.gc.y.lower );

    for( int idx = gpu::block::thread_rank(); idx < static_cast<int>( ext_dims.y * ext_dims.x ); idx += gpu::block::num_threads() ) {
        const int i = idx % ext_dims.x;
        const int j = idx / ext_dims.x;

        /// @brief r at center of cell (r position of Jr)
        const float rc = std::abs( j0 + j ) * dr;
        /// @brief r at lower edge of cell (r position of Jz, Jθ)
        const float rl = std::abs( j0 + j - 0.5f ) * dr;

        auto & Jc = current[ j * jstride + i ];

        Jc.z *= fm * dz_dt / rl;
        Jc.r *= ( j0 + j == 0 ) ? 0.0f : fm * dr_dt / rc;

        if constexpr ( real_mode ) {
            Jc.th *= dr_dt / rl;
        } else {
            Jc.th *= T( 0, static_cast<float>( 2 / ( m * dt ) ) );
        }
    }
}

/**
 * @brief Set the wall / axis image currents in the guard cells, mode m
 *
 * Idempotent; used after copy_to_gc() on the normalized current and after
 * every filter pass. Tiles not on a physical boundary exit immediately.
 *
 * @note Launch with grid( local_ntiles.x, local_ntiles.y )
 *
 * @param J_grid        View of the mode m current grid
 * @param local_bc      Local rank boundary conditions
 * @param m             Mode number
 */
template< typename T >
__global__
void current_images(
    const grid::tiled_cyl3_view<T> J_grid,
    const current::bc_type local_bc,
    const int m ) {

    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const current_bc::tile_edges edges( local_bc, tile_idx.x, tile_idx.y, J_grid.local_ntiles );

    // Uniform across the block, so the early exit is safe
    if ( ! edges.z_walls() && ! edges.r_edges() ) return;

    current_bc::tile_bc( J_grid.tile_buffer( tile_idx ), J_grid, edges, m, false );
}

} // namespace kernel

/**
 * @brief Fold the current deposited beyond walls / axis and normalize for
 *        ring particles, all modes
 *
 * @note Use exactly once per time step, on the raw deposit: after
 *       add_from_gc() and before copy_to_gc()
 *
 * @note Relies on add_from_gc() leaving the guard cells on non-periodic
 *       physical edges (including the axis) untouched, so that they still
 *       hold the current deposited beyond the wall / axis.
 */
void current::fold_and_normalize() {

    const dim3 block( 64 );
    const uint2 ntiles = J -> get_local_ntiles();
    const dim3 grid( ntiles.x, ntiles.y );

    kernel::current_fold_norm <<< grid, block >>> (
        mode0().view(), local_bc, 0, dx, dt
    );

    for( int m = 1; m < nmodes; m++ ) {
        kernel::current_fold_norm <<< grid, block >>> (
            mode(m).view(), local_bc, m, dx, dt
        );
    }
}

/**
 * @brief Set the wall / axis image currents in the guard cells, all modes
 *
 * @note Idempotent. Use after copy_to_gc() on the normalized current, and
 *       after every filter pass.
 */
void current::set_images() {

    // Nothing to do on ranks without physical (wall / axis) boundaries
    if ( ! is_conducting( local_bc.x.lower ) && ! is_conducting( local_bc.x.upper ) &&
         ! is_conducting( local_bc.y.upper ) && local_bc.y.lower != current::bc::axial ) return;

    const dim3 block( 64 );
    const uint2 ntiles = J -> get_local_ntiles();
    const dim3 grid( ntiles.x, ntiles.y );

    kernel::current_images <<< grid, block >>> (
        mode0().view(), local_bc, 0
    );

    for( int m = 1; m < nmodes; m++ ) {
        kernel::current_images <<< grid, block >>> (
            mode(m).view(), local_bc, m
        );
    }
}

/**
 * @brief Advance electric current to next iteration
 *
 * 1. Add up current deposited on guard cells (internal boundaries)
 * 2. Fold the current deposited beyond the walls / axis back into the domain
 *    (raw deposit) and normalize for ring particles
 * 3. Update guard cells and set the wall / axis images
 * 4. Apply filtering, refreshing the wall / axis images after every pass
 *
 */
void current::advance() {

    // Add up current deposited on guard cells
    J -> add_from_gc( );

    // Fold current deposited beyond walls / axis and normalize
    fold_and_normalize();

    // Update guard cell values with normalized current and set the images
    J -> copy_to_gc( );
    set_images();

    // Apply filtering
    apply_filter();

    // Advance iteration count
    iter++;

    // Moving window is checked after the iteration count is advanced so
    // that the window position matches the time label used by save()
    // (iter * dt). This only affects the axis range on output data and must
    // follow the same ordering as emf::advance()
    if ( mov_window.needs_move( iter * dt ) )
        mov_window.advance();
}

/**
 * @brief Save current density to file
 *
 * @param jc        Which current component to save (r, θ or z)
 * @param m         Mode
 */
void current::save( const fcomp::cyl jc, unsigned m ) {

    std::string vfname  = "J" + std::to_string(m);      // Dataset name
    std::string vflabel = "J^" + std::to_string(m) + "_";    // Dataset label (for plots)
    std::string path{"CURRENT"};

    switch ( jc ) {
        case( fcomp::cyl::z ) :
            vfname  += 'z';
            vflabel += 'z';
            break;
        case( fcomp::cyl::r ) :
            vfname  += 'r';
            vflabel += 'r';
            break;
        case( fcomp::cyl::th ) :
            vfname  += "θ";
            vflabel += "\\theta";
            break;
        default:
            mpi::fatal( "Invalid field component (fc) selected" );
    }

    zdf::grid_info info = {
        .name = (char *) vfname.c_str(),
        .ndims = 2,
        .label = (char *) vflabel.c_str(),
        .units = (char *) "e \\omega_n^2 / c",
    };

    zdf::iteration iteration = {
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_n"
    };

    zdf::grid_axis axis[2];
    axis[0] = (zdf::grid_axis) {
        .name = (char *) "z",
        .min = mov_window.motion(),
        .max = box.x + mov_window.motion(),
        .label = (char *) "z",
        .units = (char *) "c/\\omega_n"
    };

    axis[1] = (zdf::grid_axis) {
        .name = (char *) "r",
        .min = -dx.y/2,
        .max = box.y-dx.y/2,
        .label = (char *) "r",
        .units = (char *) "c/\\omega_n"
    };

    info.axis = axis;

    J -> save( m, jc, info, iteration, path );
}
