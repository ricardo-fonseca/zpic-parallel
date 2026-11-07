#include "emf.hpp"
#include <iostream>

#define opt_yee_block 256

using complex_t = util::complex<float>;

namespace kernel {

/**
 * @brief B advance for Yee algorithm (mode 0)
 * 
 * @param E         Pointer to 0,0 coordinate of E field
 * @param B         Pointer to 0,0 coordinate of B field
 * @param nx        Internal tile size
 * @param jstride   Stride for j coordinate
 * @param dt        Time step
 * @param dx        Cell size (z,r)
 * @param ir0       Coordinate of lower tile corner in grid
 */
__device__ void yee0_b( 
    cyl3<float> const * const __restrict__ E, 
    cyl3<float> * const __restrict__ B, 
    uint2 const tile_dims, int const jstride, 
    const double dt, float2 const dx, const int ir0 )
{
    // cylindrical cell sizes
    const float dz    = dx.x;
    const float dr    = dx.y;

    const float dt_dz = dt / dz;
    const float dt_dr = dt / dr;

    // Standard advance away from axial boundary
    const int j0 = ( ir0 > 0 ) ? -1 : 1;

    int const range_z = tile_dims.x + 2;
    int const range_r = tile_dims.y + 1 - j0;
    
    for( int idx = gpu::block::thread_rank(); idx < range_r * range_z; idx += gpu::block::num_threads() ) {
        const int i = -1 + idx % range_z;
        const int j = j0 + idx / range_z;

        /// @brief r at lower edge of j cell, normalized to Δr
        float rm  = ir0 + j - 0.5f;
        /// @brief r at upper edge of j cell, normalized to Δr
        float rp  = ir0 + j + 0.5f;
        /// @brief Δt/r at the center of j cell
        float dt_rc   = dt / ( ( ir0 + j ) * dr );

        B[ i + j*jstride ].r += (   dt_dz * ( E[(i+1) + j*jstride].th - E[i + j*jstride].th ) );  

        B[ i + j*jstride ].th += ( - dt_dz * ( E[(i+1) + j*jstride].r - E[i + j*jstride].r ) + 
                                    dt_dr * ( E[i + (j+1)*jstride].z - E[i + j*jstride].z ) );  

        B[ i + j*jstride ].z += ( - dt_rc * ( rp * E[i + (j+1)*jstride].th - rm * E[i + j*jstride].th ) );  
    }

    if ( ir0 == 0 ) {
        gpu::block::sync();
        for( int i = gpu::block::thread_rank() - 1; i < static_cast<int>(tile_dims.x) + 1; i += gpu::block::num_threads() ) {
            B[ i +   0 * jstride ].r = - B[ i + 1*jstride ].r;  
            B[ i +   0 * jstride ].th = 0;
            B[ i +   0 * jstride ].z += - 4 * dt_dr * E[ i + 1*jstride ].th;  
        }
    }
}

/**
 * @brief E advance for Yee algorithm ( no current, mode 0 )
 * 
 * @param E         Pointer to 0,0 coordinate of E field
 * @param B         Pointer to 0,0 coordinate of B field
 * @param nx        Internal tile size
 * @param jstride   Stride for j coordinate
 * @param dt        Time step
 * @param dx        Cell size (z,r)
 * @param ir0       Coordinate of lower tile corner in grid
 */
__device__ void yee0_e( 
    cyl3<float> * const __restrict__ E, 
    cyl3<float> const * const __restrict__ B, 
    uint2 const tile_dims, int const jstride, 
    const double dt, float2 const dx, const int ir0 )
{
    // cylindrical cell sizes
    const float dz    = dx.x;
    const float dr    = dx.y;

    const float dt_dz = dt / dz;
    const float dt_dr = dt / dr;

    // Standard advance away from axial boundary
    const int j0 = ( ir0 > 0 ) ? 0 : 1;

    int const range_z = tile_dims.x + 2;      // [0, nx.x + 2[
    int const range_r = tile_dims.y + 2 - j0; // [j0, nx.y + 2[

    for( int idx = gpu::block::thread_rank(); idx < range_z * range_r; idx += gpu::block::num_threads() ) {
        const int i = idx % range_z;
        const int j = idx / range_z + j0;

        /// @brief r at center of j-1 cell, normalized to Δr
        float rcm  = ir0 + j - 1;
        /// @brief r at center of j cell, normalized to Δr
        float rc   = ir0 + j;
        /// @brief Δt/r at the lower edge of j cell
        float dt_rm = dt / ( ( ir0 + j - 0.5 ) * dr );

        E[i + j*jstride].r += ( - dt_dz * ( B[i + j*jstride].th - B[(i-1) + j*jstride].th ) );

        E[i + j*jstride].th += ( + dt_dz * ( B[i + j*jstride].r - B[(i-1) + j*jstride].r ) - 
                                    dt_dr * ( B[i + j*jstride].z - B[i + (j-1)*jstride].z) );

        E[i + j*jstride].z += ( + dt_rm * ( rc * B[i + j*jstride].th - rcm * B[i + (j-1)*jstride].th ) );

    }

    if ( ir0 == 0 ) {
        gpu::block::sync();
        for( int i = gpu::block::thread_rank(); i < static_cast<int>(tile_dims.x) + 2; i += gpu::block::num_threads() ) {
            E[i +   0 *jstride].r = 0;
            E[i +   0 *jstride].th = -E[i + 1*jstride].th;
            E[i +   0 *jstride].z = E[i + 1*jstride].z;
        }
    }
}

/**
 * @brief Keep the tangential E field zero on PEC walls (inside the Yee step)
 * 
 * Called after the E advance and before the second B half advance, so that
 * the B update next to the wall uses E_tangential = 0 on the wall node.
 * 
 * Wall positions (see emf_bcz() / emf_bcr()):
 *  - lower z wall: ix = 0, tangential Er, Eθ
 *  - upper z wall: ix = nx (first upper guard cell), tangential Er, Eθ
 *  - upper r wall: iy = ny (first upper guard cell), tangential Ez, Eθ
 * 
 * @note PMC walls require no in-step correction, the tangential B
 *       components never sit on the wall node. Walls do not involve θ, so the
 *       same rule applies to every mode.
 * 
 * @param E             Pointer to 0,0 coordinate of E field (shared memory)
 * @param tile_idx      Tile index
 * @param local_ntiles  Local number of tiles
 * @param tile_dims     Tile dimensions
 * @param jstride       Stride for j coordinate
 * @param bc            Local rank boundary conditions
 */
template< typename T >
__device__
void pec_walls( 
    cyl3<T> * const __restrict__ E, 
    uint2 const tile_idx, uint2 const local_ntiles,
    uint2 const tile_dims, int const jstride,
    emf::bc_type const bc )
{
    const int nx = tile_dims.x;
    const int ny = tile_dims.y;

    // E is updated on rows / columns 0 .. n+1
    if ( tile_idx.x == 0 && bc.x.lower == emf::bc::pec ) {
        for( int j = gpu::block::thread_rank(); j < ny + 2; j += gpu::block::num_threads() ) {
            E[ j * jstride ].r  = T{};
            E[ j * jstride ].th = T{};
        }
    }

    if ( tile_idx.x == local_ntiles.x - 1 && bc.x.upper == emf::bc::pec ) {
        for( int j = gpu::block::thread_rank(); j < ny + 2; j += gpu::block::num_threads() ) {
            E[ nx + j * jstride ].r  = T{};
            E[ nx + j * jstride ].th = T{};
        }
    }

    if ( tile_idx.y == local_ntiles.y - 1 && bc.y.upper == emf::bc::pec ) {
        for( int i = gpu::block::thread_rank(); i < nx + 2; i += gpu::block::num_threads() ) {
            E[ i + ny * jstride ].z  = T{};
            E[ i + ny * jstride ].th = T{};
        }
    }
}

/**
 * @brief Set the tangential B images beyond PMC walls (inside the Yee step)
 * 
 * Called after the first B half advance and before the E advance, so that
 * the E update on the wall node uses the image (odd) tangential B values
 * beyond the wall. Without this the tangential E on a PMC wall node is
 * advanced with the unconstrained guard cell B values, which is unstable.
 * 
 * Tangential B positions (see emf_bcz() / emf_bcr()):
 *  - lower z wall (ix = 0): Br, Bθ at -1/2  <->  +1/2
 *  - upper z wall (ix = nx): Br, Bθ at nx+1/2  <->  nx-1/2
 *  - upper r wall (iy = ny): Bz, Bθ at ny Δr  <->  (ny-1) Δr
 * 
 * @note PEC walls require no in-step B correction (see pec_walls()). Walls do
 *       not involve θ, so the same rule applies to every mode.
 * 
 * @param B             Pointer to 0,0 coordinate of B field (shared memory)
 * @param tile_idx      Tile index
 * @param local_ntiles  Local number of tiles
 * @param tile_dims     Tile dimensions
 * @param jstride       Stride for j coordinate
 * @param bc            Local rank boundary conditions
 */
template< typename T >
__device__
void pmc_walls( 
    cyl3<T> * const __restrict__ B, 
    uint2 const tile_idx, uint2 const local_ntiles,
    uint2 const tile_dims, int const jstride,
    emf::bc_type const bc )
{
    const int nx = tile_dims.x;
    const int ny = tile_dims.y;

    // B is updated on rows / columns -1 .. n
    if ( tile_idx.x == 0 && bc.x.lower == emf::bc::pmc ) {
        for( int j = -1 + static_cast<int>( gpu::block::thread_rank() ); j < ny + 1; j += gpu::block::num_threads() ) {
            B[ -1 + j * jstride ].r  = - B[ 0 + j * jstride ].r;
            B[ -1 + j * jstride ].th = - B[ 0 + j * jstride ].th;
        }
    }

    if ( tile_idx.x == local_ntiles.x - 1 && bc.x.upper == emf::bc::pmc ) {
        for( int j = -1 + static_cast<int>( gpu::block::thread_rank() ); j < ny + 1; j += gpu::block::num_threads() ) {
            B[ nx + j * jstride ].r  = - B[ nx - 1 + j * jstride ].r;
            B[ nx + j * jstride ].th = - B[ nx - 1 + j * jstride ].th;
        }
    }

    if ( tile_idx.y == local_ntiles.y - 1 && bc.y.upper == emf::bc::pmc ) {
        for( int i = -1 + static_cast<int>( gpu::block::thread_rank() ); i < nx + 1; i += gpu::block::num_threads() ) {
            B[ i + ny * jstride ].z  = - B[ i + (ny-1) * jstride ].z;
            B[ i + ny * jstride ].th = - B[ i + (ny-1) * jstride ].th;
        }
    }
}

__global__ __launch_bounds__(opt_yee_block) 
void yee0( 
    grid::tiled_cyl3_view<float> E_grid, 
    grid::tiled_cyl3_view<float> B_grid, 
    float2 const dx, const double dt, emf::bc_type const local_bc ) {

    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int tile_id   = tile_idx.y * E_grid.local_ntiles.x + tile_idx.x;
    const uint2 tile_dims = E_grid.tile_dims;

    // Copy E and B into shared memory
    auto * shm = gpu::block::shared_mem<cyl3<float>>();
    cyl3<float> * __restrict__ E_local = & shm[0];
    cyl3<float> * __restrict__ B_local = & shm[ E_grid.tile_vol ];

    gpu::block::memcpy(E_local, E_grid.tile_buffer(tile_id), E_grid.tile_vol);
    gpu::block::memcpy(B_local, B_grid.tile_buffer(tile_id), B_grid.tile_vol);

    gpu::block::sync();

    cyl3<float> * const __restrict__ E = & E_local[ E_grid.inner_offset ];
    cyl3<float> * const __restrict__ B = & B_local[ B_grid.inner_offset ];

    const int jstride = E_grid.tile_ystride();
    const int ir0 = ( E_grid.local_tile_start.y + tile_idx.y ) * tile_dims.y;

    // Half B advance
    yee0_b( E, B, tile_dims, jstride, dt/2, dx, ir0 );
    gpu::block::sync();

    // Set tangential B images beyond PMC walls
    pmc_walls( B, tile_idx, E_grid.local_ntiles, tile_dims, jstride, local_bc );
    gpu::block::sync();

    // Full E advance
    yee0_e( E, B, tile_dims, jstride, dt,   dx, ir0 );
    gpu::block::sync();

    // Keep tangential E zero on PEC walls
    pec_walls( E, tile_idx, E_grid.local_ntiles, tile_dims, jstride, local_bc );
    gpu::block::sync();

    // Half B advance
    yee0_b( E, B, tile_dims, jstride, dt/2, dx, ir0 );
    gpu::block::sync();

    // Copy data to global memory
    gpu::block::memcpy(E_grid.tile_buffer(tile_id), E_local, E_grid.tile_vol);
    gpu::block::memcpy(B_grid.tile_buffer(tile_id), B_local, B_grid.tile_vol);
}

/**
 * @brief B advance for Yee algorithm (mode m)
 * 
 * @param m         Mode
 * @param E         Pointer to 0,0 coordinate of E field
 * @param B         Pointer to 0,0 coordinate of B field
 * @param nx        Internal tile size
 * @param jstride   Stride for j coordinate
 * @param dt        Time step
 * @param dx        Cell size (z,r)
 * @param ir0       Coordinate of lower tile corner in grid
 */
__device__ 
void yeem_b( 
    const int m, 
    cyl3<complex_t> const * const __restrict__ E, 
    cyl3<complex_t> * const __restrict__ B, 
    uint2 const tile_dims, int const jstride, 
    const double dt, float2 const dx, const int ir0 )
{
    // cylindrical cell sizes
    const float dz    = dx.x;
    const float dr    = dx.y;

    const float dt_dz = dt / dz;
    const float dt_dr = dt / dr;

    // Standard advance away from axial boundary
    const int j0 = ( ir0 > 0 ) ? -1 : 1;

    ///@brief m * imaginary unit
    const complex_t mI{0,static_cast<float>(m)};
    ///@brief imaginary unit
    const complex_t I{0,1};

    int const range_z = tile_dims.x + 2;
    int const range_r = tile_dims.y + 1 - j0;
    
    for( int idx = gpu::block::thread_rank(); idx < range_r * range_z; idx += gpu::block::num_threads() ) {

        const int i = -1 + idx % range_z;
        const int j = j0 + idx / range_z;

        /// @brief r/Δr at lower edge of j cell
        float rl  = ir0 + j - 0.5f;
        /// @brief r/Δr at upper edge of j cell
        float ru  = ir0 + j + 0.5f;
        /// @brief Δt/r at the center of j cell
        float dt_rc = dt / ( ( ir0 + j ) * dr );
        /// @brief Δt/r at the lower edge of j cell
        float dt_rl = dt / ( ( ir0 + j - 0.5 ) * dr );

        B[ i + j*jstride ].r += 
            - dt_rl * mI * E[ i + j * jstride ].z                       // (Δt/r) m I Ez
            + dt_dz * ( E[(i+1) + j*jstride].th - E[i + j*jstride].th );  // Δt ∂Et/∂z

        B[ i + j*jstride ].th += 
            - dt_dz * ( E[(i+1) + j*jstride].r - E[i + j*jstride].r )   // Δt ∂Er/∂z
            + dt_dr * ( E[i + (j+1)*jstride].z - E[i + j*jstride].z );  // Δt ∂Ez/∂r

        B[ i + j*jstride ].z += - dt_rc * (                             // Δt/r
            + ( ru * E[i + (j+1)*jstride].th - rl * E[i + j*jstride].th ) // ∂(r Eth)/∂r 
            - mI * E[ i + j * jstride ].r                               // m I Er
        );  
    }

    if ( ir0 == 0 ) {
        gpu::block::sync();
        if ( m == 1 ) {
            // Mode m = 1 is a special case
            for( int i = gpu::block::thread_rank() - 1; i < static_cast<int>(tile_dims.x) + 1; i += gpu::block::num_threads() ) {
                B[ i + 0 *jstride ].r  = B[ i + 1*jstride ].r;
                B[ i + 0 *jstride ].th = 0.125f * I * (9.f * B[ i + 1*jstride ].r - B[ i + 2*jstride ].r );
                B[ i + 0 *jstride ].z  = 0;
            }
        } else {
            for( int i = gpu::block::thread_rank() - 1; i < static_cast<int>(tile_dims.x) + 1; i += gpu::block::num_threads() ) {
                B[ i + 0 *jstride ].r = - B[ i + 1*jstride ].r;  // Br(r=0) = 0
                B[ i + 0 *jstride ].th= 0;
                B[ i + 0 *jstride ].z = 0;
            }
        }
    }
}

/**
 * @brief E advance for Yee algorithm ( no current, mode m )
 * 
 * @param m         Mode
 * @param E         Pointer to 0,0 coordinate of E field
 * @param B         Pointer to 0,0 coordinate of B field
 * @param nx        Internal tile size
 * @param jstride   Stride for j coordinate
 * @param dt        Time step
 * @param dx        Cell size (z,r)
 * @param ir0       Coordinate of lower tile corner in grid
 */
__device__ void yeem_e( 
    const int m,
    cyl3<complex_t> * const __restrict__ E, 
    cyl3<complex_t> const * const __restrict__ B, 
    uint2 const tile_dims, int const jstride, 
    const double dt, float2 const dx, const int ir0 )
{
    // cylindrical cell sizes
    const float dz    = dx.x;
    const float dr    = dx.y;

    const float dt_dz = dt / dz;
    const float dt_dr = dt / dr;

    // Standard advance away from axial boundary
    const int j0 = ( ir0 > 0 ) ? 0 : 1;

    /// @brief m * imaginary unit
    const complex_t mI{0,static_cast<float>(m)};

    int const range_z = tile_dims.x + 2;      // [0, nx.x + 2[
    int const range_r = tile_dims.y + 2 - j0; // [j0, nx.y + 2[

    for( int idx = gpu::block::thread_rank(); idx < range_z * range_r; idx += gpu::block::num_threads() ) {
        const int i = idx % range_z;
        const int j = idx / range_z + j0;

        /// @brief r/Δr at center of j-1 cell
        float rcm  = ir0 + j - 1;
        /// @brief rΔr at center of j cell
        float rc   = ir0 + j;
        /// @brief Δt/r at the lower edge of j cell
        float dt_rl = dt / ( ( ir0 + j - 0.5 ) * dr );
        /// @brief Δt/r at the center of j cell
        float dt_rc = dt / ( ( ir0 + j ) * dr );

        E[i + j*jstride].r += ( 
            - dt_dz * ( B[i + j*jstride].th - B[(i-1) + j*jstride].th )   // Δt ∂Bt/∂z 
            + dt_rc * mI * B[ i + j * jstride ].z                      // (Δt/r) m I Bz
        );

        E[i + j*jstride].th += ( + dt_dz * ( B[i + j*jstride].r - B[(i-1) + j*jstride].r)
                                 - dt_dr * ( B[i + j*jstride].z - B[i + (j-1)*jstride].z) );

        E[i + j*jstride].z +=  dt_rl * (                                  // Δt/r
            + rc * B[i + j * jstride].th - rcm * B[i + (j-1)*jstride].th  // ∂(r Bt)/∂r 
            - mI * B[i + j * jstride].r                                   // - m I Br
        );

    }

    if ( ir0 == 0 ) {
        gpu::block::sync();
        if ( m == 1 ) {
            // Mode m = 1 is a special case
            for( int i = gpu::block::thread_rank(); i < static_cast<int>(tile_dims.x) + 2; i += gpu::block::num_threads() ) {
                E[ i + 0 *jstride ].r = ( 4.f * E[ i + 1*jstride ].r - E[ i + 2*jstride ].r ) / 3.f;  
                E[ i + 0 *jstride ].th =  E[ i + 1 * jstride ].th;  // ∂Bt/∂r(r=0) = 0
                E[ i + 0 *jstride ].z = -E[ i + 1 * jstride ].z;  // Ez(r=0) = 0
            }
        } else {
            for( int i = gpu::block::thread_rank(); i < static_cast<int>(tile_dims.x) + 2; i += gpu::block::num_threads() ) {
                E[ i + 0 *jstride ].r = 0;
                E[ i + 0 *jstride ].th = - E[ i + 1 *jstride ].th;    // Et(r=0) = 0
                E[ i + 0 *jstride ].z = - E[ i + 1 *jstride ].z;    // Ez(r=0) = 0;  
            }
        }
    }
}

__global__ __launch_bounds__(opt_yee_block) void yeem( 
    const int m,
    grid::tiled_cyl3_view<complex_t> E_grid,
    grid::tiled_cyl3_view<complex_t> B_grid,
    float2 const dx, const double dt, emf::bc_type const local_bc ) {

    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int tile_id   = tile_idx.y * E_grid.local_ntiles.x + tile_idx.x;
    const uint2 tile_dims = E_grid.tile_dims;

    // Copy E and B into shared memory
    auto * shm = gpu::block::shared_mem<cyl3<complex_t>>();
    cyl3<complex_t> * __restrict__ E_local = & shm[0];
    cyl3<complex_t> * __restrict__ B_local = & shm[ E_grid.tile_vol ];

    gpu::block::memcpy(E_local, E_grid.tile_buffer(tile_id), E_grid.tile_vol);
    gpu::block::memcpy(B_local, B_grid.tile_buffer(tile_id), B_grid.tile_vol);

    gpu::block::sync();

    cyl3<complex_t> * const __restrict__ E = & E_local[ E_grid.inner_offset ];
    cyl3<complex_t> * const __restrict__ B = & B_local[ B_grid.inner_offset ];

    const int jstride = E_grid.tile_ystride();
    const int ir0 = ( E_grid.local_tile_start.y + tile_idx.y ) * tile_dims.y;

    // Half B advance
    yeem_b( m, E, B, tile_dims, jstride, dt/2, dx, ir0 );
    gpu::block::sync();

    // Set tangential B images beyond PMC walls
    pmc_walls( B, tile_idx, E_grid.local_ntiles, tile_dims, jstride, local_bc );
    gpu::block::sync();

    // Full E advance
    yeem_e( m, E, B, tile_dims, jstride, dt,   dx, ir0 );
    gpu::block::sync();

    // Keep tangential E zero on PEC walls
    pec_walls( E, tile_idx, E_grid.local_ntiles, tile_dims, jstride, local_bc );
    gpu::block::sync();

    // Half B advance
    yeem_b( m, E, B, tile_dims, jstride, dt/2, dx, ir0 );
    gpu::block::sync();

    // Copy data to global memory
    gpu::block::memcpy(E_grid.tile_buffer(tile_id), E_local, E_grid.tile_vol);
    gpu::block::memcpy(B_grid.tile_buffer(tile_id), B_local, B_grid.tile_vol);
}


} // namespace kernel

/**
 * @brief Move simulation window
 * 
 * When using a moving simulation window checks if a window move is due
 * at the current iteration and if so shifts left the data by one cell.
 * 
 * @note The new rightmost interior column takes the values of the first
 *       upper guard column (at the open x boundary these were evolved by
 *       the Yee kernel from stale guard values); only the last guard column
 *       is zeroed.
 */
void emf::move_window() {

    if ( mov_window.needs_move( iter * dt ) ) {

        E->x_shift_left(1);
        B->x_shift_left(1);

        mov_window.advance();
    }
}


/**
 * @brief Advance EM fields 1 time step (no current)
 * 
 */
void emf::advance() {

    // Get tile information from mode0
    auto & E0 = E -> mode0();
    auto & B0 = B -> mode0();

    const auto ntiles    = E0.get_local_ntiles();
    const auto field_vol = E0.tile_vol;

    dim3 grid( ntiles.x, ntiles.y );
    auto block = opt_yee_block;

    // Solve for mode 0
    size_t shm_size = 2 * field_vol * sizeof(cyl3<float>);
    gpu::block::set_shmem_size( kernel::yee0, shm_size );
    kernel::yee0 <<< grid, block, shm_size >>> (
        E0.view(), B0.view(), dx, dt, local_bc
    );

    // Solve for high-order modes
    if ( nmodes > 1 ) {
        shm_size = 2 * field_vol * sizeof(cyl3<complex_t>);
        gpu::block::set_shmem_size( kernel::yeem, shm_size );
    }

    for( int m = 1; m < nmodes; m++ ) { 
        auto & Em = E -> mode( m );
        auto & Bm = B -> mode( m );

        kernel::yeem <<< grid, block, shm_size >>> (
            m, Em.view(), Bm.view(), dx, dt, local_bc
        );
    }

    // Update guard cells with new values
    E -> copy_to_gc();
    B -> copy_to_gc();

    // Do additional bc calculations if needed
    process_bc();

    // Advance internal iteration number
    iter += 1;

    // Move simulation window if needed
    if ( mov_window.active() ) move_window();

}

namespace kernel {

/**
 * @brief Physical boundary conditions for the z direction
 * 
 * @note Launch with grid( 2, local_ntiles.y ): blockIdx.x selects the
 *       boundary (0 - lower, 1 - upper), blockIdx.y the tile row
 * 
 * @note All extended rows are processed; must run before emf_bcr() for
 *       correct corner values. Walls do not involve θ, so the same images
 *       apply to every mode.
 * 
 * @tparam T            float (mode 0) or complex (modes m > 0)
 * 
 * @param E_grid        View of tiled electric field grid
 * @param B_grid        View of tiled magnetic field grid
 * @param local_bc      Local rank boundary conditions
 */
template< typename T >
__global__
void emf_bcz( 
    grid::tiled_cyl3_view<T> E_grid,
    grid::tiled_cyl3_view<T> B_grid,
    emf::bc_type local_bc ) {

    int bnd_x = blockIdx.x;
    int tile_idx_y = blockIdx.y;
    const int jstride = E_grid.tile_ystride();

    if ( bnd_x == 0 ) {
        // Lower boundary
        cyl3<T> * __restrict__ E = & E_grid.tile_buffer
            (0,tile_idx_y)      // lower x boundary tile
            [ E_grid.gc.x.lower ];     // point to first x cell (ix = 0)
        cyl3<T> * __restrict__ B = & B_grid.tile_buffer
            (0,tile_idx_y)
            [ B_grid.gc.x.lower ];

        switch( local_bc.x.lower ) {
        case( emf::bc::pmc) :
            for( unsigned idx = gpu::block::thread_rank(); 
                 idx < E_grid.tile_ext_dims.y;
                 idx += gpu::block::num_threads() ) {
                // j includes the j-stride
                const int j = idx * jstride;

                E[ -1 + j ].z = -E[ 0 + j ].z;
                E[ -1 + j ].r =  E[ 1 + j ].r;
                E[ -1 + j ].th =  E[ 1 + j ].th;

                B[ -1 + j ].z =  B[ 1 + j ].z;
                B[ -1 + j ].r = -B[ 0 + j ].r;
                B[ -1 + j ].th = -B[ 0 + j ].th;
            }
            break;

        case( emf::bc::pec ) :
            for( unsigned idx = gpu::block::thread_rank(); 
                 idx < E_grid.tile_ext_dims.y;
                 idx += gpu::block::num_threads() ) {
                const int j = idx * jstride;

                E[ -1 + j ].z =  E[ 0 + j ].z;
                E[ -1 + j ].r = -E[ 1 + j ].r;
                E[ -1 + j ].th = -E[ 1 + j ].th;

                E[  0 + j ].r = T{};
                E[  0 + j ].th = T{};

                B[ -1 + j ].z = -B[ 1 + j ].z;
                B[ -1 + j ].r =  B[ 0 + j ].r;
                B[ -1 + j ].th =  B[ 0 + j ].th;

                B[  0 + j ].z = T{};
            }
            break;
        default:
            break;
        } 
    } else {
        // Upper boundary
        cyl3<T> * __restrict__ E = & E_grid.tile_buffer
            (E_grid.local_ntiles.x-1,tile_idx_y)    // upper x boundary tile
            [ E_grid.gc.x.lower + E_grid.tile_dims.x ];    // point to first upper gc (ix = tile_dims.x)
        cyl3<T> * __restrict__ B = & B_grid.tile_buffer
            (B_grid.local_ntiles.x-1,tile_idx_y)    // upper x boundary tile
            [ B_grid.gc.x.lower + B_grid.tile_dims.x ];    // point to first upper gc (ix = tile_dims.x)

        switch( local_bc.x.upper ) {
        case( emf::bc::pmc) :
            for( unsigned idx = gpu::block::thread_rank();
                 idx < E_grid.tile_ext_dims.y; 
                 idx += gpu::block::num_threads() ) {
                const int j = idx * jstride;

                E[ j ].z = -E[ -1 + j ].z;

                E[ 1 + j ].z = -E[ -2 + j ].z;
                E[ 1 + j ].r =  E[ -1 + j ].r;
                E[ 1 + j ].th =  E[ -1 + j ].th;

                B[ j ].r = -B[ -1 + j ].r;
                B[ j ].th = -B[ -1 + j ].th;

                B[ 1 + j ].z =  B[ -1 + j ].z;
                B[ 1 + j ].r = -B[ -2 + j ].r;
                B[ 1 + j ].th = -B[ -2 + j ].th;
            }
            break;

        case( emf::bc::pec) :
            for( unsigned idx = gpu::block::thread_rank();
                 idx < E_grid.tile_ext_dims.y; 
                 idx += gpu::block::num_threads() ) {
                const int j = idx * jstride;

                E[ j ].z =  E[ -1 + j ].z;
                E[ j ].r = T{};
                E[ j ].th = T{};

                E[ 1 + j ].z =  E[ -2 + j ].z;
                E[ 1 + j ].r = -E[ -1 + j ].r;
                E[ 1 + j ].th = -E[ -1 + j ].th;

                B[ j ].z = T{};
                B[ j ].r =  B[ -1 + j ].r;
                B[ j ].th =  B[ -1 + j ].th;

                B[ 1 + j ].z = -B[ -1 + j ].z;
                B[ 1 + j ].r =  B[ -2 + j ].r;
                B[ 1 + j ].th =  B[ -2 + j ].th;
            }
            break;
        default:
            break;
        }
    }
}

/**
 * @brief Physical boundary conditions for the r direction
 * 
 * @note Launch with grid( local_ntiles.x ): blockIdx.x selects the tile
 *       column. Only the upper r boundary is processed (the lower r boundary
 *       is the axis, handled inside the Yee kernels).
 * 
 * @note All extended columns are processed; must run after emf_bcz() for
 *       correct corner values.
 * 
 * @tparam T            float (mode 0) or complex (modes m > 0)
 * 
 * @param E_grid        View of tiled electric field grid
 * @param B_grid        View of tiled magnetic field grid
 * @param local_bc      Local rank boundary conditions
 */
template< typename T >
__global__
void emf_bcr( 
    grid::tiled_cyl3_view<T> E_grid,
    grid::tiled_cyl3_view<T> B_grid,
    emf::bc_type local_bc ) {

    int tile_idx_x = blockIdx.x;
    const int jstride = E_grid.tile_ystride();
    
    {
        // Upper boundary
        cyl3<T> * __restrict__ E = & E_grid.tile_buffer
            (tile_idx_x,E_grid.local_ntiles.y-1)   // upper y boundary tiles
            [ (E_grid.gc.y.lower + E_grid.tile_dims.y ) * jstride ];  // point to first upper gc (iy = tile_dims.y)
        cyl3<T> * __restrict__ B = & B_grid.tile_buffer
            (tile_idx_x,B_grid.local_ntiles.y-1)
            [ (B_grid.gc.y.lower + B_grid.tile_dims.y ) * jstride ]; 

        switch( local_bc.y.upper ) {
        case( emf::bc::pmc) :
            for( unsigned idx = gpu::block::thread_rank();
                 idx < E_grid.tile_ext_dims.x; 
                 idx += gpu::block::num_threads() ) {
                const int i = idx;

                E[ i ].r = -E[ i - 1 * jstride ].r;

                E[ i + jstride ].z  =  E[ i -1 * jstride ].z;
                E[ i + jstride ].r  = -E[ i -2 * jstride ].r;
                E[ i + jstride ].th =  E[ i -1 * jstride ].th;

                B[ i ].z  = -B[ i - 1*jstride ].z;
                B[ i ].th = -B[ i - 1*jstride ].th;

                B[ i + jstride ].z  = -B[ i - 2 * jstride ].z;
                B[ i + jstride ].r  =  B[ i - 1 * jstride ].r;
                B[ i + jstride ].th = -B[ i - 2 * jstride ].th;
            }
            break;

        case( emf::bc::pec) :
            for( unsigned idx = gpu::block::thread_rank();
                 idx < E_grid.tile_ext_dims.x; 
                 idx += gpu::block::num_threads() ) {
                const int i = idx;

                E[ i ].z = T{};
                E[ i ].r =  E[ i - 1 * jstride ].r;
                E[ i ].th = T{};

                E[ i + 1 * jstride ].z  = -E[ i - 1 * jstride ].z;
                E[ i + 1 * jstride ].r  =  E[ i - 2 * jstride ].r;
                E[ i + 1 * jstride ].th = -E[ i - 1 * jstride ].th;

                B[ i ].z  =  B[ i - 1 * jstride ].z;
                B[ i ].r  = T{};
                B[ i ].th =  B[ i - 1 * jstride ].th;

                B[ i + 1 * jstride ].z  =  B[ i - 2 * jstride ].z;
                B[ i + 1 * jstride ].r  = -B[ i - 1 * jstride ].r;
                B[ i + 1 * jstride ].th =  B[ i - 2 * jstride ].th;
            }
            break;
        default:
            break;
        }
    }
}

} // namespace kernel

/**
 * @brief Processes "physical" boundary conditions
 * 
 * @note Physical boundaries are only applied on ranks that sit on the
 *       corresponding edge of the global domain; internal partition
 *       boundaries were already handled by copy_to_gc()
 */
void emf::process_bc() {

    // Get tile information from mode0
    auto & E0 = E -> mode0();
    auto & B0 = B -> mode0();
    const uint2 ntiles = E0.get_local_ntiles();
    dim3 block( 64 );

    // z boundaries
    if ( is_conducting( local_bc.x.lower ) || is_conducting( local_bc.x.upper ) ) {
        dim3 grid( 2, ntiles.y );
        kernel::emf_bcz <<< grid, block >>> (
            E0.view(), B0.view(), local_bc
        );

        for( int m = 1; m < nmodes; m++ ) {
            kernel::emf_bcz <<< grid, block >>> (
                E -> mode(m).view(), B -> mode(m).view(), local_bc
            );
        }
    }

    // r boundaries (must run after x boundaries for correct corner values)
    if ( is_conducting( local_bc.y.upper ) ) {
        dim3 grid( ntiles.x );
        kernel::emf_bcr <<< grid, block >>> (
            E0.view(), B0.view(), local_bc
        );

        for( int m = 1; m < nmodes; m++ ) {
            kernel::emf_bcr <<< grid, block >>> (
                E -> mode(m).view(), B -> mode(m).view(), local_bc
            );
        }
    }
}


namespace kernel {

/**
 * @brief E advance for Yee algorithm including
 * 
 * @param E         Pointer to 0,0 coordinate of E field
 * @param B         Pointer to 0,0 coordinate of B field
 * @param nx        Internal tile size
 * @param ystride   Stride for y coordinate
 * @param J         Pointer to 0,0 coordinate of J field
 * @param J_ystride Stride for y coordinate of J field
 * @param dt_dx     \Delta t / \Delta x
 * @param dt_dy     \Delta t / \Delta y
 * @param dt        \Delta t
 */
__device__ void yee0J_e( 
    cyl_float3 * const __restrict__ E, 
    cyl_float3 const * const __restrict__ B, 
    uint2 const tile_dims, int const jstride, 
    cyl_float3 const * const __restrict__ J, int const J_jstride, 
    float const dt, float2 const dx, const int ir0 )
{
    // cylindrical cell sizes
    const float dz    = dx.x;
    const float dr    = dx.y;

    const float dt_dz = dt / dz;
    const float dt_dr = dt / dr;

    // Standard advance away from axial boundary
    const int j0 = ( ir0 > 0 ) ? 0 : 1;

    int const range_z = tile_dims.x + 2;      // [0, nx.x + 2[
    int const range_r = tile_dims.y + 2 - j0; // [j0, nx.y + 2[

    for( int idx = gpu::block::thread_rank(); idx < range_z * range_r; idx += gpu::block::num_threads() ) {
        const int i = idx % range_z;
        const int j = idx / range_z + j0;

        /// @brief r at center of j-1 cell, normalized to Δr
        float rcm  = ir0 + j - 1;
        /// @brief r at center of j cell, normalized to Δr
        float rc   = ir0 + j;
        /// @brief Δt/r at the lower edge of j cell
        float dt_rm = dt / ( ( ir0 + j - 0.5 ) * dr );

        E[i + j*jstride].r += - dt_dz * ( B[i + j*jstride].th - B[(i-1) + j*jstride].th)
                              - dt * J[i + j*J_jstride].r;

        E[i + j*jstride].th += ( + dt_dz * ( B[i + j*jstride].r - B[(i-1) + j*jstride].r) - 
                                   dt_dr * ( B[i + j*jstride].z - B[i + (j-1)*jstride].z) )
                               - dt * J[i + j*J_jstride].th;

        E[i + j*jstride].z += ( + dt_rm * ( rc * B[i + j*jstride].th - rcm * B[i + (j-1)*jstride].th ) )
                                - dt * J[i + j*J_jstride].z;
    }

    if ( ir0 == 0 ) {
        gpu::block::sync();
        for( int i = gpu::block::thread_rank(); i < static_cast<int>(tile_dims.x) + 2; i += gpu::block::num_threads() ) {
            E[i +   0 *jstride].r  =  0;
            E[i +   0 *jstride].th = -E[i + 1*jstride].th;
            E[i +   0 *jstride].z  =  E[i + 1*jstride].z;
        }
    }
}

__global__ __launch_bounds__(opt_yee_block) void yee0J( 
    grid::tiled_cyl3_view<float> E_grid,
    grid::tiled_cyl3_view<float> B_grid,
    grid::tiled_view<const cyl3<float>> J_grid,
    float2 const dx, float const dt,
    emf::bc_type local_bc ) {

    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int tile_id   = tile_idx.y * E_grid.local_ntiles.x + tile_idx.x;
    const uint2 tile_dims = E_grid.tile_dims;

    // Copy E and B into shared memory
    auto * shm = gpu::block::shared_mem<cyl_float3>();
    cyl_float3 * __restrict__ E_local = & shm[0];
    cyl_float3 * __restrict__ B_local = & shm[ E_grid.tile_vol ];

    gpu::block::memcpy(E_local, E_grid.tile_buffer(tile_id), E_grid.tile_vol);
    gpu::block::memcpy(B_local, B_grid.tile_buffer(tile_id), B_grid.tile_vol);

    gpu::block::sync();

    cyl_float3 * const __restrict__ E = & E_local[ E_grid.inner_offset ];
    cyl_float3 * const __restrict__ B = & B_local[ B_grid.inner_offset ];
    cyl_float3 const * const __restrict__ J = J_grid.tile_data(tile_id);

    const int jstride = E_grid.tile_ystride();
    const int J_jstride = J_grid.tile_ystride();
    const int ir0 = ( E_grid.local_tile_start.y + tile_idx.y ) * tile_dims.y;

    // Half B advance
    yee0_b( E, B, tile_dims, jstride, dt/2, dx, ir0 );
    gpu::block::sync();

    // Set tangential B images beyond PMC walls
    pmc_walls( B, tile_idx, E_grid.local_ntiles, tile_dims, jstride, local_bc );
    gpu::block::sync();

    // Full E advance including current
    yee0J_e( E, B, tile_dims, jstride, J, J_jstride, dt, dx, ir0 );
    gpu::block::sync();

    // Keep tangential E zero on PEC walls
    pec_walls( E, tile_idx, E_grid.local_ntiles, tile_dims, jstride, local_bc );
    gpu::block::sync();

    // Half B advance
    yee0_b( E, B, tile_dims, jstride, dt/2, dx, ir0 );
    gpu::block::sync();

    // Copy data to global memory
    gpu::block::memcpy(E_grid.tile_buffer(tile_id), E_local, E_grid.tile_vol);
    gpu::block::memcpy(B_grid.tile_buffer(tile_id), B_local, B_grid.tile_vol);
}

__device__ void yeemJ_e( 
    const int m,
    cyl_cfloat3 * const __restrict__ E, 
    cyl_cfloat3 const * const __restrict__ B, 
    uint2 const tile_dims, int const jstride, 
    cyl_cfloat3 const * const __restrict__ J, int const J_jstride, 
    float const dt, float2 const dx, const int ir0 )
{
    // cylindrical cell sizes
    const float dz    = dx.x;
    const float dr    = dx.y;

    const float dt_dz = dt / dz;
    const float dt_dr = dt / dr;

    // Standard advance away from axial boundary
    const int j0 = ( ir0 > 0 ) ? 0 : 1;

    /// @brief m * imaginary unit
    const complex_t mI{0,static_cast<float>(m)};

    int const range_z = tile_dims.x + 2;      // [0, nx.x + 2[
    int const range_r = tile_dims.y + 2 - j0; // [j0, nx.y + 2[

    for( int idx = gpu::block::thread_rank(); idx < range_z * range_r; idx += gpu::block::num_threads() ) {
        const int i = idx % range_z;
        const int j = idx / range_z + j0;

        /// @brief r/Δr at center of j-1 cell
        float rcm  = ir0 + j - 1;
        /// @brief rΔr at center of j cell
        float rc   = ir0 + j;
        /// @brief Δt/r at the lower edge of j cell
        float dt_rl = dt / ( ( ir0 + j - 0.5 ) * dr );
        /// @brief Δt/r at the center of j cell
        float dt_rc = dt / ( ( ir0 + j ) * dr );

        E[i + j*jstride].r +=  
            + dt_rc * mI * B[ i + j * jstride ].z                      // (Δt/r) m I Bz
            - dt_dz * ( B[i + j*jstride].th - B[(i-1) + j*jstride].th)   // Δt ∂Bth/∂z 
            - dt * J[i + j*J_jstride].r;

        E[i + j*jstride].th += 
            + dt_dz * ( B[i + j*jstride].r - B[(i-1) + j*jstride].r)
            - dt_dr * ( B[i + j*jstride].z - B[i + (j-1)*jstride].z) 
            - dt * J[i + j*J_jstride].th;

        E[i + j*jstride].z +=  dt_rl * (                                // Δt/r
            + rc * B[i + j * jstride].th - rcm * B[i + (j-1)*jstride].th  // ∂(r Bt)/∂r 
            - mI * B[i + j * jstride].r                                 // m I Br
        ) - dt * J[i + j*J_jstride].z;
    }

    if ( ir0 == 0 ) {
        gpu::block::sync();
        if ( m == 1 ) {
            // Mode m = 1 is a special case
            for( int i = gpu::block::thread_rank(); i < static_cast<int>(tile_dims.x) + 2; i += gpu::block::num_threads() ) {
                E[ i + 0 *jstride ].r = ( 4.f * E[ i + 1*jstride ].r - E[ i + 2*jstride ].r ) / 3.f;
                E[ i + 0 *jstride ].th =  E[ i + 1 * jstride ].th;   // ∂Bt/∂r(r=0) = 0
                E[ i + 0 *jstride ].z = -E[ i + 1 * jstride ].z;  // Ez(r=0) = 0
            }
        } else {
            for( int i = gpu::block::thread_rank(); i < static_cast<int>(tile_dims.x) + 2; i += gpu::block::num_threads() ) {
                E[ i + 0 *jstride ].r = 0;
                E[ i + 0 *jstride ].th = - E[ i + 1 *jstride ].th;    // Eth(r=0) = 0
                E[ i + 0 *jstride ].z = - E[ i + 1 *jstride ].z;    // Ez(r=0) = 0;  
            }
        }

    }
}

__global__ __launch_bounds__(opt_yee_block) void yeemJ( 
    const int m,
    grid::tiled_cyl3_view<complex_t> E_grid,
    grid::tiled_cyl3_view<complex_t> B_grid,
    grid::tiled_view<const cyl3<complex_t>> J_grid,
    float2 const dx, float const dt,
    emf::bc_type local_bc ) {

    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int tile_id   = tile_idx.y * E_grid.local_ntiles.x + tile_idx.x;
    const uint2 tile_dims = E_grid.tile_dims;

    // Copy E and B into shared memory
    auto * shm = gpu::block::shared_mem<cyl_cfloat3>();
    cyl_cfloat3 * __restrict__ E_local = & shm[0];
    cyl_cfloat3 * __restrict__ B_local = & shm[ E_grid.tile_vol ];

    gpu::block::memcpy(E_local, E_grid.tile_buffer(tile_id), E_grid.tile_vol);
    gpu::block::memcpy(B_local, B_grid.tile_buffer(tile_id), B_grid.tile_vol);

    gpu::block::sync();

    cyl_cfloat3 * const __restrict__ E = & E_local[ E_grid.inner_offset ];
    cyl_cfloat3 * const __restrict__ B = & B_local[ B_grid.inner_offset ];
    cyl_cfloat3 const * const __restrict__ J = J_grid.tile_data(tile_id);

    const int jstride = E_grid.tile_ystride();
    const int J_jstride = J_grid.tile_ystride();
    const int ir0 = ( E_grid.local_tile_start.y + tile_idx.y ) * tile_dims.y;

    // Half B advance
    yeem_b( m, E, B, tile_dims, jstride, dt/2, dx, ir0 );
    gpu::block::sync();

    // Set tangential B images beyond PMC walls
    pmc_walls( B, tile_idx, E_grid.local_ntiles, tile_dims, jstride, local_bc );
    gpu::block::sync();

    // Full E advance including current
    yeemJ_e( m, E, B, tile_dims, jstride, J, J_jstride, dt, dx, ir0 );
    gpu::block::sync();

    // Keep tangential E zero on PEC walls
    pec_walls( E, tile_idx, E_grid.local_ntiles, tile_dims, jstride, local_bc );
    gpu::block::sync();

    // Half B advance
    yeem_b( m, E, B, tile_dims, jstride, dt/2, dx, ir0 );
    gpu::block::sync();

    // Copy data to global memory
    gpu::block::memcpy(E_grid.tile_buffer(tile_id), E_local, E_grid.tile_vol);
    gpu::block::memcpy(B_grid.tile_buffer(tile_id), B_local, B_grid.tile_vol);

}

} // namespace kernel

/**
 * @brief Advance EM fields 1 time step including current
 * 
 * @param current   Electric current
 */
void emf::advance( current & current ) {

    // Get tile information from mode0
    auto & E0 = E -> mode0();
    auto & B0 = B -> mode0();

    const auto ntiles    = E0.get_local_ntiles();
    const auto field_vol = E0.tile_vol;

    auto & J0 = current.mode0();

    dim3 grid( ntiles.x, ntiles.y );
    auto block = opt_yee_block;
    
    // Solve for mode 0
    size_t shm_size = 2 * field_vol * sizeof(cyl_float3);
    gpu::block::set_shmem_size( kernel::yee0J, shm_size );
    kernel::yee0J <<< grid, block, shm_size >>> (
        E0.view(), B0.view(), J0.cview(),
        dx, static_cast<float>(dt), local_bc
    );

    // Solve for high-order modes
    if ( nmodes > 1 ) {
        shm_size = 2 * field_vol * sizeof(cyl_cfloat3);
        gpu::block::set_shmem_size( kernel::yeemJ, shm_size );
    }

    for( int m = 1; m < nmodes; m++ ) { 
        auto & Em = E -> mode( m );
        auto & Bm = B -> mode( m );
        auto & Jm = current.mode( m );

        kernel::yeemJ <<< grid, block, shm_size >>> (
            m, Em.view(), Bm.view(), Jm.cview(),
            dx, static_cast<float>(dt), local_bc
        );
    }

    // Update guard cells with new values
    E -> copy_to_gc();
    B -> copy_to_gc();

    // Do additional bc calculations if needed
    process_bc();

    // Advance internal iteration number
    iter += 1;

    // Move simulation window if needed
    if ( mov_window.active() ) move_window();
}

/**
 * @brief Save EM field component to file
 * 
 * @param field     Which field to save (E or B)
 * @param fc        Which field component to save (r, θ or z)
 * @param m         Mode
 */
void emf::save( emf::quantity const field, const fcomp::cyl fc, const int m ) const {
    std::string vfname;  // Dataset name
    std::string vflabel; // Dataset label (for plots)
    std::string path{"EMF"};

    switch (field ) {
        case quantity::e :
            vfname = "E" + std::to_string(m);
            vflabel = "E^" + std::to_string(m) + "_";
            break;
        case quantity::b :
            vfname = "B" + std::to_string(m);
            vflabel = "B^" + std::to_string(m) + "_";
            break;
        default:
            mpi::fatal( "Invalid quantity selected" );
    }

    switch ( fc ) {
        case( fcomp::cyl::z ) :
            vfname  += "z";
            vflabel += "z";
            break;
        case( fcomp::cyl::r ) :
            vfname  += "r";
            vflabel += "r";
            break;
        case( fcomp::cyl::th ) :
            vfname  += "θ";
            vflabel += "\\theta";
            break;
        default:
            mpi::fatal( "Invalid field component (fc) selected" );
    }

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

    zdf::grid_info info = {
        .name = (char *) vfname.c_str(),
        .label = (char *) vflabel.c_str(),
        .units = (char *) "m_e c \\omega_n e^{-1}",
        .axis = axis
    };

    zdf::iteration iteration = {
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_n"
    };

    switch (field ) {
        case quantity::e :
            E -> save( m, fc, info, iteration, path );
            break;
        case quantity::b :
            B -> save( m, fc, info, iteration, path );
            break;
        default:
            mpi::fatal( "Invalid quantity selected" );
    }
}

namespace kernel {

__global__
void get_energy( 
    grid::tiled_cyl3_view<float> E_grid,
    grid::tiled_cyl3_view<float> B_grid,
    double dz, double dr,
    double * const __restrict__ d_energy ) {

    const uint2  tile_idx = { blockIdx.x, blockIdx.y };

    cyl_float3 * const __restrict__ tile_E = E_grid.tile_data(tile_idx);
    cyl_float3 * const __restrict__ tile_B = B_grid.tile_data(tile_idx);

    auto tile_dims = E_grid.tile_dims;
    const int jstride = E_grid.tile_ystride();
    const int ir0 = ( E_grid.local_tile_start.y + tile_idx.y ) * tile_dims.y;

    cyl_double3 tile_ene_E{0,0,0};
    cyl_double3 tile_ene_B{0,0,0};

    // Axial cells are not included
    int const j0 = ( ir0 > 0 ) ? 0 : 1;
    int const range_z = tile_dims.x;
    int const range_r = tile_dims.y - j0;

    for( int idx = gpu::block::thread_rank(); idx < range_z * range_r; idx += gpu::block::num_threads() ) {
        int const i = idx % tile_dims.x;
        int const j = idx / tile_dims.x + j0;

        double rc = ( j + ir0       ) * dr;
        double rm = ( j + ir0 - 0.5 ) * dr;

        auto efld = tile_E[ j * jstride + i ];
        auto bfld = tile_B[ j * jstride + i ];

        tile_ene_E.r += rc * efld.r * efld.r;
        tile_ene_E.th += rm * efld.th * efld.th;
        tile_ene_E.z += rm * efld.z * efld.z;

        tile_ene_B.r += rm * bfld.r * bfld.r;
        tile_ene_B.th += rc * bfld.th * bfld.th;
        tile_ene_B.z += rc * bfld.z * bfld.z;
    }

    // Add up energy from all warps
    tile_ene_E.r  = gpu::warp::reduce_add( tile_ene_E.r );
    tile_ene_E.th = gpu::warp::reduce_add( tile_ene_E.th );
    tile_ene_E.z  = gpu::warp::reduce_add( tile_ene_E.z );
    
    tile_ene_B.r  = gpu::warp::reduce_add( tile_ene_B.r );
    tile_ene_B.th = gpu::warp::reduce_add( tile_ene_B.th );
    tile_ene_B.z  = gpu::warp::reduce_add( tile_ene_B.z );

    if ( gpu::warp::thread_rank() == 0 ) {
       gpu::device::atomic_fetch_add( &(d_energy[0]), tile_ene_E.r );
       gpu::device::atomic_fetch_add( &(d_energy[1]), tile_ene_E.th );
       gpu::device::atomic_fetch_add( &(d_energy[2]), tile_ene_E.z );

       gpu::device::atomic_fetch_add( &(d_energy[3]), tile_ene_B.r );
       gpu::device::atomic_fetch_add( &(d_energy[4]), tile_ene_B.th );
       gpu::device::atomic_fetch_add( &(d_energy[5]), tile_ene_B.z );
    }
}

__global__
void get_energy( 
    grid::tiled_cyl3_view<complex_t> E_grid,
    grid::tiled_cyl3_view<complex_t> B_grid,
    double dz, double dr,
    double * const __restrict__ d_energy ) {

    const uint2  tile_idx = { blockIdx.x, blockIdx.y };

    cyl_cfloat3 * const __restrict__ tile_E = E_grid.tile_data(tile_idx);
    cyl_cfloat3 * const __restrict__ tile_B = B_grid.tile_data(tile_idx);

    auto tile_dims = E_grid.tile_dims;
    const int jstride = E_grid.tile_ystride();
    const int ir0 = ( E_grid.local_tile_start.y + tile_idx.y ) * tile_dims.y;

    cyl_double3 tile_ene_E{0,0,0};
    cyl_double3 tile_ene_B{0,0,0};

    // Axial cells are not included
    int const j0 = ( ir0 > 0 ) ? 0 : 1;
    int const range_z = tile_dims.x;
    int const range_r = tile_dims.y - j0;

    for( int idx = gpu::block::thread_rank(); idx < range_z * range_r; idx += gpu::block::num_threads() ) {
        int const i = idx % tile_dims.x;
        int const j = idx / tile_dims.x + j0;

        double rc = ( j + ir0       ) * dr;
        double rm = ( j + ir0 - 0.5 ) * dr;

        auto efld = tile_E[ j * jstride + i ];
        auto bfld = tile_B[ j * jstride + i ];

        tile_ene_E.r += rc * norm( efld.r );
        tile_ene_E.th += rm * norm( efld.th );
        tile_ene_E.z += rm * norm( efld.z );

        tile_ene_B.r += rm * norm( bfld.r );
        tile_ene_B.th += rc * norm( bfld.th );
        tile_ene_B.z += rc * norm( bfld.z );
    }

    // Add up energy from all warps
    tile_ene_E.r  = gpu::warp::reduce_add( tile_ene_E.r );
    tile_ene_E.th = gpu::warp::reduce_add( tile_ene_E.th );
    tile_ene_E.z  = gpu::warp::reduce_add( tile_ene_E.z );

    tile_ene_B.r  = gpu::warp::reduce_add( tile_ene_B.r );
    tile_ene_B.th = gpu::warp::reduce_add( tile_ene_B.th );
    tile_ene_B.z  = gpu::warp::reduce_add( tile_ene_B.z );

    if ( gpu::warp::thread_rank() == 0 ) {
        gpu::device::atomic_fetch_add( &(d_energy[0]), tile_ene_E.r );
        gpu::device::atomic_fetch_add( &(d_energy[1]), tile_ene_E.th );
        gpu::device::atomic_fetch_add( &(d_energy[2]), tile_ene_E.z );

        gpu::device::atomic_fetch_add( &(d_energy[3]), tile_ene_B.r );
        gpu::device::atomic_fetch_add( &(d_energy[4]), tile_ene_B.th );
        gpu::device::atomic_fetch_add( &(d_energy[5]), tile_ene_B.z );
    }
}

} // namespace kernel


/**
 * @brief Get EM field energy
 * 
 * @note The energy will be recalculated each time this routine is called;
 * @note Axial cell values are currently being ignored
 * @note Fields are expanded as F(θ) = F_0 + Re Σ F_m exp(i m θ), so the θ
 *       average of |Re(F_m exp(i m θ))|^2 is |F_m|^2 / 2 for m > 0
 * 
* @param ene_E     Electric field energy
* @param ene_b     Magnetic field energy
* @param m         Mode
*/
void emf::get_energy( cyl_double3 & ene_E, cyl_double3 & ene_B, const int m ) const {

    // Get tile information from mode0
    auto & E0 = E -> mode0();
    auto & B0 = B -> mode0();

    const auto ntiles    = E0.get_local_ntiles();

    // Zero energy values
    gpu::device::zero( d_energy, 6 );

    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 1024 );

    if ( m < 0 || m >= nmodes ) {
        mpi::fatal( "Invalid mode (" + std::to_string(m) + ") requested" );
    }

    if ( m == 0 ) {
        kernel::get_energy <<< grid, block >>> ( 
            E0.view(), B0.view(), dx.x, dx.y, d_energy
        );
    } else {

        auto & Em = E -> mode(m);
        auto & Bm = B -> mode(m);

        kernel::get_energy <<< grid, block >>> ( 
            Em.view(), Bm.view(),
            dx.x, dx.y, d_energy
        );

    }

    // Copy results to host and normalize
    double h_energy[6];
    gpu::device::memcpy_tohost( h_energy, d_energy, 6 );

    ene_E = cyl_double3{ h_energy[0], h_energy[1], h_energy[2] };
    ene_B = cyl_double3{ h_energy[3], h_energy[4], h_energy[5] };

    const double dz = dx.x;
    const double dr = dx.y;
    // Mode 0: (1/2) |F|^2 2π r Δr Δz; modes m > 0: (1/2) (|F_m|^2/2) 2π r Δr Δz
    const double norm = ( m == 0 ) ? dz * dr * M_PI : 0.5 * dz * dr * M_PI;
    ene_E *= norm;
    ene_B *= norm;
}
