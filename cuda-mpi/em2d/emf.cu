#include "emf.hpp"

#include <iostream>
#include "grid/tiled_vec3.cuh"

#define opt_yee_block 256

namespace kernel {

/**
 * @brief B advance for Yee algorithm
 * 
 * @param E         Pointer to 0,0 coordinate of E field
 * @param B         Pointer to 0,0 coordinate of B field
 * @param tile_dims Internal tile size
 * @param ystride   Stride for y coordinate
 * @param dt_dx     \Delta t / \Delta x
 * @param dt_dy     \Delta t / \Delta y
 */
__device__ 
void yee_b( 
    float3 const * const __restrict__ E, 
    float3 * const __restrict__ B, 
    uint2 const tile_dims, int const ystride, 
    float2 const dt_dx )
{
    int const range_x = tile_dims.x + 2;
    int const range_y = tile_dims.y + 2;
    
    for( int idx = gpu::block::thread_rank(); idx < range_y * range_x; idx += gpu::block::num_threads() ) {
        const int ix = -1 + idx % range_x;
        const int iy = -1 + idx / range_x;
   
        B[ ix + iy*ystride ].x += ( - dt_dx.y * ( E[ix + (iy+1)*ystride].z - E[ix + iy*ystride].z ) );  
        B[ ix + iy*ystride ].y += (   dt_dx.x * ( E[(ix+1) + iy*ystride].z - E[ix + iy*ystride].z ) );  
        B[ ix + iy*ystride ].z += ( - dt_dx.x * ( E[(ix+1) + iy*ystride].y - E[ix + iy*ystride].y ) + 
                                        dt_dx.y * ( E[ix + (iy+1)*ystride].x - E[ix + iy*ystride].x ) );  
    }
}

/**
 * @brief E advance for Yee algorithm ( no current )
 * 
 * @param E         Pointer to 0,0 coordinate of E field
 * @param B         Pointer to 0,0 coordinate of B field
 * @param nx        Internal tile size
 * @param ystride    Stride for y coordinate
 * @param dt_dx     \Delta t / \Delta x
 * @param dt_dy     \Delta t / \Delta y
 */
__device__ 
void yee_e( 
    float3 * const __restrict__ E, 
    float3 const * const __restrict__ B, 
    uint2 const tile_dims, int const ystride, 
    float2 const dt_dx )
{
    int const range_x = tile_dims.x + 2;
    int const range_y = tile_dims.y + 2;

    for( int idx = gpu::block::thread_rank(); idx < range_y * range_x; idx += gpu::block::num_threads() ) {
        const int ix = idx % range_x;
        const int iy = idx / range_x;

        E[ix + iy*ystride].x += ( + dt_dx.y * ( B[ix + iy*ystride].z - B[ix + (iy-1)*ystride].z) );
        
        E[ix + iy*ystride].y += ( - dt_dx.x * ( B[ix + iy*ystride].z - B[(ix-1) + iy*ystride].z) );

        E[ix + iy*ystride].z += ( + dt_dx.x * ( B[ix + iy*ystride].y - B[(ix-1) + iy*ystride].y) - 
                                    dt_dx.y * ( B[ix + iy*ystride].x - B[ix + (iy-1)*ystride].x) );
    }
}

/**
 * @brief Enforce zero tangential E on PEC wall nodes
 * 
 * The bc kernels only run after the full B/2 - E - B/2 step. Without this
 * the second B half step would use the tangential E evolved on the wall
 * node (from the wall node current), polluting B next to the wall.
 * 
 * @note Must be called after the E update and before the second B half
 *       step, with a block synchronization before and after.
 * 
 * @note Corner cells may be written by both the x and y loops; both write
 *       zero to the shared (z) component, so this is benign.
 * 
 * @param E             Pointer to 0,0 coordinate of E field (shared memory)
 * @param tile_idx      Tile index
 * @param local_ntiles  Local number of tiles
 * @param tile_dims     Internal tile size
 * @param ystride       Stride for y coordinate
 * @param bc            Local rank boundary conditions
 */
__device__
void pec_walls( 
    float3 * const __restrict__ E, 
    uint2 const tile_idx, uint2 const local_ntiles,
    uint2 const tile_dims, int const ystride,
    emf::bc_type const bc )
{
    const int nx = tile_dims.x;
    const int ny = tile_dims.y;

    // E is updated on rows / columns 0 .. n+1
    if ( tile_idx.x == 0 && bc.x.lower == emf::bc::pec ) {
        for( int i = gpu::block::thread_rank(); i < ny + 2; i += gpu::block::num_threads() ) {
            E[ i * ystride ].y = 0;
            E[ i * ystride ].z = 0;
        }
    }

    if ( tile_idx.x == local_ntiles.x - 1 && bc.x.upper == emf::bc::pec ) {
        for( int i = gpu::block::thread_rank(); i < ny + 2; i += gpu::block::num_threads() ) {
            E[ nx + i * ystride ].y = 0;
            E[ nx + i * ystride ].z = 0;
        }
    }

    if ( tile_idx.y == 0 && bc.y.lower == emf::bc::pec ) {
        for( int i = gpu::block::thread_rank(); i < nx + 2; i += gpu::block::num_threads() ) {
            E[ i ].x = 0;
            E[ i ].z = 0;
        }
    }

    if ( tile_idx.y == local_ntiles.y - 1 && bc.y.upper == emf::bc::pec ) {
        for( int i = gpu::block::thread_rank(); i < nx + 2; i += gpu::block::num_threads() ) {
            E[ i + ny * ystride ].x = 0;
            E[ i + ny * ystride ].z = 0;
        }
    }
}

/**
 * @brief Complete Yee step (no current)
 * 
 * @param E_grid    Electric field
 * @param B_grid    Magnetic field
 * @param dt_dx     time step over cell size ratio
 */
__global__ __launch_bounds__(opt_yee_block) 
void yee( 
    grid::tiled_vec3_view<float> E_grid,
    grid::tiled_vec3_view<float> B_grid,
    float2 const dt_dx, emf::bc_type const local_bc ) {

    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int tile_id   = tile_idx.y * E_grid.local_ntiles.x + tile_idx.x;
    const uint2 tile_dims = E_grid.tile_dims;

    // Copy E and B into shared memory
    auto * shm = gpu::block::shared_mem<float3>();
    float3 * __restrict__ E_local = & shm[0];
    float3 * __restrict__ B_local = & shm[ E_grid.tile_vol ];

    gpu::block::memcpy(E_local, E_grid.tile_buffer(tile_id), E_grid.tile_vol);
    gpu::block::memcpy(B_local, B_grid.tile_buffer(tile_id), B_grid.tile_vol);

    gpu::block::sync();

    float3 * const __restrict__ E = & E_local[ E_grid.inner_offset ];
    float3 * const __restrict__ B = & B_local[ B_grid.inner_offset ];

    const int ystride = E_grid.tile_ystride();
    const float2 dt_dx_2 = { dt_dx.x / 2, dt_dx.y / 2 };

    // Half B advance
    yee_b( E, B, tile_dims, ystride, dt_dx_2 );
    gpu::block::sync();

    // Full E advance
    yee_e( E, B, tile_dims, ystride, dt_dx );
    gpu::block::sync();

    // Keep tangential E zero on PEC walls
    pec_walls( E, tile_idx, E_grid.local_ntiles, tile_dims, ystride, local_bc );
    gpu::block::sync();

    // Half B advance
    yee_b( E, B, tile_dims, ystride, dt_dx_2 );
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

    const auto ntiles   = E -> get_local_ntiles();

    float2 const dt_dx   = make_float2( dt/dx.x, dt/dx.y );

    dim3 grid( ntiles.x, ntiles.y );
    auto block = opt_yee_block;
    size_t shm_size = 2 * E -> tile_vol * sizeof(float3);

    gpu::block::set_shmem_size( kernel::yee, shm_size );
    kernel::yee <<< grid, block, shm_size >>> (
        E -> view(), B -> view(), dt_dx, local_bc
    );

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
 * @brief Physical boundary conditions for the x direction
 * 
 * @note Launch with grid( 2, local_ntiles.y ): blockIdx.x selects the
 *       boundary (0 - lower, 1 - upper), blockIdx.y the tile row
 * 
 * @param E_grid        View of tiled electric field grid
 * @param B_grid        View of tiled magnetic field grid
 * @param local_bc      Local rank boundary conditions
 */
__global__
void emf_bcx( 
    grid::tiled_vec3_view<float> E_grid,
    grid::tiled_vec3_view<float> B_grid,
    emf::bc_type local_bc ) {

    int bnd_x = blockIdx.x;
    int tile_idx_y = blockIdx.y;
    const int ystride = E_grid.tile_ystride();

    if ( bnd_x == 0 ) {
        // Lower boundary
        float3 * __restrict__ E = & E_grid.tile_buffer
            (0,tile_idx_y)      // lower x boundary tile
            [ E_grid.gc.x.lower ];     // point to first x cell (ix = 0)
        float3 * __restrict__ B = & B_grid.tile_buffer
            (0,tile_idx_y)
            [ B_grid.gc.x.lower ];

        switch( local_bc.x.lower ) {
        case( emf::bc::pmc) :
            for( unsigned idx = gpu::block::thread_rank(); 
                 idx < E_grid.tile_ext_dims.y;
                 idx += gpu::block::num_threads() ) {
                // iy includes the y-stride
                const int iy = idx * ystride;

                E[ -1 + iy ].x = -E[ 0 + iy ].x;
                E[ -1 + iy ].y =  E[ 1 + iy ].y;
                E[ -1 + iy ].z =  E[ 1 + iy ].z;

                B[ -1 + iy ].x =  B[ 1 + iy ].x;
                B[ -1 + iy ].y = -B[ 0 + iy ].y;
                B[ -1 + iy ].z = -B[ 0 + iy ].z;
            }
            break;

        case( emf::bc::pec ) :
            for( unsigned idx = gpu::block::thread_rank(); 
                 idx < E_grid.tile_ext_dims.y;
                 idx += gpu::block::num_threads() ) {
                const int iy = idx * ystride;

                E[ -1 + iy ].x =  E[ 0 + iy ].x;
                E[ -1 + iy ].y = -E[ 1 + iy ].y;
                E[ -1 + iy ].z = -E[ 1 + iy ].z;

                E[  0 + iy ].y = 0;
                E[  0 + iy ].z = 0;

                B[ -1 + iy ].x = -B[ 1 + iy ].x;
                B[ -1 + iy ].y =  B[ 0 + iy ].y;
                B[ -1 + iy ].z =  B[ 0 + iy ].z;

                B[  0 + iy ].x = 0;
            }
            break;
        default:
            break;
        } 
    } else {
        // Upper boundary
        float3 * __restrict__ E = & E_grid.tile_buffer
            (E_grid.local_ntiles.x-1,tile_idx_y)    // upper x boundary tile
            [ E_grid.gc.x.lower + E_grid.tile_dims.x ];    // point to first upper gc (ix = tile_dims.x)
        float3 * __restrict__ B = & B_grid.tile_buffer
            (B_grid.local_ntiles.x-1,tile_idx_y)    // upper x boundary tile
            [ B_grid.gc.x.lower + B_grid.tile_dims.x ];    // point to first upper gc (ix = tile_dims.x)

        switch( local_bc.x.upper ) {
        case( emf::bc::pmc) :
            for( unsigned idx = gpu::block::thread_rank();
                 idx < E_grid.tile_ext_dims.y; 
                 idx += gpu::block::num_threads() ) {
                const int iy = idx * ystride;

                E[ iy ].x = -E[ -1 + iy ].x;
                //E[ iy ].y =  E[ 0 + iy ].y;
                //E[ iy ].z =  E[ 0 + iy ].z;

                E[ 1 + iy ].x = -E[ -2 + iy ].x;
                E[ 1 + iy ].y =  E[ -1 + iy ].y;
                E[ 1 + iy ].z =  E[ -1 + iy ].z;

                // B[  iy ].x = -B[ iy ].x;
                B[ iy ].y = -B[ -1 + iy ].y;
                B[ iy ].z = -B[ -1 + iy ].z;

                B[ 1 + iy ].x =  B[ -1 + iy ].x;
                B[ 1 + iy ].y = -B[ -2 + iy ].y;
                B[ 1 + iy ].z = -B[ -2 + iy ].z;
            }
            break;

        case( emf::bc::pec) :
            for( unsigned idx = gpu::block::thread_rank();
                 idx < E_grid.tile_ext_dims.y; 
                 idx += gpu::block::num_threads() ) {
                const int iy = idx * ystride;

                E[ iy ].x =  E[ -1 + iy ].x;
                E[ iy ].y =  0;
                E[ iy ].z =  0;

                E[ 1 + iy ].x =  E[ -2 + iy ].x;
                E[ 1 + iy ].y = -E[ -1 + iy ].y;
                E[ 1 + iy ].z = -E[ -1 + iy ].z;

                B[ iy ].x =  0;
                B[ iy ].y =  B[ -1 + iy ].y;
                B[ iy ].z =  B[ -1 + iy ].z;

                B[ 1 + iy ].x = -B[ -1 + iy ].x;
                B[ 1 + iy ].y =  B[ -2 + iy ].y;
                B[ 1 + iy ].z =  B[ -2 + iy ].z;
            }
            break;
        default:
            break;
        }
    }
}

/**
 * @brief Physical boundary conditions for the y direction
 * 
 * @note Launch with grid( local_ntiles.x, 2 ): blockIdx.x selects the tile
 *       column, blockIdx.y the boundary (0 - lower, 1 - upper)
 * 
 * @param E_grid        View of tiled electric field grid
 * @param B_grid        View of tiled magnetic field grid
 * @param local_bc      Local rank boundary conditions
 */
__global__
void emf_bcy( 
    grid::tiled_vec3_view<float> E_grid,
    grid::tiled_vec3_view<float> B_grid,
    emf::bc_type local_bc ) {

    int tile_idx_x = blockIdx.x;
    int bnd_y = blockIdx.y;
    const int ystride = E_grid.tile_ystride();
    
    if ( bnd_y == 0 ) {
        // Lower boundary
        float3 * __restrict__ E = & E_grid.tile_buffer
            (tile_idx_x,0)              // lower y boundary tiles
            [ E_grid.gc.y.lower * ystride ];   // point to first y cell (iy = 0)
        float3 * __restrict__ B = & B_grid.tile_buffer
            (tile_idx_x,0)
            [ B_grid.gc.y.lower * ystride ];
        
        switch( local_bc.y.lower ) {
        case( emf::bc::pmc) :
            for( unsigned idx = gpu::block::thread_rank();
                 idx < E_grid.tile_ext_dims.x; 
                 idx += gpu::block::num_threads() ) {
                const int ix = idx;

                E[ ix - ystride ].x =  E[ ix + ystride ].x;
                E[ ix - ystride ].y = -E[ ix +       0 ].y;
                E[ ix - ystride ].z =  E[ ix + ystride ].z;

                B[ ix - ystride ].x = -B[ ix +       0 ].x;
                B[ ix - ystride ].y =  B[ ix + ystride ].y;
                B[ ix - ystride ].z = -B[ ix +       0 ].z;
            }
            break;

        case( emf::bc::pec ) :
            for( unsigned idx = gpu::block::thread_rank();
                 idx < E_grid.tile_ext_dims.x; 
                 idx += gpu::block::num_threads() ) {
                const int ix = idx;

                E[ ix - ystride ].x = -E[ ix + ystride ].x;
                E[ ix - ystride ].y =  E[ ix +       0 ].y;
                E[ ix - ystride ].z = -E[ ix + ystride ].z;

                E[ ix +       0 ].x = 0;
                E[ ix +       0 ].z = 0;
                
                B[ ix - ystride ].x =  B[ ix +       0 ].x;
                B[ ix - ystride ].y = -B[ ix + ystride ].y;
                B[ ix - ystride ].z =  B[ ix +       0 ].z;

                B[ ix +       0 ].y = 0;
            }
            break;
        default:
            break;
        }
    } else {
        // Upper boundary
        float3 * __restrict__ E = & E_grid.tile_buffer
            (tile_idx_x,E_grid.local_ntiles.y-1)   // upper y boundary tiles
            [ (E_grid.gc.y.lower + E_grid.tile_dims.y ) * ystride ];  // point to first upper gc (iy = tile_dims.y)
        float3 * __restrict__ B = & B_grid.tile_buffer
            (tile_idx_x,B_grid.local_ntiles.y-1)
            [ (B_grid.gc.y.lower + B_grid.tile_dims.y ) * ystride ]; 

        switch( local_bc.y.upper ) {
        case( emf::bc::pmc) :
            for( unsigned idx = gpu::block::thread_rank();
                 idx < E_grid.tile_ext_dims.x; 
                 idx += gpu::block::num_threads() ) {
                const int ix = idx;

                E[ ix ].y = -E[ ix - 1 * ystride ].y;

                E[ ix + ystride ].x =  E[ ix -1 * ystride ].x;
                E[ ix + ystride ].y = -E[ ix -2 * ystride ].y;
                E[ ix + ystride ].z =  E[ ix -1 * ystride ].z;

                B[ ix ].x = -B[ ix - 1*ystride ].x;
                B[ ix ].z = -B[ ix - 1*ystride ].z;

                B[ ix + ystride ].x = -B[ ix - 2 * ystride ].x;
                B[ ix + ystride ].y =  B[ ix - 1 * ystride ].y;
                B[ ix + ystride ].z = -B[ ix - 2 * ystride ].z;
            }
            break;

        case( emf::bc::pec) :
            for( unsigned idx = gpu::block::thread_rank();
                 idx < E_grid.tile_ext_dims.x; 
                 idx += gpu::block::num_threads() ) {
                const int ix = idx;

                E[ ix ].x =  0;
                E[ ix ].y =  E[ ix - 1 * ystride ].y;
                E[ ix ].z =  0;

                E[ ix + 1 * ystride ].x = -E[ ix- 1 * ystride ].x;
                E[ ix + 1 * ystride ].y =  E[ ix- 2 * ystride ].y;
                E[ ix + 1 * ystride ].z = -E[ ix- 1 * ystride ].z;

                B[ ix ].x =  B[ ix - 1 * ystride ].x;
                B[ ix ].y =  0;
                B[ ix ].z =  B[ ix - 1 * ystride ].z;

                B[ ix + 1 * ystride ].x =  B[ ix - 2 * ystride ].x;
                B[ ix + 1 * ystride ].y = -B[ ix - 1 * ystride ].y;
                B[ ix + 1 * ystride ].z =  B[ ix - 2 * ystride ].z;
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

    dim3 block( 64 );
    const uint2 ntiles          = E -> get_local_ntiles();

    // x boundaries
    if ( is_conducting( local_bc.x.lower ) || is_conducting( local_bc.x.upper ) ) {
        dim3 grid( 2, ntiles.y );
        kernel::emf_bcx <<< grid, block >>> (
            E -> view(), B -> view(), local_bc
        );
    }

    // y boundaries (must run after x boundaries for correct corner values)
    if ( is_conducting( local_bc.y.lower ) || is_conducting( local_bc.y.upper ) ) {
        dim3 grid( ntiles.x, 2 );
        kernel::emf_bcy <<< grid, block >>> (
            E -> view(), B -> view(), local_bc
        );
    }
}

namespace kernel {

/**
 * @brief E advance for Yee algorithm including current
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
__device__ void yeeJ_e( 
    float3 * const __restrict__ E, 
    float3 const * const __restrict__ B, 
    uint2 const tile_dims, int const ystride, 
    float3 const * const __restrict__ J, int const J_ystride, 
    float2 const dt_dx, float const dt )
{
    int const range_x = tile_dims.x + 2;
    int const range_y = tile_dims.y + 2;

    for( int idx = gpu::block::thread_rank(); idx < range_y * range_x; idx += gpu::block::num_threads() ) {
        const int ix = idx % range_x;
        const int iy = idx / range_x;

        E[ix + iy*ystride].x += ( + dt_dx.y * ( B[ix + iy*ystride].z - B[ix + (iy-1)*ystride].z) )
                                - dt * J[ix + iy*J_ystride].x;

        E[ix + iy*ystride].y += ( - dt_dx.x * ( B[ix + iy*ystride].z - B[(ix-1) + iy*ystride].z) )
                                - dt * J[ix + iy*J_ystride].y;

        E[ix + iy*ystride].z += ( + dt_dx.x * ( B[ix + iy*ystride].y - B[(ix-1) + iy*ystride].y) - 
                                    dt_dx.y * ( B[ix + iy*ystride].x - B[ix + (iy-1)*ystride].x) )
                                - dt * J[ix + iy*J_ystride ].z;
    }
}

/**
 * @brief Complete Yee step including current
 * 
 * @param E_grid    Electric field
 * @param B_grid    Magnetic field
 * @param J_grid    Current density (read only)
 * @param dt_dx     time step over cell size ratio
 * @param dt        time step
 * @param local_bc  Local rank boundary conditions
 */
__global__ __launch_bounds__(opt_yee_block) void yeeJ( 
    grid::tiled_vec3_view<float> E_grid,
    grid::tiled_vec3_view<float> B_grid,
    grid::tiled_view<const float3> J_grid,
    float2 const dt_dx, float const dt,
    emf::bc_type const local_bc ) {

    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int tile_id   = tile_idx.y * E_grid.local_ntiles.x + tile_idx.x;
    const uint2 tile_dims = E_grid.tile_dims;

    // Copy E and B into shared memory
    auto * shm = gpu::block::shared_mem<float3>();
    float3 * __restrict__ E_local = & shm[0];
    float3 * __restrict__ B_local = & shm[ E_grid.tile_vol ];

    gpu::block::memcpy(E_local, E_grid.tile_buffer(tile_id), E_grid.tile_vol);
    gpu::block::memcpy(B_local, B_grid.tile_buffer(tile_id), B_grid.tile_vol);

    gpu::block::sync();

    float3 * const __restrict__ E = & E_local[ E_grid.inner_offset ];
    float3 * const __restrict__ B = & B_local[ B_grid.inner_offset ];
    float3 const * const __restrict__ J = J_grid.tile_data(tile_id);

    const int ystride = E_grid.tile_ystride();
    const int J_ystride = J_grid.tile_ystride();
    const float2 dt_dx_2 = { dt_dx.x / 2, dt_dx.y / 2 };

    // Half B advance
    yee_b( E, B, tile_dims, ystride, dt_dx_2 );
    gpu::block::sync();

    // Full E advance including current
    yeeJ_e( E, B, tile_dims, ystride, J, J_ystride, dt_dx, dt );
    gpu::block::sync();

    // Keep tangential E zero on PEC walls
    pec_walls( E, tile_idx, E_grid.local_ntiles, tile_dims, ystride, local_bc );
    gpu::block::sync();

    // Half B advance
    yee_b( E, B, tile_dims, ystride, dt_dx_2 );
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

    const auto ntiles   = E -> get_local_ntiles();
    float2 const dt_dx   = make_float2( dt/dx.x, dt/dx.y );

    dim3 grid( ntiles.x, ntiles.y );
    auto block = opt_yee_block;
    size_t shm_size = 2 * E -> tile_vol * sizeof(float3);

    gpu::block::set_shmem_size( kernel::yeeJ, shm_size );
    kernel::yeeJ <<< grid, block, shm_size >>> (
        E -> view(), B -> view(), current.J -> cview(),
        dt_dx, dt, local_bc
    );

    // Update guard cells with new values
    E -> copy_to_gc( );
    B -> copy_to_gc( );

    // Do additional bc calculations if needed
    process_bc( );

    // Advance internal iteration number
    iter += 1;

    // Move simulation window if needed
    if ( mov_window.active() ) move_window( );
}

/**
 * @brief Save EMF data to diagnostic file
 * 
 * @param quant     Field to save (emf::quantity::e or emf::quantity::b)
 * @param fc        Field component to save (fcomp::x, fcomp::y or fcomp::z)
 */
void emf::save( const emf::quantity quant, const fcomp::cart fc ) const {

    std::string vfname;  // Dataset name
    std::string vflabel; // Dataset label (for plots)

    grid::tiled_vec3<float> * f = nullptr;

    switch ( quant ) {
        case quantity::e :
            f = E;
            vfname = "E";
            vflabel = "E_";
            break;
        case quantity::b :
            f = B;
            vfname = "B";
            vflabel = "B_";
            break;
        default:
            mpi::fatal( "Invalid quantity selected" );
    }

    switch ( fc ) {
        case( fcomp::x ) :
            vfname  += 'x';
            vflabel += 'x';
            break;
        case( fcomp::y ) :
            vfname  += 'y';
            vflabel += 'y';
            break;
        case( fcomp::z ) :
            vfname  += 'z';
            vflabel += 'z';
            break;
        default:
            mpi::fatal( "Invalid field component (fc) selected" );
    }

    zdf::grid_axis axis[2];
    axis[0] = zdf::grid_axis {
        .name = (char *) "x",
        .min = mov_window.motion(),
        .max = box.x + mov_window.motion(),
        .label = (char *) "x",
        .units = (char *) "c/\\omega_n"
    };

    axis[1] = zdf::grid_axis {
        .name = (char *) "y",
        .min = 0.0,
        .max = box.y,
        .label = (char *) "y",
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

    f -> save( fc, info, iteration, "EMF" );
}

namespace kernel {

__global__
void get_energy( 
    grid::tiled_view<const float3> E_fld, grid::tiled_view<const float3> B_fld,
    double * const __restrict__ d_energy ) {

    const uint2  tile_idx = { blockIdx.x, blockIdx.y };

    float3 const * const __restrict__ E = E_fld.tile_data( tile_idx );
    float3 const * const __restrict__ B = B_fld.tile_data( tile_idx );

    const int ystride = E_fld.tile_ystride();

    double3 ene_E = double3{0};
    double3 ene_B = double3{0};

    auto nx = E_fld.tile_dims;

    for( int idx = gpu::block::thread_rank(); idx < nx.y * nx.x; idx += gpu::block::num_threads() ) {
        int const i = idx % nx.x;
        int const j = idx / nx.x;

        float3 const efld = E[ j * ystride + i ];
        float3 const bfld = B[ j * ystride + i ];

        ene_E.x += static_cast<double>(efld.x) * efld.x;
        ene_E.y += static_cast<double>(efld.y) * efld.y;
        ene_E.z += static_cast<double>(efld.z) * efld.z;

        ene_B.x += static_cast<double>(bfld.x) * bfld.x;
        ene_B.y += static_cast<double>(bfld.y) * bfld.y;
        ene_B.z += static_cast<double>(bfld.z) * bfld.z;
    }

    // Add up energy from all warps
    ene_E.x = gpu::warp::reduce_add( ene_E.x );
    ene_E.y = gpu::warp::reduce_add( ene_E.y );
    ene_E.z = gpu::warp::reduce_add( ene_E.z );

    ene_B.x = gpu::warp::reduce_add( ene_B.x );
    ene_B.y = gpu::warp::reduce_add( ene_B.y );
    ene_B.z = gpu::warp::reduce_add( ene_B.z );

    if ( gpu::warp::thread_rank() == 0 ) {
        gpu::device::atomic_fetch_add( &(d_energy[0]), ene_E.x );
        gpu::device::atomic_fetch_add( &(d_energy[1]), ene_E.y );
        gpu::device::atomic_fetch_add( &(d_energy[2]), ene_E.z );

        gpu::device::atomic_fetch_add( &(d_energy[3]), ene_B.x );
        gpu::device::atomic_fetch_add( &(d_energy[4]), ene_B.y );
        gpu::device::atomic_fetch_add( &(d_energy[5]), ene_B.z );
    }
}

}


/**
 * @brief Get total field energy per field component
 * 
 * @warning This function will always recalculate the energy each time it is
 *          called.
 * 
 * @param ene_E     Total E-field energy (per component)
 * @param ene_B     Total B-field energy (per component)
 */
void emf::get_energy( double3 & ene_E, double3 & ene_B ) const {

    // Zero energy values
    gpu::device::zero( d_energy, 6 );

    // Add up energy from all local cells
    dim3 grid( E->get_local_ntiles().x, E->get_local_ntiles().y );
    dim3 block( 1024 );
    kernel::get_energy <<< grid, block >>> ( 
        E->cview(), B->cview(), d_energy
    );

    // Copy results to host
    double h_energy[6];
    gpu::device::memcpy_tohost( h_energy, d_energy, 6 );

    // Sum over all parallel nodes
    E -> get_part().allreduce( h_energy, 6, MPI_SUM );

    // Normalize
    ene_E.x = 0.5 * dx.x * dx.y * h_energy[0];
    ene_E.y = 0.5 * dx.x * dx.y * h_energy[1];
    ene_E.z = 0.5 * dx.x * dx.y * h_energy[2];

    ene_B.x = 0.5 * dx.x * dx.y * h_energy[3];
    ene_B.y = 0.5 * dx.x * dx.y * h_energy[4];
    ene_B.z = 0.5 * dx.x * dx.y * h_energy[5];

}