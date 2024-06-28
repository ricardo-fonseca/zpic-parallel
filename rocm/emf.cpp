#include "emf.h"

#include <iostream>
#include "zdf-cpp.h"

#define opt_yee_block 256

/**
 * @brief Construct a new EMF::EMF object
 * 
 * @param ntiles    Number of tiles
 * @param nx        Tile grid size
 * @param box       Simulation box size
 * @param dt        Time step
 * @param q         Sycl queue
 */
EMF::EMF( uint2 const ntiles, uint2 const nx, float2 const box,
    double const dt ) : 
    dx( float2{ box.x / ( nx.x * ntiles.x ), box.y / ( nx.y * ntiles.y ) } ),
    dt( dt ), box(box)
{
    // Verify Courant condition
    float cour = std::sqrt( 1.0f/( 1.0f/(dx.x*dx.x) + 1.0f/(dx.y*dx.y) ) );
    if ( dt >= cour ){
        std::cerr << "(*error*) Invalid timestep, courant condition violation.\n";
        std::cerr << "(*error*) For the current resolution [" << dx.x << "," << dx.y << "]\n";
        std::cerr << " the maximum timestep is dt = " << cour <<'\n';
        exit(-1);
    }

    // Guard cells (1 below, 2 above)
    // These are required for the Yee solver AND for field interpolation
    bnd<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {1,2};

    E = new vec3grid<float3> ( ntiles, nx, gc );
    E -> name = "Electric field";

    B = new vec3grid<float3> ( ntiles, nx, gc );
    B -> name = "Magnetic field";

    // Check that local memory can hold up to 2 times the tile buffer
    auto local_mem_size = block::shared_mem_size();
    if ( local_mem_size < 2 * E->tile_vol * sizeof( float3 ) ) {
        std::cerr << "(*error*) Tile size too large " << nx << " (plus guard cells)\n";
        std::cerr << "(*error*) Insufficient local memory (" << local_mem_size << " B) for EMF object.\n";
        abort();
    }

    // Zero fields
    E -> zero( );
    B -> zero( );

    // Reserve device memory for energy diagnostic
    d_energy = device::malloc<double>( 6 );

    // Set default boundary conditions to periodic
    bc = emf::bc_type (emf::bc::periodic);

    // Reset iteration number
    iter = 0;

}

/**
 * @brief Move simulation window
 * 
 * When using a moving simulation window checks if a window move is due
 * at the current iteration and if so shifts left the data, zeroing the
 * rightmost cells.
 * 
 */
void EMF::move_window() {

    if ( moving_window.needs_move( iter * dt ) ) {

        E->x_shift_left(1);
        B->x_shift_left(1);

        moving_window.advance();
    }
}

namespace kernel {

/**
 * @brief B advance for Yee algorithm
 * 
 * @param E         Pointer to 0,0 coordinate of E field
 * @param B         Pointer to 0,0 coordinate of B field
 * @param nx        Internal tile size
 * @param ystride   Stride for y coordinate
 * @param dt_dx     \Delta t / \Delta x
 * @param dt_dy     \Delta t / \Delta y
 */
__device__ void yee_b( 
    float3 const * const __restrict__ E, 
    float3 * const __restrict__ B, 
    uint2 const nx, int const ystride, 
    float2 const dt_dx )
{
    int const range_x = nx.x + 2;
    int const range_y = nx.y + 2;
    
    for( int idx = block_thread_rank(); idx < range_y * range_x; idx += block_num_threads() ) {
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
__device__ void yee_e( 
    float3 * const __restrict__ E, 
    float3 const * const __restrict__ B, 
    uint2 const nx, int const ystride, 
    float2 const dt_dx )
{
    int const range_x = nx.x + 2;
    int const range_y = nx.y + 2;

    for( int idx = block_thread_rank(); idx < range_y * range_x; idx += block_num_threads() ) {
        const int ix = idx % range_x;
        const int iy = idx / range_x;

        E[ix + iy*ystride].x += ( + dt_dx.y * ( B[ix + iy*ystride].z - B[ix + (iy-1)*ystride].z) );
        
        E[ix + iy*ystride].y += ( - dt_dx.x * ( B[ix + iy*ystride].z - B[(ix-1) + iy*ystride].z) );

        E[ix + iy*ystride].z += ( + dt_dx.x * ( B[ix + iy*ystride].y - B[(ix-1) + iy*ystride].y) - 
                                    dt_dx.y * ( B[ix + iy*ystride].x - B[ix + (iy-1)*ystride].x) );
    }
}

__global__ __launch_bounds__(opt_yee_block) void yee( 
    float3 * const __restrict__ E_buffer,
    float3 * const __restrict__ B_buffer,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx, unsigned int const offset, 
    float2 const dt_dx ) {

    auto * shm = block::shared_mem<float3>();

    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    const int ystride = ext_nx.x;

    const float2 dt_dx_2 = { dt_dx.x / 2, dt_dx.y / 2 };

    float3 * const __restrict__ E_local = & shm[ 0 ];
    float3 * const __restrict__ B_local = & shm[ tile_vol ];

    // Copy E and B into shared memory
    for( unsigned i = block_thread_rank(); i < tile_vol; i += block_num_threads() ) {
        E_local[i] = E_buffer[ tile_off + i ];
        B_local[i] = B_buffer[ tile_off + i ];
    }

    float3 * const __restrict__ tile_E = & E_local[ offset ];
    float3 * const __restrict__ tile_B = & B_local[ offset ];

    block_sync();

    yee_b( tile_E, tile_B, nx, ystride, dt_dx_2 );
    block_sync();

    yee_e( tile_E, tile_B, nx, ystride, dt_dx );
    block_sync();

    yee_b( tile_E, tile_B, nx, ystride, dt_dx_2 );
    block_sync();

    // Copy data to global memory
    for( unsigned i = block_thread_rank(); i < tile_vol; i += block_num_threads() ) {
        E_buffer[ tile_off + i ] = E_local[i];
        B_buffer[ tile_off + i ] = B_local[i];
    }

}

}

/**
 * @brief Advance EM fields 1 time step (no current)
 * 
 */
void EMF::advance() {

    float2 const dt_dx   = make_float2( dt/dx.x, dt/dx.y );

    dim3 grid( E -> ntiles.x, E -> ntiles.y );
    auto block = opt_yee_block;
    size_t shm_size = 2 * E -> tile_vol * sizeof(float3);

    auto err = block::set_shmem_size( kernel::yee, shm_size );
    kernel::yee <<< grid, block, shm_size >>> (
        E -> d_buffer, B -> d_buffer, 
        E -> ntiles, E -> nx, E -> ext_nx, E -> offset,
        dt_dx
    );

    // Update guard cells with new values
    E -> copy_to_gc();
    B -> copy_to_gc();

    // Do additional bc calculations if needed
    process_bc();

    // Advance internal iteration number
    iter += 1;

    // Move simulation window if needed
    if ( moving_window.active() ) move_window();

}

namespace kernel {

__global__
/**
 * @brief Physical boundary conditions for the x direction
 * 
 * @param tile_idx      Tile position on grid
 * @param E         Tile E field
 * @param B         Tile B field
 * @param nx        Number of cells
 * @param ext_nx    Number of cells including guard cells
 * @param bc        Boundary condition
 */
void emf_bcx( 
    float3 * const __restrict__ E_buffer, float3 * const __restrict__ B_buffer,
    uint2 const ntiles, const uint2 nx, const uint2 ext_nx, bnd<unsigned int> gc, 
    emf::bc_type bc ) {

    const uint2  tile_idx = { blockIdx.x * ( ntiles.x - 1 ), blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    const int ystride = ext_nx.x;
    // Start at x cell 0
    const auto x_offset = gc.x.lower;

    float3 * const __restrict__ E = & E_buffer[ tile_off + x_offset ];
    float3 * const __restrict__ B = & B_buffer[ tile_off + x_offset ];

    if ( tile_idx.x == 0 ) {
        // Lower boundary
        switch( bc.x.lower ) {
        case( emf::bc::pmc) :
            for( int idx = block_thread_rank(); idx < static_cast<int>(ext_nx.y); idx += block_num_threads() ) {
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
            for( int idx = block_thread_rank(); idx < static_cast<int>(ext_nx.y); idx += block_num_threads() ) {
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
        switch( bc.x.upper ) {
        case( emf::bc::pmc) :
            for( int idx = block_thread_rank(); idx < static_cast<int>(ext_nx.y); idx += block_num_threads() ) {
                const int iy = idx * ystride;

                E[ nx.x + iy ].x = -E[ nx.x-1 + iy ].x;
                //E[ nx.x + iy ].y =  E[ nx.x + iy ].y;
                //E[ nx.x + iy ].z =  E[ nx.x + iy ].z;

                E[ nx.x+1 + iy ].x = -E[ nx.x-2 + iy ].x;
                E[ nx.x+1 + iy ].y =  E[ nx.x-1 + iy ].y;
                E[ nx.x+1 + iy ].z =  E[ nx.x-1 + iy ].z;

                // B[ nx.x + iy ].x = -B[ nx.x + iy ].x;
                B[ nx.x + iy ].y = -B[ nx.x-1 + iy ].y;
                B[ nx.x + iy ].z = -B[ nx.x-1 + iy ].z;

                B[ nx.x+1 + iy ].x =  B[ nx.x-1 + iy ].x;
                B[ nx.x+1 + iy ].y = -B[ nx.x-2 + iy ].y;
                B[ nx.x+1 + iy ].z = -B[ nx.x-2 + iy ].z;
            }
            break;

        case( emf::bc::pec) :
            for( int idx = block_num_threads(); idx < static_cast<int>(ext_nx.y); idx += block_num_threads() ) {
                const int iy = idx * ystride;

                E[ nx.x + iy ].x =  E[ nx.x-1 + iy ].x;
                E[ nx.x + iy ].y =  0;
                E[ nx.x + iy ].z =  0;

                E[ nx.x+1 + iy ].x =  E[ nx.x-2 + iy ].x;
                E[ nx.x+1 + iy ].y = -E[ nx.x-1 + iy ].y;
                E[ nx.x+1 + iy ].z = -E[ nx.x-1 + iy ].z;

                B[ nx.x + iy ].x =  0;
                B[ nx.x + iy ].y =  B[ nx.x-1 + iy ].y;
                B[ nx.x + iy ].z =  B[ nx.x-1 + iy ].z;

                B[ nx.x+1 + iy ].x = -B[ nx.x-1 + iy ].x;
                B[ nx.x+1 + iy ].y =  B[ nx.x-2 + iy ].y;
                B[ nx.x+1 + iy ].z =  B[ nx.x-2 + iy ].z;
            }
            break;
        default:
            break;
        }
    }
}

__global__
/**
 * @brief Physical boundary conditions for the y direction
 * 
 * @param tile      Tile position on grid
 * @param E         Tile E field
 * @param B         Tile B field
 * @param nx        Number of cells
 * @param ext_nx    Number of cells including guard cells
 * @param bc        Boundary condition
 */
void emf_bcy( 
    float3 * const __restrict__ E_buffer, float3 * const __restrict__ B_buffer,
    uint2 const ntiles, const uint2 nx, const uint2 ext_nx, bnd<unsigned int> gc, 
    emf::bc_type bc )
{

    const auto tile_idx = uint2{ 
        blockIdx.x ,
        blockIdx.y * (ntiles.y-1)
    };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    const int ystride = ext_nx.x;
    // Start at y cell 0
    const auto y_offset = gc.y.lower * ext_nx.x;

    float3 * const __restrict__ E = & E_buffer[ tile_off + y_offset ];
    float3 * const __restrict__ B = & B_buffer[ tile_off + y_offset ];

    if ( tile_idx.y == 0 ) {
        // Lower boundary
        switch( bc.y.lower ) {
        case( emf::bc::pmc) :
            for( int idx = block_thread_rank(); idx < static_cast<int>(ext_nx.x); idx += block_num_threads() ) {
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
            for( int idx = block_thread_rank(); idx < static_cast<int>(ext_nx.x); idx += block_num_threads() ) {
                int ix = idx;

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
        switch( bc.y.upper ) {
        case( emf::bc::pmc) :
            for( int idx = block_thread_rank(); idx < static_cast<int>(ext_nx.x); idx += block_num_threads() ) {
                int ix = idx;

                E[ ix + nx.y * ystride ].y = -E[ ix + (nx.y-1) * ystride ].y;

                E[ ix + (nx.y+1) * ystride ].x =  E[ ix + (nx.y-1) * ystride ].x;
                E[ ix + (nx.y+1) * ystride ].y = -E[ ix + (nx.y-2) * ystride ].y;
                E[ ix + (nx.y+1) * ystride ].z =  E[ ix + (nx.y-1) * ystride ].z;

                B[ ix + (nx.y) * ystride ].x = -B[ ix + (nx.y-1)*ystride ].x;
                B[ ix + (nx.y) * ystride ].z = -B[ ix + (nx.y-1)*ystride ].z;

                B[ ix + (nx.y+1) * ystride ].x = -B[ ix + (nx.x-2) * ystride ].x;
                B[ ix + (nx.y+1) * ystride ].y =  B[ ix + (nx.x-1) * ystride ].y;
                B[ ix + (nx.y+1) * ystride ].z = -B[ ix + (nx.x-2) * ystride ].z;
            }
            break;

        case( emf::bc::pec) :
            for( int idx = block_thread_rank(); idx < static_cast<int>(ext_nx.x); idx += block_num_threads() ) {
                const int ix = idx;

                E[ ix + (nx.y)*ystride ].x =  0;
                E[ ix + (nx.y)*ystride ].y =  E[ ix + (nx.y-1)*ystride ].y;
                E[ ix + (nx.y)*ystride ].z =  0;

                E[ ix + (nx.y+1)*ystride ].x = -E[ ix + (nx.x-1) * ystride ].x;
                E[ ix + (nx.y+1)*ystride ].y =  E[ ix + (nx.x-2) * ystride ].y;
                E[ ix + (nx.y+1)*ystride ].z = -E[ ix + (nx.x-1) * ystride ].z;

                B[ ix + (nx.y)*ystride ].x =  B[ ix + (nx.y-1) * ystride ].x;
                B[ ix + (nx.y)*ystride ].y =  0;
                B[ ix + (nx.y)*ystride ].z =  B[ ix + (nx.y-1) * ystride ].z;


                B[ ix + (nx.y+1) * ystride ].x =  B[ ix + (nx.y-2) * ystride ].x;
                B[ ix + (nx.y+1) * ystride ].y = -B[ ix + (nx.y-1) * ystride ].y;
                B[ ix + (nx.y+1) * ystride ].z =  B[ ix + (nx.y-2) * ystride ].z;
            }
            break;
        default:
            break;
        }
    }
}

}

/**
 * @brief Processes "physical" boundary conditions
 * 
 */
void EMF::process_bc() {

    dim3 block( 64 );

    // x boundaries
    if ( bc.x.lower > emf::bc::periodic || bc.x.upper > emf::bc::periodic ) {
        dim3 grid( 2, E->ntiles.y );
        kernel::emf_bcx <<< grid, block >>> (
            E -> d_buffer, B -> d_buffer, E -> ntiles, E -> nx, E -> ext_nx, E -> gc, 
            bc
        );
    }

    // y boundaries
    if ( bc.y.lower > emf::bc::periodic || bc.y.upper > emf::bc::periodic ) {
        dim3 grid( E->ntiles.x, 2 );
        kernel::emf_bcx <<< grid, block >>> (
            E -> d_buffer, B -> d_buffer, E -> ntiles, E -> nx, E -> ext_nx, E -> gc, 
            bc
        );
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
__device__ void yeeJ_e( 
    float3 * const __restrict__ E, 
    float3 const * const __restrict__ B, 
    uint2 const nx, int const ystride, 
    float3 const * const __restrict__ J, int const J_ystride, 
    float2 const dt_dx, float const dt )
{
    int const range_x = nx.x + 2;
    int const range_y = nx.y + 2;

    for( int idx = block_thread_rank(); idx < range_y * range_x; idx += block_num_threads() ) {
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

__global__ __launch_bounds__(opt_yee_block) void yeeJ( 
    float3 * const __restrict__ E_buffer,
    float3 * const __restrict__ B_buffer,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx, unsigned int const offset, 
    float3 * const __restrict__ J_buffer,
    uint2 const J_ext_nx, unsigned int const J_offset, 
    float2 const dt_dx, float const dt ) {

    auto * shm = block::shared_mem<float3>();

    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    const int ystride = ext_nx.x;

    const float2 dt_dx_2 = { dt_dx.x / 2, dt_dx.y / 2 };

    float3 * const __restrict__ E_local = & shm[ 0 ];
    float3 * const __restrict__ B_local = & shm[ tile_vol ];

    // Copy E and B into shared memory
    for( unsigned i = block_thread_rank(); i < tile_vol; i += block_num_threads() ) {
        E_local[i] = E_buffer[ tile_off + i ];
        B_local[i] = B_buffer[ tile_off + i ];
    }

    float3 * const __restrict__ tile_E = & E_local[ offset ];
    float3 * const __restrict__ tile_B = & B_local[ offset ];

    const int    J_tile_vol = roundup4( J_ext_nx.x * J_ext_nx.y );
    const size_t J_tile_off = tile_id * J_tile_vol;
    const int    J_ystride  = J_ext_nx.x;
    float3 * const __restrict__ tile_J = & J_buffer[ J_tile_off + J_offset ];

    block_sync();

    yee_b( tile_E, tile_B, nx, ystride, dt_dx_2 );
    block_sync();

    yeeJ_e( tile_E, tile_B, nx, ystride, tile_J, J_ystride, dt_dx, dt );
    block_sync();

    yee_b( tile_E, tile_B, nx, ystride, dt_dx_2 );
    block_sync();

    // Copy data to global memory
    for( unsigned i = block_thread_rank(); i < tile_vol; i += block_num_threads() ) {
        E_buffer[ tile_off + i ] = E_local[i];
        B_buffer[ tile_off + i ] = B_local[i];
    }

}

}

/**
 * @brief Advance EM fields 1 time step including current
 * 
 * @param current   Electric current
 */
void EMF::advance( Current & current ) {

    float2 const dt_dx   = make_float2( dt/dx.x, dt/dx.y );

    dim3 grid( E -> ntiles.x, E -> ntiles.y );
    auto block = opt_yee_block;
    size_t shm_size = 2 * E -> tile_vol * sizeof(float3);

    auto err = block::set_shmem_size( kernel::yeeJ, shm_size );
    kernel::yeeJ <<< grid, block, shm_size >>> (
        E -> d_buffer, B -> d_buffer, 
        E -> ntiles, E -> nx, E -> ext_nx, E -> offset,
        current.J->d_buffer, current.J -> ext_nx, current.J -> offset,
        dt_dx, dt
    );

    // Update guard cells with new values
    E -> copy_to_gc( );
    B -> copy_to_gc( );

    // Do additional bc calculations if needed
    process_bc( );

    // Advance internal iteration number
    iter += 1;

    // Move simulation window if needed
    if ( moving_window.active() ) move_window( );
}

/**
 * @brief Save EMF data to diagnostic file
 * 
 * @param field     Field to save (0:E, 1:B)
 * @param fc        Field component to save (0, 1 or 2)
 */
void EMF::save( const emf::field field, fcomp::cart const fc ) {

    std::string vfname;  // Dataset name
    std::string vflabel; // Dataset label (for plots)

    vec3grid<float3> * f;

    switch (field ) {
        case emf::e :
            f = E;
            vfname = "E";
            vflabel = "E_";
            break;
        case emf::b :
            f = B;
            vfname = "B";
            vflabel = "B_";
            break;
        default:
            ABORT("Invalid field type selected, aborting");
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
            ABORT("Invalid field component (fc) selected, aborting");
    }

    zdf::grid_axis axis[2];
    axis[0] = (zdf::grid_axis) {
        .name = (char *) "x",
        .min = 0.0 + moving_window.motion(),
        .max = box.x + moving_window.motion(),
        .label = (char *) "x",
        .units = (char *) "c/\\omega_n"
    };

    axis[1] = (zdf::grid_axis) {
        .name = (char *) "y",
        .min = 0.0,
        .max = box.y,
        .label = (char *) "y",
        .units = (char *) "c/\\omega_n"
    };

    zdf::grid_info info = {
        .name = (char *) vfname.c_str(),
        .ndims = 2,
        .label = (char *) vflabel.c_str(),
        .units = (char *) "m_e c \\omega_n e^{-1}",
        .axis = axis
    };

    info.count[0] = E -> gnx.x;
    info.count[1] = E -> gnx.y;

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
    float3 * const __restrict__ E_buffer,
    float3 * const __restrict__ B_buffer,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx, unsigned int const offset, 
    double * const __restrict__ d_energy ) {

    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    float3 * const __restrict__ E_local = & E_buffer[ tile_off + offset ];
    float3 * const __restrict__ B_local = & B_buffer[ tile_off + offset ];

    const int ystride = ext_nx.x;

    double3 ene_E = double3{0};
    double3 ene_B = double3{0};

    for( int idx = block_thread_rank(); idx < nx.y * nx.x; idx += block_num_threads() ) {
        int const i = idx % nx.x;
        int const j = idx / nx.x;

        float3 const efld = E_local[ j * ystride + i ];
        float3 const bfld = B_local[ j * ystride + i ];

        ene_E.x += efld.x * efld.x;
        ene_E.y += efld.y * efld.y;
        ene_E.z += efld.z * efld.z;

        ene_B.x += bfld.x * bfld.x;
        ene_B.y += bfld.y * bfld.y;
        ene_B.z += bfld.z * bfld.z;
    }

    // Add up energy from all warps
    ene_E.x = warp::reduce_add( ene_E.x );
    ene_E.y = warp::reduce_add( ene_E.y );
    ene_E.z = warp::reduce_add( ene_E.z );

    ene_B.x = warp::reduce_add( ene_B.x );
    ene_B.y = warp::reduce_add( ene_B.y );
    ene_B.z = warp::reduce_add( ene_B.z );

    if ( warp::thread_rank() == 0 ) {
        device::atomic_fetch_add( &(d_energy[0]), ene_E.x );
        device::atomic_fetch_add( &(d_energy[1]), ene_E.y );
        device::atomic_fetch_add( &(d_energy[2]), ene_E.z );

        device::atomic_fetch_add( &(d_energy[3]), ene_B.x );
        device::atomic_fetch_add( &(d_energy[4]), ene_B.y );
        device::atomic_fetch_add( &(d_energy[5]), ene_B.z );
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
void EMF::get_energy( double3 & ene_E, double3 & ene_B ) {

    // Zero energy values
    device::zero( d_energy, 6 );

    // Add up energy from all cells
    dim3 grid( E->ntiles.x, E->ntiles.y );
    dim3 block( 1024 );
    kernel::get_energy <<< grid, block >>> ( 
        E->d_buffer, B->d_buffer,
        E->ntiles, E->nx, E->ext_nx, E->offset,
        d_energy
    );

    // Copy results to host and normalize
    double h_energy[6];
    device::memcpy_tohost( h_energy, d_energy, 6 );

    ene_E.x = 0.5 * dx.x * dx.y * h_energy[0];
    ene_E.y = 0.5 * dx.x * dx.y * h_energy[1];
    ene_E.z = 0.5 * dx.x * dx.y * h_energy[2];

    ene_B.x = 0.5 * dx.x * dx.y * h_energy[3];
    ene_B.y = 0.5 * dx.x * dx.y * h_energy[4];
    ene_B.z = 0.5 * dx.x * dx.y * h_energy[5];

}