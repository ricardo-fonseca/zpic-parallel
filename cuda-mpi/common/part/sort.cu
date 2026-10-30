#include "sort.hpp"

namespace kernel {

__global__
/**
* @brief Kernel for updating msg_np from msg_np_tile
* 
* @note Must be called with a grid size of (3,3)
* 
* @param recv_msg_np_tile      Number of particles in receive message tiles
* @param recv_msg_np           Total number of particles per receive message
* @param send_msg_np_tile      Number of particles in receive message tiles
* @param send_msg_np           Total number of particles per receive message
* @param ntiles                Grid tile dimension
*/
inline void update_msg_np( 
    int const * const __restrict__ recv_buffer, int * const __restrict__ recv_msg_np, 
    int const * const __restrict__ send_buffer, int * const __restrict__ send_msg_np, 
    const int2 ntiles ) {

    __shared__ int recv_np_dir; recv_np_dir = 0;
    __shared__ int send_np_dir; send_np_dir = 0;

    // Direction is taken from block ID
    int dir = 3 * blockIdx.y + blockIdx.x;

    // id of first edge tile for this direction
    const int start = part::edge_tile_start( dir, ntiles );
    // Number of edge tiles in this direction
    const int size  = part::edge_ntiles( dir, ntiles );

    gpu::block::sync();

    // Add contribution from all tiles in message
    int recv_np_tmp = 0;
    int send_np_tmp = 0;

    for( int i = gpu::block::thread_rank(); i < size; i+= gpu::block::num_threads() ) {
        send_np_tmp += send_buffer[ start + i ];
        recv_np_tmp += recv_buffer[ start + i ];
    }
    
    recv_np_tmp = gpu::warp::reduce_add( recv_np_tmp );
    send_np_tmp = gpu::warp::reduce_add( send_np_tmp );

    if ( gpu::warp::thread_rank() == 0 ) {
        // if gpu::block::num_threads() <= WARP_SIZE this could be replaced by a copy
        gpu::block::atomic_fetch_add( &recv_np_dir, recv_np_tmp );
        gpu::block::atomic_fetch_add( &send_np_dir, send_np_tmp );
    }

    gpu::block::sync();

    // Store result in device memory
    if ( gpu::block::thread_rank() == 0 ) {
        recv_msg_np[ dir ] = recv_np_dir;
        send_msg_np[ dir ] = send_np_dir;
    }
}

} // namespace kernel

/**
* @brief Exchange number of particles in edge cells
*
*/
void part::particle_sort::exchange_np() {

    const int2 nt = make_int2( ntiles.x, ntiles.y );

    // Post receives
    for( int dir = 0; dir < 9; dir++ ) {
        const int start = part::edge_tile_start( dir, nt );
        const int size  = part::edge_ntiles( dir, nt );

        if ( neighbor[dir] >= 0 ) {
            MPI_Irecv( &recv.buffer[start], size, MPI_INT, neighbor[dir],
                    source_tag(dir), comm, &recv.requests[dir] );
        } else {
            recv.requests[dir] = MPI_REQUEST_NULL;
        }
    }

    // Ensure send data is up to date
    gpu::device::sync();

    // Post sends
    for( int dir = 0; dir < 9; dir++ ) {
        const int start = part::edge_tile_start( dir, nt );
        const int size  = part::edge_ntiles( dir, nt );

        if ( neighbor[dir] >= 0 ) {
            MPI_Isend( &send.buffer[start], size, MPI_INT, neighbor[dir],
                    dest_tag(dir), comm, &send.requests[dir] );
        } else {
            send.requests[dir] = MPI_REQUEST_NULL;
        }
    }

    // Wait for receives to complete
    MPI_Waitall( 9, recv.requests, MPI_STATUSES_IGNORE );

    // update send.msg_np[] and recv.msg_np[]
    dim3 grid( 3, 3 );
    auto block = 32;
    kernel::update_msg_np <<< grid, block >>> ( 
        recv.buffer, recv.d_msg_np,
        send.buffer, send.d_msg_np, 
        nt
    );

    // Wait for sends to complete
    MPI_Waitall( 9, send.requests, MPI_STATUSES_IGNORE );
}
