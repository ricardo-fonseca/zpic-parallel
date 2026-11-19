#include "sort.hpp"

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
    for( int dir = 0; dir < 9; dir++ ) {
        const int start = part::edge_tile_start( dir, nt );
        const int size  = part::edge_ntiles( dir, nt );

        int send_np = 0;
        int recv_np = 0;
        for( int k = 0; k < size; k++ ) {
            send_np += send.buffer[ start + k ];
            recv_np += recv.buffer[ start + k ];
        }
        send.msg_np[dir] = send_np;
        recv.msg_np[dir] = recv_np;
    }

    // Wait for sends to complete
    MPI_Waitall( 9, send.requests, MPI_STATUSES_IGNORE );
}
