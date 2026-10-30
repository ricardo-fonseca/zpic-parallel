#pragma once

#include <cstdint>
#include "../parallel/partition.hpp"
#include "../util/memory.cuh"
#include "../util/math.hpp"

#include "comms.hpp"

namespace part {

/**
 * @brief Class for handling particle data messages
 * 
 */
class particle_message {

    private:

    /**
     * @brief Message tag for incoming messages
     * 
     * @param dir   - Communication direction (0-8)
     * @return int 
     */
    inline int source_tag( int dir ) {
        return (8 - dir) | 0x200;
    }

    /**
     * @brief Message tag for outgoing messages
     * 
     * @param dir   - Communication direction (0-8) 
     * @return int 
     */
    inline int dest_tag( int dir ) {
        return dir | 0x200;
    }

    enum class message_type { none = 0, send, receive };

    /// @brief Active message type
    message_type active;

    /// @brief Maximum data size (bytes)
    uint32_t max_size;
    /// @brief Neighbor ranks
    int neighbor[9];
    /// @brief MPI communicator for messages
    MPI_Comm comm;
    /// @brief Message handles
    MPI_Request requests[9];

    public:

    /// @brief Particle data (packed)
    uint8_t * buffer;
    /// @brief Individual message size (bytes)
    int size[9];

    /**
     * @brief Construct a new Particle Msg Buffer object
     * 
     * @param ntiles 
     */
    particle_message( mpi::cart2d & par, int2 periodic ) {

        // Buffers for particle data messages (initially empty)
        buffer = nullptr;
        max_size = 0;

        // Communicator
        comm = par.get_comm();

        // Initialize neighbor ranks and essage requests
        for( int dir = 0; dir < 9; dir++ ) {
            neighbor[ dir ] = part::dir_neighbor( dir, par, periodic );
            requests[ dir ] = MPI_REQUEST_NULL;
            size[ dir ] = 0;
        }

        active = message_type::none;
    }

    /**
     * @brief Destroy the Particle Msg Buffer object
     * 
     */
    ~particle_message() {
        if ( active != message_type::none ) {
            for( int i = 0; i < 9; i++ ) {
                if ( requests[i] != MPI_REQUEST_NULL )
                    MPI_Cancel( &requests[i] );
            }
            MPI_Waitall( 9, requests, MPI_STATUSES_IGNORE );
        }
        gpu::device::free( buffer );
    }

    particle_message( const particle_message & ) = delete;
    particle_message( particle_message && ) = delete;

    /**
     * @brief Checks if data buffer is large enough to hold all messages and grows
     *        it if necessary
     * @note Buffer is grown in multiples of 1 MB
     * 
     * @param total_size    Total required size in bytes
     */
    void check_buffer( uint32_t total_size ) {
        if ( active ==  message_type::none ) {
            if ( total_size > max_size ) {
                gpu::device::free( buffer );
                max_size = roundup<1048576>(total_size);
                buffer = gpu::device::malloc<uint8_t>( max_size );
            }
        } else {
            mpi::fatal("check_buffer() called on an active message");
        }
    }

    /**
     * @brief Start all non-blocking send messages
     * 
     */
    void isend( ) {
        if ( active != message_type::none ) {
            mpi::fatal( "isend() - Tried to send messages before other messages complete." );
        }

        active = message_type::send;

        uint32_t offset = 0;
        for( int i = 0; i < 9; i++ ) {
            if ( (neighbor[i] >= 0) && (size[i] > 0) ) {
                MPI_Isend( &buffer[offset], size[i], MPI_BYTE, neighbor[i],  dest_tag(i), comm, &requests[i]);
                offset += size[i];
            } else {
                requests[i] = MPI_REQUEST_NULL;
            }
        }
    }

    /**
     * @brief Start all non-blocking receive messages
     * 
     */
    void irecv( ) {

        if ( active != message_type::none ) {
            mpi::fatal( "irecv() - Tried to receive message before other message completes." );
        }
        active = message_type::receive;

        // Post receives
        uint32_t offset = 0;
        for( int i = 0; i < 9; i++ ) {
            if ( (neighbor[i] >= 0) && ( size[i] > 0 ) ) {
                MPI_Irecv( &buffer[offset], size[i], MPI_BYTE, neighbor[i],  source_tag(i), comm, &requests[i]);
                offset += size[i];
            } else {
                requests[i] = MPI_REQUEST_NULL;
            }
        }
    }

    /**
     * @brief Wait for all messages to complete
     * 
     * @return int 
     */
    int wait() {
        int ierr = MPI_Waitall( 9, requests, MPI_STATUSES_IGNORE );
        active = message_type::none;
        for( int i = 0; i < 9; i++ ) size[i] = 0;
        return ierr;
    }

    /**
     * @brief Update neighbor ranks
     * 
     * @param par           Parallel partition
     * @param periodic      Global periodic boundaries
     */
    void update_neighbors( const mpi::cart2d & par, const int2 periodic ) {
        if ( active != message_type::none )
            mpi::fatal( "particle_message::update_neighbors() - called on an active message" );
        for( int dir = 0; dir < 9; dir++ )
            neighbor[ dir ] = part::dir_neighbor( dir, par, periodic );
    }

};

}