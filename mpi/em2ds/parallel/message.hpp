#pragma once

#include "mpi.hpp"
#include "../util/memory.hpp"

namespace mpi {

template< typename T >
class message {
    private:

    enum type { none, send, receive };

    /// @brief Active message type
    message::type active;

    /// @brief Active / last completed message MPI handle
    MPI_Request request;

    public:

    /// @brief MPI communicator
    const MPI_Comm comm;

    /// @brief Data buffer
    T * buffer;

    /// @brief Maximum message size
    const int max_count;

    /**
     * @brief Construct a new Message object
     * 
     * @param max_count     Maximum message size
     * @param comm          MPI communicator
     */
    message( int max_count, MPI_Comm comm ) : 
        active( none ), request( MPI_REQUEST_NULL ), 
        comm( comm ), max_count( max_count )
    {
        buffer = memory::malloc<T>( max_count );
    }

    message(const message&) = delete;
    message& operator=(const message&) = delete;

    /**
     * @brief Destroy the Message object
     * 
     */
    ~message() {
        if ( active != message::none ) {
            MPI_Cancel( &request );
            MPI_Wait( &request, MPI_STATUS_IGNORE );
        }
        memory::free( buffer );
    }

    /**
     * @brief Non-blocking send message
     * 
     * @param count         Message size (must be smaller than max_count)
     * @param recipient     Target node
     * @param tag           Message tag
     * @return int          Error code from MPI_Isend (MPI_SUCCESS on success)
     */
    int isend( int count, int recipient, int tag ) {
        
        if ( count > max_count ) {
            std::cerr << "isend() - Message size too large\n";
            mpi::abort(1);
        }

        if ( active != none ) {
            std::cerr << "isend() - Tried to send message before other message completes\n";
            mpi::abort(1);
        }

        int ierr = MPI_Isend( buffer, count, mpi::data_type<T>(), recipient, tag, comm, &request);
        active = ( ierr == MPI_SUCCESS) ? message::send : message::none;
        return ierr;
    }

    /**
     * @brief Non-blocking receive message
     * 
     * @note The received message size must be <= max_count. You can use the
     *       .wait(count) method to get the received message size
     * 
     * @param sender    Source node
     * @param tag       Message tag
     * @return int      Error code from MPI_Irecv (MPI_SUCCESS on success)
     */
    int irecv( int sender, int tag ) {

        if ( active != none ) {
            std::cerr << "irecv() - Tried to receive message before other message completes\n";
            mpi::abort(1);
        }

        int ierr = MPI_Irecv( buffer, max_count, mpi::data_type<T>(), sender, tag, comm, &request);
        active = ( ierr == MPI_SUCCESS) ? message::receive : message::none;
        return ierr;
    }

    /**
     * @brief Wait for message to complete
     * 
     * @return int      Error code from MPI_Wait (MPI_SUCCESS on success)
     */
    int wait( ) {
        if ( active == message::none ) {
            std::cerr << "wait() - No active message\n";
            mpi::abort(1);
        }
        int ierr = MPI_Wait( &request, MPI_STATUS_IGNORE );
        active = message::none;
        return ierr;
    }

    /**
     * @brief Wait for receive message to complete and get message size
     * 
     * @param count     Received message size
     * @return int      Error code from MPI_Wait (MPI_SUCCESS on success)
     */
    int wait( int & count ) {
        if ( active != message::receive ) {
            std::cerr << "wait() - No active message receive\n";
            mpi::abort(1);
        }
        MPI_Status status;
        int ierr = MPI_Wait( &request, &status );
        
        // Get number of received elements
        MPI_Get_count( &status, mpi::data_type<T>(), &count );
        
        active = message::none;
        return ierr;
    }
};

}