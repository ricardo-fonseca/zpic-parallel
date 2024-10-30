
#ifndef PARALLEL_H_
#define PARALLEL_H_

#include "zpic.h"
#include "vec_types.h"

#include <mpi.h>
#include <iostream>

namespace mpi {
template< typename T > 
MPI_Datatype data_type () { 
    static_assert(0,"Invalid data type"); 
    return MPI_DATATYPE_NULL;
};

template<> constexpr MPI_Datatype data_type<int8_t  >(void) { return MPI_INT8_T; };
template<> constexpr MPI_Datatype data_type<uint8_t >(void) { return MPI_UNSIGNED_CHAR; };
template<> constexpr MPI_Datatype data_type<int16_t >(void) { return MPI_INT16_T; };
template<> constexpr MPI_Datatype data_type<uint16_t>(void) { return MPI_UINT16_T; };
template<> constexpr MPI_Datatype data_type<int32_t >(void) { return MPI_INT32_T; };
template<> constexpr MPI_Datatype data_type<uint32_t>(void) { return MPI_UINT32_T; };
template<> constexpr MPI_Datatype data_type<int64_t >(void) { return MPI_INT64_T; };
template<> constexpr MPI_Datatype data_type<uint64_t>(void) { return MPI_INT64_T; };
template<> constexpr MPI_Datatype data_type<float   >(void) { return MPI_FLOAT; };
template<> constexpr MPI_Datatype data_type<double  >(void) { return MPI_DOUBLE; };

static const MPI_Op sum = MPI_SUM;
static const int proc_null = MPI_PROC_NULL;

namespace type {
    extern MPI_Datatype float3;
    extern MPI_Datatype double3;
}

// These cannot be declared constexpr as their value is unknown at compile time
template<> inline MPI_Datatype data_type<float3 >(void) { return mpi::type::float3; };
template<> inline MPI_Datatype data_type<double3>(void) { return mpi::type::double3; };

static inline int init( int *argc, char ***argv ) {
    int ierr = MPI_Init( argc, argv );

    if ( ierr == MPI_SUCCESS ) {
        // Initialize extra types
        MPI_Type_contiguous( 3, MPI_FLOAT,  &mpi::type::float3 ); 
        MPI_Type_commit( &mpi::type::float3 );
        
        MPI_Type_contiguous( 3, MPI_DOUBLE, &mpi::type::double3 );
        MPI_Type_commit( &mpi::type::double3 );
    }
    return ierr;
}

static inline int finalize( void ) {

    // These aren't strictly necessary
    MPI_Type_free( &mpi::type::float3 );
    MPI_Type_free( &mpi::type::double3 );

    return MPI_Finalize();
}

static inline int abort( int errorcode ) { 
    return MPI_Abort(MPI_COMM_WORLD, errorcode );
}

}

/**
 * @brief Parallel partition
 * 
 */
class Partition {
    private:

    /// @brief MPI Communicator
    MPI_Comm comm;

    /// @brief Partition size
    int size;

    /// @brief Local rank
    int rank;

    /// @brief Local coordinates in partition
    int2 coords;

    /**
     * @brief Neighbor ranks
     * 
     * @note Organized as `neighbor[ydir][xdir]` where `ydir`/`xdir` take the
     * values: `0` - lower, `1` - central, `2` -upper
     */
    int neighbor[3][3];

    public:

    /// @brief Dimensions of the parallel partition
    const uint2 dims;

    /// @brief Periodicity of the parallel partition
    const int2 periodic;

    /**
     * @brief Construct a new Partition object
     * 
     * @param dims      Partition dimension
     * @param periods   Peridocity (defaults to true on both directions)
     */
    Partition( uint2 dims, int2 periodic = make_int2(1,1) ) : dims(dims), periodic(periodic) 
    {
        // Check if MPI has been initialized
        int flag;
        MPI_Initialized( &flag );

        if ( ! flag ) {
            std::cerr << "Unable to create partition object, MPI has not been initialized\n";
            exit(1);
        }

        // Get communicator size
        if ( MPI_Comm_size( MPI_COMM_WORLD, &size ) != MPI_SUCCESS ) {
            std::cerr << "Unable to get communicator size, aborting\n";
            exit(1);
        }

        // Check dimensions
        if ( dims.x < 1 ) {
            std::cerr << "Invalid partition dims.x = " << dims.x << "\n";
            exit(1);
        }

        if ( dims.y < 1 ) {
            std::cerr << "Invalid partition dims.y = " << dims.x << "\n";
            exit(1);
        }

        if ( dims.x * dims.y != (unsigned) size ) {
            std::cerr << "Partition size (" << dims.x * dims.y << ") and number of parallel nodes (" << size << ") don't match\n";
            exit(1);
        }

        
        int _dims[] = { (int) dims.x, (int) dims.y } ;
        int periods[] = { periodic.x, periodic.y };

        // Create partition
        if ( MPI_Cart_create(MPI_COMM_WORLD, 2, _dims, periods, 0, &comm ) != MPI_SUCCESS ) {
            std::cerr << "Unable to create cartesian topology\n";
            exit(1);
        }

        // Get rank
        if ( MPI_Comm_rank( comm, & rank ) != MPI_SUCCESS ) {
            std::cerr << "Unable to get communicator rank, aborting\n";
            exit(1);
        }

        int lcoords[2];
        if ( MPI_Cart_coords( comm, rank, 2, lcoords ) != MPI_SUCCESS ) {
            std::cerr << "Unable to get cartesian coordinates, aborting\n";
            exit(1);
        };
        coords = make_int2( lcoords[0], lcoords[1] );

        // Get neighbors
        // Since we also need the corner neighbors we cannor use MPI_Cart_shift()
        for( int iy = 0; iy < 3; iy ++) {
            int neighbor_coords[2];
            neighbor_coords[1] = coords.y + iy - 1;

            if ( periodic.y ) {
                if ( neighbor_coords[1] < 0 ) 
                    neighbor_coords[1] += dims.y;
                if ( neighbor_coords[1] >= (int) dims.y ) 
                    neighbor_coords[1] -= dims.y;
            } 

            for( int ix = 0; ix < 3; ix ++) {
                neighbor_coords[0] = coords.x + ix - 1;
                if ( periodic.x ) {
                    if ( neighbor_coords[0] < 0 ) 
                        neighbor_coords[0] += dims.x;
                    if ( neighbor_coords[0] >= (int) dims.x )
                        neighbor_coords[0] -= dims.x;
                }

                if ( neighbor_coords[1] >= 0 && neighbor_coords[1] < (int) dims.y && 
                     neighbor_coords[0] >= 0 && neighbor_coords[0] < (int) dims.x ) {
                    MPI_Cart_rank( comm, neighbor_coords, & neighbor[ iy ][ ix ] );
                } else {
                    neighbor[ iy ][ ix ] = -1;
                } 
            }
        }

/*
        std::cout << rank << " - neighbor:\n"
                  << neighbor[2][0] << ',' << neighbor[2][1] << ',' << neighbor[2][2] << '\n'
                  << neighbor[1][0] << ',' << neighbor[1][1] << ',' << neighbor[1][2] << '\n'
                  << neighbor[0][0] << ',' << neighbor[0][1] << ',' << neighbor[0][2] << std::endl;
*/

        // Sanity check - this should never happen
        if ( neighbor[1][1] != rank ) {
            std::cerr << "Invalid neighbor\n";
            exit(1);
        }; 
    };

    /**
     * @brief Destroy the Partition object
     * 
     */
    ~Partition() {
        MPI_Comm_free( & comm );
    };

    /**
     * @brief Prints information about local node
     * 
     */
    void info() {
        std::cout << '[' << rank << '/' << size << "] - coords " << coords << '\n';
    }

    /**
     * @brief Returs the MPI communicator
     * 
     * @return MPI_Comm     MPI Communicator
     */
    MPI_Comm get_comm() {
        return comm;
    }

    /**
     * @brief Get parallel partition size
     * 
     * @return int  Partition size
     */
    int get_size() {
        return size;
    }

    /**
     * @brief Get the local process rank
     * 
     * @return int  Local process rank
     */
    int get_rank() {
        return rank;
    }

    /**
     * @brief Get the neighbor process rank
     * 
     * @param shiftx    Shift along x direction, should be -1, 0 or +1
     * @param shifty    Shift along y direction, should be -1, 0 or +1
     * @return int      neighbor rank
     */
    int get_neighbor( int shiftx, int shifty ) {
        return neighbor[ 1 + shifty ][ 1 + shiftx ];
    }

    /**
     * @brief Get local coordinates
     * 
     * @param local_coords     Local process coordinates
     */
    int2 get_coords( ) {
        return coords;
    }

    /**
     * @brief Get coordinates of a specific process
     * 
     * @param target_rank       Target process rank
     * @param target_coords     Target process coordinates
     */
    int2 get_coords_rank( const int target_rank ) {
        int _coords[2];
        MPI_Cart_coords( comm, target_rank, 2, _coords );
        int2 target_coords = make_int2( _coords[0], _coords[1] );
        return target_coords;
    }

    /**
     * @brief Gets the rank of the process at specific coordinates
     * 
     * @param target_coords     Target coordinates
     * @return int              Target process rank
     */
    int get_rank_coords( const int2 target_coords ) {
        int cart_rank;
        int _coords[2] = { coords.x, coords.y };
        MPI_Cart_rank( comm, _coords, &cart_rank );
        return cart_rank;
    }

    /**
     * @brief Returns a unique integer based on the local coords
     * 
     * @return int 
     */
    int coords_id( ) {
        return coords.y * dims.x + coords.x;
    }

    /**
     * @brief Returns true if local node is on the edge of the partition
     * 
     * @param coord     Coordinate to check (coord::x, coord::y)
     * @param edge      Edge to check (edge::lower, edge::upper)
     * @return int      Returns 1 if node in on the requested edge
     */
    int on_edge( coord::cart coord, edge::pos edge ) {
        switch (coord) {
        case coord::x:
            switch(edge) {
                case edge::lower: return coords.x == 0;
                case edge::upper: return coords.x == (int) (dims.x-1);
            }
            break;
        case coord::y:
            switch(edge) {
                case edge::lower: return coords.y == 0;
                case edge::upper: return coords.y == (int) (dims.y-1);
            }
            break;
        }
    }

    /**
     * @brief Returns true if target node is on the edge of the partition
     * 
     * @param coord         Coordinate to check (coord::x, coord::y)
     * @param edge          Edge to check (edge::lower, edge::upper)
     * @param target_rank   Target node rank
     * @return int          Returns 1 if node in on the requested edge
     */
    int on_edge( coord::cart coord, edge::pos edge, int target_rank ) {
        
        int2 target_coords = get_coords_rank( target_rank );
        
        switch (coord) {
        case coord::x:
            switch(edge) {
                case edge::lower: return target_coords.x == 0;
                case edge::upper: return target_coords.x == (int)(dims.x-1);
            }
            break;
        case coord::y:
            switch(edge) {
                case edge::lower: return target_coords.y == 0;
                case edge::upper: return target_coords.y == (int)(dims.y-1);
            }
            break;
        }
    }

    int abort( int errorcode ) {
        return MPI_Abort( comm, errorcode );
    }

    int root() { return rank == 0;}

    void barrier() {
        if ( MPI_Barrier( comm ) != MPI_SUCCESS ) {
            std::cerr << "Error on MPI_Barrier() call\n";
            MPI_Abort( comm, 1 );
        }
    }

    /**
     * @brief Performs an MPI_Reduce operation in this parallel partition
     * 
     * @tparam T        Data type, must be supported by MPI
     * @param data      Data buffer
     * @param count     Data size
     * @param op        MPI operation
     * @param root      Target node, defaults to 0
     * @return int      Always returns 0
     */
    template< typename T >
    int reduce( T * data, int count, MPI_Op op, int root = 0 ) {

        if ( MPI_Reduce( MPI_IN_PLACE, data, count, mpi::data_type<T>(), op, root,
                         comm ) != MPI_SUCCESS ) {
            std::cerr << "MPI_Reduce operation failed, aborting\n";
            MPI_Abort( comm, 1 );
        }
        return 0;
    }

};

template< typename T >
class Message {
    private:

    enum Type { none, send, receive };

    /// @brief Active message type
    Message::Type active;

    /// @brief Active / last completed message MPI handle
    MPI_Request request;

    public:

    /// @brief MPI communicator
    MPI_Comm comm;

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
    Message( int max_count, MPI_Comm comm ) : 
        active( none ), request( MPI_REQUEST_NULL ), 
        comm( comm ), max_count( max_count )
    {
        buffer = memory::malloc<T>( max_count );
    }

    /**
     * @brief Destroy the Message object
     * 
     */
    ~Message() {
        if ( active != Message::none ) {
            MPI_Request tmp = request;
            MPI_Cancel( &tmp );
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
        
/*
        // debug
        {
            int rank; MPI_Comm_rank( comm, & rank );
            std::cout << rank << " - sending message size " << count 
                << " to node " << recipient << ", tag:" << tag << '\n';
        }
*/

        if ( count > max_count ) {
            std::cerr << "isend() - Message size too large\n";
            mpi::abort(1);
        }

        if ( active != none ) {
            std::cerr << "isend() - Tried to send message before other message completes\n";
            mpi::abort(1);
        }

        active = Message::send;
        return MPI_Isend( buffer, count, mpi::data_type<T>(), recipient, tag, comm, &request) ;
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
            std::cerr << "isend() - Tried to receive message before other message completes\n";
            mpi::abort(1);
        }

        active = Message::receive;
        return MPI_Irecv( buffer, max_count, mpi::data_type<T>(), sender, tag, comm, &request) ;
    }

    /**
     * @brief Wait for message to complete
     * 
     * @return int      Error code from MPI_Wait (MPI_SUCCESS on success)
     */
    int wait( ) {
        if ( active == Message::none ) {
            std::cerr << "wait() - No active message\n";
            mpi::abort(1);
        }
        int ierr = MPI_Wait( &request, MPI_STATUS_IGNORE );
        active = Message::none;
        return ierr;
    }

    /**
     * @brief Wait for receive message to complete and get message size
     * 
     * @param count     Received message size
     * @return int      Error code from MPI_Wait (MPI_SUCCESS on success)
     */
    int wait( int & count ) {
        if ( active != Message::receive ) {
            std::cerr << "wait() - No active message receive\n";
            mpi::abort(1);
        }
        MPI_Status status;
        int ierr = MPI_Wait( &request, &status );
        
        // Get number of received elements
        MPI_Get_count( status, mpi::data_type<T>(), &count );
        
        active = Message::none;
        return ierr;
    }
};

#endif
