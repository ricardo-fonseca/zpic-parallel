
#ifndef PARALLEL_H_
#define PARALLEL_H_

#include "zpic.h"
#include "vec_types.h"

#include <mpi.h>
#include <iostream>

namespace mpi {

namespace {
    class _mpi_cout : private std::streambuf, public std::ostream
    {   
        public:
        _mpi_cout() : std::ostream(this), new_line(true) {}
    
        private:

        bool new_line;

        int overflow(int c) override
        {
            if (c != std::char_traits<char>::eof() && new_line ) {
                int rank;
                int ierr = MPI_Comm_rank( MPI_COMM_WORLD, &rank );
                if ( ierr == MPI_SUCCESS ) {
                    std::cout << "[" << rank << "] ";
                } else {
                    std::cout << "[--] ";
                }
            }
            
            new_line = ( c == '\n' );
            std::cout.put(c);
    
            return 0;
        }
    
    };
}

static _mpi_cout cout;

template< typename T > 
MPI_Datatype data_type () { 
    static_assert(0,"Invalid data type"); 
    return MPI_DATATYPE_NULL;
};

// On some MPI implementations (namely OpenMPI 5.*) the MPI_* datatypes are
// not known at compile time so we cannot declare these as constexpr

template<> inline MPI_Datatype data_type<int8_t  >(void) { return MPI_INT8_T; };
template<> inline MPI_Datatype data_type<uint8_t >(void) { return MPI_UNSIGNED_CHAR; };
template<> inline MPI_Datatype data_type<int16_t >(void) { return MPI_INT16_T; };
template<> inline MPI_Datatype data_type<uint16_t>(void) { return MPI_UINT16_T; };
template<> inline MPI_Datatype data_type<int32_t >(void) { return MPI_INT32_T; };
template<> inline MPI_Datatype data_type<uint32_t>(void) { return MPI_UINT32_T; };
template<> inline MPI_Datatype data_type<int64_t >(void) { return MPI_INT64_T; };
template<> inline MPI_Datatype data_type<uint64_t>(void) { return MPI_UINT64_T; };
template<> inline MPI_Datatype data_type<float   >(void) { return MPI_FLOAT; };
template<> inline MPI_Datatype data_type<double  >(void) { return MPI_DOUBLE; };

static const MPI_Op sum = MPI_SUM;
static const int proc_null = MPI_PROC_NULL;

namespace type {
    extern MPI_Datatype int2;
    extern MPI_Datatype float2;
    extern MPI_Datatype float3;
    extern MPI_Datatype double3;
}

// These cannot be declared constexpr as their value is unknown at compile time
template<> inline MPI_Datatype data_type<int2 >(void)   { return mpi::type::int2; };
template<> inline MPI_Datatype data_type<float2 >(void) { return mpi::type::float2; };
template<> inline MPI_Datatype data_type<float3 >(void) { return mpi::type::float3; };
template<> inline MPI_Datatype data_type<double3>(void) { return mpi::type::double3; };

/**
 * @brief Initialize MPI environment and extra MPI types
 * 
 * @param argc      Pointer to command line argument count
 * @param argv      Pointer to command line arguments
 * @return int      MPI_SUCCESS on success, MPI_ERROR on failure
 */
static inline int init( int *argc, char ***argv ) {
    int ierr = MPI_Init( argc, argv );

    if ( ierr == MPI_SUCCESS ) {
        // Initialize extra types
        MPI_Type_contiguous( 2, MPI_INT,  &mpi::type::int2 ); 
        MPI_Type_commit( &mpi::type::int2 );

        MPI_Type_contiguous( 2, MPI_FLOAT,  &mpi::type::float2 ); 
        MPI_Type_commit( &mpi::type::float2 );

        MPI_Type_contiguous( 3, MPI_FLOAT,  &mpi::type::float3 ); 
        MPI_Type_commit( &mpi::type::float3 );
        
        MPI_Type_contiguous( 3, MPI_DOUBLE, &mpi::type::double3 );
        MPI_Type_commit( &mpi::type::double3 );
    } else {
        std::cerr << "Failed to initialize MPI\n";
    }
    return ierr;
}

/**
 * @brief Finialize MPI environment
 * 
 * @return int  MPI_SUCCESS on success, MPI_ERROR on failure
 */
static inline int finalize( void ) {

    // These aren't strictly necessary
    MPI_Type_free( &mpi::type::int2 );
    MPI_Type_free( &mpi::type::float2 );
    MPI_Type_free( &mpi::type::float3 );
    MPI_Type_free( &mpi::type::double3 );

    return MPI_Finalize();
}

/**
 * @brief Returns size of the global MPI communicator (MPI_COMM_WORLD)
 * 
 * @return int  Number of parallel nodes
 */
static inline int world_size( void ) {
    int size;
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    return size;
}

/**
 * @brief Returns MPI rank on the global MPI communicator (MPI_COMM_WORLD)
 * 
 * @return int  Node rank
 */
static inline int world_rank( void ) {
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    return rank;
}

/**
 * @brief Returns true if the calling node is the root node of the global MPI
 *        communicator
 * 
 * @return int  1 if the calling node is the root node, 0 otherwise 
 */
static inline int world_root( void ) { 
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    return rank == 0;
}

/**
 * @brief Abort the parallel code using an MPI_Abort()
 * 
 * @param errorcode     Error code to return to invoking environment
 * @return int          MPI_Abort() return value (should not return)
 */
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
            std::cerr << "(*error*) Unable to create partition object, MPI has not been initialized\n";
            std::cerr << "(*error*) aborting...\n";
            exit(1);
        }

        // Get communicator size
        if ( MPI_Comm_size( MPI_COMM_WORLD, &size ) != MPI_SUCCESS ) {
            std::cerr << "(*error*) Unable to get communicator size, aborting\n";
            std::cerr << "(*error*) aborting...\n";
            exit(1);
        }

        // Check dimensions
        if ( dims.x < 1 ) {
            std::cerr << "(*error*) Invalid partition dims.x = " << dims.x << "\n";
            std::cerr << "(*error*) aborting...\n";
            exit(1);
        }

        if ( dims.y < 1 ) {
            std::cerr << "(*error*) Invalid partition dims.y = " << dims.x << "\n";
            std::cerr << "(*error*) aborting...\n";
            exit(1);
        }

        if ( dims.x * dims.y != (unsigned) size ) {
            std::cerr << "(*error*) Partition size (" << dims.x * dims.y << ") and number of parallel nodes (" << size << ") don't match\n";
            std::cerr << "(*error*) aborting...\n";
            exit(1);
        }

        
        int _dims[] = { (int) dims.x, (int) dims.y } ;
        int periods[] = { periodic.x, periodic.y };

        // Create partition
        if ( MPI_Cart_create(MPI_COMM_WORLD, 2, _dims, periods, 0, &comm ) != MPI_SUCCESS ) {
            std::cerr << "(*error*) Unable to create cartesian topology\n";
            std::cerr << "(*error*) aborting...\n";
            exit(1);
        }

        // Get rank
        if ( MPI_Comm_rank( comm, & rank ) != MPI_SUCCESS ) {
            std::cerr << "(*error*) Unable to get communicator rank, aborting\n";
            std::cerr << "(*error*) aborting...\n";
            exit(1);
        }

        int lcoords[2];
        if ( MPI_Cart_coords( comm, rank, 2, lcoords ) != MPI_SUCCESS ) {
            std::cerr << "(*error*) Unable to get cartesian coordinates, aborting\n";
            std::cerr << "(*error*) aborting...\n";
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

        // Sanity check - this should never happen
        if ( neighbor[1][1] != rank ) {
            std::cerr << "(*error*) Invalid neighbor (bad partition)\n";
            std::cerr << "(*error*) aborting...\n";
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

    /**
     * @brief Returns true if the local node is the root node
     * 
     * @return int 
     */
    int root() { return rank == 0;}

    /**
     * @brief Performs an MPI_Barrier accross the partition
     * 
     */
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

        void *sendbuf = ( rank == root ) ? MPI_IN_PLACE : data;

        if ( MPI_Reduce( sendbuf, data, count, mpi::data_type<T>(), op, root,
                         comm ) != MPI_SUCCESS ) {
            std::cerr << "MPI_Reduce operation failed, aborting\n";
            MPI_Abort( comm, 1 );
        }

        return 0;
    }

    /**
     * @brief Performs an MPI_Allreduce operation in this parallel partition
     * 
     * 
     * @tparam T        Data type, must be supported by MPI
     * @param sendbuf   Input data
     * @param recvbuf   Output data (reduction result)
     * @param count     Number of data elements
     * @param op        Reduction operation
     * @return int      0 on success. On error the routine will abort the code.
     */
    template< typename T >
    int allreduce( const T * sendbuf, T * recvbuf, int count, MPI_Op op ) {
                
        if ( MPI_Allreduce( sendbuf, recvbuf, count, mpi::data_type<T>(), op, comm ) != MPI_SUCCESS ) {
            std::cerr << "MPI_Allreduce operation failed, aborting\n";
            MPI_Abort( comm, 1 );
        }

        return 0;
    }

    /**
     * @brief 
     * 
     * @note The operation is performed "in-place", i.e., the original data is
     * replaced by the reduction result
     * 
     * @tparam T 
     * @param data 
     * @param count 
     * @param op 
     * @return int 
     */
    template< typename T >
    int allreduce( T * data, int count, MPI_Op op ) {
        if ( MPI_Allreduce( MPI_IN_PLACE, data, count, mpi::data_type<T>(), op, comm ) != MPI_SUCCESS ) {
            std::cerr << "MPI_Allreduce operation failed, aborting\n";
            MPI_Abort( comm, 1 );
        }
        return 0;
    }

    /**
     * @brief Returns the local dimensions of a parallel grid
     * 
     * @note If the number of parallel nodes does not divide the global grid
     *       size evenly, the local grid will not have the same size on all
     *       nodes
     * 
     * @param global_size   Global grid size (x,y)
     * @return uint2        Local grid size (x,y)
     */
    inline uint2 grid_size( const uint2 global_size ) {
        uint2 local_size{ global_size.x / dims.x, global_size.y / dims.y };

        if ( coords.x < (int) (global_size.x % dims.x) ) local_size.x += 1;
        if ( coords.y < (int) (global_size.y % dims.y) ) local_size.y += 1;

        return local_size;
    }

    /**
     * @brief Returns the local offset of a parallel grid
     * 
     * @note If the number of parallel nodes does not divide the global grid
     *       size evenly, the local grid will not have the same size on all
     *       nodes
     * 
     * @param global_size   Global grid size (x,y)
     * @return uint2        Local offset on global grid (x,y)
     */
    inline uint2 grid_off( const int2 global_size ) {
        uint2 grid_size = { global_size.x / dims.x, global_size.y / dims.y };
        uint2 grid_off  = { coords.x * grid_size.x, coords.y * grid_size.y };

        if ( coords.x < (int) (global_size.x % dims.x) ) {
            grid_off.x += coords.x;
        } else {
            grid_off.x += global_size.x % dims.x;
        }

        if ( coords.y < (int) (global_size.y % dims.y) ) {
            grid_off.y += coords.y;
        } else {
            grid_off.y += global_size.y % dims.y;
        }

        return grid_off;
    }

    /**
     * @brief Get local dimensions / offset of a parallel grid
     *
     * @note If the number of parallel nodes does not divide the global grid
     *       size evenly, the local grid will not have the same size on all
     *       nodes
     * 
     * @param global_size   Global grid size (x,y)
     * @param local_size    Local grid size (x,y)
     * @param local_off     Local offset on global grid (x,y)
     */
    void grid_local( const uint2 global_size, uint2 & local_size, uint2 & local_off ) {
        // Size and offset for matched size / parallel dims
        local_size = { global_size.x / dims.x, global_size.y / dims.y };
        local_off  = { coords.x * local_size.x, coords.y * local_size.y };

        // Correct for unmatched global_size / parallel dims
        if ( coords.x < (int) (global_size.x % dims.x) ) {
            local_size.x += 1;
            local_off.x += coords.x;
        } else {
            local_off.x += global_size.x % dims.x;
        }

        if ( coords.y < (int) (global_size.y % dims.y) ) {
            local_size.y += 1;
            local_off.y += coords.y;
        } else {
            local_off.y += global_size.y % dims.y;
        }
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
