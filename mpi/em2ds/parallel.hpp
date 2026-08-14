#pragma once

#include "zpic.hpp"
#include "vec_types.hpp"

#include <mpi.h>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <string>
#include <array>
#include <source_location>

namespace mpi {

/**
 * @brief stream class that prepends [ MPI rank ] to every line
 * 
 */
class mpi_ostream : private std::streambuf, public std::ostream
{   
    public:
    mpi_ostream() : std::ostream(this), new_line(true), rank(-1) {}

    private:

    bool new_line;
    int rank;

    int overflow(int c) override
    {
        if (c != std::char_traits<char>::eof() && new_line ) {
            if ( rank < 0 ) {
                if ( MPI_Comm_rank( MPI_COMM_WORLD, &rank ) != MPI_SUCCESS )
                    rank = -1;
            }
            if ( rank >= 0 ) {
                std::cout << "[" << rank << "] ";
            } else {
                std::cout << "[--] ";
            }
        }
        
        new_line = ( c == '\n' );
        std::cout.put(c);

        return std::char_traits<char>::to_int_type(c);
    }

};

/**
 * @brief std::cout replacement, prepends [ MPI rank ] to every line
 * 
 */
inline mpi_ostream cout;

template< typename T > 
MPI_Datatype data_type () { 
    static_assert( sizeof(T) == 0,"No MPI data type for T"); 
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

template<> inline MPI_Datatype data_type<std::complex<float >>(void) { return MPI_C_FLOAT_COMPLEX ; };
template<> inline MPI_Datatype data_type<std::complex<double >>(void) { return MPI_C_DOUBLE_COMPLEX ; };

inline const MPI_Op sum = MPI_SUM;
inline constexpr int proc_null = MPI_PROC_NULL;

namespace type {
    inline MPI_Datatype mpi_int2    = MPI_DATATYPE_NULL;
    inline MPI_Datatype mpi_float2  = MPI_DATATYPE_NULL;
    inline MPI_Datatype mpi_float3  = MPI_DATATYPE_NULL;
    inline MPI_Datatype mpi_double3 = MPI_DATATYPE_NULL;
}

// These cannot be declared constexpr as their value is unknown at compile time
template<> inline MPI_Datatype data_type<int2 >()   { return mpi::type::mpi_int2; };
template<> inline MPI_Datatype data_type<float2 >() { return mpi::type::mpi_float2; };
template<> inline MPI_Datatype data_type<float3 >() { return mpi::type::mpi_float3; };
template<> inline MPI_Datatype data_type<double3>() { return mpi::type::mpi_double3; };

/**
 * @brief Initialize MPI environment and extra MPI types
 * 
 * @param argc      Pointer to command line argument count
 * @param argv      Pointer to command line arguments
 * @return int      MPI_SUCCESS on success, MPI_ERROR on failure
 */
inline int init( int *argc, char ***argv ) {
    #ifdef _OPENMP
    int provided;
    int ierr = MPI_Init_thread( argc, argv, MPI_THREAD_FUNNELED, &provided);
    if ( provided < MPI_THREAD_FUNNELED ) {
        std::cerr << "MPI library does not support MPI_THREAD_FUNNELED\n";
        return -1;
    }
    #else
    int ierr = MPI_Init( argc, argv );
    #endif

    if ( ierr == MPI_SUCCESS ) {
        // Initialize extra types
        MPI_Type_contiguous( 2, MPI_INT,  &mpi::type::mpi_int2 ); 
        MPI_Type_commit( &mpi::type::mpi_int2 );

        MPI_Type_contiguous( 2, MPI_FLOAT,  &mpi::type::mpi_float2 ); 
        MPI_Type_commit( &mpi::type::mpi_float2 );

        MPI_Type_contiguous( 3, MPI_FLOAT,  &mpi::type::mpi_float3 ); 
        MPI_Type_commit( &mpi::type::mpi_float3 );
        
        MPI_Type_contiguous( 3, MPI_DOUBLE, &mpi::type::mpi_double3 );
        MPI_Type_commit( &mpi::type::mpi_double3 );
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
inline int finalize( ) {

    // These aren't strictly necessary
    if ( mpi::type::mpi_int2    != MPI_DATATYPE_NULL ) MPI_Type_free( &mpi::type::mpi_int2 );
    if ( mpi::type::mpi_float2  != MPI_DATATYPE_NULL ) MPI_Type_free( &mpi::type::mpi_float2 );
    if ( mpi::type::mpi_float3  != MPI_DATATYPE_NULL ) MPI_Type_free( &mpi::type::mpi_float3 );
    if ( mpi::type::mpi_double3 != MPI_DATATYPE_NULL ) MPI_Type_free( &mpi::type::mpi_double3 );

    return MPI_Finalize();
}

/**
 * @brief Fatal error, outputs message and aborts the code
 * 
 * @param msg       Message to print
 * @param location  (optional) source_location object, defaults to where the function was called
 */
[[noreturn]] inline void fatal(const std::string& msg, 
    const std::source_location location =
          std::source_location::current()) {
    std::cerr << "(* fatal *) " << msg << '\n'
              << "(* fatal * ) " << location.file_name() << ':' << location.line()
              << " " << location.function_name() << '\n'
              << "(* fatal *) aborting..." << std::endl;
    MPI_Abort( MPI_COMM_WORLD, 1 );
    
    // unreachable, silences noreturn analysis
    std::exit(1);
}

/**
 * @brief Returns size of MPI communicator
 * 
 * @param comm  MPI communicator, defaults to MPI_COMM_WORLD
 * @return int 
 */
inline int size( MPI_Comm comm = MPI_COMM_WORLD ) {
    int size;
    if ( MPI_Comm_size( comm, &size ) != MPI_SUCCESS )
        mpi::fatal( "Unable to get communicator size" );
    return size;
}

/**
 * @brief Returns process rank
 * 
 * @param comm  MPI communicator, defaults to MPI_COMM_WORLD
 * @return int 
 */
inline int rank( MPI_Comm comm = MPI_COMM_WORLD ) {
    int rank;
    if ( MPI_Comm_rank( comm, &rank ) != MPI_SUCCESS )
        mpi::fatal( "Unable to get process rank");
    return rank;
}

/**
 * @brief Returns true if the calling node is the root node of the MPI
 *        communicator
 * 
 * @param comm   MPI communicator, defaults to MPI_COMM_WORLD
 * @return bool  1 if the calling node is the root node, 0 otherwise 
 */
inline bool root( MPI_Comm comm = MPI_COMM_WORLD ) {
    int rank;
    MPI_Comm_rank( comm, &rank );
    return rank == 0;
}

/**
 * @brief Performs an MPI_Barrier on the MPI communicator
 * 
 * @param comm   MPI communicator, defaults to MPI_COMM_WORLD
 * @return int 
 */
inline int barrier( MPI_Comm comm = MPI_COMM_WORLD ) {
    return MPI_Barrier( comm );
}


/**
 * @brief Abort the parallel code using an MPI_Abort()
 * 
 * @param errorcode     Error code to return to invoking environment
 * @param comm          MPI communicator, defaults to MPI_COMM_WORLD
 * @return int          MPI_Abort() return value (should not return)
 */
inline int abort( int errorcode, MPI_Comm comm = MPI_COMM_WORLD ) {
    return MPI_Abort( comm, errorcode );
}




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


/**
 * @brief Parallel partition
 * 
 */
class cart2d {
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
    std::array<std::array<int, 3>, 3> neighbor;

    public:

    /// @brief Dimensions of the parallel partition
    const uint2 dims;

    /// @brief Periodicity of the parallel partition
    const int2 periodic;

    /**
     * @brief Construct a new 2D cartesian topology object
     * 
     * @param dims      Partition dimension
     * @param periods   Peridocity (defaults to true on both directions)
     */
    cart2d( uint2 dims, int2 periodic = make_int2(1,1) ) : dims(dims), periodic(periodic) 
    {
        // Check if MPI has been initialized
        int flag; MPI_Initialized( &flag );

        if ( ! flag ) {
            std::cerr << "(*fatal*) Unable to create partition object, MPI has not been initialized\n"
                         "(*fatal*) aborting...\n";
            std::exit(1);
        }

        // Get communicator size
        size = mpi::size( MPI_COMM_WORLD );

        // Check dimensions
        if ( dims.x < 1 )
            mpi::fatal( "Invalid partition dims.x = " + std::to_string( dims.x ) );

        if ( dims.y < 1 )
            mpi::fatal( "Invalid partition dims.y = " + std::to_string( dims.y ) );

        if ( dims.x * dims.y != (unsigned) size ) {
            if ( mpi::root() ) {
                std::cerr << "(*fatal*) Partition size (" << dims.x * dims.y << ") and number of MPI parallel nodes (" << size << ") don't match\n"
                          << "(*fatal*) aborting...\n";
            }
            mpi::abort(1);
        }

        
        int _dims[] = { (int) dims.x, (int) dims.y } ;
        int periods[] = { periodic.x, periodic.y };

        // Create partition
        if ( MPI_Cart_create(MPI_COMM_WORLD, 2, _dims, periods, 0, &comm ) != MPI_SUCCESS ) {
            mpi::fatal("Unable to create cartesian topology");
        }

        // Get rank
        rank = mpi::rank( comm );

        int lcoords[2];
        if ( MPI_Cart_coords( comm, rank, 2, lcoords ) != MPI_SUCCESS ) {
            mpi::fatal( "Unable to get cartesian coordinates" );
        };
        coords = make_int2( lcoords[0], lcoords[1] );

        // Get neighbors
        // Since we also need the corner neighbors we cannot use MPI_Cart_shift()
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
            mpi::fatal( "Invalid neighbor (bad partition)" );
        }; 
    };

    /**
     * @brief Delete the default copy constructor
     * 
     */
    cart2d(const cart2d&) = delete;

    /**
     * @brief Delete the default copy constructor
     * 
     * @return cart2d& 
     */
    cart2d& operator=(const cart2d&) = delete;

    /**
     * @brief Move-construct a cart2d object
     * 
     * @note Leaves `other` in a valid but empty state (its destructor becomes
     *       a no-op, since MPI_Comm_free() cannot be called twice on the same
     *       communicator)
     * 
     * @param other     cart2d object to move from
     */
    cart2d( cart2d && other ) noexcept :
        comm( other.comm ), size( other.size ), rank( other.rank ), coords( other.coords ), 
        neighbor( other.neighbor ), dims( other.dims ), periodic( other.periodic )
    {
        other.comm = MPI_COMM_NULL;
    }
 
    // Move assignment is not available: `dims` and `periodic` are const
    // members, so they cannot be reassigned after construction. If you need
    // move-assignable cart2d objects, those members would have to lose
    // their `const` qualifier.
    cart2d& operator=(cart2d&&) = delete;

    /**
     * @brief Destroy the cart2d object
     * 
     */
    ~cart2d() {
        if ( comm != MPI_COMM_NULL ) MPI_Comm_free( & comm );
    };

    /**
     * @brief Prints information about local node
     * 
     */
    void info() const {
        mpi::cout << '[' << rank << '/' << size << "] - coords " << coords << '\n';
    }

    /**
     * @brief Returs the MPI communicator
     * 
     * @return MPI_Comm     MPI Communicator
     */
    MPI_Comm get_comm() const noexcept  {
        return comm;
    }

    /**
     * @brief Get parallel partition size
     * 
     * @return int  Partition size
     */
    int get_size() const noexcept  {
        return size;
    }

    /**
     * @brief Get the local process rank
     * 
     * @return int  Local process rank
     */
    int get_rank() const noexcept {
        return rank;
    }

    /**
     * @brief Get the neighbor process rank
     * 
     * @param shiftx    Shift along x direction, should be -1, 0 or +1
     * @param shifty    Shift along y direction, should be -1, 0 or +1
     * @return int      neighbor rank
     */
    int get_neighbor( int shiftx, int shifty ) const {
        return neighbor[ 1 + shifty ][ 1 + shiftx ];
    }

    /**
     * @brief Get local coordinates
     * 
     * @param local_coords     Local process coordinates
     */
    int2 get_coords( ) const {
        return coords;
    }

    /**
     * @brief Get coordinates of a specific process
     * 
     * @param target_rank       Target process rank
     * @param target_coords     Target process coordinates
     */
    int2 get_coords_rank( const int target_rank ) const {
        int _coords[2];
        if ( MPI_Cart_coords( comm, target_rank, 2, _coords ) != MPI_SUCCESS )
            mpi::fatal("Unable to get coordinates for rank " + std::to_string(target_rank));
        int2 target_coords = make_int2( _coords[0], _coords[1] );
        return target_coords;
    }

    /**
     * @brief Gets the rank of the process at specific coordinates
     * 
     * @param target_coords     Target coordinates
     * @return int              Target process rank
     */
    int get_rank_coords( const int2 target_coords ) const {
        int cart_rank;
        int _coords[2] = { target_coords.x, target_coords.y };
        if ( MPI_Cart_rank( comm, _coords, &cart_rank ) != MPI_SUCCESS )
            mpi::fatal( "Unable to get rank from coordinates " +
                to_string(target_coords));
        return cart_rank;
    }

    /**
     * @brief Returns a unique integer based on the local coords
     * 
     * @return int 
     */
    int coords_id( ) const {
        return coords.y * dims.x + coords.x;
    }

    private:

    /**
     * @brief Checks whether given coordinates lie on the requested edge of
     *        the partition
     * 
     * @param coord         Coordinate to check (coord::x, coord::y)
     * @param edge          Edge to check (edge::lower, edge::upper)
     * @param node_coords   Coordinates to test
     * @return int          Returns 1 if node_coords is on the requested edge
     */
    int on_edge_impl( coord::cart coord, edge::pos edge, int2 node_coords ) const {
        switch (coord) {
        case coord::x:
            switch(edge) {
                case edge::lower: return node_coords.x == 0;
                case edge::upper: return node_coords.x == (int) (dims.x-1);
            }
            break;
        case coord::y:
            switch(edge) {
                case edge::lower: return node_coords.y == 0;
                case edge::upper: return node_coords.y == (int) (dims.y-1);
            }
            break;
        default: break;
        }
        return 0;
    }

    public:

    /**
     * @brief Returns true if local node is on the edge of the partition
     * 
     * @param coord     Coordinate to check (coord::x, coord::y)
     * @param edge      Edge to check (edge::lower, edge::upper)
     * @return int      Returns 1 if node in on the requested edge
     */
    int on_edge( coord::cart coord, edge::pos edge ) const {
        return on_edge_impl( coord, edge, coords );
    }

    /**
     * @brief Returns true if target node is on the edge of the partition
     * 
     * @param coord         Coordinate to check (coord::x, coord::y)
     * @param edge          Edge to check (edge::lower, edge::upper)
     * @param target_rank   Target node rank
     * @return int          Returns 1 if node in on the requested edge
     */
    int on_edge( coord::cart coord, edge::pos edge, int target_rank ) const {
        return on_edge_impl( coord, edge, get_coords_rank( target_rank ) );
    }

    /**
     * @brief Returns true if the local node is the root node
     * 
     * @return bool 
     */
    bool root() const { return rank == 0;}

    /**
     * @brief Performs an MPI_Barrier accross the partition
     * 
     */
    void barrier() {
        if ( MPI_Barrier( comm ) != MPI_SUCCESS ) {
            mpi::fatal( "Barrier failed" );
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
     */
    template< typename T >
    void reduce( T * data, int count, MPI_Op op, int root = 0 ) {

        void *sendbuf = ( rank == root ) ? MPI_IN_PLACE : data;

        if ( MPI_Reduce( sendbuf, data, count, mpi::data_type<T>(), op, root,
                         comm ) != MPI_SUCCESS ) {
            mpi::fatal("Reduce operation failed");
        }
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
     */
    template< typename T >
    void allreduce( const T * sendbuf, T * recvbuf, int count, MPI_Op op ) {
                
        if ( MPI_Allreduce( sendbuf, recvbuf, count, mpi::data_type<T>(), op, comm ) != MPI_SUCCESS ) {
            mpi::fatal( "Allreduce operation failed");
        }
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
    void allreduce( T * data, int count, MPI_Op op ) {
        if ( MPI_Allreduce( MPI_IN_PLACE, data, count, mpi::data_type<T>(), op, comm ) != MPI_SUCCESS ) {
            mpi::fatal( "Allreduce operation failed");
        }
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
    inline uint2 grid_size( const uint2 global_size ) const {
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
    inline uint2 grid_off( const uint2 global_size ) const {
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
     * @param local_start   Local start position on global grid (x,y)
     */
    void grid_local( const uint2 global_size, uint2 & local_size, uint2 & local_start ) const {
        // Size and offset for matched size / parallel dims
        local_size = { global_size.x / dims.x, global_size.y / dims.y };
        local_start  = { coords.x * local_size.x, coords.y * local_size.y };

        // Correct for unmatched global_size / parallel dims
        if ( coords.x < (int) (global_size.x % dims.x) ) {
            local_size.x += 1;
            local_start.x += coords.x;
        } else {
            local_start.x += global_size.x % dims.x;
        }

        if ( coords.y < (int) (global_size.y % dims.y) ) {
            local_size.y += 1;
            local_start.y += coords.y;
        } else {
            local_start.y += global_size.y % dims.y;
        }
    }
};

}





