#pragma once

#include <array>

#include "mpi.hpp"
#include "../core/vec_types.hpp"
#include "../core/dims.hpp"

namespace mpi {
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