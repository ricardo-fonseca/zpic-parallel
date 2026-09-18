#pragma once

#include "mpi.h"
#include "../parallel/partition.hpp"

#include "../core/vec_types.hpp"
#include "../core/bounds.hpp"

#include "../util/memory.hpp"

#include "../zdf/zdf.hpp"

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>

namespace part {

/**
 * @brief Particle quantity identifiers
 * 
 */
enum class quantity { x, y, ux, uy, uz };

namespace bnd_t {
    enum type { none = 0, periodic, comm };
}

/**
 * @brief Local boundary type
 * 
 */
typedef bounds_2d<bnd_t::type> bnd_type;

/**
 * @brief edge tile direction from shift (dx, dy)
 * 
 * Returns:
 * 
 * | Δy | Δx | dir |
 * | -- | -- | --- |
 * | -1 | -1 |  0  |
 * | -1 |  0 |  1  |
 * | -1 | +1 |  2  |
 * |  0 | -1 |  3  |
 * |  0 |  0 |  4  |
 * |  0 | +1 |  5  |
 * | +1 | -1 |  6  |
 * | +1 |  0 |  7  |
 * | +1 | +1 |  8  |
 * 
 * @param dx    x edge tile shift (-1, 0 or 1)
 * @param dy    y edge tile shift (-1, 0 or 1)
 * @return int  Direction (0-8)
 */
inline constexpr int edge_dir_shift( const int dx, const int dy ) {
    return (dy + 1)*3 + (dx + 1);
}

/**
 * @brief edge tile shift (dx, dy) from direction
 * 
 * @details
 * Returns:
 * 
 * | dir | Δy | Δx |
 * | --- | -- | -- |
 * |  0  | -1 | -1 |
 * |  1  | -1 |  0 |
 * |  2  | -1 | +1 |
 * |  3  |  0 | -1 |
 * |  4  |  0 |  0 |
 * |  5  |  0 | +1 |
 * |  6  | +1 | -1 |
 * |  7  | +1 |  0 |
 * |  8  | +1 | +1 |
 * 
 * @param dir       Direction index (0-8)
 * @param dx        x edge tile direction (-1, 0 or 1)
 * @param dy        y edge tile direction (-1, 0 or 1)
 */
inline constexpr void edge_shift_dir( const int dir, int & dx, int & dy ) {
    dx = dir % 3 - 1;
    dy = dir / 3 - 1;
}

/**
 * @brief Number of edge tiles per direction
 * 
 * Returns:
 * 
 * | dir | ntiles   |
 * | --- | -------- |
 * |  0  | 1        |
 * |  1  | ntiles.x |
 * |  2  | 1        |
 * |  3  | ntiles.y |
 * |  4  | 0        |
 * |  5  | ntiles.y |
 * |  6  | 1        |
 * |  7  | ntiles.x |
 * |  8  | 1        |
 * 
 * @note Direction complies to `edge_shift_dir()`
 * 
 * @param dir               Direction (0-8)
 * @param ntiles            Number of local tiles (x,y)
 * @return unsigned int     Number of edge tiles in the specified direction
 */
inline constexpr int edge_ntiles( const int dir, const int2 ntiles ) {
    int size = 1;                                 // corners
    if ( dir == 1 || dir == 7 ) size = ntiles.x;  // y boundary
    if ( dir == 3 || dir == 5 ) size = ntiles.y;  // x boundary
    if ( dir == 4 ) size = 0;                     // local

    return size;
}

/**
 * @brief Offset (from first edge tile) per direction
 * 
 * Returns:
 * 
 * | idx |            start            |
 * | --- | --------------------------- |
 * |  0  | 0                           |
 * |  1  | 1                           |
 * |  2  | 1 +   ntiles.x              |
 * |  3  | 2 +   ntiles.x              |
 * |  4  | 2 +   ntiles.x +   ntiles.y |
 * |  5  | 2 +   ntiles.x +   ntiles.y |
 * |  6  | 2 +   ntiles.x + 2*ntiles.y |
 * |  7  | 3 +   ntiles.x + 2*ntiles.y |
 * |  8  | 3 + 2*ntiles.x + 2*ntiles.y |
 * 
 * @note This assumes the edge tile (number of particle) information is stored
 *       in a contiguous buffer, following the same order as set by
 *      `edge_shift_dir()` and sizes according to `edge_ntiles()`
 * 
 * @param dir       Direction (0-8)
 * @param ntiles    Number of local tiles (x,y)
 * @return int      Start of edge tiles in the specified direction
 */
inline constexpr int edge_tile_start( const int dir, const int2 ntiles ) {
    int a, b, c;
    a = b = c = 0;

    if (dir > 0) a = 1;
    if (dir > 2) a = 2;
    if (dir > 6) a = 3;

    if (dir > 1) b =     ntiles.x;
    if (dir > 7) b = 2 * ntiles.x;

    if (dir > 3) c =     ntiles.y;
    if (dir > 5) c = 2 * ntiles.y;

    return a + b + c ;
}

/**
 * @brief Checks whether selected edge is connectec to another node
 *
 * @details Returns true if communication along the selected edge will be
 *          required. Direction 4 (local) always returns true; callers are
 *          expected to handle it separately.
 *
 * @param dir           Direction (0-8)
 * @param local_bnd     Local boundary type (none, local periodic, comm)
 * @return bool         True if the direction corresponds to a comm boundary
 */
inline bool edge_dir_comm( const int dir, part::bnd_type const local_bnd ) {
    int dx, dy;
    part::edge_shift_dir( dir, dx, dy );

    if ( dx < 0 && local_bnd.x.lower != part::bnd_t::comm ) return false;
    if ( dx > 0 && local_bnd.x.upper != part::bnd_t::comm ) return false;
    if ( dy < 0 && local_bnd.y.lower != part::bnd_t::comm ) return false;
    if ( dy > 0 && local_bnd.y.upper != part::bnd_t::comm ) return false;

    return true;
}

/**
 * @brief Gets tile id from coordinates, including edge tiles
 * 
 * @note Assumes edge tile information is in the same tile buffer 
 *       beggining at the end of the local tile information (position 
 *       `ntiles.y * ntiles.x`) and following the order set by
 *       `edge_shift_dir()`
 * 
 * @param coords        Tile coordinates
 * @param ntiles        Tile grid dimensions
 * @param local_bnd     local boundary type (none, local periodic, comm)
 * @return int          Tile id on success, -1 on out of bounds
 */
inline int tid_coords( int2 coords, int2 const ntiles, part::bnd_type const local_bnd ) {

    // assert( coords.x >= -1 && coords.x <= ntiles.x );
    // assert( coords.y >= -1 && coords.y <= ntiles.y );

    // Local (non-parallel) periodic wrap
    if ( local_bnd.x.lower == part::bnd_t::periodic ) {
        if      ( coords.x < 0 )         coords.x += ntiles.x; 
        else if ( coords.x >= ntiles.x ) coords.x -= ntiles.x;
    }

    // Local (non-parallel) y periodic
    if ( local_bnd.y.lower == part::bnd_t::periodic ) {
        if      ( coords.y < 0 )         coords.y += ntiles.y;
        else if ( coords.y >= ntiles.y ) coords.y -= ntiles.y;
    }

    // Edge (communication with other nodes) shift
    const int dx  = ( coords.x >= ntiles.x ) - ( coords.x < 0 );
    const int dy  = ( coords.y >= ntiles.y ) - ( coords.y < 0 );
    const int dir = part::edge_dir_shift( dx, dy );

    // Local tile
    if ( dir == 4 ) return coords.y * ntiles.x + coords.x;

    // No neighbor in this direction
    if ( ! part::edge_dir_comm( dir, local_bnd ) ) return -1;

    int tid = ntiles.y * ntiles.x +                 // base
              part::edge_tile_start( dir, ntiles );

    if      ( dy == 0 ) tid += coords.y;    // x boundary (dir 3, 5)
    else if ( dx == 0 ) tid += coords.x;    // y boundary (dir 1, 7)
                                            // corners take no offset

    return tid;
}

/**
 * @brief Local tile id receiving the k-th edge tile of a given direction
 *
 * @details Edge tile data arrives ordered along the shared boundary. This maps
 *          position `k` within direction `dir` onto the local tile grid: the
 *          bottom row for dir 1, the left column for dir 3, the single corner
 *          tile for dir 0, and so on.
 *
 * @note Direction complies to `edge_shift_dir()`. Valid for
 *       `0 <= k < edge_ntiles( dir, ntiles )`.
 *
 * @param dir       Direction (0-8)
 * @param k         Position within the direction's edge tiles
 * @param ntiles    Number of local tiles (x,y)
 * @return int      Local tile id
 */
inline constexpr int local_edge_tid( const int dir, const int k, const int2 ntiles ) {
    const int stride = ( dir == 3 || dir == 5 ) ? ntiles.x : 1;

    const int xoff = ( dir % 3 == 2 ) ? ntiles.x - 1 : 0;
    const int yoff = ( dir / 3 == 2 ) ? ( ntiles.y - 1 ) * ntiles.x : 0;

    return k * stride + yoff + xoff;
}

/**
 * @brief Total number of edge tiles
 * 
 * @param ntiles    Local number of tiles (x,y)
 * @return int      Total number of tiles
 */
inline constexpr int msg_tiles( const uint2 ntiles ) {
    return  2 * ntiles.y +          // x boundary
            2 * ntiles.x +          // y boundary
            4;                      // corners
}

/**
 * @brief Total number of local tiles
 * 
 * @param ntiles    Local number of tiles (x,y)
 * @return int      Total number of tiles
 */
inline constexpr int local_tiles( const uint2 ntiles ) {
    return ntiles.x * ntiles.y;
}

/**
 * @brief Total number of tiles, including edge tiles
 * 
 * @param ntiles    Local number of tiles (x,y)
 * @return int      Total number of tiles
 */
inline constexpr int all_tiles( const uint2 ntiles ) {
    return local_tiles( ntiles ) + msg_tiles( ntiles );
}

/**
 * @brief   Data structure to hold particle sort data
 * 
 * @warning This is meant to be used only as a superclass for particle_sort. The
 *          struct does not include methods for allocating / deallocating
 *          memory
 *
 * 
 */
struct particle_sort_view {
    /// @brief Maximum number of particles to be sorted
    uint32_t max_part;
    /// @brief Particle index list [max_part]
    int *idx;
    /// @brief Number of particles in index list [local_ntiles]
    int * nidx;
    /// @brief Number of particles leaving tile in all directions [ntiles * 9]
    int * npt;
    /**
     * @brief New number of particles per tile
     * @note  Includes incoming/outgoing particles per edge tile
     */
    int * new_np;
    /// @brief Local number of tiles
    const uint2 ntiles;

    particle_sort_view( const uint32_t max_part, const uint2 ntiles ) : 
        max_part(max_part), ntiles(ntiles) {};
};

/**
 * @brief Class for particle sorting data
 * 
 * @note This class does not hold any actual particle data, only particle
 *       inidices and counts. It should work for any type of particle data.
 * 
 */
class particle_sort : public particle_sort_view {

    struct Message {
        /// @brief Buffer for all 8 messages
        int * buffer;
        /// @brief Number of incoming particles per message
        int msg_np[9];
        /// @brief Message requests
        MPI_Request requests[9];
    };

    /// @brief MPI communicator
    MPI_Comm comm;
    /// @brief Neighbor ranks
    int neighbor[9];

    private:

    /**
     * @brief Message tag for incoming messages
     * 
     * @param dir   Communication direction (0-8)
     * @return int 
     */
    inline int source_tag( int dir ) {
        return (8 - dir) | 0x100;
    }

    /**
     * @brief Message tag for outgoing messages
     * 
     * @param dir   Communication direction (0-8) 
     * @return int 
     */
    inline int dest_tag( int dir ) {
        return dir | 0x100;
    }
    
    public:

    /// @brief Incoming messages
    particle_sort::Message recv;
    /// @brief Outgoing messages
    particle_sort::Message send;

    /**
     * @brief Construct a new Particle Sort object
     * 
     * @param ntiles        Local number of tiles
     * @param max_part      Maximum number of particles in buffer
     * @param par           Parallel partition
     */
    particle_sort( uint2 const ntiles, uint32_t const max_part, mpi::cart2d & par ) :
        particle_sort_view( max_part, ntiles )
    {
        idx = memory::malloc<int>( max_part );

        // Number of local ltiles
        int local_tiles = part::local_tiles(ntiles);

        // Number of edge tiles for messaging
        int msg_tiles = part::msg_tiles(ntiles);

         // Include send buffer for number of particles leaving node
        new_np = memory::malloc<int>( local_tiles + msg_tiles );
        
        // Number of particles leaving each local tile
        nidx   = memory::malloc<int>( local_tiles );

        // Particle can move in 9 different directions
        npt = memory::malloc<int>( 9 * local_tiles );

        // Send buffer
        send.buffer = &new_np[ local_tiles ];

        // Receive buffer
        recv.buffer = memory::malloc<int>( msg_tiles );
        memory::zero( recv.buffer, msg_tiles );

        // MPI Communicator
        comm = par.get_comm();

        // Local MPI rank
        auto local = par.get_rank(); 

        // Neighbor MPI ranks and messages
        for( int dir = 0; dir < 9; dir++ ) {
            int shiftx, shifty;
            part::edge_shift_dir( dir, shiftx, shifty );
            neighbor[ dir ] = par.get_neighbor( shiftx, shifty );

            // Disable all messages to self
            // Single node periodic boundaries are handled without messages
            if ( neighbor[dir] == local ) neighbor[dir] = -1;

            // Initialize messages
            recv.requests[dir] = send.requests[dir] = MPI_REQUEST_NULL;
            recv.msg_np[dir]   = send.msg_np[dir]   = 0;
        }
    }

    /**
     * @brief Destroy the Particle Sort object
     * 
     */
    ~particle_sort() {
        memory::free( recv.buffer );
        
        memory::free( npt );
        memory::free( nidx );
        memory::free( new_np );
        memory::free( idx );
    }

    particle_sort( const particle_sort & ) = delete;
    particle_sort( particle_sort && ) = delete;

    /**
     * @brief Checks if idx buffer is large enough, grows if needed
     *
     * @note idx buffer is grown in multiples of 1 kB
     * 
     * @param new_max   New maximum number of particles required
     */
    void check_buffer( uint32_t new_max ) {
        if ( new_max > max_part ) {
            memory::free( idx );
            max_part = roundup<65536>(new_max);
            idx = memory::malloc<int>( max_part );
        }
    }

    /**
     * @brief Sets np values to 0
     * 
     */
    void reset() {
        // Reset local data and outgoing data buffer
        memory::zero( new_np, part::all_tiles( ntiles ) );
    }

    /**
     * @brief Exchange number of particles in edge cells
     *
     */
    void exchange_np( );

};


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

    enum class MessageType { none = 0, send, receive };

    /// @brief Active message type
    MessageType active;

    /// @brief Maximum data size (bytes)
    uint32_t max_size;
    /// @brief Neighbor ranks (includes self)
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
    particle_message( mpi::cart2d & par ) {

        // Buffers for particle data messages (initially empty)
        buffer = nullptr;
        max_size = 0;

        // Communicator
        comm = par.get_comm();

        // Initialize neighbor ranks and essage requests
        for( int dir = 0; dir < 9; dir++ ) {
            int shiftx, shifty;
            part::edge_shift_dir( dir, shiftx, shifty );
            neighbor[ dir ] = par.get_neighbor( shiftx, shifty );
            requests[ dir ] = MPI_REQUEST_NULL;
            size[ dir ] = 0;
        }

        active = MessageType::none;
    }

    /**
     * @brief Destroy the Particle Msg Buffer object
     * 
     */
    ~particle_message() {
        if ( active != MessageType::none ) {
            for( int i = 0; i < 9; i++ ) {
                if ( requests[i] != MPI_REQUEST_NULL )
                    MPI_Cancel( &requests[i] );
            }
            MPI_Waitall( 9, requests, MPI_STATUSES_IGNORE );
        }
        memory::free( buffer );
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
        if ( active ==  MessageType::none ) {
            if ( total_size > max_size ) {
                memory::free( buffer );
                max_size = roundup<1048576>(total_size);
                buffer = memory::malloc<uint8_t>( max_size );
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
        if ( active != MessageType::none ) {
            mpi::fatal( "isend() - Tried to send messages before other messages complete." );
        }

        active = MessageType::send;

        uint32_t offset = 0;
        for( int i = 0; i < 9; i++ ) {
            if ( (i != 4) && (size[i] > 0) ) {
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

        if ( active != MessageType::none ) {
            mpi::fatal( "irecv() - Tried to receive message before other message completes." );
        }
        active = MessageType::receive;

        // Post receives
        uint32_t offset = 0;
        for( int i = 0; i < 9; i++ ) {
            if ( ( i != 4 ) && ( size[i] > 0 ) ) {
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
        active = MessageType::none;
        for( int i = 0; i < 9; i++ ) size[i] = 0;
        return ierr;
    }
};

/**
 * @brief   Data structure to hold particle data
 * 
 * @warning This is meant to be used only as a superclass for particles. The
 *          struct does not include methods for allocating / deallocating
 *          memory
 * 
 * @note    Declaring a function parameter as `func(particles_view p)` and calling
 *          the function with a `particles` object parameter will automatically
 *          cast the value to `particles_view`. This means that we will not be
 *          creating a full copy of the `particles` object and therefore data
 *          will not be destroyed when the function reaches the end.
 */
struct particles_view {

    /// @brief Global number of tiles (x,y)
    uint2 global_ntiles;
    /// @brief Local Number of tiles (x,y)
    uint2 local_ntiles;
    /// @brief Tile grid size
    const uint2 tile_dims;
    /// @brief Start position of local tiles in global tile grid
    uint2 local_tile_start;

    /// @brief Number of particles in tile
    int * tile_np;
    /// @brief Tile particle position on global array
    int * tile_offset;

    /// @brief Particle position (cell index)
    int2 *ix;
    /// @brief Particle position (position inside cell) normalized to cell size [-0.5,0.5)
    float2 *x;
    /// @brief Particle velocity
    float3 *u;

    /// @brief Maximum number of particles in the buffer
    uint32_t max_part;

    particles_view( const uint2 global_ntiles, const uint2 tile_dims, const uint32_t max_part ) :
        global_ntiles( global_ntiles ),
        tile_dims( tile_dims ),
        max_part( max_part ) {};
};

/**
 * @brief Class for particle data
 * 
 */
class particles : public particles_view {

    protected:

    /// @brief Global periodic boundaries (x,y)
    int2 periodic;

    /// @brief Local node boundary type
    part::bnd_type local_bnd;

    /// @brief Outgoing particle data messages
    particle_message send;

    /// @brief Incoming particle data messages
    particle_message recv;

    public:

    /// @brief Parallel partition
    mpi::cart2d & parallel;

    /**
     * @brief Construct a new particles object
     * 
     * @param global_ntiles     Global number of tiles
     * @param tile_dims         Individual tile grid size
     * @param max_part          Maximum number of particles
     */
    particles( const uint2 global_ntiles, const uint2 tile_dims, const uint32_t max_part, mpi::cart2d & parallel ) :
        particles_view( global_ntiles, tile_dims, max_part ),
        send( parallel ), recv( parallel ),
        parallel( parallel )
    {

        // Get local number of tiles and position on tile grid
        parallel.grid_local( global_ntiles, local_ntiles, local_tile_start );
        
        ///@brief Total number of local tiles including edge tiles
        const size_t bsize = part::all_tiles( local_ntiles );

        // Tile information
        tile_np = memory::malloc<int>( bsize );
        tile_offset = memory::malloc<int>( bsize );

        // Initially empty
        memory::zero( tile_np, bsize );
        memory::zero( tile_offset, bsize );

        // Particle data
        ix = memory::malloc<int2>( max_part );
        x = memory::malloc<float2>( max_part );
        u = memory::malloc<float3>( max_part );

        // Default global periodic boundaries to parallel partition type
        periodic = parallel.periodic;

        // Set local bnd values
        update_local_bnd();
    }

    /**
     * @brief Destroy the particles object
     * 
     */
    ~particles() {
        memory::free( u );
        memory::free( x );
        memory::free( ix );

        memory::free( tile_offset );
        memory::free( tile_np );
    }

    particles( const particles & ) = delete;
    particles( particles && ) = delete;

    /**
     * @brief Local grid size
     * 
     * @return uint2 
     */
    uint2 local_dims() const {
        return local_ntiles * tile_dims;
    };

    /**
     * @brief Global grid size
     * 
     * @return uint2 
     */
    uint2 global_dims() const {
        return global_ntiles * tile_dims;
    }

    /**
     * @brief Update local node boundary types
     * 
     */
    void update_local_bnd() {
        
        // Default to none
        local_bnd = part::bnd_t::none;

        // Get communication boundaries
        if ( parallel.get_neighbor(-1, 0) >= 0 ) local_bnd.x.lower = part::bnd_t::comm;
        if ( parallel.get_neighbor(+1, 0) >= 0 ) local_bnd.x.upper = part::bnd_t::comm;

        if ( parallel.get_neighbor( 0,-1) >= 0 ) local_bnd.y.lower = part::bnd_t::comm;
        if ( parallel.get_neighbor( 0,+1) >= 0 ) local_bnd.y.upper = part::bnd_t::comm;

        // Correct for local node periodic
        if ( periodic.x && parallel.dims.x == 1 ) 
            local_bnd.x.lower = local_bnd.x.upper = part::bnd_t::periodic;

        if ( periodic.y && parallel.dims.y == 1 ) 
            local_bnd.y.lower = local_bnd.y.upper = part::bnd_t::periodic;

    }

    /**
     * @brief Get local node boundary types
     * 
     */
    part::bnd_type get_local_bnd() const {
        return local_bnd;
    }

    /**
     * @brief Set global periodic boundary settings
     * 
     * @param new_periodic 
     */
    void set_periodic( int2 new_periodic ) {
        // Check x direction
        if ( ( new_periodic.x ) && 
             ( (! parallel.periodic.x ) && ( parallel.dims.x > 1 )) ) {
            mpi::fatal( "particles::set_periodic() - Attempting to set "
                        "parallel x boundaries on non-parallel comm direction." );
        }

        // Check y direction
        if ( ( new_periodic.y ) && 
             ( (! parallel.periodic.y ) && ( parallel.dims.y > 1 )) ) {
            mpi::fatal( "particles::set_periodic() - Attempting to set "
                        "parallel y boundaries on non-parallel comm direction" );
        }

        // Store new global periodic values
        periodic = new_periodic;

        // update local bnd values
        update_local_bnd();
    }

    /**
     * @brief Get global periodic boundary settings
     * 
     * @return int2 
     */
    int2 get_periodic( ) const { return periodic; }

    /**
     * @brief Sets the number of particles per tile to 0
     * 
     */
    void zero_np() {
        memory::zero( tile_np, part::all_tiles( local_ntiles ) );
    }

    /**
     * @brief Grows particle data buffers
     * 
     * @warning Particle data is not copied, previous values, if any,
     *          are destroyed
     * 
     * @param new_max   New buffer size. Will be rounded up to multiple
     *                  of 64k.
     */
    void grow_buffer( uint32_t new_max ) {
        if ( new_max > max_part ) {
            memory::free( u );
            memory::free( x );
            memory::free( ix );

            // Grow in multiples 64k blocks
            max_part = roundup<65536>(new_max);

            ix = memory::malloc<int2>  ( max_part );
            x  = memory::malloc<float2>( max_part );
            u  = memory::malloc<float3>( max_part );
        }
    }

    /**
     * @brief Swaps buffers between 2 particle objects
     * 
     * @param a     Object a
     * @param b     Object b
     */
    friend void swap_buffers( particles & a, particles & b ) {
        
        // assert( a.local_ntiles == b.local_ntiles );
        // assert( a.global_ntiles == b.global_ntiles );

        std::swap( a.ix, b.ix );
        std::swap( a.x,  b.x );
        std::swap( a.u,  b.u );

        std::swap( a.max_part, b.max_part );

        std::swap( a.tile_np,     b.tile_np );
        std::swap( a.tile_offset, b.tile_offset );
    }

    /**
     * @brief Gets (node) local number of particles
     * 
     * @return uint32_t 
     */
    uint32_t local_np() const {

        // sum up number of particles in each tile
        // This works even if the buffer is not compact
        uint32_t local_np = 0;
        for( unsigned i = 0; i < local_ntiles.x*local_ntiles.y; i++ )
            local_np += tile_np[i];

/*
        // Since the buffer is kept compact we could just look at the last tile
        auto idx = ntiles.x*ntiles.y - 1;
        uint32_t np_total = offset[idx] + np[idx];
*/
        return local_np;
    }

    /**
     * @brief Gets global number of particles
     *
     * @note When all = 0 (default) returns 0 on non-root ranks
     * 
     * @param all           Return result on all parallel nodes (defaults to false)
     * @return uint64_t     Global number of particles
     */
    uint64_t global_np( bool all = false ) const {

        uint64_t local = local_np();

        if ( parallel.get_size() > 1 ) {
            if ( all ) {
                uint64_t global;
                parallel.allreduce( &local, &global, 1, mpi::sum );
                return global;
            } else {
                parallel.reduce( &local, 1, mpi::sum );
                if ( ! parallel.root() ) local = 0;
                return local;
            }
        }

        return local;
    }

    /**
     * @brief Gets maximum number of particles in a single tile
     * 
     * @return int 
     */
    int tile_np_max() const {
        int max_np = tile_np[0];
        for( unsigned i = 1; i < local_ntiles.x*local_ntiles.y; i++ ) {
            if ( tile_np[i] > max_np ) max_np = tile_np[i];
        }
        return max_np;
    }

    /**
     * @brief Gets minimum number of particles in a single tile
     * 
     * @return int 
     */
    int tile_np_min() const {
        int min_np = tile_np[0];
        for( unsigned i = 1; i < local_ntiles.x*local_ntiles.y; i++ ) {
            if ( tile_np[i] < min_np ) min_np = tile_np[i];
        }
        return min_np;
    }
    /**
     * @brief Returns local grid range
     * 
     * @return bounds_2d<uint32_t> 
     */
    bounds_2d<uint32_t> local_range() const { 
        uint2 dims = local_dims();
        
        bounds_2d<uint32_t> range;
        range.x = bounds<uint32_t>( 0, dims.x - 1 );
        range.y = bounds<uint32_t>( 0, dims.y - 1 );

        return range;
    };

    /**
     * @brief Gather data from a specific particle quantity
     * 
     * @param quant     Quantity to gather
     * @param d_data    Output data buffer, assumed to have size >= np
     */
    void gather( part::quantity quant, float * const __restrict__ d_data  );

    /**
     * @brief Gather data from a specific particle quantity, scaling values
     * 
     * @note Data (val) will be returned as `scale.x * val + scale.y`
     * 
     * @param quant     Quantity to gather
     * @param scale     Scale factor for data
     * @param d_data    Output data buffer, assumed to have size >= np
     */
    void gather( part::quantity quant, const float2 scale, float * const __restrict__ d_data );

    /**
     * @brief Validates particle data
     * 
     * @details In case of invalid particle data prints out an error message and aborts
     *          the program
     * 
     * @param msg   Message to print in case of error
     * @param over  Amount of extra cells indices beyond limit allowed. Used
     *              when checking the buffer before tile_sort(). Defaults to 0
     */
    void validate( std::string msg, int const over = 0 );

    /**
     * @brief Shifts particle cells by the required amount
     * 
     * @details Cells are shited by adding the parameter `shift` to the particle cell
     *          indexes
     * 
     * @note Does not check if the particles are still inside the tile after
     *       the shift
     * 
     * @param shift     Cell shift in both directions
     */
    void cell_shift( int2 const shift );
    
    /**
     * @brief Moves particles to the correct tiles
     * 
     * @warning This version of `tile_sort()` is provided for debug only;
     *          temporary buffers are created and destroyed on every call.
     * 
     * @param extra     (optional) Additional space to add to each tile. Leaves
     *                  room for particles to be injected later.
     */
    void tile_sort( const int * __restrict__ extra = nullptr ){
        // Create temporary buffers
        particles    tmp( global_ntiles, tile_dims, max_part, parallel );
        particle_sort sort( local_ntiles, max_part, parallel );
        
        // Call sort routine
        tile_sort( tmp, sort, extra );
    };

    /**
     * @brief Moves particles to the correct tiles
     * 
     * @note particles are only expected to have moved no more than 1 tile
     *       in each direction
     * 
     * @param tmp       Temporary particle buffer
     * @param sort      Temporary sort index 
     * @param extra     (optional) Additional space to add to each tile. Leaves
     *                  room for particles to be injected later.
     */
    void tile_sort( particles &tmp, particle_sort &sort, 
                    const int * __restrict__ extra = nullptr ); 

    /**
     * @brief Save particle data to disk
     * 
     * @param quants    Quantities to save
     * @param metadata  Particle metadata (name, labels, units, etc.). Information is used to set file name
     * @param iter      Iteration metadata
     * @param path      Path where to save the file
     */
    void save( const part::quantity quants[], zdf::part_info &metadata, zdf::iteration &iter, std::string path );


    /**
     * @brief Size (in bytes) of a single particle
     * 
     * @return size_t 
     */
    static constexpr size_t particle_size() {
        return sizeof(*ix) + sizeof(*x) + sizeof(*u);
    };

    /**
     * @brief Byte offsets of each quantity block inside a packed message
     *
     * @details Messages are packed as `np` cell indices, followed by `np`
     *          cell positions, followed by `np` generalized velocities.
     *          Offsets are relative to the start of the message.
     */
    struct packed_offsets {
        size_t ix, x, u;

        explicit constexpr packed_offsets( uint32_t const np ) :
            ix( 0 ),
            x ( ix + np * sizeof( * particles_view::ix ) ),
            u ( x  + np * sizeof( * particles_view::x ) ) {}
    };

    /**
     * @brief Prepare particle receive buffers and start receive
     * 
     * @param sort      Temporary sort index
     * @param recv      Receive message object
     */
    void irecv_msg( particle_sort &sort, particle_message &recv );

    /**
     * @brief Pack particles moving out of the node into a message buffer and start send
     * 
     * @param tmp       Temporary buffer holding particles moving away from tiles
     * @param sort      Temporary sort index
     * @param send      Send message object
     */
    void isend_msg( particles &tmp, particle_sort &sort, particle_message &send );

    /**
     * @brief Unpack received particle data into main particle data buffer
     * 
     * @param sort      Temporary sort index
     * @param recv      Receive message object
     */
    void unpack_msg( particle_sort &sort, particle_message &recv );

    /**
     * @brief Print information on the number of particles per tile
     * 
     * @warning Used for debug purposes only
     * 
     * @param msg   (optional) Message to print before printing particle information
     */
    void info_np( std::string msg = "" ) const {
        
        parallel.barrier();

        if ( ! msg.empty() && ( parallel.get_rank() == 0)) {
            std::cout << "-------------[info]> " << msg << '\n';
        }

        for( int k = 0; k < parallel.get_size() ; k++ ) {
            if ( k == parallel.get_rank() ) {
                std::cout << '\n';
                mpi::cout << "#particles per tile:\n";

                for( unsigned j = 0; j < local_ntiles.y; j++ ) {
                    mpi::cout << j << ':';
                    for( unsigned i = 0; i < local_ntiles.x; i++ ) {
                        int tid = j * local_ntiles.x + i;
                        mpi::cout << " " << tile_np[tid];
                    }
                    mpi::cout << '\n';
                }

                mpi::cout << "#particles total: " << local_np() << '\n';
            }
            parallel.barrier();
        }
    }
};

} // end of part namespace

