#ifndef PARTICLES_H_
#define PARTICLES_H_

#include "parallel.h"

#include "zpic.h"
#include "vec_types.h"
#include "bnd.h"

#include "zdf-cpp.h"

namespace part {

/**
 * @brief Particle quantity identifiers
 * 
 */
enum quant { x, y, ux, uy, uz };

namespace bnd_t {
    enum type { none = 0, periodic, comm };
}

/**
 * @brief Local boundary type
 * 
 */
typedef bnd<bnd_t::type> bnd_type;

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
 * @return int  Direction (0-9)
 */
__host__ __device__
inline int edge_dir_shift( const int dx, const int dy ) {
    return (dy + 1)*3 + (dx + 1);
}

/**
 * @brief edge tile shift (dx, dy) from direction
 * 
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
 * @param dir       Direction index (0-9)
 * @param dx        x edge tile direction (-1, 0 or 1)
 * @param dy        y edge tile direction (-1, 0 or 1)
 */
inline void edge_shift_dir( const int dir, int & dx, int & dy ) {
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
 * @param dir       Direction (0-9)
 * @param ntiles    Number of local tiles (x,y)
 * @return int      Number of edge tiles in the specified direction
 */
inline unsigned int edge_ntiles( const int dir, const uint2 ntiles ) {
    unsigned int size = 1;                        // corners
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
 * | idx |            offset           |
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
 * @return int      Offset of edge tiles in the specified direction
 */
inline int edge_tile_off( const int dir, const uint2 ntiles ) {
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
__host__ __device__
inline int tid_coords( int2 coords, int2 const ntiles, part::bnd_type const local_bnd ) {

    // Local (non-parallel) x periodic
    if ( local_bnd.x.lower == part::bnd_t::periodic ) {
        if      ( coords.x < 0 )         coords.x += ntiles.x; 
        else if ( coords.x >= ntiles.x ) coords.x -= ntiles.x;
    }

    // Local (non-parallel) y periodic
    if ( local_bnd.y.lower == part::bnd_t::periodic ) {
        if      ( coords.y < 0 )         coords.y += ntiles.y;
        else if ( coords.y >= ntiles.y ) coords.y -= ntiles.y;
    }

    // Parallel shift
    int xshift = ( coords.x >= ntiles.x ) - ( coords.x < 0 );
    int yshift = ( coords.y >= ntiles.y ) - ( coords.y < 0 );
    int dir    = part::edge_dir_shift( xshift, yshift );

    int tid = -1;
    switch (dir)
    {
    case 0: // lower y, lower x
        if (( local_bnd.y.lower == part::bnd_t::comm ) &&
            ( local_bnd.x.lower == part::bnd_t::comm ))
            tid = ntiles.y * ntiles.x; // base
        break;
    case 1: // lower y
        if ( local_bnd.y.lower == part::bnd_t::comm )
            tid = ntiles.y * ntiles.x + // base
                  1 +                   // 0
                  coords.x;
        break;
    case 2: // lower y, upper x
        if (( local_bnd.y.lower == part::bnd_t::comm ) &&
            ( local_bnd.x.upper == part::bnd_t::comm ))
            tid = ntiles.y * ntiles.x + // base
                  1 +                   // 0
                  ntiles.x;             // 1
        break;
    case 3: // lower x
        if ( local_bnd.x.lower == part::bnd_t::comm )
            tid = ntiles.y * ntiles.x + // base
                  1 +                   // 0
                  ntiles.x +            // 1
                  1 +                   // 2
                  coords.y;
        break;
    case 4: // local tiles
        tid = coords.y * ntiles.x + coords.x;
        break;
    case 5: // upper x
        if ( local_bnd.x.upper == part::bnd_t::comm )
            tid = ntiles.y * ntiles.x + // base
                1 +                     // 0
                ntiles.x +              // 1
                1 +                     // 2
                ntiles.y +              // 3
                coords.y;
        break;
    case 6: // upper y, lower x
        if (( local_bnd.y.upper == part::bnd_t::comm ) &&
            ( local_bnd.x.lower == part::bnd_t::comm ))
            tid = ntiles.y * ntiles.x + // base
                1 +                     // 0
                ntiles.x +              // 1
                1 +                     // 2
                ntiles.y +              // 3
                ntiles.y;               // 5
        break;
    case 7: // upper y
        if ( local_bnd.y.upper == part::bnd_t::comm )
            tid = ntiles.y * ntiles.x + // base
                1 +                     // 0
                ntiles.x +              // 1
                1 +                     // 2
                ntiles.y +              // 3
                ntiles.y +              // 5
                1 +                     // 6
                coords.x;
        break;
    case 8: // upper y, upper x
        if (( local_bnd.y.upper == part::bnd_t::comm ) &&
            ( local_bnd.x.upper == part::bnd_t::comm ))
            tid = ntiles.y * ntiles.x + // base
                1 +                     // 0
                ntiles.x +              // 1
                1 +                     // 2
                ntiles.y +              // 3
                ntiles.y +              // 5
                1 +                     // 6
                ntiles.x;               // 7
        break;
    default:
        tid = -1;
        break;
    }
    return tid;
}

/**
 * @brief Total number of edge tiles
 * 
 * @param ntiles    Local number of tiles (x,y)
 * @return int      Total number of tiles
 */
__host__ __device__
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
__host__ __device__
inline constexpr int local_tiles( const uint2 ntiles ) {
    return ntiles.x * ntiles.y;
}

/**
 * @brief Total number of tiles, including edge tiles
 * 
 * @param ntiles    Local number of tiles (x,y)
 * @return int      Total number of tiles
 */
__host__ __device__
inline constexpr int all_tiles( const uint2 ntiles ) {
    return local_tiles( ntiles ) + msg_tiles( ntiles );
}

}

/**
 * @brief   Data structure to hold particle sort data
 * 
 * @warning This is meant to be used only as a superclass for ParticleSort. The
 *          struct does not include methods for allocating / deallocating
 *          memory
 *
 * 
 */
struct ParticleSortData {
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
    /// @brief Total number of tiles
    const uint2 ntiles;

    struct Message {
        /// @brief Buffer for all 8 messages (in device memory)
        int * buffer;
        /// @brief Number of incoming particles per message
        int msg_np[9];

        int * d_msg_np;

        /// @brief Total number of particles to be exchanged
        int total_np;
        /// @brief Message requests
        MPI_Request requests[9];
    };

    /// @brief Incoming messages
    ParticleSortData::Message recv;
    /// @brief Outgoing messages
    ParticleSortData::Message send;

    /// @brief MPI communicator
    MPI_Comm comm;
    /// @brief Neighbor ranks
    int neighbor[9];

    ParticleSortData( const uint2 ntiles, Partition & par ) : 
        ntiles(ntiles) {
            recv.d_msg_np = device::malloc<int>(9);
            send.d_msg_np = device::malloc<int>(9);
        };
    
    ~ParticleSortData() {
        device::free( send.d_msg_np );
        device::free( recv.d_msg_np );
    }
};

/**
 * @brief Class for particle sorting data
 * 
 * @note This class does not hold any actual particle data, only particle
 *       inidices and counts. It should work for any type of particle data.
 * 
 */
class ParticleSort : public ParticleSortData {
    
    private:

    /**
     * @brief Message tag for incoming messages
     * 
     * @param dir   - Communication direction (0-8)
     * @return int 
     */
    inline int source_tag( int dir ) {
        return (8 - dir) | 0x100;
    }

    /**
     * @brief Message tag for outgoing messages
     * 
     * @param dir   - Communication direction (0-8) 
     * @return int 
     */
    inline int dest_tag( int dir ) {
        return dir | 0x100;
    }

    
    public:

    /**
     * @brief Construct a new Particle Sort object
     * 
     * @param ntiles        Local number of tiles
     * @param max_part      Maximum number of particles in buffer
     * @param par           Parallel partition
     */
    ParticleSort( uint2 const ntiles, uint32_t const max_part, Partition & par ) :
        ParticleSortData( ntiles, par )
    {
        idx = device::malloc<int>( max_part );

        auto local_tiles = ntiles.x * ntiles.y;

        auto edge_tiles = 2 * ntiles.y + // x boundary
                          2 * ntiles.x + // y boundary
                          4;             // corners

        // Include send / receive buffers for number of particles leaving/entering node
        new_np = device::malloc<int>( local_tiles + 2 * edge_tiles );
        
        // Number of particles leaving each local tile
        nidx   = device::malloc<int>( local_tiles );

        // Particle can move in 9 different directions
        npt = device::malloc<int>( 9 * local_tiles );

        // Send buffer
        send.buffer = &new_np[ local_tiles ];

        // Receive buffer
        recv.buffer = &new_np[ local_tiles + edge_tiles ];

        // Communicator
        comm = par.get_comm();

        // Neighbor ranks
        for( int dir = 0; dir < 9; dir++ ) {
            int shiftx, shifty;
            part::edge_shift_dir( dir, shiftx, shifty );
            neighbor[ dir ] = par.get_neighbor( shiftx, shifty );
        }
    }

    /**
     * @brief Destroy the Particle Sort object
     * 
     */
    ~ParticleSort() {
        device::free( npt );
        device::free( nidx );
        device::free( new_np );
        device::free( idx );
    }

    /**
     * @brief Sets np values to 0
     * 
     */
    void reset() {
        auto local_tiles = ntiles.x * ntiles.y;

        auto edge_tiles = 2 * ntiles.y + // x boundary
                          2 * ntiles.x + // y boundary
                          4;             // corners

        // No need to reset incoming message buffers
        device::zero( new_np, local_tiles + edge_tiles );
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
class ParticleMessage {

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
    ParticleMessage( Partition & par ) {

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
        }

        active = MessageType::none;
    }

    /**
     * @brief Destroy the Particle Msg Buffer object
     * 
     */
    ~ParticleMessage() {
        if ( active != MessageType::none ) {
            for( int i = 0; i < 9; i++ ) {
                MPI_Request tmp = requests[i];
                MPI_Cancel( &tmp );
            }
        }
        device::free( buffer );
    }

    /**
     * @brief Checks if data buffer is large enough to hold all messages and grows
     *        it if necessary
     * @note Buffer is grown in multiples of 1 MB
     * 
     * @param total_size    Total required size in bytes
     */
    void check_buffer( uint32_t total_size ) {
        if ( total_size > max_size ) {
            device::free( buffer );
            max_size = roundup<1048576>(total_size);
            buffer = device::malloc<uint8_t>( max_size );
        }
    }

    /**
     * @brief Start all non-blocking send messages
     * 
     */
    void isend( ) {
        if ( active != MessageType::none ) {
            std::cerr << "isend() - Tried to send messages before other messages complete\n";
            mpi::abort(1);
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
            std::cerr << "irecv() - Tried to receive message before other message completes\n";
            mpi::abort(1);
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
        return ierr;
    }
};

#if 0
/**
 * @brief   Data structure to hold particle data
 * 
 * @warning This is meant to be used only as a superclass for Particles. The
 *          struct does not include methods for allocating / deallocating
 *          memory
 * 
 * @note    Declaring a function parameter as `func(ParticleData p)` and calling
 *          the function with a `Particles` object parameter will automatically
 *          cast the value to `ParticleData`. This means that we will not be
 *          creating a full copy of the `Particles` object and therefore data
 *          will not be destroyed when the function reaches the end.
 * 
 *          
 */
struct ParticleData {

    /// @brief Number of tiles (x,y)
    const uint2 ntiles;  
    /// @brief Tile grid size
    const uint2 nx;
    /// @brief Number of particles in tile
    int * np;
    /// @brief Tile particle position on global array
    int * offset;

    /// @brief Particle position (cell index)
    int2 *ix;
    /// @brief Particle position (position inside cell) normalized to cell size [-0.5,0.5)
    float2 *x;
    /// @brief Particle velocity
    float3 *u;

    /// @brief Maximum number of particles in the buffer
    uint32_t max_part;

    ParticleData( const uint2 ntiles, const uint2 nx, const uint32_t max_part ) :
        ntiles( ntiles ), nx( nx ), max_part( max_part ) {};
};


/**
 * @brief Class for particle data
 * 
 */
 class Particles : public ParticleData {

    private:

    /// @brief Device variable for returning uint32 values
    device::Var<uint32_t> dev_tmp_uint32;

    void bnd_check( ParticleSort & sort );
    void copy_out( ParticleData & tmp, const ParticleSortData & sort );
    void copy_in( ParticleData & tmp );
    void copy_sorted( ParticleData & tmp, const ParticleSortData & sort );

    public:

    /// @brief Sets periodic boundaries (x,y)
    int2 periodic;

    /// Global grid size
    const uint2 gnx;

    /**
     * @brief Construct a new Particles object
     * 
     * @param ntiles    Number of tiles
     * @param nx        Tile grid size
     * @param max_part  Maximum number of particles
     */
    Particles( const uint2 ntiles, const uint2 nx, const uint32_t max_part ) :
        ParticleData( ntiles, nx, max_part ), dev_tmp_uint32(), 
        periodic( int2{1,1} ), gnx ( uint2{ntiles.x * nx.x, ntiles.y * nx.y} )
    {
        const size_t bsize = ntiles.x * ntiles.y;
        
        // Tile information
        np     = device::malloc<int>( bsize );
        offset = device::malloc<int>( bsize );

        // Initially empty
        device::zero( np, bsize );
        device::zero( offset, bsize );

        // Particle data
        ix = device::malloc<int2>  ( max_part );
        x  = device::malloc<float2>( max_part );
        u  = device::malloc<float3>( max_part );
    }

    ~Particles() {
        device::free( u );
        device::free( x );
        device::free( ix );

        device::free( offset );
        device::free( np );
    }

    /**
     * @brief Sets the number of particles per tile to 0
     * 
     */
    void zero_np( ) {
        device::zero( np, ntiles.x * ntiles.y );
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
            device::free( u );
            device::free( x );
            device::free( ix );

            // Grow in multiples 64k blocks
            max_part = roundup<65536>(new_max);

            ix = device::malloc<int2>( max_part );
            x  = device::malloc<float2>( max_part );
            u  = device::malloc<float3>( max_part );
        }
    }

    /**
     * @brief Swaps buffers between 2 particle objects
     * 
     * @param a     Object a
     * @param b     Object b
     */
    friend void swap_buffers( Particles & a, Particles & b ) {
        swap( a.ix, b.ix );
        swap( a.x,  b.x );
        swap( a.u,  b.u );

        auto tmp_max_part = b.max_part;
        b.max_part = a.max_part;
        a.max_part = tmp_max_part;

        swap( a.np,     b.np );
        swap( a.offset, b.offset );
    }

    /**
     * @brief Gets total number of particles
     * 
     * @return uint32_t 
     */
    uint32_t np_total();

    /**
     * @brief Gets maximum number of particles in a single tile
     * 
     * @return uint32_t 
     */
    uint32_t np_max_tile();

    /**
     * @brief Gets minimum number of particles in a single tile
     * 
     * @return uint32_t 
     */
    uint32_t np_min_tile();

    /**
     * @brief Returns global grid range
     * 
     * @return bnd<uint32_t> 
     */
    bnd<uint32_t> g_range() { 
        bnd<uint32_t> range;
        range.x = { .lower = 0, .upper = gnx.x - 1 };
        range.y = { .lower = 0, .upper = gnx.y - 1 };

        return range;
    };

    /**
     * @brief Gather data from a specific particle quantity
     * 
     * @param quant     Quantity to gather
     * @param d_data    Output data buffer, assumed to have size >= np
     */
    void gather( part::quant quant, float * const __restrict__ d_data );

    void gather_host( part::quant quant, float * const __restrict__ d_data, float * const __restrict__ h_data,
    uint32_t const np );

    /**
     * @brief Gather data from a specific particle quantity, scaling values
     * 
     * @note Data (val) will be returned as `scale.x * val + scale.y`
     * 
     * @param quant     Quantity to gather
     * @param d_data    Output data buffer, assumed to have size >= np
     * @param scale     Scale factor for data
     */
    void gather( part::quant quant, float * const __restrict__ d_data, const float2 scale );

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
     *          temporary buffers are created and removed each time the
     *          function is called.
     * 
     * @param extra     (optional) Additional space to add to each tile. Leaves
     *                  room for particles to be injected later.
     */
    void tile_sort( const int * __restrict__ extra = nullptr ){
        // Create temporary buffers
        Particles tmp( ntiles, nx, max_part );
        ParticleSort sort( ntiles, max_part );
        
        // Call sort routine
        tile_sort( tmp, sort, extra );
    };

    /**
     * @brief Moves particles to the correct tiles
     * 
     * @note Particles are only expected to have moved no more than 1 tile
     *       in each direction
     * 
     * @param tmp       Temporary particle buffer
     * @param sort      Temporary sort index 
     * @param extra     (optional) Additional space to add to each tile. Leaves
     *                  room for particles to be injected later.
     */
    void tile_sort( Particles &tmp, ParticleSort &sort,
                    const int * __restrict__ extra = nullptr ); 

    /**
     * @brief Save particle data to disk
     * 
     * @param info  Particle metadata (name, labels, units, etc.). Information is used to set file name
     * @param iter  Iteration metadata
     * @param path  Path where to save the file
     */
    void save( zdf::part_info &metadata, zdf::iteration &iter, std::string path );

};

#endif

/**
 * @brief   Data structure to hold particle data
 * 
 * @warning This is meant to be used only as a superclass for Particles. The
 *          struct does not include methods for allocating / deallocating
 *          memory
 * 
 * @note    Declaring a function parameter as `func(ParticleData p)` and calling
 *          the function with a `Particles` object parameter will automatically
 *          cast the value to `ParticleData`. This means that we will not be
 *          creating a full copy of the `Particles` object and therefore data
 *          will not be destroyed when the function reaches the end.
 */
 struct ParticleData {

    /// @brief Global number of tiles (x,y)
    uint2 global_ntiles;
    /// @brief Local Number of tiles (x,y)
    uint2 ntiles;
    /// @brief Tile grid size
    const uint2 nx;
    /// @brief Offset of local tiles in global tile grid
    uint2 tile_off;

    /// @brief Number of particles in tile
    int * np;
    /// @brief Tile particle position on global array
    int * offset;

    /// @brief Particle position (cell index)
    int2 *ix;
    /// @brief Particle position (position inside cell) normalized to cell size [-0.5,0.5)
    float2 *x;
    /// @brief Particle velocity
    float3 *u;

    /// @brief Maximum number of particles in the buffer
    uint32_t max_part;

    ParticleData( const uint2 global_ntiles, const uint2 nx, const uint32_t max_part ) :
        global_ntiles( global_ntiles ),
        nx( nx ),
        max_part( max_part ) {};
};

/**
 * @brief Class for particle data
 * 
 */
 class Particles : public ParticleData {

    protected:

    /// @brief Local grid size
    uint2 local_nx;

    /// @brief Global grid size
    uint2 global_nx;

    /// @brief Global periodic boundaries (x,y)
    int2 periodic;

    /// @brief Local node boundary type
    part::bnd_type local_bnd;

    /// @brief Outgoing particle data messages
    ParticleMessage send;

    /// @brief Incoming particle data messages
    ParticleMessage recv;

    private:

    /// @brief Device variable for returning uint32 values
    device::Var<uint32_t> dev_tmp_uint32;

    public:

    /// @brief Parallel partition
    Partition & parallel;

    /**
     * @brief Construct a new Particles object
     * 
     * @param global_ntiles     Global number of tiles
     * @param nx                Individual tile grid size
     * @param max_part          Maximum number of particles
     */
    Particles( const uint2 global_ntiles, const uint2 nx, const uint32_t max_part, Partition & parallel ) :
        ParticleData( global_ntiles, nx, max_part ),
        send( parallel ), recv( parallel ),
        parallel( parallel )
    {

        // Get local number of tiles and position on tile grid
        parallel.grid_local( global_ntiles, ntiles, tile_off );

        // Global grid size
        global_nx = global_ntiles * nx;

        // Local grid size
        local_nx = ntiles * nx;
        
        ///@brief Total number of local tiles including edge tiles
        const size_t bsize = part::all_tiles( ntiles );

        // Tile information
        np = device::malloc<int>( bsize );
        offset = device::malloc<int>( bsize );

        // Initially empty
        device::zero( np, bsize );
        device::zero( offset, bsize );

        // Particle data
        ix = device::malloc<int2>( max_part );
        x = device::malloc<float2>( max_part );
        u = device::malloc<float3>( max_part );

        // Default global periodic boundaries to parallel partition type
        periodic = parallel.periodic;

        // Set local bnd values
        update_local_bnd();
    }

    ~Particles() {
        device::free( u );
        device::free( x );
        device::free( ix );

        device::free( offset );
        device::free( np );
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
    part::bnd_type get_local_bnd() {
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
            std::cerr << "Particles::set_periodic() - Attempting to set ";
            std::cerr << "parallel boundaries on non-parallel comm direction\n";
            exit(1);
        }

        // Check y direction
        if ( ( new_periodic.y ) && 
             ( (! parallel.periodic.y ) && ( parallel.dims.y > 1 )) ) {
            std::cerr << "Particles::set_periodic() - Attempting to set ";
            std::cerr << "parallel boundaries on non-parallel comm direction\n";
            exit(1);
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
    auto get_periodic( ) { return periodic; }

    /**
     * @brief Get the local grid size
     * 
     * @return auto
     */
    auto get_local_nx() { return local_nx; }

    /**
     * @brief Sets the number of particles per tile to 0
     * 
     */
    void zero_np() {
        device::zero( np, part::all_tiles( ntiles ) );
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
            device::free( u );
            device::free( x );
            device::free( ix );

            // Grow in multiples 64k blocks
            max_part = roundup<65536>(new_max);

            ix = device::malloc<int2>  ( max_part );
            x  = device::malloc<float2>( max_part );
            u  = device::malloc<float3>( max_part );
        }
    }

    /**
     * @brief Swaps buffers between 2 particle objects
     * 
     * @param a     Object a
     * @param b     Object b
     */
    friend void swap_buffers( Particles & a, Particles & b ) {
        swap( a.ix, b.ix );
        swap( a.x,  b.x );
        swap( a.u,  b.u );

        auto tmp_max_part = b.max_part;
        b.max_part = a.max_part;
        a.max_part = tmp_max_part;

        swap( a.np,     b.np );
        swap( a.offset, b.offset );
    }

    /**
     * @brief Gets (node) local number of particles
     * 
     * @return uint32_t 
     */
    uint32_t np_local(); // implemented in particles.cpp

    /**
     * @brief Gets global number of particles
     * 
     * @param all           Return result on all parallel nodes (defaults to false)
     * @return uint64_t     Global number of particles
     */
    uint64_t np_global( bool all = false ) {

        uint64_t local = np_local();

        if ( parallel.get_size() > 1 ) {
            if ( all ) {
                uint64_t global;
                parallel.allreduce( &local, &global, 1, mpi::sum );
                return global;
            } else {
                parallel.reduce( &local, 1, mpi::sum );
                return local;
            }
        }

        return local;
    }

    /**
     * @brief Gets maximum number of particles in a single tile
     * 
     * @return uint32_t 
     */
    uint32_t np_max_tile(); // implemented in particles.cpp

    /**
     * @brief Gets minimum number of particles in a single tile
     * 
     * @return uint32_t 
     */
    uint32_t np_min_tile(); // implemented in particles.cpp
    
    /**
     * @brief Returns local grid range
     * 
     * @return bnd<uint32_t> 
     */
    bnd<uint32_t> local_range() { 
        bnd<uint32_t> range;
        range.x = pair<uint32_t>( 0, local_nx.x - 1 );
        range.y = pair<uint32_t>( 0, local_nx.y - 1 );

        return range;
    };

    /**
     * @brief Gather data from a specific particle quantity
     * 
     * @param quant     Quantity to gather
     * @param d_data    Output device data buffer, assumed to have size >= np
     */
    void gather( part::quant quant, float * const __restrict__ d_data );

    /**
     * @brief Gather data from a specific particle quantity, scaling values
     * 
     * @note Data (val) will be returned as `scale.x * val + scale.y`
     * 
     * @param quant     Quantity to gather
     * @param d_data    Output device data buffer, assumed to have size >= np
     * @param scale     Scale factor for data
     */
    void gather( part::quant quant, const float2 scale, float * const __restrict__ d_data );
    
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
     *          temporary buffers are created and removed each time the
     *          function is called.
     * 
     * @param extra     (optional) Additional space to add to each tile. Leaves
     *                  room for particles to be injected later.
     */
    void tile_sort( Partition & parallel, const int * __restrict__ extra = nullptr ){
        // Create temporary buffers
        Particles    tmp( global_ntiles, nx, max_part, parallel );
        ParticleSort sort( ntiles, max_part, parallel );
        
        // Call sort routine
        tile_sort( tmp, sort, extra );
    };

    /**
     * @brief Moves particles to the correct tiles
     * 
     * @note Particles are only expected to have moved no more than 1 tile
     *       in each direction
     * 
     * @param tmp       Temporary particle buffer
     * @param sort      Temporary sort index 
     * @param extra     (optional) Additional space to add to each tile. Leaves
     *                  room for particles to be injected later.
     */
    void tile_sort( Particles &tmp, ParticleSort &sort, 
                    const int * __restrict__ extra = nullptr ); 

    /**
     * @brief Save particle data to disk
     * 
     * @param quants    Quantities to save
     * @param metadata  Particle metadata (name, labels, units, etc.). Information is used to set file name
     * @param iter      Iteration metadata
     * @param path      Path where to save the file
     */
    void save( const part::quant quants[], zdf::part_info &metadata, zdf::iteration &iter, std::string path );


    /**
     * @brief Size (in bytes) of a single particle
     * 
     * @return size_t 
     */
    size_t constexpr particle_size() {
        return sizeof(int2) + sizeof(float2) + sizeof(float3);
    };


    /**
     * @brief Prepare particle receive buffers and start receive
     * 
     * @param sort      Temporary sort index
     * @param recv      Receive message object
     */
    void irecv_msg( ParticleSortData &sort, ParticleMessage &recv );

    /**
     * @brief Pack particles moving out of the node into a message buffer and start send
     * 
     * @param tmp       Temporary buffer holding particles moving away from tiles
     * @param sort      Temporary sort index
     * @param send      Send message object
     */
    void isend_msg( Particles &tmp, ParticleSortData &sort, ParticleMessage &send );

    /**
     * @brief Unpack received particle data into main particle data buffer
     * 
     * @param sort      Temporary sort index
     * @param recv      Receive message object
     */
    void unpack_msg( ParticleSortData &sort, ParticleMessage &recv );

    /**
     * @brief Print information on the number of particles per tile
     * 
     * @warning Used for debug purposes only
     * 
     * @param msg   (optional) Message to print before printing particle information
     */
    void info_np( std::string msg = "" ) {
        
        parallel.barrier();

        int * h_np = host::malloc<int>( ntiles.x * ntiles.y );
        device::memcpy_tohost( h_np, np, ntiles.x * ntiles.y );


        if ( ! msg.empty() && ( parallel.get_rank() == 0)) {
            std::cout << "-------------[info]> " << msg << '\n';
        }

        for( int k = 0; k < parallel.get_size() ; k++ ) {
            if ( k == parallel.get_rank() ) {
                std::cout << '\n';
                mpi::cout << "#particles per tile:\n";

                for( unsigned int j = 0; j < ntiles.y; j++ ) {
                    mpi::cout << j << ':';
                    for( unsigned int i = 0; i < ntiles.x; i++ ) {
                        int tid = j * ntiles.x + i;
                        mpi::cout << " " << h_np[tid];
                    }
                    mpi::cout << '\n';
                }

                mpi::cout << "#particles total: " << np_local() << '\n';
            }
            parallel.barrier();
        }
    }
};

#endif
