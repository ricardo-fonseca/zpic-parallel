#ifndef GRID_H_
#define GRID_H_

#include "parallel.h"

#include "vec_types.h"
#include "bnd.h"
#include "zdf-cpp.h"

#include <iostream>

namespace source {
    enum tag { lower = 0, upper = 1 };
}

namespace dest {
    enum tag { upper = 0, lower = 1 };
}

/**
 * @brief 
 * 
 */
template <class T>
class grid {
    protected:

    /// @brief Local number of tiles
    uint2 ntiles;

    /// @brief Offset of local tiles in global tile grid
    uint2 tile_off;

    /// @brief Consider local boundaries periodic
    int2 periodic;

    /// @brief Local grid size
    uint2 gnx;

    /// @brief Buffers for sending messages
    pair< Message<T>* > msg_send;

    /// @brief Buffers for receiving messages
    pair< Message<T>* > msg_recv;

    /**
     * @brief Set the local node information. This information will (may) be
     * different for each parallel node
     * 
     * @note Global periodic information is taken from the parallel partition
     * 
     */
    void initialize( ) {

        // Get local number of tiles and position
        part.grid_local( global_ntiles, ntiles, tile_off );

        // Get local grid size
        gnx = ntiles * nx;

        // Get local periodic flag
        periodic.x = part.periodic.x && (part.dims.x == 1);
        periodic.y = part.periodic.y && (part.dims.y == 1);

        // Allocate main data buffer
        d_buffer = memory::malloc<T>( buffer_size() );

        // Get maximum message size
        int max_msg_size = max(
            ( ntiles.y * ext_nx.y ) * max( gc.x.lower, gc.x.upper ),
            max( gc.y.lower, gc.y.upper ) * ( ntiles.x * ext_nx.x )
        );

        // Allocate message buffers
        msg_recv.lower = new Message<T>( max_msg_size, part.get_comm() );
        msg_recv.upper = new Message<T>( max_msg_size, part.get_comm() );
        msg_send.lower = new Message<T>( max_msg_size, part.get_comm() );
        msg_send.upper = new Message<T>( max_msg_size, part.get_comm() );

    }

    private:

    /**
     * @brief Validate grid / parallel parameters. The execution will stop if
     * errors are found.
     * 
     */
    void validate_parameters() {
        // Grid parameters
        if ( global_ntiles.x == 0 || global_ntiles.y == 0 ) {
            std::cerr << "Invalid number of tiles " << global_ntiles << '\n';
            mpi::abort(1);
        }

        if ( nx.x == 0 || nx.y == 0 ) {
            std::cerr << "Invalid tiles size" << nx << '\n';
            mpi::abort(1);
        }

        // Parallel partition
        if ( (unsigned) part.dims.x > global_ntiles.x ) {
            std::cerr << "Number of parallel nodes along x (" ;
            std::cerr << part.dims.x << ") is larger than number of tiles along x(";
            std::cerr << global_ntiles.x << '\n';
            mpi::abort(1);
        }

        if ( (unsigned) part.dims.y > global_ntiles.y ) {
            std::cerr << "Number of parallel nodes along y (" ;
            std::cerr << part.dims.y << ") is larger than number of tiles along y(";
            std::cerr << global_ntiles.y << '\n';
            mpi::abort(1);
        }
    }

    public:

    /// @brief Parallel partition
    Partition & part;

    /// @brief Data buffer
    T * d_buffer;

    /// @brief Global number of tiles
    const uint2 global_ntiles;

    /// @brief Tile grid size
    const uint2 nx;
    
    /// @brief Tile guard cells
    const bnd<unsigned int> gc;
    
    /// @brief Tile grize including guard cells
    const uint2 ext_nx;

    /// @brief Local offset in cells between lower tile corner and position (0,0)
    const unsigned int offset;

    /// @brief Tile volume (may be larger than product of cells for alignment)
    const unsigned int tile_vol;

    /// @brief Object name
    std::string name;

    /**
     * @brief Construct a new grid object
     * 
     * @param global_ntiles     Global number of tiles
     * @param nx                Individual tile size
     * @param gc                Number of guard cells
     * @param part              Parallel partition
     */
    grid( uint2 const global_ntiles, uint2 const nx, bnd<unsigned int> const gc, Partition & part ):
        part( part ),
        d_buffer( nullptr ), 
        global_ntiles( global_ntiles ),
        nx( nx ),
        gc(gc),
        ext_nx( make_uint2( gc.x.lower + nx.x + gc.x.upper,
                            gc.y.lower + nx.y + gc.y.upper )),
        offset( gc.y.lower * ext_nx.x + gc.x.lower ),
        tile_vol( roundup4( ext_nx.x * ext_nx.y ) ),
        name( "grid" )
    {
        // Validate parameters
        validate_parameters();

        // Set local information (ntiles, tile_start and local_periodic)
        initialize();
    };

    /**
     * @brief Construct a new grid object
     * 
     * @note: The number of guard cells is set to 0
     * 
     * @param global_ntiles     Global number of tiles
     * @param nx                Individual tile size
     * @param part              Parallel partition
     */
    grid( uint2 const global_ntiles, uint2 const nx, Partition & part ):
        part( part ),
        d_buffer( nullptr ),
        global_ntiles( global_ntiles ),
        nx( nx ),
        gc( 0 ),
        ext_nx( make_uint2( nx.x, nx.y )),
        offset( 0 ),
        tile_vol( roundup4( nx.x * nx.y )),
        name( "grid" )
    {
        // Validate parameters
        validate_parameters();

        // Set local information (ntiles, tile_start and periodic)
        initialize();
    };

    /**
     * @brief Get the local number of tiles
     * 
     * @return int2 
     */
    uint2 get_ntiles() { return ntiles; };

    /**
     * @brief Returns the local tile offset in the global MPI tile grid
     * 
     * @return uint2 
     */
    uint2 get_tile_off() { return tile_off; };

    /**
     * @brief grid destructor
     * 
     */
    ~grid(){
        delete msg_recv.lower;
        delete msg_recv.upper;
        delete msg_send.lower;
        delete msg_send.upper;

        memory::free( d_buffer );
    };

    /**
     * @brief Stream extraction
     * 
     * @param os 
     * @param obj 
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, const grid<T>& obj) {
        os << obj.name << "{";
        os << "(" << obj.ntiles.x << " x " << obj.ntiles.y << " tiles)";
        os << ", (" << obj.nx.x << " x " << obj.nx.y << " points/tile)";
        os << "}";
        return os;
    }


    /**
     * @brief Buffer size
     * 
     * @return total size of data buffers
     */
    const std::size_t buffer_size() {
        return (static_cast <std::size_t> (tile_vol)) * ( ntiles.x * ntiles.y ) ;
    };

    /**
     * @brief zero device data on a grid grid
     * 
     * @return int       Returns 0 on success, -1 on error
     */
    int zero() {
        memory::zero( d_buffer, buffer_size() );
        return 0;
    };

    /**
     * @brief Sets data to a constant value
     * 
     * @param val       Value
     */
    void set( T const & val ){

        const size_t size = buffer_size( );
        for( size_t i = 0; i < size; i++ ) d_buffer[i] = val;
    };

    /**
     * @brief Scalar assigment
     * 
     * @param val       Value
     * @return T        Returns the same value
     */
    T operator=( T const val ) {
        set( val );
        return val;
    }

    /**
     * @brief Adds another grid object on top of local object
     * 
     * @param rhs         Other object to add
     * @return grid&    Reference to local object
     */
    void add( const grid<T> &rhs ) {
        size_t const size = buffer_size( );

        for( size_t i = 0; i < size; i++ ) d_buffer[i] += rhs.d_buffer[i];
    };

    /**
     * @brief Operator +=
     * 
     * @param rhs           Other grid to add
     * @return grid<T>& 
     */
    grid<T>& operator+=(const grid<T>& rhs) {
        add( rhs );
        return *this;
    }

    /**
     * @brief Gather local grid into a contiguous grid
     * 
     * Used mostly for diagnostic output
     * 
     * @param out               Output buffer
     * @return unsigned int     Total number of cells
     */
    unsigned int gather( T * const __restrict__ out ) {

        // Loop over tiles
        for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
            for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {

                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol + offset;

                auto * const __restrict__ tile_data = & d_buffer[ tile_off ];

                // Loop inside tile
                for( unsigned iy = 0; iy < nx.y; iy ++ ) {
                    for( unsigned ix = 0; ix < nx.x; ix ++ ) {
                        auto const gix = tile_idx.x * nx.x + ix;
                        auto const giy = tile_idx.y * nx.y + iy;

                        auto const out_idx = giy * gnx.x + gix;

                        out[ out_idx ] = tile_data[ iy * ext_nx.x + ix ];
                    }
                }
            }
        }

        return gnx.x * gnx.y;
    }

#ifdef _OPENMP

    /**
     * @brief Copies edge values to X neighboring guard cells
     * 
     */
    void local_copy_to_gc_x() {

        // Loop over tiles
        #pragma omp parallel for
        for( unsigned tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

            const auto tile_idx = make_int2( tid % ntiles.x, tid / ntiles.x );
            const auto tile_off = tid * tile_vol;

            auto * __restrict__ local = & d_buffer[ tile_off ];

            {   // Copy from lower neighbour
                int neighbor_tx = tile_idx.x;
                neighbor_tx -= 1;
                if ( periodic.x && neighbor_tx < 0 ) neighbor_tx += ntiles.x;

                if ( neighbor_tx >= 0 ) {
                    auto * __restrict__ x_lower = d_buffer + (tile_idx.y * ntiles.x + neighbor_tx) * tile_vol;
                    for( unsigned j = 0; j < ext_nx.y; j++ ) {
                        for( unsigned i = 0; i < gc.x.lower; i++ ) {
                            local[ i + j * ext_nx.x ] = x_lower[ nx.x + i + j * ext_nx.x ];
                        }
                    }
                }
            }

            {   // Copy from upper neighbour
                int neighbor_tx = tile_idx.x;
                neighbor_tx += 1;
                if ( periodic.x && neighbor_tx >= static_cast<int>(ntiles.x) ) neighbor_tx -= ntiles.x;

                if ( neighbor_tx < static_cast<int>(ntiles.x) ) {
                    auto * __restrict__ x_upper = d_buffer + (tile_idx.y * ntiles.x + neighbor_tx) * tile_vol;
                    for( unsigned j = 0; j < ext_nx.y; j++ ) {
                        for( unsigned i = 0; i < gc.x.upper; i++ ) {
                            local[ gc.x.lower + nx.x + i + j * ext_nx.x ] = x_upper[ gc.x.lower + i + j * ext_nx.x ];
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief Copies edge values to Y neighboring guard cells
     * 
     */
    void local_copy_to_gc_y() {

        #pragma omp parallel for
        for( unsigned tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

            const auto tile_idx = make_int2( tid % ntiles.x, tid / ntiles.x );
            const auto tile_off = tid * tile_vol;

            auto * __restrict__ local = d_buffer + tile_off;
            
            {   // Copy from lower neighbour
                int neighbor_ty = tile_idx.y;
                neighbor_ty -= 1;
                if ( periodic.y && neighbor_ty < 0 ) neighbor_ty += ntiles.y;

                if ( neighbor_ty >= 0 ) {
                    auto * __restrict__ y_lower = d_buffer + (neighbor_ty * ntiles.x + tile_idx.x) * tile_vol;
                    for( unsigned j = 0; j < gc.y.lower; j++ ) {
                        for( unsigned i = 0; i < ext_nx.x; i++ ) {
                            local[ i + j * ext_nx.x ] = y_lower[ i + ( nx.y + j ) * ext_nx.x ];
                        }
                    }
                }
            }

            {   // Copy from upper neighbour
                int neighbor_ty = tile_idx.y;
                neighbor_ty += 1;
                if ( periodic.y && neighbor_ty >= static_cast<int>(ntiles.y) ) neighbor_ty -= ntiles.y;

                if ( neighbor_ty < static_cast<int>(ntiles.y) ) {
                    auto * __restrict__ y_upper = d_buffer + (neighbor_ty * ntiles.x + tile_idx.x) * tile_vol;
                    for( unsigned j = 0; j < gc.y.upper; j++ ) {
                        for( unsigned i = 0; i < ext_nx.x; i++ ) {
                            local[ i + ( gc.y.lower + nx.y + j ) * ext_nx.x ] = y_upper[ i + ( gc.y.lower + j ) * ext_nx.x ];
                        }
                    }
                }
            }
        }
    }

#else

    /**
     * @brief Copies edge values to X neighboring guard cells
     * 
     */
    void local_copy_to_gc_x() {

        // Loop over tiles
        for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
            for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {

                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];

                {   // Copy from lower neighbour
                    int neighbor_tx = tile_idx.x;
                    neighbor_tx -= 1;
                    if ( periodic.x && neighbor_tx < 0 ) neighbor_tx += ntiles.x;

                    if ( neighbor_tx >= 0 ) {
                        auto * __restrict__ x_lower = d_buffer + (tile_idx.y * ntiles.x + neighbor_tx) * tile_vol;
                        for( unsigned j = 0; j < ext_nx.y; j++ ) {
                            for( unsigned i = 0; i < gc.x.lower; i++ ) {
                                local[ i + j * ext_nx.x ] = x_lower[ nx.x + i + j * ext_nx.x ];
                            }
                        }
                    }
                }

                {   // Copy from upper neighbour
                    int neighbor_tx = tile_idx.x;
                    neighbor_tx += 1;
                    if ( periodic.x && neighbor_tx >= static_cast<int>(ntiles.x) ) neighbor_tx -= ntiles.x;

                    if ( neighbor_tx < static_cast<int>(ntiles.x) ) {
                        auto * __restrict__ x_upper = d_buffer + (tile_idx.y * ntiles.x + neighbor_tx) * tile_vol;
                        for( unsigned j = 0; j < ext_nx.y; j++ ) {
                            for( unsigned i = 0; i < gc.x.upper; i++ ) {
                                local[ gc.x.lower + nx.x + i + j * ext_nx.x ] = x_upper[ gc.x.lower + i + j * ext_nx.x ];
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief Copies edge values to Y neighboring guard cells
     * 
     */
    void local_copy_to_gc_y() {

        // Loop over tiles
        for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
            for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {

                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = d_buffer + tile_off;
                
                {   // Copy from lower neighbour
                    int neighbor_ty = tile_idx.y;
                    neighbor_ty -= 1;
                    if ( periodic.y && neighbor_ty < 0 ) neighbor_ty += ntiles.y;

                    if ( neighbor_ty >= 0 ) {
                        auto * __restrict__ y_lower = d_buffer + (neighbor_ty * ntiles.x + tile_idx.x) * tile_vol;
                        for( unsigned j = 0; j < gc.y.lower; j++ ) {
                            for( unsigned i = 0; i < ext_nx.x; i++ ) {
                                local[ i + j * ext_nx.x ] = y_lower[ i + ( nx.y + j ) * ext_nx.x ];
                            }
                        }
                    }
                }

                {   // Copy from upper neighbour
                    int neighbor_ty = tile_idx.y;
                    neighbor_ty += 1;
                    if ( periodic.y && neighbor_ty >= static_cast<int>(ntiles.y) ) neighbor_ty -= ntiles.y;

                    if ( neighbor_ty < static_cast<int>(ntiles.y) ) {
                        auto * __restrict__ y_upper = d_buffer + (neighbor_ty * ntiles.x + tile_idx.x) * tile_vol;
                        for( unsigned j = 0; j < gc.y.upper; j++ ) {
                            for( unsigned i = 0; i < ext_nx.x; i++ ) {
                                local[ i + ( gc.y.lower + nx.y + j ) * ext_nx.x ] = y_upper[ i + ( gc.y.lower + j ) * ext_nx.x ];
                            }
                        }
                    }
                }
            }
        }
    }

#endif

    /**
     * @brief Copies x values to neighboring guard cells, including cells on 
     *        other parallel nodes
     * 
     */
    void copy_to_gc_x() {

        // Get x neighbors
        int lnode = part.get_neighbor(-1, 0 );
        int unode = part.get_neighbor(+1, 0 );

        // Disable messages if local periodic
        if ( periodic.x ) lnode = unode = -1;

        // Post message receives
        if ( lnode >= 0 ) msg_recv.lower->irecv( lnode, source::lower );
        if ( unode >= 0 ) msg_recv.upper->irecv( unode, source::upper );

        // Send message - lower neighbor
        if ( lnode >= 0 ) {
            unsigned int tx = 0;
            for( unsigned ty = 0; ty < ntiles.y; ty++ ) {
                
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_send.lower-> buffer [ ty * ext_nx.y * gc.x.upper ];


                for( unsigned j = 0; j < ext_nx.y; j++ ) {
                    for( unsigned i = 0; i < gc.x.upper; i++ ) {
                        msg[ j * gc.x.upper + i ] = local[ j * ext_nx.x + gc.x.lower + i ];
                    }
                }
            }

            int msg_size = ( ext_nx.y * ntiles.y ) * gc.x.upper;
            msg_send.lower->isend( msg_size, lnode, dest::lower );
        }

        // Send message - upper neighbor
        if ( unode >= 0 ) {
            unsigned int tx = ntiles.x - 1;
            for( unsigned ty = 0; ty < ntiles.y; ty++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_send.upper-> buffer[ ty * ext_nx.y * gc.x.lower ];

                for( unsigned j = 0; j < ext_nx.y; j++ ) {
                    for( unsigned i = 0; i < gc.x.lower; i++ ) {
                        msg[ j * gc.x.lower + i ] = local[ j * ext_nx.x + nx.x + i ];
                    }
                }
            }

            int msg_size = ( ext_nx.y * ntiles.y ) * gc.x.lower;
            msg_send.upper->isend( msg_size, unode, dest::upper );
        }

        // Process local tiles
        local_copy_to_gc_x();

        // Receive message - lower neighbor
        if ( lnode >= 0 ) {
            msg_recv.lower-> wait();

            unsigned int tx = 0;
            for( unsigned ty = 0; ty < ntiles.y; ty++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_recv.lower-> buffer[ ty * ext_nx.y * gc.x.lower ];

                for( unsigned j = 0; j < ext_nx.y; j++ ) {
                    for( unsigned i = 0; i < gc.x.lower; i++ ) {
                        local[ j * ext_nx.x + i ] = msg[ j * gc.x.lower + i ];
                    }
                }
            }
        }

        // Receive message - upper neighbor
        if ( unode >= 0 ) {
            msg_recv.upper-> wait();

            unsigned int tx = ntiles.x-1;
            for( unsigned ty = 0; ty < ntiles.y; ty++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_recv.upper -> buffer[ ty * ext_nx.y * gc.x.upper ];

                for( unsigned j = 0; j < ext_nx.y; j++ ) {
                    for( unsigned i = 0; i < gc.x.upper; i++ ) {
                        local[ j * ext_nx.x + gc.x.lower + nx.x + i ] = msg[ j * gc.x.upper + i ];
                    }
                }
            }
        }

        // Wait for send messages to complete
        if ( lnode >= 0 ) msg_send.lower->wait( );
        if ( unode >= 0 ) msg_send.upper->wait( );
    }

    /**
     * @brief Copies y values to neighboring guard cells, including cells on 
     *        other parallel nodes
     * 
     */
    void copy_to_gc_y() {

        // Get y neighbors
        int lnode = part.get_neighbor(0, -1);
        int unode = part.get_neighbor(0, +1);

        // Disable messages if local periodic
        if ( periodic.y ) lnode = unode = -1;
        
        // Post message receives
        if ( lnode >= 0 ) msg_recv.lower->irecv( lnode, source::lower );
        if ( unode >= 0 ) msg_recv.upper->irecv( unode, source::upper );

        // Post message sends
        if ( lnode >= 0 ) {
            unsigned int ty = 0;
            for( unsigned tx = 0; tx < ntiles.x; tx++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_send.lower-> buffer[ tx * ext_nx.x * gc.y.upper ];
                
                for( unsigned j = 0; j < gc.y.upper; j++ ) {
                    for( unsigned i = 0; i < ext_nx.x; i++ ) {
                        msg[ j * ext_nx.x + i ] = local[ ( gc.y.lower + j ) * ext_nx.x + i ];
                    }
                }
            }

            int msg_size =  gc.y.upper * ( ntiles.x * ext_nx.x );
            msg_send.lower->isend( msg_size, lnode, dest::lower );
        }

        if ( unode >= 0 ) {
            unsigned int ty = ntiles.y-1;
            for( unsigned tx = 0; tx < ntiles.x; tx++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_send.upper -> buffer[ tx * ext_nx.x * gc.y.lower ];

                for( unsigned j = 0; j < gc.y.lower; j++ ) {
                    for( unsigned i = 0; i < ext_nx.x; i++ ) {
                        msg[ j * ext_nx.x + i ] = local[ ( nx.y + j ) * ext_nx.x + i ];
                    }
                }
            }

            int msg_size = gc.y.lower * ( ntiles.x * ext_nx.x );
            msg_send.upper -> isend( msg_size, unode, dest::upper );
        }

        // Process local tiles
        local_copy_to_gc_y();

        // Wait for receive messages to complete and copy data
        if ( lnode >= 0 ) {
            msg_recv.lower-> wait();

            unsigned int ty = 0;
            for( unsigned tx = 0; tx < ntiles.x; tx++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_recv.lower-> buffer[ tx * ext_nx.x * gc.y.lower ];

                for( unsigned j = 0; j < gc.y.lower; j++ ) {
                    for( unsigned i = 0; i < ext_nx.x; i++ ) {
                        local[ j * ext_nx.x + i ] =  msg[ j * ext_nx.x + i ];
                    }
                }
            }
        }

        if ( unode >= 0 ) {
            msg_recv.upper-> wait();

            unsigned int ty = ntiles.y - 1;
            for( unsigned tx = 0; tx < ntiles.x; tx++ ) {
                
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_recv.upper-> buffer[ tx * ext_nx.x * gc.y.upper ];

                for( unsigned j = 0; j < gc.y.upper; j++ ) {
                    for( unsigned i = 0; i < ext_nx.x; i++ ) {
                        local[ ( gc.y.lower + nx.y + j ) * ext_nx.x + i ] =  msg[ j * ext_nx.x + i ];
                    }
                }
            }
        }

        // Wait for send messages to complete
        if ( lnode >= 0 ) msg_send.lower->wait( );
        if ( unode >= 0 ) msg_send.upper->wait( );
    }

    /**
     * @brief Copies edge values to neighboring guard cells, including other
     *        parallel nodes
     * 
     */
    void copy_to_gc()  {

        // Copy along x direction
        copy_to_gc_x();

        // Copy along y direction
        copy_to_gc_y();

    };

#ifdef _OPENMP

    /**
     * @brief Adds values from neighboring x guard cells to local data
     * 
     */
    void local_add_from_gc_x() {
        // Add along x direction

        // Loop over tiles
        #pragma omp parallel for
        for( unsigned tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

            const auto tile_idx = make_int2( tid % ntiles.x, tid / ntiles.x );
            const auto tile_off = tid * tile_vol;

            auto * __restrict__ local = d_buffer + tile_off;
            
            {   // Add from lower neighbour
                int neighbor_tx = tile_idx.x;
                neighbor_tx -= 1;
                if ( periodic.x && neighbor_tx < 0 ) neighbor_tx += ntiles.x;

                if ( neighbor_tx >= 0 ) {
                    T * __restrict__ x_lower = d_buffer + (tile_idx.y * ntiles.x + neighbor_tx) * tile_vol;
                    for( unsigned j = 0; j < ext_nx.y; j++ ) {
                        for( unsigned i = 0; i < gc.x.upper; i++ ) {
                            local[ gc.x.lower + i + j * ext_nx.x ] += x_lower[ gc.x.lower + nx.x + i + j * ext_nx.x ];
                        }
                    }
                }
            }

            {   // Add from upper neighbour
                int neighbor_tx = tile_idx.x;
                neighbor_tx += 1;
                if ( periodic.x && neighbor_tx >= static_cast<int>(ntiles.x) ) neighbor_tx -= ntiles.x;

                if ( neighbor_tx < static_cast<int>(ntiles.x) ) {
                    auto * __restrict__ x_upper = d_buffer + (tile_idx.y * ntiles.x + neighbor_tx) * tile_vol;
                    for( unsigned j = 0; j < ext_nx.y; j++ ) {
                        for( unsigned i = 0; i < gc.x.lower; i++ ) {
                            local[ nx.x + i + j * ext_nx.x ] += x_upper[ i + j * ext_nx.x ];
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief Adds values from neighboring y guard cells to local data
     * 
     */
    void local_add_from_gc_y(){

        // Add along y direction

        // Loop over tiles
        #pragma omp parallel for
        for( unsigned tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

            const auto tile_idx = make_int2( tid % ntiles.x, tid / ntiles.x );
            const auto tile_off = tid * tile_vol;

            auto * __restrict__ local = d_buffer + tile_off;
            
            {   // Add from lower neighbour
                int neighbor_ty = tile_idx.y;
                neighbor_ty -= 1;
                if ( periodic.y && neighbor_ty < 0 ) neighbor_ty += ntiles.y;

                if ( neighbor_ty >= 0 ) {
                    auto * __restrict__ y_lower = d_buffer + (neighbor_ty * ntiles.x + tile_idx.x) * tile_vol;
                    for( unsigned j = 0; j < gc.y.upper; j++ ) {
                        for( unsigned i = 0; i < ext_nx.x; i++ ) {
                            local[ i + ( gc.y.lower + j ) * ext_nx.x ] += y_lower[ i + ( gc.y.lower + nx.y + j ) * ext_nx.x ];
                        }
                    }
                }
            }

            {   // Add from upper neighbour
                int neighbor_ty = tile_idx.y;
                neighbor_ty += 1;
                if ( periodic.y && neighbor_ty >= static_cast<int>(ntiles.y) ) neighbor_ty -= ntiles.y;

                if ( neighbor_ty < static_cast<int>(ntiles.y) ) {
                    auto * __restrict__ y_upper = d_buffer + (neighbor_ty * ntiles.x + tile_idx.x) * tile_vol;
                    for( unsigned j = 0; j < gc.y.lower; j++ ) {
                        for( unsigned i = 0; i < ext_nx.x; i++ ) {
                            local[ i + ( nx.y + j ) * ext_nx.x ] += y_upper[ i + j * ext_nx.x ];
                        }
                    }
                }
            }
        }
    };


#else

    /**
     * @brief Adds values from neighboring x guard cells to local data
     * 
     */
    void local_add_from_gc_x() {
        // Loop over tiles
        for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
            for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {

                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = d_buffer + tile_off;
                
                {   // Add from lower neighbour
                    int neighbor_tx = tile_idx.x;
                    neighbor_tx -= 1;
                    if ( periodic.x && neighbor_tx < 0 ) neighbor_tx += ntiles.x;

                    if ( neighbor_tx >= 0 ) {
                        T * __restrict__ x_lower = d_buffer + (tile_idx.y * ntiles.x + neighbor_tx) * tile_vol;
                        for( unsigned j = 0; j < ext_nx.y; j++ ) {
                            for( unsigned i = 0; i < gc.x.upper; i++ ) {
                                local[ gc.x.lower + i + j * ext_nx.x ] += x_lower[ gc.x.lower + nx.x + i + j * ext_nx.x ];
                            }
                        }
                    }
                }

                {   // Add from upper neighbour
                    int neighbor_tx = tile_idx.x;
                    neighbor_tx += 1;
                    if ( periodic.x && neighbor_tx >= static_cast<int>(ntiles.x) ) neighbor_tx -= ntiles.x;

                    if ( neighbor_tx < static_cast<int>(ntiles.x) ) {
                        auto * __restrict__ x_upper = d_buffer + (tile_idx.y * ntiles.x + neighbor_tx) * tile_vol;
                        for( unsigned j = 0; j < ext_nx.y; j++ ) {
                            for( unsigned i = 0; i < gc.x.lower; i++ ) {
                                local[ nx.x + i + j * ext_nx.x ] += x_upper[ i + j * ext_nx.x ];
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief Adds values from neighboring x guard cells to local data
     * 
     */
    void local_add_from_gc_y() {
        for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
            for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {

                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = d_buffer + tile_off;
                
                {   // Add from lower neighbour
                    int neighbor_ty = tile_idx.y;
                    neighbor_ty -= 1;
                    if ( periodic.y && neighbor_ty < 0 ) neighbor_ty += ntiles.y;

                    if ( neighbor_ty >= 0 ) {
                        auto * __restrict__ y_lower = d_buffer + (neighbor_ty * ntiles.x + tile_idx.x) * tile_vol;
                        for( unsigned j = 0; j < gc.y.upper; j++ ) {
                            for( unsigned i = 0; i < ext_nx.x; i++ ) {
                                local[ i + ( gc.y.lower + j ) * ext_nx.x ] += y_lower[ i + ( gc.y.lower + nx.y + j ) * ext_nx.x ];
                            }
                        }
                    }
                }

                {   // Add from upper neighbour
                    int neighbor_ty = tile_idx.y;
                    neighbor_ty += 1;
                    if ( periodic.y && neighbor_ty >= static_cast<int>(ntiles.y) ) neighbor_ty -= ntiles.y;

                    if ( neighbor_ty < static_cast<int>(ntiles.y) ) {
                        auto * __restrict__ y_upper = d_buffer + (neighbor_ty * ntiles.x + tile_idx.x) * tile_vol;
                        for( unsigned j = 0; j < gc.y.lower; j++ ) {
                            for( unsigned i = 0; i < ext_nx.x; i++ ) {
                                local[ i + ( nx.y + j ) * ext_nx.x ] += y_upper[ i + j * ext_nx.x ];
                            }
                        }
                    }
                }
            }
        }
    }

#endif

    void add_from_gc_x() {

        // Get x neighbors
        int lnode = part.get_neighbor(-1, 0 );
        int unode = part.get_neighbor(+1, 0 );

        // Disable messages if local periodic
        if ( periodic.x ) lnode = unode = -1;

        // Post message receives
        if ( lnode >= 0 ) msg_recv.lower->irecv( lnode, source::lower );
        if ( unode >= 0 ) msg_recv.upper->irecv( unode, source::upper );

        // Send message - lower neighbor
        if ( lnode >= 0 ) {

            unsigned int tx = 0;
            for( unsigned ty = 0; ty < ntiles.y; ty++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                
                T * __restrict__ msg = & msg_send.lower-> buffer [ ty * ext_nx.y * gc.x.lower ];

                for( unsigned j = 0; j < ext_nx.y; j++ ) {
                    for( unsigned i = 0; i < gc.x.lower; i++ ) {
                        msg[ j * gc.x.lower + i ] = local[ j * ext_nx.x + i ];
                    }
                }
            }

            int msg_size = ( ntiles.y * ext_nx.y ) * gc.x.lower;
            msg_send.lower->isend( msg_size, lnode, dest::lower );
        }

        // Send message - upper neighbor
        if ( unode >= 0 ) {

            unsigned int tx = ntiles.x - 1;
            for( unsigned ty = 0; ty < ntiles.y; ty++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_send.upper-> buffer[ ty * ext_nx.y * gc.x.upper ];

                for( unsigned j = 0; j < ext_nx.y; j++ ) {
                    for( unsigned i = 0; i < gc.x.upper; i++ ) {
                        msg[ j * gc.x.upper + i ] = local[ j * ext_nx.x + gc.x.lower + nx.x + i ];
                    }
                }
            }

            int msg_size = ( ntiles.y * ext_nx.y ) * gc.x.upper;
            msg_send.upper->isend( msg_size, unode, dest::upper );
        }

        // Process local tiles
        local_add_from_gc_x();

        // Receive message - lower neighbor
        if ( lnode >= 0 ) {
            msg_recv.lower-> wait();

            unsigned int tx = 0;
            for( unsigned ty = 0; ty < ntiles.y; ty++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_recv.lower-> buffer[ ty * ext_nx.y * gc.x.upper ];

                for( unsigned j = 0; j < ext_nx.y ; j++ ) {
                    for( unsigned i = 0; i < gc.x.upper; i++ ) {
                        local[ j * ext_nx.x + gc.x.lower + i ] += msg[ j * gc.x.upper + i ] ;
                    }
                }
            }
        }

        // Receive message - upper neighbor
        if ( unode >= 0 ) {
            msg_recv.upper-> wait();

            unsigned int tx = ntiles.x - 1;
            for( unsigned ty = 0; ty < ntiles.y; ty++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_recv.upper-> buffer[ ty * ext_nx.y * gc.x.lower ];
                
                for( unsigned j = 0; j < ext_nx.y; j++ ) {
                    for( unsigned i = 0; i < gc.x.lower; i++ ) {
                        local[ j * ext_nx.x + nx.x + i ] += msg[ j * gc.x.lower + i ] ;
                    }
                }
            }
        }

        // Wait for send messages to complete
        if ( lnode >= 0 ) msg_send.lower->wait( );
        if ( unode >= 0 ) msg_send.upper->wait( );
    }

    /**
     * @brief Adds values from neighboring y guard cells to local data,
     *        including cells from other parallel nodes
     * 
     */
    void add_from_gc_y() {
        // Get y neighbors
        int lnode = part.get_neighbor( 0, -1 );
        int unode = part.get_neighbor( 0, +1 );

        // Disable messages if local periodic
        if ( periodic.y ) lnode = unode = -1;

        // Post message receives
        if ( lnode >= 0 ) msg_recv.lower->irecv( lnode, source::lower );
        if ( unode >= 0 ) msg_recv.upper->irecv( unode, source::upper );

        // Send message - lower neighbor
        if ( lnode >= 0 ) {
            unsigned int ty = 0;
            for( unsigned tx = 0; tx < ntiles.x; tx++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_send.lower-> buffer[ tx * ( ext_nx.x * gc.y.lower ) ];

                for( unsigned j = 0; j < gc.y.lower; j++ ) {
                    for( unsigned i = 0; i < ext_nx.x; i++ ) {
                        msg[ j * ext_nx.x + i ] = local[ j * ext_nx.x + i ];
                    }
                }
            }

            int msg_size = gc.y.lower * ( ntiles.x * ext_nx.x );
            msg_send.lower->isend( msg_size, lnode, dest::lower );
        }

        // Send message - upper neighbor
        if ( unode >= 0 ) {

            unsigned int ty = ntiles.y-1;
            for( unsigned tx = 0; tx < ntiles.x; tx++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_send.upper-> buffer[ tx * gc.y.upper * ext_nx.x ];

                for( unsigned j = 0; j < gc.y.upper; j++ ) {
                    for( unsigned i = 0; i < ext_nx.x; i++ ) {
                        msg[ j * ext_nx.x + i ] = local[ ( gc.y.lower + nx.y + j ) * ext_nx.x + i ];
                    }
                }
            }

            int msg_size    = gc.y.upper * ( ntiles.x * ext_nx.x );
            msg_send.upper->isend( msg_size, unode, dest::upper );
        }

        // Process local tiles
        local_add_from_gc_y();

        // Receive message - lower neighbor
        if ( lnode >= 0 ) {
            msg_recv.lower-> wait();

            unsigned int ty = 0;
            for( unsigned tx = 0; tx < ntiles.x; tx++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_recv.lower-> buffer[ tx * gc.y.upper * ext_nx.x ];

                for( unsigned j = 0; j < gc.y.upper; j++ ) {
                    for( unsigned i = 0; i < ext_nx.x; i++ ) {
                       local[ ( gc.y.lower + j ) * ext_nx.x + i ] += msg[ j * ext_nx.x + i ];
                    }
                }
            }
        }

        // Receive message - upper neighbor
        if ( unode >= 0 ) {
            msg_recv.upper-> wait();

            unsigned int ty = ntiles.y - 1;
            for( unsigned tx = 0; tx < ntiles.x; tx++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_recv.upper -> buffer[ tx * (gc.y.lower * ext_nx.x) ];

                for( unsigned j = 0; j < gc.y.lower; j++ ) {
                    for( unsigned i = 0; i < ext_nx.x; i++ ) {
                        local[ ( nx.y + j ) * ext_nx.x + i ] +=  msg[ j * ext_nx.x + i ];
                    }
                }
            }
        }

        // Wait for send messages to complete
        if ( lnode >= 0 ) msg_send.lower->wait( );
        if ( unode >= 0 ) msg_send.upper->wait( );
    }

    /**
     * @brief Adds values from neighboring guard cells to local data, including
     *        values from other parallel nodes
     * 
     */
    void add_from_gc() {
        // Add along x direction
        add_from_gc_x();

        // Add along y direction
        add_from_gc_y();
    }

    /**
     * @brief Left shifts data for a specified amount
     * 
     * @warning This operation is only allowed if the number of upper x guard cells
     * is greater or equal to the requested shift
     * 
     * @param shift Number of cells to shift
     */
    void x_shift_left( unsigned int const shift ) {

        /*
         * Warning: This version cannot be made parallel inside tiles 
         */
        if ( gc.x.upper >= shift ) {

            const int ystride = ext_nx.x;

            // Loop over tiles
            for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {

                    const auto tile_idx = make_uint2( tx, ty );
                    const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                    const auto tile_off = tid * tile_vol ;
                    
                    auto * __restrict__ buffer = d_buffer + tile_off;
                    
                    for( unsigned iy = 0; iy < ext_nx.y; iy++ ) {
                        for( unsigned ix = 0; ix < ext_nx.x - shift; ix++ ) {
                            buffer[ ix + iy * ystride ] = buffer[ ix + shift + iy * ystride ]; 
                        }
                        for( unsigned ix = ext_nx.x - shift; ix < ext_nx.x; ix++ ) {
                            buffer[ ix + iy * ystride ] = T{0};
                        }
                    }
                }
            }

            // Copy x guard cells
            copy_to_gc_x();

        } else {
            std::cerr << "x_shift_left(), shift value too large, must be <= gc.x.upper\n";
            exit(1);
        }
    }

    /**
     * @brief Left shifts data for a specified amount
     * 
     * @warning This operation is only allowed if the number of upper x guard cells
     * is greater or equal to the requested shift
     * 
     * @param shift Number of cells to shift
     */
    void x_shift_left_mk2( unsigned int const shift ) {

        if ( gc.x.upper >= shift ) {

            const int ystride = ext_nx.x;

            // Loop over tiles
            for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {
                    
                    // On a GPU this would be on shared memory
                    T local[tile_vol];

                    const auto tile_idx = make_uint2( tx, ty );
                    const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                    const auto tile_off = tid * tile_vol ;
                    
                    auto * __restrict__ buffer = d_buffer + tile_off;
                    
                    for( unsigned j = 0; j < ext_nx.y; j++ ) {
                        for( unsigned i = 0; i < ext_nx.x - shift; i++ ) {
                            local[ i + j * ystride ] = buffer[ i + shift + j * ystride ]; 
                        }
                        for( unsigned i = ext_nx.x - shift; i < ext_nx.x; i++ ) {
                            local[ i + j * ystride ] = 0;
                        }
                    }

                    // sync ...
                    for( unsigned i = 0; i < tile_vol; i++ ) buffer[i] = local[i];

                }
            }

            // Copy x guard cells
            copy_to_gc_x();

        } else {
            std::cerr << "x_shift_left(), shift value too large, must be <= gc.x.upper\n";
            exit(1);
        }
    }

    /**
     * @brief Perform a convolution with a 3 point kernel [a,b,c] along x
     * 
     * @param a     Kernel value a
     * @param b     Kernel value b
     * @param c     Kernel value c
     */
    template < typename S >
    void kernel3_x( S const a, S const b, S const c ) {
        if (( gc.x.lower > 0) && (gc.x.upper > 0)) {

            const int ystride = ext_nx.x;

            // On a GPU these would be on block shared memory
            T A[ tile_vol ];
            T B[ tile_vol ];

            // Loop over tiles
            for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {

                    const auto tile_idx = make_uint2( tx, ty );
                    const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                    const auto tile_off = tid * tile_vol ;

                    auto * __restrict__ buffer = d_buffer + tile_off;

                    // Copy data from tile buffer
                    for( unsigned i = 0; i < tile_vol; i++ ) A[i] = buffer[i];

                    // Apply kernel locally
                    for( unsigned iy = 0; iy < ext_nx.y; iy++ ) {
                        for( unsigned ix = gc.x.lower; ix < nx.x + gc.x.lower; ix ++) {
                            B [ iy * ystride + ix ] = A[ iy * ystride + (ix-1) ] * a +
                                                      A[ iy * ystride +  ix    ] * b +
                                                      A[ iy * ystride + (ix+1) ] * c;
                        }
                    }

                    // Copy data back to tile buffer
                    for( unsigned i = 0; i < tile_vol; i++ ) buffer[i] = B[i];
                }
            }

            // Update guard cells
            copy_to_gc_x();

        } else {
            std::cerr << "kernel_x3() requires at least 1 guard cell at both the lower and upper x boundaries.\n";
            exit(1);
        }

    }

    /**
     * @brief Perform a convolution with a 3 point kernel [a,b,c] along y
     * 
     * @param a     Kernel value a
     * @param b     Kernel value b
     * @param c     Kernel value c
     */
    template < typename S >
    void kernel3_y( S const a, S const b, S const c ) {

        if (( gc.y.lower > 0) && (gc.y.upper > 0)) {

            const int ystride = ext_nx.x;

            // On a GPU these would be on block shared memory
            T A[ tile_vol ];
            T B[ tile_vol ];

            // Loop over tiles
            for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {

                    const auto tile_idx = make_uint2( tx, ty );
                    const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                    const auto tile_off = tid * tile_vol;

                    auto * __restrict__ buffer = d_buffer + tile_off;

                    // Copy data from tile buffer
                    for( unsigned i = 0; i < tile_vol; i++ ) A[i] = buffer[i];

                    // Apply kernel locally
                    for( unsigned iy = 0; iy < ext_nx.y; iy++ ) {
                        for( unsigned ix = gc.x.lower; ix < nx.x + gc.x.lower; ix ++) {
                            B [ iy * ystride + ix ] = A[ (iy-1) * ystride + ix ] * a +
                                                      A[    iy  * ystride + ix ] * b +
                                                      A[ (iy+1) * ystride + ix ] * c;
                        }
                    }

                    // Copy data back to tile buffer
                    for( unsigned i = 0; i < tile_vol; i++ ) buffer[i] = B[i];
                }
            }

            // Update guard cells
            copy_to_gc_y();

        } else {
            std::cerr << "kernel3_y() requires at least 1 guard cell at both the lower and upper y boundaries.\n";
            exit(1);
        }

    }

    /**
     * @brief Save grid values to disk with full metadata
     * 
     * The field type <T> must be supported by ZDF file format
     * 
     */
    void save( zdf::grid_info &info, zdf::iteration &iter, std::string path ) {

        // Fill in global grid dimensions
        info.ndims = 2;
        info.count[0] = global_ntiles.x * nx.x;
        info.count[1] = global_ntiles.y * nx.y;

        // Allocate buffer on host to gather data
        T * h_data = memory::malloc<T>( gnx.x * gnx.y );

        // Gather data on contiguous grid
        gather( h_data );

        // Information on local chunk of grid data
        zdf::chunk chunk;
        chunk.count[0] = gnx.x;
        chunk.count[1] = gnx.y;
        chunk.start[0] = tile_off.x * nx.x;
        chunk.start[1] = tile_off.y * nx.y;
        chunk.stride[0] = chunk.stride[1] = 1;
        chunk.data = (void *) h_data;

        // Save data
        zdf::save_grid<T>( chunk, info, iter, path, part.get_comm() );

        // Free temporary buffer
        memory::free( h_data );
    };

    /**
     * @brief Save grid values to disk
     * 
     * @param filename      Output file name
     */
    void save( std::string filename ) {
        // Allocate buffer on host to gather data
        T * h_data = memory::malloc<T>( gnx.x * gnx.y );

        // Gather data on contiguous grid
        gather( h_data );

        uint64_t global[2] = { global_ntiles.x * nx.x, global_ntiles.y * nx.y };
        uint64_t start[2] = { tile_off.x * nx.x, tile_off.y * nx.y };
        uint64_t local[2] = { gnx.x, gnx.y };

        zdf::save_grid( h_data, 2, global, start, local, name, filename, part.get_comm() );

        // Free temporary buffer
        memory::free( h_data );
    }

};

#endif