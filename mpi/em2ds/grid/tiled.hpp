#pragma once

#include "../utils.hpp"
#include "../bounds.hpp"
#include "../vec_types.hpp"
#include "../parallel.hpp"

#include "../zdf/zdf.hpp"

#include <iostream>


namespace grid {

/**
 * @brief Tiled grid class with MPI support
 * 
 */
template <class T>
class tiled {
    protected:

    // Tags are paired so that a message sent with dest::lower is received
    // with source::upper (both have value 0). This ensures MPI tag matching
    // between sender and receiver without extra bookkeeping.

    /// @brief tags for outgoing messages
    struct source { enum tag { lower = 0, upper = 1 }; };
    /// @brief tags for incoming messages
    struct dest   { enum tag { upper = 0, lower = 1 }; };

    /// @brief Parallel partition
    const Partition & part;

    /// @brief Local number of tiles
    uint2 local_ntiles;

    /// @brief Start position of local tiles in global tile grid
    uint2 local_tile_start;

    /// @brief Consider local boundaries periodic
    int2 local_periodic;

    /// @brief Local grid dimensions (all local tiles)
    uint2 local_dims;

    /// @brief Buffers for sending messages
    bounds< Message<T>* > msg_send;

    /// @brief Buffers for receiving messages
    bounds< Message<T>* > msg_recv;

    /// @brief Data buffer
    T * d_buffer;

    /**
     * @brief Set the local node information. This information will (may) be
     * different for each parallel node
     * 
     * @note Global periodic information is taken from the parallel partition
     * 
     */
    void initialize( ) {

        // Get local number of tiles and position
        part.grid_local( global_ntiles, local_ntiles, local_tile_start );

        // Get local grid size
        local_dims = local_ntiles * tile_dims;

        // Get local periodic flag
        local_periodic.x = part.periodic.x && (part.dims.x == 1);
        local_periodic.y = part.periodic.y && (part.dims.y == 1);

        // Allocate main data buffer
        d_buffer = memory::malloc<T>( buffer_size() );

        // Get maximum message size
        int max_msg_size = std::max(
            ( local_ntiles.y * tile_ext_dims.y ) * std::max( gc.x.lower, gc.x.upper ),
            std::max( gc.y.lower, gc.y.upper ) * ( local_ntiles.x * tile_ext_dims.x )
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

        if ( tile_dims.x == 0 || tile_dims.y == 0 ) {
            std::cerr << "Invalid tile dimensions" << tile_dims << '\n';
            mpi::abort(1);
        }

        // Parallel partition
        if ( part.dims.x > global_ntiles.x ) {
            std::cerr << "Number of parallel nodes along x (" ;
            std::cerr << part.dims.x << ") is larger than number of tiles along x(";
            std::cerr << global_ntiles.x << '\n';
            mpi::abort(1);
        }

        if ( part.dims.y > global_ntiles.y ) {
            std::cerr << "Number of parallel nodes along y (" ;
            std::cerr << part.dims.y << ") is larger than number of tiles along y(";
            std::cerr << global_ntiles.y << '\n';
            mpi::abort(1);
        }
    }

    public:

    /// @brief Global number of tiles
    const uint2 global_ntiles;

    /// @brief Tile grid dimensions
    const uint2 tile_dims;
    
    /// @brief Tile guard cells
    const bounds_2d<unsigned int> gc;
    
    /// @brief Tile grid dimensions including guard cells
    const uint2 tile_ext_dims;

    /// @brief Local offset in cells between lower tile corner and position (0,0)
    const unsigned int offset;

    /// @brief Tile volume (may be larger than tile_ext_dim.x * tile_ext_dim.y for alignment)
    const std::size_t tile_vol;

    /// @brief Object name
    std::string name;

    /**
     * @brief Construct a new grid object
     * 
     * @param global_ntiles     Global number of tiles
     * @param tile_dims         Individual tile size
     * @param gc                Number of guard cells
     * @param part              Parallel partition
     */
    tiled( uint2 const global_ntiles, uint2 const tile_dims, bounds_2d<unsigned int> const gc, const Partition & part ):
        part( part ),
        d_buffer( nullptr ), 
        global_ntiles( global_ntiles ),
        tile_dims( tile_dims ),
        gc(gc),
        tile_ext_dims( make_uint2( gc.x.lower + tile_dims.x + gc.x.upper,
                            gc.y.lower + tile_dims.y + gc.y.upper )),
        offset( gc.y.lower * tile_ext_dims.x + gc.x.lower ),
        tile_vol( roundup4( tile_ext_dims.x * tile_ext_dims.y ) ),
        name( "tiled grid" )
    {
        // Validate parameters
        validate_parameters();

        // Set local information (ntiles, tile_start and local_periodic)
        initialize();
    };

    /**
     * @brief Construct a new tile grid object
     * 
     * @note: The number of guard cells is set to 0
     * 
     * @param global_ntiles     Global number of tiles
     * @param tile_dims                Individual tile size
     * @param part              Parallel partition
     */
    tiled( uint2 const global_ntiles, uint2 const tile_dims, const Partition & part ):
        part( part ),
        d_buffer( nullptr ),
        global_ntiles( global_ntiles ),
        tile_dims( tile_dims ),
        gc( 0 ),
        tile_ext_dims( make_uint2( tile_dims.x, tile_dims.y )),
        offset( 0 ),
        tile_vol( roundup4( tile_dims.x * tile_dims.y )),
        name( "tiled grid" )
    {
        // Validate parameters
        validate_parameters();

        // Set local information (ntiles, tile_start and periodic)
        initialize();
    };

    /**
    * @brief Move constructor
    *
    * @note Steals the data buffer and message buffers from `other`, leaving
    *       it in a valid but empty state (safe to destruct).
    *
    * @param other     tiled grid to move from
    */
    tiled( tiled && other ) noexcept :
        part( other.part ),
        local_ntiles( other.local_ntiles ),
        local_tile_start( other.local_tile_start ),
        local_periodic( other.local_periodic ),
        local_dims( other.local_dims ),
        msg_send( other.msg_send ),
        msg_recv( other.msg_recv ),
        d_buffer( other.d_buffer ),
        global_ntiles( other.global_ntiles ),
        tile_dims( other.tile_dims ),
        gc( other.gc ),
        tile_ext_dims( other.tile_ext_dims ),
        offset( other.offset ),
        tile_vol( other.tile_vol ),
        name( std::move( other.name ) )
    {
        // Null out other's owned resources so its destructor is a no-op
        other.d_buffer = nullptr;

        other.msg_send.lower = nullptr;
        other.msg_send.upper = nullptr;
        other.msg_recv.lower = nullptr;
        other.msg_recv.upper = nullptr;
    }

    /**
     * @brief tiled grid destructor
     * 
     */
    ~tiled(){
        delete msg_recv.lower;
        delete msg_recv.upper;
        delete msg_send.lower;
        delete msg_send.upper;

        if ( d_buffer != nullptr ) memory::free( d_buffer );
    };

    /**
     * @brief Delete default copy constructor
     * 
     */
    tiled(const tiled&) = delete;

    /**
     * @brief Delete default copy constructor
     * 
     */
    tiled& operator=(const tiled&) = delete;

    /**
     * @brief Get a pointer to the data buffer
     * 
     * @return T* 
     */
    T * data() const noexcept { return d_buffer; }

    /**
     * @brief Get a pointer to a specific tile
     * 
     * @param tid   Tile index (flat)
     * @return T* 
     */
    T * tile_data( const unsigned int tid ) const noexcept {
        return & d_buffer[ tid * tile_vol ];
    }

    /**
     * @brief Get a pointer to a specific tile
     * 
     * @param tx    x tile index
     * @param ty    y tile index
     * @return T* 
     */
    T * tile_data( const unsigned int tx, const unsigned int ty ) const noexcept {
        return & d_buffer[ (ty * local_ntiles.x + tx) * tile_vol ];
    }

    /**
     * @brief Get a pointer to a specific tile
     * 
     * @param tid   Tile index (x,y)
     * @return T* 
     */
    T * tile_data( const uint2 tid ) const noexcept {
        return & d_buffer[ (tid.y * local_ntiles.x + tid.x) * tile_vol ];
    }

    /**
     * @brief Get the local number of tiles
     * 
     * @return uint2 
     */
    uint2 get_local_ntiles() const noexcept { return local_ntiles; };

    /**
     * @brief Returns the local tile offset in the global MPI tile grid
     * 
     * @return uint2 
     */
    uint2 get_local_tile_start() const noexcept { return local_tile_start; };

    /**
     * @brief Get the global grid dimensions
     * 
     * @return uint2 
     */
    uint2 get_global_dims() const noexcept { return global_ntiles  * tile_dims; }

    /**
     * @brief Get the local dims object
     * 
     * @return uint2 
     */
    uint2 get_local_dims() const noexcept { return local_dims; }

    const Partition & get_part() const noexcept { return  part; }

    /**
     * @brief Stream extraction
     * 
     * @param os 
     * @param obj 
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, const tiled & obj) {
        return os 
            << obj.name << "{ " 
            << obj.local_ntiles << " local tiles, "
            << obj.local_tile_start << " local start, "
            << obj.global_ntiles << " global tiles, "
            << obj.tile_dims << " points/tile }";
    }


    /**
     * @brief Buffer size
     * 
     * @return total size of data buffers (in elements)
     */
    std::size_t buffer_size() const noexcept {
        return (static_cast <std::size_t> (tile_vol)) * ( local_ntiles.x * local_ntiles.y ) ;
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
        #pragma omp parallel for
        for( size_t i = 0; i < buffer_size( ); i++ ) d_buffer[i] = val;
    };

    /**
     * @brief Adds another grid object on top of local object
     * 
     * @param rhs         Other object to add
     */
    void add( const tiled &rhs ) {
        size_t const size = buffer_size( );

        #pragma omp parallel for
        for( size_t i = 0; i < size; i++ ) d_buffer[i] += rhs.d_buffer[i];
    };

    /**
     * @brief Operator +=
     * 
     * @param rhs           Other grid to add
     * @return grid<T>& 
     */
    tiled & operator+=(const tiled & rhs) {
        add( rhs );
        return *this;
    }

    /**
     * @brief Gather tiled grid values
     * 
     * @note
     * The default behavior is to output data into a contiguous array of 
     * dimensions local_dims.y * local_dims.x. The stride parameter can be used
     * to specify different memory layouts.
     * 
     * @tparam T2               Output buffer datatype, must support T2 = T assignment
     * @param out               Output buffer
     * @param stride            (optional) Output buffer stride, defaults to a
     *                          contiguous memory layout
     * @return unsigned int     Total number of cells
     */
    template< typename T2 >
    unsigned int gather( T2 * const __restrict__ out, uint2 stride = {0,0} ) const {

        // Default to contiguous memory layout
        if ( stride.x == 0 ) {
            stride = make_uint2( 1, local_dims.x );
        }

        if ( stride.x == 1 ) {
            // Optimized version for x stride == 1
            #pragma omp parallel for collapse(2)
            for( unsigned ty = 0; ty < local_ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < local_ntiles.x; tx ++ ) {
                    // const auto tid = ty * local_ntiles.x + tx;
                    // const auto start = tid * tile_vol + offset;
                    T * const __restrict__ tdata = tile_data( tx, ty ) + offset;

                    const auto gix0 = tx * tile_dims.x;
                    const auto giy0 = ty * tile_dims.y;

                    // Loop inside tile
                    for( unsigned iy = 0; iy < tile_dims.y; iy ++ ) {
                        for( unsigned ix = 0; ix < tile_dims.x; ix ++ ) {
                            out[ (giy0 + iy) * stride.y + (gix0 + ix) ] =
                                tdata[ iy * tile_ext_dims.x + ix ];
                        }
                    }
                }
            }
        } else {
            // Arbitrary x stride
            #pragma omp parallel for collapse(2)
            for( unsigned ty = 0; ty < local_ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < local_ntiles.x; tx ++ ) {
                    const auto tid = ty * local_ntiles.x + tx;
                    const auto start = tid * tile_vol + offset;
                    T * const __restrict__ tile_data = & d_buffer[ start ];

                    const auto gix0 = tx * tile_dims.x;
                    const auto giy0 = ty * tile_dims.y;

                    // Loop inside tile
                    for( unsigned iy = 0; iy < tile_dims.y; iy ++ ) {
                        for( unsigned ix = 0; ix < tile_dims.x; ix ++ ) {
                            out[ (giy0 + iy) * stride.y + (gix0 + ix) * stride.x ] =
                                tile_data[ iy * tile_ext_dims.x + ix ];
                        }
                    }
                }
            }
        }

        return local_dims.x * local_dims.y;
    }

    /**
     * @brief Scatter data into the tile grid and updates guard cell values
     * 
     * @note
     * The default behavior is consider that the input data is a contiguous
     * array of dimensions local_dims.y * local_dims.x. The stride parameter
     * can be used to specify different memory layouts.
     * 
     * @tparam T2               Input buffer datatype
     * @param d_in              Input buffer
     * @param in_stride         (optional) Input buffer stride, defaults to a
     *                          contiguous memory layout
     * @return unsigned int     Total number of cells
     */
    template< typename T2 >
    unsigned int scatter( T2 const * const __restrict__ d_in, uint2 stride = {0,0} ) {

        // Default to contiguous memory layout
        if ( stride.x == 0 ) {
            stride = make_uint2( 1, local_dims.x );
        }

        if ( stride.x == 1 ) {
            // Optimized version for x stride == 1
            #pragma omp parallel for collapse(2)
            for( unsigned ty = 0; ty < local_ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < local_ntiles.x; tx ++ ) {
                    const auto tid = ty * local_ntiles.x + tx;
                    const auto start = tid * tile_vol + offset;
                    T * const __restrict__ tile_data = & d_buffer[ start ];

                    const auto gix0 = tx * tile_dims.x;
                    const auto giy0 = ty * tile_dims.y;

                    // Loop inside tile
                    for( unsigned iy = 0; iy < tile_dims.y; iy ++ ) {
                        for( unsigned ix = 0; ix < tile_dims.x; ix ++ ) {
                            tile_data[ iy * tile_ext_dims.x + ix ] = 
                                d_in[ (giy0 + iy) * stride.y + (gix0 + ix) ];
                        }
                    }
                }
            }

        } else {
            // Arbitrary x stride
            #pragma omp parallel for collapse(2)
            for( unsigned ty = 0; ty < local_ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < local_ntiles.x; tx ++ ) {
                    const auto tid = ty * local_ntiles.x + tx;
                    const auto start = tid * tile_vol + offset;
                    T * const __restrict__ tile_data = & d_buffer[ start ];

                    const auto gix0 = tx * tile_dims.x;
                    const auto giy0 = ty * tile_dims.y;

                    // Loop inside tile
                    for( unsigned iy = 0; iy < tile_dims.y; iy ++ ) {
                        for( unsigned ix = 0; ix < tile_dims.x; ix ++ ) {
                            tile_data[ iy * tile_ext_dims.x + ix ] = 
                                d_in[ (giy0 + iy) * stride.y + (gix0 + ix) * stride.x ];
                        }
                    }
                }
            }
        }

        // Update guard cell values
        copy_to_gc();

        return local_dims.x * local_dims.y;
    }

    /**
     * @brief Scatter data into the tile grid and updates guard cell values
     * 
     * @note
     * The default behavior is consider that the input data is a contiguous
     * array of dimensions local_dims.y * local_dims.x. The stride parameter
     * can be used to specify different memory layouts.
     *
     * The operation T2 = T3 * T must be supported.
     * 
     * @tparam T2               Input buffer datatype
     * @tparam T3               Scale factor datatype
     * @param d_in              Input buffer
     * @param scale             Scale factor
     * @param in_stride         (optional) Input buffer stride, defaults to a
     *                          contiguous memory layout
     * @return unsigned int     Total number of cells
     */
    template< typename T2, typename T3 >
    unsigned int scatter( T2 const * const __restrict__ d_in, T3 const scale, 
        uint2 stride = {0,0} ) {

        // Default to contiguous memory layout
        if ( stride.x == 0 ) {
            stride = make_uint2( 1, local_dims.x );
        }

        if ( stride.x == 1 ) {
            // Optimized version for x stride == 1
            #pragma omp parallel for collapse(2)
            for( unsigned ty = 0; ty < local_ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < local_ntiles.x; tx ++ ) {
                    const auto tid = ty * local_ntiles.x + tx;
                    const auto start = tid * tile_vol + offset;
                    T * const __restrict__ tile_data = & d_buffer[ start ];

                    const auto gix0 = tx * tile_dims.x;
                    const auto giy0 = ty * tile_dims.y;

                    // Loop inside tile
                    for( unsigned iy = 0; iy < tile_dims.y; iy ++ ) {
                        for( unsigned ix = 0; ix < tile_dims.x; ix ++ ) {
                            tile_data[ iy * tile_ext_dims.x + ix ] = 
                                d_in[ (giy0 + iy) * stride.y + (gix0 + ix) ] * scale;
                        }
                    }
                }
            }

        } else {
            // Arbitrary x stride
            #pragma omp parallel for collapse(2)
            for( unsigned ty = 0; ty < local_ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < local_ntiles.x; tx ++ ) {
                    const auto tid = ty * local_ntiles.x + tx;
                    const auto start = tid * tile_vol + offset;
                    T * const __restrict__ tile_data = & d_buffer[ start ];

                    const auto gix0 = tx * tile_dims.x;
                    const auto giy0 = ty * tile_dims.y;

                    // Loop inside tile
                    for( unsigned iy = 0; iy < tile_dims.y; iy ++ ) {
                        for( unsigned ix = 0; ix < tile_dims.x; ix ++ ) {
                            tile_data[ iy * tile_ext_dims.x + ix ] = 
                                d_in[ (giy0 + iy) * stride.y + (gix0 + ix) * stride.x ] *
                                scale;
                        }
                    }
                }
            }
        }

        // Update guard cell values
        copy_to_gc();

        return local_dims.x * local_dims.y;
    }

    /**
     * @brief Copies edge values to X neighboring guard cells
     * 
     */
    void local_copy_to_gc_x() {

        // Loop over tiles
        #pragma omp parallel for
        for( unsigned tid = 0; tid < local_ntiles.y * local_ntiles.x; tid ++ ) {

            const auto tile_idx = make_int2( tid % local_ntiles.x, tid / local_ntiles.x );
            const auto tile_off = tid * tile_vol;
            const auto ystride  = tile_ext_dims.x;

            auto * __restrict__ local = & d_buffer[ tile_off ];

            {   // Copy from lower neighbour
                int neighbor_tx = tile_idx.x - 1;
                if ( local_periodic.x && neighbor_tx < 0 ) neighbor_tx += local_ntiles.x;

                if ( neighbor_tx >= 0 ) {
                    const auto neighbor_off = (tile_idx.y * local_ntiles.x + neighbor_tx) * tile_vol;
                    auto * __restrict__ x_lower = & d_buffer[ neighbor_off ] ;
                    for( unsigned j = 0; j < tile_ext_dims.y; j++ ) {
                        for( unsigned i = 0; i < gc.x.lower; i++ ) {
                            local[ i + j * ystride ] = x_lower[ tile_dims.x + i + j * ystride ];
                        }
                    }
                }
            }

            {   // Copy from upper neighbour
                int neighbor_tx = tile_idx.x + 1;
                if ( local_periodic.x && neighbor_tx >= static_cast<int>(local_ntiles.x) ) neighbor_tx -= local_ntiles.x;

                if ( neighbor_tx < static_cast<int>(local_ntiles.x) ) {
                    const auto neighbor_off = (tile_idx.y * local_ntiles.x + neighbor_tx) * tile_vol;
                    auto * __restrict__ x_upper =  & d_buffer[ neighbor_off ] ;
                    for( unsigned j = 0; j < tile_ext_dims.y; j++ ) {
                        for( unsigned i = 0; i < gc.x.upper; i++ ) {
                            local[ gc.x.lower + tile_dims.x + i + j * ystride ] = x_upper[ gc.x.lower + i + j * ystride ];
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
        for( unsigned tid = 0; tid < local_ntiles.y * local_ntiles.x; tid ++ ) {

            const auto tile_idx = make_int2( tid % local_ntiles.x, tid / local_ntiles.x );
            const auto tile_off = tid * tile_vol;
            const auto ystride  = tile_ext_dims.x;

            auto * __restrict__ local = & d_buffer[ tile_off ];
            
            {   // Copy from lower neighbour
                int neighbor_ty = tile_idx.y - 1;
                if ( local_periodic.y && neighbor_ty < 0 ) neighbor_ty += local_ntiles.y;

                if ( neighbor_ty >= 0 ) {
                    const auto neighbor_off = (neighbor_ty * local_ntiles.x + tile_idx.x) * tile_vol;
                    auto * __restrict__ y_lower = & d_buffer [ neighbor_off ] ;
                    for( int j = 0; j < gc.y.lower; j++ ) {
                        for( int i = 0; i < tile_ext_dims.x; i++ ) {
                            local[ i + j * ystride ] = y_lower[ i + ( tile_dims.y + j ) * ystride ];
                        }
                    }
                }
            }

            {   // Copy from upper neighbour
                int neighbor_ty = tile_idx.y + 1;
                if ( local_periodic.y && neighbor_ty >= static_cast<int>(local_ntiles.y) ) neighbor_ty -= local_ntiles.y;

                if ( neighbor_ty < static_cast<int>(local_ntiles.y) ) {
                    const auto neighbor_off = (neighbor_ty * local_ntiles.x + tile_idx.x) * tile_vol;
                    auto * __restrict__ y_upper = & d_buffer [ neighbor_off ];
                    for( int j = 0; j < gc.y.upper; j++ ) {
                        for( int i = 0; i < tile_ext_dims.x; i++ ) {
                            local[ i + ( gc.y.lower + tile_dims.y + j ) * ystride ] = y_upper[ i + ( gc.y.lower + j ) * ystride ];
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief Copies x values to neighboring guard cells, including cells on 
     *        other parallel nodes
     * 
     */
    void copy_to_gc_x() {

        // Get x neighbors
        int lnode = part.get_neighbor(-1, 0 );
        int unode = part.get_neighbor(+1, 0 );

        // Disable messages if only 1 node along  x direction
        if ( part.dims.x == 1 ) lnode = unode = -1;

        // Post message receives
        if ( lnode >= 0 ) msg_recv.lower->irecv( lnode, source::lower );
        if ( unode >= 0 ) msg_recv.upper->irecv( unode, source::upper );

        // Send message - lower neighbor
        if ( lnode >= 0 ) {
            unsigned int tx = 0;
            for( unsigned ty = 0; ty < local_ntiles.y; ty++ ) {
                
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * local_ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_send.lower-> buffer [ ty * tile_ext_dims.y * gc.x.upper ];


                for( unsigned j = 0; j < tile_ext_dims.y; j++ ) {
                    for( unsigned i = 0; i < gc.x.upper; i++ ) {
                        msg[ j * gc.x.upper + i ] = local[ j * tile_ext_dims.x + gc.x.lower + i ];
                    }
                }
            }

            int msg_size = ( tile_ext_dims.y * local_ntiles.y ) * gc.x.upper;
            msg_send.lower->isend( msg_size, lnode, dest::lower );
        }

        // Send message - upper neighbor
        if ( unode >= 0 ) {
            unsigned int tx = local_ntiles.x - 1;
            for( unsigned ty = 0; ty < local_ntiles.y; ty++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * local_ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_send.upper-> buffer[ ty * tile_ext_dims.y * gc.x.lower ];

                for( unsigned j = 0; j < tile_ext_dims.y; j++ ) {
                    for( unsigned i = 0; i < gc.x.lower; i++ ) {
                        msg[ j * gc.x.lower + i ] = local[ j * tile_ext_dims.x + tile_dims.x + i ];
                    }
                }
            }

            int msg_size = ( tile_ext_dims.y * local_ntiles.y ) * gc.x.lower;
            msg_send.upper->isend( msg_size, unode, dest::upper );
        }

        // Process local tiles
        local_copy_to_gc_x();

        // Receive message - lower neighbor
        if ( lnode >= 0 ) {
            msg_recv.lower-> wait();

            unsigned int tx = 0;
            for( unsigned ty = 0; ty < local_ntiles.y; ty++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * local_ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_recv.lower-> buffer[ ty * tile_ext_dims.y * gc.x.lower ];

                for( unsigned j = 0; j < tile_ext_dims.y; j++ ) {
                    for( unsigned i = 0; i < gc.x.lower; i++ ) {
                        local[ j * tile_ext_dims.x + i ] = msg[ j * gc.x.lower + i ];
                    }
                }
            }
        }

        // Receive message - upper neighbor
        if ( unode >= 0 ) {
            msg_recv.upper-> wait();

            unsigned int tx = local_ntiles.x-1;
            for( unsigned ty = 0; ty < local_ntiles.y; ty++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * local_ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_recv.upper -> buffer[ ty * tile_ext_dims.y * gc.x.upper ];

                for( unsigned j = 0; j < tile_ext_dims.y; j++ ) {
                    for( unsigned i = 0; i < gc.x.upper; i++ ) {
                        local[ j * tile_ext_dims.x + gc.x.lower + tile_dims.x + i ] = msg[ j * gc.x.upper + i ];
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

        // Disable messages if only 1 node along y direction
        if ( part.dims.y == 1 ) lnode = unode = -1;
        
        // Post message receives
        if ( lnode >= 0 ) msg_recv.lower->irecv( lnode, source::lower );
        if ( unode >= 0 ) msg_recv.upper->irecv( unode, source::upper );

        // Post message sends
        if ( lnode >= 0 ) {
            unsigned int ty = 0;
            for( unsigned tx = 0; tx < local_ntiles.x; tx++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * local_ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_send.lower-> buffer[ tx * tile_ext_dims.x * gc.y.upper ];
                
                for( unsigned j = 0; j < gc.y.upper; j++ ) {
                    for( unsigned i = 0; i < tile_ext_dims.x; i++ ) {
                        msg[ j * tile_ext_dims.x + i ] = local[ ( gc.y.lower + j ) * tile_ext_dims.x + i ];
                    }
                }
            }

            int msg_size =  gc.y.upper * ( local_ntiles.x * tile_ext_dims.x );
            msg_send.lower->isend( msg_size, lnode, dest::lower );
        }

        if ( unode >= 0 ) {
            unsigned int ty = local_ntiles.y-1;
            for( unsigned tx = 0; tx < local_ntiles.x; tx++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * local_ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_send.upper -> buffer[ tx * tile_ext_dims.x * gc.y.lower ];

                for( unsigned j = 0; j < gc.y.lower; j++ ) {
                    for( unsigned i = 0; i < tile_ext_dims.x; i++ ) {
                        msg[ j * tile_ext_dims.x + i ] = local[ ( tile_dims.y + j ) * tile_ext_dims.x + i ];
                    }
                }
            }

            int msg_size = gc.y.lower * ( local_ntiles.x * tile_ext_dims.x );
            msg_send.upper -> isend( msg_size, unode, dest::upper );
        }

        // Process local tiles
        local_copy_to_gc_y();

        // Wait for receive messages to complete and copy data
        if ( lnode >= 0 ) {
            msg_recv.lower-> wait();

            unsigned int ty = 0;
            for( unsigned tx = 0; tx < local_ntiles.x; tx++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * local_ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_recv.lower-> buffer[ tx * tile_ext_dims.x * gc.y.lower ];

                for( unsigned j = 0; j < gc.y.lower; j++ ) {
                    for( unsigned i = 0; i < tile_ext_dims.x; i++ ) {
                        local[ j * tile_ext_dims.x + i ] =  msg[ j * tile_ext_dims.x + i ];
                    }
                }
            }
        }

        if ( unode >= 0 ) {
            msg_recv.upper-> wait();

            unsigned int ty = local_ntiles.y - 1;
            for( unsigned tx = 0; tx < local_ntiles.x; tx++ ) {
                
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * local_ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_recv.upper-> buffer[ tx * tile_ext_dims.x * gc.y.upper ];

                for( unsigned j = 0; j < gc.y.upper; j++ ) {
                    for( unsigned i = 0; i < tile_ext_dims.x; i++ ) {
                        local[ ( gc.y.lower + tile_dims.y + j ) * tile_ext_dims.x + i ] =  msg[ j * tile_ext_dims.x + i ];
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

    /**
     * @brief Adds values from neighboring x guard cells to local data
     * 
     */
    void local_add_from_gc_x() {
        // Add along x direction

        // Loop over tiles
        #pragma omp parallel for
        for( int tid = 0; tid < local_ntiles.y * local_ntiles.x; tid ++ ) {

            const auto tile_idx = make_int2( tid % local_ntiles.x, tid / local_ntiles.x );
            const auto tile_off = tid * tile_vol;
            const auto ystride  = tile_ext_dims.x;

            auto * __restrict__ local = & d_buffer[ tile_off ];
            
            {   // Add from lower neighbour
                int neighbor_tx = tile_idx.x - 1;
                if ( local_periodic.x && neighbor_tx < 0 ) neighbor_tx += local_ntiles.x;

                if ( neighbor_tx >= 0 ) {
                    const auto neighbor_off = (tile_idx.y * local_ntiles.x + neighbor_tx) * tile_vol;
                    T * __restrict__ x_lower = & d_buffer[neighbor_off]; 
                    for( unsigned j = 0; j < tile_ext_dims.y; j++ ) {
                        for( unsigned i = 0; i < gc.x.upper; i++ ) {
                            local[ gc.x.lower + i + j * ystride ] += x_lower[ gc.x.lower + tile_dims.x + i + j * ystride ];
                        }
                    }
                }
            }

            {   // Add from upper neighbour
                int neighbor_tx = tile_idx.x + 1;
                if ( local_periodic.x && neighbor_tx >= static_cast<int>(local_ntiles.x) ) neighbor_tx -= local_ntiles.x;

                if ( neighbor_tx < static_cast<int>(local_ntiles.x) ) {
                    const auto neighbor_off = (tile_idx.y * local_ntiles.x + neighbor_tx) * tile_vol;
                    auto * __restrict__ x_upper = & d_buffer[neighbor_off]; 
                    for( int j = 0; j < tile_ext_dims.y; j++ ) {
                        for( int i = 0; i < gc.x.lower; i++ ) {
                            local[ tile_dims.x + i + j * ystride ] += x_upper[ i + j * ystride ];
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
        for( int tid = 0; tid < local_ntiles.y * local_ntiles.x; tid ++ ) {

            const auto tile_idx = make_int2( tid % local_ntiles.x, tid / local_ntiles.x );
            const auto tile_off = tid * tile_vol;
            const auto ystride  = tile_ext_dims.x;

            auto * __restrict__ local = & d_buffer[ tile_off ];
            
            {   // Add from lower neighbour
                int neighbor_ty = tile_idx.y - 1;
                if ( local_periodic.y && neighbor_ty < 0 ) neighbor_ty += local_ntiles.y;

                if ( neighbor_ty >= 0 ) {
                    const auto neighbor_off = (neighbor_ty * local_ntiles.x + tile_idx.x) * tile_vol;
                    auto * __restrict__ y_lower = & d_buffer [ neighbor_off ]; 
                    for( int j = 0; j < gc.y.upper; j++ ) {
                        for( int i = 0; i < tile_ext_dims.x; i++ ) {
                            local[ i + ( gc.y.lower + j ) * ystride ] += y_lower[ i + ( gc.y.lower + tile_dims.y + j ) * ystride ];
                        }
                    }
                }
            }

            {   // Add from upper neighbour
                int neighbor_ty = tile_idx.y + 1;
                if ( local_periodic.y && neighbor_ty >= static_cast<int>(local_ntiles.y) ) neighbor_ty -= local_ntiles.y;

                if ( neighbor_ty < static_cast<int>(local_ntiles.y) ) {
                    const auto neighbor_off = (neighbor_ty * local_ntiles.x + tile_idx.x) * tile_vol;
                    auto * __restrict__ y_upper = & d_buffer [ neighbor_off ]; 
                    for( unsigned j = 0; j < gc.y.lower; j++ ) {
                        for( unsigned i = 0; i < tile_ext_dims.x; i++ ) {
                            local[ i + ( tile_dims.y + j ) * ystride ] += y_upper[ i + j * ystride ];
                        }
                    }
                }
            }
        }
    };

    /**
     * @brief Adds values from neighboring x guard cells to local data,
     *        including cells from other parallel nodes
     */
    void add_from_gc_x() {

        // Get x neighbors
        int lnode = part.get_neighbor(-1, 0 );
        int unode = part.get_neighbor(+1, 0 );

        // Disable messages if only 1 node along  x direction
        if ( part.dims.x == 1 ) lnode = unode = -1;

        // Post message receives
        if ( lnode >= 0 ) msg_recv.lower->irecv( lnode, source::lower );
        if ( unode >= 0 ) msg_recv.upper->irecv( unode, source::upper );

        // Send message - lower neighbor
        if ( lnode >= 0 ) {

            unsigned int tx = 0;
            for( unsigned ty = 0; ty < local_ntiles.y; ty++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * local_ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                
                T * __restrict__ msg = & msg_send.lower-> buffer [ ty * tile_ext_dims.y * gc.x.lower ];

                for( unsigned j = 0; j < tile_ext_dims.y; j++ ) {
                    for( unsigned i = 0; i < gc.x.lower; i++ ) {
                        msg[ j * gc.x.lower + i ] = local[ j * tile_ext_dims.x + i ];
                    }
                }
            }

            int msg_size = ( local_ntiles.y * tile_ext_dims.y ) * gc.x.lower;
            msg_send.lower->isend( msg_size, lnode, dest::lower );
        }

        // Send message - upper neighbor
        if ( unode >= 0 ) {

            unsigned int tx = local_ntiles.x - 1;
            for( unsigned ty = 0; ty < local_ntiles.y; ty++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * local_ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_send.upper-> buffer[ ty * tile_ext_dims.y * gc.x.upper ];

                for( unsigned j = 0; j < tile_ext_dims.y; j++ ) {
                    for( unsigned i = 0; i < gc.x.upper; i++ ) {
                        msg[ j * gc.x.upper + i ] = local[ j * tile_ext_dims.x + gc.x.lower + tile_dims.x + i ];
                    }
                }
            }

            int msg_size = ( local_ntiles.y * tile_ext_dims.y ) * gc.x.upper;
            msg_send.upper->isend( msg_size, unode, dest::upper );
        }

        // Process local tiles
        local_add_from_gc_x();

        // Receive message - lower neighbor
        if ( lnode >= 0 ) {
            msg_recv.lower-> wait();

            unsigned int tx = 0;
            for( unsigned ty = 0; ty < local_ntiles.y; ty++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * local_ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_recv.lower-> buffer[ ty * tile_ext_dims.y * gc.x.upper ];

                for( unsigned j = 0; j < tile_ext_dims.y ; j++ ) {
                    for( unsigned i = 0; i < gc.x.upper; i++ ) {
                        local[ j * tile_ext_dims.x + gc.x.lower + i ] += msg[ j * gc.x.upper + i ] ;
                    }
                }
            }
        }

        // Receive message - upper neighbor
        if ( unode >= 0 ) {
            msg_recv.upper-> wait();

            unsigned int tx = local_ntiles.x - 1;
            for( unsigned ty = 0; ty < local_ntiles.y; ty++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * local_ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_recv.upper-> buffer[ ty * tile_ext_dims.y * gc.x.lower ];
                
                for( unsigned j = 0; j < tile_ext_dims.y; j++ ) {
                    for( unsigned i = 0; i < gc.x.lower; i++ ) {
                        local[ j * tile_ext_dims.x + tile_dims.x + i ] += msg[ j * gc.x.lower + i ] ;
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
     */
    void add_from_gc_y() {
        // Get y neighbors
        int lnode = part.get_neighbor( 0, -1 );
        int unode = part.get_neighbor( 0, +1 );

        // Disable messages if only 1 node along y direction
        if ( part.dims.y == 1 ) lnode = unode = -1;

        // Post message receives
        if ( lnode >= 0 ) msg_recv.lower->irecv( lnode, source::lower );
        if ( unode >= 0 ) msg_recv.upper->irecv( unode, source::upper );

        // Send message - lower neighbor
        if ( lnode >= 0 ) {
            unsigned int ty = 0;
            for( unsigned tx = 0; tx < local_ntiles.x; tx++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * local_ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_send.lower-> buffer[ tx * ( tile_ext_dims.x * gc.y.lower ) ];

                for( unsigned j = 0; j < gc.y.lower; j++ ) {
                    for( unsigned i = 0; i < tile_ext_dims.x; i++ ) {
                        msg[ j * tile_ext_dims.x + i ] = local[ j * tile_ext_dims.x + i ];
                    }
                }
            }

            int msg_size = gc.y.lower * ( local_ntiles.x * tile_ext_dims.x );
            msg_send.lower->isend( msg_size, lnode, dest::lower );
        }

        // Send message - upper neighbor
        if ( unode >= 0 ) {

            unsigned int ty = local_ntiles.y-1;
            for( unsigned tx = 0; tx < local_ntiles.x; tx++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * local_ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_send.upper-> buffer[ tx * gc.y.upper * tile_ext_dims.x ];

                for( unsigned j = 0; j < gc.y.upper; j++ ) {
                    for( unsigned i = 0; i < tile_ext_dims.x; i++ ) {
                        msg[ j * tile_ext_dims.x + i ] = local[ ( gc.y.lower + tile_dims.y + j ) * tile_ext_dims.x + i ];
                    }
                }
            }

            int msg_size    = gc.y.upper * ( local_ntiles.x * tile_ext_dims.x );
            msg_send.upper->isend( msg_size, unode, dest::upper );
        }

        // Process local tiles
        local_add_from_gc_y();

        // Receive message - lower neighbor
        if ( lnode >= 0 ) {
            msg_recv.lower-> wait();

            unsigned int ty = 0;
            for( unsigned tx = 0; tx < local_ntiles.x; tx++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * local_ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_recv.lower-> buffer[ tx * gc.y.upper * tile_ext_dims.x ];

                for( unsigned j = 0; j < gc.y.upper; j++ ) {
                    for( unsigned i = 0; i < tile_ext_dims.x; i++ ) {
                       local[ ( gc.y.lower + j ) * tile_ext_dims.x + i ] += msg[ j * tile_ext_dims.x + i ];
                    }
                }
            }
        }

        // Receive message - upper neighbor
        if ( unode >= 0 ) {
            msg_recv.upper-> wait();

            unsigned int ty = local_ntiles.y - 1;
            for( unsigned tx = 0; tx < local_ntiles.x; tx++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * local_ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];
                T * __restrict__ msg = & msg_recv.upper -> buffer[ tx * (gc.y.lower * tile_ext_dims.x) ];

                for( unsigned j = 0; j < gc.y.lower; j++ ) {
                    for( unsigned i = 0; i < tile_ext_dims.x; i++ ) {
                        local[ ( tile_dims.y + j ) * tile_ext_dims.x + i ] +=  msg[ j * tile_ext_dims.x + i ];
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

        if ( shift > 0 && shift < gc.x.upper ) {

            const int ystride = tile_ext_dims.x;

            // Loop over tiles
            #pragma omp parallel for
            for( int tid = 0; tid < local_ntiles.y * local_ntiles.x; tid ++ ) {
                const auto tile_off = tid * tile_vol ;
                
                auto * __restrict__ buffer = & d_buffer[ tile_off ];
                
                for( int iy = 0; iy < tile_ext_dims.y; iy++ ) {
                    for( int ix = 0; ix < tile_ext_dims.x - shift; ix++ ) {
                        buffer[ ix + iy * ystride ] = buffer[ (ix + shift) + iy * ystride ]; 
                    }
                    for( int ix = tile_ext_dims.x - shift; ix < tile_ext_dims.x; ix++ ) {
                        buffer[ ix + iy * ystride ] = T{0};
                    }
                }
            }

            // Copy x guard cells
            copy_to_gc_x();

        } else {
            std::cerr << "x_shift_left(), invalid shift value, must be 0 < shift <= gc.x.upper\n";
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

            const int ystride = tile_ext_dims.x;

            // Loop over tiles
            #pragma omp parallel for
            for( int tid = 0; tid < local_ntiles.y * local_ntiles.x; tid++ ) {

                // On a GPU these would be on block shared memory
                T A[ tile_vol ];
                T B[ tile_vol ];

                const auto tile_off = tid * tile_vol ;

                auto * __restrict__ buffer = & d_buffer[ tile_off ];

                // Copy data from tile buffer
                for( int i = 0; i < tile_vol; i++ ) {
                    A[i] = B[i] = buffer[i];
                }

                // Apply kernel locally
                for( int iy = 0; iy < tile_ext_dims.y; iy++ ) {
                    for( int ix = gc.x.lower; ix < gc.x.lower + tile_dims.x; ix ++) {
                        B[ iy * ystride + ix ] = A[ iy * ystride + (ix-1) ] * a +
                                                 A[ iy * ystride +  ix    ] * b +
                                                 A[ iy * ystride + (ix+1) ] * c;
                    }
                }

                // Copy data back to tile buffer
                for( int i = 0; i < tile_vol; i++ ) buffer[i] = B[i];
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

            const int ystride = tile_ext_dims.x;

            // Loop over tiles
            #pragma omp parallel for
            for( int tid = 0; tid < local_ntiles.y * local_ntiles.x; tid++ ) {

                // On a GPU these would be on block shared memory
                T A[ tile_vol ];
                T B[ tile_vol ];

                const auto tile_off = tid * tile_vol ;

                auto * __restrict__ buffer = & d_buffer[ tile_off ];

                // Copy data from tile buffer
                for( int i = 0; i < tile_vol; i++ ) {
                    A[i] = B[i] = buffer[i];
                }

                // Apply kernel locally
                for( int iy = gc.y.lower; iy < tile_dims.y + gc.y.lower; iy++ ) {
                    for( int ix = 0; ix < tile_ext_dims.x; ix ++) {
                        B [ iy * ystride + ix ] = A[ (iy-1) * ystride + ix ] * a +
                                                  A[    iy  * ystride + ix ] * b +
                                                  A[ (iy+1) * ystride + ix ] * c;
                    }
                }

                // Copy data back to tile buffer
                for( int i = 0; i < tile_vol; i++ ) buffer[i] = B[i];
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
     * @note Data can be converted to a different datatype
     * 
     * @tparam T2       Datatype to be used for output, defaults to the grid 
     *                  datatype. Must be supported by the ZDF library.
     * @param info      Grid metadata
     * @param iter      Iteration value
     * @param path      File path
     */
    template< typename T2 = T >
    void save( zdf::grid_info &info, const zdf::iteration &iter, const std::string & path ) {

        // Fill in global grid dimensions
        info.ndims = 2;
        info.count[0] = global_ntiles.x * tile_dims.x;
        info.count[1] = global_ntiles.y * tile_dims.y;

        // Allocate buffer on host to gather data
        T2 * h_data = memory::malloc<T2>( local_dims.x * local_dims.y );

        // Gather data on contiguous grid
        gather( h_data );

        // Information on local chunk of grid data
        zdf::chunk chunk;
        chunk.count[0] = local_dims.x;
        chunk.count[1] = local_dims.y;
        chunk.start[0] = local_tile_start.x * tile_dims.x;
        chunk.start[1] = local_tile_start.y * tile_dims.y;
        chunk.stride[0] = chunk.stride[1] = 1;
        chunk.data = (void *) h_data;

        // Save data
        zdf::save_grid<T2>( chunk, info, iter, path, part.get_comm() );

        // Free temporary buffer
        memory::free( h_data );
    };

    /**
     * @brief Save grid values to disk
     *
     * @note Data can be converted to a different datatype
     * 
     * @tparam T2       Datatype to be used for output, defaults to the grid 
     *                  datatype. Must be supported by the ZDF library.
     * @param filename  Output file name
     */
    template< typename T2 = T >
    void save( const std::string & filename ) {
        // Allocate buffer on host to gather data
        T2 * h_data = memory::malloc<T2>( local_dims.x * local_dims.y );

        // Gather data on contiguous grid
        gather( h_data );

        uint64_t global[2] = { global_ntiles.x * tile_dims.x, global_ntiles.y * tile_dims.y };
        uint64_t start[2]  = { local_tile_start.x * tile_dims.x, local_tile_start.y * tile_dims.y };
        uint64_t local[2]  = { local_dims.x, local_dims.y };

        zdf::save_grid( h_data, 2, global, start, local, name, filename, part.get_comm() );

        // Free temporary buffer
        memory::free( h_data );
    }

};

} // end of namespace grid
