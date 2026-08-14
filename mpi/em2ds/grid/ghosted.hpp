#pragma once

#include "../parallel.hpp"

#include "../vec_types.hpp"
#include "../bounds.hpp"
#include "../zdf/zdf.hpp"


namespace grid {

/**
 * @brief Basic grid class
 * 
 * @note Used as a contiguous 2D array in device memory
 * 
 * @tparam T    grid datatype
 */
template <class T>
class ghosted{

    protected:

    // Tags are paired so that a message sent with dest::lower is received
    // with source::upper (both have value 0). This ensures MPI tag matching
    // between sender and receiver without extra bookkeeping.

    /// @brief tags for outgoing messages
    struct source { enum tag { lower = 0, upper = 1 }; };
    /// @brief tags for incoming messages
    struct dest   { enum tag { upper = 0, lower = 1 }; };

    /// @brief Parallel partition
    const mpi::cart2d & part;

    /// @brief Local grid size
    uint2 local_dims;

    /// @brief Local grid size including guard cells
    uint2 local_ext_dims;

    /// @brief Start position of local grid on global grid
    uint2 local_start;

    /// @brief Consider local boundaries periodic
    int2 local_periodic;

    /// @brief Local offset in cells between buffer[0] and position (0,0)
    unsigned int offset;

    /// @brief Buffers for sending messages
    bounds< mpi::message<T>* > msg_send;

    /// @brief Buffers for receiving messages
    bounds< mpi::message<T>* > msg_recv;

    /// @brief Data buffer   
    T * d_buffer;    

    /// @brief Global grid size
    uint2 global_dims;

    /// @brief Tile guard cells
    bounds_2d<unsigned int> gc;

    public:

    /// @brief Object name
    std::string name ="unnamed_grid";
        
    /**
     * @brief Construct a new basic grid object
     * 
     * @note The granularity parameter controls how the the is split over multiple parallel domains.
     *       The local grid size will always be a multiple of this parameter.
     * 
     * @param global_dims   Global grid dimensions
     * @param gc            Number of guard cells
     * @param part          Parallel partition
     * @param granularity   Granularity for splitting grid across parallel nodes
     */
    ghosted( uint2 const global_dims, bounds_2d<unsigned int> const gc, const mpi::cart2d & part, 
        uint2 const granularity = {1,1} ):
        part( part ),
        d_buffer( nullptr ), 
        global_dims( global_dims ),
        gc(gc) {
        
        if ( global_dims.x == 0 || global_dims.y == 0 ) {
            mpi::fatal( "Invalid global grid dimensions: "  + to_string(global_dims) );
        }
        
        /// @brief global number of chunks
        auto global_chunks = global_dims / granularity;

        if ( global_chunks.x * granularity.x != global_dims.x || global_chunks.y * granularity.y != global_dims.y  ) {
            mpi::fatal( "Invalid granularity (" + to_string(granularity) +
                        "), the global_dims do not divide evenly by this value");
        }
        
        /// @brief local number of chunks
        uint2 local_chunks;
        
        /// @brief position offset of local chunks on global grid
        uint2 local_chunk_offset;

        // Get local number of chunks and local offset
        part.grid_local( global_chunks, local_chunks, local_chunk_offset );

        // Set local grid size and position on global grid
        local_dims = local_chunks * granularity;
        local_ext_dims = { gc.x.lower + local_dims.x + gc.x.upper,
                           gc.y.lower + local_dims.y + gc.y.upper };
        local_start  = local_chunk_offset * granularity;

        offset = gc.y.lower * local_ext_dims.x + gc.x.lower;

        // Get local periodic flag
        local_periodic.x = part.periodic.x && (part.dims.x == 1);
        local_periodic.y = part.periodic.y && (part.dims.y == 1);

        // Allocate main data buffer
        d_buffer = memory::malloc<T>( buffer_size() );

        // Get maximum message size
        int max_msg_size = std::max(
            ( local_ext_dims.y ) * std::max( gc.x.lower, gc.x.upper ),
            std::max( gc.y.lower, gc.y.upper ) * ( local_ext_dims.x )
        );

        // Allocate message buffers
        msg_recv.lower = new mpi::message<T>( max_msg_size, part.get_comm() );
        msg_recv.upper = new mpi::message<T>( max_msg_size, part.get_comm() );
        msg_send.lower = new mpi::message<T>( max_msg_size, part.get_comm() );
        msg_send.upper = new mpi::message<T>( max_msg_size, part.get_comm() );
    }

    /**
     * @brief Move constructor
     *
     * @note Transfers ownership of the data buffer and message buffers from
     *       `other`. After the move, `other` is left in a valid but empty
     *       state: its pointers are null so its destructor is a no-op.
     *
     * @param other     Source grid (will be left empty)
     */
    ghosted( ghosted && other ) noexcept :
        part( other.part ),
        local_dims( other.local_dims ),
        local_ext_dims( other.local_ext_dims ),
        local_start( other.local_start ),
        local_periodic( other.local_periodic ),
        offset( other.offset ),
        msg_send( other.msg_send ),
        msg_recv( other.msg_recv ),
        d_buffer( other.d_buffer ),
        global_dims( other.global_dims ),
        gc( other.gc ),
        name( std::move( other.name ) ) {
            
        // Leave `other` in a destructible but empty state.
        other.d_buffer       = nullptr;
        other.msg_send.lower = nullptr;
        other.msg_send.upper = nullptr;
        other.msg_recv.lower = nullptr;
        other.msg_recv.upper = nullptr;
    }

    ghosted( uint2 const global_dims, uint2 const local_dims_, uint2 const local_start_, const mpi::cart2d & part ) :
        part( part ),
        d_buffer( nullptr ), 
        global_dims( global_dims ),
        gc( bounds_2d<unsigned int>{0} ) {

        local_dims = local_dims_;
        local_ext_dims = local_dims;
        local_start = local_start_;
        offset = 0;

        // Get local periodic flag
        local_periodic.x = part.periodic.x && (part.dims.x == 1);
        local_periodic.y = part.periodic.y && (part.dims.y == 1);

        // Allocate main data buffer
        d_buffer = memory::malloc<T>( buffer_size() );

        // Messages are not required
        msg_send.lower = nullptr;
        msg_send.upper = nullptr;
        msg_recv.lower = nullptr;
        msg_recv.upper = nullptr;
    }

    /**
     * @brief Destroy the basic grid object
     * 
     */
    ~ghosted() {

        delete msg_recv.lower;
        delete msg_recv.upper;
        delete msg_send.lower;
        delete msg_send.upper;

        if ( d_buffer != nullptr ) memory::free( d_buffer );
    }

    /**
     * @brief Delete default copy constructor
     * 
     */
    ghosted(const ghosted&) = delete;

    /**
     * @brief Delete default copy constructor
     * 
     */
    ghosted& operator=(const ghosted&) = delete;

    /**
     * @brief Get a pointer to the data buffer
     * 
     * @return T* 
     */
    T* data() const noexcept { return d_buffer; }

    /**
     * @brief Get the global dims object
     * 
     * @return uint2 
     */
    uint2 get_global_dims() const noexcept { return global_dims; }

    /**
     * @brief Get the gc object
     * 
     * @return uint2 
     */
    bounds_2d<unsigned int> get_gc() const noexcept { return gc; }

    /**
     * @brief Get the local dims object
     * 
     * @return uint2 
     */
    uint2 get_local_dims() const noexcept { return local_dims; }

    /**
     * @brief Get the local ext dims object
     * 
     * @return uint2 
     */
    uint2 get_local_ext_dims() const noexcept { return local_ext_dims; }

    /**
     * @brief Get the position of the local grid on the global grid
     * 
     * @return uint2 
     */
    uint2 get_local_start() const noexcept { return local_start; }

    /**
     * @brief Get the offset object
     * 
     * @return unsigned int 
     */
    unsigned int get_offset() const noexcept { return offset; }

    /**
     * @brief Get the parallel topology
     * 
     * @return const mpi::cart2d& 
     */
    const mpi::cart2d & get_part() const noexcept { return part; }

    /**
     * @brief Buffer size
     * 
     * @return total size of data buffers (in elements)
     */
    std::size_t buffer_size() const noexcept {
        return local_ext_dims.y * local_ext_dims.x;
    };

    /**
     * @brief Stream extraction
     * 
     * @param os 
     * @param obj 
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, const ghosted<T>& obj) {
        return os << obj.name << '{'
           << "local: " << obj.local_dims
           << ", start: " << obj.local_start
           << ", global: " << obj.global_dims
           << '}';
    }

    /**
     * @brief zero device data on a grid grid
     * 
     */
    void zero( ) {
        memory::zero( d_buffer, buffer_size() );
    };

    /**
     * @brief Sets data to a constant value
     * 
     * @param val       Value
     */
    void set( T const & val ){
        #pragma omp parallel for
        for( size_t i = 0; i < buffer_size(); i++ ) {
            d_buffer[i] = val;
        }
    };

    /**
     * @brief Adds another grid object on top of local object
     * 
     * @param rhs         Other object to add
     */
    void add( const ghosted<T> &rhs ) {
        if ( rhs.local_ext_dims != local_ext_dims ) {
            mpi::fatal( 
                "add(): incompatible grid sizes (" + name + ": " + to_string(local_ext_dims) +
                " vs " + rhs.name + ": " + to_string(rhs.local_ext_dims) + ')' 
            );
        }
        
        size_t const size = buffer_size( );
        #pragma omp parallel for
        for( size_t i = 0; i < size; i++ ) d_buffer[i] += rhs.d_buffer[i];
    };

    /**
     * @brief Operator +=
     * 
     * @param rhs           Other grid to add
     * @return ghosted<T>& 
     */
    ghosted<T>& operator+=(const ghosted<T>& rhs) {
        add( rhs );
        return *this;
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
            T * __restrict__ msg = msg_send.lower-> buffer;
            for( unsigned j = 0; j < local_ext_dims.y; j++ ) {
                for( unsigned i = 0; i < gc.x.upper; i++ ) {
                    msg[ j * gc.x.upper + i ] = d_buffer[ j * local_ext_dims.x + gc.x.lower + i ];
                }
            }
            msg_send.lower->isend( local_ext_dims.y * gc.x.upper, lnode, dest::lower );
        }

        // Send message - upper neighbor
        if ( unode >= 0 ) {
            T * __restrict__ msg = msg_send.upper-> buffer;
            for( unsigned j = 0; j < local_ext_dims.y; j++ ) {
                for( unsigned i = 0; i < gc.x.lower; i++ ) {
                    msg[ j * gc.x.lower + i ] = d_buffer[  j * local_ext_dims.x + local_dims.x + i ];
                }
            }
            msg_send.upper->isend( local_ext_dims.y * gc.x.lower, unode, dest::upper );
        }

        // Process local parallel
        if ( local_periodic.x ) {
            for( unsigned j = 0; j < local_ext_dims.y; j++ ) {
                for( unsigned i = 0; i < gc.x.upper; i++ ) {
                    d_buffer[ j * local_ext_dims.x + gc.x.lower + local_dims.x + i ] = 
                        d_buffer[ j * local_ext_dims.x + gc.x.lower + i ];
                }
                for( unsigned i = 0; i < gc.x.lower; i++ ) {
                    d_buffer[ j * local_ext_dims.x + i ] = 
                        d_buffer[  j * local_ext_dims.x + local_dims.x + i ];
                }
            }
        }

        // Receive message - lower neighbor
        if ( lnode >= 0 ) {
            msg_recv.lower-> wait();
            T * __restrict__ msg = msg_recv.lower-> buffer;

            for( unsigned j = 0; j < local_ext_dims.y; j++ ) {
                for( unsigned i = 0; i < gc.x.lower; i++ ) {
                    d_buffer[ j * local_ext_dims.x + i ] = msg[ j * gc.x.lower + i ];
                }
            }
        }

        // Receive message - upper neighbor
        if ( unode >= 0 ) {
            msg_recv.upper-> wait();
            T * __restrict__ msg = msg_recv.upper -> buffer;

            for( unsigned j = 0; j < local_ext_dims.y; j++ ) {
                for( unsigned i = 0; i < gc.x.upper; i++ ) {
                    d_buffer[ j * local_ext_dims.x + gc.x.lower + local_dims.x + i ] = 
                        msg[ j * gc.x.upper + i ];
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
            T * __restrict__ msg = msg_send.lower-> buffer;
            for( unsigned j = 0; j < gc.y.upper; j++ ) {
                for( unsigned i = 0; i < local_ext_dims.x; i++ ) {
                    msg[ j * local_ext_dims.x + i ] = d_buffer[ ( gc.y.lower + j ) * local_ext_dims.x + i ];
                }
            }
            msg_send.lower->isend( gc.y.upper * local_ext_dims.x, lnode, dest::lower );
        }

        if ( unode >= 0 ) {
            T * __restrict__ msg = msg_send.upper -> buffer;
            for( unsigned j = 0; j < gc.y.lower; j++ ) {
                for( unsigned i = 0; i < local_ext_dims.x; i++ ) {
                    msg[ j * local_ext_dims.x + i ] = d_buffer[ ( local_dims.y + j ) * local_ext_dims.x + i ];
                }
            }
            msg_send.upper -> isend( gc.y.lower * local_ext_dims.x, unode, dest::upper );
        }

        // Process local parallel
        if ( local_periodic.y ) {
            for( unsigned j = 0; j < gc.y.lower; j++ ) {
                for( unsigned i = 0; i < local_ext_dims.x; i++ ) {
                    d_buffer[ j * local_ext_dims.x + i ] =  
                        d_buffer[ ( local_dims.y + j ) * local_ext_dims.x + i ];
                }
            }

            for( unsigned j = 0; j < gc.y.upper; j++ ) {
                for( unsigned i = 0; i < local_ext_dims.x; i++ ) {
                    d_buffer[ ( gc.y.lower + local_dims.y + j ) * local_ext_dims.x + i ] =  
                        d_buffer[ ( gc.y.lower + j ) * local_ext_dims.x + i ];
                }
            }
        }


        // Wait for receive messages to complete and copy data
        if ( lnode >= 0 ) {
            msg_recv.lower-> wait();
            T * __restrict__ msg = msg_recv.lower-> buffer;

            for( unsigned j = 0; j < gc.y.lower; j++ ) {
                for( unsigned i = 0; i < local_ext_dims.x; i++ ) {
                    d_buffer[ j * local_ext_dims.x + i ] =  
                        msg[ j * local_ext_dims.x + i ];
                }
            }
        }

        if ( unode >= 0 ) {
            msg_recv.upper-> wait();
            T * __restrict__ msg = msg_recv.upper-> buffer;

            for( unsigned j = 0; j < gc.y.upper; j++ ) {
                for( unsigned i = 0; i < local_ext_dims.x; i++ ) {
                    d_buffer[ ( gc.y.lower + local_dims.y + j ) * local_ext_dims.x + i ] =  
                        msg[ j * local_ext_dims.x + i ];
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

            T * __restrict__ msg = msg_send.lower-> buffer;

            for( int j = 0; j < local_ext_dims.y; j++ ) {
                for( int i = 0; i < gc.x.lower; i++ ) {
                    msg[ j * gc.x.lower + i ] = d_buffer[ j * local_ext_dims.x + i ];
                }
            }

            msg_send.lower->isend( local_ext_dims.y * gc.x.lower, lnode, dest::lower );
        }

        // Send message - upper neighbor
        if ( unode >= 0 ) {
            T * __restrict__ msg = msg_send.upper-> buffer;

            for( unsigned j = 0; j < local_ext_dims.y; j++ ) {
                for( unsigned i = 0; i < gc.x.upper; i++ ) {
                    msg[ j * gc.x.upper + i ] = d_buffer[ j * local_ext_dims.x + gc.x.lower + local_dims.x + i ];
                }
            }

            msg_send.upper->isend( local_ext_dims.y * gc.x.upper, unode, dest::upper );
        }

        // Process local periodic boundaries
        if ( local_periodic.x ) {
            for( unsigned j = 0; j < local_ext_dims.y ; j++ ) {
                for( unsigned i = 0; i < gc.x.upper; i++ ) {
                    d_buffer[ j * local_ext_dims.x + gc.x.lower + i ] += 
                        d_buffer[ j * local_ext_dims.x + gc.x.lower + local_dims.x + i ];
                }

                for( unsigned i = 0; i < gc.x.lower; i++ ) {
                    d_buffer[ j * local_ext_dims.x + local_dims.x + i ] += 
                        d_buffer[ j * local_ext_dims.x + i ];
                }
            }

        }

        // Receive message - lower neighbor
        if ( lnode >= 0 ) {
            msg_recv.lower-> wait();
            T * __restrict__ msg = msg_recv.lower-> buffer;

            for( unsigned j = 0; j < local_ext_dims.y ; j++ ) {
                for( unsigned i = 0; i < gc.x.upper; i++ ) {
                    d_buffer[ j * local_ext_dims.x + gc.x.lower + i ] += msg[ j * gc.x.upper + i ] ;
                }
            }
        }

        // Receive message - upper neighbor
        if ( unode >= 0 ) {
            msg_recv.upper-> wait();
            T * __restrict__ msg = msg_recv.upper-> buffer;

            for( unsigned j = 0; j < local_ext_dims.y; j++ ) {
                for( unsigned i = 0; i < gc.x.lower; i++ ) {
                    d_buffer[ j * local_ext_dims.x + local_dims.x + i ] += msg[ j * gc.x.lower + i ] ;
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
            T * __restrict__ msg = msg_send.lower-> buffer;
            for( unsigned j = 0; j < gc.y.lower; j++ ) {
                for( unsigned i = 0; i < local_ext_dims.x; i++ ) {
                    msg[ j * local_ext_dims.x + i ] = d_buffer[ j * local_ext_dims.x + i ];
                }
            }

            msg_send.lower->isend( gc.y.lower * local_ext_dims.x, lnode, dest::lower );
        }

        // Send message - upper neighbor
        if ( unode >= 0 ) {
            T * __restrict__ msg = msg_send.upper-> buffer;
            for( unsigned j = 0; j < gc.y.upper; j++ ) {
                for( unsigned i = 0; i < local_ext_dims.x; i++ ) {
                    msg[ j * local_ext_dims.x + i ] = 
                        d_buffer[ ( gc.y.lower + local_dims.y + j ) * local_ext_dims.x + i ];
                }
            }

            msg_send.upper->isend( gc.y.upper * local_ext_dims.x, unode, dest::upper );
        }

        // Process local parallel boundary
        if ( local_periodic.y ) {
            for( unsigned j = 0; j < gc.y.upper; j++ ) {
                for( unsigned i = 0; i < local_ext_dims.x; i++ ) {
                    d_buffer[ ( gc.y.lower + j ) * local_ext_dims.x + i ] += d_buffer[ ( gc.y.lower + local_dims.y + j ) * local_ext_dims.x + i ];
                }
            }

            for( unsigned j = 0; j < gc.y.lower; j++ ) {
                for( unsigned i = 0; i < local_ext_dims.x; i++ ) {
                    d_buffer[ ( local_dims.y + j ) * local_ext_dims.x + i ] +=  d_buffer[ j * local_ext_dims.x + i ];
                }
            }
        }

        // Receive message - lower neighbor
        if ( lnode >= 0 ) {
            msg_recv.lower-> wait();
            T * __restrict__ msg = msg_recv.lower-> buffer;

            for( unsigned j = 0; j < gc.y.upper; j++ ) {
                for( unsigned i = 0; i < local_ext_dims.x; i++ ) {
                    d_buffer[ ( gc.y.lower + j ) * local_ext_dims.x + i ] += msg[ j * local_ext_dims.x + i ];
                }
            }
        }

        // Receive message - upper neighbor
        if ( unode >= 0 ) {
            msg_recv.upper-> wait();
            T * __restrict__ msg = msg_recv.upper -> buffer;

            for( unsigned j = 0; j < gc.y.lower; j++ ) {
                for( unsigned i = 0; i < local_ext_dims.x; i++ ) {
                    d_buffer[ ( local_dims.y + j ) * local_ext_dims.x + i ] +=  msg[ j * local_ext_dims.x + i ];
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

        if ( shift > 0 && shift <= gc.x.upper ) {

            const int ystride = local_ext_dims.x;

            #pragma omp parallel for
            for( unsigned iy = 0; iy < local_ext_dims.y; iy++ ) {
                for( unsigned ix = 0; ix < local_ext_dims.x - shift; ix++ ) {
                    d_buffer[ ix + iy * ystride ] = d_buffer[ (ix + shift) + iy * ystride ]; 
                }
                for( unsigned ix = local_ext_dims.x - shift; ix < local_ext_dims.x; ix++ ) {
                    d_buffer[ ix + iy * ystride ] = T{0};
                }
            }

            // Copy x guard cells
            copy_to_gc_x();

        } else {
            mpi::fatal( "x_shift_left(), invalid shift value, must be 0 < shift <= gc.x.upper" );
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

            const int ystride = local_ext_dims.x;
            auto * __restrict__ data = & d_buffer[ offset ];

            #pragma omp parallel for
            for( int iy = 0; iy < static_cast<int>(local_dims.y); iy ++ ) {
                auto prev = data[ iy * ystride - 1 ];
                auto curr = data[ iy * ystride + 0 ];
                for( int ix = 0; ix < static_cast<int>(local_dims.x); ix ++ ) {
                    auto next = data[ iy * ystride + ix + 1 ];
                    data[ iy * ystride + ix ] = prev * a + curr * b + next * c;
                    prev = curr;
                    curr = next;
                }
            }

            // Update guard cells
            copy_to_gc();

        } else {
            mpi::fatal( "kernel_x3() requires at least 1 guard cell at both the lower and upper x boundaries." );
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

            const int ystride = local_ext_dims.x;
            auto * __restrict__ data = & d_buffer[ offset ];

            std::vector<T> prev_row( local_dims.x );
            std::vector<T> curr_row( local_dims.x );

            #pragma omp parallel for
            for( int ix = 0; ix < local_dims.x; ix++ ) {
                prev_row[ix] = data[ -1 * ystride + ix ];
                curr_row[ix] = data[  0 * ystride + ix ];
            }

            for( int iy = 0; iy < local_dims.y; iy++ ) {
                #pragma omp parallel for
                for( int ix = 0; ix < local_dims.x; ix++ ) {
                    T next = data[ (iy+1) * ystride + ix ];

                    data[ iy * ystride + ix ] = prev_row[ix] * a + curr_row[ix] * b + next * c; 

                    prev_row[ix] = curr_row[ix];
                    curr_row[ix] = next;
                }
            }

            copy_to_gc();

        } else {
            mpi::fatal( "kernel3_y() requires at least 1 guard cell at both the lower and upper y boundaries." );
        }

    }

    /**
     * @brief Transpose the grid
     * 
     * @note The operation requires a temporary buffer that must be at least
     *       local_dim.x * local_dim.y size
     * 
     * @param send_buffer   Temporary buffer for transpose operation
     */
    void transpose( T * send_buffer) {
        // Check parallel partition
        if ( part.dims.x != 1 ) {
            mpi::fatal( "only 1D parallel partitions along y are supported." );
        }

        if ( local_dims.x % part.dims.y != 0 ) {
            mpi::fatal( "The x dimension must divide evenly by the number of y parallel nodes." );
        }

        int2 block_dims = make_int2( local_dims.x / part.dims.y, local_dims.y );
        std::size_t block_size = static_cast<std::size_t> ( block_dims.x ) * block_dims.y;

        // Transpose data and pack send message buffer
        const T* __restrict__ data = &d_buffer[ offset ];
        for( int p = 0; p < part.dims.y; p++ ) {
            // The loop order is optimized for the memory writes to be contiguous
            for( int ix = 0; ix < block_dims.x; ix++ ) {
                for( int iy = 0; iy < block_dims.y; iy++ ) {
                    send_buffer[ p * block_size + ix * block_dims.y + iy ] = 
                        data[ iy * local_ext_dims.x + ( p * block_dims.x + ix ) ];
                }
            }
        }

        // Reshape grid - only grid parameters are modified, the data buffer remains unchanged
        local_start.y = (local_start.y * global_dims.x ) / global_dims.y;
        global_dims = { global_dims.y, global_dims.x };
        local_dims  = make_uint2( global_dims.x, block_dims.x );       
        std::swap( gc.x, gc.y );
        local_ext_dims = make_uint2(
            gc.x.lower + local_dims.x + gc.x.upper,
            gc.y.lower + local_dims.y + gc.y.upper
        );
        offset = gc.y.lower * local_ext_dims.x + gc.x.lower;

        // Prepare receive MPI type
        MPI_Datatype tmp_type, recv_type;
        MPI_Type_vector( block_dims.x, block_dims.y, local_ext_dims.x, mpi::data_type<T>(), &tmp_type);
        MPI_Type_create_resized( tmp_type, 0, block_dims.y * sizeof(T), &recv_type );
        MPI_Type_free( &tmp_type );
        MPI_Type_commit( &recv_type );
        
        // Exchange data and unpack
        auto * __restrict__ out_data = & d_buffer[ offset ];
        MPI_Alltoall( 
            send_buffer, block_size, mpi::data_type<T>(), 
            out_data, 1, recv_type, 
            part.get_comm()
        );

        // Free receive type
        MPI_Type_free( &recv_type );
    }

    /**
     * @brief Transpose the data
     * 
     * @note Guard cells are not correct after transpose, if required call
     *       copy_to_gc().
     * 
     */
    void transpose() {
        T* tmp = memory::malloc<T>( local_dims.x * local_dims.y );
        transpose( tmp );
        memory::free( tmp );
    }

    /**
     * @brief Save grid values to disk
     * 
     * @param filename      Output file name (includes path)
     */
    template< typename T2 = T >
    void save( const std::string & filename ) {
        // Allocate buffer on host to gather data
        T2 * out = memory::malloc<T2>( local_dims.x * local_dims.y );

        // Gather data on contiguous grid
        #pragma omp parallel for
        for( int iy = 0; iy < local_dims.y; iy++ ) {
            for( int ix = 0; ix < local_dims.x; ix++ ) {
                out[ iy * local_dims.x + ix ] = d_buffer[ offset + iy * local_ext_dims.x + ix ];
            }
        }

        uint64_t global[2] = { global_dims.x, global_dims.y };
        uint64_t start[2]  = { local_start.x, local_start.y };
        uint64_t local[2]  = { local_dims.x, local_dims.y };

        zdf::save_grid( out, 2, global, start, local, name, filename, part.get_comm() );

        // Free temporary buffer
        memory::free( out );
    }
};

}