#ifndef BASIC_GRID_H_
#define BASIC_GRID_H_

#include "parallel.h"

#include "vec_types.h"
#include "bnd.h"
#include "zdf-cpp.h"


// Tags are paired so that a message sent with dest::lower is received
// with source::upper (both have value 0). This ensures MPI tag matching
// between sender and receiver without extra bookkeeping.

namespace source {
    enum tag { lower = 0, upper = 1 };
}

namespace dest {
    enum tag { upper = 0, lower = 1 };
}

/**
 * @brief Basic grid class
 * 
 * @note Used as a contiguous 2D array in device memory
 * 
 * @tparam T    grid datatype
 */
template <class T>
class basic_grid{

    protected:

    /// @brief Parallel partition
    const Partition & part;

    /// @brief Local grid size
    uint2 local_dims;

    /// @brief Local grid size including guard cells
    uint2 local_ext_dims;

    /// @brief Local grid position on global grid
    uint2 local_pos;

    /// @brief Consider local boundaries periodic
    int2 local_periodic;

    /// @brief Local offset in cells between buffer[0] and position (0,0)
    unsigned int offset;

    /// @brief Buffers for sending messages
    pair< Message<T>* > msg_send;

    /// @brief Buffers for receiving messages
    pair< Message<T>* > msg_recv;

    public:

    /// @brief Data buffer   
    T * d_buffer;    

    /// @brief Global grid size
    const uint2 global_dims;

    /// @brief Tile guard cells
    const bnd<unsigned int> gc;

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
    basic_grid( uint2 const global_dims, bnd<unsigned int> const gc, const Partition & part, 
        uint2 const granularity = {1,1} ):
        part( part ),
        d_buffer( nullptr ), 
        global_dims( global_dims ),
        gc(gc) {
        
        if ( global_dims.x == 0 || global_dims.y == 0 ) {
            std::cerr << "Invalid global grid dimension " << global_dims << '\n';
            mpi::abort(1);
        }
        
        /// @brief global number of chunks
        auto global_chunks = global_dims / granularity;

        if ( global_chunks.x * granularity.x != global_dims.x || global_chunks.y * granularity.y != global_dims.y  ) {
            std::cerr << "Invalid granularity " << granularity 
                      << ", the global_dims do not divide evenly by this value \n";
            mpi::abort(1);
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
        local_pos  = local_chunk_offset * granularity;

        offset = gc.y.lower * local_ext_dims.x + gc.x.lower;

        // Get local periodic flag
        local_periodic.x = part.periodic.x && (part.dims.x == 1);
        local_periodic.y = part.periodic.y && (part.dims.y == 1);

        // Allocate main data buffer
        d_buffer = memory::malloc<T>( buffer_size() );

        // Get maximum message size
        int max_msg_size = max(
            ( local_ext_dims.y ) * max( gc.x.lower, gc.x.upper ),
            max( gc.y.lower, gc.y.upper ) * ( local_ext_dims.x )
        );

        // Allocate message buffers
        msg_recv.lower = new Message<T>( max_msg_size, part.get_comm() );
        msg_recv.upper = new Message<T>( max_msg_size, part.get_comm() );
        msg_send.lower = new Message<T>( max_msg_size, part.get_comm() );
        msg_send.upper = new Message<T>( max_msg_size, part.get_comm() );
    }

    /**
     * @brief Destroy the basic grid object
     * 
     */
    ~basic_grid() {

        delete msg_recv.lower;
        delete msg_recv.upper;
        delete msg_send.lower;
        delete msg_send.upper;

        memory::free( d_buffer );
    }

    /**
     * @brief Delete default copy constructor
     * 
     */
    basic_grid(const basic_grid&) = delete;

    /**
     * @brief Delete default copy constructor
     * 
     */
    basic_grid& operator=(const basic_grid&) = delete;

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
     * @brief Get the local pos object
     * 
     * @return uint2 
     */
    uint2 get_local_pos() const noexcept { return local_pos; }

    /**
     * @brief Get the offset object
     * 
     * @return unsigned int 
     */
    unsigned int get_offset() const noexcept { return offset; }

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
    friend std::ostream& operator<<(std::ostream& os, const basic_grid<T>& obj) {
        os << obj.name << '{'
           << "local: " << obj.local_dims
           << ", position: " << obj.local_pos
           << ", global: " << obj.global_dims
           << '}';
        return os;
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
    void add( const basic_grid<T> &rhs ) {
        if ( rhs.local_ext_dims != local_ext_dims ) {
            std::cerr << "add(): incompatible grid sizes (" << name << ": " << local_ext_dims
                      << " vs " << rhs.name << ": " << rhs.local_ext_dims << ")\n";
            mpi::abort(1);
        }
        
        size_t const size = buffer_size( );
        #pragma omp parallel for
        for( size_t i = 0; i < size; i++ ) d_buffer[i] += rhs.d_buffer[i];
    };

    /**
     * @brief Operator +=
     * 
     * @param rhs           Other grid to add
     * @return basic_grid<T>& 
     */
    basic_grid<T>& operator+=(const basic_grid<T>& rhs) {
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
            std::cerr << "x_shift_left(), invalid shift value, must be 0 < shift <= gc.x.upper\n";
            mpi::abort(1);
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
            std::cerr << "kernel_x3() requires at least 1 guard cell at both the lower and upper x boundaries.\n";
            mpi::abort(1);
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
            std::cerr << "kernel3_y() requires at least 1 guard cell at both the lower and upper y boundaries.\n";
            mpi::abort(1);
        }

    }

    /**
     * @brief Save grid values to disk
     * 
     * @param filename      Output file name (includes path)
     */
    void save( std::string filename ) {
        // Allocate buffer on host to gather data
        T * out = memory::malloc<T>( local_dims.x * local_dims.y );

        // Gather data on contiguous grid
        #pragma omp parallel for
        for( int iy = 0; iy < local_dims.y; iy++ ) {
            for( int ix = 0; ix < local_dims.x; ix++ ) {
                out[ iy * local_dims.x + ix ] = d_buffer[ offset + iy * local_ext_dims.x + ix ];
            }
        }

        uint64_t global[2] = { global_dims.x, global_dims.y };
        uint64_t start[2]  = { local_pos.x, local_pos.y };
        uint64_t local[2]  = { local_dims.x, local_dims.y };

        zdf::save_grid( out, 2, global, start, local, name, filename, part.get_comm() );

        // Free temporary buffer
        memory::free( out );
    }
};

#endif