#pragma once

#include "../utils.hpp"
#include "../parallel.hpp"
#include "../vec_types.hpp"
#include "../zdf-cpp.h"


namespace grid {

/**
 * @brief Flat grid class with contiguous memory layout in an parallel partition
 * 
 * @tparam T    grid datatype
 */
template <class T>
class flat{

    protected:

    /// @brief Parallel partition
    const Partition & part;

    /// @brief Local grid size
    uint2 local_dims;

    /// @brief Start position of local grid on global grid
    uint2 local_start;

    /// @brief Consider local boundaries periodic
    int2 local_periodic;

    /// @brief Data buffer
    T * d_buffer;

    /// @brief Global grid size
    uint2 global_dims;

    public:

    /// @brief Object name
    std::string name ="flat_grid";
        
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
    flat( uint2 const global_dims, const Partition & part, 
        uint2 const granularity = {1,1} ):
        part( part ),
        d_buffer( nullptr ), 
        global_dims( global_dims ) {
        
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
        local_start = local_chunk_offset * granularity;

        // Get local periodic flag
        local_periodic.x = part.periodic.x && (part.dims.x == 1);
        local_periodic.y = part.periodic.y && (part.dims.y == 1);

        // Allocate main data buffer
        d_buffer = memory::malloc<T>( buffer_size() );
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
    flat( flat && other ) noexcept :
        part( other.part ),
        local_dims( other.local_dims ),
        local_start( other.local_start ),
        local_periodic( other.local_periodic ),
        d_buffer( other.d_buffer ),
        global_dims( other.global_dims ),
        name( std::move( other.name ) ) {
            
        // Leave `other` in a destructible but empty state.
        other.d_buffer       = nullptr;
    }

    /**
     * @brief Construct a new flat3 grid with specified local information
     *
     * @warning The routine does not validate if the  grid is consistent across
     *          the parallel domain
     *
     * @note 
     * The local_size parameter allows allocating data buffers larger than
     * local_dims.y * local_dims.x, so the buffer can be used with libraries
     * requiring some additional space (in particular, FFTW)
     * 
     * @param global_dims       Global grid dimensions
     * @param local_dims_       Local grid dimensions
     * @param local_start_      Start position of local grid
     * @param part              Parallel partition
     * @param local_size        (optional) Size (in elements) to use for data
     *                          buffers, must be larger than local_dims.y * local_dims.x
     */
    flat( uint2 const global_dims, uint2 const local_dims, uint2 const local_start, const Partition & part, 
        size_t local_size = 0 ) :
        part( part ),
        local_dims ( local_dims ),
        local_start ( local_start ),
        d_buffer( nullptr ),
        global_dims( global_dims )
        {

        // Get local periodic flag
        local_periodic.x = part.periodic.x && (part.dims.x == 1);
        local_periodic.y = part.periodic.y && (part.dims.y == 1);

        // If local_size was not specified default to standard buffer size
        if ( local_size == 0 ) local_size = buffer_size();

        // If local_size was specified, verify that it is large enough
        if ( local_size < buffer_size() ) {
            std::cerr << "Requested local_size for flat<> grid is too small, "
                         "must be at least " << buffer_size() << '\n';
            mpi::abort(1);
        }

        // Allocate main data buffer
        d_buffer = memory::malloc<T>(local_size );
    }

    /**
     * @brief Destroy the basic grid object
     * 
     */
    ~flat() {
        if ( d_buffer != nullptr ) memory::free( d_buffer );
    }

    /**
     * @brief Delete default copy constructor
     * 
     */
    flat(const flat&) = delete;

    /**
     * @brief Delete default copy constructor
     * 
     */
    flat& operator=(const flat&) = delete;

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
     * @brief Get the local dims object
     * 
     * @return uint2 
     */
    uint2 get_local_dims() const noexcept { return local_dims; }

    /**
     * @brief Get the local pos object
     * 
     * @return uint2 
     */
    uint2 get_local_start() const noexcept { return local_start; }

    /**
     * @brief Get the part object
     * 
     * @return const Partition& 
     */
    const Partition & get_part() const noexcept { return  part; }

    /**
     * @brief Buffer size
     * 
     * @return total size of data buffers (in elements)
     */
    std::size_t buffer_size() const noexcept {
        return local_dims.y * local_dims.x;
    };

    /**
     * @brief Stream extraction
     * 
     * @param os 
     * @param obj 
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, const flat<T>& obj) {
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
    void add( const flat<T> &rhs ) {
        if ( rhs.local_dims != local_dims ) {
            std::cerr << "add(): incompatible grid sizes (" << name << ": " << local_dims
                      << " vs " << rhs.name << ": " << rhs.local_dims << ")\n";
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
     * @return flat<T>& 
     */
    flat<T>& operator+=(const flat<T>& rhs) {
        add( rhs );
        return *this;
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
            std::cerr << "only 1D parallel partitions along y are supported\n";
            mpi::abort(1);
        }

        if ( local_dims.x % part.dims.y != 0 ) {
            std::cerr << "The x dimension must divide evenly by the number of y parallel nodes \n";
            mpi::abort(1);
        }

        int2 block_dims = make_int2( local_dims.x / part.dims.y, local_dims.y );
        std::size_t block_size = static_cast<std::size_t> ( block_dims.x ) * block_dims.y;

        // Transpose data and pack send message buffer
        const T* __restrict__ data = &d_buffer[ 0 ];
        for( int p = 0; p < part.dims.y; p++ ) {
            // The loop order is optimized for the memory writes to be contiguous
            for( int ix = 0; ix < block_dims.x; ix++ ) {
                for( int iy = 0; iy < block_dims.y; iy++ ) {
                    send_buffer[ p * block_size + ix * block_dims.y + iy ] = 
                        data[ iy * local_dims.x + ( p * block_dims.x + ix ) ];
                }
            }
        }

        // Reshape grid - only grid parameters are modified, the data buffer remains unchanged
        local_start.y = (local_start.y * global_dims.x ) / global_dims.y;
        global_dims = { global_dims.y, global_dims.x };
        local_dims  = make_uint2( global_dims.x, block_dims.x );       

        // Prepare receive MPI type
        MPI_Datatype tmp_type, recv_type;
        MPI_Type_vector( block_dims.x, block_dims.y, local_dims.x, mpi::data_type<T>(), &tmp_type);
        MPI_Type_create_resized( tmp_type, 0, block_dims.y * sizeof(T), &recv_type );
        MPI_Type_free( &tmp_type );
        MPI_Type_commit( &recv_type );
        
        // Exchange data and unpack
        auto * __restrict__ out_data = & d_buffer[ 0 ];
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
        T* tmp = memory::malloc<T>( buffer_size() );
        transpose( tmp );
        memory::free( tmp );
    }

    /**
     * @brief Save grid values to disk
     * 
     * @param filename      Output file name (includes path)
     */
    void save( std::string filename ) {
        uint64_t global[2] = { global_dims.x, global_dims.y };
        uint64_t start[2]  = { local_start.x, local_start.y };
        uint64_t local[2]  = { local_dims.x, local_dims.y };

        zdf::save_grid( d_buffer, 2, global, start, local, name, filename, part.get_comm() );
    }
};

}