#pragma once

#include "../utils.hpp"
#include "../parallel.hpp"
#include "../vec_types.hpp"
#include "../zdf/zdf.hpp"

// Proveides fcomp::cart
#include "tiled_vec3.hpp"
#include <sstream>
#include <string>

namespace grid {

/**
 * @brief Flat grid class with contiguous memory layout with 3 different buffers
 * 
 * @tparam T    grid datatype
 */
template <class T>
class flat3{

    protected:

    /// @brief Parallel partition
    const mpi::cart2d & part;

    /// @brief Local grid size
    uint2 local_dims;

    /// @brief Local grid position on global grid
    uint2 local_start;

    /// @brief Consider local boundaries periodic
    int2 local_periodic;

    /// @brief x data buffer
    T * x_buffer = nullptr;

    /// @brief x data buffer
    T * y_buffer = nullptr;

    /// @brief x data buffer
    T * z_buffer = nullptr;

    /// @brief Global grid size
    uint2 global_dims;

    public:

    /// @brief Object name
    std::string name ="flat3_grid";
        
    /**
     * @brief Construct a new flat3 grid object
     * 
     * @note 
     * The granularity parameter controls how the the is split over multiple parallel domains.
     * The local grid size will always be a multiple of this parameter.
     * 
     * @param global_dims   Global grid dimensions
     * @param gc            Number of guard cells
     * @param part          Parallel partition
     * @param granularity   Granularity for splitting grid across parallel nodes
     */
    flat3( uint2 const global_dims, const mpi::cart2d & part, 
        uint2 const granularity = {1,1} ):
        part( part ),
        x_buffer( nullptr ), y_buffer( nullptr ), z_buffer( nullptr ), 
        global_dims( global_dims ) {
        
        if ( global_dims.x == 0 || global_dims.y == 0 ) {
            mpi::fatal( "Invalid global grid dimensions: " + to_string(global_dims) );
        }
        
        /// @brief global number of chunks
        auto global_chunks = global_dims / granularity;

        if ( global_chunks.x * granularity.x != global_dims.x || global_chunks.y * granularity.y != global_dims.y  ) {
            mpi::fatal( "Invalid granularity (" + to_string(granularity) + 
                        "), the global_dims do not divide evenly by this value" );
        }
        
        /// @brief local number of chunks
        uint2 local_chunks;
        
        /// @brief position offset of local chunks on global grid
        uint2 local_chunk_offset;

        // Get local number of chunks and local offset
        part.grid_local( global_chunks, local_chunks, local_chunk_offset );

        // Set local grid size and position on global grid
        local_dims = local_chunks * granularity;
        local_start  = local_chunk_offset * granularity;

        // Get local periodic flag
        local_periodic.x = part.periodic.x && (part.dims.x == 1);
        local_periodic.y = part.periodic.y && (part.dims.y == 1);

        // Allocate main data buffer
        x_buffer = memory::malloc<T>( buffer_size() );
        y_buffer = memory::malloc<T>( buffer_size() );
        z_buffer = memory::malloc<T>( buffer_size() );
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
    flat3( flat3 && other ) noexcept :
        part( other.part ),
        local_dims( other.local_dims ),
        local_start( other.local_start ),
        local_periodic( other.local_periodic ),
        x_buffer( other.x_buffer ), y_buffer( other.y_buffer ), z_buffer( other.z_buffer ),
        global_dims( other.global_dims ),
        name( std::move( other.name ) ) {
            
        // Leave `other` in a destructible but empty state.
        other.x_buffer       = nullptr;
        other.y_buffer       = nullptr;
        other.z_buffer       = nullptr;
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
    flat3( uint2 const global_dims, uint2 const local_dims, uint2 const local_start, const mpi::cart2d & part, 
        size_t local_size = 0 ) :
        part( part ),
        local_dims ( local_dims ),
        local_start ( local_start ),
        x_buffer( nullptr ), y_buffer( nullptr ), z_buffer( nullptr ), 
        global_dims( global_dims )
        {

        // Get local periodic flag
        local_periodic.x = part.periodic.x && (part.dims.x == 1);
        local_periodic.y = part.periodic.y && (part.dims.y == 1);

        // If local_size was not specified default to standard buffer size
        if ( local_size == 0 ) local_size = buffer_size();

        // If local_size was specified, verify that it is large enough
        if ( local_size < buffer_size() ) {
            mpi::fatal( "Requested local_size for flat3<> grid is too small, "
                         "must be at least " + std::to_string(buffer_size()) );
        }

        // Allocate main data buffers using local_size
        x_buffer = memory::malloc<T>( local_size );
        y_buffer = memory::malloc<T>( local_size );
        z_buffer = memory::malloc<T>( local_size );
    }

    /**
     * @brief Destroy the basic grid object
     * 
     */
    ~flat3() {
        if ( x_buffer != nullptr ) memory::free( x_buffer );
        if ( y_buffer != nullptr ) memory::free( y_buffer );
        if ( z_buffer != nullptr ) memory::free( z_buffer );
    }

    /**
     * @brief Delete default copy constructor
     * 
     */
    flat3(const flat3 &) = delete;

    /**
     * @brief Delete default copy constructor
     * 
     */
    flat3& operator=(const flat3 &) = delete;

    /**
     * @brief Get a pointer to the x data buffer
     * 
     * @return T* 
     */
    T* x() const noexcept { return x_buffer; }

    /**
     * @brief Get a pointer to the y data buffer
     * 
     * @return T* 
     */
    T* y() const noexcept { return y_buffer; }

    /**
     * @brief Get a pointer to the z data buffer
     * 
     * @return T* 
     */
    T* z() const noexcept { return z_buffer; }

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
    const mpi::cart2d & get_part() const noexcept { return  part; }

    /**
     * @brief Buffer size
     * 
     * @return total size of each data buffer (in elements)
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
    friend std::ostream& operator<<(std::ostream& os, const flat3 & obj) {
        os << obj.name << '{'
           << "local: " << obj.local_dims
           << ", position: " << obj.local_start
           << ", global: " << obj.global_dims
           << '}';
        return os;
    }

    /**
     * @brief zero device data on a grid grid
     * 
     */
    void zero( ) {
        memory::zero( x_buffer, buffer_size() );
        memory::zero( y_buffer, buffer_size() );
        memory::zero( z_buffer, buffer_size() );
    };

    /**
     * @brief Sets data to a constant value
     * 
     * @param val       Value
     */
    void set( T const & val_x, T const & val_y, T const & val_z  ){
        
        size_t const size = buffer_size( );
        #pragma omp parallel for
        for( int i = 0; i < size; i++ ) {
            x_buffer[i] = val_x;
            y_buffer[i] = val_y;
            z_buffer[i] = val_z;
        }
    };

    /**
     * @brief Adds another grid object on top of local object
     * 
     * @param rhs         Other object to add
     */
    void add( const flat3<T> &rhs ) {
        if ( rhs.local_dims != local_dims ) {
            std::ostringstream msg;
            msg << "add(): incompatible grid sizes (" << name << ": " << local_dims
                      << " vs " << rhs.name << ": " << rhs.local_dims << ')';
            mpi::fatal(msg.str());
        }

        size_t const size = buffer_size( );
        #pragma omp parallel for
        for( int i = 0; i < size; i++ ) {
            x_buffer[i] += rhs.x_buffer[i];
            y_buffer[i] += rhs.y_buffer[i];
            z_buffer[i] += rhs.z_buffer[i];
        }
    };

    /**
     * @brief Operator +=
     * 
     * @param rhs           Other grid to add
     * @return flat<T>& 
     */
    flat3 & operator+=(const flat3<T> & rhs) {
        add( rhs );
        return *this;
    }

#if 0
    /**
     * @brief Transpose the grid
     * 
     * @note The operation requires a temporary buffer that must be at least
     *       local_dim.x * local_dim.y size
     * 
     * @param send_buffer   Temporary buffer for transpose operation
     */
    void transpose( T * send_buffer) {
        
        NOT IMPLEMENTED YET!

        // Check parallel partition
        if ( part.dims.x != 1 ) {
            mpi::fatal( "only 1D parallel partitions along y are supported" );
        }

        if ( local_dims.x % part.dims.y != 0 ) {
            mpi:.fatal( "The x dimension must divide evenly by the number of y parallel nodes" );
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
#endif

    /**
     * @brief Save grid values to disk
     * 
     * @param filename      Output file name (includes path)
     */
    void save( fcomp::cart fc, std::string filename ) {
        uint64_t global[2] = { global_dims.x, global_dims.y };
        uint64_t start[2]  = { local_start.x, local_start.y };
        uint64_t local[2]  = { local_dims.x, local_dims.y };

        switch( fc ) {
        case fcomp::cart::z : 
            zdf::save_grid( z_buffer, 2, global, start, local, name + "-z", filename, part.get_comm() );
            break;
        case fcomp::cart::y :
            zdf::save_grid( y_buffer, 2, global, start, local, name + "-y", filename, part.get_comm() );
            break;
        case fcomp::cart::x :
            zdf::save_grid( x_buffer, 2, global, start, local, name + "-x", filename, part.get_comm() );
            break;
        default:
            mpi::fatal( "flat3::save() - Invalid fc" );
        }
    }

    /**
     * @brief Save grid values to disk with full metadata
     * 
     * @param fc        Field component to save
     * @param info      Grid metadata
     * @param iter      Iteration value
     * @param path      File path
     */
    void save( fcomp::cart fc, zdf::grid_info &info, const zdf::iteration &iter, const std::string & path ) {
        // Fill in global grid dimensions
        info.ndims = 2;
        info.count[0] = global_dims.x;
        info.count[1] = global_dims.y;

        // Information on local chunk of grid data
        zdf::chunk chunk;
        chunk.count[0] = local_dims.x;
        chunk.count[1] = local_dims.y;
        chunk.start[0] = local_start.x;
        chunk.start[1] = local_start.y;
        chunk.stride[0] = chunk.stride[1] = 1;

        switch( fc ) {
        case fcomp::cart::z : 
            chunk.data = (void *) z_buffer;
            break;
        case fcomp::cart::y :
            chunk.data = (void *) y_buffer;
            break;
        case fcomp::cart::x :
            chunk.data = (void *) x_buffer;
            break;
        default:
            mpi::fatal( "flat3::save() - Invalid fc" );
        }

        zdf::save_grid<T>( chunk, info, iter, path, part.get_comm() );
    }


};

}