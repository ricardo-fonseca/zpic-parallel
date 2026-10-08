#pragma once

#include "../util/memory.cuh"
#include "../parallel/partition.hpp"
#include "../core/vec_types.cuh"
#include "../zdf/zdf.hpp"

#include <sstream>
#include <string>
#include <functional>

namespace grid {

template<class T>
struct flat3_view {
    /// @brief Data buffers
    T * x_buffer, * y_buffer, * z_buffer;
    /// @brief Local grid size
    uint2 local_dims;
    /// @brief Start position of local grid on global grid
    uint2 local_start;
    /// @brief Global grid size
    uint2 global_dims;
};

template< class T >
using flat3_deleter = std::function< void( T * ) >;

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

    /// @brief y data buffer
    T * y_buffer = nullptr;

    /// @brief z data buffer
    T * z_buffer = nullptr;

    /// @brief Allocated buffer size (for each buffer, in elements), >= buffer_size()
    std::size_t capacity = 0;

    /// @brief Release function for d_buffer
    flat3_deleter<T> release;

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

        // In this situation total capacity equals buffer size
        capacity = buffer_size();

        // Allocate main data buffers
        x_buffer = gpu::device::malloc<T>( capacity );
        y_buffer = gpu::device::malloc<T>( capacity );
        z_buffer = gpu::device::malloc<T>( capacity );

        // Set buffer release function
        release = []( T * ptr ) { gpu::device::free( ptr ); };
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
        capacity( other.capacity ),
        release( std::move( other.release ) ),
        global_dims( other.global_dims ),
        name( std::move( other.name ) ) {
            
        // Leave `other` in a destructible but empty state.
        other.x_buffer       = nullptr;
        other.y_buffer       = nullptr;
        other.z_buffer       = nullptr;

        other.capacity       = 0;
        other.release        = nullptr;
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
        T * source_x, T * source_y, T * source_z, size_t capacity, flat3_deleter<T> release ) :
        part( part ),
        local_dims ( local_dims ),
        local_start ( local_start ),
        x_buffer( source_x ), y_buffer( source_y ), z_buffer( source_z ), 
        capacity( capacity ),
        release( std::move( release ) ),
        global_dims( global_dims )
        {

        // Get local periodic flag
        local_periodic.x = part.periodic.x && (part.dims.x == 1);
        local_periodic.y = part.periodic.y && (part.dims.y == 1);

        // Check source pointer
        if ( source_x == nullptr || source_y == nullptr || source_z == nullptr ) {
            mpi::fatal("Invalid source pointer(s)");
        }

        // Check capacity
        if ( capacity < buffer_size() ) {
            mpi::fatal( "Provided capacity for flat<> grid is too small, "
                        "must be at least " + std::to_string(buffer_size()) + " elements");
        }
    }

    /**
     * @brief Destroy the basic grid object
     * 
     */
    ~flat3() {
        if ( x_buffer != nullptr && release ) release( x_buffer );
        if ( y_buffer != nullptr && release ) release( y_buffer );
        if ( z_buffer != nullptr && release ) release( z_buffer );
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
     * @brief Returns a view of the tiled grid
     * 
     * @return tiled_view<T> 
     */
    flat3_view<T> view() noexcept {
        return { 
            x_buffer, y_buffer, z_buffer,
            local_dims,
            local_start,
            global_dims
        };
    }

    /**
     * @brief Returns a read-only view of the tiled grid
     * 
     * @note const qualified so that a const tiled grid can still hand a view
     *       to a read-only kernel. view() stays non-const: handing out a
     *       mutable view is a mutating operation on the grid.
     * 
     * @return tiled_view<const T> 
     */
    flat3_view<const T> cview() const noexcept {
        return { 
            x_buffer, y_buffer, z_buffer,
            local_dims,
            local_start,
            global_dims
        };
    }

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
        return static_cast<std::size_t>( local_dims.y ) * local_dims.x;
    };

    /**
     * @brief Stream extraction
     * 
     * @param os 
     * @param obj 
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, const flat3 & obj) {
        os << obj.name << " {"
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

        // These could be done in different streams
        gpu::device::zero( x_buffer, buffer_size() );
        gpu::device::zero( y_buffer, buffer_size() );
        gpu::device::zero( z_buffer, buffer_size() );
    };

    /**
     * @brief Sets data to a constant value
     * 
     * @param val       Value
     */
    void set( T const & val_x, T const & val_y, T const & val_z  ){
        
        size_t const size = buffer_size( );
        
        // These could be done in different streams
        gpu::device::set( x_buffer, size, val_x );
        gpu::device::set( y_buffer, size, val_y );
        gpu::device::set( z_buffer, size, val_z );
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

        // These could be done in different streams
        gpu::device::add( x_buffer, rhs.x_buffer, buffer_size() );
        gpu::device::add( y_buffer, rhs.y_buffer, buffer_size() );
        gpu::device::add( z_buffer, rhs.z_buffer, buffer_size() );
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

    /**
     * @brief Save grid values to disk
     * 
     * @param filename      Output file name (includes path)
     */
    void save( fcomp::cart fc, std::string filename ) {
        uint64_t global[2] = { global_dims.x, global_dims.y };
        uint64_t start[2]  = { local_start.x, local_start.y };
        uint64_t local[2]  = { local_dims.x, local_dims.y };

        std::string lname = name;

        T * h_buffer = gpu::host::malloc<T>( buffer_size() );

        switch( fc ) {
        case fcomp::cart::z : 
            gpu::device::memcpy_tohost( h_buffer, z_buffer, buffer_size() );
            lname += "-z";
            break;
        case fcomp::cart::y :
            gpu::device::memcpy_tohost( h_buffer, y_buffer, buffer_size() );
            lname += "-y";
            break;
        case fcomp::cart::x :
            gpu::device::memcpy_tohost( h_buffer, x_buffer, buffer_size() );
            lname += "-x";
            break;
        default:
            mpi::fatal( "flat3::save() - Invalid fc" );
        }

        zdf::save_grid( h_buffer, 2, global, start, local, lname, filename, part.get_comm() );
        gpu::host::free( h_buffer );
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

        chunk.data = gpu::host::malloc<T>( buffer_size() );

        switch( fc ) {
        case fcomp::cart::z : 
            gpu::device::memcpy_tohost( reinterpret_cast<T*>(chunk.data), z_buffer, buffer_size() );
            break;
        case fcomp::cart::y :
            gpu::device::memcpy_tohost( reinterpret_cast<T*>(chunk.data), y_buffer, buffer_size() );
            break;
        case fcomp::cart::x :
            gpu::device::memcpy_tohost( reinterpret_cast<T*>(chunk.data), x_buffer, buffer_size() );
            break;
        default:
            mpi::fatal( "flat3::save() - Invalid fc" );
        }

        zdf::save_grid<T>( chunk, info, iter, path, part.get_comm() );

        gpu::host::free( chunk.data );
    }


};

}