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
struct flat_view {
    /// @brief Data buffer
    T * d_buffer;
    /// @brief Local grid size
    uint2 local_dims;
    /// @brief Start position of local grid on global grid
    uint2 local_start;
    /// @brief Global grid size
    uint2 global_dims;
};

/**
 * @brief Lambda deallocator for flat grid buffer
 * 
 * @tparam T 
 */
template< class T >
using flat_deleter = std::function< void( T * ) >;

/**
 * @brief Flat grid class with contiguous memory layout in an parallel partition
 * 
 * @tparam T    grid datatype
 */
template <class T>
class flat{

    protected:

    /// @brief Parallel partition
    const mpi::cart2d & part;

    /// @brief Local grid size
    uint2 local_dims;

    /// @brief Start position of local grid on global grid
    uint2 local_start;

    /// @brief Consider local boundaries periodic
    int2 local_periodic;

    /// @brief Data buffer
    T * d_buffer;

    /// @brief Allocated buffer size (in elements), >= buffer_size()
    std::size_t capacity = 0;

    /// @brief Release function for d_buffer
    flat_deleter<T> release;

    /// @brief Global grid size
    uint2 global_dims;

    public:

    /// @brief Object name
    std::string name = "flat_grid";
        
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
    flat( uint2 const global_dims, const mpi::cart2d & part, 
        uint2 const granularity = {1,1} ):
        part( part ),
        d_buffer( nullptr ), 
        global_dims( global_dims ) {
        
        if ( global_dims.x == 0 || global_dims.y == 0 ) {
            mpi::fatal( "Invalid global grid dimension " + to_string(global_dims) );
        }
        
        /// @brief global number of chunks
        auto global_chunks = global_dims / granularity;

        if ( global_chunks.x * granularity.x != global_dims.x || global_chunks.y * granularity.y != global_dims.y  ) {
            mpi::fatal(
                 "Invalid granularity " + to_string(granularity) +
                 ", the global_dims do not divide evenly by this value." );
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

        // In this situation total capacity equals buffer size
        capacity = buffer_size();

        // Allocate main data buffer
        d_buffer = gpu::device::malloc<T>( capacity );

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
    flat( flat && other ) noexcept :
        part( other.part ),
        local_dims( other.local_dims ),
        local_start( other.local_start ),
        local_periodic( other.local_periodic ),
        d_buffer( other.d_buffer ),
        capacity( other.capacity ),
        release( std::move( other.release ) ),
        global_dims( other.global_dims ),
        name( std::move( other.name ) ) {
            
        // Leave `other` in a destructible but empty state.
        other.d_buffer       = nullptr;
        other.capacity       = 0;
        other.release        = nullptr;
    }

    /**
     * @brief Construct a new flat object from pre-allocated buffer
     * 
     * @warning The routine does not validate if the grid is consistent across
     *          the parallel domain
     *
     * @param global_dims       Global grid dimensions
     * @param local_dims        Local grid dimensions
     * @param local_start       Start position of local grid
     * @param part              Parallel partition
     * @param source            Pre-allocated buffer
     * @param capacity          Capacity of pre-allocated buffer (number of elements)
     * @param release           Buffer release function (called when flat object is destroyed)
     *
     */
    flat( uint2 const global_dims, uint2 const local_dims, uint2 const local_start, const mpi::cart2d & part, 
        T * source, size_t capacity, flat_deleter<T> release ) :
        part( part ),
        local_dims ( local_dims ),
        local_start ( local_start ),
        d_buffer( source ),
        capacity( capacity ),
        release( std::move( release ) ),
        global_dims( global_dims )
        {

        // Get local periodic flag
        local_periodic.x = part.periodic.x && (part.dims.x == 1);
        local_periodic.y = part.periodic.y && (part.dims.y == 1);

        // Check source pointer
        if ( source == nullptr ) {
            mpi::fatal("Invalid source pointer");
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
     * @note Calls the buffer release function; for custom allocators this may
     *       be collective (e.g. NVSHMEM)
     */
    ~flat() {
        if ( d_buffer != nullptr && release ) release( d_buffer );
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
     * @brief Returns a view of the tiled grid
     * 
     * @return tiled_view<T> 
     */
    flat_view<T> view() noexcept {
        return { 
            d_buffer, 
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
    flat_view<const T> cview() const noexcept {
        return { 
            d_buffer, 
            local_dims,
            local_start,
            global_dims
        };
    }

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
    const mpi::cart2d & get_part() const noexcept { return  part; }

    /**
     * @brief Buffer size
     * 
     * @return total size of data buffers (in elements)
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
    friend std::ostream& operator<<(std::ostream& os, const flat<T>& obj) {
        return os << obj.name << " {"
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
        gpu::device::zero( d_buffer, buffer_size() );
    };

    /**
     * @brief Sets data to a constant value
     * 
     * @param val       Value
     */
    void set( T const & val ){
        gpu::device::set( d_buffer, buffer_size(), val );
    };

    /**
     * @brief Adds another grid object on top of local object
     * 
     * @param rhs         Other object to add
     */
    void add( const flat<T> &rhs ) {
        if ( rhs.local_dims != local_dims ) {
            std::ostringstream msg;
            msg << "add(): incompatible grid sizes (" << name << ": " << local_dims
                << " vs " << rhs.name << ": " << rhs.local_dims << ')';
            mpi::fatal(msg.str());
        }
        
        gpu::device::add( d_buffer, rhs.d_buffer, buffer_size() );
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
     * @brief Save grid values to disk
     * 
     * @param filename      Output file name (includes path)
     */
    void save( std::string filename ) {
        uint64_t global[2] = { global_dims.x, global_dims.y };
        uint64_t start[2]  = { local_start.x, local_start.y };
        uint64_t local[2]  = { local_dims.x, local_dims.y };

        T * h_buffer = gpu::host::malloc<T>( buffer_size() );
        gpu::device::memcpy_tohost( h_buffer, d_buffer, buffer_size() );

        zdf::save_grid( h_buffer, 2, global, start, local, name, filename, part.get_comm() );

        gpu::host::free( h_buffer );
    }

    /**
     * @brief Save grid values to disk with full metadata
     * 
     * @param info      Grid metadata
     * @param iter      Iteration value
     * @param path      File path
     */
    void save( zdf::grid_info &info, const zdf::iteration &iter, const std::string & path ) {
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
        gpu::device::memcpy_tohost( reinterpret_cast<T*>(chunk.data), d_buffer, buffer_size() );

        zdf::save_grid<T>( chunk, info, iter, path, part.get_comm() );

        gpu::host::free( chunk.data );
    }
};

}