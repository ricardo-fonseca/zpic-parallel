#pragma once

#include "../util/memory.hpp"
#include "../util/error.hpp"
#include "../core/vec_types.hpp"
#include "zdf.hpp"

#include <sstream>
#include <string>
#include <cstring>
#include <functional>

namespace grid {

/**
 * @brief Deallocator for flat grid buffer
 * 
 * @tparam T 
 */
template< class T >
using flat_deleter = std::function< void( T * ) >;

/**
 * @brief Flat grid class with contiguous memory layout
 * 
 * @tparam T    grid datatype
 */
template <class T>
class flat{

    protected:

    /// @brief Grid size
    uint2 grid_dims;

    /// @brief Data buffer
    T * d_buffer;

    /// @brief Allocated buffer size (in elements), >= buffer_size()
    std::size_t capacity = 0;

    /// @brief Release function for d_buffer
    flat_deleter<T> release;

    public:

    /// @brief Object name
    std::string name ="flat_grid";
        
    /**
     * @brief Construct a new flat grid object
     * 
     * @param dims          Grid dimensions
     */
    flat( uint2 const dims ):
        grid_dims( dims ),
        d_buffer( nullptr ) {
        
        if ( dims.x == 0 || dims.y == 0 ) {
            util::fatal( "Invalid grid dimensions " + to_string(dims) );
        }

        // In this situation total capacity equals buffer size
        capacity = buffer_size();

        // Allocate main data buffer
        d_buffer = memory::malloc<T>( capacity );

        // Set buffer release function
        release = []( T * ptr ) { memory::free( ptr ); };
    }

    /**
     * @brief Move constructor
     *
     * @note Transfers ownership of the data buffer from `other`. After the
     *       move, `other` is left in a valid but empty state: its pointer is
     *       null so its destructor is a no-op.
     *
     * @param other     Source grid (will be left empty)
     */
    flat( flat && other ) noexcept :
        grid_dims( other.grid_dims ),
        d_buffer( other.d_buffer ),
        capacity( other.capacity ),
        release( std::move( other.release ) ),
        name( std::move( other.name ) ) {
            
        // Leave `other` in a destructible but empty state.
        other.d_buffer       = nullptr;
        other.capacity       = 0;
        other.release        = nullptr;
    }

    /**
     * @brief Construct a new flat object from pre-allocated buffer
     * 
     * @note The capacity parameter allows using data buffers larger than
     *       grid_dims.y * grid_dims.x, so the buffer can be used with
     *       libraries requiring some additional space (in particular, FFTW).
     *       The grid takes ownership of the buffer.
     * 
     * @param dims              Grid dimensions
     * @param source            Pre-allocated buffer
     * @param capacity          Capacity of pre-allocated buffer (number of elements)
     * @param release           Buffer release function (called when flat object is destroyed)
     *
     */
    flat( uint2 const dims, T * source, size_t capacity, flat_deleter<T> release ) :
        grid_dims( dims ),
        d_buffer( source ),
        capacity( capacity ),
        release( std::move( release ) )
        {

        // Check source pointer
        if ( source == nullptr ) {
            util::fatal("Invalid source pointer");
        }

        // Check capacity
        if ( capacity < buffer_size() ) {
            util::fatal( "Provided capacity for flat<> grid is too small, "
                        "must be at least " + std::to_string(buffer_size()) + " elements");
        }
    }

    /**
     * @brief Destroy the basic grid object
     * 
     * @note Calls the buffer release function
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
     * @brief Get a pointer to the data buffer
     * 
     * @return T* 
     */
    T* data() const noexcept { return d_buffer; }

    /**
     * @brief Get the grid dimensions
     * 
     * @return uint2 
     */
    uint2 get_grid_dims() const noexcept { return grid_dims; }

    /**
     * @brief Buffer size
     * 
     * @return total size of data buffers (in elements)
     */
    std::size_t buffer_size() const noexcept {
        return static_cast<std::size_t>( grid_dims.y ) * grid_dims.x;
    };

    /**
     * @brief Stream extraction
     * 
     * @param os 
     * @param obj 
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, const flat<T>& obj) {
        return os << obj.name << '{' << obj.grid_dims << '}';
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
        size_t const size = buffer_size( );
        #pragma omp parallel for
        for( size_t i = 0; i < size; i++ ) {
            d_buffer[i] = val;
        }
    };

    /**
     * @brief Adds another grid object on top of local object
     * 
     * @param rhs         Other object to add
     */
    void add( const flat<T> &rhs ) {
        if ( rhs.grid_dims != grid_dims ) {
            std::ostringstream msg;
            msg << "add(): incompatible grid sizes (" << name << ": " << grid_dims
                      << " vs " << rhs.name << ": " << rhs.grid_dims << ')';
            util::fatal(msg.str());
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
     *       grid_dims.x * grid_dims.y size
     * 
     * @param tmp   Temporary buffer for transpose operation
     */
    void transpose( T * tmp ) {

        const auto nx = grid_dims.x;
        const auto ny = grid_dims.y;

        // Transpose data into temporary buffer
        // The loop order is optimized for the memory writes to be contiguous
        const T * __restrict__ data = d_buffer;
        T * __restrict__ out = tmp;

        #pragma omp parallel for
        for( unsigned ix = 0; ix < nx; ix++ ) {
            for( unsigned iy = 0; iy < ny; iy++ ) {
                out[ ix * ny + iy ] = data[ iy * nx + ix ];
            }
        }

        // Copy back to data buffer
        std::memcpy( (void *) d_buffer, (void *) tmp, buffer_size() * sizeof(T) );

        // Reshape grid
        grid_dims = make_uint2( ny, nx );
    }

    /**
     * @brief Transpose the data
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
        uint64_t dims[2] = { grid_dims.x, grid_dims.y };
        zdf::save_grid( d_buffer, 2, dims, name, filename );
    }

    /**
     * @brief Save grid values to disk with full metadata
     * 
     * @param info      Grid metadata
     * @param iter      Iteration value
     * @param path      File path
     */
    void save( zdf::grid_info &info, const zdf::iteration &iter, const std::string & path ) {
        // Fill in grid dimensions
        info.ndims = 2;
        info.count[0] = grid_dims.x;
        info.count[1] = grid_dims.y;

        zdf::save_grid<T>( d_buffer, info, iter, path );
    }
};

}
