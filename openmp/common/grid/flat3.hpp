#pragma once

#include "../util/memory.hpp"
#include "../util/error.hpp"
#include "../core/vec_types.hpp"
#include "../core/dims.hpp"
#include "zdf.hpp"

#include <sstream>
#include <string>
#include <functional>

namespace grid {

/**
 * @brief Deallocator for flat3 grid buffers
 * 
 * @tparam T 
 */
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

    /// @brief Grid size
    uint2 grid_dims;

    /// @brief x data buffer
    T * x_buffer = nullptr;

    /// @brief y data buffer
    T * y_buffer = nullptr;

    /// @brief z data buffer
    T * z_buffer = nullptr;

    /// @brief Allocated buffer size (for each buffer, in elements), >= buffer_size()
    std::size_t capacity = 0;

    /// @brief Release function for data buffers
    flat3_deleter<T> release;

    /**
     * @brief Returns the data buffer for the selected component
     * 
     * @param fc    Field component
     * @return T* 
     */
    T * component( fcomp::cart fc ) const {
        switch( fc ) {
        case fcomp::cart::x : return x_buffer;
        case fcomp::cart::y : return y_buffer;
        case fcomp::cart::z : return z_buffer;
        }
        util::fatal( "flat3 - Invalid fc" );
    }

    public:

    /// @brief Object name
    std::string name = "flat3_grid";
        
    /**
     * @brief Construct a new flat3 grid object
     * 
     * @param dims          Grid dimensions
     */
    flat3( uint2 const dims ):
        grid_dims( dims ),
        x_buffer( nullptr ), y_buffer( nullptr ), z_buffer( nullptr ) {
        
        if ( dims.x == 0 || dims.y == 0 ) {
            util::fatal( "Invalid grid dimensions: " + to_string(dims) );
        }

        // In this situation total capacity equals buffer size
        capacity = buffer_size();

        // Allocate main data buffers
        x_buffer = memory::malloc<T>( capacity );
        y_buffer = memory::malloc<T>( capacity );
        z_buffer = memory::malloc<T>( capacity );

        // Set buffer release function
        release = []( T * ptr ) { memory::free( ptr ); };
    }

    /**
     * @brief Move constructor
     *
     * @note Transfers ownership of the data buffers from `other`. After the
     *       move, `other` is left in a valid but empty state: its pointers are
     *       null so its destructor is a no-op.
     *
     * @param other     Source grid (will be left empty)
     */
    flat3( flat3 && other ) noexcept :
        grid_dims( other.grid_dims ),
        x_buffer( other.x_buffer ), y_buffer( other.y_buffer ), z_buffer( other.z_buffer ),
        capacity( other.capacity ),
        release( std::move( other.release ) ),
        name( std::move( other.name ) ) {
            
        // Leave `other` in a destructible but empty state.
        other.x_buffer       = nullptr;
        other.y_buffer       = nullptr;
        other.z_buffer       = nullptr;

        other.capacity       = 0;
        other.release        = nullptr;
    }

    /**
     * @brief Construct a new flat3 object from pre-allocated buffers
     *
     * @note The capacity parameter allows using data buffers larger than
     *       grid_dims.y * grid_dims.x, so the buffers can be used with
     *       libraries requiring some additional space (in particular, FFTW).
     *       The grid takes ownership of the buffers.
     * 
     * @param dims              Grid dimensions
     * @param source_x          Pre-allocated buffer (x component)
     * @param source_y          Pre-allocated buffer (y component)
     * @param source_z          Pre-allocated buffer (z component)
     * @param capacity          Capacity of each pre-allocated buffer (number of elements)
     * @param release           Buffer release function (called for each buffer when
     *                          the flat3 object is destroyed)
     */
    flat3( uint2 const dims, T * source_x, T * source_y, T * source_z, 
        size_t capacity, flat3_deleter<T> release ) :
        grid_dims( dims ),
        x_buffer( source_x ), y_buffer( source_y ), z_buffer( source_z ), 
        capacity( capacity ),
        release( std::move( release ) )
        {

        // Check source pointers
        if ( source_x == nullptr || source_y == nullptr || source_z == nullptr ) {
            util::fatal("Invalid source pointer(s)");
        }

        // Check capacity
        if ( capacity < buffer_size() ) {
            util::fatal( "Provided capacity for flat3<> grid is too small, "
                        "must be at least " + std::to_string(buffer_size()) + " elements");
        }
    }

    /**
     * @brief Destroy the flat3 grid object
     * 
     * @note Calls the buffer release function for each buffer
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
     * @brief Delete default copy assignment
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
     * @brief Get the grid dimensions
     * 
     * @return uint2 
     */
    uint2 get_grid_dims() const noexcept { return grid_dims; }

    /**
     * @brief Buffer size
     * 
     * @return total size of each data buffer (in elements)
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
    friend std::ostream& operator<<(std::ostream& os, const flat3 & obj) {
        return os << obj.name << '{' << obj.grid_dims << '}';
    }

    /**
     * @brief Zero data on all buffers
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
     * @param val_x     x value
     * @param val_y     y value
     * @param val_z     z value
     */
    void set( T const & val_x, T const & val_y, T const & val_z  ){
        
        size_t const size = buffer_size( );
        #pragma omp parallel for
        for( size_t i = 0; i < size; i++ ) {
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
        if ( rhs.grid_dims != grid_dims ) {
            std::ostringstream msg;
            msg << "add(): incompatible grid sizes (" << name << ": " << grid_dims
                      << " vs " << rhs.name << ": " << rhs.grid_dims << ')';
            util::fatal(msg.str());
        }

        size_t const size = buffer_size( );
        #pragma omp parallel for
        for( size_t i = 0; i < size; i++ ) {
            x_buffer[i] += rhs.x_buffer[i];
            y_buffer[i] += rhs.y_buffer[i];
            z_buffer[i] += rhs.z_buffer[i];
        }
    };

    /**
     * @brief Operator +=
     * 
     * @param rhs           Other grid to add
     * @return flat3<T>& 
     */
    flat3 & operator+=(const flat3<T> & rhs) {
        add( rhs );
        return *this;
    }

    /**
     * @brief Save grid values to disk
     * 
     * @param fc        Field component to save
     * @param filename  Output file name (includes path)
     */
    void save( fcomp::cart fc, std::string filename ) {
        uint64_t dims[2] = { grid_dims.x, grid_dims.y };
        zdf::save_grid( component( fc ), 2, dims, name + "-" + fcomp::name(fc), filename );
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
        // Fill in grid dimensions
        info.ndims = 2;
        info.count[0] = grid_dims.x;
        info.count[1] = grid_dims.y;

        zdf::save_grid<T>( component( fc ), info, iter, path );
    }
};

}
