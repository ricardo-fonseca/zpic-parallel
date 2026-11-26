#pragma once

#include "../core/vec_types.hpp"
#include "../core/bounds.hpp"
#include "../util/memory.hpp"
#include "../util/error.hpp"
#include "zdf.hpp"

#include <vector>
#include <string>
#include <utility>

namespace grid {

/**
 * @brief Basic grid class with guard cells (halo)
 * 
 * @note Used as a contiguous 2D array, including guard cells
 * 
 * @tparam T    grid datatype
 */
template <class T>
class halo{

    protected:

    /// @brief Grid size
    uint2 grid_dims;

    /// @brief Grid size including guard cells
    uint2 grid_ext_dims;

    /// @brief Periodic boundaries
    int2 grid_periodic;

    /// @brief Offset in cells between buffer[0] and position (0,0)
    unsigned int offset;

    /// @brief Data buffer   
    T * d_buffer;    

    /// @brief Guard cells
    bounds_2d<unsigned int> gc;

    public:

    /// @brief Object name
    std::string name ="unnamed_grid";
        
    /**
     * @brief Construct a new halo grid object
     * 
     * @param dims          Grid dimensions
     * @param gc            Number of guard cells
     * @param periodic      Periodic boundaries
     */
    halo( uint2 const dims, bounds_2d<unsigned int> const gc, int2 const periodic ):
        grid_dims( dims ),
        grid_periodic( periodic ),
        d_buffer( nullptr ), 
        gc(gc) {
        
        if ( dims.x == 0 || dims.y == 0 ) {
            util::fatal( "Invalid grid dimensions: "  + to_string(dims) );
        }

        grid_ext_dims = { gc.x.lower + grid_dims.x + gc.x.upper,
                          gc.y.lower + grid_dims.y + gc.y.upper };

        offset = gc.y.lower * grid_ext_dims.x + gc.x.lower;

        // Allocate main data buffer
        d_buffer = memory::malloc<T>( buffer_size() );
    }

    /**
     * @brief Construct a new halo grid object with no guard cells
     * 
     * @param dims          Grid dimensions
     * @param periodic      Periodic boundaries
     */
    halo( uint2 const dims, int2 const periodic ) :
        halo( dims, bounds_2d<unsigned int>{0}, periodic ) { }

    /**
     * @brief Move constructor
     *
     * @note Transfers ownership of the data buffer from `other`. After the
     *       move, `other` is left in a valid but empty state: its pointer is
     *       null so its destructor is a no-op.
     *
     * @param other     Source grid (will be left empty)
     */
    halo( halo && other ) noexcept :
        grid_dims( other.grid_dims ),
        grid_ext_dims( other.grid_ext_dims ),
        grid_periodic( other.grid_periodic ),
        offset( other.offset ),
        d_buffer( other.d_buffer ),
        gc( other.gc ),
        name( std::move( other.name ) ) {
            
        // Leave `other` in a destructible but empty state.
        other.d_buffer = nullptr;
    }

    /**
     * @brief Destroy the basic grid object
     * 
     */
    ~halo() {
        if ( d_buffer != nullptr ) memory::free( d_buffer );
    }

    /**
     * @brief Delete default copy constructor
     * 
     */
    halo(const halo&) = delete;

    /**
     * @brief Delete default copy constructor
     * 
     */
    halo& operator=(const halo&) = delete;

    /**
     * @brief Get a pointer to the data buffer
     * 
     * @return T* 
     */
    T* data() const noexcept { return d_buffer; }

    /**
     * @brief Get the guard cells
     * 
     * @return bounds_2d<unsigned int> 
     */
    bounds_2d<unsigned int> get_gc() const noexcept { return gc; }

    /**
     * @brief Get the grid dimensions
     * 
     * @return uint2 
     */
    uint2 get_grid_dims() const noexcept { return grid_dims; }

    /**
     * @brief Get the grid dimensions including guard cells
     * 
     * @return uint2 
     */
    uint2 get_grid_ext_dims() const noexcept { return grid_ext_dims; }

    /**
     * @brief Get the periodic boundaries flags
     * 
     * @return int2 
     */
    int2 get_grid_periodic() const noexcept { return grid_periodic; }

    /**
     * @brief Get the offset between buffer[0] and position (0,0)
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
        return static_cast<std::size_t>( grid_ext_dims.y ) * grid_ext_dims.x;
    };

    /**
     * @brief Stream extraction
     * 
     * @param os 
     * @param obj 
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, const halo<T>& obj) {
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
    void add( const halo<T> &rhs ) {
        if ( rhs.grid_ext_dims != grid_ext_dims ) {
            util::fatal( 
                "add(): incompatible grid sizes (" + name + ": " + to_string(grid_ext_dims) +
                " vs " + rhs.name + ": " + to_string(rhs.grid_ext_dims) + ')' 
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
     * @return halo<T>& 
     */
    halo<T>& operator+=(const halo<T>& rhs) {
        add( rhs );
        return *this;
    }

    /**
     * @brief Copies x values to neighboring guard cells
     * 
     * @note Guard cells are only updated for periodic boundaries
     */
    void copy_to_gc_x() {
        if ( grid_periodic.x ) {
            #pragma omp parallel for
            for( unsigned j = 0; j < grid_ext_dims.y; j++ ) {
                for( unsigned i = 0; i < gc.x.upper; i++ ) {
                    d_buffer[ j * grid_ext_dims.x + gc.x.lower + grid_dims.x + i ] = 
                        d_buffer[ j * grid_ext_dims.x + gc.x.lower + i ];
                }

                for( unsigned i = 0; i < gc.x.lower; i++ ) {
                    d_buffer[ j * grid_ext_dims.x + i ] = 
                        d_buffer[  j * grid_ext_dims.x + grid_dims.x + i ];
                }
            }
        }
    }

    /**
     * @brief Copies y values to neighboring guard cells
     * 
     * @note Guard cells are only updated for periodic boundaries
     */
    void copy_to_gc_y() {
        if ( grid_periodic.y ) {
            #pragma omp parallel for
            for( unsigned j = 0; j < gc.y.lower; j++ ) {
                for( unsigned i = 0; i < grid_ext_dims.x; i++ ) {
                    d_buffer[ j * grid_ext_dims.x + i ] =  
                        d_buffer[ ( grid_dims.y + j ) * grid_ext_dims.x + i ];
                }
            }

            #pragma omp parallel for
            for( unsigned j = 0; j < gc.y.upper; j++ ) {
                for( unsigned i = 0; i < grid_ext_dims.x; i++ ) {
                    d_buffer[ ( gc.y.lower + grid_dims.y + j ) * grid_ext_dims.x + i ] =  
                        d_buffer[ ( gc.y.lower + j ) * grid_ext_dims.x + i ];
                }
            }
        }
    }

    /**
     * @brief Copies edge values to neighboring guard cells
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
     * @note Guard cells are only added for periodic boundaries
     */
    void add_from_gc_x() {
        if ( grid_periodic.x ) {
            #pragma omp parallel for
            for( unsigned j = 0; j < grid_ext_dims.y ; j++ ) {
                for( unsigned i = 0; i < gc.x.upper; i++ ) {
                    d_buffer[ j * grid_ext_dims.x + gc.x.lower + i ] += 
                        d_buffer[ j * grid_ext_dims.x + gc.x.lower + grid_dims.x + i ];
                }

                for( unsigned i = 0; i < gc.x.lower; i++ ) {
                    d_buffer[ j * grid_ext_dims.x + grid_dims.x + i ] += 
                        d_buffer[ j * grid_ext_dims.x + i ];
                }
            }
        }
    }

    /**
     * @brief Adds values from neighboring y guard cells to local data
     * 
     * @note Guard cells are only added for periodic boundaries
     */
    void add_from_gc_y() {
        if ( grid_periodic.y ) {
            #pragma omp parallel for
            for( unsigned j = 0; j < gc.y.upper; j++ ) {
                for( unsigned i = 0; i < grid_ext_dims.x; i++ ) {
                    d_buffer[ ( gc.y.lower + j ) * grid_ext_dims.x + i ] += d_buffer[ ( gc.y.lower + grid_dims.y + j ) * grid_ext_dims.x + i ];
                }
            }

            #pragma omp parallel for
            for( unsigned j = 0; j < gc.y.lower; j++ ) {
                for( unsigned i = 0; i < grid_ext_dims.x; i++ ) {
                    d_buffer[ ( grid_dims.y + j ) * grid_ext_dims.x + i ] +=  d_buffer[ j * grid_ext_dims.x + i ];
                }
            }
        }
    }

    /**
     * @brief Adds values from neighboring guard cells to local data
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

            const int ystride = grid_ext_dims.x;

            #pragma omp parallel for
            for( unsigned iy = 0; iy < grid_ext_dims.y; iy++ ) {
                for( unsigned ix = 0; ix < grid_ext_dims.x - shift; ix++ ) {
                    d_buffer[ ix + iy * ystride ] = d_buffer[ (ix + shift) + iy * ystride ]; 
                }

                for( unsigned ix = grid_ext_dims.x - shift; ix < grid_ext_dims.x; ix++ ) {
                    d_buffer[ ix + iy * ystride ] = T{0};
                }
            }

            // Copy x guard cells
            copy_to_gc_x();

        } else {
            util::fatal( "x_shift_left(), invalid shift value, must be 0 < shift <= gc.x.upper" );
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

            const int ystride = grid_ext_dims.x;
            auto * __restrict__ data = & d_buffer[ offset ];

            #pragma omp parallel for
            for( int iy = 0; iy < static_cast<int>(grid_dims.y); iy ++ ) {
                auto prev = data[ iy * ystride - 1 ];
                auto curr = data[ iy * ystride + 0 ];

                for( int ix = 0; ix < static_cast<int>(grid_dims.x); ix ++ ) {
                    auto next = data[ iy * ystride + ix + 1 ];
                    data[ iy * ystride + ix ] = prev * a + curr * b + next * c;
                    prev = curr;
                    curr = next;
                }
            }

            // Update guard cells
            copy_to_gc();

        } else {
            util::fatal( "kernel3_x() requires at least 1 guard cell at both the lower and upper x boundaries." );
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

            const int ystride = grid_ext_dims.x;
            auto * __restrict__ data = & d_buffer[ offset ];

            std::vector<T> prev_row( grid_dims.x );
            std::vector<T> curr_row( grid_dims.x );

            #pragma omp parallel for
            for( int ix = 0; ix < static_cast<int>(grid_dims.x); ix++ ) {
                prev_row[ix] = data[ -1 * ystride + ix ];
                curr_row[ix] = data[  0 * ystride + ix ];
            }

            for( int iy = 0; iy < static_cast<int>(grid_dims.y); iy++ ) {
                #pragma omp parallel for
                for( int ix = 0; ix < static_cast<int>(grid_dims.x); ix++ ) {
                    T next = data[ (iy+1) * ystride + ix ];
                    data[ iy * ystride + ix ] = prev_row[ix] * a + curr_row[ix] * b + next * c; 
                    prev_row[ix] = curr_row[ix];
                    curr_row[ix] = next;
                }
            }

            copy_to_gc();
        } else {
            util::fatal( "kernel3_y() requires at least 1 guard cell at both the lower and upper y boundaries." );
        }
    }

    /**
     * @brief Transpose the grid
     * 
     * @note The operation requires a temporary buffer that must be at least
     *       grid_dims.x * grid_dims.y size. Guard cells are swapped (x <-> y)
     *       but their values are not correct after the transpose, if
     *       required call copy_to_gc().
     * 
     * @param tmp   Temporary buffer for transpose operation
     */
    void transpose( T * tmp ) {

        const auto nx = grid_dims.x;
        const auto ny = grid_dims.y;

        // Transpose data into temporary buffer
        // The loop order is optimized for the memory writes to be contiguous
        {
            const T* __restrict__ data = &d_buffer[ offset ];
            const auto ystride = grid_ext_dims.x;

            #pragma omp parallel for
            for( unsigned ix = 0; ix < nx; ix++ ) {
                for( unsigned iy = 0; iy < ny; iy++ ) {
                    tmp[ ix * ny + iy ] = data[ iy * ystride + ix ];
                }
            }
        }

        // Reshape grid - the data buffer size remains unchanged
        grid_dims = make_uint2( ny, nx );
        std::swap( gc.x, gc.y );
        std::swap( grid_periodic.x, grid_periodic.y );
        grid_ext_dims = make_uint2(
            gc.x.lower + grid_dims.x + gc.x.upper,
            gc.y.lower + grid_dims.y + gc.y.upper
        );
        offset = gc.y.lower * grid_ext_dims.x + gc.x.lower;

        // Copy transposed data back into the (reshaped) data buffer
        {
            T * __restrict__ data = &d_buffer[ offset ];
            const auto ystride = grid_ext_dims.x;

            #pragma omp parallel for
            for( unsigned iy = 0; iy < grid_dims.y; iy++ ) {
                for( unsigned ix = 0; ix < grid_dims.x; ix++ ) {
                    data[ iy * ystride + ix ] = tmp[ iy * grid_dims.x + ix ];
                }
            }
        }
    }

    /**
     * @brief Transpose the data
     * 
     * @note Guard cells are not correct after transpose, if required call
     *       copy_to_gc().
     * 
     */
    void transpose() {
        T* tmp = memory::malloc<T>( static_cast<std::size_t>( grid_dims.x ) * grid_dims.y );
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
        T2 * out = memory::malloc<T2>( static_cast<std::size_t>( grid_dims.x ) * grid_dims.y );

        // Gather data on contiguous grid
        #pragma omp parallel for
        for( int iy = 0; iy < static_cast<int>(grid_dims.y); iy++ ) {
            for( int ix = 0; ix < static_cast<int>(grid_dims.x); ix++ ) {
                out[ iy * grid_dims.x + ix ] = d_buffer[ offset + iy * grid_ext_dims.x + ix ];
            }
        }

        uint64_t dims[2] = { grid_dims.x, grid_dims.y };
        zdf::save_grid( out, 2, dims, name, filename );

        // Free temporary buffer
        memory::free( out );
    }
};

}
