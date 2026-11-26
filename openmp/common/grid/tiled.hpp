#pragma once

#include <iostream>
#include <cassert>
#include <algorithm>
#include <string>
#include <cstddef>
#include <cstring>

#include "../core/vec_types.hpp"
#include "../core/bounds.hpp"
#include "zdf.hpp"

#include "../util/memory.hpp"
#include "../util/math.hpp"
#include "../util/error.hpp"

namespace grid {

/**
 * @brief Tile volume alignment (number of elements)
 * 
 * @details Tile buffers are stored contiguously (tile `tid` starts at element
 *          `tid * tile_vol`), and `tile_vol` is rounded up to a multiple of
 *          this value. With a 64 byte aligned base buffer, every tile buffer is
 *          then 64 byte aligned for any element type with sizeof(T) >= 4
 *          (float, float3, double, ...), as required by aligned SIMD
 *          loads/stores up to AVX512 (e.g. `vec_memadd()`).
 */
constexpr unsigned int tile_vol_align = 16;

template <class T>
struct tiled_view {

    /// @brief Number of tiles
    const uint2 grid_ntiles;

    /// @brief Data buffer
    T * d_buffer;

    /// @brief Tile grid dimensions
    const uint2 tile_dims;
    
    /// @brief Tile grid dimensions including guard cells
    const uint2 tile_ext_dims;

    /// @brief Offset, in cells, from the start of a tile's data buffer to its local (0,0) point
    const unsigned int inner_offset;

    /// @brief Tile guard cells
    const bounds_2d<unsigned int> gc;

    /// @brief Tile volume (may be larger than tile_ext_dim.x * tile_ext_dim.y for alignment)
    const std::size_t tile_vol;

    /**
     * @brief Buffer size
     * 
     * @return total size of data buffers (in elements)
     */
    inline std::size_t buffer_size() const noexcept {
        return tile_vol * static_cast<std::size_t>(grid_ntiles.x) * grid_ntiles.y;
    };

    /**
     * @brief Get a pointer to the start of a specific tile's data buffer
     * 
     * @note This points at the first guard cell of the tile (if any), not at
     *       the tile's local (0,0) point. Use tile_data() for a pointer to
     *       the local (0,0) point instead.
     * 
     * @param tid   Tile index (flat)
     * @return T* 
     */
    inline T * tile_buffer( const unsigned int tid ) const noexcept {
        return & d_buffer[ tid * tile_vol ];
    }

    /**
     * @brief Get a pointer to the start of a specific tile's data buffer
     * 
     * @note This points at the first guard cell of the tile (if any), not at
     *       the tile's local (0,0) point. Use tile_data() for a pointer to
     *       the local (0,0) point instead.
     * 
     * @param tx    x tile index
     * @param ty    y tile index
     * @return T* 
     */
    inline T * tile_buffer( const unsigned int tx, const unsigned int ty ) const noexcept {
        return & d_buffer[ (ty * grid_ntiles.x + tx) * tile_vol ];
    }

    /**
     * @brief Get a pointer to the start of a specific tile's data buffer
     * 
     * @note This points at the first guard cell of the tile (if any), not at
     *       the tile's local (0,0) point. Use tile_data() for a pointer to
     *       the local (0,0) point instead.
     * 
     * @param tid   Tile index (x,y)
     * @return T* 
     */
    inline  T * tile_buffer( const uint2 tid ) const noexcept {
        return & d_buffer[ (tid.y * grid_ntiles.x + tid.x) * tile_vol ];
    }

    /**
     * @brief Get a pointer to a specific tile's local (0,0) point
     * 
     * @note This skips past the tile's guard cells (if any). Use
     *       tile_buffer() for a pointer to the start of the tile's data
     *       buffer instead.
     * 
     * @param tid   Tile index (flat)
     * @return T* 
     */
    inline T * tile_data( const unsigned int tid ) const noexcept {
        return & d_buffer[ tid * tile_vol + inner_offset ];
    }

    /**
     * @brief Get a pointer to a specific tile's local (0,0) point
     * 
     * @note This skips past the tile's guard cells (if any). Use
     *       tile_buffer() for a pointer to the start of the tile's data
     *       buffer instead.
     * 
     * @param tx    x tile index
     * @param ty    y tile index
     * @return T* 
     */
    inline T * tile_data( const unsigned int tx, const unsigned int ty ) const noexcept {
        return & d_buffer[ (ty * grid_ntiles.x + tx) * tile_vol + inner_offset ];
    }

    /**
     * @brief Get a pointer to a specific tile's local (0,0) point
     * 
     * @note This skips past the tile's guard cells (if any). Use
     *       tile_buffer() for a pointer to the start of the tile's data
     *       buffer instead.
     * 
     * @param tid   Tile index (x,y)
     * @return T* 
     */
    inline T * tile_data( const uint2 tid ) const noexcept {
        return & d_buffer[ (tid.y * grid_ntiles.x + tid.x) * tile_vol + inner_offset ];
    }

    /**
     * @brief Y stride in each tile (same as tile_ext_dims.x)
     * 
     * @return unsigned int 
     */
    inline unsigned int tile_ystride() const noexcept {
        return tile_ext_dims.x;
    }
};

/**
 * @brief Tiled grid class
 * 
 */
template <class T>
class tiled {
    protected:

    /// @brief Data buffer
    T * d_buffer;

    private:

    /**
     * @brief Validate grid parameters. The execution will stop if errors are
     * found.
     * 
     */
    void validate_parameters() {
        // Grid parameters
        if ( grid_ntiles.x == 0 || grid_ntiles.y == 0 ) {
            util::fatal( "Invalid number of tiles: " + to_string(grid_ntiles) );
        }

        if ( tile_dims.x == 0 || tile_dims.y == 0 ) {
            util::fatal( "Invalid tile dimensions: " + to_string(tile_dims) );
        }

        if ( gc.x.lower > tile_dims.x || gc.x.upper > tile_dims.x )
            util::fatal( "Number of x guard cells exceeds tile size along x" );

        if ( gc.y.lower > tile_dims.y || gc.y.upper > tile_dims.y )
            util::fatal( "Number of y guard cells exceeds tile size along y" );
    }

    public:

    /// @brief Number of tiles
    const uint2 grid_ntiles;

    /// @brief Tile grid dimensions
    const uint2 tile_dims;
    
    /// @brief Tile guard cells
    const bounds_2d<unsigned int> gc;
    
    /// @brief Tile grid dimensions including guard cells
    const uint2 tile_ext_dims;

    /// @brief Offset, in cells, from the start of a tile's data buffer to its local (0,0) point
    const unsigned int inner_offset;

    /// @brief Tile volume (may be larger than tile_ext_dim.x * tile_ext_dim.y for alignment)
    const std::size_t tile_vol;

    /// @brief Grid dimensions (all tiles, excluding guard cells)
    const uint2 grid_dims;

    /// @brief Periodic boundaries
    const int2 grid_periodic;

    /// @brief Object name
    std::string name;

    /**
     * @brief Construct a new grid object
     * 
     * @param ntiles        Number of tiles
     * @param tile_dims     Individual tile size
     * @param gc            Number of guard cells
     * @param periodic      Periodic boundaries
     */
    tiled( uint2 const ntiles, uint2 const tile_dims, bounds_2d<unsigned int> const gc, int2 const periodic ):
        d_buffer( nullptr ), 
        grid_ntiles( ntiles ),
        tile_dims( tile_dims ),
        gc(gc),
        tile_ext_dims( make_uint2( gc.x.lower + tile_dims.x + gc.x.upper,
                            gc.y.lower + tile_dims.y + gc.y.upper )),
        inner_offset( gc.y.lower * tile_ext_dims.x + gc.x.lower ),
        tile_vol( roundup<tile_vol_align>( tile_ext_dims.x * tile_ext_dims.y ) ),
        grid_dims( ntiles * tile_dims ),
        grid_periodic( periodic ),
        name( "tiled grid" )
    {
        // Validate parameters
        validate_parameters();

        // Allocate main data buffer
        d_buffer = memory::malloc<T>( buffer_size() );
    };

    /**
     * @brief Construct a new tile grid object
     * 
     * @note: The number of guard cells is set to 0
     * 
     * @param ntiles        Number of tiles
     * @param tile_dims     Individual tile size
     * @param periodic      Periodic boundaries
     */
    tiled( uint2 const ntiles, uint2 const tile_dims, int2 const periodic ):
        d_buffer( nullptr ),
        grid_ntiles( ntiles ),
        tile_dims( tile_dims ),
        gc( 0 ),
        tile_ext_dims( make_uint2( tile_dims.x, tile_dims.y )),
        inner_offset( 0 ),
        tile_vol( roundup<tile_vol_align>( tile_dims.x * tile_dims.y )),
        grid_dims( ntiles * tile_dims ),
        grid_periodic( periodic ),
        name( "tiled grid" )
    {
        // Validate parameters
        validate_parameters();

        // Allocate main data buffer
        d_buffer = memory::malloc<T>( buffer_size() );
    };

    /**
    * @brief Move constructor
    *
    * @note Steals the data buffer from `other`, leaving it in a valid but
    *       empty state (safe to destruct).
    *
    * @param other     tiled grid to move from
    */
    tiled( tiled && other ) noexcept :
        d_buffer( other.d_buffer ),
        grid_ntiles( other.grid_ntiles ),
        tile_dims( other.tile_dims ),
        gc( other.gc ),
        tile_ext_dims( other.tile_ext_dims ),
        inner_offset( other.inner_offset ),
        tile_vol( other.tile_vol ),
        grid_dims( other.grid_dims ),
        grid_periodic( other.grid_periodic ),
        name( std::move( other.name ) )
    {
        // Null out other's owned resources so its destructor is a no-op
        other.d_buffer = nullptr;
    }

    /**
     * @brief tiled grid destructor
     * 
     */
    ~tiled(){
        if ( d_buffer != nullptr ) memory::free( d_buffer );
    };

    /**
     * @brief Delete default copy constructor
     * 
     */
    tiled(const tiled&) = delete;

    /**
     * @brief Delete default copy assignment
     * 
     */
    tiled& operator=(const tiled&) = delete;

    /**
     * @brief Returns a view of the tiled grid
     * 
     * @return tiled_view<T> 
     */
    tiled_view<T> view() noexcept {
        return { 
            grid_ntiles, 
            d_buffer, 
            tile_dims,
            tile_ext_dims, 
            inner_offset, 
            gc,
            tile_vol };
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
    tiled_view<const T> cview() const noexcept {
        return { 
            grid_ntiles, 
            d_buffer, 
            tile_dims,
            tile_ext_dims, 
            inner_offset, 
            gc,
            tile_vol };
    }

    /**
     * @brief Get a pointer to the data buffer
     * 
     * @return T* 
     */
    T * buffer() const noexcept { return d_buffer; }

    /**
     * @brief Get a pointer to the start of a specific tile's data buffer
     * 
     * @note This points at the first guard cell of the tile (if any), not at
     *       the tile's local (0,0) point. Use tile_data() for a pointer to
     *       the local (0,0) point instead.
     * 
     * @param tid   Tile index (flat)
     * @return T* 
     */
    T * tile_buffer( const unsigned int tid ) const noexcept {
        return & d_buffer[ tid * tile_vol ];
    }

    /**
     * @brief Get a pointer to the start of a specific tile's data buffer
     * 
     * @note This points at the first guard cell of the tile (if any), not at
     *       the tile's local (0,0) point. Use tile_data() for a pointer to
     *       the local (0,0) point instead.
     * 
     * @param tx    x tile index
     * @param ty    y tile index
     * @return T* 
     */
    T * tile_buffer( const unsigned int tx, const unsigned int ty ) const noexcept {
        return & d_buffer[ (ty * grid_ntiles.x + tx) * tile_vol ];
    }

    /**
     * @brief Get a pointer to the start of a specific tile's data buffer
     * 
     * @note This points at the first guard cell of the tile (if any), not at
     *       the tile's local (0,0) point. Use tile_data() for a pointer to
     *       the local (0,0) point instead.
     * 
     * @param tid   Tile index (x,y)
     * @return T* 
     */
    T * tile_buffer( const uint2 tid ) const noexcept {
        return & d_buffer[ (tid.y * grid_ntiles.x + tid.x) * tile_vol ];
    }

    /**
     * @brief Get a pointer to a specific tile's local (0,0) point
     * 
     * @note This skips past the tile's guard cells (if any). Use
     *       tile_buffer() for a pointer to the start of the tile's data
     *       buffer instead.
     * 
     * @param tid   Tile index (flat)
     * @return T* 
     */
    T * tile_data( const unsigned int tid ) const noexcept {
        return & d_buffer[ tid * tile_vol + inner_offset ];
    }

    /**
     * @brief Get a pointer to a specific tile's local (0,0) point
     * 
     * @note This skips past the tile's guard cells (if any). Use
     *       tile_buffer() for a pointer to the start of the tile's data
     *       buffer instead.
     * 
     * @param tx    x tile index
     * @param ty    y tile index
     * @return T* 
     */
    T * tile_data( const unsigned int tx, const unsigned int ty ) const noexcept {
        return & d_buffer[ (ty * grid_ntiles.x + tx) * tile_vol + inner_offset ];
    }

    /**
     * @brief Get a pointer to a specific tile's local (0,0) point
     * 
     * @note This skips past the tile's guard cells (if any). Use
     *       tile_buffer() for a pointer to the start of the tile's data
     *       buffer instead.
     * 
     * @param tid   Tile index (x,y)
     * @return T* 
     */
    T * tile_data( const uint2 tid ) const noexcept {
        return & d_buffer[ (tid.y * grid_ntiles.x + tid.x) * tile_vol + inner_offset ];
    }

    /**
     * @brief Get the number of tiles
     * 
     * @return uint2 
     */
    uint2 get_grid_ntiles() const noexcept { return grid_ntiles; };

    /**
     * @brief Get the grid dimensions (all tiles, excluding guard cells)
     * 
     * @return uint2 
     */
    uint2 get_grid_dims() const noexcept { return grid_dims; }

    /**
     * @brief Get the periodic boundaries flags
     * 
     * @return int2 
     */
    int2 get_grid_periodic() const noexcept { return grid_periodic; }

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
            << obj.grid_ntiles << " tiles, "
            << obj.tile_dims << " points/tile }";
    }

    /**
     * @brief Buffer size
     * 
     * @return total size of data buffers (in elements)
     */
    std::size_t buffer_size() const noexcept {
        return (static_cast <std::size_t> (tile_vol)) * ( grid_ntiles.x * grid_ntiles.y ) ;
    };

    /**
     * @brief zero device data on a grid grid
     * 
     */
    void zero() {
        memory::zero( d_buffer, buffer_size() );
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
        // Check that the objects are compatible
        assert(( grid_ntiles == rhs.get_grid_ntiles() && tile_vol == rhs.tile_vol ));

        #pragma omp parallel for
        for( size_t i = 0; i < buffer_size(); i++ ) 
            d_buffer[i] += rhs.d_buffer[i];
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
     * dimensions grid_dims.y * grid_dims.x. The stride parameter can be used
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
            stride = make_uint2( 1, grid_dims.x );
        }

        // Check that y stride is valid
        assert(( stride.y > 0 ));

        if ( stride.x == 1 ) {
            // Optimized version for x stride == 1
            #pragma omp parallel for collapse(2)
            for( unsigned ty = 0; ty < grid_ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < grid_ntiles.x; tx ++ ) {
                    T * const __restrict__ tdata = tile_data( tx, ty );

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
            for( unsigned ty = 0; ty < grid_ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < grid_ntiles.x; tx ++ ) {
                    T * const __restrict__ tdata = tile_data( tx, ty );

                    const auto gix0 = tx * tile_dims.x;
                    const auto giy0 = ty * tile_dims.y;

                    // Loop inside tile
                    for( unsigned iy = 0; iy < tile_dims.y; iy ++ ) {
                        for( unsigned ix = 0; ix < tile_dims.x; ix ++ ) {
                            out[ (giy0 + iy) * stride.y + (gix0 + ix) * stride.x ] =
                                tdata[ iy * tile_ext_dims.x + ix ];
                        }
                    }
                }
            }
        }

        return grid_dims.x * grid_dims.y;
    }

    /**
     * @brief Scatter data into the tile grid and updates guard cell values
     * 
     * @note
     * The default behavior is consider that the input data is a contiguous
     * array of dimensions grid_dims.y * grid_dims.x. The stride parameter
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
            stride = make_uint2( 1, grid_dims.x );
        }

        if ( stride.x == 1 ) {
            // Optimized version for x stride == 1
            #pragma omp parallel for collapse(2)
            for( unsigned ty = 0; ty < grid_ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < grid_ntiles.x; tx ++ ) {
                    T * const __restrict__ tdata = tile_data( tx, ty );

                    const auto gix0 = tx * tile_dims.x;
                    const auto giy0 = ty * tile_dims.y;

                    // Loop inside tile
                    for( unsigned iy = 0; iy < tile_dims.y; iy ++ ) {
                        for( unsigned ix = 0; ix < tile_dims.x; ix ++ ) {
                            tdata[ iy * tile_ext_dims.x + ix ] = 
                                d_in[ (giy0 + iy) * stride.y + (gix0 + ix) ];
                        }
                    }
                }
            }

        } else {
            // Arbitrary x stride
            #pragma omp parallel for collapse(2)
            for( unsigned ty = 0; ty < grid_ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < grid_ntiles.x; tx ++ ) {
                    T * const __restrict__ tdata = tile_data( tx, ty );

                    const auto gix0 = tx * tile_dims.x;
                    const auto giy0 = ty * tile_dims.y;

                    // Loop inside tile
                    for( unsigned iy = 0; iy < tile_dims.y; iy ++ ) {
                        for( unsigned ix = 0; ix < tile_dims.x; ix ++ ) {
                            tdata[ iy * tile_ext_dims.x + ix ] = 
                                d_in[ (giy0 + iy) * stride.y + (gix0 + ix) * stride.x ];
                        }
                    }
                }
            }
        }

        // Update guard cell values
        copy_to_gc();

        return grid_dims.x * grid_dims.y;
    }

    /**
     * @brief Scatter data into the tile grid and updates guard cell values
     * 
     * @note
     * The default behavior is consider that the input data is a contiguous
     * array of dimensions grid_dims.y * grid_dims.x. The stride parameter
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
            stride = make_uint2( 1, grid_dims.x );
        }

        if ( stride.x == 1 ) {
            // Optimized version for x stride == 1
            #pragma omp parallel for collapse(2)
            for( unsigned ty = 0; ty < grid_ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < grid_ntiles.x; tx ++ ) {
                    T * const __restrict__ tdata = tile_data( tx, ty );

                    const auto gix0 = tx * tile_dims.x;
                    const auto giy0 = ty * tile_dims.y;

                    // Loop inside tile
                    for( unsigned iy = 0; iy < tile_dims.y; iy ++ ) {
                        for( unsigned ix = 0; ix < tile_dims.x; ix ++ ) {
                            tdata[ iy * tile_ext_dims.x + ix ] = 
                                d_in[ (giy0 + iy) * stride.y + (gix0 + ix) ] * scale;
                        }
                    }
                }
            }

        } else {
            // Arbitrary x stride
            #pragma omp parallel for collapse(2)
            for( unsigned ty = 0; ty < grid_ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < grid_ntiles.x; tx ++ ) {
                    T * const __restrict__ tdata = tile_data( tx, ty );

                    const auto gix0 = tx * tile_dims.x;
                    const auto giy0 = ty * tile_dims.y;

                    // Loop inside tile
                    for( unsigned iy = 0; iy < tile_dims.y; iy ++ ) {
                        for( unsigned ix = 0; ix < tile_dims.x; ix ++ ) {
                            tdata[ iy * tile_ext_dims.x + ix ] = 
                                d_in[ (giy0 + iy) * stride.y + (gix0 + ix) * stride.x ] *
                                scale;
                        }
                    }
                }
            }
        }

        // Update guard cell values
        copy_to_gc();

        return grid_dims.x * grid_dims.y;
    }

    /**
     * @brief Copies edge values to X neighboring guard cells
     */
    void copy_to_gc_x() {

        const auto ystride  = tile_ext_dims.x;

        // Loop over tiles
        #pragma omp parallel for
        for( int tid = 0; tid < static_cast<int>(grid_ntiles.y * grid_ntiles.x); tid ++ ) {

            const auto tile_idx = make_int2( tid % grid_ntiles.x, tid / grid_ntiles.x );
            
            auto * __restrict__ local = tile_buffer( tid );

            {   // Copy from lower neighbor
                int neighbor_tx = tile_idx.x - 1;
                if ( grid_periodic.x && neighbor_tx < 0 ) neighbor_tx += grid_ntiles.x;

                if ( neighbor_tx >= 0 ) {
                    auto * __restrict__ x_lower = tile_buffer( neighbor_tx, tile_idx.y );
                    for( unsigned j = 0; j < tile_ext_dims.y; j++ ) {
                        for( unsigned i = 0; i < gc.x.lower; i++ ) {
                            local[ i + j * ystride ] = x_lower[ tile_dims.x + i + j * ystride ];
                        }
                    }
                }
            }

            {   // Copy from upper neighbor
                int neighbor_tx = tile_idx.x + 1;
                if ( grid_periodic.x && neighbor_tx >= static_cast<int>(grid_ntiles.x) ) neighbor_tx -= grid_ntiles.x;

                if ( neighbor_tx < static_cast<int>(grid_ntiles.x) ) {
                    auto * __restrict__ x_upper = tile_buffer( neighbor_tx, tile_idx.y );
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
     */
    void copy_to_gc_y() {

        const auto ystride  = tile_ext_dims.x;

        #pragma omp parallel for
        for( int tid = 0; tid < static_cast<int>(grid_ntiles.y * grid_ntiles.x); tid ++ ) {

            const int2 tile_idx = make_int2( tid % grid_ntiles.x, tid / grid_ntiles.x );
            T * __restrict__ local = tile_buffer( tid );
            
            {   // Copy from lower neighbor
                int neighbor_ty = tile_idx.y - 1;
                if ( grid_periodic.y && neighbor_ty < 0 ) neighbor_ty += grid_ntiles.y;

                if ( neighbor_ty >= 0 ) {
                    auto * __restrict__ y_lower = tile_buffer( tile_idx.x, neighbor_ty );
                    for( int j = 0; j < static_cast<int>(gc.y.lower); j++ ) {
                        for( int i = 0; i < static_cast<int>(tile_ext_dims.x); i++ ) {
                            local[ i + j * ystride ] = y_lower[ i + ( tile_dims.y + j ) * ystride ];
                        }
                    }
                }
            }

            {   // Copy from upper neighbor
                int neighbor_ty = tile_idx.y + 1;
                if ( grid_periodic.y && neighbor_ty >= static_cast<int>(grid_ntiles.y) ) neighbor_ty -= grid_ntiles.y;

                if ( neighbor_ty < static_cast<int>(grid_ntiles.y) ) {
                    auto * __restrict__ y_upper = tile_buffer( tile_idx.x, neighbor_ty );
                    for( int j = 0; j < static_cast<int>(gc.y.upper); j++ ) {
                        for( int i = 0; i < static_cast<int>(tile_ext_dims.x); i++ ) {
                            local[ i + ( gc.y.lower + tile_dims.y + j ) * ystride ] = y_upper[ i + ( gc.y.lower + j ) * ystride ];
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief Copies edge values to neighboring guard cells
     */
    void copy_to_gc()  {

        // Copy along x direction
        copy_to_gc_x();

        // Copy along y direction
        copy_to_gc_y();

    };

    protected:

    // add_from_gc_x() / add_from_gc_y() are kept internal: a partial guard
    // cell reduction along a single direction leaves the tile corners
    // unaccounted for, so callers should use add_from_gc(), which performs
    // both directions in the required order. Promote them if a caller
    // genuinely needs a single direction.

    /**
     * @brief Adds values from neighboring x guard cells to local data
     * 
     * @note Use add_from_gc() instead.
     */
    void add_from_gc_x() {
        // Add along x direction

        const auto ystride  = tile_ext_dims.x;

        // Loop over tiles
        #pragma omp parallel for
        for( int tid = 0; tid < static_cast<int>(grid_ntiles.y * grid_ntiles.x); tid ++ ) {

            const auto tile_idx = make_int2( tid % grid_ntiles.x, tid / grid_ntiles.x );

            auto * __restrict__ local = tile_buffer( tid );
            
            {   // Add from lower neighbour
                int neighbor_tx = tile_idx.x - 1;
                if ( grid_periodic.x && neighbor_tx < 0 ) neighbor_tx += grid_ntiles.x;

                if ( neighbor_tx >= 0 ) {
                    T * __restrict__ x_lower = tile_buffer( neighbor_tx, tile_idx.y );
                    for( unsigned j = 0; j < tile_ext_dims.y; j++ ) {
                        for( unsigned i = 0; i < gc.x.upper; i++ ) {
                            local[ gc.x.lower + i + j * ystride ] += x_lower[ gc.x.lower + tile_dims.x + i + j * ystride ];
                        }
                    }
                }
            }

            {   // Add from upper neighbour
                int neighbor_tx = tile_idx.x + 1;
                if ( grid_periodic.x && neighbor_tx >= static_cast<int>(grid_ntiles.x) ) neighbor_tx -= grid_ntiles.x;

                if ( neighbor_tx < static_cast<int>(grid_ntiles.x) ) {
                    T * __restrict__ x_upper = tile_buffer( neighbor_tx, tile_idx.y );
                    for( int j = 0; j < static_cast<int>(tile_ext_dims.y); j++ ) {
                        for( int i = 0; i <static_cast<int>(gc.x.lower); i++ ) {
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
     * @note Use add_from_gc() instead.
     */
    void add_from_gc_y(){

        const auto ystride  = tile_ext_dims.x;

        // Add along y direction

        // Loop over tiles
        #pragma omp parallel for
        for( int tid = 0; tid < static_cast<int>(grid_ntiles.y * grid_ntiles.x); tid ++ ) {

            const int2 tile_idx = make_int2( tid % grid_ntiles.x, tid / grid_ntiles.x );
            T * __restrict__ local = tile_buffer( tid );
            
            {   // Add from lower neighbour
                int neighbor_ty = tile_idx.y - 1;
                if ( grid_periodic.y && neighbor_ty < 0 ) 
                    neighbor_ty += grid_ntiles.y;

                if ( neighbor_ty >= 0 ) {
                    T * __restrict__ y_lower = tile_buffer( tile_idx.x, neighbor_ty );
                    for( int j = 0; j < static_cast<int>(gc.y.upper); j++ ) {
                        for( int i = 0; i < static_cast<int>(tile_ext_dims.x); i++ ) {
                            local[ i + ( gc.y.lower + j ) * ystride ] += 
                                y_lower[ i + ( gc.y.lower + tile_dims.y + j ) * ystride ];
                        }
                    }
                }
            }

            {   // Add from upper neighbour
                int neighbor_ty = tile_idx.y + 1;
                if ( grid_periodic.y && neighbor_ty >= static_cast<int>(grid_ntiles.y) ) 
                    neighbor_ty -= grid_ntiles.y;

                if ( neighbor_ty < static_cast<int>(grid_ntiles.y) ) {
                    T * __restrict__ y_upper = tile_buffer( tile_idx.x, neighbor_ty );
                    for( unsigned j = 0; j < gc.y.lower; j++ ) {
                        for( unsigned i = 0; i < tile_ext_dims.x; i++ ) {
                            local[ i + ( tile_dims.y + j ) * ystride ] += y_upper[ i + j * ystride ];
                        }
                    }
                }
            }
        }
    };

    public:

    /**
     * @brief Adds values from neighboring guard cells to local data
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

            const int ystride = tile_ext_dims.x;

            // Loop over tiles
            #pragma omp parallel for
            for( int tid = 0; tid < static_cast<int>(grid_ntiles.y * grid_ntiles.x); tid ++ ) {
                T * __restrict__ buffer = tile_buffer( tid );
                
                for( int iy = 0; iy < static_cast<int>(tile_ext_dims.y); iy++ ) {
                    for( int ix = 0; ix < static_cast<int>(tile_ext_dims.x - shift); ix++ ) {
                        buffer[ ix + iy * ystride ] = buffer[ (ix + shift) + iy * ystride ]; 
                    }
                    for( int ix = tile_ext_dims.x - shift; ix < static_cast<int>(tile_ext_dims.x); ix++ ) {
                        buffer[ ix + iy * ystride ] = T{};
                    }
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
     * @note Guard cells must be up to date on entry. Guard cells are updated
     *       on exit.
     * 
     * @param a             Kernel value a
     * @param b             Kernel value b
     * @param c             Kernel value c
     * @param upper_ext     (optional) Also compute the result on the first
     *                      upper x guard column and first upper y guard row,
     *                      see kernel::kernel3_x(). Requires gc.x.upper >= 2
     *                      and gc.y.upper >= 1
     */
    template < typename S >
    void kernel3_x( S const a, S const b, S const c, bool const upper_ext = false ) {

        if (( gc.x.lower < 1 ) || ( gc.x.upper < 1 )) {
            util::fatal( "kernel3_x() requires at least 1 guard cell at both the lower and upper x boundaries." );
        }

        if ( upper_ext && ( gc.x.upper < 2 || gc.y.upper < 1 ) ) {
            util::fatal( "kernel3_x() with upper_ext requires at least 2 upper x guard cells"
                        " and 1 upper y guard cell." );
        }

        const int ext = ( upper_ext ) ? 1 : 0;
        const int ystride = tile_ext_dims.x;
        const int nx      = tile_dims.x + ext;
        const int ny      = tile_dims.y + ext;
        const int offset  = inner_offset;

        // Loop over tiles
        #pragma omp parallel for
        for( int tid = 0; tid < static_cast<int>(grid_ntiles.y * grid_ntiles.x); tid++ ) {

            // On a GPU these would be on block shared memory
            T A[ tile_vol ];

            T * __restrict__ buffer = tile_buffer( tid );

            // Copy data from tile buffer
            std::memcpy( A, buffer, tile_vol * sizeof(T) );

            // Apply kernel, written directly to the tile buffer
            for( int idx = 0; idx < ny * nx; idx ++ ) {
                const int iy = idx / nx;
                const int ix = idx % nx;
                const int k  = iy * ystride + ix + offset;
                buffer[ k ] = A[ k - 1 ] * a +
                              A[ k     ] * b +
                              A[ k + 1 ] * c;
            }
        }

        // Update guard cells (both directions: only the tile interior, plus
        // the optional upper extension, was written)
        copy_to_gc();
    }

    /**
     * @brief Perform a convolution with a 3 point kernel [a,b,c] along y
     * 
     * @note Guard cells must be up to date on entry. Guard cells are updated
     *       on exit.
     * 
     * @param a             Kernel value a
     * @param b             Kernel value b
     * @param c             Kernel value c
     * @param upper_ext     (optional) Also compute the result on the first
     *                      upper x guard column and first upper y guard row,
     *                      see kernel::kernel3_y(). Requires gc.y.upper >= 2
     *                      and gc.x.upper >= 1
     */
    template < typename S >
    void kernel3_y( S const a, S const b, S const c, bool const upper_ext = false ) {

        if (( gc.y.lower < 1 ) || ( gc.y.upper < 1 )) {
            util::fatal( "kernel3_y() requires at least 1 guard cell at both the lower and upper y boundaries." );
        }

        if ( upper_ext && ( gc.y.upper < 2 || gc.x.upper < 1 ) ) {
            util::fatal( "kernel3_y() with upper_ext requires at least 2 upper y guard cells"
                        " and 1 upper x guard cell." );
        }

        const int ext     = ( upper_ext ) ? 1 : 0;
        const int ystride = tile_ext_dims.x;
        const int nx      = tile_dims.x + ext;
        const int ny      = tile_dims.y + ext;
        const int offset  = inner_offset;

        // Loop over tiles
        #pragma omp parallel for
        for( int tid = 0; tid < static_cast<int>(grid_ntiles.y * grid_ntiles.x); tid++ ) {

            // On a GPU these would be on block shared memory
            T A[ tile_vol ];

            auto * __restrict__ buffer = tile_buffer( tid );

            // Copy data from tile buffer
            std::memcpy( A, buffer, tile_vol * sizeof(T) );

            // Apply kernel, written directly to the tile buffer
            for( int idx = 0; idx < ny * nx; idx ++ ) {
                const int iy = idx / nx;
                const int ix = idx % nx;
                const int k  = iy * ystride + ix + offset;
                buffer[ k ] = A[ k - ystride ] * a +
                              A[ k           ] * b +
                              A[ k + ystride ] * c;
            }
        }

        // Update guard cells (both directions: only the tile interior, plus
        // the optional upper extension, was written)
        copy_to_gc();
    }

    /**
     * @brief Perform a separable 2D convolution with a 3 point kernel [a,b,c]
     *        along x and y
     * 
     * @note Equivalent to kernel3_x( a, b, c ) followed by kernel3_y( a, b, c ),
     *       i.e. a 3x3 kernel [a,b,c] ⊗ [a,b,c], but with a single kernel
     *       launch and a single guard cell update.
     * 
     * @note Guard cells (including corners) must be up to date on entry, as
     *       with kernel3_x() / kernel3_y(). Guard cells are updated on exit.
     * 
     * @param a             Kernel value a
     * @param b             Kernel value b
     * @param c             Kernel value c
     * @param upper_ext     (optional) Also compute the result on the first
     *                      upper x guard column and first upper y guard row,
     *                      see kernel::kernel3_xy(). Requires gc.x.upper >= 2
     *                      and gc.y.upper >= 2
     */
    template < typename S >
    void kernel3_xy( S const a, S const b, S const c, bool const upper_ext = false ) {

        if (( gc.x.lower < 1 ) || (gc.x.upper < 1 ) || 
            ( gc.y.lower < 1 ) || (gc.y.upper < 1 )) {
            util::fatal( "kernel3_xy() requires at least 1 guard cell at all x and y boundaries." );
        }

        if ( upper_ext && ( gc.x.upper < 2 || gc.y.upper < 2 ) ) {
            util::fatal( "kernel3_xy() with upper_ext requires at least 2 upper x and y guard cells." );
        }

        const int ext     = ( upper_ext ) ? 1 : 0;
        const int ystride = tile_ext_dims.x;
        const int nx      = tile_dims.x + ext;
        const int ny      = tile_dims.y + ext;
        const int offset  = inner_offset;

        // Loop over tiles
        #pragma omp parallel for
        for( int tid = 0; tid < static_cast<int>(grid_ntiles.y * grid_ntiles.x); tid++ ) {

            // On a GPU these would be on block shared memory
            T A[ tile_vol ];
            T B[ tile_vol ];

            auto * __restrict__ buffer = tile_buffer( tid );

            // Copy data from tile buffer
            std::memcpy( A, buffer, tile_vol * sizeof(T) );

            // x pass, rows -1 to ny (inclusive)
            for( int idx = 0; idx < ( ny + 2 ) * nx; idx ++ ) {
                const int iy = idx / nx - 1;
                const int ix = idx % nx;
                const int k  = iy * ystride + ix + offset;
                B[ k ] = A[ k - 1 ] * a +
                        A[ k     ] * b +
                        A[ k + 1 ] * c;
            }

            // y pass, written directly to the tile buffer
            for( int idx = 0; idx < ny * nx; idx ++ ) {
                const int iy = idx / nx;
                const int ix = idx % nx;
                const int k  = iy * ystride + ix + offset;
                buffer[ k ] = B[ k - ystride ] * a +
                              B[ k           ] * b +
                              B[ k + ystride ] * c;
            }
        }

        // Update guard cells
        copy_to_gc();
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

        // Fill in grid dimensions
        info.ndims = 2;
        info.count[0] = grid_dims.x;
        info.count[1] = grid_dims.y;

        // Allocate buffer on host to gather data
        T2 * h_data = memory::malloc<T2>( static_cast<std::size_t>(grid_dims.x) * grid_dims.y );

        // Gather data on contiguous grid
        gather( h_data );

        // Save data
        zdf::save_grid<T2>( h_data, info, iter, path );

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
        T2 * h_data = memory::malloc<T2>( static_cast<std::size_t>(grid_dims.x) * grid_dims.y );

        // Gather data on contiguous grid
        gather( h_data );

        uint64_t dims[2] = { grid_dims.x, grid_dims.y };
        zdf::save_grid( h_data, 2, dims, name, filename );

        // Free temporary buffer
        memory::free( h_data );
    }

};

} // end of namespace grid
