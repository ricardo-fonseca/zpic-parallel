/**
 * @file particles_common.hpp
 * @brief Generic tiled particle buffer (common implementation)
 *
 * @details This file is not meant to be included directly. Each particle
 *          type creates a local wrapper header (e.g. `cyl_part/particles.hpp`)
 *          that defines the namespace and the particle quantities and then
 *          includes this file:
 *
 *              #pragma once
 *              #define PARTICLES_NAMESPACE cyl_part
 *              #define PARTICLES_QUANTITIES( X ) \
 *                  X( float2, x,  "Position inside cell [-0.5,0.5)" ) \
 *                  X( float3, u,  "Generalized velocity" ) \
 *                  X( float,  q,  "Individual particle charge" ) \
 *                  X( float2, th, "Angular position { cos(θ), sin(θ) }" )
 *              #include "part/particles.hpp"
 *
 *          and a local source file (e.g. `cyl_part/particles.cpp`):
 *
 *              #include "particles.hpp"
 *              #define PARTICLES_NAMESPACE cyl_part
 *              #include "part/particles_impl.hpp"
 *
 *          The cell index `int2 ix` is always present and must not be listed.
 *          Each entry is `X( type, name, description )`; `description` is
 *          documentation only. Quantity types must be trivially copyable and
 *          have a size that is a multiple of 4 bytes (e.g. float, float2,
 *          float3, double, double2, int, uint64_t).
 *
 *          Quantity names must not collide with the names of the members of
 *          `particles_view` / `particles` (e.g. ix, tile_np, tile_offset,
 *          max_part, load, store, ...).
 *
 *          If a quantity named `x` exists, it is assumed to be the position
 *          inside the cell, and `validate()` checks that its .x / .y
 *          components are in [-0.5, 0.5).
 *
 * @note    This file intentionally has no include guard, so that several
 *          particle types may be used in the same translation unit. The
 *          wrapper headers should use `#pragma once`. Both macros are
 *          undefined at the end of this file.
 *
 * @note    The helper files (part_tiles.hpp, sort.hpp) do not depend on the
 *          particle quantities and are shared by all particle types, in the
 *          `part` namespace.
 */

#ifndef PARTICLES_NAMESPACE
#error "PARTICLES_NAMESPACE must be defined before including common/particles.hpp"
#endif

#ifndef PARTICLES_QUANTITIES
#error "PARTICLES_QUANTITIES must be defined before including common/particles.hpp"
#endif

#include "../core/vec_types.hpp"
#include "../core/bounds.hpp"
#include "../util/memory.hpp"
#include "../util/math.hpp"
#include "../util/error.hpp"

#include "part_tiles.hpp"
#include "sort.hpp"

#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <type_traits>

/*
 * Helper X macros, applied to PARTICLES_QUANTITIES. All are undefined at the
 * end of the file.
 */

/// @brief Declare quantity value (e.g. `float2 x;`)
#define PARTICLES_DECL_VALUE_( T, n, d )    T n;
/// @brief Declare quantity pointer (e.g. `float2 * x;`)
#define PARTICLES_DECL_PTR_( T, n, d )      T * n;
/// @brief Comma separated list of quantity names
#define PARTICLES_NAME_LIST_( T, n, d )     n,
/// @brief Sum of quantity sizes
#define PARTICLES_SIZE_SUM_( T, n, d )      + sizeof( T )
/// @brief Allocate device buffer for quantity
#define PARTICLES_ALLOC_( T, n, d )         n = memory::malloc< T >( max_part );
/// @brief Free device buffer for quantity
#define PARTICLES_FREE_( T, n, d )          memory::free( n );
/// @brief Swap quantity buffers between objects a and b
#define PARTICLES_SWAP_( T, n, d )          std::swap( a.n, b.n );
/// @brief Compile time checks on quantity type
#define PARTICLES_CHECK_TYPE_( T, n, d ) \
    static_assert( std::is_trivially_copyable_v< T >, \
        "Particle quantity '" #n "' must be trivially copyable" ); \
    static_assert( sizeof( T ) % 4 == 0, \
        "Particle quantity '" #n "' size must be a multiple of 4 bytes" );

namespace PARTICLES_NAMESPACE {

/**
 * @brief Implementation details (helper functions)
 * 
 */
namespace particle_detail {

/**
 * @brief Checks if all (floating point) components of a value are finite
 * 
 * @note Non floating point types (e.g. int, uint64_t, int2) are always valid
 */
inline bool finite( float  const v ) { return std::isfinite( v ); }
inline bool finite( double const v ) { return std::isfinite( v ); }
inline bool finite( float2 const v ) { return std::isfinite( v.x ) && std::isfinite( v.y ); }
inline bool finite( float3 const v ) { return std::isfinite( v.x ) && std::isfinite( v.y ) && std::isfinite( v.z ); }
inline bool finite( double2 const v ) { return std::isfinite( v.x ) && std::isfinite( v.y ); }
inline bool finite( double3 const v ) { return std::isfinite( v.x ) && std::isfinite( v.y ) && std::isfinite( v.z ); }
template< class T >
inline bool finite( T const & ) { return true; }

/**
 * @brief Detects if a type has a member named x
 */
template< class T, class = void >
struct has_x : std::false_type {};

template< class T >
struct has_x< T, std::void_t< decltype( std::declval< T & >().x ) > > : std::true_type {};

template< class T, class = void >
struct has_y : std::false_type {};

template< class T >
struct has_y< T, std::void_t< decltype( std::declval< T & >().y ) > > : std::true_type {};

/**
 * @brief Checks the position inside the cell, if a quantity named `x` exists
 * 
 * @details Checks that the .x / .y components of `p.x` (when present) are in
 *          [-0.5, 0.5). This must be a template so that the `if constexpr`
 *          branches are discarded when `x` (or its components) do not exist.
 * 
 * @param p         Particle values
 * @param i         Particle index inside tile (for error message)
 * @param tile_idx  Tile index (for error message)
 * @return int      1 if the position is invalid, 0 otherwise
 */
template< class P >
inline int check_cell_position( P const & p, int const i, int2 const tile_idx ) {
    int err = 0;
    if constexpr ( has_x< P >::value ) {
        using X = std::remove_cv_t< decltype( p.x ) >;
        if constexpr ( has_x< X >::value ) {
            if ( p.x.x < -0.5f || p.x.x >= 0.5f ) {
                std::printf( "[%d,%d] Invalid x[%d].x position (%f), range = [-0.5,0.5[\n",
                    tile_idx.x, tile_idx.y, i, double( p.x.x ) );
                err = 1;
            }
        }
        if constexpr ( has_y< X >::value ) {
            if ( p.x.y < -0.5f || p.x.y >= 0.5f ) {
                std::printf( "[%d,%d] Invalid x[%d].y position (%f), range = [-0.5,0.5[\n",
                    tile_idx.x, tile_idx.y, i, double( p.x.y ) );
                err = 1;
            }
        }
    }
    return err;
}

} // namespace particle_detail

// Compile time checks on quantity types
PARTICLES_QUANTITIES( PARTICLES_CHECK_TYPE_ )

/**
 * @brief Values of a single particle (all quantities)
 * 
 * @note Used to move particles between buffers, see `particles_view::load()`
 *       and `particles_view::store()`
 */
struct particle_t {
    /// @brief Particle position (cell index)
    int2 ix;
    // Additional quantities
    PARTICLES_QUANTITIES( PARTICLES_DECL_VALUE_ )
};

/**
 * @brief   Data structure to hold particle data
 * 
 * @warning This is meant to be used only as a superclass for particles. The
 *          struct does not include methods for allocating / deallocating
 *          memory
 * 
 * @note    Declaring a function parameter as `func(particles_view p)` and calling
 *          the function with a `particles` object parameter will automatically
 *          cast the value to `particles_view`. This means that we will not be
 *          creating a full copy of the `particles` object and therefore data
 *          will not be destroyed when the function reaches the end.
 */
struct particles_view {

    /// @brief Number of tiles (x,y)
    uint2 grid_ntiles;
    /// @brief Tile grid size
    const uint2 tile_dims;

    /// @brief Number of particles in tile
    int * tile_np;
    /// @brief Tile particle position on global array
    int * tile_offset;

    /// @brief Particle position (cell index)
    int2 *ix;
    // Additional quantities (pointers to device buffers)
    PARTICLES_QUANTITIES( PARTICLES_DECL_PTR_ )

    /// @brief Maximum number of particles in the buffer
    uint32_t max_part;

    /**
     * @brief Reads all quantities of a single particle
     * 
     * @param idx_          Particle index in global buffer
     * @return particle_t   Particle values
     */
    inline
    particle_t load( int const idx_ ) const {
        particle_t p_;
        p_.ix = this -> ix[ idx_ ];
#define PARTICLES_LOAD_( T, n, d ) p_.n = this -> n[ idx_ ];
        PARTICLES_QUANTITIES( PARTICLES_LOAD_ )
#undef PARTICLES_LOAD_
        return p_;
    }

    /**
     * @brief Writes all quantities of a single particle
     * 
     * @param idx_      Particle index in global buffer
     * @param p_        Particle values
     */
    inline
    void store( int const idx_, particle_t const & p_ ) const {
        this -> ix[ idx_ ] = p_.ix;
#define PARTICLES_STORE_( T, n, d ) this -> n[ idx_ ] = p_.n;
        PARTICLES_QUANTITIES( PARTICLES_STORE_ )
#undef PARTICLES_STORE_
    }

    /**
     * @brief Checks that all floating point values of a particle are finite
     * 
     * @note Prints an error message for each invalid quantity
     * 
     * @param idx_          Particle index in global buffer
     * @param tile_idx_     Tile index (for error message)
     * @return int          1 if any invalid values were found, 0 otherwise
     */
    int check_finite( int const idx_, int2 const tile_idx_ ) const {
        int err_ = 0;
#define PARTICLES_CHECK_FINITE_( T, n, d ) \
        if ( ! particle_detail::finite( this -> n[ idx_ ] ) ) { \
            std::printf( "[%d,%d] Invalid " #n "[%d] value (not finite)\n", \
                tile_idx_.x, tile_idx_.y, idx_ ); \
            err_ = 1; \
        }
        PARTICLES_QUANTITIES( PARTICLES_CHECK_FINITE_ )
#undef PARTICLES_CHECK_FINITE_
        return err_;
    }
};

/**
 * @brief Class for particle data
 * 
 */
class particles {

    /// @brief Number of tiles (x,y)
    uint2 grid_ntiles;
    /// @brief Tile grid size
    const uint2 tile_dims;

    /// @brief Number of particles in tile
    int * tile_np;
    /// @brief Tile particle position on global array
    int * tile_offset;

    /// @brief Particle position (cell index)
    int2 *ix;
    // Additional quantities (pointers to device buffers)
    PARTICLES_QUANTITIES( PARTICLES_DECL_PTR_ )

    /// @brief Maximum number of particles in the buffer
    uint32_t max_part;

    protected:

    /// @brief Periodic boundaries (x,y)
    int2 periodic;

    public:

    /**
     * @brief Construct a new particles object
     * 
     * @param ntiles            Number of tiles
     * @param tile_dims         Individual tile grid size
     * @param max_part          Maximum number of particles
     * @param periodic          Periodic boundaries
     */
    particles( const uint2 ntiles, const uint2 tile_dims, const uint32_t max_part, const int2 periodic ) :
        grid_ntiles( ntiles ), tile_dims( tile_dims ), max_part( max_part ),
        periodic( periodic )
    {
        // Validade parameters

        // Grid parameters
        if ( ntiles.x == 0 || ntiles.y == 0 ) {
            util::fatal( "Invalid number of tiles: " + to_string(ntiles) );
        }

        if ( tile_dims.x == 0 || tile_dims.y == 0 ) {
            util::fatal( "Invalid tile dimensions: " + to_string(tile_dims) );
        }

        ///@brief Total number of tiles
        const size_t bsize = part::grid_tiles( grid_ntiles );

        // Tile information
        tile_np = memory::malloc<int>( bsize );
        tile_offset = memory::malloc<int>( bsize );

        // Initially empty
        memory::zero( tile_np, bsize );
        memory::zero( tile_offset, bsize );

        // Particle data - cell
        ix = memory::malloc<int2>( max_part );
        // Particle data - remaining quantities
        PARTICLES_QUANTITIES( PARTICLES_ALLOC_ )
    }

    /**
     * @brief Destroy the particles object
     * 
     */
    ~particles() {
        PARTICLES_QUANTITIES( PARTICLES_FREE_ )
        memory::free( ix );

        memory::free( tile_offset );
        memory::free( tile_np );
    }

    particles( const particles & ) = delete;
    particles( particles && ) = delete;

    particles_view view() noexcept {
        return {
            grid_ntiles,
            tile_dims,
            tile_np,
            tile_offset,
            ix,
            PARTICLES_QUANTITIES( PARTICLES_NAME_LIST_ )
            max_part
        };
    }

    int * get_tile_offset() const noexcept {
        return tile_offset;
    };

    int * get_tile_np() const noexcept {
        return tile_np;
    }

    /**
     * @brief Get the tile dimensions
     * 
     * @return uint2 
     */
    uint2 get_tile_dims() const noexcept { return tile_dims; };

    /**
     * @brief Get the number of tiles
     * 
     * @return uint2 
     */
    uint2 get_grid_ntiles() const noexcept { return grid_ntiles; };

    /**
     * @brief Grid size
     * 
     * @return uint2 
     */
    uint2 get_grid_dims() const {
        return grid_ntiles * tile_dims;
    };

    /**
     * @brief Set periodic boundary settings
     * 
     * @param new_periodic 
     */
    void set_periodic( int2 new_periodic ) {
        periodic = new_periodic;
    }

    /**
     * @brief Get periodic boundary settings
     * 
     * @return int2 
     */
    int2 get_periodic( ) const { return periodic; }

    /**
     * @brief Sets the number of particles per tile to 0
     * 
     */
    void zero_np() {
        memory::zero( tile_np, part::grid_tiles( grid_ntiles ) );
    }

    /**
     * @brief Grows particle data buffers
     * 
     * @warning Particle data is not copied, previous values, if any,
     *          are destroyed
     * 
     * @param new_max   New buffer size. Will be rounded up to multiple
     *                  of 64k.
     */
    void grow_buffer( uint32_t new_max ) {
        if ( new_max > max_part ) {
            PARTICLES_QUANTITIES( PARTICLES_FREE_ )
            memory::free( ix );

            // Grow in multiples 64k blocks
            max_part = roundup<65536>(new_max);

            ix = memory::malloc<int2>( max_part );
            PARTICLES_QUANTITIES( PARTICLES_ALLOC_ )
        }
    }

    /**
     * @brief Swaps buffers between 2 particle objects
     * 
     * @param a     Object a
     * @param b     Object b
     */
    friend void swap_buffers( particles & a, particles & b ) {
        
        // assert( a.grid_ntiles == b.grid_ntiles );

        std::swap( a.ix, b.ix );
        PARTICLES_QUANTITIES( PARTICLES_SWAP_ )

        std::swap( a.max_part, b.max_part );

        std::swap( a.tile_np,     b.tile_np );
        std::swap( a.tile_offset, b.tile_offset );
    }

    /**
     * @brief Gets number of particles
     * 
     * @return uint32_t 
     */
    uint32_t np() const; // implemented in particles_impl.hpp

    /**
     * @brief Gets maximum number of particles in a single tile
     * 
     * @return int 
     */
    uint32_t tile_np_max() const; // implemented in particles_impl.hpp

    /**
     * @brief Gets minimum number of particles in a single tile
     * 
     * @return int 
     */
    uint32_t tile_np_min() const; // implemented in particles_impl.hpp

    /**
     * @brief Returns grid range
     * 
     * @return bounds_2d<uint32_t> 
     */
    bounds_2d<uint32_t> grid_range() const { 
        uint2 dims = get_grid_dims();
        
        bounds_2d<uint32_t> range;
        range.x = bounds<uint32_t>( 0, dims.x - 1 );
        range.y = bounds<uint32_t>( 0, dims.y - 1 );

        return range;
    };

    /**
     * @brief Validates particle data
     * 
     * @details In case of invalid particle data prints out an error message and aborts
     *          the program
     * 
     * @param msg   Message to print in case of error
     * @param over  Amount of extra cells indices beyond limit allowed. Used
     *              when checking the buffer before tile_sort(). Defaults to 0
     */
    void validate( std::string msg, int const over = 0 );

    /**
     * @brief Shifts particle cells by the required amount
     * 
     * @details Cells are shited by adding the parameter `shift` to the particle cell
     *          indexes
     * 
     * @note Does not check if the particles are still inside the tile after
     *       the shift
     * 
     * @param shift     Cell shift in both directions
     */
    void cell_shift( int2 const shift );
    
    /**
     * @brief Moves particles to the correct tiles
     * 
     * @warning This version of `tile_sort()` is provided for debug only;
     *          temporary buffers are created and destroyed on every call.
     * 
     * @param extra     (optional) Additional space to add to each tile. Leaves
     *                  room for particles to be injected later.
     */
    void tile_sort( const int * __restrict__ extra = nullptr ){
        // Create temporary buffers
        particles    tmp( grid_ntiles, tile_dims, max_part, periodic );
        part::particle_sort sort( grid_ntiles, max_part );
        
        // Call sort routine
        tile_sort( tmp, sort, extra );
    };

    /**
     * @brief Moves particles to the correct tiles
     * 
     * @note particles are only expected to have moved no more than 1 tile
     *       in each direction
     * 
     * @note If the particle buffer is too small to hold all particles after
     *       the sort (including `extra` space),
     *       the buffer is grown and all particles are copied in a single
     *       pass (`copy_sorted()`). The `tmp` and `sort` buffers are grown
     *       to match.
     * 
     * @param tmp       Temporary particle buffer
     * @param sort      Temporary sort index 
     * @param extra     (optional) Additional space to add to each tile. Leaves
     *                  room for particles to be injected later.
     */
    void tile_sort( particles &tmp, part::particle_sort &sort, 
                    const int * __restrict__ extra = nullptr ); 

    /**
     * @brief Size (in bytes) of a single particle
     * 
     * @return size_t 
     */
    static constexpr size_t particle_size() {
        return sizeof( int2 ) PARTICLES_QUANTITIES( PARTICLES_SIZE_SUM_ );
    };

    /**
     * @brief Print information on the number of particles per tile
     * 
     * @warning Used for debug purposes only
     * 
     * @param msg   (optional) Message to print before printing particle information
     */
    void info_np( std::string msg = "" ) {
        
        if ( ! msg.empty() ) {
            std::cout << "-------------[info]> " << msg << '\n';
        }

        std::cout << "#particles per tile:\n";

        for( unsigned j = 0; j < grid_ntiles.y; j++ ) {
            std::cout << j << ':';
            for( unsigned i = 0; i < grid_ntiles.x; i++ ) {
                int tid = j * grid_ntiles.x + i;
                std::cout << " " << tile_np[tid];
            }
            std::cout << '\n';
        }

        std::cout << "#particles total: " << np() << '\n';
    }
};

} // namespace PARTICLES_NAMESPACE

#undef PARTICLES_DECL_VALUE_
#undef PARTICLES_DECL_PTR_
#undef PARTICLES_NAME_LIST_
#undef PARTICLES_SIZE_SUM_
#undef PARTICLES_ALLOC_
#undef PARTICLES_FREE_
#undef PARTICLES_SWAP_
#undef PARTICLES_CHECK_TYPE_

#undef PARTICLES_QUANTITIES
#undef PARTICLES_NAMESPACE
