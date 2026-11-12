/**
 * @file particles.hpp
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
 *              #include "../common/particles.hpp"
 *
 *          and a local source file (e.g. `cyl_part/particles.cu`):
 *
 *              #include "particles.hpp"
 *              #define PARTICLES_NAMESPACE cyl_part
 *              #include "../common/particles_template.cu"
 *
 *          The cell index `int2 ix` is always present and must not be listed.
 *          Each entry is `X( type, name, description )`; `description` is
 *          documentation only. Quantity types must be trivially copyable and
 *          have a size that is a multiple of 4 bytes (e.g. float, float2,
 *          float3, double, double2, int, uint64_t).
 *
 *          Quantity names must not collide with the names of the members of
 *          `particles_view` / `particles` (e.g. ix, tile_np, tile_offset,
 *          max_part, load, store, pack, unpack, ...).
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
 * @note    The helper files (bnd_type.hpp, comms.hpp, part_tiles.cuh,
 *          message.hpp, sort.hpp) do not depend on the particle quantities
 *          and are shared by all particle types, in the `part` namespace.
 */

#ifndef PARTICLES_NAMESPACE
#error "PARTICLES_NAMESPACE must be defined before including common/particles.hpp"
#endif

#ifndef PARTICLES_QUANTITIES
#error "PARTICLES_QUANTITIES must be defined before including common/particles.hpp"
#endif

#include "../parallel/partition.hpp"
#include "../core/vec_types.cuh"
#include "../core/bounds.hpp"
#include "../util/memory.cuh"
#include "../util/math.hpp"

#include "bnd_type.hpp"
#include "part_tiles.cuh"
#include "message.hpp"
#include "sort.hpp"

#include <cstdlib>
#include <cstdint>
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
/// @brief Declare quantity offset (e.g. `size_t x;`)
#define PARTICLES_DECL_OFFSET_( T, n, d )   size_t n;
/// @brief Comma separated list of quantity names
#define PARTICLES_NAME_LIST_( T, n, d )     n,
/// @brief Sum of quantity sizes
#define PARTICLES_SIZE_SUM_( T, n, d )      + sizeof( T )
/// @brief Allocate device buffer for quantity
#define PARTICLES_ALLOC_( T, n, d )         n = gpu::device::malloc< T >( max_part );
/// @brief Free device buffer for quantity
#define PARTICLES_FREE_( T, n, d )          gpu::device::free( n );
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
__device__ inline bool finite( float  const v ) { return ::isfinite( v ); }
__device__ inline bool finite( double const v ) { return ::isfinite( v ); }
__device__ inline bool finite( float2 const v ) { return ::isfinite( v.x ) && ::isfinite( v.y ); }
__device__ inline bool finite( float3 const v ) { return ::isfinite( v.x ) && ::isfinite( v.y ) && ::isfinite( v.z ); }
__device__ inline bool finite( float4 const v ) { return ::isfinite( v.x ) && ::isfinite( v.y ) && ::isfinite( v.z ) && ::isfinite( v.w ); }
__device__ inline bool finite( double2 const v ) { return ::isfinite( v.x ) && ::isfinite( v.y ); }
__device__ inline bool finite( double3 const v ) { return ::isfinite( v.x ) && ::isfinite( v.y ) && ::isfinite( v.z ); }
template< class T >
__device__ inline bool finite( T const & ) { return true; }

/**
 * @brief Copies particle quantity data into a (packed) message buffer
 * 
 * @note Data is copied as 32 bit words, so that it works with the 4 byte
 *       alignment of the packed message buffer.
 * 
 * @warning Must be called by all threads in the block
 */
template< class T >
__device__ inline void pack_words( uint8_t * __restrict__ dst, T const * __restrict__ src, int const np ) {
    gpu::block::memcpy( reinterpret_cast< uint32_t * >( dst ),
                        reinterpret_cast< uint32_t const * >( src ),
                        np * ( sizeof( T ) / 4 ) );
}

/**
 * @brief Copies particle quantity data from a (packed) message buffer
 * 
 * @warning Must be called by all threads in the block
 */
template< class T >
__device__ inline void unpack_words( T * __restrict__ dst, uint8_t const * __restrict__ src, int const np ) {
    gpu::block::memcpy( reinterpret_cast< uint32_t * >( dst ),
                        reinterpret_cast< uint32_t const * >( src ),
                        np * ( sizeof( T ) / 4 ) );
}

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
 * @brief Byte offsets of each quantity block inside a packed message
 *
 * @details Messages are packed as `np` cell indices, followed by `np`
 *          values of each of the additional quantities, in the order
 *          they are defined. Offsets are relative to the start of the
 *          message.
 */
struct packed_offsets {
    size_t ix;
    PARTICLES_QUANTITIES( PARTICLES_DECL_OFFSET_ )

    /**
     * @brief Construct a new packed offsets object
     * 
     * @param offset_   Number of particles in the message before this group
     * @param total_    Total number of particles in the message
     */
    __host__ __device__
    explicit packed_offsets( size_t const offset_, uint32_t const total_ ) {
        /// @brief Size (bytes) of all previous quantities (single particle)
        [[maybe_unused]] size_t prev_ = sizeof( int2 );
        ix = offset_ * sizeof( int2 );
#define PARTICLES_SET_OFFSET_( T, n, d ) \
        n = offset_ * sizeof( T ) + total_ * prev_; prev_ += sizeof( T );
        PARTICLES_QUANTITIES( PARTICLES_SET_OFFSET_ )
#undef PARTICLES_SET_OFFSET_
    }
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

    /// @brief Global number of tiles (x,y)
    uint2 global_ntiles;
    /// @brief Local Number of tiles (x,y)
    uint2 local_ntiles;
    /// @brief Tile grid size
    const uint2 tile_dims;
    /// @brief Start position of local tiles in global tile grid
    uint2 local_tile_start;

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
    __device__ __forceinline__
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
    __device__ __forceinline__
    void store( int const idx_, particle_t const & p_ ) const {
        this -> ix[ idx_ ] = p_.ix;
#define PARTICLES_STORE_( T, n, d ) this -> n[ idx_ ] = p_.n;
        PARTICLES_QUANTITIES( PARTICLES_STORE_ )
#undef PARTICLES_STORE_
    }

    /**
     * @brief Packs particle data into a message buffer
     * 
     * @warning Must be called by all threads in the block
     * 
     * @param buffer_   Message buffer (for this direction)
     * @param off_      Offsets inside message buffer
     * @param src_      Index of first particle to pack in global buffer
     * @param np_       Number of particles to pack
     */
    __device__
    void pack( uint8_t * __restrict__ buffer_, packed_offsets const & off_, int const src_, int const np_ ) const {
        particle_detail::pack_words( &buffer_[ off_.ix ], &( this -> ix[ src_ ] ), np_ );
#define PARTICLES_PACK_( T, n, d ) \
        particle_detail::pack_words( &buffer_[ off_.n ], &( this -> n[ src_ ] ), np_ );
        PARTICLES_QUANTITIES( PARTICLES_PACK_ )
#undef PARTICLES_PACK_
    }

    /**
     * @brief Unpacks particle data from a message buffer
     * 
     * @warning Must be called by all threads in the block
     * 
     * @param buffer_   Message buffer (for this direction)
     * @param off_      Offsets inside message buffer
     * @param dst_      Index of first particle in global buffer
     * @param np_       Number of particles to unpack
     */
    __device__
    void unpack( uint8_t const * __restrict__ buffer_, packed_offsets const & off_, int const dst_, int const np_ ) const {
        particle_detail::unpack_words( &( this -> ix[ dst_ ] ), &buffer_[ off_.ix ], np_ );
#define PARTICLES_UNPACK_( T, n, d ) \
        particle_detail::unpack_words( &( this -> n[ dst_ ] ), &buffer_[ off_.n ], np_ );
        PARTICLES_QUANTITIES( PARTICLES_UNPACK_ )
#undef PARTICLES_UNPACK_
    }

    /**
     * @brief Checks that all floating point values of a particle are finite
     * 
     * @note Prints an error message for each invalid quantity
     * 
     * @param idx_          Particle index in global buffer
     * @param rank_         MPI rank (for error message)
     * @param tile_idx_     Tile index (for error message)
     * @return int          1 if any invalid values were found, 0 otherwise
     */
    __device__
    int check_finite( int const idx_, int const rank_, int2 const tile_idx_ ) const {
        int err_ = 0;
#define PARTICLES_CHECK_FINITE_( T, n, d ) \
        if ( ! particle_detail::finite( this -> n[ idx_ ] ) ) { \
            printf( "[%d] [%d,%d] Invalid " #n "[%d] value (not finite)\n", \
                rank_, tile_idx_.x, tile_idx_.y, idx_ ); \
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

    /// @brief Global number of tiles (x,y)
    uint2 global_ntiles;
    /// @brief Local Number of tiles (x,y)
    uint2 local_ntiles;
    /// @brief Tile grid size
    const uint2 tile_dims;
    /// @brief Start position of local tiles in global tile grid
    uint2 local_tile_start;

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

    /// @brief Device variable for returning uint32 values
    gpu::device::var<uint32_t> dev_tmp_uint32;

    protected:

    /// @brief Global periodic boundaries (x,y)
    int2 periodic;

    /// @brief Local node boundary type
    part::bnd_type local_bnd;

    /// @brief Outgoing particle data messages
    part::particle_message send;

    /// @brief Incoming particle data messages
    part::particle_message recv;

    public:

    /// @brief Parallel partition
    mpi::cart2d & parallel;

    /**
     * @brief Construct a new particles object
     * 
     * @param global_ntiles     Global number of tiles
     * @param tile_dims         Individual tile grid size
     * @param max_part          Maximum number of particles
     * @param parallel          Parallel partition
     * @param periodic          Periodic boundaries
     */
    particles( const uint2 global_ntiles, const uint2 tile_dims, const uint32_t max_part, mpi::cart2d & parallel, const int2 periodic ) :
        global_ntiles( global_ntiles ), tile_dims( tile_dims ), max_part( max_part ),
        send( parallel, periodic ),
        recv( parallel, periodic ),
        parallel( parallel ),
        periodic( periodic )
    {
        // Validade parameters

        // Grid parameters
        if ( global_ntiles.x == 0 || global_ntiles.y == 0 ) {
            mpi::fatal( "Invalid number of tiles: " + to_string(global_ntiles) );
        }

        if ( tile_dims.x == 0 || tile_dims.y == 0 ) {
            mpi::fatal( "Invalid tile dimensions: " + to_string(tile_dims) );
        }

        // Parallel partition
        if ( parallel.dims.x > global_ntiles.x ) {
            mpi::fatal ( "Number of parallel nodes along x (" +
                         std::to_string(parallel.dims.x) +
                         ") is larger than number of tiles along x(" +
                         std::to_string(global_ntiles.x) + ')');
        }

        if ( parallel.dims.y > global_ntiles.y ) {
            mpi::fatal ( "Number of parallel nodes along y (" +
                         std::to_string(parallel.dims.y) +
                         ") is larger than number of tiles along y(" +
                         std::to_string(global_ntiles.y) + ')');
        }

        // Periodic boundaries (x)
        if ( ( periodic.x ) && 
             ( (! parallel.periodic.x ) && ( parallel.dims.x > 1 )) ) {
            mpi::fatal( "Periodic boundaries (x) on non-parallel comm direction." );
        }

        // Periodic boundaries (y)
        if ( ( periodic.y ) && 
             ( (! parallel.periodic.y ) && ( parallel.dims.y > 1 )) ) {
            mpi::fatal( "Periodic boundaries (y) on non-parallel comm direction." );
        }

        // Get local number of tiles and position on tile grid
        parallel.grid_local( global_ntiles, local_ntiles, local_tile_start );
        
        ///@brief Total number of local tiles including edge tiles
        const size_t bsize = part::all_tiles( local_ntiles );

        // Tile information
        tile_np = gpu::device::malloc<int>( bsize );
        tile_offset = gpu::device::malloc<int>( bsize );

        // Check shared memory (for update_tile_info)
        auto local_mem_size = gpu::block::shared_mem_size();
        if ( local_mem_size < bsize * sizeof(int) ) {
            std::ostringstream msg;
            msg << "Too many local tiles " << bsize 
                << ", insufficient shared memory (" << local_mem_size << " B) for particles object.";
            mpi::fatal( msg.str() );
        }

        // Initially empty
        gpu::device::zero( tile_np, bsize );
        gpu::device::zero( tile_offset, bsize );

        // Particle data - cell
        ix = gpu::device::malloc<int2>( max_part );
        // Particle data - remaining quantities
        PARTICLES_QUANTITIES( PARTICLES_ALLOC_ )

        // Set local bnd values
        update_local_bnd();
    }

    /**
     * @brief Construct a new particles object
     *
     * @note Periodic boundaries are assumed to be the same as the parallel partition
     * 
     * @param global_ntiles     Global number of tiles
     * @param tile_dims         Individual tile grid size
     * @param max_part          Maximum number of particles
     * @param parallel          Parallel partition
     */
    particles( const uint2 global_ntiles, const uint2 tile_dims, const uint32_t max_part, mpi::cart2d & parallel ) :
        particles( global_ntiles, tile_dims, max_part, parallel, parallel.periodic ) {}

    /**
     * @brief Destroy the particles object
     * 
     */
    ~particles() {
        PARTICLES_QUANTITIES( PARTICLES_FREE_ )
        gpu::device::free( ix );

        gpu::device::free( tile_offset );
        gpu::device::free( tile_np );
    }

    particles( const particles & ) = delete;
    particles( particles && ) = delete;

    particles_view view() noexcept {
        return {
            global_ntiles,
            local_ntiles,
            tile_dims,
            local_tile_start,
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
     * @brief Get the local number of tiles
     * 
     * @return uint2 
     */
    uint2 get_tile_dims() const noexcept { return tile_dims; };

    /**
     * @brief Get the local number of tiles
     * 
     * @return uint2 
     */
    uint2 get_local_ntiles() const noexcept { return local_ntiles; };

    /**
     * @brief Returns the local tile offset in the global MPI tile grid
     * 
     * @return uint2 
     */
    uint2 get_local_tile_start() const noexcept { return local_tile_start; };

    /**
     * @brief Get the global number of tiles
     * 
     * @return uint2 
     */
    uint2 get_global_ntiles() const noexcept { return global_ntiles; };


    /**
     * @brief Local grid size
     * 
     * @return uint2 
     */
    uint2 get_local_dims() const {
        return local_ntiles * tile_dims;
    };

    /**
     * @brief Global grid size
     * 
     * @return uint2 
     */
    uint2 get_global_dims() const {
        return global_ntiles * tile_dims;
    }

    /**
     * @brief Get the parallel partition
     * 
     * @return const mpi::cart2d& 
     */
    const mpi::cart2d & get_parallel() const noexcept { return  parallel; }

    /**
     * @brief Update local node boundary types
     * 
     */
    void update_local_bnd() {
        
        // Default to none
        local_bnd = part::bnd_t::none;

        // Get communication boundaries
        if ( part::shift_neighbor(-1, 0, parallel, periodic) >= 0 ) local_bnd.x.lower = part::bnd_t::comm;
        if ( part::shift_neighbor(+1, 0, parallel, periodic) >= 0 ) local_bnd.x.upper = part::bnd_t::comm;

        if ( part::shift_neighbor( 0,-1, parallel, periodic) >= 0 ) local_bnd.y.lower = part::bnd_t::comm;
        if ( part::shift_neighbor( 0,+1, parallel, periodic) >= 0 ) local_bnd.y.upper = part::bnd_t::comm;

        // Correct for local node periodic
        if ( periodic.x && parallel.dims.x == 1 ) 
            local_bnd.x.lower = local_bnd.x.upper = part::bnd_t::periodic;

        if ( periodic.y && parallel.dims.y == 1 ) 
            local_bnd.y.lower = local_bnd.y.upper = part::bnd_t::periodic;

        send.update_neighbors(parallel, periodic);
        recv.update_neighbors(parallel, periodic);
    }

    /**
     * @brief Get local node boundary types
     * 
     */
    part::bnd_type get_local_bnd() const {
        return local_bnd;
    }

    /**
     * @brief Set global periodic boundary settings
     * 
     * @param new_periodic 
     */
    void set_periodic( int2 new_periodic ) {
        // Check x direction
        if ( ( new_periodic.x ) && 
             ( (! parallel.periodic.x ) && ( parallel.dims.x > 1 )) ) {
            mpi::fatal( "particles::set_periodic() - Attempting to set "
                        "parallel x boundaries on non-parallel comm direction." );
        }

        // Check y direction
        if ( ( new_periodic.y ) && 
             ( (! parallel.periodic.y ) && ( parallel.dims.y > 1 )) ) {
            mpi::fatal( "particles::set_periodic() - Attempting to set "
                        "parallel y boundaries on non-parallel comm direction" );
        }

        // Store new global periodic values
        periodic = new_periodic;

        // update local bnd values
        update_local_bnd();
    }

    /**
     * @brief Get global periodic boundary settings
     * 
     * @return int2 
     */
    int2 get_periodic( ) const { return periodic; }

    /**
     * @brief Sets the number of particles per tile to 0
     * 
     */
    void zero_np() {
        gpu::device::zero( tile_np, part::all_tiles( local_ntiles ) );
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
            gpu::device::free( ix );

            // Grow in multiples 64k blocks
            max_part = roundup<65536>(new_max);

            ix = gpu::device::malloc<int2>( max_part );
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
        
        // assert( a.local_ntiles == b.local_ntiles );
        // assert( a.global_ntiles == b.global_ntiles );

        std::swap( a.ix, b.ix );
        PARTICLES_QUANTITIES( PARTICLES_SWAP_ )

        std::swap( a.max_part, b.max_part );

        std::swap( a.tile_np,     b.tile_np );
        std::swap( a.tile_offset, b.tile_offset );
    }

    /**
     * @brief Gets (node) local number of particles
     * 
     * @return uint32_t 
     */
    uint32_t local_np(); // implemented in particles.cu

    /**
     * @brief Gets global number of particles
     *
     * @note When all = 0 (default) returns 0 on non-root ranks
     * 
     * @param all           Return result on all parallel nodes (defaults to false)
     * @return uint64_t     Global number of particles
     */
    uint64_t global_np( bool all = false ) {

        uint64_t local = local_np();

        if ( parallel.get_size() > 1 ) {
            if ( all ) {
                uint64_t global;
                parallel.allreduce( &local, &global, 1, mpi::sum );
                return global;
            } else {
                parallel.reduce( &local, 1, mpi::sum );
                if ( ! parallel.root() ) local = 0;
                return local;
            }
        }

        return local;
    }

    /**
     * @brief Gets maximum number of particles in a single tile
     * 
     * @return int 
     */
    uint32_t tile_np_max(); // implemented in particles.cu

    /**
     * @brief Gets minimum number of particles in a single tile
     * 
     * @return int 
     */
    uint32_t tile_np_min(); // implemented in particles.cu

    /**
     * @brief Returns local grid range
     * 
     * @return bounds_2d<uint32_t> 
     */
    bounds_2d<uint32_t> local_range() const { 
        uint2 dims = get_local_dims();
        
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
        particles    tmp( global_ntiles, tile_dims, max_part, parallel );
        part::particle_sort sort( local_ntiles, max_part, parallel, periodic );
        
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
     *       the sort (including incoming MPI particles and `extra` space),
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
    __host__ __device__
    static constexpr size_t particle_size() {
        return sizeof( int2 ) PARTICLES_QUANTITIES( PARTICLES_SIZE_SUM_ );
    };

    /// @brief Byte offsets of each quantity block inside a packed message
    using packed_offsets = PARTICLES_NAMESPACE::packed_offsets;

    /**
     * @brief Prepare particle receive buffers and start receive
     * 
     * @param sort      Temporary sort index
     * @param recv      Receive message object
     */
    void irecv_msg( part::particle_sort &sort, part::particle_message &recv );

    /**
     * @brief Pack particles moving out of the node into a message buffer and start send
     * 
     * @param tmp       Temporary buffer holding particles moving away from tiles
     * @param sort      Temporary sort index
     * @param send      Send message object
     */
    void isend_msg( particles &tmp, part::particle_sort &sort, part::particle_message &send );

    /**
     * @brief Unpack received particle data into main particle data buffer
     * 
     * @param sort      Temporary sort index
     * @param recv      Receive message object
     */
    void unpack_msg( part::particle_sort &sort, part::particle_message &recv );

    /**
     * @brief Print information on the number of particles per tile
     * 
     * @warning Used for debug purposes only
     * 
     * @param msg   (optional) Message to print before printing particle information
     */
    void info_np( std::string msg = "" ) {
        
        parallel.barrier();

        int * h_tile_np = gpu::host::malloc<int>( local_ntiles.x * local_ntiles.y );
        gpu::device::memcpy_tohost( h_tile_np, tile_np, local_ntiles.x * local_ntiles.y );

        
        if ( ! msg.empty() && parallel.root() ) {
            std::cout << "-------------[info]> " << msg << '\n';
        }

        uint64_t np = local_np();

        for( int k = 0; k < parallel.get_size() ; k++ ) {
            if ( k == parallel.get_rank() ) {
                std::cout << '\n';
                mpi::cout << "#particles per tile:\n";

                for( unsigned j = 0; j < local_ntiles.y; j++ ) {
                    mpi::cout << j << ':';
                    for( unsigned i = 0; i < local_ntiles.x; i++ ) {
                        int tid = j * local_ntiles.x + i;
                        mpi::cout << " " << h_tile_np[tid];
                    }
                    mpi::cout << '\n';
                }

                mpi::cout << "#particles total: " << np << '\n';
            }
            parallel.barrier();
        }

        parallel.reduce( &np, 1, mpi::sum, 0 );
        if ( parallel.root() ) {
            mpi::cout << "#particles all: " << np << '\n';
        }

        gpu::host::free( h_tile_np );
    }
};

} // namespace PARTICLES_NAMESPACE

#undef PARTICLES_DECL_VALUE_
#undef PARTICLES_DECL_PTR_
#undef PARTICLES_DECL_OFFSET_
#undef PARTICLES_NAME_LIST_
#undef PARTICLES_SIZE_SUM_
#undef PARTICLES_ALLOC_
#undef PARTICLES_FREE_
#undef PARTICLES_SWAP_
#undef PARTICLES_CHECK_TYPE_

#undef PARTICLES_QUANTITIES
#undef PARTICLES_NAMESPACE
