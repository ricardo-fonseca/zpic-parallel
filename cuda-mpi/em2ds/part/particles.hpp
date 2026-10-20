#pragma once

#include "../parallel/partition.hpp"
#include "../core/vec_types.cuh"
#include "../core/bounds.hpp"
#include "../util/memory.cuh"
#include "../util/math.hpp"
#include "../zdf/zdf.hpp"


#include "bnd_type.hpp"
#include "part_tiles.cuh"
#include "message.hpp"
#include "sort.hpp"

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>

namespace part {

/**
 * @brief Particle quantity identifiers (for gather)
 * 
 */
enum class quantity { x, y, ux, uy, uz };


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
    /// @brief Particle position (position inside cell) normalized to cell size [-0.5,0.5)
    float2 *x;
    /// @brief Particle velocity
    float3 *u;

    /// @brief Maximum number of particles in the buffer
    uint32_t max_part;
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
    /// @brief Particle position (position inside cell) normalized to cell size [-0.5,0.5)
    float2 *x;
    /// @brief Particle velocity
    float3 *u;

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
    particle_message send;

    /// @brief Incoming particle data messages
    particle_message recv;

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

        // Particle data
        ix = gpu::device::malloc<int2>( max_part );
        x  = gpu::device::malloc<float2>( max_part );
        u  = gpu::device::malloc<float3>( max_part );

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
        gpu::device::free( u );
        gpu::device::free( x );
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
            x,
            u,
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
            gpu::device::free( u );
            gpu::device::free( x );
            gpu::device::free( ix );

            // Grow in multiples 64k blocks
            max_part = roundup<65536>(new_max);

            ix = gpu::device::malloc<int2>  ( max_part );
            x  = gpu::device::malloc<float2>( max_part );
            u  = gpu::device::malloc<float3>( max_part );
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
        std::swap( a.x,  b.x );
        std::swap( a.u,  b.u );

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
     * @brief Gather data from a specific particle quantity
     * 
     * @param quant     Quantity to gather
     * @param d_data    Output device data buffer, assumed to have size >= np
     */
    void gather( part::quantity quant, float * const __restrict__ d_data );

    /**
     * @brief Gather data from a specific particle quantity, scaling values
     * 
     * @note Data (val) will be returned as `scale.x * val + scale.y`
     * 
     * @param quant     Quantity to gather
     * @param scale     Scale factor for data
     * @param d_data    Output device data buffer, assumed to have size >= np
     */
    void gather( part::quantity quant, const float2 scale, float * const __restrict__ d_data );

    /**
     * @brief Save particle data to disk
     * 
     * @param quants    Quantities to save
     * @param metadata  Particle metadata (name, labels, units, etc.). Information is used to set file name
     * @param iter      Iteration metadata
     * @param path      Path where to save the file
     */
    void save( const part::quantity quants[], zdf::part_info &metadata, zdf::iteration &iter, std::string path );


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
        particle_sort sort( local_ntiles, max_part, parallel, periodic );
        
        // Call sort routine
        tile_sort( tmp, sort, extra );
    };

    /**
     * @brief Moves particles to the correct tiles
     * 
     * @note particles are only expected to have moved no more than 1 tile
     *       in each direction
     * 
     * @param tmp       Temporary particle buffer
     * @param sort      Temporary sort index 
     * @param extra     (optional) Additional space to add to each tile. Leaves
     *                  room for particles to be injected later.
     */
    void tile_sort( particles &tmp, particle_sort &sort, 
                    const int * __restrict__ extra = nullptr ); 

    /**
     * @brief Size (in bytes) of a single particle
     * 
     * @return size_t 
     */
    __host__ __device__
    static constexpr size_t particle_size() {
        return sizeof(*ix) + sizeof(*x) + sizeof(*u);
    };

    /**
     * @brief Byte offsets of each quantity block inside a packed message
     *
     * @details Messages are packed as `np` cell indices, followed by `np`
     *          cell positions, followed by `np` generalized velocities.
     *          Offsets are relative to the start of the message.
     */
    struct packed_offsets {
        private:
        static constexpr size_t size_ix = sizeof( * particles_view::ix );
        static constexpr size_t size_x  = sizeof( * particles_view::x  );
        static constexpr size_t size_u  = sizeof( * particles_view::u  );

        public:
        size_t ix, x, u;
        
        __host__ __device__
        explicit constexpr packed_offsets( size_t offset, uint32_t const total ) :
            ix( offset * size_ix                              ),
            x ( offset * size_x  + total * (size_ix          )),
            u ( offset * size_u  + total * (size_ix + size_x )) {}
    };

    /**
     * @brief Prepare particle receive buffers and start receive
     * 
     * @param sort      Temporary sort index
     * @param recv      Receive message object
     */
    void irecv_msg( particle_sort &sort, particle_message &recv );

    /**
     * @brief Pack particles moving out of the node into a message buffer and start send
     * 
     * @param tmp       Temporary buffer holding particles moving away from tiles
     * @param sort      Temporary sort index
     * @param send      Send message object
     */
    void isend_msg( particles &tmp, particle_sort &sort, particle_message &send );

    /**
     * @brief Unpack received particle data into main particle data buffer
     * 
     * @param sort      Temporary sort index
     * @param recv      Receive message object
     */
    void unpack_msg( particle_sort &sort, particle_message &recv );

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

} // end of part namespace

