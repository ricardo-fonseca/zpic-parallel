#pragma once

#include <cstdint>
#include "../core/vec_types.cuh"
#include "../parallel/partition.hpp"
#include "../core/gpu.cuh"
#include "../util/memory.cuh"
#include "../util/math.hpp"

#include "comms.hpp"

namespace part {

/**
 * @brief   Data structure to hold particle sort data
 * 
 * @warning This is meant to be used only as a superclass for particle_sort. The
 *          struct does not include methods for allocating / deallocating
 *          memory
 *
 * 
 */
struct particle_sort_view {
    /// @brief Maximum number of particles to be sorted
    uint32_t max_part;
    /// @brief Particle index list [max_part]
    int *idx;
    /// @brief Number of particles in index list [local_ntiles]
    int * nidx;
    /// @brief Number of particles leaving tile in all directions [ntiles * 9]
    int * npt;
    /**
     * @brief New number of particles per tile
     * @note  Includes incoming/outgoing particles per edge tile
     */
    int * new_np;
    /// @brief Local number of tiles
    const uint2 ntiles;
};

/**
 * @brief Class for particle sorting data
 * 
 * @note This class does not hold any actual particle data, only particle
 *       inidices and counts. It should work for any type of particle data.
 * 
 */
class particle_sort {

    /// @brief Maximum number of particles to be sorted
    uint32_t max_part;
    /// @brief Particle index list [max_part]
    int *idx;
    /// @brief Number of particles in index list [local_ntiles]
    int * nidx;
    /// @brief Number of particles leaving tile in all directions [ntiles * 9]
    int * npt;
    /**
     * @brief New number of particles per tile
     * @note  Includes incoming/outgoing particles per edge tile
     */
    int * new_np;
    /// @brief Local number of tiles
    const uint2 ntiles;

    struct message {
        /// @brief Buffer for all 8 messages
        int * buffer;
        /// @brief Number of incoming particles per message (in device memory)
        int * d_msg_np;
        /// @brief Number of incoming particles per message (in host memory)
        int * h_msg_np;
        /// @brief Total number of particles to be exchanged
        int total_np;
        /// @brief Message requests
        MPI_Request requests[9];
    };

    /// @brief MPI communicator
    MPI_Comm comm;
    /// @brief Neighbor ranks
    int neighbor[9];

    private:

    /**
     * @brief Message tag for incoming messages
     * 
     * @param dir   Communication direction (0-8)
     * @return int 
     */
    inline int source_tag( int dir ) {
        return (8 - dir) | 0x100;
    }

    /**
     * @brief Message tag for outgoing messages
     * 
     * @param dir   Communication direction (0-8) 
     * @return int 
     */
    inline int dest_tag( int dir ) {
        return dir | 0x100;
    }
    
    public:

    /// @brief Incoming messages
    particle_sort::message recv;
    /// @brief Outgoing messages
    particle_sort::message send;

    /**
     * @brief Construct a new Particle Sort object
     * 
     * @param ntiles        Local number of tiles
     * @param max_part      Maximum number of particles in buffer
     * @param par           Parallel partition
     */
    particle_sort( uint2 const ntiles, uint32_t const max_part, mpi::cart2d & par, const int2 periodic ) :
        max_part( max_part ), ntiles( ntiles )
    {
        idx = gpu::device::malloc<int>( max_part );

        // Number of local ltiles
        int local_tiles = part::local_tiles(ntiles);

        // Number of edge tiles for messaging
        int msg_tiles = part::msg_tiles(ntiles);

         // Include send buffer for number of particles leaving node
        new_np = gpu::device::malloc<int>( local_tiles + msg_tiles );
        
        // Number of particles leaving each local tile
        nidx   = gpu::device::malloc<int>( local_tiles );

        // Particle can move in 9 different directions
        npt = gpu::device::malloc<int>( 9 * local_tiles );

        // Send buffer
        send.buffer = &new_np[ local_tiles ];
        send.d_msg_np = gpu::device::malloc<int>(9);
        send.h_msg_np = gpu::host::malloc<int>(9);
        gpu::host::zero( send.h_msg_np, 9 );
        gpu::device::zero( send.d_msg_np, 9 );

        // Receive buffer
        recv.buffer = gpu::device::malloc<int>( msg_tiles );
        gpu::device::zero( recv.buffer, msg_tiles );
        
        recv.d_msg_np = gpu::device::malloc<int>(9);
        recv.h_msg_np = gpu::host::malloc<int>(9);
        gpu::host::zero( recv.h_msg_np, 9 );
        gpu::device::zero( recv.d_msg_np, 9 );

        // MPI Communicator
        comm = par.get_comm();


        // Neighbor MPI ranks and messages
        for( int dir = 0; dir < 9; dir++ ) {
            // This will disable messages to self
            neighbor[ dir ] = part::dir_neighbor( dir, par, periodic );
            // Initialize messages
            recv.requests[dir] = send.requests[dir] = MPI_REQUEST_NULL;
        }
    }

    /**
     * @brief Destroy the Particle Sort object
     * 
     */
    ~particle_sort() {
        gpu::device::free( send.d_msg_np );
        gpu::device::free( recv.d_msg_np );

        gpu::host::free( send.h_msg_np );
        gpu::host::free( recv.h_msg_np );

        gpu::device::free( recv.buffer );

        gpu::device::free( npt );
        gpu::device::free( nidx );
        gpu::device::free( new_np );
        gpu::device::free( idx );
    }

    particle_sort( const particle_sort & ) = delete;
    particle_sort( particle_sort && ) = delete;

    particle_sort_view view() const noexcept {
        return {
            max_part,
            idx,
            nidx,
            npt,
            new_np,
            ntiles
        };
    }

    /**
     * @brief Checks if idx buffer is large enough, grows if needed
     *
     * @note idx buffer is grown in multiples of 64k elements
     * 
     * @param new_max   New maximum number of particles required
     */
    void check_buffer( uint32_t new_max ) {
        if ( new_max > max_part ) {
            gpu::device::free( idx );
            max_part = roundup<65536>(new_max);
            idx = gpu::device::malloc<int>( max_part );
        }
    }

    /**
     * @brief Sets np values to 0
     * 
     */
    void reset() {
        // Reset local data and outgoing data buffer
        gpu::device::zero( new_np, part::all_tiles( ntiles ) );
    }

    /**
    * @brief Exchange number of particles in edge cells
    *
    */
    void exchange_np();

    void update_neighbors( const mpi::cart2d & par, const int2 periodic ) {
        for( int dir = 0; dir < 9; dir++ )
            neighbor[ dir ] = part::dir_neighbor( dir, par, periodic );
    }
};


}