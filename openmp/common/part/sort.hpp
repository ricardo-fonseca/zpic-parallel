#pragma once

#include <cstdint>
#include "../core/vec_types.hpp"
#include "../util/memory.hpp"
#include "../util/math.hpp"

#include "part_tiles.hpp"

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
    /// @brief Number of particles in index list [ntiles]
    int * nidx;
    /// @brief Number of particles leaving tile in all directions [ntiles * 9]
    int * npt;
    /// @brief New number of particles per tile [ntiles]
    int * new_np;
    /// @brief Number of tiles
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
    /// @brief Number of particles in index list [ntiles]
    int * nidx;
    /// @brief Number of particles leaving tile in all directions [ntiles * 9]
    int * npt;
    /// @brief New number of particles per tile [ntiles]
    int * new_np;
    /// @brief Number of tiles
    const uint2 ntiles;

    private:

    /**
     * @brief Size of the per-tile counters buffer (new_np, nidx and npt)
     * 
     * @return int  Number of int elements
     */
    inline int counters_size() const {
        return 11 * part::grid_tiles( ntiles );
    }

    public:

    /**
     * @brief Construct a new Particle Sort object
     * 
     * @param ntiles        Number of tiles
     * @param max_part      Maximum number of particles in buffer
     */
    particle_sort( uint2 const ntiles, uint32_t const max_part ) :
        max_part( max_part ), ntiles( ntiles )
    {
        idx = memory::malloc<int>( max_part );

        // Number of tiles
        const int grid_tiles = part::grid_tiles(ntiles);

        // Single buffer for all per-tile counters, so they can be zeroed
        // together in reset():
        //   [ new_np (tiles) | nidx (tiles) | npt (9 * tiles) ]
        new_np = memory::malloc<int>( counters_size() );
        nidx   = &new_np[ grid_tiles ];
        npt    = &nidx[ grid_tiles ];

        memory::zero( new_np, counters_size() );
    }

    /**
     * @brief Destroy the Particle Sort object
     * 
     */
    ~particle_sort() {
        // new_np, nidx and npt share the same buffer
        memory::free( new_np );
        memory::free( idx );
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
            memory::free( idx );
            max_part = roundup<65536>(new_max);
            idx = memory::malloc<int>( max_part );
        }
    }

    /**
     * @brief Sets np values to 0
     * 
     * @note `bnd_check()` accumulates `new_np` using atomics, so it must be
     *       zeroed before it is called. `nidx` and `npt` share the same
     *       buffer, so all counters are zeroed with a single memset.
     */
    void reset() {
        memory::zero( new_np, counters_size() );
    }
};

}
