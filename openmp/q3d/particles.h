#ifndef PARTICLES_H_
#define PARTICLES_H_

#include "zpic.h"
#include "vec_types.h"

#include "bnd.h"

#include "zdf-cpp.h"

namespace part {

/**
 * @brief Particle quantity identifiers
 * 
 */
enum quant { z, r, cosθ, sinθ, q, ux, uy, uz };

namespace bnd_t {
    enum type { none = 0, periodic };
}

/**
 * @brief Local boundary type
 * 
 */
typedef bnd<bnd_t::type> bnd_type;


/**
 * @brief Gets tile id from coordinates
 * 
 * @param coords        Tile coordinates
 * @param ntiles        Tile grid dimensions
 * @param periodic_z    Periodic boundary (along z) information
 * @return int          Tile id on success, -1 on out of bounds
 */
inline int tid_coords( int2 coords, int2 const ntiles, int const periodic_z ) {

    // x periodic
    if ( periodic_z ) {
        if      ( coords.x < 0 )         coords.x += ntiles.x; 
        else if ( coords.x >= ntiles.x ) coords.x -= ntiles.x;
    }

    // Out of bounds (non-periodic boundary)
    int outx = ( coords.x >= ntiles.x ) || ( coords.x < 0 );
    int outy = ( coords.y >= ntiles.y ) || ( coords.y < 0 );

    return ( outx || outy ) ? -1 : coords.y * ntiles.x + coords.x;
}

/**
 * @brief Total number of tiles
 * 
 * @param ntiles    Local number of tiles (x,y)
 * @return int      Total number of tiles
 */
inline constexpr int all_tiles( const uint2 ntiles ) {
    return ntiles.x * ntiles.y;
}

}

/**
 * @brief   Data structure to hold particle sort data
 * 
 * @warning This is meant to be used only as a superclass for ParticleSort. The
 *          struct does not include methods for allocating / deallocating
 *          memory
 *
 * 
 */
struct ParticleSortData {
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

    ParticleSortData( const uint2 ntiles ) : 
        ntiles(ntiles) {};
};

/**
 * @brief Class for particle sorting data
 * 
 * @note This class does not hold any actual particle data, only particle
 *       inidices and counts. It should work for any type of particle data.
 * 
 */
class ParticleSort : public ParticleSortData {

    private:
    
    public:

    /**
     * @brief Construct a new Particle Sort object
     * 
     * @param ntiles        Local number of tiles
     * @param max_part      Maximum number of particles in buffer
     */
    ParticleSort( uint2 const ntiles, uint32_t const max_part ) :
        ParticleSortData( ntiles )
    {
        idx = memory::malloc<int>( max_part );

        auto local_tiles = ntiles.x * ntiles.y;

        new_np = memory::malloc<int>( local_tiles );
        
        // Number of particles leaving each local tile
        nidx   = memory::malloc<int>( local_tiles );

        // Particle can move in 9 different directions
        npt = memory::malloc<int>( 9 * local_tiles );
    }

    /**
     * @brief Destroy the Particle Sort object
     * 
     */
    ~ParticleSort() {
        memory::free( npt );
        memory::free( nidx );
        memory::free( new_np );
        memory::free( idx );
    }

    /**
     * @brief Sets np values to 0
     * 
     */
    void reset() {
        // Reset local data and outgoing data buffer
        memory::zero( new_np, part::all_tiles( ntiles ) );
    }

    /**
     * @brief Exchange number of particles in edge cells
     *
     */
    void exchange_np( );

};


/**
 * @brief   Data structure to hold particle data
 * 
 * @warning This is meant to be used only as a superclass for Particles. The
 *          struct does not include methods for allocating / deallocating
 *          memory
 * 
 * @note    Declaring a function parameter as `func(ParticleData p)` and calling
 *          the function with a `Particles` object parameter will automatically
 *          cast the value to `ParticleData`. This means that we will not be
 *          creating a full copy of the `Particles` object and therefore data
 *          will not be destroyed when the function reaches the end.
 */
struct ParticleData {

    /// @brief Number of tiles (x,y)
    uint2 ntiles;
    /// @brief Tile grid size
    const uint2 nx;

    /// @brief Number of particles in tile
    int * np;
    /// @brief Tile particle position on global array
    int * offset;

    /// @brief Particle position (cell index)
    int2 *ix;
    /// @brief Particle position (position inside cell) normalized to cell size [-0.5,0.5)
    float2 *x;
    /// @brief Particle velocity
    float3 *u;

    /// @brief Individual particle charge
    float *q;

    /// @brief Angular position stored as { cos(θ), sin(θ) }
    float2 *θ;

    /// @brief Maximum number of particles in the buffer
    uint32_t max_part;

    ParticleData( const uint2 ntiles, const uint2 nx, const uint32_t max_part ) :
        ntiles( ntiles ),
        nx( nx ),
        max_part( max_part ) {};
};

/**
 * @brief Class for particle data
 * 
 */
class Particles : public ParticleData {

    protected:

    /// @brief Grid size
    uint2 dims;

    public:

    /// @brief Periodic boundaries (z)
    int periodic_z;

    /**
     * @brief Construct a new Particles object
     * 
     * @param global_ntiles     Global number of tiles
     * @param nx                Individual tile grid size
     * @param max_part          Maximum number of particles
     */
    Particles( const uint2 global_ntiles, const uint2 nx, const uint32_t max_part ) :
        ParticleData( global_ntiles, nx, max_part ),
        dims( ntiles * nx ), periodic_z( 1 )
    {        
        const size_t bsize = ntiles.y * ntiles.x;

        // Tile information
        np = memory::malloc<int>( bsize );
        offset = memory::malloc<int>( bsize );

        // Initially empty
        memory::zero( np, bsize );
        memory::zero( offset, bsize );

        // Particle data
        ix = memory::malloc<int2>( max_part );
        x  = memory::malloc<float2>( max_part );
        u  = memory::malloc<float3>( max_part );

        q  = memory::malloc<float>( max_part );
        θ  = memory::malloc<float2>( max_part );
    }

    ~Particles() {
        memory::free( θ );
        memory::free( q );


        memory::free( u );
        memory::free( x );
        memory::free( ix );

        memory::free( offset );
        memory::free( np );
    }

    /**
     * @brief Get global periodic boundary settings for z
     * 
     * @return int
     */
    auto get_periodic_z( ) { return periodic_z; }

    /**
     * @brief Get the local grid size
     * 
     * @return auto
     */
    auto get_dims() { return dims; }

    /**
     * @brief Sets the number of particles per tile to 0
     * 
     */
    void zero_np() {
        memory::zero( np, ntiles.y * ntiles.x );
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
            memory::free( θ );
            memory::free( q );

            memory::free( u );
            memory::free( x );
            memory::free( ix );

            // Grow in multiples 64k blocks
            max_part = roundup<65536>(new_max);

            ix = memory::malloc<int2>  ( max_part );
            x  = memory::malloc<float2>( max_part );
            u  = memory::malloc<float3>( max_part );

            q  = memory::malloc<float>( max_part );
            θ  = memory::malloc<float2>( max_part );

        }
    }

    /**
     * @brief Swaps buffers between 2 particle objects
     * 
     * @param a     Object a
     * @param b     Object b
     */
    friend void swap_buffers( Particles & a, Particles & b ) {
        swap( a.ix, b.ix );
        swap( a.x,  b.x );
        swap( a.u,  b.u );

        swap( a.q,  b.q );
        swap( a.θ,  b.θ );


        auto tmp_max_part = b.max_part;
        b.max_part = a.max_part;
        a.max_part = tmp_max_part;

        swap( a.np,     b.np );
        swap( a.offset, b.offset );
    }

    /**
     * @brief Gets total number of particles
     * 
     * @return uint32_t 
     */
    uint32_t np_total() {

        // sum up number of particles in each tile
        // This works even if the buffer is not compact
        uint32_t np_total = 0;
        for( unsigned i = 0; i < ntiles.x*ntiles.y; i++ )
            np_total += np[i];

/*
        // Since the buffer is kept compact we could just look at the last tile
        auto idx = ntiles.x*ntiles.y - 1;
        uint32_t np_total = offset[idx] + np[idx];
*/
        return np_total;
    }

    /**
     * @brief Gets maximum number of particles in a single tile
     * 
     * @return uint32_t 
     */
    uint32_t np_max_tile() {
        int max_np = np[0];
        for( unsigned i = 1; i < ntiles.x*ntiles.y; i++ ) {
            if ( np[i] > max_np ) max_np = np[i];
        }
        return max_np;
    }

    /**
     * @brief Gets minimum number of particles in a single tile
     * 
     * @return uint32_t 
     */
    uint32_t np_min_tile() {
        int min_np = np[0];
        for( unsigned i = 1; i < ntiles.x*ntiles.y; i++ ) {
            if ( np[i] < min_np ) min_np = np[i];
        }
        return min_np;
    }
    /**
     * @brief Returns local grid range
     * 
     * @return bnd<uint32_t> 
     */
    bnd<uint32_t> local_range() { 
        bnd<uint32_t> range;
        range.x = pair<uint32_t>( 0, dims.x - 1 );
        range.y = pair<uint32_t>( 0, dims.y - 1 );

        return range;
    };

    /**
     * @brief Gather data from a specific particle quantity
     * 
     * @param quant     Quantity to gather
     * @param d_data    Output data buffer, assumed to have size >= np
     */
    void gather( part::quant quant, float * const __restrict__ d_data  );

    /**
     * @brief Gather data from a specific particle quantity, scaling values
     * 
     * @note Data (val) will be returned as `scale.x * val + scale.y`
     * 
     * @param quant     Quantity to gather
     * @param scale     Scale factor for data
     * @param d_data    Output data buffer, assumed to have size >= np
     */
    void gather( part::quant quant, const float2 scale, float * const __restrict__ d_data );

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
     *          temporary buffers are created and removed each time the
     *          function is called.
     * 
     * @param extra     (optional) Additional space to add to each tile. Leaves
     *                  room for particles to be injected later.
     */
    void tile_sort( const int * __restrict__ extra = nullptr ){
        // Create temporary buffers
        Particles    tmp( ntiles, nx, max_part );
        ParticleSort sort( ntiles, max_part );
        
        // Call sort routine
        tile_sort( tmp, sort, extra );
    };

    /**
     * @brief Moves particles to the correct tiles
     * 
     * @note Particles are only expected to have moved no more than 1 tile
     *       in each direction
     * 
     * @param tmp       Temporary particle buffer
     * @param sort      Temporary sort index 
     * @param extra     (optional) Additional space to add to each tile. Leaves
     *                  room for particles to be injected later.
     */
    void tile_sort( Particles &tmp, ParticleSort &sort, 
                    const int * __restrict__ extra = nullptr ); 

    /**
     * @brief Save particle data to disk
     * 
     * @param quants    Quantities to save
     * @param metadata  Particle metadata (name, labels, units, etc.). Information is used to set file name
     * @param iter      Iteration metadata
     * @param path      Path where to save the file
     */
    void save( const part::quant quants[], zdf::part_info &metadata, zdf::iteration &iter, std::string path );

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

        std::cout << '\n';
        std::cout << "#particles per tile:\n";

        for( unsigned j = 0; j < ntiles.y; j++ ) {
            std::cout << j << ':';
            for( unsigned i = 0; i < ntiles.x; i++ ) {
                int tid = j * ntiles.x + i;
                std::cout << " " << np[tid];
            }
            std::cout << '\n';
        }

        std::cout << "#particles total: " << np_total() << '\n';
    }
};

#endif
