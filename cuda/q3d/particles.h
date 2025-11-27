#ifndef PARTICLES_H_
#define PARTICLES_H_

#include "zpic.h"

#include "vec_types.h"

#include "zdf-cpp.h"
#include "bnd.h"

namespace part {

/**
 * @brief Particle quantity identifiers
 * 
 */
enum quant { z, r, cos_th, sin_th, q, ux, uy, uz };

namespace bnd_t {
    enum type { none = 0, periodic };
}

/**
 * @brief Local boundary type
 * 
 */
typedef bnd<bnd_t::type> bnd_type;

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
 */
struct ParticleSortData {
    /// @brief Particle index list [max_part]
    int *idx;
    /// @brief Number of particles in index list [ntiles]
    int * nidx;
    /// @brief Number of particles leaving tile in all directions [ntiles * 9]
    int * npt;
    /// @brief New number of particles per tile
    int * new_np;
    /// @brief Total number of tiles
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
        idx    = device::malloc<int>( max_part );

        auto local_tiles = ntiles.x * ntiles.y;
        new_np = device::malloc<int>( local_tiles );
        // Number of particles leaving each local tile
        nidx   = device::malloc<int>( local_tiles );
        // Particles can move in 9 different directions
        npt    = device::malloc<int>( 9 * local_tiles );
    }

    /**
     * @brief Destroy the Particle Sort object
     * 
     */
    ~ParticleSort() {
        device::free( npt );
        device::free( nidx );
        device::free( new_np );
        device::free( idx );
    }

    /**
     * @brief Sets np values to 0
     * 
     */
    void reset( ) {
        device::zero( new_np, part::all_tiles( ntiles ) );
    }
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
 * 
 *          
 */
struct ParticleData {

    /// @brief Number of tiles (x,y)
    const uint2 ntiles;  
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
    float2 *th;

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

    private:

    /// @brief Device variable for returning uint32 values
    device::Var<uint32_t> dev_tmp_uint32;

    void bnd_check( ParticleSort & sort );
    void copy_out( ParticleData & tmp, const ParticleSortData & sort );
    void copy_in( ParticleData & tmp );
    void copy_sorted( ParticleData & tmp, const ParticleSortData & sort );

    public:

    /// @brief Periodic boundaries (z)
    int periodic_z;

    /**
     * @brief Construct a new Particles object
     * 
     * @param ntiles    Number of tiles
     * @param nx        Tile grid size
     * @param max_part  Maximum number of particles
     */
    Particles( const uint2 ntiles, const uint2 nx, const uint32_t max_part ) :
        ParticleData( ntiles, nx, max_part ), dev_tmp_uint32(), 
        dims( ntiles * nx ), periodic_z( 1 )
    {
        const size_t bsize = ntiles.x * ntiles.y;
        
        // Tile information
        np     = device::malloc<int>( bsize );
        offset = device::malloc<int>( bsize );

        // Initially empty
        device::zero( np, bsize );
        device::zero( offset, bsize );

        // Particle data
        ix = device::malloc<int2>  ( max_part );
        x  = device::malloc<float2>( max_part );
        u  = device::malloc<float3>( max_part );

        q  = device::malloc<float>( max_part );
        th = device::malloc<float2>( max_part );
    }

    ~Particles() {
        device::free( th );
        device::free( q );

        device::free( u );
        device::free( x );
        device::free( ix );

        device::free( offset );
        device::free( np );
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
    void zero_np( ) {
        device::zero( np, ntiles.x * ntiles.y );
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
            device::free( th );
            device::free( q );

            device::free( u );
            device::free( x );
            device::free( ix );

            // Grow in multiples 64k blocks
            max_part = roundup<65536>(new_max);

            ix = device::malloc<int2>( max_part );
            x  = device::malloc<float2>( max_part );
            u  = device::malloc<float3>( max_part );

            q  = device::malloc<float>( max_part );
            th  = device::malloc<float2>( max_part );

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
        swap( a.th,  b.th );

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
    uint32_t np_total();

    /**
     * @brief Gets maximum number of particles in a single tile
     * 
     * @return uint32_t 
     */
    uint32_t np_max_tile();

    /**
     * @brief Gets minimum number of particles in a single tile
     * 
     * @return uint32_t 
     */
    uint32_t np_min_tile();

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
    void gather( part::quant quant, float * const __restrict__ d_data );

    /**
     * @brief Gather data from a specific particle quantity, scaling values
     * 
     * @note Data (val) will be returned as `scale.x * val + scale.y`
     * 
     * @param quant     Quantity to gather
     * @param d_data    Output data buffer, assumed to have size >= np
     * @param scale     Scale factor for data
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
        Particles tmp( ntiles, nx, max_part );
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
     * @param info  Particle metadata (name, labels, units, etc.). Information is used to set file name
     * @param iter  Iteration metadata
     * @param path  Path where to save the file
     */
    void save( zdf::part_info &metadata, zdf::iteration &iter, std::string path );

    /**
     * @brief Print information on the number of particles per tile
     * 
     * @warning Used for debug purposes only
     * 
     * @param msg   (optional) Message to print before printing particle information
     */
    void info_np( std::string msg = "" ) {
        
        int * tmp_np = host::malloc<int>( part::all_tiles( ntiles ) );
        device::memcpy_tohost( tmp_np, np, 6 );

        if ( ! msg.empty() ) {
            std::cout << "-------------[info]> " << msg << '\n';
        }

        std::cout << '\n';
        std::cout << "#particles per tile:\n";

        uint32_t total = 0;

        for( unsigned j = 0; j < ntiles.y; j++ ) {
            std::cout << j << ':';
            for( unsigned i = 0; i < ntiles.x; i++ ) {
                int tid = j * ntiles.x + i;
                std::cout << " " << tmp_np[tid];
                total += tmp_np[tid];
            }
            std::cout << '\n';
        }

        std::cout << "#particles total: " << total << '\n';
    }
};

#endif
