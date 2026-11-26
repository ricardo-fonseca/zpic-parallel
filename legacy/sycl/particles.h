#ifndef PARTICLES_H_
#define PARTICLES_H_

#include "zpic.h"
#include "vec_types.h"
#include "bnd.h"

#include "zdf-cpp.h"

/**
 * @brief Particle quantity identifiers
 * 
 */
namespace part {
    enum quant { x, y, ux, uy, uz };
}

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
    const int ntiles;

    ParticleSortData( const int ntiles ) : ntiles(ntiles) {};
};

class ParticleSort : public ParticleSortData {
    public:

    sycl::queue & queue;

    ParticleSort( uint2 const ntiles, uint32_t const max_part, sycl::queue & queue ) :
        ParticleSortData( ntiles.x * ntiles.y ), queue(queue)
    {
        idx = device::malloc<int>( max_part, queue );

        const size_t bsize = ntiles.x * ntiles.y;
        new_np = device::malloc<int>( bsize, queue );
        nidx  = device::malloc<int>( bsize, queue );
        // Particles can move in 9 different directions
        npt   = device::malloc<int>( 9 * bsize, queue );
    }

    ~ParticleSort() {
        device::free( npt, queue );
        device::free( nidx, queue );
        device::free( new_np, queue );
        device::free( idx, queue );
    }

    /**
     * @brief Sets np values to 0
     * 
     */
    void reset( ) {
        device::zero( new_np, ntiles, queue );
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

    /// @brief Maximum number of particles in the buffer
    uint32_t max_part;

    ParticleData( const uint2 ntiles, const uint2 nx, const uint32_t max_part ) :
        ntiles( ntiles ), nx( nx ), max_part( max_part ) {};
};

/**
 * @brief Class for particle data
 * 
 */
class Particles : public ParticleData {

    private:

    /// @brief Device variable for returning uint32 values
    device::Var<uint32_t> _dev_tmp_uint32;

    public:

    /// @brief Associated sycl queue
    sycl::queue & queue;

    /// @brief Sets periodic boundaries (x,y)
    int2 periodic;

    /// Global grid size
    const uint2 gnx;

    /**
     * @brief Construct a new Particles object
     * 
     * @param ntiles    Number of tiles
     * @param nx        Tile grid size
     * @param max_part  Maximum number of particles
     */
    Particles( const uint2 ntiles, const uint2 nx, const uint32_t max_part, sycl::queue & queue ) :
        ParticleData( ntiles, nx, max_part ),
        _dev_tmp_uint32(queue), queue(queue),
        periodic( make_int2(1,1) ), gnx ( make_uint2( ntiles.x * nx.x, ntiles.y * nx.y ) )
    {
        const size_t bsize = ntiles.x * ntiles.y;
        
        // Tile information
        np     = device::malloc<int>( bsize, queue );
        offset = device::malloc<int>( bsize, queue );

        // Initially empty
        device::zero( np, bsize, queue );
        device::zero( offset, bsize, queue );

        // Particle data
        ix = device::malloc<int2>  ( max_part, queue );
        x  = device::malloc<float2>( max_part, queue );
        u  = device::malloc<float3>( max_part, queue );
    }

    ~Particles() {
        device::free( u, queue );
        device::free( x, queue );
        device::free( ix, queue );

        device::free( offset, queue );
        device::free( np, queue );
    }

    /**
     * @brief Sets the number of particles per tile to 0
     * 
     */
    void zero_np( ) {
        device::zero( np, ntiles.x * ntiles.y, queue );
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
            device::free( u , queue );
            device::free( x , queue );
            device::free( ix, queue );

            // Grow in multiples 64k blocks
            max_part = roundup<65536>(new_max);

            ix = device::malloc<int2>( max_part, queue );
            x  = device::malloc<float2>( max_part, queue );
            u  = device::malloc<float3>( max_part, queue );
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

        _dev_tmp_uint32 = 0;
        queue.submit([&](sycl::handler &h) {

            auto size = ntiles.x*ntiles.y;
            auto block = ( size < 8 ) ? size : 8 ;

            sycl::range<1> local{ block };
            sycl::range<1> global{ size };

            auto dev_np = _dev_tmp_uint32.ptr();
            auto np = this -> np;

            h.parallel_for( 
                sycl::nd_range{ global, local },
                [=](sycl::nd_item<1> it) {
                
                uint32_t local_np = np[ it.get_global_id(0) ];

                auto sg = it.get_sub_group();
                local_np = device::subgroup::reduce_add( sg, local_np );

                if ( sg.get_local_linear_id() == 0 ) {
                    auto v = sycl::atomic_ref<uint32_t,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                                (dev_np[0]);
                    v.fetch_add(local_np);
                }
            });
        });
        queue.wait();

        return _dev_tmp_uint32.get();
    }

    /**
     * @brief Gets maximum number of particles in a single tile
     * 
     * @return uint32_t 
     */
    uint32_t np_max_tile() {
        _dev_tmp_uint32 = 0;
        queue.submit([&](sycl::handler &h) {

            auto size = ntiles.x*ntiles.y;
            auto block = ( size < 1024 ) ? size : 1024 ;

            sycl::range<1> local{ block };
            sycl::range<1> global{ size };

            auto dev_max = _dev_tmp_uint32.ptr();
            auto np = this -> np;

            h.parallel_for( 
                sycl::nd_range{ global, local },
                [=](sycl::nd_item<1> it) {
                
                uint32_t local_max = 0;
                for( auto i = it.get_local_id(); i < size; i += it.get_local_range() ) {
                    auto tile_np = np[i];
                    if ( tile_np > local_max ) local_max = tile_np;
                }

                auto sg = it.get_sub_group();
                local_max = device::subgroup::reduce_max( sg, local_max );

                if ( sg.get_local_linear_id() == 0 ) {
                    auto v = sycl::atomic_ref<uint32_t,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                                (dev_max[0]);
                    v.fetch_max(local_max);
                }
            });
        });
        queue.wait();

       return _dev_tmp_uint32.get();
    }

    /**
     * @brief Gets minimum number of particles in a single tile
     * 
     * @return uint32_t 
     */
    uint32_t np_min_tile() {
        _dev_tmp_uint32 = 0;
        queue.submit([&](sycl::handler &h) {

            auto size = ntiles.x*ntiles.y;
            auto block = ( size < 1024 ) ? size : 1024 ;

            sycl::range<1> local{ block };
            sycl::range<1> global{ size };

            auto dev_min = _dev_tmp_uint32.ptr();
            auto np = this -> np;

            h.parallel_for( 
                sycl::nd_range{ global, local },
                [=](sycl::nd_item<1> it) {
                
                uint32_t local_min = std::numeric_limits<uint32_t>::max();
                for( auto i = it.get_local_id(); i < size; i += it.get_local_range() ) {
                    auto tile_min = np[i];
                    if ( tile_min < local_min ) local_min = tile_min;
                }

                auto sg = it.get_sub_group();
                local_min = device::subgroup::reduce_min( sg, local_min );

                if ( sg.get_local_linear_id() == 0 ) {
                    auto v = sycl::atomic_ref<uint32_t,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                                (dev_min[0]);
                    v.fetch_min(local_min);
                }
            });
        });
        queue.wait();

       return _dev_tmp_uint32.get();
    }

    /**
     * @brief Returns global grid range
     * 
     * @return bnd<uint32_t> 
     */
    bnd<uint32_t> g_range() { 
        bnd<uint32_t> range;
        range.x = { .lower = 0, .upper = gnx.x - 1 };
        range.y = { .lower = 0, .upper = gnx.y - 1 };

        return range;
    };

    /**
     * @brief Gather data from a specific particle quantity
     * 
     * @param quant     Quantity to gather
     * @param d_data    Output data buffer, assumed to have size >= np
     */
    void gather( part::quant quant, float * const __restrict__ d_data );

    void gather_host( part::quant quant, float * const __restrict__ d_data, float * const __restrict__ h_data,
    uint32_t const np );

    /**
     * @brief Gather data from a specific particle quantity, scaling values
     * 
     * @note Data (val) will be returned as `scale.x * val + scale.y`
     * 
     * @param quant     Quantity to gather
     * @param d_data    Output data buffer, assumed to have size >= np
     * @param scale     Scale factor for data
     */
    void gather( part::quant quant, float * const __restrict__ d_data, const float2 scale );

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
        Particles tmp( ntiles, nx, max_part, queue );
        ParticleSort sort( ntiles, max_part, queue );
        
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

};

#endif
