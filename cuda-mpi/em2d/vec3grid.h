#ifndef VEC3_GRID_H_
#define VEC3_GRID_H_

#include "vec_types.h"
#include "grid.h"

 /**
 * @brief Field components (x,y,z)
 * 
 */
namespace fcomp {
    enum cart  { x = 0, y, z };
}

/**
 * @brief CUDA Kernels for the various functions
 * 
 * @note This namespace is kept anonymous so these functions can only be
 *       accessed from within this file
 */
namespace {

/**
 * @brief CUDA kernel for gather operation
 * 
 * @tparam fc           Field component
 * @tparam S            Datatype (scalar)
 * @tparam V            Datatype (vector)
 * @param d_out         Output buffer
 * @param d_buffer      Input buffer (includes offset to cell [0,0])
 * @param ntiles        Number of tiles
 * @param nx            Tile size
 * @param ext_nx        Tile size (including guard cells)
 */
template< fcomp::cart fc, typename S, typename V >
__global__
void gather_kernel( 
    S * const __restrict__ d_out, V const * const __restrict__ d_buffer,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx )
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    const uint2  local_nx = { ntiles.x * nx.x, ntiles.y * nx.y };

    auto * const __restrict__ tile_data = & d_buffer[ tile_off ];

    for( int i = block_thread_rank(); i < nx.x * nx.y; i+= block_num_threads() ) {
        const auto ix = i % nx.x;
        const auto iy = i / nx.x;

        auto const gix = tile_idx.x * nx.x + ix;
        auto const giy = tile_idx.y * nx.y + iy;

        auto const out_idx = giy * local_nx.x + gix;

        if constexpr ( fc == fcomp::x ) d_out[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].x;
        if constexpr ( fc == fcomp::y ) d_out[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].y;
        if constexpr ( fc == fcomp::z ) d_out[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].z;
    }
}

}
/**
 * End of CUDA kernels namespace
 */


template< typename T >
struct vec3_scalar {
    T vec3;
    
    // Verify the components exist and are arithmetic
    static_assert( 
        std::is_arithmetic_v<decltype(vec3.x)> &&
        std::is_arithmetic_v<decltype(vec3.y)> &&
        std::is_arithmetic_v<decltype(vec3.z)>,
        "x, y and z components must be arithmetic"
    );
    
    // Verify they are all of the same type
    static_assert( 
        std::is_same_v<decltype(vec3.x), decltype(vec3.y)> &&
        std::is_same_v<decltype(vec3.x), decltype(vec3.z)>,
        "x, y and z components must all be of the same type"
    );

    public:
    using type = decltype(vec3.x);
};

template < typename V > 
class vec3grid : public grid< V >
{
    protected:

    // Get the base scalar type
    using S = typename vec3_scalar<V>::type;

    using grid< V > :: ntiles;
    using grid< V > :: tile_off;
    using grid< V > :: local_periodic;

    using grid< V > :: initialize;

    public:

    using grid< V > :: part;
    using grid< V > :: name;

    using grid< V > :: d_buffer;
    using grid< V > :: nx;

    using grid< V > :: gc;

    using grid< V > :: global_ntiles;
    using grid< V > :: local_nx;
    using grid< V > :: ext_nx;

    using grid< V > :: offset;

    using grid< V > :: tile_vol;

    using grid< V > :: grid;
    using grid< V > :: operator=;

    using grid< V > :: gather;
    using grid< V > :: copy_to_gc_x;
    using grid< V > :: copy_to_gc_y;

    /**
     * @brief Gather specific field component values from all tiles into a 
     * contiguous grid
     * 
     * Used mostly for diagnostic output
     * 
     * @param fc                Field component to output
     * @param out               Scalar output buffer
     * @return unsigned int     Total number of cells
     */
    unsigned int gather( const enum fcomp::cart fc, S * const __restrict__ d_out ) {

        dim3 block( 64 );
        dim3 grid( ntiles.x, ntiles.y );

        switch( fc ) {
        case( fcomp::x ):
            gather_kernel<fcomp::x> <<< grid, block >>> (
                d_out, d_buffer + offset,
                ntiles, nx, ext_nx );
            break;
        case( fcomp::y ):
            gather_kernel<fcomp::y> <<< grid, block >>> (
                d_out, d_buffer + offset,
                ntiles, nx, ext_nx );
            break;
        case( fcomp::z ):
            gather_kernel<fcomp::z> <<< grid, block >>> (
                d_out, d_buffer + offset,
                ntiles, nx, ext_nx );
            break;
        default:
            ABORT( "vec3grid::gather() - Invalid fc");
        }        

        return local_nx.x * local_nx.y;
    }

    /**
     * @brief Save specific field component to disk
     * 
     * @note The scalar field type <S> must be supported by ZDF file format
     * 
     * @param fc    Field component to save
     * @param info  Grid metadata (label, units, axis, etc.). Information is used to set file name
     * @param iter  Iteration metadata
     * @param path  Path where to save the file
     */
    void save( const enum fcomp::cart fc, zdf::grid_info &info, zdf::iteration &iter, std::string path ) {

        // Fill in grid dimensions
        info.ndims = 2;
        info.count[0] = global_ntiles.x * nx.x;
        info.count[1] = global_ntiles.y * nx.y;

        const std::size_t bsize = local_nx.x * local_nx.y;

        // Allocate buffers on host and device to gather data
        S * h_data = host::malloc<S>( bsize );
        S * d_data = device::malloc<S>( bsize );

        // Gather field component data on contiguous grid
        gather( fc, d_data );

        // Copy to host and free device memory
        device::memcpy_tohost( h_data, d_data, bsize );
        device::free( d_data );

        // Information on local chunk of grid data
        zdf::chunk chunk;
        chunk.count[0] = local_nx.x;
        chunk.count[1] = local_nx.y;
        chunk.start[0] = tile_off.x * nx.x;
        chunk.start[1] = tile_off.y * nx.y;
        chunk.stride[0] = chunk.stride[1] = 1;
        chunk.data = (void *) h_data;

        // Save data
        zdf::save_grid<S>( chunk, info, iter, path, part.get_comm() );

        // Free remaining temporary buffer        
        host::free( h_data );
    }

    /**
     * @brief Save grid values to disk
     * 
     * @param fc        Field component to save
     * @param filename  Output file name (includes path)
     */
    void save( const enum fcomp::cart fc, std::string filename ) {
        
        const std::size_t bsize = local_nx.x * local_nx.y;

        // Allocate buffers on host and device to gather data
        S * h_data = host::malloc<S>( bsize );
        S * d_data = device::malloc<S>( bsize );

        // Gather data on contiguous grid
        gather( fc, d_data );

        // Copy to host and free device memory
        device::memcpy_tohost( h_data, d_data, bsize );
        device::free( d_data );

        uint64_t global[2] = { global_ntiles.x * nx.x, global_ntiles.y * nx.y };
        uint64_t start[2]  = { tile_off.x * nx.x, tile_off.y * nx.y };
        uint64_t local[2]  = { local_nx.x, local_nx.y };

        // Save data
        std::string comp[] = { "x", "y", "z" };

        zdf::save_grid( h_data, 2, global, start, local, name + "-" + comp[fc], filename, part.get_comm() );

        // Free remaining temporary buffer 
        host::free( h_data );
    }

};

#endif