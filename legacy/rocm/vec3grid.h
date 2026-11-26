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
 * @tparam T 
 * @param d_out     Output buffer
 * @param d_buffer  Input buffer (includes offset to cell [0,0])
 * @param gnx       Global grid size
 * @param nx        Local grid size
 * @param ext_nx    Local grid size including guard cells
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

    const uint2  gnx = { ntiles.x * nx.x, ntiles.y * nx.y };

    auto * const __restrict__ tile_data = & d_buffer[ tile_off ];

    for( int i = block_thread_rank(); i < nx.x * nx.y; i+= block_num_threads() ) {
        const auto ix = i % nx.x;
        const auto iy = i / nx.x;

        auto const gix = tile_idx.x * nx.x + ix;
        auto const giy = tile_idx.y * nx.y + iy;

        auto const out_idx = giy * gnx.x + gix;

        if ( fc == fcomp::x ) d_out[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].x;
        if ( fc == fcomp::y ) d_out[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].y;
        if ( fc == fcomp::z ) d_out[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].z;
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
    
    public:

    using grid<V> :: d_buffer;
    using grid<V> :: nx;

    using grid<V> :: gc;

    using grid<V> :: ntiles;
    using grid<V> :: gnx;
    using grid<V> :: ext_nx;

    using grid<V> :: offset;

    using grid<V> :: tile_vol;

    using grid<V> :: grid;
    using grid<V> :: operator=;

    using grid<V> :: gather;

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

        return gnx.x * gnx.y;
    }

    /**
     * @brief Save specific field component to disk
     * 
     * The scalar field type <S> must be supported by ZDF file format
     * 
     * @param fc    Field component to save
     * @param info  Grid metadata (label, units, axis, etc.). Information is used to set file name
     * @param iter  Iteration metadata
     * @param path  Path where to save the file
     */
    void save( const enum fcomp::cart fc, zdf::grid_info &info, zdf::iteration &iter, std::string path ) {

        // Fill in grid dimensions
        info.ndims = 2;
        info.count[0] = gnx.x;
        info.count[1] = gnx.y;

        const std::size_t bsize = gnx.x * gnx.y;

        S * d_data = device::malloc<S>( bsize );
        S * h_data = host::malloc<S>( bsize );

        gather( fc, d_data );

        device::memcpy_tohost( h_data, d_data, bsize );

        zdf::save_grid( h_data, info, iter, path );

        host::free( h_data );
        device::free( d_data );
    }

    void save( const enum fcomp::cart fc, std::string path ) {
        
        std::cout << "in vec3grid::save( fc, path )\n";

        // Prepare file info
        zdf::grid_axis axis[2];
        axis[0] = (zdf::grid_axis) {
            .name = (char *) "x",
            .min = 0.,
            .max = 1. * gnx.x,
            .label = (char *) "x",
            .units = (char *) ""
        };

        axis[1] = (zdf::grid_axis) {
            .name = (char *) "y",
            .min = 0.,
            .max = 1. * gnx.y,
            .label = (char *) "y",
            .units = (char *) ""
        };

        std::string fcomp_name[] = {"x","y","z"};

        std::string grid_name = "cuda-vec3-" + fcomp_name[fc];
        std::string grid_label = "cuda vec3 test (" + fcomp_name[fc] + ")";

        zdf::grid_info info = {
            .name = (char *) grid_name.c_str(),
            .label = (char *) grid_label.c_str(),
            .units = (char *) "",
            .axis  = axis
        };

        zdf::iteration iter = {
            .name = (char *) "ITERATION",
            .n = 0,
            .t = 0,
            .time_units = (char *) ""
        };

        save( fc, info, iter, path );
    }

};

#endif