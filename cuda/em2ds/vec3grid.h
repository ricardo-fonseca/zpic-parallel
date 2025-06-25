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
namespace vec3_grid_kernel{

/**
 * @brief CUDA kernel for gather operation (single component)
 * 
 * @tparam fc       Field component
 * @tparam S        Scalar type
 * @tparam V        Vector type 
 * @param d_out     (out) Output buffer
 * @param d_buffer  (in) Input buffer (includes offset to cell [0,0])
 * @param ntiles    Number of tiles
 * @param nx        tile grid size
 * @param ext_nx    tile grid size including guard cells
 */
template< fcomp::cart fc, typename S, typename V >
__global__
void gather_fcomp( 
    S * const __restrict__ d_out, V const * const __restrict__ d_buffer,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx )
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    const uint2  dims = { ntiles.x * nx.x, ntiles.y * nx.y };

    auto * const __restrict__ tile_data = & d_buffer[ tile_off ];

    for( int i = block_thread_rank(); i < nx.x * nx.y; i+= block_num_threads() ) {
        const auto ix = i % nx.x;
        const auto iy = i / nx.x;

        auto const gix = tile_idx.x * nx.x + ix;
        auto const giy = tile_idx.y * nx.y + iy;

        auto const out_idx = giy * dims.x + gix;

        if constexpr ( fc == fcomp::x ) d_out[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].x;
        if constexpr ( fc == fcomp::y ) d_out[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].y;
        if constexpr ( fc == fcomp::z ) d_out[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].z;
    }
}

/**
 * @brief CUDA kernel for scatter operation (single component)
 * 
 * @tparam fc       Field component
 * @tparam S        Scalar type
 * @tparam V        Vector type 
 * @param d_buffer  (out) Tile buffer (includes offset to cell [0,0])
 * @param ntiles    Number of tiles
 * @param nx        tile grid size
 * @param ext_nx    tile grid size including guard cells
 * @param d_in      (in) Contiguous data buffer
 */
template< fcomp::cart fc, typename S, typename V >
__global__
void scatter_fcomp( 
    V const * const __restrict__ d_buffer,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx,
    S * const __restrict__ d_in )
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    const uint2  dims = { ntiles.x * nx.x, ntiles.y * nx.y };

    auto * const __restrict__ tile_data = & d_buffer[ tile_off ];

    for( int i = block_thread_rank(); i < nx.x * nx.y; i+= block_num_threads() ) {
        const auto ix = i % nx.x;
        const auto iy = i / nx.x;

        auto const gix = tile_idx.x * nx.x + ix;
        auto const giy = tile_idx.y * nx.y + iy;

        auto const out_idx = giy * dims.x + gix;

        if constexpr ( fc == fcomp::x ) tile_data[ iy * ext_nx.x + ix ].x = d_in[ out_idx ];
        if constexpr ( fc == fcomp::y ) tile_data[ iy * ext_nx.x + ix ].y = d_in[ out_idx ];
        if constexpr ( fc == fcomp::z ) tile_data[ iy * ext_nx.x + ix ].z = d_in[ out_idx ];
    }
}

/**
 * @brief CUDA kernel for gather operation (all components)
 * 
 * @note The output array will be organized in 3 sequential blocks (i.e. SoA):
 *       x components, y components and z components
 * 
 * @tparam S        Scalar type
 * @tparam V        Vector type 
 * @param d_out     (out) Output buffer
 * @param d_buffer  (in) Input buffer
 * @param ntiles    Number of tiles
 * @param nx        tile grid size
 * @param ext_nx    tile grid size including guard cells
 */
template< typename S, typename V >
__global__
void gather( 
    S * const __restrict__ d_out, V const * const __restrict__ d_buffer, const unsigned int offset,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx )
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    const uint2  dims = { ntiles.x * nx.x, ntiles.y * nx.y };

    S * const __restrict__ out_x = & d_out[                             0 ];
    S * const __restrict__ out_y = & d_out[     dims.x * dims.y ];
    S * const __restrict__ out_z = & d_out[ 2 * dims.x * dims.y ];

    // Copy tile data into shared memory
    V * __restrict__ local = block::shared_mem<V>();
    block::memcpy( local, & d_buffer[ tile_off ], tile_vol );
    block_sync();

    for( int i = block_thread_rank(); i < nx.x * nx.y; i+= block_num_threads() ) {
        const auto ix = i % nx.x;
        const auto iy = i / nx.x;

        auto const gix = tile_idx.x * nx.x + ix;
        auto const giy = tile_idx.y * nx.y + iy;

        auto const out_idx = giy * dims.x + gix;
        auto const idx     = iy * ext_nx.x + ix + offset;

        out_x[ out_idx ] = local[ idx ].x;
        out_y[ out_idx ] = local[ idx ].y;
        out_z[ out_idx ] = local[ idx ].z;
    }

}

/**
 * @brief CUDA kernel for scatter operation (all components)
 * 
 * @note The input array will be organized in 3 sequential blocks (i.e. SoA):
 *       x components, y components and z components
 * 
 * @tparam S        Scalar type
 * @tparam V        Vector type 
 * @param d_buffer  (out) Tile buffer (includes offset to cell [0,0])
 * @param ntiles    Number of tiles
 * @param nx        tile grid size
 * @param ext_nx    tile grid size including guard cells
 * @param d_in      (in) Input buffer ( 3 x contiguous grids )
 */
template< typename S, typename V >
__global__
void scatter( 
    V * const __restrict__ d_buffer, const unsigned int offset,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx,
    S * const __restrict__ d_in )
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    const uint2  dims = { ntiles.x * nx.x, ntiles.y * nx.y };

    S * __restrict__ in_x = & d_in[                 0 ];
    S * __restrict__ in_y = & d_in[     dims.x * dims.y ];
    S * __restrict__ in_z = & d_in[ 2 * dims.x * dims.y ];

    auto * shm = block::shared_mem<V>();
    V * __restrict__ local = & shm[0];

    for( int i = block_thread_rank(); i < nx.x * nx.y; i+= block_num_threads() ) {
        const auto ix = i % nx.x;
        const auto iy = i / nx.x;

        auto const gix = tile_idx.x * nx.x + ix;
        auto const giy = tile_idx.y * nx.y + iy;

        auto const in_idx = giy * dims.x + gix;

        local[ iy * ext_nx.x + ix + offset ].x = in_x[ in_idx ] ;
        local[ iy * ext_nx.x + ix + offset ].y = in_y[ in_idx ] ;
        local[ iy * ext_nx.x + ix + offset ].z = in_z[ in_idx ] ;
    }

    block_sync();
    block::memcpy( & d_buffer[ tile_off ], local, tile_vol );
}

template< typename S, typename V >
__global__
void scatter_scale( 
    V * const __restrict__ d_buffer, const unsigned int offset,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx,
    S const * const __restrict__ d_in, S const scale )
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    const uint2  dims = { ntiles.x * nx.x, ntiles.y * nx.y };

    S const * const __restrict__ in_x = & d_in[                             0 ];
    S const * const __restrict__ in_y = & d_in[     dims.x * dims.y ];
    S const * const __restrict__ in_z = & d_in[ 2 * dims.x * dims.y ];

    auto * shm = block::shared_mem<V>();
    V * __restrict__ local = & shm[0];

    for( int i = block_thread_rank(); i < nx.y * nx.x; i+= block_num_threads() ) {
        const auto ix = i % nx.x;
        const auto iy = i / nx.x;

        auto const gix = tile_idx.x * nx.x + ix;
        auto const giy = tile_idx.y * nx.y + iy;

        auto const idx    = iy * ext_nx.x + ix + offset;
        auto const in_idx = giy * dims.x + gix;

        local[ idx ].x = scale * in_x[ in_idx ] ;
        local[ idx ].y = scale * in_y[ in_idx ] ;
        local[ idx ].z = scale * in_z[ in_idx ] ;
    }

    block_sync();
    block::memcpy( & d_buffer[ tile_off ], local, tile_vol );
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

    using grid< V > :: d_buffer;
    using grid< V > :: ntiles;
    using grid< V > :: nx;
    using grid< V > :: dims;
    using grid< V > :: gc;
    using grid< V > :: ext_nx;
    using grid< V > :: offset;
    using grid< V > :: tile_vol;
    using grid< V > :: name;
    using grid< V > :: periodic;

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
            vec3_grid_kernel::gather_fcomp<fcomp::x> <<< grid, block >>> (
                d_out, d_buffer + offset,
                ntiles, nx, ext_nx );
            break;
        case( fcomp::y ):
            vec3_grid_kernel::gather_fcomp<fcomp::y> <<< grid, block >>> (
                d_out, d_buffer + offset,
                ntiles, nx, ext_nx );
            break;
        case( fcomp::z ):
            vec3_grid_kernel::gather_fcomp<fcomp::z> <<< grid, block >>> (
                d_out, d_buffer + offset,
                ntiles, nx, ext_nx );
            break;
        default:
            ABORT( "vec3grid::gather() - Invalid fc");
        }        

        return dims.x * dims.y;
    }

    /**
     * @brief Gather all field component values from tiles into a 
     *        contiguous grid
     * 
     * @note The output array will be organized in 3 sequential blocks (i.e. SoA):
     *       x components, y components and z components
     * 
     * @param out               Scalar output buffer
     * @return unsigned int     Total number of cells * 3
     */
    unsigned int gather( S * const __restrict__ d_out ) {
        dim3 block( 64 );
        dim3 grid( ntiles.x, ntiles.y );

        size_t shm_size = tile_vol * sizeof(V);
        block::set_shmem_size( vec3_grid_kernel::gather<S,V>, shm_size );

        vec3_grid_kernel::gather<S,V> <<< grid, block, shm_size >>> (
                d_out, d_buffer, offset,
                ntiles, nx, ext_nx );     

        return 3 * dims.x * dims.y;
    }

    /**
     * @brief Scatter specific field component values from contiguous grid
     *        into tiles
     * 
     * @param fc                Field component to input
     * @param d_in              Scalar output buffer
     * @return unsigned int     Total number of cells
     */
    unsigned int scatter( const enum fcomp::cart fc, S * const __restrict__ d_in ) {

        dim3 block( 64 );
        dim3 grid( ntiles.x, ntiles.y );

        switch( fc ) {
        case( fcomp::x ):
            vec3_grid_kernel::scatter_fcomp<fcomp::x> <<< grid, block >>> (
                d_buffer + offset,
                ntiles, nx, ext_nx,
                d_in );
            break;
        case( fcomp::y ):
            vec3_grid_kernel::scatter_fcomp<fcomp::y> <<< grid, block >>> (
                d_buffer + offset,
                ntiles, nx, ext_nx,
                d_in );
            break;
        case( fcomp::z ):
            vec3_grid_kernel::scatter_fcomp<fcomp::z> <<< grid, block >>> (
                d_buffer + offset,
                ntiles, nx, ext_nx,
                d_in );
            break;
        default:
            ABORT( "vec3grid::scatter() - Invalid fc");
        }        

        return dims.x * dims.y;
    }

    /**
     * @brief Scatter all field component values from contiguous grid
     *        into tiles
     *
     * @note Input array must be organized in 3 sequential blocks (i.e. SoA):
     *       x components, y components and z components
     *  
     * @param fc                Field component to output
     * @param d_in              Input buffer
     * @return unsigned int     Total number of cells * 3
     */
    unsigned int scatter( S * const __restrict__ d_in ) {

        dim3 block( 64 );
        dim3 grid( ntiles.x, ntiles.y );

        size_t shm_size = tile_vol * sizeof(V);
        block::set_shmem_size( vec3_grid_kernel::scatter<S,V>, shm_size );
        vec3_grid_kernel::scatter<S,V> <<< grid, block, shm_size >>> (
            d_buffer, offset,
            ntiles, nx, ext_nx,
            d_in );

        return dims.x * dims.y * 3;
    }

    /**
     * @brief Scatter all field component values from contiguous grid
     *        into tiles and scale values
     * 
     * @param d_in              Input buffer    
     * @param scale             Scale value
     * @return unsigned int     Total number of cells * 3
     */
    unsigned int scatter( S * const __restrict__ d_in, S const scale ) {

        dim3 block( 64 );
        dim3 grid( ntiles.x, ntiles.y );

        size_t shm_size = tile_vol * sizeof(V);
        block::set_shmem_size( vec3_grid_kernel::scatter_scale<S,V>, shm_size );
        vec3_grid_kernel::scatter_scale<S,V> <<< grid, block, shm_size >>> (
            d_buffer, offset,
            ntiles, nx, ext_nx,
            d_in, scale );

        return dims.x * dims.y * 3;
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
        info.count[0] = dims.x;
        info.count[1] = dims.y;

        const std::size_t bsize = dims.x * dims.y;

        S * d_data = device::malloc<S>( bsize );
        S * h_data = host::malloc<S>( bsize );

        gather( fc, d_data );

        device::memcpy_tohost( h_data, d_data, bsize );

        zdf::save_grid( h_data, info, iter, path );

        host::free( h_data );
        device::free( d_data );
    }

    /**
     * @brief Save specific field component to disk (no metadata)
     * 
     * @note The scalar field type <S> must be supported by ZDF file format
     * 
     * @param fc        Field component to save
     * @param filename  Outout file name (includes path) 
     */
    void save( const enum fcomp::cart fc, std::string filename ) {

        const std::size_t bsize = dims.x * dims.y;

        S * d_data = device::malloc<S>( bsize );
        S * h_data = host::malloc<S>( bsize );

        gather( fc, d_data );

        device::memcpy_tohost( h_data, d_data, bsize );

        uint64_t grid_dims[] = {dims.x, dims.y};

        std::string fcomp_name[] = {"x","y","z"};
        std::string name_fc = name + "-" + fcomp_name[fc];
        zdf::save_grid( h_data, 2, grid_dims, name_fc, filename );

        host::free( h_data );
        device::free( d_data );
    }

};

#endif