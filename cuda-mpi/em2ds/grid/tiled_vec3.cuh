#pragma once

#include "tiled.cuh"
#include "../core/vec_types.cuh"

namespace grid {

template< typename S >
using tiled_vec3_view = grid::tiled_view< vec3<S> >;

namespace kernel {
namespace tiled_vec3 {

template< fcomp::cart fc, typename S, typename V >
__global__
void gather_fcomp( 
    S * const __restrict__ d_out, unsigned int const out_stride_y,
    const tiled_view<V> tiles )
{
    unsigned const tx = blockIdx.x;
    unsigned const ty = blockIdx.y;
    const V * const __restrict__ tdata = tiles.tile_data( tx, ty );

    const auto tile_dims  =tiles.tile_dims;
    const auto gix0 = tx * tile_dims.x;
    const auto giy0 = ty * tile_dims.y;
    const auto tile_stride_y = tiles.tile_ystride();

    for( int i = gpu::block::thread_rank(); i < tile_dims.x * tile_dims.y; i+= gpu::block::num_threads() ) {
        const auto ix = i % tile_dims.x;
        const auto iy = i / tile_dims.x;

        const auto out_idx = (giy0 + iy) * out_stride_y + (gix0 + ix);

        if constexpr ( fc == fcomp::x ) d_out[ out_idx ] = tdata[ iy * tile_stride_y + ix ].x;
        if constexpr ( fc == fcomp::y ) d_out[ out_idx ] = tdata[ iy * tile_stride_y + ix ].y;
        if constexpr ( fc == fcomp::z ) d_out[ out_idx ] = tdata[ iy * tile_stride_y + ix ].z;
    }
}

template< fcomp::cart fc, typename S, typename V >
__global__
void gather_fcomp( 
    S * const __restrict__ d_out, uint2 const out_stride,
    const tiled_view<V> tiles )
{
    unsigned const tx = blockIdx.x;
    unsigned const ty = blockIdx.y;
    const V * const __restrict__ tdata = tiles.tile_data( tx, ty );

    const auto tile_dims  =tiles.tile_dims;
    const auto gix0 = tx * tile_dims.x;
    const auto giy0 = ty * tile_dims.y;
    const auto tile_stride_y = tiles.tile_ystride();

    for( int i = gpu::block::thread_rank(); i < tile_dims.x * tile_dims.y; i+= gpu::block::num_threads() ) {
        const auto ix = i % tile_dims.x;
        const auto iy = i / tile_dims.x;

        const auto out_idx = (giy0 + iy) * out_stride.y + (gix0 + ix) * out_stride.x;

        if constexpr ( fc == fcomp::x ) d_out[ out_idx ] = tdata[ iy * tile_stride_y + ix ].x;
        if constexpr ( fc == fcomp::y ) d_out[ out_idx ] = tdata[ iy * tile_stride_y + ix ].y;
        if constexpr ( fc == fcomp::z ) d_out[ out_idx ] = tdata[ iy * tile_stride_y + ix ].z;
    }
}

template< typename S, typename V >
__global__
void gather( 
    S * const __restrict__ out_x, 
    S * const __restrict__ out_y, 
    S * const __restrict__ out_z,
    unsigned int const out_stride_y,
    const tiled_view<V> tiles) {

    unsigned const tx = blockIdx.x;
    unsigned const ty = blockIdx.y;
    const V * const __restrict__ tdata = tiles.tile_data( tx, ty );

    const auto tile_dims  =tiles.tile_dims;
    const auto gix0 = tx * tile_dims.x;
    const auto giy0 = ty * tile_dims.y;
    const auto tile_stride_y = tiles.tile_ystride();

    for( int i = gpu::block::thread_rank(); i < tile_dims.x * tile_dims.y; i+= gpu::block::num_threads() ) {
        const auto ix = i % tile_dims.x;
        const auto iy = i / tile_dims.x;

        const auto out_idx = (giy0 + iy) * out_stride_y + (gix0 + ix);

        out_x[ out_idx ] = tdata[ iy * tile_stride_y + ix ].x;
        out_y[ out_idx ] = tdata[ iy * tile_stride_y + ix ].y;
        out_z[ out_idx ] = tdata[ iy * tile_stride_y + ix ].z;
    }
}


template< typename S, typename V >
__global__
void gather( 
    S * const __restrict__ out_x, 
    S * const __restrict__ out_y, 
    S * const __restrict__ out_z,
    uint2 const out_stride,
    const tiled_view<V> tiles) {

    unsigned const tx = blockIdx.x;
    unsigned const ty = blockIdx.y;
    const V * const __restrict__ tdata = tiles.tile_data( tx, ty );

    const auto tile_dims  =tiles.tile_dims;
    const auto gix0 = tx * tile_dims.x;
    const auto giy0 = ty * tile_dims.y;
    const auto tile_stride_y = tiles.tile_ystride();

    for( int i = gpu::block::thread_rank(); i < tile_dims.x * tile_dims.y; i+= gpu::block::num_threads() ) {
        const auto ix = i % tile_dims.x;
        const auto iy = i / tile_dims.x;

        const auto out_idx = (giy0 + iy) * out_stride.y + (gix0 + ix) * out_stride.x;

        out_x[ out_idx ] = tdata[ iy * tile_stride_y + ix ].x;
        out_y[ out_idx ] = tdata[ iy * tile_stride_y + ix ].y;
        out_z[ out_idx ] = tdata[ iy * tile_stride_y + ix ].z;
    }
}

template< typename S, typename T, typename V >
__global__
void scatter( 
    const S * const __restrict__ in_x, 
    const S * const __restrict__ in_y, 
    const S * const __restrict__ in_z,
    const T scale,
    const unsigned int in_stride_y,
    tiled_view<V> tiles) {

    unsigned const tx = blockIdx.x;
    unsigned const ty = blockIdx.y;
    V * const __restrict__ tdata = tiles.tile_data( tx, ty );

    const auto tile_dims  =tiles.tile_dims;
    const auto gix0 = tx * tile_dims.x;
    const auto giy0 = ty * tile_dims.y;
    const auto tile_stride_y = tiles.tile_ystride();

    for( int i = gpu::block::thread_rank(); i < tile_dims.x * tile_dims.y; i+= gpu::block::num_threads() ) {
        const auto ix = i % tile_dims.x;
        const auto iy = i / tile_dims.x;

        V val;
        val.x = in_x[ (giy0 + iy) * in_stride_y + (gix0 + ix) ] * scale;
        val.y = in_y[ (giy0 + iy) * in_stride_y + (gix0 + ix) ] * scale;
        val.z = in_z[ (giy0 + iy) * in_stride_y + (gix0 + ix) ] * scale;
        tdata[ iy * tile_stride_y + ix ] = val;
    }
}

template< typename S, typename T, typename V >
__global__
void scatter( 
    const S * const __restrict__ in_x, 
    const S * const __restrict__ in_y, 
    const S * const __restrict__ in_z,
    const T scale,
    const uint2 in_stride,
    tiled_view<V> tiles) {

    unsigned const tx = blockIdx.x;
    unsigned const ty = blockIdx.y;
    V * const __restrict__ tdata = tiles.tile_data( tx, ty );

    const auto tile_dims  =tiles.tile_dims;
    const auto gix0 = tx * tile_dims.x;
    const auto giy0 = ty * tile_dims.y;
    const auto tile_stride_y = tiles.tile_ystride();

    for( int i = gpu::block::thread_rank(); i < tile_dims.x * tile_dims.y; i+= gpu::block::num_threads() ) {
        const auto ix = i % tile_dims.x;
        const auto iy = i / tile_dims.x;

        V val;
        val.x = in_x[ (giy0 + iy) * in_stride.y + (gix0 + ix) * in_stride.x ] * scale;
        val.y = in_y[ (giy0 + iy) * in_stride.y + (gix0 + ix) * in_stride.x ] * scale;
        val.z = in_z[ (giy0 + iy) * in_stride.y + (gix0 + ix) * in_stride.x ] * scale;
        tdata[ iy * tile_stride_y + ix ] = val;
    }
}

} // tiled_vec3 namespace
} // kernel namespace

template < typename S > 
class tiled_vec3 : public grid::tiled< vec3<S> >
{
    private:

    using V = vec3<S>;

    using grid::tiled< V > :: local_ntiles;
    using grid::tiled< V > :: local_tile_start;
    using grid::tiled< V > :: local_periodic;
    using grid::tiled< V > :: d_buffer;

    using grid::tiled< V > :: initialize;

    public:

    using grid::tiled< V > :: part;
    using grid::tiled< V > :: name;

    using grid::tiled< V > :: global_ntiles;
    using grid::tiled< V > :: tile_dims;
    using grid::tiled< V > :: tile_ext_dims;

    using grid::tiled< V > :: gc;
    using grid::tiled< V > :: local_dims;

    using grid::tiled< V > :: tile_vol;

    using grid::tiled< V > :: tiled;

    using grid::tiled< V > :: set;

    using grid::tiled< V > :: gather;
    using grid::tiled< V > :: scatter;
    using grid::tiled< V > :: copy_to_gc_x;
    using grid::tiled< V > :: copy_to_gc_y;
    using grid::tiled< V > :: copy_to_gc;

    using grid::tiled< V > :: buffer;
    using grid::tiled< V > :: tile_data;
    using grid::tiled< V > :: tile_buffer;

    using grid::tiled< V > :: view;
    using grid::tiled< V > :: cview;
    
    /**
     * @brief Gather specific field component values from all tiles
     * 
     * @note
     * The default behavior is to output data into a contiguous array of dimensions 
     * local_dims.y * local_dims.x. The stride parameter can be used to specify different
     * memory layouts.
     *
     * @param fc            Field component to output
     * @param out           Scalar output buffer
     * @return uint         Total number of elements copied
     */
    template< typename S2 >
    unsigned int gather( const enum fcomp::cart fc, S2 * const __restrict__ out, 
        uint2 stride = {0,0} ) const {
        
        // Default to contiguous memory layout
        if ( stride.x == 0 ) {
            stride = make_uint2( 1, local_dims.x );
        }

        // Check that y stride is valid
        assert(( stride.y > 0 ));

        dim3 block( 64 );
        dim3 grid( local_ntiles.x, local_ntiles.y );

        if ( stride.x == 1 ) {
            // Optimized versions for x stride == 1
            switch( fc ) {
                case( fcomp::x ):
                kernel::tiled_vec3::gather_fcomp<fcomp::x> <<< grid, block >>>
                    (out, stride.y, cview());
                break;
                case( fcomp::y ):
                kernel::tiled_vec3::gather_fcomp<fcomp::y> <<< grid, block >>>
                    (out, stride.y, cview());
                break;
                case( fcomp::z ):
                kernel::tiled_vec3::gather_fcomp<fcomp::z> <<< grid, block >>>
                    (out, stride.y, cview());
                break;
                default:
                mpi::fatal( "invalid fc value");
            }

        } else {
            // Arbitrary x stride
            switch( fc ) {
                case( fcomp::x ):
                kernel::tiled_vec3::gather_fcomp<fcomp::x> <<< grid, block >>>
                    (out, stride, cview());
                break;
                case( fcomp::y ):
                kernel::tiled_vec3::gather_fcomp<fcomp::y> <<< grid, block >>>
                    (out, stride, cview());
                break;
                case( fcomp::z ):
                kernel::tiled_vec3::gather_fcomp<fcomp::z> <<< grid, block >>>
                    (out, stride, cview());
                break;
                default:
                mpi::fatal( "invalid fc value");
            }
        }

        return local_dims.x * local_dims.y;
    }

    /**
     * @brief Gather data from tiled grid
     *
     * @note By default, the output buffers are assumed to use a contiguous
     *       memory layout. Other layouts may be specified using the stride
     *       parameter
     * 
     * @tparam S2               Output grids datatype
     * @param out_x             x component buffer
     * @param out_y             y component buffer
     * @param out_z             z component buffer
     * @param stride            Output buffers stride, defaults to a contiguous
     *                          memory layout
     * @return unsigned int     Number of cells written
     */
    template< typename S2 >
    unsigned int gather( 
        S2 * const __restrict__ out_x, 
        S2 * const __restrict__ out_y, 
        S2 * const __restrict__ out_z,
        uint2 stride = {0,0} ) const {

        // Default to contiguous memory layout
        if ( stride.x == 0 ) {
            stride = make_uint2( 1, local_dims.x );
        }

        // Check that y stride is valid
        assert(( stride.y > 0 ));

        dim3 block( 64 );
        dim3 grid( local_ntiles.x, local_ntiles.y );

        if ( stride.x == 1 ) {
            kernel::tiled_vec3::gather <<< grid, block >>> 
                ( out_x, out_y, out_z, stride.y, cview() );
        } else {
            // Arbitrary x stride
            kernel::tiled_vec3::gather <<< grid, block >>> 
                ( out_x, out_y, out_z, stride, cview() );
        }

        return local_dims.x * local_dims.y;
    }

    /**
     * @brief Scatter data into the tile grid and update guard cell values
     * 
     * @tparam S2               Input buffer datatype
     * @tparam S3               Scale factor datatype
     * @param in_x              x component buffer
     * @param in_y              y component buffer
     * @param in_z              z component buffer
     * @param scale             Scale factor
     * @param stride            Input buffer stride, defaults to a
     *                          contiguous memory layout
     * @return unsigned int     Total number of cells
     */
    template< typename S2, typename S3 >
    unsigned int scatter( 
        const S2 * const __restrict__ in_x, 
        const S2 * const __restrict__ in_y, 
        const S2 * const __restrict__ in_z,
        const S3 scale, 
        uint2 stride = {0,0} ) {

        // Default to contiguous memory layout
        if ( stride.x == 0 ) {
            stride = make_uint2( 1, local_dims.x );
        }

        // Check that y stride is valid
        assert(( stride.y > 0 ));

        dim3 block( 64 );
        dim3 grid( local_ntiles.x, local_ntiles.y );

        if ( stride.x == 1 ) {
            kernel::tiled_vec3::scatter <<< grid, block >>> 
                ( in_x, in_y, in_z, scale, stride.y, view() );
        } else {
            // Arbitrary x stride
            kernel::tiled_vec3::scatter <<< grid, block >>> 
                ( in_x, in_y, in_z, scale, stride, view() );
        }

        // Update guard cell values
        copy_to_gc();

        return local_dims.x * local_dims.y;
    }


    /**
     * @brief Save specific field component to disk
     * 
     * The field type <T> must be supported by ZDF file format
     * 
     * @param fc    Field component to save
     * @param info  Grid metadata (label, units, axis, etc.). Information is used to set file name
     * @param iter  Iteration metadata
     * @param path  Path where to save the file
     */
    template< typename S2 = S >
    void save( const enum fcomp::cart fc, zdf::grid_info &metadata, const zdf::iteration &iter, const std::string & path ) {

        // Fill in grid dimensions
        metadata.ndims = 2;
        metadata.count[0] = global_ntiles.x * tile_dims.x;
        metadata.count[1] = global_ntiles.y * tile_dims.y;

        // Allocate buffer on host to gather data
        const std::size_t bsize = static_cast<std::size_t>(local_dims.x) * local_dims.y;
        S2 * d_data = gpu::device::malloc<S2>( bsize );
        S2 * h_data = gpu::host::malloc<S2>( bsize );

        gather( fc, d_data );

        // Copy to host and free device memory
        gpu::device::memcpy_tohost( h_data, d_data, bsize );
        gpu::device::free( d_data );


        // Information on local chunk of grid data
        zdf::chunk chunk;
        chunk.count[0] = local_dims.x;
        chunk.count[1] = local_dims.y;
        chunk.start[0] = local_tile_start.x * tile_dims.x;
        chunk.start[1] = local_tile_start.y * tile_dims.y;
        chunk.stride[0] = chunk.stride[1] = 1;
        chunk.data = (void *) h_data;

        // Save data
        zdf::save_grid<S2>( chunk, metadata, iter, path, part.get_comm() );

        gpu::host::free( h_data );
    }

    template< typename S2 = S >
    void save( const enum fcomp::cart fc, const std::string & filename ) {
        
        // Allocate buffers on host and device to gather data
        const std::size_t bsize = local_dims.x * local_dims.y;
        S2 * h_data = gpu::host::malloc<S2>( bsize );
        S2 * d_data = gpu::device::malloc<S2>( bsize );

        // Gather data on contiguous grid
        gather( fc, d_data );

        // Copy to host and free device memory
        gpu::device::memcpy_tohost( h_data, d_data, bsize );
        gpu::device::free( d_data );

        uint64_t global[2] = { global_ntiles.x * tile_dims.x, global_ntiles.y * tile_dims.y };
        uint64_t start[2]  = { local_tile_start.x * tile_dims.x, local_tile_start.y * tile_dims.y };
        uint64_t local[2]  = { local_dims.x, local_dims.y };

        // Save data
        zdf::save_grid( h_data, 2, global, start, local, name + "-" + fcomp::name(fc), filename, part.get_comm() );

        // Free remaining temporary buffer 
        gpu::host::free( h_data );
    }
};

}
