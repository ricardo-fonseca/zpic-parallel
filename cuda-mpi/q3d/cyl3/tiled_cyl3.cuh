#pragma once

#include "cyl3.cuh"
#include "grid/tiled.cuh"

/**
 * @brief MPI datatypes for cyl3 values (used for guard cell exchanges)
 *
 * @note Contiguous types are valid because cyl3<T> has no padding
 *       (sizeof(cyl3<T>) == 3 * sizeof(T), checked in cyl3.cuh)
 */
namespace mpi {

template<> inline MPI_Datatype data_type< cyl3<float>  >() { return data_type<float3>(); }
template<> inline MPI_Datatype data_type< cyl3<double> >() { return data_type<double3>(); }

template<> inline MPI_Datatype data_type< cyl3<util::complex64> >() {
    static const MPI_Datatype t = detail::contiguous_type( 3, data_type<util::complex64>() );
    return t;
}

template<> inline MPI_Datatype data_type< cyl3<util::complex128> >() {
    static const MPI_Datatype t = detail::contiguous_type( 3, data_type<util::complex128>() );
    return t;
}

} // namespace mpi

namespace grid {

template< typename S >
using tiled_cyl3_view = grid::tiled_view< cyl3<S> >;

namespace kernel {
namespace tiled_cyl3 {

/**
 * @brief Gather a single field component, x stride == 1
 *
 * @tparam fc       Field component
 * @tparam S        Output datatype
 * @tparam V        Tile datatype (cyl3<T>)
 * @param d_out         Output buffer
 * @param out_stride_y  Output buffer y stride
 * @param tiles         Tiled grid view
 */
template< fcomp::cyl fc, typename S, typename V >
__global__
void gather_fcomp(
    S * const __restrict__ d_out, unsigned int const out_stride_y,
    const tiled_view<V> tiles )
{
    unsigned const tx = blockIdx.x;
    unsigned const ty = blockIdx.y;
    const V * const __restrict__ tdata = tiles.tile_data( tx, ty );

    const auto tile_dims = tiles.tile_dims;
    const auto gix0 = tx * tile_dims.x;
    const auto giy0 = ty * tile_dims.y;
    const auto tile_stride_y = tiles.tile_ystride();

    for( unsigned i = gpu::block::thread_rank(); i < tile_dims.x * tile_dims.y; i += gpu::block::num_threads() ) {
        const auto ix = i % tile_dims.x;
        const auto iy = i / tile_dims.x;

        const auto out_idx = (giy0 + iy) * out_stride_y + (gix0 + ix);
        const auto & v = tdata[ iy * tile_stride_y + ix ];

        if constexpr ( fc == fcomp::cyl::z  ) d_out[ out_idx ] = v.z;
        if constexpr ( fc == fcomp::cyl::r  ) d_out[ out_idx ] = v.r;
        if constexpr ( fc == fcomp::cyl::th ) d_out[ out_idx ] = v.th;
    }
}

/**
 * @brief Gather a single field component, arbitrary stride
 *
 * @tparam fc       Field component
 * @tparam S        Output datatype
 * @tparam V        Tile datatype (cyl3<T>)
 * @param d_out         Output buffer
 * @param out_stride    Output buffer stride
 * @param tiles         Tiled grid view
 */
template< fcomp::cyl fc, typename S, typename V >
__global__
void gather_fcomp(
    S * const __restrict__ d_out, uint2 const out_stride,
    const tiled_view<V> tiles )
{
    unsigned const tx = blockIdx.x;
    unsigned const ty = blockIdx.y;
    const V * const __restrict__ tdata = tiles.tile_data( tx, ty );

    const auto tile_dims = tiles.tile_dims;
    const auto gix0 = tx * tile_dims.x;
    const auto giy0 = ty * tile_dims.y;
    const auto tile_stride_y = tiles.tile_ystride();

    for( unsigned i = gpu::block::thread_rank(); i < tile_dims.x * tile_dims.y; i += gpu::block::num_threads() ) {
        const auto ix = i % tile_dims.x;
        const auto iy = i / tile_dims.x;

        const auto out_idx = (giy0 + iy) * out_stride.y + (gix0 + ix) * out_stride.x;
        const auto & v = tdata[ iy * tile_stride_y + ix ];

        if constexpr ( fc == fcomp::cyl::z  ) d_out[ out_idx ] = v.z;
        if constexpr ( fc == fcomp::cyl::r  ) d_out[ out_idx ] = v.r;
        if constexpr ( fc == fcomp::cyl::th ) d_out[ out_idx ] = v.th;
    }
}

/**
 * @brief Gather all field components into separate buffers, x stride == 1
 *
 */
template< typename S, typename V >
__global__
void gather(
    S * const __restrict__ out_z,
    S * const __restrict__ out_r,
    S * const __restrict__ out_th,
    unsigned int const out_stride_y,
    const tiled_view<V> tiles )
{
    unsigned const tx = blockIdx.x;
    unsigned const ty = blockIdx.y;
    const V * const __restrict__ tdata = tiles.tile_data( tx, ty );

    const auto tile_dims = tiles.tile_dims;
    const auto gix0 = tx * tile_dims.x;
    const auto giy0 = ty * tile_dims.y;
    const auto tile_stride_y = tiles.tile_ystride();

    for( unsigned i = gpu::block::thread_rank(); i < tile_dims.x * tile_dims.y; i += gpu::block::num_threads() ) {
        const auto ix = i % tile_dims.x;
        const auto iy = i / tile_dims.x;

        const auto out_idx = (giy0 + iy) * out_stride_y + (gix0 + ix);
        const auto v = tdata[ iy * tile_stride_y + ix ];

        out_z [ out_idx ] = v.z;
        out_r [ out_idx ] = v.r;
        out_th[ out_idx ] = v.th;
    }
}

/**
 * @brief Gather all field components into separate buffers, arbitrary stride
 *
 */
template< typename S, typename V >
__global__
void gather(
    S * const __restrict__ out_z,
    S * const __restrict__ out_r,
    S * const __restrict__ out_th,
    uint2 const out_stride,
    const tiled_view<V> tiles )
{
    unsigned const tx = blockIdx.x;
    unsigned const ty = blockIdx.y;
    const V * const __restrict__ tdata = tiles.tile_data( tx, ty );

    const auto tile_dims = tiles.tile_dims;
    const auto gix0 = tx * tile_dims.x;
    const auto giy0 = ty * tile_dims.y;
    const auto tile_stride_y = tiles.tile_ystride();

    for( unsigned i = gpu::block::thread_rank(); i < tile_dims.x * tile_dims.y; i += gpu::block::num_threads() ) {
        const auto ix = i % tile_dims.x;
        const auto iy = i / tile_dims.x;

        const auto out_idx = (giy0 + iy) * out_stride.y + (gix0 + ix) * out_stride.x;
        const auto v = tdata[ iy * tile_stride_y + ix ];

        out_z [ out_idx ] = v.z;
        out_r [ out_idx ] = v.r;
        out_th[ out_idx ] = v.th;
    }
}

/**
 * @brief Scatter separate component buffers into the tiles, with scaling,
 *        x stride == 1
 *
 * @note Uses V( ... ) rather than V{ ... } so that the same implicit
 *       conversions as the base class scatter are allowed (e.g. double
 *       input into a float grid).
 */
template< typename S, typename T, typename V >
__global__
void scatter(
    const S * const __restrict__ in_z,
    const S * const __restrict__ in_r,
    const S * const __restrict__ in_th,
    const T scale,
    const unsigned int in_stride_y,
    tiled_view<V> tiles )
{
    unsigned const tx = blockIdx.x;
    unsigned const ty = blockIdx.y;
    V * const __restrict__ tdata = tiles.tile_data( tx, ty );

    const auto tile_dims = tiles.tile_dims;
    const auto gix0 = tx * tile_dims.x;
    const auto giy0 = ty * tile_dims.y;
    const auto tile_stride_y = tiles.tile_ystride();

    for( unsigned i = gpu::block::thread_rank(); i < tile_dims.x * tile_dims.y; i += gpu::block::num_threads() ) {
        const auto ix = i % tile_dims.x;
        const auto iy = i / tile_dims.x;

        const auto in_idx = (giy0 + iy) * in_stride_y + (gix0 + ix);

        tdata[ iy * tile_stride_y + ix ] = V(
            in_z [ in_idx ] * scale,
            in_r [ in_idx ] * scale,
            in_th[ in_idx ] * scale );
    }
}

/**
 * @brief Scatter separate component buffers into the tiles, with scaling,
 *        arbitrary stride
 *
 */
template< typename S, typename T, typename V >
__global__
void scatter(
    const S * const __restrict__ in_z,
    const S * const __restrict__ in_r,
    const S * const __restrict__ in_th,
    const T scale,
    const uint2 in_stride,
    tiled_view<V> tiles )
{
    unsigned const tx = blockIdx.x;
    unsigned const ty = blockIdx.y;
    V * const __restrict__ tdata = tiles.tile_data( tx, ty );

    const auto tile_dims = tiles.tile_dims;
    const auto gix0 = tx * tile_dims.x;
    const auto giy0 = ty * tile_dims.y;
    const auto tile_stride_y = tiles.tile_ystride();

    for( unsigned i = gpu::block::thread_rank(); i < tile_dims.x * tile_dims.y; i += gpu::block::num_threads() ) {
        const auto ix = i % tile_dims.x;
        const auto iy = i / tile_dims.x;

        const auto in_idx = (giy0 + iy) * in_stride.y + (gix0 + ix) * in_stride.x;

        tdata[ iy * tile_stride_y + ix ] = V(
            in_z [ in_idx ] * scale,
            in_r [ in_idx ] * scale,
            in_th[ in_idx ] * scale );
    }
}

} // tiled_cyl3 namespace
} // kernel namespace

/**
 * @brief Tiled grid of cylindrical vector values, cyl3<S>
 *
 * @note Grid x is the z (axial) direction and grid y is the r (radial)
 *       direction (see coord::cyl). The parallel partition must not be
 *       periodic along r.
 *
 * @note copy_to_gc() (also called by scatter()) does not fill the lower r
 *       guard cells on the node holding the axis, since there is no lower
 *       neighbour. Any axis boundary condition must be applied separately.
 *
 * @tparam S    Component datatype: float, double, util::complex64 or
 *              util::complex128
 */
template < typename S >
class tiled_cyl3 : public grid::tiled< cyl3<S> >
{
    private:

    using V = cyl3<S>;

    // Protected in grid::tiled, used internally (keep private)
    using grid::tiled< V > :: part;
    using grid::tiled< V > :: local_ntiles;
    using grid::tiled< V > :: local_tile_start;
    using grid::tiled< V > :: local_dims;

    /**
     * @brief Check that the parallel partition is valid for a cylindrical grid
     *
     */
    void validate_cylindrical() const {
        if ( part.periodic.y ) {
            mpi::fatal( "tiled_cyl3: the parallel partition must not be periodic along r (y)" );
        }
    }

    /**
     * @brief Launch the single component gather kernel
     *
     * @tparam fc       Field component
     * @tparam S2       Output datatype
     * @param out       Output buffer
     * @param stride    Output buffer stride (already defaulted)
     */
    template< fcomp::cyl fc, typename S2 >
    void gather_fcomp( S2 * const __restrict__ out, const uint2 stride ) const {
        dim3 block( 64 );
        dim3 grid( local_ntiles.x, local_ntiles.y );

        if ( stride.x == 1 ) {
            // Optimized version for x stride == 1
            kernel::tiled_cyl3::gather_fcomp<fc> <<< grid, block >>>
                ( out, stride.y, cview() );
        } else {
            // Arbitrary x stride
            kernel::tiled_cyl3::gather_fcomp<fc> <<< grid, block >>>
                ( out, stride, cview() );
        }
    }

    public:

    // Public in grid::tiled, listed for unqualified use and to un-hide
    // the base class gather() / scatter() overloads
    using grid::tiled< V > :: name;

    using grid::tiled< V > :: global_ntiles;
    using grid::tiled< V > :: tile_dims;
    using grid::tiled< V > :: tile_ext_dims;

    using grid::tiled< V > :: gc;
    using grid::tiled< V > :: tile_vol;

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
     * @brief Construct a new tiled_cyl3 object with guard cells
     *
     * @param global_ntiles     Global number of tiles (z,r)
     * @param tile_dims         Tile dimensions (z,r)
     * @param gc                Number of guard cells
     * @param part              Parallel partition, must not be periodic along r
     */
    tiled_cyl3( uint2 const global_ntiles, uint2 const tile_dims,
        bounds_2d<unsigned int> const gc, const mpi::cart2d & part ) :
        grid::tiled< V >( global_ntiles, tile_dims, gc, part )
    {
        validate_cylindrical();
    }

    /**
     * @brief Construct a new tiled_cyl3 object without guard cells
     *
     * @param global_ntiles     Global number of tiles (z,r)
     * @param tile_dims         Tile dimensions (z,r)
     * @param part              Parallel partition, must not be periodic along r
     */
    tiled_cyl3( uint2 const global_ntiles, uint2 const tile_dims,
        const mpi::cart2d & part ) :
        grid::tiled< V >( global_ntiles, tile_dims, part )
    {
        validate_cylindrical();
    }

    /**
     * @brief Gather specific field component values from all tiles
     *
     * @note
     * The default behavior is to output data into a contiguous array of dimensions
     * local_dims.y * local_dims.x. The stride parameter can be used to specify different
     * memory layouts.
     *
     * @tparam S2           Output datatype, must support S2 = S assignment
     * @param fc            Field component to output
     * @param out           Scalar output buffer
     * @param stride        (optional) Output buffer stride, defaults to a
     *                      contiguous memory layout
     * @return unsigned int Total number of elements copied
     */
    template< typename S2 >
    unsigned int gather( const enum fcomp::cyl fc, S2 * const __restrict__ out,
        uint2 stride = {0,0} ) const {

        // Default to contiguous memory layout
        if ( stride.x == 0 ) {
            stride = make_uint2( 1, local_dims.x );
        }

        // Check that y stride is valid
        assert(( stride.y > 0 ));

        switch( fc ) {
        case( fcomp::cyl::z ):
            gather_fcomp< fcomp::cyl::z  >( out, stride ); break;
        case( fcomp::cyl::r ):
            gather_fcomp< fcomp::cyl::r  >( out, stride ); break;
        case( fcomp::cyl::th ):
            gather_fcomp< fcomp::cyl::th >( out, stride ); break;
        default:
            mpi::fatal( "invalid fc value" );
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
     * @param out_z             z component buffer
     * @param out_r             r component buffer
     * @param out_th            θ component buffer
     * @param stride            Output buffers stride, defaults to a contiguous
     *                          memory layout
     * @return unsigned int     Number of cells written
     */
    template< typename S2 >
    unsigned int gather(
        S2 * const __restrict__ out_z,
        S2 * const __restrict__ out_r,
        S2 * const __restrict__ out_th,
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
            // Optimized version for x stride == 1
            kernel::tiled_cyl3::gather <<< grid, block >>>
                ( out_z, out_r, out_th, stride.y, cview() );
        } else {
            // Arbitrary x stride
            kernel::tiled_cyl3::gather <<< grid, block >>>
                ( out_z, out_r, out_th, stride, cview() );
        }

        return local_dims.x * local_dims.y;
    }

    /**
     * @brief Scatter data into the tile grid with scaling and update guard
     *        cell values
     *
     * @tparam S2               Input buffer datatype
     * @tparam S3               Scale factor datatype, S = S2 * S3 must be
     *                          supported
     * @param in_z              z component buffer
     * @param in_r              r component buffer
     * @param in_th             θ component buffer
     * @param scale             Scale factor
     * @param stride            Input buffer stride, defaults to a
     *                          contiguous memory layout
     * @return unsigned int     Total number of cells
     */
    template< typename S2, typename S3 >
    unsigned int scatter(
        const S2 * const __restrict__ in_z,
        const S2 * const __restrict__ in_r,
        const S2 * const __restrict__ in_th,
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
            // Optimized version for x stride == 1
            kernel::tiled_cyl3::scatter <<< grid, block >>>
                ( in_z, in_r, in_th, scale, stride.y, view() );
        } else {
            // Arbitrary x stride
            kernel::tiled_cyl3::scatter <<< grid, block >>>
                ( in_z, in_r, in_th, scale, stride, view() );
        }

        // Update guard cell values
        copy_to_gc();

        return local_dims.x * local_dims.y;
    }

    /**
     * @brief Scatter data into the tile grid and update guard cell values
     *
     * @note Implemented as a scatter with scale 1, which is exact for both
     *       real and complex values.
     *
     * @tparam S2               Input buffer datatype
     * @param in_z              z component buffer
     * @param in_r              r component buffer
     * @param in_th             θ component buffer
     * @param stride            Input buffer stride, defaults to a
     *                          contiguous memory layout
     * @return unsigned int     Total number of cells
     */
    template< typename S2 >
    unsigned int scatter(
        const S2 * const __restrict__ in_z,
        const S2 * const __restrict__ in_r,
        const S2 * const __restrict__ in_th,
        uint2 stride = {0,0} ) {

        return scatter( in_z, in_r, in_th, 1, stride );
    }

    /**
     * @brief Save specific field component to disk
     *
     * @tparam S2       Datatype to be used for output, defaults to the grid
     *                  component datatype. Must be supported by the ZDF library.
     * @param fc        Field component to save
     * @param metadata  Grid metadata (label, units, axis, etc.). Information is used to set file name
     * @param iter      Iteration metadata
     * @param path      Path where to save the file
     */
    template< typename S2 = S >
    void save( const enum fcomp::cyl fc, zdf::grid_info &metadata, const zdf::iteration &iter, const std::string & path ) {

        // Fill in grid dimensions
        metadata.ndims = 2;
        metadata.count[0] = global_ntiles.x * tile_dims.x;
        metadata.count[1] = global_ntiles.y * tile_dims.y;

        // Allocate buffers on host and device to gather data
        const std::size_t bsize = static_cast<std::size_t>(local_dims.x) * local_dims.y;
        S2 * h_data = gpu::host::malloc<S2>( bsize );
        S2 * d_data = gpu::device::malloc<S2>( bsize );

        // Gather data on contiguous grid
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

        // Free temporary buffer
        gpu::host::free( h_data );
    }

    /**
     * @brief Save specific field component to disk
     *
     * @note The dataset is named <name>-<component>
     *
     * @tparam S2       Datatype to be used for output, defaults to the grid
     *                  component datatype. Must be supported by the ZDF library.
     * @param fc        Field component to save
     * @param filename  Output file name
     */
    template< typename S2 = S >
    void save( const enum fcomp::cyl fc, const std::string & filename ) {

        // Allocate buffers on host and device to gather data
        const std::size_t bsize = static_cast<std::size_t>(local_dims.x) * local_dims.y;
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

        // Free temporary buffer
        gpu::host::free( h_data );
    }
};

} // namespace grid
