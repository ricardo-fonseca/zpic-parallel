#pragma once

#include <iostream>
#include <cassert>
#include <algorithm>
#include <string>
#include <cstddef>

#include "../core/gpu.cuh"

#include "../core/vec_types.cuh"
#include "../core/bounds.hpp"
#include "../parallel/parallel.hpp"
#include "../zdf/zdf-parallel.hpp"

#include "../util/memory.cuh"
#include "../util/math.hpp"

namespace grid {

template <class T>
struct tiled_view {

    /// @brief Local number of tiles
    const uint2 local_ntiles;

    /// @brief Data buffer
    T * d_buffer;

    /// @brief Tile grid dimensions
    const uint2 tile_dims;
    
    /// @brief Tile grid dimensions including guard cells
    const uint2 tile_ext_dims;

    /// @brief Offset, in cells, from the start of a tile's data buffer to its local (0,0) point
    const unsigned int inner_offset;

    /// @brief Tile guard cells
    const bounds_2d<unsigned int> gc;

    /// @brief Tile volume (may be larger than tile_ext_dim.x * tile_ext_dim.y for alignment)
    const std::size_t tile_vol;

    /**
     * @brief Buffer size
     * 
     * @return total size of data buffers (in elements)
     */
    __host__ __device__
    inline std::size_t buffer_size() const noexcept {
        return tile_vol * static_cast<std::size_t>(local_ntiles.x) * local_ntiles.y;
    };

    /**
     * @brief Get a pointer to the start of a specific tile's data buffer
     * 
     * @note This points at the first guard cell of the tile (if any), not at
     *       the tile's local (0,0) point. Use tile_data() for a pointer to
     *       the local (0,0) point instead.
     * 
     * @param tid   Tile index (flat)
     * @return T* 
     */
    __host__ __device__
    inline T * tile_buffer( const unsigned int tid ) const noexcept {
        return & d_buffer[ tid * tile_vol ];
    }

    /**
     * @brief Get a pointer to the start of a specific tile's data buffer
     * 
     * @note This points at the first guard cell of the tile (if any), not at
     *       the tile's local (0,0) point. Use tile_data() for a pointer to
     *       the local (0,0) point instead.
     * 
     * @param tx    x tile index
     * @param ty    y tile index
     * @return T* 
     */
    __host__ __device__
    inline T * tile_buffer( const unsigned int tx, const unsigned int ty ) const noexcept {
        return & d_buffer[ (ty * local_ntiles.x + tx) * tile_vol ];
    }

    /**
     * @brief Get a pointer to the start of a specific tile's data buffer
     * 
     * @note This points at the first guard cell of the tile (if any), not at
     *       the tile's local (0,0) point. Use tile_data() for a pointer to
     *       the local (0,0) point instead.
     * 
     * @param tid   Tile index (x,y)
     * @return T* 
     */
    __host__ __device__
    inline  T * tile_buffer( const uint2 tid ) const noexcept {
        return & d_buffer[ (tid.y * local_ntiles.x + tid.x) * tile_vol ];
    }

    /**
     * @brief Get a pointer to a specific tile's local (0,0) point
     * 
     * @note This skips past the tile's guard cells (if any). Use
     *       tile_buffer() for a pointer to the start of the tile's data
     *       buffer instead.
     * 
     * @param tid   Tile index (flat)
     * @return T* 
     */
    __host__ __device__
    inline T * tile_data( const unsigned int tid ) const noexcept {
        return & d_buffer[ tid * tile_vol + inner_offset ];
    }

    /**
     * @brief Get a pointer to a specific tile's local (0,0) point
     * 
     * @note This skips past the tile's guard cells (if any). Use
     *       tile_buffer() for a pointer to the start of the tile's data
     *       buffer instead.
     * 
     * @param tx    x tile index
     * @param ty    y tile index
     * @return T* 
     */
    __host__ __device__
    inline T * tile_data( const unsigned int tx, const unsigned int ty ) const noexcept {
        return & d_buffer[ (ty * local_ntiles.x + tx) * tile_vol + inner_offset ];
    }

    /**
     * @brief Get a pointer to a specific tile's local (0,0) point
     * 
     * @note This skips past the tile's guard cells (if any). Use
     *       tile_buffer() for a pointer to the start of the tile's data
     *       buffer instead.
     * 
     * @param tid   Tile index (x,y)
     * @return T* 
     */
    __host__ __device__
    inline T * tile_data( const uint2 tid ) const noexcept {
        return & d_buffer[ (tid.y * local_ntiles.x + tid.x) * tile_vol + inner_offset ];
    }

    /**
     * @brief Y stride in each tile (same as tile_ext_dims.x)
     *  
     * @return unsigned int 
     */
    __host__ __device__
    inline unsigned int tile_ystride() const noexcept {
        return tile_ext_dims.x;
    }
};

/**
 * @brief CUDA kernels for tiled::grid
 * 
 */
namespace kernel {

/**
 * @brief CUDA kernel for add operation
 * 
 * @tparam T 
 * @param a 
 * @param b 
 * @param size 
 * @return __global__ 
 */
template< class T >
__global__
void add(T * __restrict__ a, T const * __restrict__ b, size_t const size ) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < size ) {
        a[idx] += b[idx];
    }
}

template< class T2, class T1 >
__global__
void gather( 
    T2 * const __restrict__ d_out, const unsigned int out_stride_y,
    const tiled_view<T1> tiles )
{
    unsigned const tx = blockIdx.x;
    unsigned const ty = blockIdx.y;

    T1 * const __restrict__ tdata = tiles.tile_data( tx, ty );

    const auto tile_dims  =tiles.tile_dims;

    const auto gix0 = tx * tile_dims.x;
    const auto giy0 = ty * tile_dims.y;

    const auto tile_stride_y = tiles.tile_ystride();

    for( int i = gpu::block::thread_rank(); i < tile_dims.x * tile_dims.y; i+= gpu::block::num_threads() ) {
        const auto ix = i % tile_dims.x;
        const auto iy = i / tile_dims.x;

        d_out[ (giy0 + iy) * out_stride_y + (gix0 + ix) ] =
                                tdata[ iy * tile_stride_y + ix ];
    }
}

template< class T2, class T1 >
__global__
void gather( 
    T2 * const __restrict__ d_out, const uint2 out_stride,
    const tiled_view<T1> tiles )
{
    unsigned const tx = blockIdx.x;
    unsigned const ty = blockIdx.y;

    T1 const * const __restrict__ tdata = tiles.tile_data( tx, ty );

    const auto tile_dims  = tiles.tile_dims;

    const auto gix0 = tx * tile_dims.x;
    const auto giy0 = ty * tile_dims.y;

    const auto tile_stride_y = tiles.tile_ystride();

    for( int i = gpu::block::thread_rank(); i < tile_dims.x * tile_dims.y; i+= gpu::block::num_threads() ) {
        const auto ix = i % tile_dims.x;
        const auto iy = i / tile_dims.x;
        d_out[ (giy0 + iy) * out_stride.y + (gix0 + ix) * out_stride.x ] =
                                tdata[ iy * tile_stride_y + ix ];
    }
}

template< class T2, class T1 >
__global__
void scatter( 
    tiled_view<T1> tiles, 
    T2 const * const __restrict__ d_in, const unsigned int in_stride_y
){
    unsigned const tx = blockIdx.x;
    unsigned const ty = blockIdx.y;

    T1 * const __restrict__ tdata = tiles.tile_data( tx, ty );

    const auto tile_dims  = tiles.tile_dims;

    const auto gix0 = tx * tile_dims.x;
    const auto giy0 = ty * tile_dims.y;

    const auto tile_stride_y = tiles.tile_ystride();

    for( int i = gpu::block::thread_rank(); i < tile_dims.x * tile_dims.y; i+= gpu::block::num_threads() ) {
        const auto ix = i % tile_dims.x;
        const auto iy = i / tile_dims.x;
        tdata[ iy * tile_stride_y + ix ] = 
            d_in[ (giy0 + iy) * in_stride_y + (gix0 + ix) ];
    }
}

template< class T2, class T1 >
__global__
void scatter( 
    tiled_view<T1> tiles, 
    T2 const * const __restrict__ d_in, const uint2 in_stride
){
    unsigned const tx = blockIdx.x;
    unsigned const ty = blockIdx.y;

    T1 * const __restrict__ tdata = tiles.tile_data( tx, ty );

    const auto tile_dims  = tiles.tile_dims;

    const auto gix0 = tx * tile_dims.x;
    const auto giy0 = ty * tile_dims.y;

    const auto tile_stride_y = tiles.tile_ystride();

    for( int i = gpu::block::thread_rank(); i < tile_dims.x * tile_dims.y; i+= gpu::block::num_threads() ) {
        const auto ix = i % tile_dims.x;
        const auto iy = i / tile_dims.x;
        tdata[ iy * tile_stride_y + ix ] = 
            d_in[ (giy0 + iy) * in_stride.y + (gix0 + ix) * in_stride.x ];
    }
}

template< class T3, class T2, class T1 >
__global__
void scatter( 
    tiled_view<T1> tiles, 
    T2 const * const __restrict__ d_in, unsigned int const in_stride_y, T3 const scale
){
    unsigned const tx = blockIdx.x;
    unsigned const ty = blockIdx.y;

    T1 * const __restrict__ tdata = tiles.tile_data( tx, ty );

    const auto tile_dims  = tiles.tile_dims;

    const auto gix0 = tx * tile_dims.x;
    const auto giy0 = ty * tile_dims.y;

    const auto tile_stride_y = tiles.tile_ystride();

    for( int i = gpu::block::thread_rank(); i < tile_dims.x * tile_dims.y; i+= gpu::block::num_threads() ) {
        const auto ix = i % tile_dims.x;
        const auto iy = i / tile_dims.x;
        tdata[ iy * tile_stride_y + ix ] = 
            d_in[ (giy0 + iy) * in_stride_y + (gix0 + ix) ] * scale;
    }
}

template< class T3, class T2, class T1 >
__global__
void scatter( 
    tiled_view<T1> tiles, 
    T2 const * const __restrict__ d_in, uint2 const in_stride, T3 const scale
){
    unsigned const tx = blockIdx.x;
    unsigned const ty = blockIdx.y;

    T1 * const __restrict__ tdata = tiles.tile_data( tx, ty );

    const auto tile_dims  = tiles.tile_dims;

    const auto gix0 = tx * tile_dims.x;
    const auto giy0 = ty * tile_dims.y;

    const auto tile_stride_y = tiles.tile_ystride();

    for( int i = gpu::block::thread_rank(); i < tile_dims.x * tile_dims.y; i+= gpu::block::num_threads() ) {
        const auto ix = i % tile_dims.x;
        const auto iy = i / tile_dims.x;
        tdata[ iy * tile_stride_y + ix ] = 
            d_in[ (giy0 + iy) * in_stride.y + (gix0 + ix) * in_stride.x ] * scale;
    }
}

template< typename T >
__global__
void local_copy_to_gc_x( tiled_view<T> tiles, const int periodic_x )
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };

    T * __restrict__ local = tiles.tile_buffer( tile_idx );
    auto const ystride  = tiles.tile_ystride();
    auto const gc_x_lower = tiles.gc.x.lower;
    auto const gc_x_upper = tiles.gc.x.upper;
    auto const dims_x = tiles.tile_dims.x;

    {   // Copy from lower neighbour
        int neighbor_tx = tile_idx.x - 1;
        if ( periodic_x && neighbor_tx < 0 )
            neighbor_tx += tiles.local_ntiles.x;

        if ( neighbor_tx >= 0 ) {
            T * __restrict__ x_lower = tiles.tile_buffer( neighbor_tx, tile_idx.y );
            for( unsigned idx = gpu::block::thread_rank(); idx < tiles.tile_ext_dims.y * gc_x_lower; idx += gpu::block::num_threads() ) {
                const auto i = idx % gc_x_lower;
                const auto j = idx / gc_x_lower; 
                local[ i + j * ystride ] = x_lower[ dims_x + i + j * ystride ];
            }
        }
    }

    {   // Copy from upper neighbour
        int neighbor_tx = tile_idx.x + 1;
        if ( periodic_x && neighbor_tx >= static_cast<int>(tiles.local_ntiles.x) )
            neighbor_tx -= tiles.local_ntiles.x;

        if ( neighbor_tx < static_cast<int>(tiles.local_ntiles.x) ) {
            T * __restrict__ x_upper = tiles.tile_buffer( neighbor_tx, tile_idx.y );
            for( unsigned idx = gpu::block::thread_rank(); idx < tiles.tile_ext_dims.y * gc_x_upper; idx += gpu::block::num_threads() ) {
                const auto i = idx % gc_x_upper;
                const auto j = idx / gc_x_upper; 
                local[ gc_x_lower + dims_x + i + j * ystride ] = 
                    x_upper[ gc_x_lower + i + j * ystride ];
            }
        }
    }
}

template< typename T >
__global__
void local_copy_to_gc_y( tiled_view<T> tiles, const int periodic_y )
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };

    T * __restrict__ local = tiles.tile_buffer( tile_idx );
    auto const ystride  = tiles.tile_ystride();
    auto const gc_y_lower = tiles.gc.y.lower;
    auto const gc_y_upper = tiles.gc.y.upper;
    auto const ext_dims_x = tiles.tile_ext_dims.x;
    auto const dims_y = tiles.tile_dims.y;

    {   // Copy from lower neighbour
        int neighbor_ty = tile_idx.y - 1;
        if ( periodic_y && neighbor_ty < 0 )
            neighbor_ty += tiles.local_ntiles.y;

        if ( neighbor_ty >= 0 ) {
            T * __restrict__ y_lower = tiles.tile_buffer( tile_idx.x, neighbor_ty );
            for( unsigned idx = gpu::block::thread_rank(); idx < gc_y_lower * ext_dims_x; idx += gpu::block::num_threads() ) {
                const auto i = idx % ext_dims_x;
                const auto j = idx / ext_dims_x; 
                local[ i + j * ystride ] = 
                    y_lower[ i + ( dims_y + j ) * ystride ];
            }
        }
    }

    {   // Copy from upper neighbour
        int neighbor_ty = tile_idx.y + 1;
        if ( periodic_y && neighbor_ty >= static_cast<int>(tiles.local_ntiles.y) )
            neighbor_ty -= tiles.local_ntiles.y;

        if ( neighbor_ty < static_cast<int>(tiles.local_ntiles.y) ) {
            T * __restrict__ y_upper = tiles.tile_buffer( tile_idx.x, neighbor_ty );
            for( unsigned idx = gpu::block::thread_rank(); idx < gc_y_upper * ext_dims_x; idx += gpu::block::num_threads() ) {
                const auto i = idx % ext_dims_x;
                const auto j = idx / ext_dims_x; 
                local[ i + ( gc_y_lower + dims_y + j ) * ystride ] = 
                    y_upper[ i + ( gc_y_lower + j ) * ystride ];
            }
        }
    }
}

template< typename T >
__global__
void local_add_from_gc_x( tiled_view<T> tiles, const int periodic_x )
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    T * __restrict__ local = tiles.tile_buffer( tile_idx );
    auto const ystride  = tiles.tile_ystride();
    auto const gc_x_lower = tiles.gc.x.lower;
    auto const gc_x_upper = tiles.gc.x.upper;
    auto const dims_x = tiles.tile_dims.x;

    {   // Add from lower neighbour
        int neighbor_tx = tile_idx.x - 1;
        if ( periodic_x && neighbor_tx < 0 )
            neighbor_tx += tiles.local_ntiles.x;

        if ( neighbor_tx >= 0 ) {
            T * __restrict__ x_lower = tiles.tile_buffer( neighbor_tx, tile_idx.y );
            for( unsigned idx = gpu::block::thread_rank(); idx < tiles.tile_ext_dims.y * gc_x_upper; idx += gpu::block::num_threads() ) {
                const auto i = idx % gc_x_upper;
                const auto j = idx / gc_x_upper; 
                local[ gc_x_lower + i + j * ystride ] += x_lower[ gc_x_lower + dims_x + i + j * ystride ];
            }
        }
    }

    {   // Add from upper neighbour
        int neighbor_tx = tile_idx.x + 1;
        if ( periodic_x && neighbor_tx >= static_cast<int>(tiles.local_ntiles.x) )
            neighbor_tx -= tiles.local_ntiles.x;

        if ( neighbor_tx < static_cast<int>(tiles.local_ntiles.x) ) {
            T * __restrict__ x_upper = tiles.tile_buffer( neighbor_tx, tile_idx.y );
            for( unsigned idx = gpu::block::thread_rank(); idx < tiles.tile_ext_dims.y * gc_x_lower; idx += gpu::block::num_threads() ) {
                const auto i = idx % gc_x_lower;
                const auto j = idx / gc_x_lower; 
                local[ dims_x + i + j * ystride ] += x_upper[ i + j * ystride ];
            }
        }
    }
}

template< typename T >
__global__
void local_add_from_gc_y( tiled_view<T> tiles, const int periodic_y )
{
    const uint2 tile_idx = { blockIdx.x, blockIdx.y };

    auto * __restrict__ local = tiles.tile_buffer( tile_idx );
    auto const ystride  = tiles.tile_ystride();
    auto const gc_y_lower = tiles.gc.y.lower;
    auto const gc_y_upper = tiles.gc.y.upper;
    auto const ext_dims_x = tiles.tile_ext_dims.x;
    auto const dims_y = tiles.tile_dims.y;

    {   // Add from lower neighbour
        int neighbor_ty = tile_idx.y - 1;
        if ( periodic_y && neighbor_ty < 0 )
            neighbor_ty += tiles.local_ntiles.y;

        if ( neighbor_ty >= 0 ) {
            T * __restrict__ y_lower = tiles.tile_buffer( tile_idx.x, neighbor_ty );
            for( unsigned idx = gpu::block::thread_rank(); idx < gc_y_upper * ext_dims_x; idx += gpu::block::num_threads() ) {
                const auto i = idx % ext_dims_x;
                const auto j = idx / ext_dims_x; 
                local[ i + ( gc_y_lower + j ) * ystride ] += 
                    y_lower[ i + ( gc_y_lower + dims_y + j ) * ystride ];
            }
        }
    }

    {   // Add from upper neighbour
        int neighbor_ty = tile_idx.y + 1;
        if ( periodic_y && neighbor_ty >= static_cast<int>(tiles.local_ntiles.y) )
            neighbor_ty -= tiles.local_ntiles.y;

        if ( neighbor_ty < static_cast<int>(tiles.local_ntiles.y) ) {
            auto * __restrict__ y_upper = tiles.tile_buffer( tile_idx.x, neighbor_ty );
            for( unsigned idx = gpu::block::thread_rank(); idx < gc_y_lower * ext_dims_x; idx += gpu::block::num_threads() ) {
                const auto i = idx % ext_dims_x;
                const auto j = idx / ext_dims_x; 
                local[ i + ( dims_y + j ) * ystride ] += 
                    y_upper[ i + j * ystride ];
            }
        }
    }
}

template< typename T >
__global__
void x_shift_left( tiled_view<T> tiles, const unsigned shift )
{

    auto * local = gpu::block::shared_mem<T>();

    const uint2 tile_idx = { blockIdx.x, blockIdx.y };

    const int ystride = tiles.tile_ystride();
    const int tile_vol = tiles.tile_ext_dims.x * tiles.tile_ext_dims.y;

    auto * __restrict__ buffer = tiles.tile_buffer( tile_idx );

    for( int idx = gpu::block::thread_rank(); idx < tile_vol; idx += gpu::block::num_threads() ) {
        const int i = idx % tiles.tile_ext_dims.x;
        const int j = idx / tiles.tile_ext_dims.x;
        if ( i + shift < tiles.tile_ext_dims.x ) {
            local[ i + j * ystride ] = buffer[ (i + shift) + j * ystride ];
        } else {
            local[ i + j * ystride ] = T{0};
        }
    }

    gpu::block::sync();

    for( int idx = gpu::block::thread_rank(); idx < tile_vol; idx += gpu::block::num_threads() )
        buffer[idx] = local[idx];
}

/**
 * @brief CUDA kernel for a 3 point convolution [a,b,c] along x
 * 
 * @note The tile is copied to shared memory and the result is written
 *       directly to global memory. Guard cells are not modified (except as
 *       noted below) and must be updated afterwards.
 * 
 * @note With ext = 1 the result is also computed on the first upper x guard
 *       column and the first upper y guard row (ix = nx, iy = ny). Requires
 *       gc.x.upper >= 2 and gc.y.upper >= 1. This is needed when a physical
 *       boundary node lies on the first upper guard cell (e.g. PEC / PMC
 *       walls on a staggered grid); on internal tile boundaries these values
 *       are overwritten by the subsequent guard cell update.
 * 
 * @param ext   0 - interior only, 1 - include first upper guard row/column
 */
template< typename T, typename S >
__global__
void kernel3_x( tiled_view<T> tiles, S const a, S const b, S const c, int const ext )
{
    auto * __restrict__ A = gpu::block::shared_mem<T>();

    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    T * __restrict__ buffer = tiles.tile_buffer( tile_idx );

    // Copy data from tile buffer
    for( int i = gpu::block::thread_rank(); i < tiles.tile_vol; i += gpu::block::num_threads() )
        A[i] = buffer[i];

    // Synchronize 
    gpu::block::sync();

    // Apply kernel, written directly to the tile buffer
    const int ystride = tiles.tile_ystride();
    const int nx      = tiles.tile_dims.x + ext;
    const int ny      = tiles.tile_dims.y + ext;
    const int offset  = tiles.inner_offset;

    for( int idx = gpu::block::thread_rank(); idx < ny * nx; idx += gpu::block::num_threads() ) {
        const int iy = idx / nx;
        const int ix = idx % nx;
        const int k  = iy * ystride + ix + offset;
        buffer[ k ] = A[ k - 1 ] * a +
                      A[ k     ] * b +
                      A[ k + 1 ] * c;
    }
}

/**
 * @brief CUDA kernel for a 3 point convolution [a,b,c] along y
 * 
 * @note The tile is copied to shared memory and the result is written
 *       directly to global memory. Guard cells are not modified (except as
 *       noted below) and must be updated afterwards.
 * 
 * @note With ext = 1 the result is also computed on the first upper x guard
 *       column and the first upper y guard row (ix = nx, iy = ny). Requires
 *       gc.y.upper >= 2 and gc.x.upper >= 1. See kernel3_x().
 * 
 * @param ext   0 - interior only, 1 - include first upper guard row/column
 */
template< typename T, typename S >
__global__
void kernel3_y( tiled_view<T> tiles, S const a, S const b, S const c, int const ext )
{
    auto * __restrict__ A = gpu::block::shared_mem<T>();

    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    T * __restrict__ buffer = tiles.tile_buffer( tile_idx );

    // Copy data from tile buffer
    for( int i = gpu::block::thread_rank(); i < tiles.tile_vol; i += gpu::block::num_threads() )
        A[i] = buffer[i];

    // Synchronize 
    gpu::block::sync();

    // Apply kernel, written directly to the tile buffer
    const int ystride = tiles.tile_ystride();
    const int nx      = tiles.tile_dims.x + ext;
    const int ny      = tiles.tile_dims.y + ext;
    const int offset  = tiles.inner_offset;

    for( int idx = gpu::block::thread_rank(); idx < ny * nx; idx += gpu::block::num_threads() ) {
        const int iy = idx / nx;
        const int ix = idx % nx;
        const int k  = iy * ystride + ix + offset;
        buffer[ k ] = A[ k - ystride ] * a +
                      A[ k           ] * b +
                      A[ k + ystride ] * c;
    }
}

/**
 * @brief CUDA kernel for a separable 3x3 convolution, [a,b,c] along x
 *        followed by [a,b,c] along y
 * 
 * @note The x pass is applied to the tile's interior rows and to the first
 *       lower / upper y guard rows, so that the y pass can be done locally.
 *       This reads the corner guard cells, which must be up to date (as is
 *       the case after copy_to_gc()).
 * 
 * @note With ext = 1 the result is also computed on the first upper x guard
 *       column and the first upper y guard row (ix = nx, iy = ny). Requires
 *       gc.x.upper >= 2 and gc.y.upper >= 2. See kernel3_x().
 * 
 * @note Only the tile interior (plus the extension above, if requested) is
 *       written; guard cells must be updated afterwards.
 * 
 * @param ext   0 - interior only, 1 - include first upper guard row/column
 */
template< typename T, typename S >
__global__
void kernel3_xy( tiled_view<T> tiles, S const a, S const b, S const c, int const ext )
{
    auto * shm = gpu::block::shared_mem<T>();

    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    T * __restrict__ buffer = tiles.tile_buffer( tile_idx );

    T * __restrict__ A = & shm[0];
    T * __restrict__ B = & shm[tiles.tile_vol];

    // Copy data from tile buffer
    for( int i = gpu::block::thread_rank(); i < tiles.tile_vol; i += gpu::block::num_threads() )
        A[i] = buffer[i];

    // Synchronize 
    gpu::block::sync();

    const int ystride = tiles.tile_ystride();
    const int nx      = tiles.tile_dims.x + ext;
    const int ny      = tiles.tile_dims.y + ext;
    const int offset  = tiles.inner_offset;

    // x pass, rows -1 to ny (inclusive)
    for( int idx = gpu::block::thread_rank(); idx < ( ny + 2 ) * nx; idx += gpu::block::num_threads() ) {
        const int iy = idx / nx - 1;
        const int ix = idx % nx;
        const int k  = iy * ystride + ix + offset;
        B[ k ] = A[ k - 1 ] * a +
                 A[ k     ] * b +
                 A[ k + 1 ] * c;
    }

    // Synchronize 
    gpu::block::sync();

    // y pass, written directly to the tile buffer
    for( int idx = gpu::block::thread_rank(); idx < ny * nx; idx += gpu::block::num_threads() ) {
        const int iy = idx / nx;
        const int ix = idx % nx;
        const int k  = iy * ystride + ix + offset;
        buffer[ k ] = B[ k - ystride ] * a +
                      B[ k           ] * b +
                      B[ k + ystride ] * c;
    }
}

template< typename T >
__global__
void copy_to_gc_x_send( 
    T * const __restrict__ lower, T * const __restrict__ upper,
    tiled_view<T> tiles ) {
    
    auto const ext_dims = tiles.tile_ext_dims;
    auto const gc_x_lower = tiles.gc.x.lower;
    auto const gc_x_upper = tiles.gc.x.upper;
    auto const ystride = tiles.tile_ystride();
    auto const dims_x = tiles.tile_dims.x;
    auto const ty = blockIdx.y;

    // Note that blockIdx.x is constant inside a block so there is no thread divergence
    if ( blockIdx.x == 0 ) {
        // lower message
        if ( lower != nullptr ) {
            const unsigned int tx = 0;
            T * __restrict__ local = tiles.tile_buffer( tx, ty );
            T * __restrict__ msg = & lower [ ty * ext_dims.y * gc_x_upper ];
            for( int idx = gpu::block::thread_rank(); idx < ext_dims.y * gc_x_upper; idx += gpu::block::num_threads() ) {
                const int i = idx % gc_x_upper;
                const int j = idx / gc_x_upper;
                msg[ j * gc_x_upper + i ] = local[ j * ystride + gc_x_lower + i ];
            }
        }
    } else {
        // upper message
        if ( upper != nullptr ) {
            const unsigned int tx = tiles.local_ntiles.x - 1;
            T * __restrict__ local = tiles.tile_buffer( tx, ty );
            T * __restrict__ msg = & upper[ ty * ext_dims.y * gc_x_lower ];
            for( int idx = gpu::block::thread_rank(); idx < ext_dims.y * gc_x_lower; idx += gpu::block::num_threads() ) {
                const int i = idx % gc_x_lower;
                const int j = idx / gc_x_lower;
                msg[ j * gc_x_lower + i ] = local[ j * ystride + dims_x + i ];
            }
        }
    }
}

template< typename T >
__global__
void copy_to_gc_x_recv( 
    T * const __restrict__ lower, T * const __restrict__ upper,
    tiled_view<T> tiles ) {

    auto const ext_dims = tiles.tile_ext_dims;
    auto const gc_x_lower = tiles.gc.x.lower;
    auto const gc_x_upper = tiles.gc.x.upper;
    auto const ystride = tiles.tile_ystride();
    auto const dims_x = tiles.tile_dims.x;
    auto const ty = blockIdx.y;

    if ( blockIdx.x == 0 ) {
        // lower message
        if ( lower != nullptr ) {
            const unsigned int tx = 0;
            T * __restrict__ local = tiles.tile_buffer( tx, ty );
            T * __restrict__ msg = & lower [ ty * ext_dims.y * gc_x_lower ];
            for( int idx = gpu::block::thread_rank(); idx < ext_dims.y * gc_x_lower; idx += gpu::block::num_threads() ) {
                const int i = idx % gc_x_lower;
                const int j = idx / gc_x_lower;
                local[ j * ystride + i ] = msg[ j * gc_x_lower + i ];
            }
        }
    } else {
        // upper message
        if ( upper != nullptr ) {
            const unsigned int tx = tiles.local_ntiles.x-1;
            T * __restrict__ local = tiles.tile_buffer( tx, ty );
            T * __restrict__ msg = & upper[ ty * ext_dims.y * gc_x_upper ];
            for( int idx = gpu::block::thread_rank(); idx < ext_dims.y * gc_x_upper; idx += gpu::block::num_threads() ) {
                const int i = idx % gc_x_upper;
                const int j = idx / gc_x_upper;
                local[ j * ystride + gc_x_lower + dims_x + i ] = msg[ j * gc_x_upper + i ];
            }
        }
    }
}

template< typename T >
__global__
void copy_to_gc_y_send(  
    T * const __restrict__ lower, T * const __restrict__ upper,
    tiled_view<T> tiles ) {

    auto const ext_dims = tiles.tile_ext_dims;
    auto const gc_y_lower = tiles.gc.y.lower;
    auto const gc_y_upper = tiles.gc.y.upper;
    auto const ystride = tiles.tile_ystride();
    auto const dims_y = tiles.tile_dims.y;
    auto const tx = blockIdx.x;

    if ( blockIdx.y == 0 ) {
        // lower message
        if ( lower != nullptr ) {
            const unsigned int ty = 0;
            T * __restrict__ local = tiles.tile_buffer( tx, ty );
            T * __restrict__ msg = & lower [ tx * ext_dims.x * gc_y_upper ];
            for( int idx = gpu::block::thread_rank(); idx < gc_y_upper * ext_dims.x; idx += gpu::block::num_threads() ) {
                const int i = idx % ext_dims.x;
                const int j = idx / ext_dims.x;
                msg[ j * ext_dims.x + i ] = local[ ( gc_y_lower + j ) * ystride + i ];
            }
        }
    } else {
        // upper message
        if ( upper != nullptr ) {
            const unsigned int ty = tiles.local_ntiles.y - 1;
            T * __restrict__ local = tiles.tile_buffer( tx, ty );
            T * __restrict__ msg = & upper[ tx * ext_dims.x * gc_y_lower ];
            for( int idx = gpu::block::thread_rank(); idx < gc_y_lower * ext_dims.x; idx += gpu::block::num_threads() ) {
                const int i = idx % ext_dims.x;
                const int j = idx / ext_dims.x;
                msg[ j * ext_dims.x + i ] = local[ ( dims_y + j ) * ystride + i ];
            }
        }
    }
}

template< typename T >
__global__
void copy_to_gc_y_recv( 
    T * const __restrict__ lower, T * const __restrict__ upper,
    tiled_view<T> tiles ) {

    auto const ext_dims = tiles.tile_ext_dims;
    auto const gc_y_lower = tiles.gc.y.lower;
    auto const gc_y_upper = tiles.gc.y.upper;
    auto const ystride = tiles.tile_ystride();
    auto const dims_y = tiles.tile_dims.y;
    auto const tx = blockIdx.x;
    
    if ( blockIdx.y == 0 ) {
        // lower message
        if ( lower != nullptr ) {
            const unsigned int ty = 0;
            T * __restrict__ local = tiles.tile_buffer( tx, ty );
            T * __restrict__ msg = & lower [ tx * ext_dims.x * gc_y_lower ];
            for( int idx = gpu::block::thread_rank(); idx < gc_y_lower * ext_dims.x; idx += gpu::block::num_threads() ) {
                const int i = idx % ext_dims.x;
                const int j = idx / ext_dims.x;
                local[ j * ystride + i ] =  msg[ j * ext_dims.x + i ];
            }
        }
    } else {
        // upper message
        if ( upper != nullptr ) {
            const unsigned int ty = tiles.local_ntiles.y - 1;
            T * __restrict__ local = tiles.tile_buffer( tx, ty );
            T * __restrict__ msg = & upper[ tx * ext_dims.x * gc_y_upper ];
            for( int idx = gpu::block::thread_rank(); idx < gc_y_upper * ext_dims.x; idx += gpu::block::num_threads() ) {
                const int i = idx % ext_dims.x;
                const int j = idx / ext_dims.x;
                local[ ( gc_y_lower + dims_y + j ) * ystride + i ] =  msg[ j * ext_dims.x + i ];
            }
        }
    }
}

template< typename T >
__global__
void add_from_gc_x_send( 
    T * const __restrict__ lower, T * const __restrict__ upper,
    tiled_view<T> tiles ) {

    auto const ext_dims = tiles.tile_ext_dims;
    auto const gc_x_lower = tiles.gc.x.lower;
    auto const gc_x_upper = tiles.gc.x.upper;
    auto const ystride = tiles.tile_ystride();
    auto const dims_x = tiles.tile_dims.x;
    auto const ty = blockIdx.y;
    
    if ( blockIdx.x == 0 ) {
        // lower message
        if ( lower != nullptr ) {
            const unsigned int tx = 0;
            T * __restrict__ local = tiles.tile_buffer( tx, ty );
            T * __restrict__ msg = & lower [ ty * ext_dims.y * gc_x_lower ];
            for( int idx = gpu::block::thread_rank(); idx < ext_dims.y * gc_x_lower; idx += gpu::block::num_threads() ) {
                const int i = idx % gc_x_lower;
                const int j = idx / gc_x_lower;
                msg[ j * gc_x_lower + i ] = local[ j * ystride + i ];
            }
        }
    } else {
        // upper message
        if ( upper != nullptr ) {
            const unsigned int tx = tiles.local_ntiles.x - 1;
            T * __restrict__ local = tiles.tile_buffer( tx, ty );
            T * __restrict__ msg = & upper[ ty * ext_dims.y * gc_x_upper ];
            for( int idx = gpu::block::thread_rank(); idx < ext_dims.y * gc_x_upper; idx += gpu::block::num_threads() ) {
                const int i = idx % gc_x_upper;
                const int j = idx / gc_x_upper;
                msg[ j * gc_x_upper + i ] = local[ j * ystride + gc_x_lower + dims_x + i ];
            }
        }
    }
}

template< typename T >
__global__
void add_from_gc_x_recv( 
    T * const __restrict__ lower, T * const __restrict__ upper,
    tiled_view<T> tiles ) {

    auto const ext_dims = tiles.tile_ext_dims;
    auto const gc_x_lower = tiles.gc.x.lower;
    auto const gc_x_upper = tiles.gc.x.upper;
    auto const ystride = tiles.tile_ystride();
    auto const dims_x = tiles.tile_dims.x;
    auto const ty = blockIdx.y;
    
    if ( blockIdx.x == 0 ) {
        // lower message
        if ( lower != nullptr ) {
            const unsigned int tx = 0;
            T * __restrict__ local = tiles.tile_buffer( tx, ty );
            T * __restrict__ msg = & lower [ ty * ext_dims.y * gc_x_upper ];
            for( int idx = gpu::block::thread_rank(); idx < ext_dims.y * gc_x_upper; idx += gpu::block::num_threads() ) {
                const int i = idx % gc_x_upper;
                const int j = idx / gc_x_upper;
                local[ j * ystride + gc_x_lower + i ] += msg[ j * gc_x_upper + i ] ;
            }
        }
    } else {
        // upper message
        if ( upper != nullptr ) {
            const unsigned int tx = tiles.local_ntiles.x - 1;
            T * __restrict__ local = tiles.tile_buffer( tx, ty );
            T * __restrict__ msg = & upper[ ty * ext_dims.y * gc_x_lower ];
            for( int idx = gpu::block::thread_rank(); idx < ext_dims.y * gc_x_lower; idx += gpu::block::num_threads() ) {
                const int i = idx % gc_x_lower;
                const int j = idx / gc_x_lower;
                local[ j * ystride + dims_x + i ] += msg[ j * gc_x_lower + i ] ;
            }
        }
    }
}

template< typename T >
__global__
void add_from_gc_y_send( 
    T * const __restrict__ lower, T * const __restrict__ upper,
    tiled_view<T> tiles ) {

    auto const ext_dims = tiles.tile_ext_dims;
    auto const gc_y_lower = tiles.gc.y.lower;
    auto const gc_y_upper = tiles.gc.y.upper;
    auto const ystride = tiles.tile_ystride();
    auto const dims_y = tiles.tile_dims.y;
    auto const tx = blockIdx.x;

    if ( blockIdx.y == 0 ) {
        // lower message
        if ( lower != nullptr ) {
            const unsigned int ty = 0;
            T * __restrict__ local = tiles.tile_buffer( tx, ty );
            T * __restrict__ msg = & lower [ tx * ( ext_dims.x * gc_y_lower ) ];
            for( int idx = gpu::block::thread_rank(); idx < gc_y_lower * ext_dims.x; idx += gpu::block::num_threads() ) {
                const int i = idx % ext_dims.x;
                const int j = idx / ext_dims.x;
                msg[ j * ext_dims.x + i ] = local[ j * ystride + i ];
            }
        }
    } else {
        // upper message
        if ( upper != nullptr ) {
            const unsigned int ty = tiles.local_ntiles.y-1;
            T * __restrict__ local = tiles.tile_buffer( tx, ty );
            T * __restrict__ msg = & upper[ tx * gc_y_upper * ext_dims.x ];
            for( int idx = gpu::block::thread_rank(); idx < gc_y_upper * ext_dims.x; idx += gpu::block::num_threads() ) {
                const int i = idx % ext_dims.x;
                const int j = idx / ext_dims.x;
                msg[ j * ext_dims.x + i ] = local[ ( gc_y_lower + dims_y + j ) * ystride + i ];
            }
        }
    }
}

template< typename T >
__global__
void add_from_gc_y_recv( 
    T * const __restrict__ lower, T * const __restrict__ upper,
    tiled_view<T> tiles ) {

    auto const ext_dims = tiles.tile_ext_dims;
    auto const gc_y_lower = tiles.gc.y.lower;
    auto const gc_y_upper = tiles.gc.y.upper;
    auto const ystride = tiles.tile_ystride();
    auto const dims_y = tiles.tile_dims.y;
    auto const tx = blockIdx.x;
    
    if ( blockIdx.y == 0 ) {
        // lower message
        if ( lower != nullptr ) {
            const unsigned int ty = 0;
            T * __restrict__ local = tiles.tile_buffer( tx, ty );
            T * __restrict__ msg = & lower [ tx * gc_y_upper * ext_dims.x ];
            for( int idx = gpu::block::thread_rank(); idx < gc_y_upper * ext_dims.x; idx += gpu::block::num_threads() ) {
                const int i = idx % ext_dims.x;
                const int j = idx / ext_dims.x;
                local[ ( gc_y_lower + j ) * ystride + i ] += msg[ j * ext_dims.x + i ];
            }
        }
    } else {
        // upper message
        if ( upper != nullptr ) {
            const unsigned int ty = tiles.local_ntiles.y - 1;
            T * __restrict__ local = tiles.tile_buffer( tx, ty );
            T * __restrict__ msg = & upper[ tx * gc_y_lower * ext_dims.x ];
            for( int idx = gpu::block::thread_rank(); idx < gc_y_lower * ext_dims.x; idx += gpu::block::num_threads() ) {
                const int i = idx % ext_dims.x;
                const int j = idx / ext_dims.x;
                local[ ( dims_y + j ) * ystride + i ] +=  msg[ j * ext_dims.x + i ];
            }
        }
    }
}

} // namespace kernel

/**
 * @brief Tiled grid class with MPI support
 * 
 */
template <class T>
class tiled {
    protected:

    // Tags are paired so that a message sent with dest::lower is received with
    // source::upper (both have the same value). This ensures MPI tag matching
    // between sender and receiver without extra bookkeeping.

    /// @brief tags for outgoing messages
    struct source { enum tag { lower = 0, upper = 1 }; };
    /// @brief tags for incoming messages
    struct dest   { enum tag { upper = 0, lower = 1 }; };

    /// @brief Parallel partition
    const mpi::cart2d & part;

    /// @brief Local number of tiles
    uint2 local_ntiles;

    /// @brief Start position of local tiles in global tile grid
    uint2 local_tile_start;

    /// @brief Consider local boundaries periodic
    int2 local_periodic;

    /// @brief Local grid dimensions (all local tiles)
    uint2 local_dims;

    /// @brief Buffers for sending messages
    bounds< mpi::message<T>* > msg_send;

    /// @brief Buffers for receiving messages
    bounds< mpi::message<T>* > msg_recv;

    /// @brief Data buffer
    T * d_buffer;

    /**
     * @brief Set the local node information. This information will (may) be
     * different for each parallel node
     * 
     * @note Global periodic information is taken from the parallel partition
     * 
     */
    void initialize( ) {

        // Get local number of tiles and position
        part.grid_local( global_ntiles, local_ntiles, local_tile_start );

        // Get local grid size
        local_dims = local_ntiles * tile_dims;

        // Get local periodic flag
        local_periodic.x = part.periodic.x && (part.dims.x == 1);
        local_periodic.y = part.periodic.y && (part.dims.y == 1);

        // Allocate main data buffer
        d_buffer = gpu::device::malloc<T>( buffer_size() );

        // Get maximum message size
        int max_msg_size = std::max(
            ( local_ntiles.y * tile_ext_dims.y ) * std::max( gc.x.lower, gc.x.upper ),
            std::max( gc.y.lower, gc.y.upper ) * ( local_ntiles.x * tile_ext_dims.x )
        );

        // Allocate message buffers
        msg_recv.lower = new mpi::message<T>( max_msg_size, part.get_comm() );
        msg_recv.upper = new mpi::message<T>( max_msg_size, part.get_comm() );
        msg_send.lower = new mpi::message<T>( max_msg_size, part.get_comm() );
        msg_send.upper = new mpi::message<T>( max_msg_size, part.get_comm() );

    }

    private:

    /**
     * @brief Validate grid / parallel parameters. The execution will stop if
     * errors are found.
     * 
     */
    void validate_parameters() {
        // Grid parameters
        if ( global_ntiles.x == 0 || global_ntiles.y == 0 ) {
            mpi::fatal( "Invalid number of tiles: " + to_string(global_ntiles) );
        }

        if ( tile_dims.x == 0 || tile_dims.y == 0 ) {
            mpi::fatal( "Invalid tile dimensions: " + to_string(tile_dims) );
        }

        if ( gc.x.lower > tile_dims.x || gc.x.upper > tile_dims.x )
            mpi::fatal( "Number of x guard cells exceeds tile size along x" );

        if ( gc.y.lower > tile_dims.y || gc.y.upper > tile_dims.y )
            mpi::fatal( "Number of y guard cells exceeds tile size along y" );

        // Parallel partition
        if ( part.dims.x > global_ntiles.x ) {
            mpi::fatal ( "Number of parallel nodes along x (" +
                         std::to_string(part.dims.x) +
                         ") is larger than number of tiles along x(" +
                         std::to_string(global_ntiles.x) + ')');
        }

        if ( part.dims.y > global_ntiles.y ) {
            mpi::fatal ( "Number of parallel nodes along y (" +
                         std::to_string(part.dims.y) +
                         ") is larger than number of tiles along y(" +
                         std::to_string(global_ntiles.y) + ')');
        }
    }

    public:

    /// @brief Global number of tiles
    const uint2 global_ntiles;

    /// @brief Tile grid dimensions
    const uint2 tile_dims;
    
    /// @brief Tile guard cells
    const bounds_2d<unsigned int> gc;
    
    /// @brief Tile grid dimensions including guard cells
    const uint2 tile_ext_dims;

    /// @brief Offset, in cells, from the start of a tile's data buffer to its local (0,0) point
    const unsigned int inner_offset;

    /// @brief Tile volume (may be larger than tile_ext_dim.x * tile_ext_dim.y for alignment)
    const std::size_t tile_vol;

    /// @brief Object name
    std::string name;

    /**
     * @brief Construct a new grid object
     * 
     * @param global_ntiles     Global number of tiles
     * @param tile_dims         Individual tile size
     * @param gc                Number of guard cells
     * @param part              Parallel partition
     */
    tiled( uint2 const global_ntiles, uint2 const tile_dims, bounds_2d<unsigned int> const gc, const mpi::cart2d & part ):
        part( part ),
        d_buffer( nullptr ), 
        global_ntiles( global_ntiles ),
        tile_dims( tile_dims ),
        gc(gc),
        tile_ext_dims( make_uint2( gc.x.lower + tile_dims.x + gc.x.upper,
                            gc.y.lower + tile_dims.y + gc.y.upper )),
        inner_offset( gc.y.lower * tile_ext_dims.x + gc.x.lower ),
        tile_vol( roundup<4>( tile_ext_dims.x * tile_ext_dims.y ) ),
        name( "tiled grid" )
    {
        // Validate parameters
        validate_parameters();

        // Set local information (ntiles, tile_start and local_periodic)
        initialize();
    };

    /**
     * @brief Construct a new tile grid object
     * 
     * @note: The number of guard cells is set to 0
     * 
     * @param global_ntiles     Global number of tiles
     * @param tile_dims                Individual tile size
     * @param part              Parallel partition
     */
    tiled( uint2 const global_ntiles, uint2 const tile_dims, const mpi::cart2d & part ):
        part( part ),
        d_buffer( nullptr ),
        global_ntiles( global_ntiles ),
        tile_dims( tile_dims ),
        gc( 0 ),
        tile_ext_dims( make_uint2( tile_dims.x, tile_dims.y )),
        inner_offset( 0 ),
        tile_vol( roundup<4>( tile_dims.x * tile_dims.y )),
        name( "tiled grid" )
    {
        // Validate parameters
        validate_parameters();

        // Set local information (ntiles, tile_start and periodic)
        initialize();
    };

    /**
    * @brief Move constructor
    *
    * @note Steals the data buffer and message buffers from `other`, leaving
    *       it in a valid but empty state (safe to destruct).
    *
    * @param other     tiled grid to move from
    */
    tiled( tiled && other ) noexcept :
        part( other.part ),
        local_ntiles( other.local_ntiles ),
        local_tile_start( other.local_tile_start ),
        local_periodic( other.local_periodic ),
        local_dims( other.local_dims ),
        msg_send( other.msg_send ),
        msg_recv( other.msg_recv ),
        d_buffer( other.d_buffer ),
        global_ntiles( other.global_ntiles ),
        tile_dims( other.tile_dims ),
        gc( other.gc ),
        tile_ext_dims( other.tile_ext_dims ),
        inner_offset( other.inner_offset ),
        tile_vol( other.tile_vol ),
        name( std::move( other.name ) )
    {
        // Null out other's owned resources so its destructor is a no-op
        other.d_buffer = nullptr;

        other.msg_send.lower = nullptr;
        other.msg_send.upper = nullptr;
        other.msg_recv.lower = nullptr;
        other.msg_recv.upper = nullptr;
    }

    /**
     * @brief tiled grid destructor
     * 
     */
    ~tiled(){
        delete msg_recv.lower;
        delete msg_recv.upper;
        delete msg_send.lower;
        delete msg_send.upper;

        if ( d_buffer != nullptr ) gpu::device::free( d_buffer );
    };

    /**
     * @brief Delete default copy constructor
     * 
     */
    tiled(const tiled&) = delete;

    /**
     * @brief Delete default copy assignment
     * 
     */
    tiled& operator=(const tiled&) = delete;

    /**
     * @brief Returns a view of the tiled grid
     * 
     * @return tiled_view<T> 
     */
    tiled_view<T> view() noexcept {
        return { 
            local_ntiles, 
            d_buffer, 
            tile_dims,
            tile_ext_dims, 
            inner_offset, 
            gc,
            tile_vol };
    }

    /**
     * @brief Returns a read-only view of the tiled grid
     * 
     * @note const qualified so that a const tiled grid can still hand a view
     *       to a read-only kernel. view() stays non-const: handing out a
     *       mutable view is a mutating operation on the grid.
     * 
     * @return tiled_view<const T> 
     */
    tiled_view<const T> cview() const noexcept {
        return { 
            local_ntiles, 
            d_buffer, 
            tile_dims,
            tile_ext_dims, 
            inner_offset, 
            gc,
            tile_vol };
    }

    /**
     * @brief Get a pointer to the data buffer
     * 
     * @return T* 
     */
    T * buffer() const noexcept { return d_buffer; }

    /**
     * @brief Get a pointer to the start of a specific tile's data buffer
     * 
     * @note This points at the first guard cell of the tile (if any), not at
     *       the tile's local (0,0) point. Use tile_data() for a pointer to
     *       the local (0,0) point instead.
     * 
     * @param tid   Tile index (flat)
     * @return T* 
     */
    T * tile_buffer( const unsigned int tid ) const noexcept {
        return & d_buffer[ tid * tile_vol ];
    }

    /**
     * @brief Get a pointer to the start of a specific tile's data buffer
     * 
     * @note This points at the first guard cell of the tile (if any), not at
     *       the tile's local (0,0) point. Use tile_data() for a pointer to
     *       the local (0,0) point instead.
     * 
     * @param tx    x tile index
     * @param ty    y tile index
     * @return T* 
     */
    T * tile_buffer( const unsigned int tx, const unsigned int ty ) const noexcept {
        return & d_buffer[ (ty * local_ntiles.x + tx) * tile_vol ];
    }

    /**
     * @brief Get a pointer to the start of a specific tile's data buffer
     * 
     * @note This points at the first guard cell of the tile (if any), not at
     *       the tile's local (0,0) point. Use tile_data() for a pointer to
     *       the local (0,0) point instead.
     * 
     * @param tid   Tile index (x,y)
     * @return T* 
     */
    T * tile_buffer( const uint2 tid ) const noexcept {
        return & d_buffer[ (tid.y * local_ntiles.x + tid.x) * tile_vol ];
    }

    /**
     * @brief Get a pointer to a specific tile's local (0,0) point
     * 
     * @note This skips past the tile's guard cells (if any). Use
     *       tile_buffer() for a pointer to the start of the tile's data
     *       buffer instead.
     * 
     * @param tid   Tile index (flat)
     * @return T* 
     */
    T * tile_data( const unsigned int tid ) const noexcept {
        return & d_buffer[ tid * tile_vol + inner_offset ];
    }

    /**
     * @brief Get a pointer to a specific tile's local (0,0) point
     * 
     * @note This skips past the tile's guard cells (if any). Use
     *       tile_buffer() for a pointer to the start of the tile's data
     *       buffer instead.
     * 
     * @param tx    x tile index
     * @param ty    y tile index
     * @return T* 
     */
    T * tile_data( const unsigned int tx, const unsigned int ty ) const noexcept {
        return & d_buffer[ (ty * local_ntiles.x + tx) * tile_vol + inner_offset ];
    }

    /**
     * @brief Get a pointer to a specific tile's local (0,0) point
     * 
     * @note This skips past the tile's guard cells (if any). Use
     *       tile_buffer() for a pointer to the start of the tile's data
     *       buffer instead.
     * 
     * @param tid   Tile index (x,y)
     * @return T* 
     */
    T * tile_data( const uint2 tid ) const noexcept {
        return & d_buffer[ (tid.y * local_ntiles.x + tid.x) * tile_vol + inner_offset ];
    }

    /**
     * @brief Get the local number of tiles
     * 
     * @return uint2 
     */
    uint2 get_local_ntiles() const noexcept { return local_ntiles; };

    /**
     * @brief Returns the local tile offset in the global MPI tile grid
     * 
     * @return uint2 
     */
    uint2 get_local_tile_start() const noexcept { return local_tile_start; };

    /**
     * @brief Get the global grid dimensions
     * 
     * @return uint2 
     */
    uint2 get_global_dims() const noexcept { return global_ntiles  * tile_dims; }

    /**
     * @brief Get the local dims object
     * 
     * @return uint2 
     */
    uint2 get_local_dims() const noexcept { return local_dims; }

    /**
     * @brief Get the parallel 
     * 
     * @return const mpi::cart2d& 
     */
    const mpi::cart2d & get_part() const noexcept { return  part; }

    /**
     * @brief Stream extraction
     * 
     * @param os 
     * @param obj 
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, const tiled & obj) {
        return os 
            << obj.name << "{ " 
            << obj.local_ntiles << " local tiles, "
            << obj.local_tile_start << " local start, "
            << obj.global_ntiles << " global tiles, "
            << obj.tile_dims << " points/tile }";
    }


    /**
     * @brief Buffer size
     * 
     * @return total size of data buffers (in elements)
     */
    std::size_t buffer_size() const noexcept {
        return (static_cast <std::size_t> (tile_vol)) * ( local_ntiles.x * local_ntiles.y ) ;
    };

    /**
     * @brief zero device data on a grid grid
     * 
     */
    void zero() {
        gpu::device::zero( d_buffer, buffer_size() );
    };

    /**
     * @brief Sets data to a constant value
     * 
     * @param val       Value
     */
    void set( T const & val ){
        gpu::device::set( d_buffer, buffer_size(), val );
    };

    /**
     * @brief Adds another grid object on top of local object
     * 
     * @param rhs         Other object to add
     */
    void add( const tiled &rhs ) {
        // Check that the objects are compatible
        assert(( local_ntiles == rhs.get_local_ntiles() && tile_vol == rhs.tile_vol ));

        size_t const size = buffer_size( );
        int const block = 512;
        int const grid = (size -1) / block + 1;

        kernel::add <<< grid, block >>> ( d_buffer, rhs.d_buffer, size );
    };

    /**
     * @brief Operator +=
     * 
     * @param rhs           Other grid to add
     * @return grid<T>& 
     */
    tiled & operator+=(const tiled & rhs) {
        add( rhs );
        return *this;
    }

    /**
     * @brief Gather tiled grid values
     * 
     * @note
     * The default behavior is to output data into a contiguous array of 
     * dimensions local_dims.y * local_dims.x. The stride parameter can be used
     * to specify different memory layouts.
     * 
     * @tparam T2               Output buffer datatype, must support T2 = T assignment
     * @param out               Output buffer
     * @param stride            (optional) Output buffer stride, defaults to a
     *                          contiguous memory layout
     * @return unsigned int     Total number of cells
     */
    template< typename T2 >
    unsigned int gather( T2 * const __restrict__ d_out, uint2 stride = {0,0} ) const {

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
            kernel::gather <<< grid, block >>> (
                d_out, stride.y, cview()
            );
        } else {
            // Arbitrary x stride
            kernel::gather <<< grid, block >>> (
                d_out, stride, cview()
            );
        }

        return local_dims.x * local_dims.y;
    }

    /**
     * @brief Scatter data into the tile grid and updates guard cell values
     * 
     * @note
     * The default behavior is consider that the input data is a contiguous
     * array of dimensions local_dims.y * local_dims.x. The stride parameter
     * can be used to specify different memory layouts.
     * 
     * @tparam T2               Input buffer datatype
     * @param d_in              Input buffer
     * @param in_stride         (optional) Input buffer stride, defaults to a
     *                          contiguous memory layout
     * @return unsigned int     Total number of cells
     */
    template< typename T2 >
    unsigned int scatter( T2 const * const __restrict__ d_in, uint2 stride = {0,0} ) {

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
            kernel::scatter <<< grid, block >>> (
                view(), d_in, stride.y
            );
        } else {
            // Arbitrary x stride
            kernel::scatter <<< grid, block >>> (
                view(), d_in, stride
            );

        }

        // Update guard cell values
        copy_to_gc();

        return local_dims.x * local_dims.y;
    }

    /**
     * @brief Scatter data into the tile grid and updates guard cell values
     * 
     * @note
     * The default behavior is consider that the input data is a contiguous
     * array of dimensions local_dims.y * local_dims.x. The stride parameter
     * can be used to specify different memory layouts.
     *
     * The operation T2 = T3 * T must be supported.
     * 
     * @tparam T2               Input buffer datatype
     * @tparam T3               Scale factor datatype
     * @param d_in              Input buffer
     * @param scale             Scale factor
     * @param in_stride         (optional) Input buffer stride, defaults to a
     *                          contiguous memory layout
     * @return unsigned int     Total number of cells
     */
    template< typename T2, typename T3 >
    unsigned int scatter( T2 const * const __restrict__ d_in, T3 const scale, 
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
            kernel::scatter <<< grid, block >>> (
                view(), d_in, stride.y, scale
            );
        } else {
            // Arbitrary x stride
            kernel::scatter <<< grid, block >>> (
                view(), d_in, stride, scale
            );
        }

        // Update guard cell values
        copy_to_gc();

        return local_dims.x * local_dims.y;
    }

    protected:

    // The local_* guard cell routines below only handle tile-to-tile copies
    // inside this node. Used on their own in a multi-node run they leave the
    // guard cells on the node boundary stale, with no diagnostic. They are
    // implementation details of copy_to_gc_x() / copy_to_gc_y(); external
    // callers should use those instead.

    /**
     * @brief Copies edge values to X neighboring guard cells
     * 
     * @note Node local operation, does not exchange data with other parallel
     *       nodes. Use copy_to_gc_x() instead.
     */
    void local_copy_to_gc_x() {

        dim3 grid( local_ntiles.x, local_ntiles.y );
        dim3 block( 64 );

        kernel::local_copy_to_gc_x <<< grid, block >>> (
            view(), local_periodic.x
        );
    }

    /**
     * @brief Copies edge values to Y neighboring guard cells
     * 
     * @note Node local operation, does not exchange data with other parallel
     *       nodes. Use copy_to_gc_y() instead.
     */
    void local_copy_to_gc_y() {

        dim3 grid( local_ntiles.x, local_ntiles.y );
        dim3 block( 64 );

        kernel::local_copy_to_gc_y <<< grid, block >>> (
            view(), local_periodic.y
        );
    }

    public:

    /**
     * @brief Copies x values to neighboring guard cells, including cells on 
     *        other parallel nodes
     * 
     * @warning Collective operation, must be called by all parallel nodes.
     */
    void copy_to_gc_x() {

        // Get x neighbors
        int lnode = part.get_neighbor(-1, 0 );
        int unode = part.get_neighbor(+1, 0 );

        // Disable messages if only 1 node along  x direction
        if ( part.dims.x == 1 ) lnode = unode = -1;

        // Active messages
        const bool send_lower = lnode >= 0 && gc.x.upper > 0;
        const bool send_upper = unode >= 0 && gc.x.lower > 0;
        const bool recv_lower = lnode >= 0 && gc.x.lower > 0;
        const bool recv_upper = unode >= 0 && gc.x.upper > 0;

        // Post message receives
        if ( recv_lower ) msg_recv.lower->irecv( lnode, source::lower );
        if ( recv_upper ) msg_recv.upper->irecv( unode, source::upper );

        // Pack send messages
        if ( send_lower || send_upper ) {
            dim3 grid( 2, local_ntiles.y );
            dim3 block( 64 );

            T * lower = ( send_lower ) ? msg_send.lower -> buffer : nullptr;
            T * upper = ( send_upper ) ? msg_send.upper -> buffer : nullptr;

            kernel::copy_to_gc_x_send <<< grid, block >>> (
                lower, upper, view()
            );
            gpu::device::sync();
        }

        // Send message - lower neighbor
        if ( send_lower ) {
            const int msg_size = ( tile_ext_dims.y * local_ntiles.y ) * gc.x.upper;
            msg_send.lower->isend( msg_size, lnode, dest::lower );
        }

        // Send message - upper neighbor
        if ( send_upper ) {
            const int msg_size = ( tile_ext_dims.y * local_ntiles.y ) * gc.x.lower;
            msg_send.upper->isend( msg_size, unode, dest::upper );
        }

        // Process local tiles
        local_copy_to_gc_x();

        // Wait for receive messages
        if ( recv_lower ) msg_recv.lower-> wait();
        if ( recv_upper ) msg_recv.upper-> wait();

        // Unpack receive messages
        if ( recv_lower || recv_upper ) {
            dim3 grid( 2, local_ntiles.y );
            dim3 block( 64 );

            T * lower = ( recv_lower ) ? msg_recv.lower -> buffer : nullptr;
            T * upper = ( recv_upper ) ? msg_recv.upper -> buffer : nullptr;

            kernel::copy_to_gc_x_recv <<< grid, block >>>(
                lower, upper, view()
            );
            gpu::device::sync();
        }


        // Wait for send messages to complete
        if ( send_lower ) msg_send.lower->wait( );
        if ( send_upper ) msg_send.upper->wait( );
    }

    /**
     * @brief Copies y values to neighboring guard cells, including cells on 
     *        other parallel nodes
     * 
     * @warning Collective operation, must be called by all parallel nodes.
     */
    void copy_to_gc_y() {

        // Get y neighbors
        int lnode = part.get_neighbor(0, -1);
        int unode = part.get_neighbor(0, +1);

        // Disable messages if only 1 node along y direction
        if ( part.dims.y == 1 ) lnode = unode = -1;

        // Active messages
        const bool send_lower = lnode >= 0 && gc.y.upper > 0;
        const bool send_upper = unode >= 0 && gc.y.lower > 0;
        const bool recv_lower = lnode >= 0 && gc.y.lower > 0;
        const bool recv_upper = unode >= 0 && gc.y.upper > 0;

        // Post message receives
        if ( recv_lower ) msg_recv.lower->irecv( lnode, source::lower );
        if ( recv_upper ) msg_recv.upper->irecv( unode, source::upper );

        // Pack send messages
        if ( send_lower || send_upper ) {
            dim3 grid( local_ntiles.x, 2 );
            dim3 block( 64 );

            T * lower = ( send_lower ) ? msg_send.lower -> buffer : nullptr;
            T * upper = ( send_upper ) ? msg_send.upper -> buffer : nullptr;

            kernel::copy_to_gc_y_send <<< grid, block >>> (
                lower, upper, view()
            );
            gpu::device::sync();
        }

        // Post message sends
        if ( send_lower ) {
            const int msg_size =  gc.y.upper * ( local_ntiles.x * tile_ext_dims.x );
            msg_send.lower->isend( msg_size, lnode, dest::lower );
        }
        if ( send_upper ) {
            const int msg_size = gc.y.lower * ( local_ntiles.x * tile_ext_dims.x );
            msg_send.upper -> isend( msg_size, unode, dest::upper );
        }

        // Process local tiles
        local_copy_to_gc_y();

        // Wait for receive messages
        if ( recv_lower ) msg_recv.lower-> wait();
        if ( recv_upper ) msg_recv.upper-> wait();

        // Unpack receive messages
        if ( recv_lower || recv_upper ) {
            dim3 grid( local_ntiles.x, 2 );
            dim3 block( 64 );

            T * lower = ( recv_lower ) ? msg_recv.lower -> buffer : nullptr;
            T * upper = ( recv_upper ) ? msg_recv.upper -> buffer : nullptr;

            kernel::copy_to_gc_y_recv <<< grid, block >>>(
                lower, upper, view()
            );
            gpu::device::sync();
        }

        // Wait for send messages to complete
        if ( send_lower ) msg_send.lower->wait( );
        if ( send_upper ) msg_send.upper->wait( );
    }

    /**
     * @brief Copies edge values to neighboring guard cells, including other
     *        parallel nodes
     * 
     * @warning Collective operation, must be called by all parallel nodes.
     */
    void copy_to_gc()  {

        // Copy along x direction
        copy_to_gc_x();

        // Copy along y direction
        copy_to_gc_y();

    };

    protected:

    // As above for the local_* routines. add_from_gc_x() / add_from_gc_y()
    // are also kept internal: a partial guard cell reduction along a single
    // direction leaves the tile corners unaccounted for, so callers should
    // use add_from_gc(), which performs both directions in the required
    // order. Promote them if a caller genuinely needs a single direction.

    /**
     * @brief Adds values from neighboring x guard cells to local data
     * 
     * @note Node local operation, does not exchange data with other parallel
     *       nodes. Use add_from_gc() instead.
     */
    void local_add_from_gc_x() {
        dim3 grid( local_ntiles.x, local_ntiles.y );
        dim3 block( 64 );

        kernel::local_add_from_gc_x <<< grid, block >>> (
            view(), local_periodic.x
        );
    }

    /**
     * @brief Adds values from neighboring y guard cells to local data
     * 
     * @note Node local operation, does not exchange data with other parallel
     *       nodes. Use add_from_gc() instead.
     */
    void local_add_from_gc_y(){
        
        dim3 grid( local_ntiles.x, local_ntiles.y );
        dim3 block( 64 );

        kernel::local_add_from_gc_y <<< grid, block >>> (
            view(), local_periodic.y
        );
    };

    /**
     * @brief Adds values from neighboring x guard cells to local data,
     *        including cells from other parallel nodes
     * 
     * @warning Collective operation, must be called by all parallel nodes.
     */
    void add_from_gc_x() {

        // Get x neighbors
        int lnode = part.get_neighbor(-1, 0 );
        int unode = part.get_neighbor(+1, 0 );

        // Disable messages if only 1 node along  x direction
        if ( part.dims.x == 1 ) lnode = unode = -1;

        // Active messages
        const bool send_lower = lnode >= 0 && gc.x.lower > 0;
        const bool send_upper = unode >= 0 && gc.x.upper > 0;
        const bool recv_lower = lnode >= 0 && gc.x.upper > 0;
        const bool recv_upper = unode >= 0 && gc.x.lower > 0;

        // Post message receives
        if ( recv_lower ) msg_recv.lower->irecv( lnode, source::lower );
        if ( recv_upper ) msg_recv.upper->irecv( unode, source::upper );

        // Pack send messages
        if ( send_lower || send_upper ) {
            dim3 grid( 2, local_ntiles.y );
            dim3 block( 64 );

            T * lower = ( send_lower ) ? msg_send.lower -> buffer : nullptr;
            T * upper = ( send_upper ) ? msg_send.upper -> buffer : nullptr;

            kernel::add_from_gc_x_send <<< grid, block >>> (
                lower, upper, view()
            );
            gpu::device::sync();
        }

        // Send messages
        if ( send_lower ) {
            const int msg_size = ( local_ntiles.y * tile_ext_dims.y ) * gc.x.lower;
            msg_send.lower->isend( msg_size, lnode, dest::lower );
        }
        if ( send_upper ) {
            const int msg_size = ( local_ntiles.y * tile_ext_dims.y ) * gc.x.upper;
            msg_send.upper->isend( msg_size, unode, dest::upper );
        }

        // Process local tiles
        local_add_from_gc_x();

        // Wait for receive messages
        if ( recv_lower ) msg_recv.lower-> wait();
        if ( recv_upper ) msg_recv.upper-> wait();

        // Unpack receive messages
        if ( recv_lower || recv_upper ) {
            dim3 grid( 2, local_ntiles.y );
            dim3 block( 64 );

            T * lower = ( recv_lower ) ? msg_recv.lower -> buffer : nullptr;
            T * upper = ( recv_upper ) ? msg_recv.upper -> buffer : nullptr;

            kernel::add_from_gc_x_recv <<< grid, block >>>(
                lower, upper, view()
            );
            gpu::device::sync();
        }

        // Wait for send messages to complete
        if ( send_lower ) msg_send.lower->wait( );
        if ( send_upper ) msg_send.upper->wait( );
    }

    /**
     * @brief Adds values from neighboring y guard cells to local data,
     *        including cells from other parallel nodes
     * 
     * @warning Collective operation, must be called by all parallel nodes.
     */
    void add_from_gc_y() {
        // Get y neighbors
        int lnode = part.get_neighbor( 0, -1 );
        int unode = part.get_neighbor( 0, +1 );

        // Disable messages if only 1 node along y direction
        if ( part.dims.y == 1 ) lnode = unode = -1;

        // Active messages
        const bool send_lower = lnode >= 0 && gc.y.lower > 0;
        const bool send_upper = unode >= 0 && gc.y.upper > 0;
        const bool recv_lower = lnode >= 0 && gc.y.upper > 0;
        const bool recv_upper = unode >= 0 && gc.y.lower > 0;

        // Post message receives
        if ( recv_lower ) msg_recv.lower->irecv( lnode, source::lower );
        if ( recv_upper ) msg_recv.upper->irecv( unode, source::upper );

        // Pack send messages
        if ( send_lower || send_upper ) {
            dim3 grid( local_ntiles.x, 2 );
            dim3 block( 64 );

            T * lower = ( send_lower ) ? msg_send.lower -> buffer : nullptr;
            T * upper = ( send_upper ) ? msg_send.upper -> buffer : nullptr;

            kernel::add_from_gc_y_send <<< grid, block >>> (
                lower, upper, view()
            );
            gpu::device::sync();
        }

        // Send message - lower neighbor
        if ( send_lower ) {
            const int msg_size = gc.y.lower * ( local_ntiles.x * tile_ext_dims.x );
            msg_send.lower->isend( msg_size, lnode, dest::lower );
        }

        // Send message - upper neighbor
        if ( send_upper ) {
            const int msg_size    = gc.y.upper * ( local_ntiles.x * tile_ext_dims.x );
            msg_send.upper->isend( msg_size, unode, dest::upper );
        }

        // Process local tiles
        local_add_from_gc_y();

        // Wait for receive messages
        if ( recv_lower ) msg_recv.lower-> wait();
        if ( recv_upper ) msg_recv.upper-> wait();

        // Unpack receive messages
        if ( recv_lower || recv_upper ) {
            dim3 grid( local_ntiles.x, 2 );
            dim3 block( 64 );

            T * lower = ( recv_lower ) ? msg_recv.lower -> buffer : nullptr;
            T * upper = ( recv_upper ) ? msg_recv.upper -> buffer : nullptr;

            kernel::add_from_gc_y_recv <<< grid, block >>>(
                lower, upper, view()
            );
            gpu::device::sync();
        }

        // Wait for send messages to complete
        if ( send_lower ) msg_send.lower->wait( );
        if ( send_upper ) msg_send.upper->wait( );
    }

    public:

    /**
     * @brief Adds values from neighboring guard cells to local data, including
     *        values from other parallel nodes
     * 
     * @warning Collective operation, must be called by all parallel nodes.
     */
    void add_from_gc() {
        // Add along x direction
        add_from_gc_x();

        // Add along y direction
        add_from_gc_y();
    }

    /**
     * @brief Left shifts data for a specified amount
     * 
     * @warning This operation is only allowed if the number of upper x guard cells
     * is greater or equal to the requested shift
     * 
     * @param shift Number of cells to shift
     */
    void x_shift_left( unsigned int const shift ) {

        if ( shift > 0 && shift <= gc.x.upper ) {

            dim3 grid( local_ntiles.x, local_ntiles.y );

            // Shift data using guard cell values
            size_t shm_size = tile_vol * sizeof(T);
            gpu::block::set_shmem_size( kernel::x_shift_left<T>, shm_size );
            kernel::x_shift_left <<< grid, 1024, shm_size >>> ( 
                view(), shift
            );

            // Copy x guard cells
            copy_to_gc_x();
        } else {
            mpi::fatal( "x_shift_left(), invalid shift value, must be 0 < shift <= gc.x.upper" );
        }
    }

    /**
     * @brief Perform a convolution with a 3 point kernel [a,b,c] along x
     * 
     * @note Guard cells must be up to date on entry. Guard cells are updated
     *       on exit.
     * 
     * @param a             Kernel value a
     * @param b             Kernel value b
     * @param c             Kernel value c
     * @param upper_ext     (optional) Also compute the result on the first
     *                      upper x guard column and first upper y guard row,
     *                      see kernel::kernel3_x(). Requires gc.x.upper >= 2
     *                      and gc.y.upper >= 1
     */
    template < typename S >
    void kernel3_x( S const a, S const b, S const c, bool const upper_ext = false ) {

        if (( gc.x.lower > 0) && (gc.x.upper > 0)) {

            if ( upper_ext && ( gc.x.upper < 2 || gc.y.upper < 1 ) ) {
                mpi::fatal( "kernel3_x() with upper_ext requires at least 2 upper x guard cells"
                            " and 1 upper y guard cell." );
            }

            dim3 grid( local_ntiles.x, local_ntiles.y );

            size_t shm_size = tile_vol * sizeof(T);

            if ( shm_size > gpu::block::shared_mem_size() ) {
                mpi::fatal("grid::kernel3_x(), tile size too large, insufficient shared memory");
            }

            gpu::block::set_shmem_size( kernel::kernel3_x<T,S>, shm_size );
            kernel::kernel3_x <<< grid, 1024, shm_size >>> ( 
                view(), a, b, c, upper_ext ? 1 : 0
            ); 

            // Update guard cells
            copy_to_gc();

        } else {
            mpi::fatal( "kernel3_x() requires at least 1 guard cell at both the lower and upper x boundaries." );
        }

    }

    /**
     * @brief Perform a convolution with a 3 point kernel [a,b,c] along y
     * 
     * @note Guard cells must be up to date on entry. Guard cells are updated
     *       on exit.
     * 
     * @param a             Kernel value a
     * @param b             Kernel value b
     * @param c             Kernel value c
     * @param upper_ext     (optional) Also compute the result on the first
     *                      upper x guard column and first upper y guard row,
     *                      see kernel::kernel3_y(). Requires gc.y.upper >= 2
     *                      and gc.x.upper >= 1
     */
    template < typename S >
    void kernel3_y( S const a, S const b, S const c, bool const upper_ext = false ) {

        if (( gc.y.lower > 0) && (gc.y.upper > 0)) {

            if ( upper_ext && ( gc.y.upper < 2 || gc.x.upper < 1 ) ) {
                mpi::fatal( "kernel3_y() with upper_ext requires at least 2 upper y guard cells"
                            " and 1 upper x guard cell." );
            }

            dim3 grid( local_ntiles.x, local_ntiles.y );

            size_t shm_size = tile_vol * sizeof(T);

            if ( shm_size > gpu::block::shared_mem_size() ) {
                mpi::fatal("grid::kernel3_y(), tile size too large, insufficient shared memory");
            }

            gpu::block::set_shmem_size( kernel::kernel3_y<T,S>, shm_size );
            kernel::kernel3_y <<< grid, 1024, shm_size >>> ( 
                view(), a, b, c, upper_ext ? 1 : 0
            ); 

            // Update guard cells
            copy_to_gc();

        } else {
            mpi::fatal( "kernel3_y() requires at least 1 guard cell at both the lower and upper y boundaries." );
        }

    }

    /**
     * @brief Perform a separable 2D convolution with a 3 point kernel [a,b,c]
     *        along x and y
     * 
     * @note Equivalent to kernel3_x( a, b, c ) followed by kernel3_y( a, b, c ),
     *       i.e. a 3x3 kernel [a,b,c] ⊗ [a,b,c], but with a single kernel
     *       launch and a single guard cell update.
     * 
     * @note Guard cells (including corners) must be up to date on entry, as
     *       with kernel3_x() / kernel3_y(). Guard cells are updated on exit.
     * 
     * @param a             Kernel value a
     * @param b             Kernel value b
     * @param c             Kernel value c
     * @param upper_ext     (optional) Also compute the result on the first
     *                      upper x guard column and first upper y guard row,
     *                      see kernel::kernel3_xy(). Requires gc.x.upper >= 2
     *                      and gc.y.upper >= 2
     */
    template < typename S >
    void kernel3_xy( S const a, S const b, S const c, bool const upper_ext = false ) {

        if (( gc.x.lower > 0) && (gc.x.upper > 0) && 
            ( gc.y.lower > 0) && (gc.y.upper > 0)) {

            if ( upper_ext && ( gc.x.upper < 2 || gc.y.upper < 2 ) ) {
                mpi::fatal( "kernel3_xy() with upper_ext requires at least 2 upper x and y guard cells." );
            }

            dim3 grid( local_ntiles.x, local_ntiles.y );

            size_t shm_size = 2 * tile_vol * sizeof(T);

            if ( shm_size > gpu::block::shared_mem_size() ) {
                mpi::fatal("grid::kernel3_xy(), tile size too large, insufficient shared memory");
            }

            gpu::block::set_shmem_size( kernel::kernel3_xy<T,S>, shm_size );
            kernel::kernel3_xy <<< grid, 1024, shm_size >>> ( 
                view(), a, b, c, upper_ext ? 1 : 0
            ); 

            // Update guard cells
            copy_to_gc();

        } else {
            mpi::fatal( "kernel3_xy() requires at least 1 guard cell at all x and y boundaries." );
        }

    }
    
    /**
     * @brief Save grid values to disk with full metadata
     *
     * @note Data can be converted to a different datatype
     * 
     * @tparam T2       Datatype to be used for output, defaults to the grid 
     *                  datatype. Must be supported by the ZDF library.
     * @param info      Grid metadata
     * @param iter      Iteration value
     * @param path      File path
     */
    template< typename T2 = T >
    void save( zdf::grid_info &info, const zdf::iteration &iter, const std::string & path ) {

        // Fill in global grid dimensions
        info.ndims = 2;
        info.count[0] = global_ntiles.x * tile_dims.x;
        info.count[1] = global_ntiles.y * tile_dims.y;

        // Allocate buffers on host and device to gather data
        const std::size_t bsize = local_dims.x * local_dims.y;
        T2 * h_data = gpu::host::malloc<T2>( bsize );
        T2 * d_data = gpu::device::malloc<T2>( bsize );

        // Gather data on contiguous grid
        gather( d_data );

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
        zdf::save_grid<T2>( chunk, info, iter, path, part.get_comm() );

        // Free temporary buffer
        gpu::host::free( h_data );
    };

    /**
     * @brief Save grid values to disk
     *
     * @note Data can be converted to a different datatype
     * 
     * @tparam T2       Datatype to be used for output, defaults to the grid 
     *                  datatype. Must be supported by the ZDF library.
     * @param filename  Output file name
     */
    template< typename T2 = T >
    void save( const std::string & filename ) {
        // Allocate buffers on host and device to gather data
        const std::size_t bsize = local_dims.x * local_dims.y;
        T2 * h_data = gpu::host::malloc<T2>( bsize );
        T2 * d_data = gpu::device::malloc<T2>( bsize );

        // Gather data on contiguous grid
        gather( d_data );

        // Copy to host and free device memory
        gpu::device::memcpy_tohost( h_data, d_data, bsize );
        gpu::device::free( d_data );

        uint64_t global[2] = { global_ntiles.x * tile_dims.x, global_ntiles.y * tile_dims.y };
        uint64_t start[2]  = { local_tile_start.x * tile_dims.x, local_tile_start.y * tile_dims.y };
        uint64_t local[2]  = { local_dims.x, local_dims.y };

        zdf::save_grid( h_data, 2, global, start, local, name, filename, part.get_comm() );

        // Free temporary buffer
        gpu::host::free( h_data );
    }

};

} // end of namespace grid
