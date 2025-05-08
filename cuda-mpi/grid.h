#ifndef GRID_H_
#define GRID_H_

#include "parallel.h"

#include "vec_types.h"
#include "bnd.h"
#include "zdf-cpp.h"

#include <iostream>

/**
 * @brief CUDA Kernels for the various functions
 * 
 * @note This namespace is kept anonymous so these functions can only be
 *       accessed from within this file
 */
namespace {

/**
 * @brief CUDA kernel for adding two grid objects
 * 
 * @tparam T 
 * @param a     Output grid
 * @param b     Input grid
 * @param size  Grid size
 */
template< typename T >
__global__
void add_kernel(T * __restrict__ a, T const * __restrict__ b, size_t const size ) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < size ) {
        a[idx] += b[idx];
    }
}

/**
 * @brief CUDA kernel for gather operation
 * 
 * @tparam T        Data type
 * @param d_out     Output buffer
 * @param d_buffer  Input buffer (includes offset to cell [0,0])
 * @param ntiles    Number of tiles
 * @param nx        Tile size
 * @param ext_nx    Tile size including guard cells
 */
template< typename T >
__global__
void gather_kernel( 
    T * const __restrict__ d_out, T const * const __restrict__ d_buffer,
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

        d_out[ out_idx ] = tile_data[ iy * ext_nx.x + ix ];
    }
}

template< typename T >
__global__
void copy_to_gc_x_kernel( T * const __restrict__ d_buffer,
    const uint2 ntiles, const uint2 nx, const uint2 ext_nx, 
    const int periodic_x, const int gc_x_lower, const int gc_x_upper )
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    auto * __restrict__ local = & d_buffer[ tile_off ];

    {   // Copy from lower neighbour
        int neighbor_tx = tile_idx.x - 1;
        if ( periodic_x && neighbor_tx < 0 )
            neighbor_tx += ntiles.x;

        if ( neighbor_tx >= 0 ) {
            auto * __restrict__ x_lower = d_buffer + (tile_idx.y * ntiles.x + neighbor_tx) * tile_vol;
            for( unsigned idx = block_thread_rank(); idx < ext_nx.y * gc_x_lower; idx += block_num_threads() ) {
                const auto i = idx % gc_x_lower;
                const auto j = idx / gc_x_lower; 
                local[ i + j * ext_nx.x ] = x_lower[ nx.x + i + j * ext_nx.x ];
            }
        }
    }

    {   // Copy from upper neighbour
        int neighbor_tx = tile_idx.x + 1;
        if ( periodic_x && neighbor_tx >= static_cast<int>(ntiles.x) )
            neighbor_tx -= ntiles.x;

        if ( neighbor_tx < static_cast<int>(ntiles.x) ) {
            auto * __restrict__ x_upper = d_buffer + (tile_idx.y * ntiles.x + neighbor_tx) * tile_vol;
            for( unsigned idx = block_thread_rank(); idx < ext_nx.y * gc_x_upper; idx += block_num_threads() ) {
                const auto i = idx % gc_x_upper;
                const auto j = idx / gc_x_upper; 
                local[ gc_x_lower + nx.x + i + j * ext_nx.x ] = x_upper[ gc_x_lower + i + j * ext_nx.x ];
            }
        }
    }
}

template< typename T >
__global__
void copy_to_gc_y_kernel( T * const __restrict__ d_buffer,
    const uint2 ntiles, const uint2 nx, const uint2 ext_nx, 
    const int periodic_y, const int gc_y_lower, const int gc_y_upper )
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    auto * __restrict__ local = & d_buffer[ tile_off ];

    {   // Copy from lower neighbour
        int neighbor_ty = tile_idx.y - 1;
        if ( periodic_y && neighbor_ty < 0 )
            neighbor_ty += ntiles.y;

        if ( neighbor_ty >= 0 ) {
            auto * __restrict__ y_lower = d_buffer + (neighbor_ty * ntiles.x + tile_idx.x) * tile_vol;
            for( unsigned idx = block_thread_rank(); idx < gc_y_lower * ext_nx.x; idx += block_num_threads() ) {
                const auto i = idx % ext_nx.x;
                const auto j = idx / ext_nx.x; 
                local[ i + j * ext_nx.x ] = y_lower[ i + ( nx.y + j ) * ext_nx.x ];
            }
        }
    }

    {   // Copy from upper neighbour
        int neighbor_ty = tile_idx.y + 1;
        if ( periodic_y && neighbor_ty >= static_cast<int>(ntiles.y) )
            neighbor_ty -= ntiles.y;

        if ( neighbor_ty < static_cast<int>(ntiles.y) ) {
            auto * __restrict__ y_upper = d_buffer + (neighbor_ty * ntiles.x + tile_idx.x) * tile_vol;
            for( unsigned idx = block_thread_rank(); idx < gc_y_upper * ext_nx.x; idx += block_num_threads() ) {
                const auto i = idx % ext_nx.x;
                const auto j = idx / ext_nx.x; 
                local[ i + ( gc_y_lower + nx.y + j ) * ext_nx.x ] = y_upper[ i + ( gc_y_lower + j ) * ext_nx.x ];
            }
        }
    }
}

template< typename T >
__global__
void add_from_gc_x_kernel( T * const __restrict__ d_buffer,
    const uint2 ntiles, const uint2 nx, const uint2 ext_nx, 
    const int periodic_x, const int gc_x_lower, const int gc_x_upper )
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    auto * __restrict__ local = & d_buffer[ tile_off ];

    {   // Add from lower neighbour
        int neighbor_tx = tile_idx.x - 1;
        if ( periodic_x && neighbor_tx < 0 )
            neighbor_tx += ntiles.x;

        if ( neighbor_tx >= 0 ) {
            T * __restrict__ x_lower = d_buffer + (tile_idx.y * ntiles.x + neighbor_tx) * tile_vol;
            for( unsigned idx = block_thread_rank(); idx < ext_nx.y * gc_x_upper; idx += block_num_threads() ) {
                const auto i = idx % gc_x_upper;
                const auto j = idx / gc_x_upper; 
                local[ gc_x_lower + i + j * ext_nx.x ] += x_lower[ gc_x_lower + nx.x + i + j * ext_nx.x ];
            }
        }
    }

    {   // Add from upper neighbour
        int neighbor_tx = tile_idx.x + 1;
        if ( periodic_x && neighbor_tx >= static_cast<int>(ntiles.x) )
            neighbor_tx -= ntiles.x;

        if ( neighbor_tx < static_cast<int>(ntiles.x) ) {
            auto * __restrict__ x_upper = d_buffer + (tile_idx.y * ntiles.x + neighbor_tx) * tile_vol;
            for( unsigned idx = block_thread_rank(); idx < ext_nx.y * gc_x_lower; idx += block_num_threads() ) {
                const auto i = idx % gc_x_lower;
                const auto j = idx / gc_x_lower; 
                local[ nx.x + i + j * ext_nx.x ] += x_upper[ i + j * ext_nx.x ];
            }
        }
    }
}

template< typename T >
__global__
void add_from_gc_y_kernel( T * const __restrict__ d_buffer,
    const uint2 ntiles, const uint2 nx, const uint2 ext_nx, 
    const int periodic_y, const int gc_y_lower, const int gc_y_upper )
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    auto * __restrict__ local = & d_buffer[ tile_off ];

    {   // Add from lower neighbour
        int neighbor_ty = tile_idx.y - 1;
        if ( periodic_y && neighbor_ty < 0 )
            neighbor_ty += ntiles.y;

        if ( neighbor_ty >= 0 ) {
            auto * __restrict__ y_lower = d_buffer + (neighbor_ty * ntiles.x + tile_idx.x) * tile_vol;
            for( unsigned idx = block_thread_rank(); idx < gc_y_upper * ext_nx.x; idx += block_num_threads() ) {
                const auto i = idx % ext_nx.x;
                const auto j = idx / ext_nx.x; 
                local[ i + ( gc_y_lower + j ) * ext_nx.x ] += y_lower[ i + ( gc_y_lower + nx.y + j ) * ext_nx.x ];
            }
        }
    }

    {   // Add from upper neighbour
        int neighbor_ty = tile_idx.y + 1;
        if ( periodic_y && neighbor_ty >= static_cast<int>(ntiles.y) ) neighbor_ty -= ntiles.y;

        if ( neighbor_ty < static_cast<int>(ntiles.y) ) {
            auto * __restrict__ y_upper = d_buffer + (neighbor_ty * ntiles.x + tile_idx.x) * tile_vol;
            for( unsigned idx = block_thread_rank(); idx < gc_y_lower * ext_nx.x; idx += block_num_threads() ) {
                const auto i = idx % ext_nx.x;
                const auto j = idx / ext_nx.x; 
                local[ i + ( nx.y + j ) * ext_nx.x ] += y_upper[ i + j * ext_nx.x ];
            }
        }
    }

}

template< typename T >
__global__
void x_shift_left_kernel( T * const __restrict__ d_buffer, const uint2 ntiles, const uint2 ext_nx, const unsigned shift )
{

    auto * local = block::shared_mem<T>();

    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    const int ystride = ext_nx.x;

    auto * __restrict__ buffer = & d_buffer[ tile_off ];

    for( int idx = block_thread_rank(); idx < tile_vol; idx += block_num_threads() ) {
        const auto i = idx % ext_nx.x;
        const auto j = idx / ext_nx.x;
        if ( i < ext_nx.x - shift ) {
            local[ i + j * ystride ] = buffer[ (i + shift) + j * ystride ];
        } else {
            local[ i + j * ystride ] = {0};
        }
    }

    block_sync();

    for( int idx = block_thread_rank(); idx < tile_vol; idx += block_num_threads() )
        buffer[idx] = local[idx];
}

template< typename T, typename S >
__global__
void kernel3_x_kernel( T * const __restrict__ d_buffer, const uint2 ntiles,
    const uint2 nx, const uint2 ext_nx, const int gc_x_lower, 
    S const a, S const b, S const c )
{
    auto * shm = block::shared_mem<T>();

    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;
    const int    ystride  = ext_nx.x;

    T * __restrict__ buffer = & d_buffer[ tile_off ];
    T * __restrict__ A = & shm[0];
    T * __restrict__ B = & shm[tile_vol];

    // Copy data from tile buffer
    for( int i = block_thread_rank(); i < tile_vol; i += block_num_threads() )
        A[i] = buffer[i];

    // Synchronize 
    block_sync();

    // Apply kernel locally
    for( int idx = block_thread_rank(); idx < ext_nx.y * nx.x; idx += block_num_threads() ) {
        const auto iy = idx / nx.x;
        const auto ix = idx % nx.x + gc_x_lower;
        B [ iy * ystride + ix ] = A[ iy * ystride + (ix-1) ] * a +
                                  A[ iy * ystride +  ix    ] * b +
                                  A[ iy * ystride + (ix+1) ] * c;
    }

    // Synchronize 
    block_sync();

    // Copy data back to tile buffer
    for( int i = block_thread_rank(); i < tile_vol; i += block_num_threads() )
        buffer[i] = B[i];
}


template< typename T, typename S >
__global__
void kernel3_y_kernel( T * const __restrict__ d_buffer, const uint2 ntiles,
    const uint2 nx, const uint2 ext_nx, const int gc_y_lower, 
    S const a, S const b, S const c )
{
    auto * shm = block::shared_mem<T>();

    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;
    const int    ystride  = ext_nx.x;

    T * __restrict__ buffer = & d_buffer[ tile_off ];
    T * __restrict__ A = & shm[0];
    T * __restrict__ B = & shm[tile_vol];

    // Copy data from tile buffer
    for( int i = block_thread_rank(); i < tile_vol; i += block_num_threads() )
        A[i] = buffer[i];

    // Synchronize 
    block_sync();

    // Apply kernel locally
    for( int idx = block_thread_rank(); idx < nx.y * ext_nx.x; idx += block_num_threads() ) {
        const auto iy = idx / ext_nx.x + gc_y_lower;
        const auto ix = idx % ext_nx.x;

        B [ iy * ystride + ix ] = A[ (iy-1) * ystride + ix ] * a +
                                  A[    iy  * ystride + ix ] * b +
                                  A[ (iy+1) * ystride + ix ] * c;
    }

    // Synchronize 
    block_sync();

    // Copy data back to tile buffer
    for( int i = block_thread_rank(); i < tile_vol; i += block_num_threads() )
        buffer[i] = B[i];
}

/**
 * @brief CUDA kernel for packing x boundaries copy_to send messages
 * 
 * @note Must be called with grid( 2, ntiles.y )
 * 
 * @tparam T            Data type
 * @param lower         Lower neighbor send message buffer
 * @param upper         Upper neighbor send message buffer
 * @param d_buffer      Grid data
 * @param ntiles        Number of local tiles (x,y)
 * @param nx            Tile size
 * @param ext_nx        External tile size
 * @param gc_x_lower    Number of guard cells in x lower boundary
 * @param gc_x_upper    Number of guard cells in x upper boundary
 */
template< typename T >
__global__
void kernel_copy_to_gc_x_send( 
    T * const __restrict__ lower, T * const __restrict__ upper,
    T * const __restrict__ d_buffer, const uint2 ntiles, 
    const uint2 nx, const uint2 ext_nx, 
    const int gc_x_lower, const int gc_x_upper ) {

    unsigned int tx = ( blockIdx.x == 0 ) ? 0 : ntiles.x - 1;    
    unsigned int ty = blockIdx.y;

    const auto tid = ty * ntiles.x + tx;
    const auto tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const auto tile_off = tid * tile_vol;
    auto * __restrict__ local = & d_buffer[ tile_off ];
    
    // Note that blockIdx.x is constant inside a block so there is no thread divergence
    if ( blockIdx.x == 0 ) {
        // lower message
        if ( lower != nullptr ) {
            T * __restrict__ msg = & lower [ ty * ext_nx.y * gc_x_upper ];
            for( int idx = block_thread_rank(); idx < ext_nx.y * gc_x_upper; idx += block_num_threads() ) {
                const int i = idx % gc_x_upper;
                const int j = idx / gc_x_upper;
                msg[ j * gc_x_upper + i ] = local[ j * ext_nx.x + gc_x_lower + i ];
            }
        }
    } else {
        // upper message
        if ( upper != nullptr ) {
            T * __restrict__ msg = & upper[ ty * ext_nx.y * gc_x_lower ];
            for( int idx = block_thread_rank(); idx < ext_nx.y * gc_x_lower; idx += block_num_threads() ) {
                const int i = idx % gc_x_lower;
                const int j = idx / gc_x_lower;
                msg[ j * gc_x_lower + i ] = local[ j * ext_nx.x + nx.x + i ];
            }
        }
    }
}

/**
 * @brief CUDA kernel for unpacking x boundaries copy_to receive messages
 * 
 * @note Must be called with grid( 2, ntiles.y )
 * 
 * @tparam T            Data type
 * @param lower         Lower neighbor receive message buffer
 * @param upper         Upper neighbor receive message buffer
 * @param d_buffer      Grid data
 * @param ntiles        Number of local tiles (x,y)
 * @param nx            Tile size
 * @param ext_nx        External tile size
 * @param gc_x_lower    Number of guard cells in x lower boundary
 * @param gc_x_upper    Number of guard cells in x upper boundary
 */
template< typename T >
__global__
void kernel_copy_to_gc_x_recv( 
    T * const __restrict__ lower, T * const __restrict__ upper,
    T * const __restrict__ d_buffer, const uint2 ntiles, 
    const uint2 nx, const uint2 ext_nx,
    const int gc_x_lower, const int gc_x_upper ) {

    unsigned int tx = ( blockIdx.x == 0 ) ? 0 : ntiles.x - 1;    
    unsigned int ty = blockIdx.y;

    const auto tid = ty * ntiles.x + tx;
    const auto tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const auto tile_off = tid * tile_vol;
    auto * __restrict__ local = & d_buffer[ tile_off ];
    
    // Note that blockIdx.x is constant inside a block so there is no thread divergence
    if ( blockIdx.x == 0 ) {
        // lower message
        if ( lower != nullptr ) {
            T * __restrict__ msg = & lower [ ty * ext_nx.y * gc_x_lower ];
            for( int idx = block_thread_rank(); idx < ext_nx.y * gc_x_lower; idx += block_num_threads() ) {
                const int i = idx % gc_x_lower;
                const int j = idx / gc_x_lower;
                local[ j * ext_nx.x + i ] = msg[ j * gc_x_lower + i ];
            }
        }
    } else {
        // upper message
        if ( upper != nullptr ) {
            T * __restrict__ msg = & upper[ ty * ext_nx.y * gc_x_upper ];
            for( int idx = block_thread_rank(); idx < ext_nx.y * gc_x_upper; idx += block_num_threads() ) {
                const int i = idx % gc_x_upper;
                const int j = idx / gc_x_upper;
                local[ j * ext_nx.x + gc_x_lower + nx.x + i ] = msg[ j * gc_x_upper + i ];
            }
        }
    }
}

/**
 * @brief CUDA kernel for packing y boundaries copy_to send messages
 * 
 * @note Must be called with grid( ntiles.x, 2 )
 * 
 * @tparam T            Data type
 * @param lower         Lower neighbor send message buffer
 * @param upper         Upper neighbor send message buffer
 * @param d_buffer      Grid data
 * @param ntiles        Number of local tiles (x,y)
 * @param nx            Tile size
 * @param ext_nx        External tile size
 * @param gc_y_lower    Number of guard cells in y lower boundary
 * @param gc_y_upper    Number of guard cells in y upper boundary
 */
template< typename T >
__global__
void kernel_copy_to_gc_y_send( 
    T * const __restrict__ lower, T * const __restrict__ upper,
    T * const __restrict__ d_buffer, const uint2 ntiles, 
    const uint2 nx, const uint2 ext_nx, 
    const int gc_y_lower, const int gc_y_upper ) {

    unsigned int tx = blockIdx.x;
    unsigned int ty = ( blockIdx.y == 0 ) ? 0 : ntiles.y - 1;    

    const auto tid = ty * ntiles.x + tx;
    const auto tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const auto tile_off = tid * tile_vol;
    auto * __restrict__ local = & d_buffer[ tile_off ];
    
    // Note that blockIdx.y is constant inside a block so there is no thread divergence
    if ( blockIdx.y == 0 ) {
        // lower message
        if ( lower != nullptr ) {
            T * __restrict__ msg = & lower [ tx * ext_nx.x * gc_y_upper ];
            for( int idx = block_thread_rank(); idx < gc_y_upper * ext_nx.x; idx += block_num_threads() ) {
                const int i = idx % ext_nx.x;
                const int j = idx / ext_nx.x;
                msg[ j * ext_nx.x + i ] = local[ ( gc_y_lower + j ) * ext_nx.x + i ];
            }
        }
    } else {
        // upper message
        if ( upper != nullptr ) {
            T * __restrict__ msg = & upper[ tx * ext_nx.x * gc_y_lower ];
            for( int idx = block_thread_rank(); idx < gc_y_lower * ext_nx.x; idx += block_num_threads() ) {
                const int i = idx % ext_nx.x;
                const int j = idx / ext_nx.x;
                msg[ j * ext_nx.x + i ] = local[ ( nx.y + j ) * ext_nx.x + i ];
            }
        }
    }
}

/**
 * @brief CUDA kernel for unpacking y boundaries copy_to receive messages
 * 
 * @note Must be called with grid( ntiles.x, 2 )
 * 
 * @tparam T            Data type
 * @param lower         Lower neighbor receive message buffer
 * @param upper         Upper neighbor receive message buffer
 * @param d_buffer      Grid data
 * @param ntiles        Number of local tiles (x,y)
 * @param nx            Tile size
 * @param ext_nx        External tile size
 * @param gc_y_lower    Number of guard cells in y lower boundary
 * @param gc_y_upper    Number of guard cells in y upper boundary
 */
template< typename T >
__global__
void kernel_copy_to_gc_y_recv( 
    T * const __restrict__ lower, T * const __restrict__ upper,
    T * const __restrict__ d_buffer, const uint2 ntiles, 
    const uint2 nx, const uint2 ext_nx, 
    const int gc_y_lower, const int gc_y_upper ) {

    unsigned int tx = blockIdx.x;
    unsigned int ty = ( blockIdx.y == 0 ) ? 0 : ntiles.y - 1;    

    const auto tid = ty * ntiles.x + tx;
    const auto tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const auto tile_off = tid * tile_vol;
    auto * __restrict__ local = & d_buffer[ tile_off ];
    
    // Note that blockIdx.y is constant inside a block so there is no thread divergence
    if ( blockIdx.y == 0 ) {
        // lower message
        if ( lower != nullptr ) {
            T * __restrict__ msg = & lower [ tx * ext_nx.x * gc_y_lower ];
            for( int idx = block_thread_rank(); idx < gc_y_lower * ext_nx.x; idx += block_num_threads() ) {
                const int i = idx % ext_nx.x;
                const int j = idx / ext_nx.x;
                local[ j * ext_nx.x + i ] =  msg[ j * ext_nx.x + i ];
            }
        }
    } else {
        // upper message
        if ( upper != nullptr ) {
            T * __restrict__ msg = & upper[ tx * ext_nx.x * gc_y_upper ];
            for( int idx = block_thread_rank(); idx < gc_y_upper * ext_nx.x; idx += block_num_threads() ) {
                const int i = idx % ext_nx.x;
                const int j = idx / ext_nx.x;
                local[ ( gc_y_lower + nx.y + j ) * ext_nx.x + i ] =  msg[ j * ext_nx.x + i ];
            }
        }
    }
}

/**
 * @brief CUDA kernel for packing x boundaries add_from send messages
 * 
 * @note Must be called with grid( 2, ntiles.y )
 * 
 * @tparam T            Data type
 * @param lower         Lower neighbor send message buffer
 * @param upper         Upper neighbor send message buffer
 * @param d_buffer      Grid data
 * @param ntiles        Number of local tiles (x,y)
 * @param nx            Tile size
 * @param ext_nx        External tile size
 * @param gc_x_lower    Number of guard cells in x lower boundary
 * @param gc_x_upper    Number of guard cells in x upper boundary
 */
template< typename T >
__global__
void kernel_add_from_gc_x_send( 
    T * const __restrict__ lower, T * const __restrict__ upper,
    T * const __restrict__ d_buffer, const uint2 ntiles, 
    const uint2 nx, const uint2 ext_nx, 
    const int gc_x_lower, const int gc_x_upper ) {

    unsigned int tx = ( blockIdx.x == 0 ) ? 0 : ntiles.x - 1;    
    unsigned int ty = blockIdx.y;

    const auto tid = ty * ntiles.x + tx;
    const auto tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const auto tile_off = tid * tile_vol;
    auto * __restrict__ local = & d_buffer[ tile_off ];
    
    // Note that blockIdx.x is constant inside a block so there is no thread divergence
    if ( blockIdx.x == 0 ) {
        // lower message
        if ( lower != nullptr ) {
            T * __restrict__ msg = & lower [ ty * ext_nx.y * gc_x_lower ];
            for( int idx = block_thread_rank(); idx < ext_nx.y * gc_x_lower; idx += block_num_threads() ) {
                const int i = idx % gc_x_lower;
                const int j = idx / gc_x_lower;
                msg[ j * gc_x_lower + i ] = local[ j * ext_nx.x + i ];
            }
        }
    } else {
        // upper message
        if ( upper != nullptr ) {
            T * __restrict__ msg = & upper[ ty * ext_nx.y * gc_x_upper ];
            for( int idx = block_thread_rank(); idx < ext_nx.y * gc_x_upper; idx += block_num_threads() ) {
                const int i = idx % gc_x_upper;
                const int j = idx / gc_x_upper;
                msg[ j * gc_x_upper + i ] = local[ j * ext_nx.x + gc_x_lower + nx.x + i ];
            }
        }
    }
}


/**
 * @brief CUDA kernel for unpacking x boundaries add_from receive messages
 * 
 * @note Must be called with grid( 2, ntiles.y )
 * 
 * @tparam T            Data type
 * @param lower         Lower neighbor receive message buffer
 * @param upper         Upper neighbor receive message buffer
 * @param d_buffer      Grid data
 * @param ntiles        Number of local tiles (x,y)
 * @param nx            Tile size
 * @param ext_nx        External tile size
 * @param gc_x_lower    Number of guard cells in x lower boundary
 * @param gc_x_upper    Number of guard cells in x upper boundary
 */
template< typename T >
__global__
void kernel_add_from_gc_x_recv( 
    T * const __restrict__ lower, T * const __restrict__ upper,
    T * const __restrict__ d_buffer, const uint2 ntiles, 
    const uint2 nx, const uint2 ext_nx, 
    const int gc_x_lower, const int gc_x_upper ) {

    unsigned int tx = ( blockIdx.x == 0 ) ? 0 : ntiles.x - 1;    
    unsigned int ty = blockIdx.y;

    const auto tid = ty * ntiles.x + tx;
    const auto tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const auto tile_off = tid * tile_vol;
    auto * __restrict__ local = & d_buffer[ tile_off ];
    
    // Note that blockIdx.x is constant inside a block so there is no thread divergence
    if ( blockIdx.x == 0 ) {
        // lower message
        if ( lower != nullptr ) {
            T * __restrict__ msg = & lower [ ty * ext_nx.y * gc_x_upper ];
            for( int idx = block_thread_rank(); idx < ext_nx.y * gc_x_upper; idx += block_num_threads() ) {
                const int i = idx % gc_x_upper;
                const int j = idx / gc_x_upper;
                local[ j * ext_nx.x + gc_x_lower + i ] += msg[ j * gc_x_upper + i ] ;
            }
        }
    } else {
        // upper message
        if ( upper != nullptr ) {
            T * __restrict__ msg = & upper[ ty * ext_nx.y * gc_x_lower ];
            for( int idx = block_thread_rank(); idx < ext_nx.y * gc_x_lower; idx += block_num_threads() ) {
                const int i = idx % gc_x_lower;
                const int j = idx / gc_x_lower;
                local[ j * ext_nx.x + nx.x + i ] += msg[ j * gc_x_lower + i ] ;
            }
        }
    }
}

/**
 * @brief CUDA kernel for packing x boundaries add_from send messages
 * 
 * @note Must be called with grid( ntiles.x, 2 )
 * 
 * @tparam T            Data type
 * @param lower         Lower neighbor send message buffer
 * @param upper         Upper neighbor send message buffer
 * @param d_buffer      Grid data
 * @param ntiles        Number of local tiles (x,y)
 * @param nx            Tile size
 * @param ext_nx        External tile size
 * @param gc_y_lower    Number of guard cells in y lower boundary
 * @param gc_y_upper    Number of guard cells in y upper boundary
 */
template< typename T >
__global__
void kernel_add_from_gc_y_send( 
    T * const __restrict__ lower, T * const __restrict__ upper,
    T * const __restrict__ d_buffer, const uint2 ntiles, 
    const uint2 nx, const uint2 ext_nx, 
    const int gc_y_lower, const int gc_y_upper ) {

    unsigned int tx = blockIdx.x;
    unsigned int ty = ( blockIdx.y == 0 ) ? 0 : ntiles.y - 1;    

    const auto tid = ty * ntiles.x + tx;
    const auto tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const auto tile_off = tid * tile_vol;
    auto * __restrict__ local = & d_buffer[ tile_off ];
    
    // Note that blockIdx.y is constant inside a block so there is no thread divergence
    if ( blockIdx.y == 0 ) {
        // lower message
        if ( lower != nullptr ) {
            T * __restrict__ msg = & lower [ tx * ( ext_nx.x * gc_y_lower ) ];
            for( int idx = block_thread_rank(); idx < gc_y_lower * ext_nx.x; idx += block_num_threads() ) {
                const int i = idx % ext_nx.x;
                const int j = idx / ext_nx.x;
                msg[ j * ext_nx.x + i ] = local[ j * ext_nx.x + i ];
            }
        }
    } else {
        // upper message
        if ( upper != nullptr ) {
            T * __restrict__ msg = & upper[ tx * gc_y_upper * ext_nx.x ];
            for( int idx = block_thread_rank(); idx < gc_y_upper * ext_nx.x; idx += block_num_threads() ) {
                const int i = idx % ext_nx.x;
                const int j = idx / ext_nx.x;
                msg[ j * ext_nx.x + i ] = local[ ( gc_y_lower + nx.y + j ) * ext_nx.x + i ];
            }
        }
    }
}

/**
 * @brief CUDA kernel for unpacking y boundaries add_from send messages
 * 
 * @note Must be called with grid( ntiles.x, 2 )
 * 
 * @tparam T            Data type
 * @param lower         Lower neighbor send message buffer
 * @param upper         Upper neighbor send message buffer
 * @param d_buffer      Grid data
 * @param ntiles        Number of local tiles (x,y)
 * @param nx            Tile size
 * @param ext_nx        External tile size
 * @param gc_y_lower    Number of guard cells in y lower boundary
 * @param gc_y_upper    Number of guard cells in y upper boundary
 */
template< typename T >
__global__
void kernel_add_from_gc_y_recv( 
    T * const __restrict__ lower, T * const __restrict__ upper,
    T * const __restrict__ d_buffer, const uint2 ntiles, 
    const uint2 nx, const uint2 ext_nx, 
    const int gc_y_lower, const int gc_y_upper ) {

    unsigned int tx = blockIdx.x;
    unsigned int ty = ( blockIdx.y == 0 ) ? 0 : ntiles.y - 1;    

    const auto tid = ty * ntiles.x + tx;
    const auto tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const auto tile_off = tid * tile_vol;
    auto * __restrict__ local = & d_buffer[ tile_off ];
    
    // Note that blockIdx.y is constant inside a block so there is no thread divergence
    if ( blockIdx.y == 0 ) {
        // lower message
        if ( lower != nullptr ) {
            T * __restrict__ msg = & lower [ tx * gc_y_upper * ext_nx.x ];
            for( int idx = block_thread_rank(); idx < gc_y_upper * ext_nx.x; idx += block_num_threads() ) {
                const int i = idx % ext_nx.x;
                const int j = idx / ext_nx.x;
                local[ ( gc_y_lower + j ) * ext_nx.x + i ] += msg[ j * ext_nx.x + i ];
            }
        }
    } else {
        // upper message
        if ( upper != nullptr ) {
            T * __restrict__ msg = & upper[ tx * (gc_y_lower * ext_nx.x) ];
            for( int idx = block_thread_rank(); idx < gc_y_lower * ext_nx.x; idx += block_num_threads() ) {
                const int i = idx % ext_nx.x;
                const int j = idx / ext_nx.x;
                local[ ( nx.y + j ) * ext_nx.x + i ] +=  msg[ j * ext_nx.x + i ];
            }
        }
    }
}


/**
 * @brief End of CUDA kernels namespace
 * 
 */
}

namespace source {
    enum tag { lower = 0, upper = 1 };
}

namespace dest {
    enum tag { upper = 0, lower = 1 };
}


/**
 * @brief 
 * 
 */
template <class T>
class grid {
    protected:

    /// @brief Local number of tiles
    uint2 ntiles;

    /// @brief Offset of local tiles in global tile grid
    uint2 tile_off;

    /// @brief Consider local boundaries periodic
    int2 periodic;

    /// @brief Local grid size
    uint2 local_nx;

    /// @brief Buffers for sending messages
    pair< Message<T>* > msg_send;

    /// @brief Buffers for receiving messages
    pair< Message<T>* > msg_recv;

    /**
     * @brief Set the local node information. This information will (may) be
     * different for each parallel node
     * 
     * @note Global periodic information is taken from the parallel partition
     * 
     */
    void initialize( ) {

        // Get local number of tiles and position
        part.grid_local( global_ntiles, ntiles, tile_off );

        // Get local grid size
        local_nx = ntiles * nx;

        // Get local periodic flag
        periodic.x = part.periodic.x && (part.dims.x == 1);
        periodic.y = part.periodic.y && (part.dims.y == 1);

        // Allocate main data buffer on device memory
        d_buffer = device::malloc<T>( buffer_size() );

        // Get maximum message size
        int max_msg_size = max(
            ( ntiles.y * ext_nx.y ) * max( gc.x.lower, gc.x.upper ),
            max( gc.y.lower, gc.y.upper ) * ( ntiles.x * ext_nx.x )
        );

        // Allocate message buffers
        msg_recv.lower = new Message<T>( max_msg_size, part.get_comm() );
        msg_recv.upper = new Message<T>( max_msg_size, part.get_comm() );
        msg_send.lower = new Message<T>( max_msg_size, part.get_comm() );
        msg_send.upper = new Message<T>( max_msg_size, part.get_comm() );

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
            std::cerr << "Invalid number of tiles " << global_ntiles << '\n';
            mpi::abort(1);
        }

        if ( nx.x == 0 || nx.y == 0 ) {
            std::cerr << "Invalid tiles size" << nx << '\n';
            mpi::abort(1);
        }

        // Parallel partition
        if ( (unsigned) part.dims.x > global_ntiles.x ) {
            std::cerr << "Number of parallel nodes along x (" ;
            std::cerr << part.dims.x << ") is larger than number of tiles along x(";
            std::cerr << global_ntiles.x << '\n';
            mpi::abort(1);
        }

        if ( (unsigned) part.dims.y > global_ntiles.y ) {
            std::cerr << "Number of parallel nodes along y (" ;
            std::cerr << part.dims.y << ") is larger than number of tiles along y(";
            std::cerr << global_ntiles.y << '\n';
            mpi::abort(1);
        }
    }

    public:

    /// @brief Parallel partition
    Partition & part;

    /// @brief Data buffer   
    T * d_buffer;

    /// @brief Global number of tiles
    const uint2 global_ntiles;

    /// @brief Tile grid size
    const uint2 nx;
    
    /// @brief Tile guard cells
    const bnd<unsigned int> gc;

    /// @brief Tile grize including guard cells
    const uint2 ext_nx;

    /// @brief Local offset in cells between lower tile corner and position (0,0)
    const unsigned int offset;

    /// @brief Tile volume (may be larger than product of cells for alignment)
    const unsigned int tile_vol;

    /// @brief Object name
    std::string name;

    /**
     * @brief Construct a new grid object
     * 
     * @param global_ntiles     Global number of tiles
     * @param nx                Individual tile size
     * @param gc                Number of guard cells
     * @param part              Parallel partition
     */
    grid( uint2 const global_ntiles, uint2 const nx, bnd<unsigned int> const gc, Partition & part ):
        part( part ),
        d_buffer( nullptr ), 
        global_ntiles( global_ntiles ),
        nx( nx ),
        gc(gc),
        ext_nx( make_uint2( gc.x.lower + nx.x + gc.x.upper,
                            gc.y.lower + nx.y + gc.y.upper )),
        offset( gc.y.lower * ext_nx.x + gc.x.lower ),
        tile_vol( roundup4( ext_nx.x * ext_nx.y ) ),
        name( "grid" )
    {
        // Validate parameters
        validate_parameters();

        // Set local information (ntiles, tile_start and local_periodic)
        initialize();
    };   

    /**
     * @brief Construct a new grid object
     * 
     * @note: The number of guard cells is set to 0
     * 
     * @param global_ntiles     Global number of tiles
     * @param nx                Individual tile size
     * @param part              Parallel partition
     */
    grid( uint2 const global_ntiles, uint2 const nx, Partition & part ):
        part( part ),
        d_buffer( nullptr ),
        global_ntiles( global_ntiles ),
        nx( nx ),
        gc( 0 ),
        ext_nx( make_uint2( nx.x, nx.y )),
        offset( 0 ),
        tile_vol( roundup4( nx.x * nx.y )),
        name( "grid" )
    {
        // Validate parameters
        validate_parameters();

        // Set local information (ntiles, tile_start and periodic)
        initialize();
    };

    /**
     * @brief Get the local number of tiles
     * 
     * @return int2 
     */
    uint2 get_ntiles() { return ntiles; };

    /**
     * @brief Returns the local tile offset in the global MPI tile grid
     * 
     * @return uint2 
     */
    uint2 get_tile_off() { return tile_off; };

    /**
     * @brief Get the global grid size
     * 
     * @return uint2 
     */
    uint2 get_global_nx() { return global_ntiles  * nx; }

    /**
     * @brief grid destructor
     * 
     */
    ~grid(){
        delete msg_recv.lower;
        delete msg_recv.upper;
        delete msg_send.lower;
        delete msg_send.upper;

        device::free( d_buffer );
    };

    /**
     * @brief Stream extraction
     * 
     * @param os 
     * @param obj 
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, const grid<T>& obj) {
        os << obj.name << "{";
        os << "(" << obj.ntiles.x << " x " << obj.ntiles.y << " tiles)";
        os << ", (" << obj.nx.x << " x " << obj.nx.y << " points/tile)";
        os << "}";
        return os;
    }

    /**
     * @brief Buffer size
     * 
     * @return total size of data buffers
     */
    std::size_t buffer_size() {
        return (static_cast <std::size_t> (tile_vol)) * ( ntiles.x * ntiles.y ) ;
    };

    /**
     * @brief zero device data on a grid grid
     * 
     * @return int       Returns 0 on success, -1 on error
     */
    int zero( ) {
        device::zero( d_buffer, buffer_size() );
        return 0;
    };

    /**
     * @brief Sets data to a constant value
     * 
     * @param val       Value
     */
    void set( T const & val ){
        device::setval( d_buffer, buffer_size(), val );
    };

    /**
     * @brief Scalar assigment
     * 
     * @param val       Value
     * @return T        Returns the same value
     */
    T operator=( T const val ) {
        set( val );
        return val;
    }

    /**
     * @brief Adds another grid object on top of local object
     * 
     * @param rhs         Other object to add
     */
    void add( const grid<T> &rhs ) {
        size_t const size = buffer_size( );
        int const block = 32;
        int const grid = (size -1) / block + 1;

        add_kernel <<< grid, block >>> ( d_buffer, rhs.d_buffer, size );
    };

    /**
     * @brief Operator +=
     * 
     * @param rhs           Other grid to add
     * @return grid<T>& 
     */
    grid<T>& operator+=(const grid<T>& rhs) {
        add( rhs );
        return *this;
    }

    /**
     * @brief Gather field into a contiguos grid
     * 
     * Used mostly for diagnostic output
     * 
     * @param out               Output buffer
     * @return unsigned int     Total number of cells
     */
    unsigned int gather( T * const __restrict__ d_out ) {

        dim3 block( 64 );
        dim3 grid( ntiles.x, ntiles.y );

        gather_kernel <<< grid, block >>> (
            d_out, d_buffer + offset,
            ntiles, nx, ext_nx );

        return local_nx.x * local_nx.y;
    }


    /**
     * @brief Copies edge values to X neighboring guard cells
     * 
     */
    void local_copy_to_gc_x() {
        dim3 grid( ntiles.x, ntiles.y );
        dim3 block( 64 );

        // Copy along x direction
        copy_to_gc_x_kernel <<< grid, block >>> (
            d_buffer, ntiles, nx, ext_nx,
            periodic.x, gc.x.lower, gc.x.upper
        );
    }

    /**
     * @brief Copies x values to neighboring guard cells, including cells on 
     *        other parallel nodes
     * 
     */
    void copy_to_gc_x() {

        // Get x neighbors
        int lnode = part.get_neighbor(-1, 0 );
        int unode = part.get_neighbor(+1, 0 );

        // Disable messages if local periodic
        if ( periodic.x ) lnode = unode = -1;

        // Post message receives
        if ( lnode >= 0 ) msg_recv.lower->irecv( lnode, source::lower );
        if ( unode >= 0 ) msg_recv.upper->irecv( unode, source::upper );

        // Pack send messages
        if ( lnode >= 0 || unode >= 0 ) {
            dim3 grid( 2, ntiles.y );
            dim3 block( 64 );

            T * lower = ( lnode >= 0 ) ? msg_send.lower -> buffer : nullptr;
            T * upper = ( unode >= 0 ) ? msg_send.upper -> buffer : nullptr;

            kernel_copy_to_gc_x_send <<< grid, block >>> (
                lower, upper, d_buffer, ntiles,
                nx, ext_nx, gc.x.lower, gc.x.upper
            );
        }

        // Send messages
        if ( lnode >= 0 ) {
            int msg_size = ( ext_nx.y * ntiles.y ) * gc.x.upper;
            msg_send.lower->isend( msg_size, lnode, dest::lower );
        }
        if ( unode >= 0 ) {
            int msg_size = ( ext_nx.y * ntiles.y ) * gc.x.lower;
            msg_send.upper->isend( msg_size, unode, dest::upper );
        }

        // Process local tiles
        local_copy_to_gc_x();

        // Wait for receive messages
        if ( lnode >= 0 ) msg_recv.lower-> wait();
        if ( unode >= 0 ) msg_recv.upper-> wait();

        // Unpack receive messages
        if ( lnode >= 0 || unode >= 0 ) {
            dim3 grid( 2, ntiles.y );
            dim3 block( 64 );

            T * lower = ( lnode >= 0 ) ? msg_recv.lower -> buffer : nullptr;
            T * upper = ( unode >= 0 ) ? msg_recv.upper -> buffer : nullptr;

            kernel_copy_to_gc_x_recv <<< grid, block >>>(
                lower, upper, d_buffer, ntiles,
                nx, ext_nx, gc.x.lower, gc.x.upper
            );
        }

        // Wait for send messages to complete
        if ( lnode >= 0 ) msg_send.lower->wait( );
        if ( unode >= 0 ) msg_send.upper->wait( );
    }



    /**
     * @brief Copies edge values to Y neighboring guard cells
     * 
     */
    void local_copy_to_gc_y() {
        dim3 grid( ntiles.x, ntiles.y );
        dim3 block( 64 );

        // Copy along y direction
        copy_to_gc_y_kernel <<< grid, block >>> (
            d_buffer, ntiles, nx, ext_nx,
            periodic.x, gc.x.lower, gc.x.upper
        );
    }

    /**
     * @brief Copies y values to neighboring guard cells, including cells on 
     *        other parallel nodes
     * 
     */
    void copy_to_gc_y() {

        // Get y neighbors
        int lnode = part.get_neighbor(0, -1);
        int unode = part.get_neighbor(0, +1);

        // Disable messages if local periodic
        if ( periodic.y ) lnode = unode = -1;
        
        // Post message receives
        if ( lnode >= 0 ) msg_recv.lower->irecv( lnode, source::lower );
        if ( unode >= 0 ) msg_recv.upper->irecv( unode, source::upper );

        // Pack send messages
        if ( lnode >= 0 || unode >= 0 ) {
            dim3 grid( ntiles.x, 2 );
            dim3 block( 64 );

            T * lower = ( lnode >= 0 ) ? msg_send.lower -> buffer : nullptr;
            T * upper = ( unode >= 0 ) ? msg_send.upper -> buffer : nullptr;

            kernel_copy_to_gc_y_send <<< grid, block >>> (
                lower, upper, d_buffer, ntiles,
                nx, ext_nx, gc.y.lower, gc.y.upper
            );
        }

        // Send messages
        if ( lnode >= 0 ) {
            int msg_size =  gc.y.upper * ( ntiles.x * ext_nx.x );
            msg_send.lower->isend( msg_size, lnode, dest::lower );
        }
        if ( unode >= 0 ) {
            int msg_size = gc.y.lower * ( ntiles.x * ext_nx.x );
            msg_send.upper -> isend( msg_size, unode, dest::upper );
        }

        // Process local tiles
        local_copy_to_gc_y();

        // Wait for receive messages
        if ( lnode >= 0 ) msg_recv.lower-> wait();
        if ( unode >= 0 ) msg_recv.upper-> wait();

        // Unpack receive messages
        if ( lnode >= 0 || unode >= 0 ) {
            dim3 grid( ntiles.x, 2 );
            dim3 block( 64 );

            T * lower = ( lnode >= 0 ) ? msg_recv.lower -> buffer : nullptr;
            T * upper = ( unode >= 0 ) ? msg_recv.upper -> buffer : nullptr;

            kernel_copy_to_gc_y_recv <<< grid, block >>>(
                lower, upper, d_buffer, ntiles,
                nx, ext_nx, gc.y.lower, gc.y.upper
            );
        }

        // Wait for send messages to complete
        if ( lnode >= 0 ) msg_send.lower->wait( );
        if ( unode >= 0 ) msg_send.upper->wait( );
    }

    /**
     * @brief Copies edge values to neighboring guard cells, including other
     *        parallel nodes
     * 
     */
    void copy_to_gc()  {

        // Copy along x direction
        copy_to_gc_x();

        // Copy along y direction
        copy_to_gc_y();

    };

    /**
     * @brief Adds values from neighboring x guard cells to local data
     * 
     */
    void local_add_from_gc_x() {
        dim3 grid( ntiles.x, ntiles.y );
        dim3 block( 64 );

        // Add along x direction
        add_from_gc_x_kernel <<< grid, block >>> (
            d_buffer, ntiles, nx, ext_nx,
            periodic.x, gc.x.lower, gc.x.upper
        );
    }

    /**
     * @brief Adds values from neighboring y guard cells to local data
     * 
     */
    void local_add_from_gc_y() {
        dim3 grid( ntiles.x, ntiles.y );
        dim3 block( 64 );

        // Add along y direction
        add_from_gc_y_kernel <<< grid, block >>> (
            d_buffer, ntiles, nx, ext_nx,
            periodic.y, gc.y.lower, gc.y.upper
        );
    }

    /**
     * @brief Adds values from neighboring y guard cells to local data,
     *        including cells from other parallel nodes
     * 
     */
    void add_from_gc_x() {

        // Get x neighbors
        int lnode = part.get_neighbor(-1, 0 );
        int unode = part.get_neighbor(+1, 0 );

        // Disable messages if local periodic
        if ( periodic.x ) lnode = unode = -1;

        // Post message receives
        if ( lnode >= 0 ) msg_recv.lower->irecv( lnode, source::lower );
        if ( unode >= 0 ) msg_recv.upper->irecv( unode, source::upper );

        // Pack send messages
        if ( lnode >= 0 || unode >= 0 ) {
            dim3 grid( 2, ntiles.y );
            dim3 block( 64 );

            T * lower = ( lnode >= 0 ) ? msg_send.lower -> buffer : nullptr;
            T * upper = ( unode >= 0 ) ? msg_send.upper -> buffer : nullptr;

            kernel_add_from_gc_x_send <<< grid, block >>> (
                lower, upper, d_buffer, ntiles,
                nx, ext_nx, gc.x.lower, gc.x.upper
            );
        }

        // Send messages
        if ( lnode >= 0 ) {
            int msg_size = ( ext_nx.y * ntiles.y ) * gc.x.lower;
            msg_send.lower->isend( msg_size, lnode, dest::lower );
        }
        if ( unode >= 0 ) {
            int msg_size = ( ext_nx.y * ntiles.y ) * gc.x.upper;
            msg_send.upper->isend( msg_size, unode, dest::upper );
        }

        // Process local tiles
        local_add_from_gc_x();

        // Wait for receive messages
        if ( lnode >= 0 ) msg_recv.lower-> wait();
        if ( unode >= 0 ) msg_recv.upper-> wait();

        // Unpack receive messages
        if ( lnode >= 0 || unode >= 0 ) {
            dim3 grid( 2, ntiles.y );
            dim3 block( 64 );

            T * lower = ( lnode >= 0 ) ? msg_recv.lower -> buffer : nullptr;
            T * upper = ( unode >= 0 ) ? msg_recv.upper -> buffer : nullptr;

            kernel_add_from_gc_x_recv <<< grid, block >>>(
                lower, upper, d_buffer, ntiles,
                nx, ext_nx, gc.x.lower, gc.x.upper
            );
        }

        // Wait for send messages to complete
        if ( lnode >= 0 ) msg_send.lower->wait( );
        if ( unode >= 0 ) msg_send.upper->wait( );
    }

    /**
     * @brief Adds values from neighboring y guard cells to local data,
     *        including cells from other parallel nodes
     * 
     */
    void add_from_gc_y() {
        // Get y neighbors
        int lnode = part.get_neighbor( 0, -1 );
        int unode = part.get_neighbor( 0, +1 );

        // Disable messages if local periodic
        if ( periodic.y ) lnode = unode = -1;

        // Post message receives
        if ( lnode >= 0 ) msg_recv.lower->irecv( lnode, source::lower );
        if ( unode >= 0 ) msg_recv.upper->irecv( unode, source::upper );

        // Pack send messages
        if ( lnode >= 0 || unode >= 0 ) {
            dim3 grid( ntiles.x, 2 );
            dim3 block( 64 );

            T * lower = ( lnode >= 0 ) ? msg_send.lower -> buffer : nullptr;
            T * upper = ( unode >= 0 ) ? msg_send.upper -> buffer : nullptr;

            kernel_add_from_gc_y_send <<< grid, block >>> (
                lower, upper, d_buffer, ntiles,
                nx, ext_nx, gc.x.lower, gc.x.upper
            );
        }

        // Send messages
        if ( lnode >= 0 ) {
            int msg_size = gc.y.lower * ( ntiles.x * ext_nx.x );
            msg_send.lower->isend( msg_size, lnode, dest::lower );
        }
        if ( unode >= 0 ) {
            int msg_size    = gc.y.upper * ( ntiles.x * ext_nx.x );
            msg_send.upper->isend( msg_size, unode, dest::upper );
        }

        // Process local tiles
        local_add_from_gc_y();

        // Wait for receive messages
        if ( lnode >= 0 ) msg_recv.lower-> wait();
        if ( unode >= 0 ) msg_recv.upper-> wait();

        // Unpack receive messages
        if ( lnode >= 0 || unode >= 0 ) {
            dim3 grid( ntiles.x, 2 );
            dim3 block( 64 );

            T * lower = ( lnode >= 0 ) ? msg_recv.lower -> buffer : nullptr;
            T * upper = ( unode >= 0 ) ? msg_recv.upper -> buffer : nullptr;

            kernel_add_from_gc_y_recv <<< grid, block >>>(
                lower, upper, d_buffer, ntiles,
                nx, ext_nx, gc.y.lower, gc.y.upper
            );
        }

        // Wait for send messages to complete
        if ( lnode >= 0 ) msg_send.lower->wait( );
        if ( unode >= 0 ) msg_send.upper->wait( );
    }

    /**
     * @brief Adds values from neighboring guard cells to local data, including
     *        values from other parallel nodes
     * 
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
     * @note The routine assumes the (x) guard cell values are correct. If this
     *       is not the case, you will need to update them first.
     * 
     * @warning This operation is only allowed if the number of upper x guard
     *          cells is greater or equal to the requested shift.
     * 
     * @param shift Number of cells to shift
     */
    void x_shift_left( unsigned int const shift ) {

        if ( gc.x.upper >= shift ) {

            dim3 grid( ntiles.x, ntiles.y );

            // Shift data using guard cell values
            size_t shm_size = tile_vol * sizeof(T);
            
            block::set_shmem_size( x_shift_left_kernel<T>, shm_size );
            x_shift_left_kernel <<< grid, 1024, shm_size >>> ( 
                d_buffer, ntiles, ext_nx, shift
            );

            copy_to_gc_x();

        } else {
            ABORT( "grid::x_shift_left(), shift value too large, must be <= gc.x.upper" );
        }
    }

    /**
     * @brief Perform a convolution with a 3 point kernel [a,b,c] along x
     * 
     * @param a     Kernel value a
     * @param b     Kernel value b
     * @param c     Kernel value c
     */
    template < typename S >
    void kernel3_x( S const a, S const b, S const c ) {

        if (( gc.x.lower > 0) && (gc.x.upper > 0)) {

            dim3 grid( ntiles.x, ntiles.y );

            size_t shm_size = 2 * tile_vol * sizeof(T);

            if ( shm_size > block::shared_mem_size() ) {
                ABORT("grid::x_shift_left(), tile size too large, insufficient shared memory");
            }

            block::set_shmem_size( kernel3_x_kernel<T,S>, shm_size );
            kernel3_x_kernel <<< grid, 1024, shm_size >>> ( 
                d_buffer, ntiles, nx, ext_nx, gc.x.lower, a, b, c
            ); 

            copy_to_gc_x();

        } else {
            ABORT("grid::kernel3_x() requires at least 1 guard cell at both the lower and upper x boundaries.");
        }
    }

    /**
     * @brief Perform a convolution with a 3 point kernel [a,b,c] along y
     * 
     * @param a     Kernel value a
     * @param b     Kernel value b
     * @param c     Kernel value c
     */
    template < typename S >
    void kernel3_y( S const a, S const b, S const c ) {

        if (( gc.y.lower > 0) && (gc.y.upper > 0)) {

            dim3 grid( ntiles.x, ntiles.y );

            size_t shm_size = 2 * tile_vol * sizeof(T);

            block::set_shmem_size( kernel3_y_kernel<T,S>, shm_size );
            kernel3_y_kernel <<< grid, 1024, shm_size >>> ( 
                d_buffer, ntiles, nx, ext_nx, gc.y.lower, a, b, c
            ); 

            copy_to_gc_y();

        } else {
            ABORT("grid::kernel3_y() requires at least 1 guard cell at both the lower and upper y boundaries.");
        }
    }

    /**
     * @brief Save field values to disk
     * 
     * The field type <T> must be supported by ZDF file format
     * 
     */

    /**
     * @brief Save field values to disk
     * 
     * @note The field type <T> must be supported by ZDF file format
     * 
     * @param info      Grid metadata (label, units, axis, etc.). Information is used to set file name
     * @param iter      Iteration metadata
     * @param path      Base path for file
     */
    void save( zdf::grid_info &info, zdf::iteration &iter, std::string path ) {

        // Fill in grid dimensions
        info.ndims = 2;
        info.count[0] = global_ntiles.x * nx.x;
        info.count[1] = global_ntiles.y * nx.y;

        const std::size_t bsize = local_nx.x * local_nx.y;

        // Allocate buffers on host and device to gather data
        T * h_data = host::malloc<T>( bsize );
        T * d_data = device::malloc<T>( bsize );

        // Gather data on contiguous grid
        gather( d_data );

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
        zdf::save_grid<T>( chunk, info, iter, path, part.get_comm() );

        // Free remaining temporary buffer        
        host::free( h_data );
    };

    /**
     * @brief Save grid values to disk
     * 
     * @param filename      Output file name (includes path)
     */
    void save( std::string filename ) {
   
        const std::size_t bsize = local_nx.x * local_nx.y;

        // Allocate buffers on host and device to gather data
        T * h_data = host::malloc<T>( bsize );
        T * d_data = device::malloc<T>( bsize );

        // Gather data on contiguous grid
        gather( d_data );

        // Copy to host and free device memory
        device::memcpy_tohost( h_data, d_data, bsize );
        device::free( d_data );

        uint64_t global[2] = { global_ntiles.x * nx.x, global_ntiles.y * nx.y };
        uint64_t start[2]  = { tile_off.x * nx.x, tile_off.y * nx.y };
        uint64_t local[2]  = { local_nx.x, local_nx.y };

        // Save data
        zdf::save_grid( h_data, 2, global, start, local, name, filename, part.get_comm() );

        // Free remaining temporary buffer 
        host::free( h_data );
    }

};

#endif