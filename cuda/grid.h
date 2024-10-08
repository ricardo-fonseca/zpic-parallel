#ifndef GRID_H_
#define GRID_H_

#include <iostream>

#include "utils.h"
#include "bnd.h"
#include "zdf-cpp.h"

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
 * @tparam T 
 * @param d_out     Output buffer
 * @param d_buffer  Input buffer (includes offset to cell [0,0])
 * @param gnx       Global grid size
 * @param nx        Local grid size
 * @param ext_nx    Local grid size including guard cells
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

    const uint2  gnx = { ntiles.x * nx.x, ntiles.y * nx.y };

    auto * const __restrict__ tile_data = & d_buffer[ tile_off ];

    for( int i = block_thread_rank(); i < nx.x * nx.y; i+= block_num_threads() ) {
        const auto ix = i % nx.x;
        const auto iy = i / nx.x;

        auto const gix = tile_idx.x * nx.x + ix;
        auto const giy = tile_idx.y * nx.y + iy;

        auto const out_idx = giy * gnx.x + gix;

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
 * @brief End of CUDA kernels namespace
 * 
 */
}

/**
 * @brief 
 * 
 */
template <class T>
class grid {

    protected:

    public:

    T * d_buffer;

    /// Number of tiles
    const uint2 ntiles;

    /// Tile grid size
    const uint2 nx;
    
    /// Tile guard cells
    const bnd<unsigned int> gc;

    /// Consider boundaries to be periodic
    int2 periodic;

    /// Global grid size
    const uint2 gnx;
    
    /// Tile grize including guard cells
    const uint2 ext_nx;

    /// Offset in cells between lower tile corner and position (0,0)
    const unsigned int offset;

    /// Tile volume (may be larger than product of cells for alingment)
    const unsigned int tile_vol;

    /// Object name
    std::string name;

    /**
     * @brief Construct a new grid object
     * 
     * @param ntiles    Number of tiles
     * @param nx        Individual tile size
     * @param gc        Number of guard cells
     */
    grid( uint2 const ntiles, uint2 const nx, bnd<unsigned int> const gc ):
        d_buffer( nullptr ), 
        ntiles( ntiles ),
        nx( nx ),
        gc(gc),
        periodic( { 1, 1 } ),
        gnx( { ntiles.x * nx.x, ntiles.y * nx.y } ),
        ext_nx( { gc.x.lower +  nx.x + gc.x.upper,
                  gc.y.lower +  nx.y + gc.y.upper } ),
        offset( { gc.y.lower * ext_nx.x + gc.x.lower } ),
        tile_vol( roundup4( ext_nx.x * ext_nx.y ) ),
        name( "grid" )
    {
        d_buffer = device::malloc<T>( buffer_size() );
    };

    /**
     * @brief Construct a new grid object
     * 
     * The number of guard cells is set to 0
     * 
     * @param ntiles    Number of tiles
     * @param nx        Individual tile size
     */
    grid( uint2 const ntiles, uint2 const nx ):
        d_buffer( nullptr ),
        ntiles( ntiles ),
        nx( nx ),
        gc( 0 ),
        periodic( { 1, 1 } ),
        gnx( { ntiles.x * nx.x, ntiles.y * nx.y } ),
        ext_nx( { nx.x, nx.y } ),
        offset( 0 ),
        tile_vol( roundup4( nx.x * nx.y )),
        name( "grid" )
    {
        d_buffer = device::malloc<T>( buffer_size() );
    };

    /**
     * @brief grid destructor
     * 
     */
    ~grid(){
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

        return gnx.x * gnx.y;
    }

    /**
     * @brief Copies edge values to neighboring guard cells
     * 
     */
    void copy_to_gc( )  {

        dim3 block( 64 );
        dim3 grid( ntiles.x, ntiles.y );

        // Copy along x direction
        copy_to_gc_x_kernel <<< grid, block >>> (
            d_buffer, ntiles, nx, ext_nx,
            periodic.x, gc.x.lower, gc.x.upper
        );

        // Copy along y direction
        copy_to_gc_y_kernel <<< grid, block >>> (
            d_buffer, ntiles, nx, ext_nx,
            periodic.y, gc.y.lower, gc.y.upper
        );
    };

    /**
     * @brief Adds values from neighboring guard cells to local data
     * 
     */
    void add_from_gc( ){

        dim3 block( 64 );
        dim3 grid( ntiles.x, ntiles.y );

        // Add along x direction
        add_from_gc_x_kernel <<< grid, block >>> (
            d_buffer, ntiles, nx, ext_nx,
            periodic.x, gc.x.lower, gc.x.upper
        );

        // Add along y direction
        add_from_gc_y_kernel <<< grid, block >>> (
            d_buffer, ntiles, nx, ext_nx,
            periodic.y, gc.y.lower, gc.y.upper
        );

    };

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
     * @param q     Sycl queue
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

            // Update x guard cells not changing lower and upper global guard cells
            copy_to_gc_x_kernel <<< grid, 128 >>> (
                d_buffer, ntiles, nx, ext_nx,
                0, gc.x.lower, gc.x.upper
            );

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

            copy_to_gc_x_kernel <<< grid, 128 >>> (
                d_buffer, ntiles, nx, ext_nx,
                periodic.x, gc.x.lower, gc.x.upper
            );

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

            copy_to_gc_y_kernel <<< grid, 128 >>> (
                d_buffer, ntiles, nx, ext_nx,
                periodic.y, gc.y.lower, gc.y.upper
            );

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
    void save( zdf::grid_info &info, zdf::iteration &iter, std::string path ) {

        // Fill in grid dimensions
        info.ndims = 2;
        info.count[0] = gnx.x;
        info.count[1] = gnx.y;

        const std::size_t bsize = gnx.x * gnx.y;

        T * h_data = host::malloc<T>( bsize );

        T * d_data = device::malloc<T>( bsize );

        gather( d_data );

        device::memcpy_tohost( h_data, d_data, bsize );

        device::free( d_data );

        zdf::save_grid( h_data, info, iter, path );

        host::free( h_data );
    };

    void save( std::string path ) {
        
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

        std::string grid_name = "cuda";
        std::string grid_label = "cuda test";

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

        save( info, iter, path );
    }

};

#endif