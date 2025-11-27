#ifndef CYL3GRID_H_
#define CYL3GRID_H_

#include "cyl3.h"
#include "grid.h"

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
template< int fc, typename T, typename V >
__global__
void gather_kernel( 
    T * const __restrict__ d_out, V const * const __restrict__ d_buffer,
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

        if constexpr ( fc == 0 ) d_out[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].z;
        if constexpr ( fc == 1 ) d_out[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].r;
        if constexpr ( fc == 2 ) d_out[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].th;
    }
}

}
/**
 * End of CUDA kernels namespace
 */



template < class T > 
class cyl3grid : public grid< cyl3<T> >
{
    protected:

    // Get the vector type, must be supported by <cyl3.h>
    using V = cyl3<T>;

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
    unsigned int gather( const enum fcomp::cyl fc, T * const __restrict__ d_out ) {

        dim3 block( 64 );
        dim3 grid( ntiles.x, ntiles.y );

        switch( fc ) {
            case( fcomp::z ):
                gather_kernel<0> <<< grid, block >>> (
                    d_out, &d_buffer[ offset ],
                    ntiles, nx, ext_nx );
                break;
            case( fcomp::r ):
                gather_kernel<1> <<< grid, block >>> (
                    d_out, &d_buffer[ offset ],
                    ntiles, nx, ext_nx );
                break;

            case( fcomp::th ):
                gather_kernel<2> <<< grid, block >>> (
                    d_out, &d_buffer[ offset ],
                    ntiles, nx, ext_nx );
                break;
        }

        return dims.x * dims.y;
    }

#if 0
    /**
     * @brief Gather all field component values from tiles into a 
     *        contiguous grid
     * 
     * @note The output array will be organized in 3 sequential blocks (i.e. SoA):
     *       z components, r components and t components
     * 
     * @param out               Scalar output buffer
     * @return unsigned int     Total number of cells * 3
     */
    unsigned int gather( T * const __restrict__ d_out ) {

        T * const __restrict__ out_z = & d_out[                   0 ];
        T * const __restrict__ out_r = & d_out[     dims.x * dims.y ];
        T * const __restrict__ out_t = & d_out[ 2 * dims.x * dims.y ];

        for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
            for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {

                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol + offset;

                V * const __restrict__ tile_data = & d_buffer[ tile_off ]; 

                // Loop inside tile
                for( unsigned iy = 0; iy < nx.y; iy ++ ) {
                    for( unsigned ix = 0; ix < nx.x; ix ++ ) {

                        auto const gix = tile_idx.x * nx.x + ix;
                        auto const giy = tile_idx.y * nx.y + iy;

                        auto const out_idx = giy * dims.x + gix;

                        out_z[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].z;
                        out_r[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].r;
                        out_t[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].t;
                    }
                }
            }
        }

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
    unsigned int scatter( const enum fcomp::cyl fc, T * const __restrict__ d_in ) {

        switch( fc ) {
            case( fcomp::z ):
                #pragma omp parallel for
                for( int tid = 0; tid < ntiles.x * ntiles.y; tid++ )  {
                    const auto tile_idx = make_uint2( tid % ntiles.x, tid / ntiles.x );
                    const auto tile_off = tid * tile_vol + offset;

                    V * const __restrict__ tile_data = & d_buffer[ tile_off ]; 

                    // Loop inside tile
                    for( unsigned iy = 0; iy < nx.y; iy ++ ) {
                        for( unsigned ix = 0; ix < nx.x; ix ++ ) {

                            auto const gix = tile_idx.x * nx.x + ix;
                            auto const giy = tile_idx.y * nx.y + iy;

                            auto const in_idx = giy * dims.x + gix;

                            tile_data[ iy * ext_nx.x + ix ].z = d_in[ in_idx ];
                        }
                    }
                }
                break;
            case( fcomp::r ):
                #pragma omp parallel for
                for( int tid = 0; tid < ntiles.x * ntiles.y; tid++ )  {
                    const auto tile_idx = make_uint2( tid % ntiles.x, tid / ntiles.x );
                    const auto tile_off = tid * tile_vol + offset;

                    V * const __restrict__ tile_data = & d_buffer[ tile_off ]; 

                    // Loop inside tile
                    for( unsigned iy = 0; iy < nx.y; iy ++ ) {
                        for( unsigned ix = 0; ix < nx.x; ix ++ ) {

                            auto const gix = tile_idx.x * nx.x + ix;
                            auto const giy = tile_idx.y * nx.y + iy;

                            auto const in_idx = giy * dims.x + gix;

                            tile_data[ iy * ext_nx.x + ix ].r = d_in[ in_idx ];
                        }
                    }
                }
                break;

            case( fcomp::t ):
                #pragma omp parallel for
                for( int tid = 0; tid < ntiles.x * ntiles.y; tid++ )  {
                    const auto tile_idx = make_uint2( tid % ntiles.x, tid / ntiles.x );
                    const auto tile_off = tid * tile_vol + offset;

                    V * const __restrict__ tile_data = & d_buffer[ tile_off ]; 

                    // Loop inside tile
                    for( unsigned iy = 0; iy < nx.y; iy ++ ) {
                        for( unsigned ix = 0; ix < nx.x; ix ++ ) {

                            auto const gix = tile_idx.x * nx.x + ix;
                            auto const giy = tile_idx.y * nx.y + iy;

                            auto const in_idx = giy * dims.x + gix;

                            tile_data[ iy * ext_nx.x + ix ].t = d_in[ in_idx ];
                        }
                    }
                }
                break;
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
    unsigned int scatter( T const * const __restrict__ d_in ) {

        T const * const __restrict__ in_z = d_in[                   0 ];
        T const * const __restrict__ in_r = d_in[     dims.x * dims.y ];
        T const * const __restrict__ in_t = d_in[ 2 * dims.x * dims.y ];

        #pragma omp parallel for
        for( int tid = 0; tid < ntiles.x * ntiles.y; tid++ )  {
            const auto tile_idx = make_uint2( tid % ntiles.x, tid / ntiles.x );
            const auto tile_off = tid * tile_vol + offset;

            V * const __restrict__ tile_data = & d_buffer[ tile_off ]; 

            // Loop inside tile
            for( unsigned iy = 0; iy < nx.y; iy ++ ) {
                for( unsigned ix = 0; ix < nx.x; ix ++ ) {

                    auto const gix = tile_idx.x * nx.x + ix;
                    auto const giy = tile_idx.y * nx.y + iy;

                    auto const in_idx = giy * dims.x + gix;

                    tile_data[ iy * ext_nx.x + ix ].z = in_z[ in_idx ];
                    tile_data[ iy * ext_nx.x + ix ].r = in_r[ in_idx ];
                    tile_data[ iy * ext_nx.x + ix ].t = in_t[ in_idx ];
                }
            }
        }

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
    unsigned int scatter( T const * const __restrict__ d_in, T const scale ) {

        T const * const __restrict__ in_z = & d_in[                   0 ];
        T const * const __restrict__ in_r = & d_in[     dims.x * dims.y ];
        T const * const __restrict__ in_t = & d_in[ 2 * dims.x * dims.y ];

        #pragma omp parallel for
        for( int tid = 0; tid < ntiles.x * ntiles.y; tid++ )  {
            const auto tile_idx = make_uint2( tid % ntiles.x, tid / ntiles.x );
            const auto tile_off = tid * tile_vol + offset;

            V * const __restrict__ tile_data = & d_buffer[ tile_off ]; 

            // Loop inside tile
            for( unsigned iy = 0; iy < nx.y; iy ++ ) {
                for( unsigned ix = 0; ix < nx.x; ix ++ ) {

                    auto const gix = tile_idx.x * nx.x + ix;
                    auto const giy = tile_idx.y * nx.y + iy;

                    auto const in_idx = giy * dims.x + gix;

                    tile_data[ iy * ext_nx.x + ix ].z = in_z[ in_idx ] * scale;
                    tile_data[ iy * ext_nx.x + ix ].r = in_r[ in_idx ] * scale;
                    tile_data[ iy * ext_nx.x + ix ].t = in_t[ in_idx ] * scale;
                }
            }
        }

        return dims.x * dims.y * 3;
    }
#endif

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
    void save( const enum fcomp::cyl fc, zdf::grid_info &metadata, zdf::iteration &iter, std::string path ) {

        // Fill in grid dimensions
        metadata.ndims = 2;
        metadata.count[0] = dims.x;
        metadata.count[1] = dims.y;

        const std::size_t bsize = dims.x * dims.y;
        T * h_data = host::malloc<T>( bsize );
        T * d_data = device::malloc<T>( bsize );

        gather( fc, d_data );

        device::memcpy_tohost( h_data, d_data, bsize );
        device::free( d_data );

        // Save data
        zdf::save_grid( h_data, metadata, iter, path );

        host::free( h_data );
    }

    /**
     * @brief Save grid values to disk
     * 
     * @param fc        Field component to save
     * @param filename  Output file name (includes path)
     */
    void save( const enum fcomp::cyl fc, std::string filename ) {
        
        const std::size_t bsize = dims.x * dims.y;
        T * h_data = host::malloc<T>( bsize );
        T * d_data = device::malloc<T>( bsize );

        gather( fc, d_data );

        device::memcpy_tohost( h_data, d_data, bsize );
        device::free( d_data );

        // Save data
        uint64_t grid_dims[] = {dims.x, dims.y};
        std::string fcomp_name[] = {"z","r","θ"};
        std::string name_fc = name + "-" + fcomp_name[fc];
        zdf::save_grid( h_data, 2, grid_dims, name_fc, filename );

        // Free remaining temporary buffer 
        host::free( h_data );
    }

};

#endif