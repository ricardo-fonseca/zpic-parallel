#ifndef CYL3GRID_H_
#define CYL3GRID_H_

#include "cyl3.h"
#include "grid.h"

template < class T > 
class cyl3grid : public grid< cyl3<T> >
{
    protected:

    // Get the vector type, must be supported by <cyl3.h>
    using V = cyl3<T>;

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
    unsigned int gather( const enum fcomp::cyl fc, T * const __restrict__ out ) {
        
        switch( fc ) {
            case( fcomp::z ):
                #pragma omp parallel for
                for( unsigned int tid = 0; tid < ntiles.x * ntiles.y; tid++ )  {
                    const auto tile_idx = make_uint2( tid % ntiles.x, tid / ntiles.x );
                    const auto tile_off = tid * tile_vol + offset;

                    V * const __restrict__ tile_data = & d_buffer[ tile_off ]; 

                    // Loop inside tile
                    for( unsigned iy = 0; iy < nx.y; iy ++ ) {
                        for( unsigned ix = 0; ix < nx.x; ix ++ ) {

                            auto const gix = tile_idx.x * nx.x + ix;
                            auto const giy = tile_idx.y * nx.y + iy;

                            auto const out_idx = giy * local_nx.x + gix;

                            out[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].z;
                        }
                    }
                }
                break;
            case( fcomp::r ):
                #pragma omp parallel for
                for( unsigned int tid = 0; tid < ntiles.x * ntiles.y; tid++ )  {
                    const auto tile_idx = make_uint2( tid % ntiles.x, tid / ntiles.x );
                    const auto tile_off = tid * tile_vol + offset;

                    V * const __restrict__ tile_data = & d_buffer[ tile_off ]; 

                    // Loop inside tile
                    for( unsigned iy = 0; iy < nx.y; iy ++ ) {
                        for( unsigned ix = 0; ix < nx.x; ix ++ ) {

                            auto const gix = tile_idx.x * nx.x + ix;
                            auto const giy = tile_idx.y * nx.y + iy;

                            auto const out_idx = giy * local_nx.x + gix;

                            out[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].r;
                        }
                    }
                }
                break;

            case( fcomp::θ ):
                #pragma omp parallel for
                for( unsigned int tid = 0; tid < ntiles.x * ntiles.y; tid++ )  {
                    const auto tile_idx = make_uint2( tid % ntiles.x, tid / ntiles.x );
                    const auto tile_off = tid * tile_vol + offset;

                    V * const __restrict__ tile_data = & d_buffer[ tile_off ]; 

                    // Loop inside tile
                    for( unsigned iy = 0; iy < nx.y; iy ++ ) {
                        for( unsigned ix = 0; ix < nx.x; ix ++ ) {

                            auto const gix = tile_idx.x * nx.x + ix;
                            auto const giy = tile_idx.y * nx.y + iy;

                            auto const out_idx = giy * local_nx.x + gix;

                            out[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].θ;
                        }
                    }
                }
                break;
        }

        return local_nx.x * local_nx.y;
    }

    /**
     * @brief Gather all field component values from tiles into a 
     *        contiguous grid
     * 
     * @note The output array will be organized in 3 sequential blocks (i.e. SoA):
     *       z components, r components and θ components
     * 
     * @param out               Scalar output buffer
     * @return unsigned int     Total number of cells * 3
     */
    unsigned int gather( T * const __restrict__ d_out ) {

        T * const __restrict__ out_z = & d_out[                           0 ];
        T * const __restrict__ out_r = & d_out[     local_nx.x * local_nx.y ];
        T * const __restrict__ out_θ = & d_out[ 2 * local_nx.x * local_nx.y ];

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

                        auto const out_idx = giy * local_nx.x + gix;

                        out_z[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].z;
                        out_r[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].r;
                        out_θ[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].θ;
                    }
                }
            }
        }

        return 3 * local_nx.x * local_nx.y;
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

                            auto const in_idx = giy * local_nx.x + gix;

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

                            auto const in_idx = giy * local_nx.x + gix;

                            tile_data[ iy * ext_nx.x + ix ].r = d_in[ in_idx ];
                        }
                    }
                }
                break;

            case( fcomp::θ ):
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

                            auto const in_idx = giy * local_nx.x + gix;

                            tile_data[ iy * ext_nx.x + ix ].θ = d_in[ in_idx ];
                        }
                    }
                }
                break;
        }

        return local_nx.x * local_nx.y;
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

        T const * const __restrict__ in_z = d_in[                           0 ];
        T const * const __restrict__ in_r = d_in[     local_nx.x * local_nx.y ];
        T const * const __restrict__ in_θ = d_in[ 2 * local_nx.x * local_nx.y ];

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

                    auto const in_idx = giy * local_nx.x + gix;

                    tile_data[ iy * ext_nx.x + ix ].z = in_z[ in_idx ];
                    tile_data[ iy * ext_nx.x + ix ].r = in_r[ in_idx ];
                    tile_data[ iy * ext_nx.x + ix ].θ = in_θ[ in_idx ];
                }
            }
        }

        return local_nx.x * local_nx.y * 3;
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

        T const * const __restrict__ in_z = & d_in[                           0 ];
        T const * const __restrict__ in_r = & d_in[     local_nx.x * local_nx.y ];
        T const * const __restrict__ in_θ = & d_in[ 2 * local_nx.x * local_nx.y ];

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

                    auto const in_idx = giy * local_nx.x + gix;

                    tile_data[ iy * ext_nx.x + ix ].z = in_z[ in_idx ] * scale;
                    tile_data[ iy * ext_nx.x + ix ].r = in_r[ in_idx ] * scale;
                    tile_data[ iy * ext_nx.x + ix ].θ = in_θ[ in_idx ] * scale;
                }
            }
        }

        return local_nx.x * local_nx.y * 3;
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
    void save( const enum fcomp::cyl fc, zdf::grid_info &info, zdf::iteration &iter, std::string path ) {

        // Fill in grid dimensions
        info.ndims = 2;
        info.count[0] = global_ntiles.x * nx.x;
        info.count[1] = global_ntiles.y * nx.y;

        // Allocate buffer on host to gather data
        T * h_data = memory::malloc<T>( local_nx.x * local_nx.y );

        gather( fc, h_data );

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

        memory::free( h_data );
    }

    /**
     * @brief Save grid values to disk
     * 
     * @param fc        Field component to save
     * @param filename  Output file name (includes path)
     */
    void save( const enum fcomp::cyl fc, std::string filename ) {
        
        const std::size_t bsize = local_nx.x * local_nx.y;

        // Allocate buffers on host and device to gather data
        T * h_data = memory::malloc<T>( bsize );

        // Gather data on contiguous grid
        gather( fc, h_data );

        // Save data
        uint64_t global[2] = { global_ntiles.x * nx.x, global_ntiles.y * nx.y };
        uint64_t start[2] = { tile_off.x * nx.x, tile_off.y * nx.y };
        uint64_t local[2]  = { local_nx.x, local_nx.y };

        std::string fcomp_name[] = {"z","r","θ"};
        std::string name_fc = name + "-" + fcomp_name[fc];

        zdf::save_grid( h_data, 2, global, start, local, name_fc, filename, part.get_comm() );

        // Free remaining temporary buffer 
        memory::free( h_data );
    }

};

#endif