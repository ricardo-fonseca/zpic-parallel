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

template< class T >
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

template < class V > 
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
    unsigned int gather( const enum fcomp::cart fc, S * const __restrict__ out ) {
        
        switch( fc ) {
            case( fcomp::x ):
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

                                out[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].x;
                            }
                        }
                    }
                }
                break;
            case( fcomp::y ):
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

                                out[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].y;
                            }
                        }
                    }
                }
                break;

            case( fcomp::z ):
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

                                out[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].z;
                            }
                        }
                    }
                }
                break;
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

        S * const __restrict__ out_x = & d_out[                   0 ];
        S * const __restrict__ out_y = & d_out[     dims.x * dims.y ];
        S * const __restrict__ out_z = & d_out[ 2 * dims.x * dims.y ];

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

                        out_x[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].x;
                        out_y[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].y;
                        out_z[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].z;
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
    unsigned int scatter( const enum fcomp::cart fc, S * const __restrict__ d_in ) {

        switch( fc ) {
            case( fcomp::x ):
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

                                auto const in_idx = giy * dims.x + gix;

                                tile_data[ iy * ext_nx.x + ix ].x = d_in[ in_idx ];
                            }
                        }
                    }
                }
                break;
            case( fcomp::y ):
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

                                auto const in_idx = giy * dims.x + gix;

                                tile_data[ iy * ext_nx.x + ix ].y = d_in[ in_idx ];
                            }
                        }
                    }
                }
                break;

            case( fcomp::z ):
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

                                auto const in_idx = giy * dims.x + gix;

                                tile_data[ iy * ext_nx.x + ix ].z = d_in[ in_idx ];
                            }
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
    unsigned int scatter( S const * const __restrict__ d_in ) {

        S const * const __restrict__ in_x = d_in[                   0 ];
        S const * const __restrict__ in_y = d_in[     dims.x * dims.y ];
        S const * const __restrict__ in_z = d_in[ 2 * dims.x * dims.y ];


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

                        auto const in_idx = giy * dims.x + gix;

                        tile_data[ iy * ext_nx.x + ix ].x = in_x[ in_idx ];
                        tile_data[ iy * ext_nx.x + ix ].y = in_y[ in_idx ];
                        tile_data[ iy * ext_nx.x + ix ].z = in_z[ in_idx ];
                    }
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
    unsigned int scatter( S const * const __restrict__ d_in, S const scale ) {

        S const * const __restrict__ in_x = & d_in[                   0 ];
        S const * const __restrict__ in_y = & d_in[     dims.x * dims.y ];
        S const * const __restrict__ in_z = & d_in[ 2 * dims.x * dims.y ];

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

                        auto const in_idx = giy * dims.x + gix;

                        tile_data[ iy * ext_nx.x + ix ].x = in_x[ in_idx ] * scale;
                        tile_data[ iy * ext_nx.x + ix ].y = in_y[ in_idx ] * scale;
                        tile_data[ iy * ext_nx.x + ix ].z = in_z[ in_idx ] * scale;
                    }
                }
            }
        }

        return dims.x * dims.y * 3;
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
    void save( const enum fcomp::cart fc, zdf::grid_info &metadata, zdf::iteration &iter, std::string path ) {

        // Fill in grid dimensions
        metadata.ndims = 2;
        metadata.count[0] = dims.x;
        metadata.count[1] = dims.y;

        // Allocate buffer on host to gather data
        S * h_data = memory::malloc<S>( metadata.count[0] * metadata.count[1] );

        gather( fc, h_data );

        // Save data
        zdf::save_grid( h_data, metadata, iter, path );

        memory::free( h_data );
    }

    /**
     * @brief Save grid values to disk
     * 
     * @param fc        Field component to save
     * @param filename  Output file name (includes path)
     */
    void save( const enum fcomp::cart fc, std::string filename ) {
        
        const std::size_t bsize = dims.x * dims.y;

        // Allocate buffers on host and device to gather data
        S * h_data = memory::malloc<S>( bsize );

        // Gather data on contiguous grid
        gather( fc, h_data );

        // Save data
        uint64_t grid_dims[] = {dims.x, dims.y};
        std::string fcomp_name[] = {"x","y","z"};
        std::string name_fc = name + "-" + fcomp_name[fc];
        zdf::save_grid( h_data, 2, grid_dims, name_fc, filename );

        // Free remaining temporary buffer 
        memory::free( h_data );        
    }

};

#endif