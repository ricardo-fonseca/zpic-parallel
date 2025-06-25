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
    using grid< V > :: nx;

    using grid< V > :: gc;

    using grid< V > :: ntiles;
    using grid< V > :: gnx;
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

                                auto const out_idx = giy * gnx.x + gix;

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

                                auto const out_idx = giy * gnx.x + gix;

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

                                auto const out_idx = giy * gnx.x + gix;

                                out[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].z;
                            }
                        }
                    }
                }
                break;
        }

        return gnx.x * gnx.y;
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
        metadata.count[0] = ntiles.x * nx.x;
        metadata.count[1] = ntiles.y * nx.y;

        // Allocate buffer on host to gather data
        S * h_data = memory::malloc<S>( metadata.count[0] * metadata.count[1] );

        gather( fc, h_data );
        zdf::save_grid( h_data, metadata, iter, path );

        memory::free( h_data );
    }

};

#endif
