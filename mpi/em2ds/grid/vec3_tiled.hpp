#pragma once

#include "tiled.hpp"
#include "../vec_types.hpp"

/**
 * @brief Field components (x,y,z)
 * 
 */
namespace fcomp {
    enum cart  { x = 0, y, z };
}

namespace grid {

template < typename S > 
class vec3_tiled : public grid::tiled< vec3<S> >
{
    protected:

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

    // Is this necessary?
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

        if ( stride.x == 1 ) {
            // Optimized versions for x stride == 1
            switch( fc ) {
                case( fcomp::x ):
                #pragma omp parallel for collapse(2)
                for( unsigned ty = 0; ty < local_ntiles.y; ty ++ ) {
                    for( unsigned tx = 0; tx < local_ntiles.x; tx ++ ) {
                        V * const __restrict__ data = tile_data(tx,ty);

                        const auto gix0 = tx * tile_dims.x;
                        const auto giy0 = ty * tile_dims.y;

                        // Loop inside tile
                        for( unsigned iy = 0; iy < tile_dims.y; iy ++ ) {
                            for( unsigned ix = 0; ix < tile_dims.x; ix ++ ) {
                                out[ (giy0 + iy) * stride.y + (gix0 + ix) ] = data[ iy * tile_ext_dims.x + ix ].x;
                            }
                        }
                    }
                }
                break;

                case( fcomp::y ):
                #pragma omp parallel for collapse(2)
                for( unsigned ty = 0; ty < local_ntiles.y; ty ++ ) {
                    for( unsigned tx = 0; tx < local_ntiles.x; tx ++ ) {
                        V * const __restrict__ data = tile_data(tx,ty);

                        const auto gix0 = tx * tile_dims.x;
                        const auto giy0 = ty * tile_dims.y;

                        // Loop inside tile
                        for( unsigned iy = 0; iy < tile_dims.y; iy ++ ) {
                            for( unsigned ix = 0; ix < tile_dims.x; ix ++ ) {
                                out[ (giy0 + iy) * stride.y + (gix0 + ix) ] = data[ iy * tile_ext_dims.x + ix ].y;
                            }
                        }
                    }
                }
                break;

                case( fcomp::z ):
                #pragma omp parallel for collapse(2)
                for( unsigned ty = 0; ty < local_ntiles.y; ty ++ ) {
                    for( unsigned tx = 0; tx < local_ntiles.x; tx ++ ) {
                        V * const __restrict__ data = tile_data(tx,ty);

                        const auto gix0 = tx * tile_dims.x;
                        const auto giy0 = ty * tile_dims.y;

                        // Loop inside tile
                        for( unsigned iy = 0; iy < tile_dims.y; iy ++ ) {
                            for( unsigned ix = 0; ix < tile_dims.x; ix ++ ) {
                                out[ (giy0 + iy) * stride.y + (gix0 + ix) ] = data[ iy * tile_ext_dims.x + ix ].z;
                            }
                        }
                    }
                }
                break;
            }

        } else {
            // Arbitrary x stride
            switch( fc ) {
                case( fcomp::x ):
                #pragma omp parallel for collapse(2)
                for( unsigned ty = 0; ty < local_ntiles.y; ty ++ ) {
                    for( unsigned tx = 0; tx < local_ntiles.x; tx ++ ) {
                        V * const __restrict__ data = tile_data(tx,ty);

                        const auto gix0 = tx * tile_dims.x;
                        const auto giy0 = ty * tile_dims.y;

                        // Loop inside tile
                        for( unsigned iy = 0; iy < tile_dims.y; iy ++ ) {
                            for( unsigned ix = 0; ix < tile_dims.x; ix ++ ) {
                                out[ (giy0 + iy) * stride.y + (gix0 + ix) * stride.x ] = data[ iy * tile_ext_dims.x + ix ].x;
                            }
                        }
                    }
                }
                break;

                case( fcomp::y ):
                #pragma omp parallel for collapse(2)
                for( unsigned ty = 0; ty < local_ntiles.y; ty ++ ) {
                    for( unsigned tx = 0; tx < local_ntiles.x; tx ++ ) {
                        V * const __restrict__ data = tile_data(tx,ty);

                        const auto gix0 = tx * tile_dims.x;
                        const auto giy0 = ty * tile_dims.y;

                        // Loop inside tile
                        for( unsigned iy = 0; iy < tile_dims.y; iy ++ ) {
                            for( unsigned ix = 0; ix < tile_dims.x; ix ++ ) {
                                out[ (giy0 + iy) * stride.y + (gix0 + ix) * stride.x ] = data[ iy * tile_ext_dims.x + ix ].y;
                            }
                        }
                    }
                }
                break;

                case( fcomp::z ):
                #pragma omp parallel for collapse(2)
                for( unsigned ty = 0; ty < local_ntiles.y; ty ++ ) {
                    for( unsigned tx = 0; tx < local_ntiles.x; tx ++ ) {
                        V * const __restrict__ data = tile_data(tx,ty);

                        const auto gix0 = tx * tile_dims.x;
                        const auto giy0 = ty * tile_dims.y;

                        // Loop inside tile
                        for( unsigned iy = 0; iy < tile_dims.y; iy ++ ) {
                            for( unsigned ix = 0; ix < tile_dims.x; ix ++ ) {
                                out[ (giy0 + iy) * stride.y + (gix0 + ix) * stride.x ] = data[ iy * tile_ext_dims.x + ix ].z;
                            }
                        }
                    }
                }
                break;
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

        if ( stride.x == 1 ) {
            // Optimized version for x stride == 1
            #pragma omp parallel for collapse(2)
            for( unsigned ty = 0; ty < local_ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < local_ntiles.x; tx ++ ) {
                    V * const __restrict__ data = tile_data(tx,ty);

                    const auto gix0 = tx * tile_dims.x;
                    const auto giy0 = ty * tile_dims.y;

                    // Loop inside tile
                    for( unsigned iy = 0; iy < tile_dims.y; iy ++ ) {
                        for( unsigned ix = 0; ix < tile_dims.x; ix ++ ) {
                            V val = data[ iy * tile_ext_dims.x + ix ];
                            out_x[ (giy0 + iy) * stride.y + (gix0 + ix) ] = val.x;
                            out_y[ (giy0 + iy) * stride.y + (gix0 + ix) ] = val.y;
                            out_z[ (giy0 + iy) * stride.y + (gix0 + ix) ] = val.z;
                        }
                    }
                }
            }
        } else {
            // Arbitrary x stride
            #pragma omp parallel for collapse(2)
            for( unsigned ty = 0; ty < local_ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < local_ntiles.x; tx ++ ) {
                    V * const __restrict__ data = tile_data(tx,ty);

                    const auto gix0 = tx * tile_dims.x;
                    const auto giy0 = ty * tile_dims.y;

                    // Loop inside tile
                    for( unsigned iy = 0; iy < tile_dims.y; iy ++ ) {
                        for( unsigned ix = 0; ix < tile_dims.x; ix ++ ) {
                            V val = data[ iy * tile_ext_dims.x + ix ];
                            out_x[ (giy0 + iy) * stride.y + (gix0 + ix) * stride.x ] = val.x;
                            out_y[ (giy0 + iy) * stride.y + (gix0 + ix) * stride.x ] = val.y;
                            out_z[ (giy0 + iy) * stride.y + (gix0 + ix) * stride.x ] = val.z;
                        }
                    }
                }
            }
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

        if ( stride.x == 1 ) {
            // Optimized version for x stride == 1
            #pragma omp parallel for collapse(2)
            for( unsigned ty = 0; ty < local_ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < local_ntiles.x; tx ++ ) {
                    V * const __restrict__ data = tile_data(tx,ty);

                    const auto gix0 = tx * tile_dims.x;
                    const auto giy0 = ty * tile_dims.y;

                    // Loop inside tile
                    for( unsigned iy = 0; iy < tile_dims.y; iy ++ ) {
                        for( unsigned ix = 0; ix < tile_dims.x; ix ++ ) {
                            V val;
                            val.x = in_x[ (giy0 + iy) * stride.y + (gix0 + ix) ] * scale;
                            val.y = in_y[ (giy0 + iy) * stride.y + (gix0 + ix) ] * scale;
                            val.z = in_z[ (giy0 + iy) * stride.y + (gix0 + ix) ] * scale;
                            data[ iy * tile_ext_dims.x + ix ] = val;
                        }
                    }
                }
            }
        } else {
            // Arbitrary x stride
            #pragma omp parallel for collapse(2)
            for( unsigned ty = 0; ty < local_ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < local_ntiles.x; tx ++ ) {
                    V * const __restrict__ data = tile_data(tx,ty);

                    const auto gix0 = tx * tile_dims.x;
                    const auto giy0 = ty * tile_dims.y;

                    // Loop inside tile
                    for( unsigned iy = 0; iy < tile_dims.y; iy ++ ) {
                        for( unsigned ix = 0; ix < tile_dims.x; ix ++ ) {
                            V val;
                            val.x = in_x[ (giy0 + iy) * stride.y + (gix0 + ix) * stride.x ] * scale;
                            val.y = in_y[ (giy0 + iy) * stride.y + (gix0 + ix) * stride.x ] * scale;
                            val.z = in_z[ (giy0 + iy) * stride.y + (gix0 + ix) * stride.x ] * scale;
                            data[ iy * tile_ext_dims.x + ix ] = val;

                        }
                    }
                }
            }
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
    void save( const enum fcomp::cart fc, zdf::grid_info &metadata, zdf::iteration &iter, const std::string & path ) {

        // Fill in grid dimensions
        metadata.ndims = 2;
        metadata.count[0] = global_ntiles.x * tile_dims.x;
        metadata.count[1] = global_ntiles.y * tile_dims.y;

        // Allocate buffer on host to gather data
        S2 * h_data = memory::malloc<S2>( metadata.count[0] * metadata.count[1] );

        gather( fc, h_data );

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

        memory::free( h_data );
    }

    template< typename S2 = S >
    void save( const enum fcomp::cart fc, const std::string & filename ) {
        
        const std::size_t bsize = local_dims.x * local_dims.y;

        // Allocate buffers on host and device to gather data
        S2 * h_data = memory::malloc<S2>( bsize );

        // Gather data on contiguous grid
        gather( fc, h_data );

        uint64_t global[2] = { global_ntiles.x * tile_dims.x, global_ntiles.y * tile_dims.y };
        uint64_t start[2]  = { local_tile_start.x * tile_dims.x, local_tile_start.y * tile_dims.y };
        uint64_t local[2]  = { local_dims.x, local_dims.y };

        // Save data
        std::string comp[] = { "x", "y", "z" };

        zdf::save_grid( h_data, 2, global, start, local, name + "-" + comp[fc], filename, part.get_comm() );

        // Free remaining temporary buffer 
        memory::free( h_data );
    }
};

}
