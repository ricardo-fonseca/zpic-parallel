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

    using grid< V > :: q;

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
    unsigned int gather( const enum fcomp::cart fc, S * const __restrict__ d_out ) {

        q.submit([&](sycl::handler &h) {

            const auto ntiles   = this->ntiles;
            const auto tile_vol = this->tile_vol;
            const auto nx       = this->nx;
            const auto gnx      = this->gnx;
            const auto ext_nx   = this->ext_nx;
            const auto offset   = this->offset;
            auto * __restrict__ d_buffer = this->d_buffer;

            // 8×1 work items per group
            sycl::range<2> local{ 8, 1 };

            // ntiles.x × ntiles.y groups
            sycl::range<2> global{ ntiles.x, ntiles.y };

            switch( fc ) {
            case( fcomp::x ):
                h.parallel_for( 
                    sycl::nd_range{global * local , local},
                    [=](sycl::nd_item<2> it) { 

                    const auto tile_idx = make_uint2( it.get_group(0), it.get_group(1) );
                    const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                    const auto tile_off = tid * tile_vol + offset;

                    auto * const __restrict__ tile_data = & d_buffer[ tile_off ];

                    // Loop inside tile
                    for( unsigned idx = it.get_local_id(0); idx < nx.y * nx.x; idx += it.get_local_range(0) ) {
                        auto const iy = idx / nx.x;
                        auto const ix = idx % nx.x;
                        auto const gix = tile_idx.x * nx.x + ix;
                        auto const giy = tile_idx.y * nx.y + iy;
                        auto const out_idx = giy * gnx.x + gix;
                        d_out[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].x;
                    }

                });
                break;
            case( fcomp::y ):
                h.parallel_for( 
                    sycl::nd_range{global * local , local},
                    [=](sycl::nd_item<2> it) { 

                    const auto tile_idx = make_uint2( it.get_group(0), it.get_group(1) );
                    const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                    const auto tile_off = tid * tile_vol + offset;

                    auto * const __restrict__ tile_data = & d_buffer[ tile_off ];

                    // Loop inside tile
                    for( unsigned idx = it.get_local_id(0); idx < nx.y * nx.x; idx += it.get_local_range(0) ) {
                        auto const iy = idx / nx.x;
                        auto const ix = idx % nx.x;
                        auto const gix = tile_idx.x * nx.x + ix;
                        auto const giy = tile_idx.y * nx.y + iy;
                        auto const out_idx = giy * gnx.x + gix;
                        d_out[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].y;
                    }

                });
                break;
            case( fcomp::z ):
                h.parallel_for( 
                    sycl::nd_range{global * local , local},
                    [=](sycl::nd_item<2> it) { 

                    const auto tile_idx = make_uint2( it.get_group(0), it.get_group(1) );
                    const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                    const auto tile_off = tid * tile_vol + offset;

                    auto * const __restrict__ tile_data = & d_buffer[ tile_off ];

                    // Loop inside tile
                    for( unsigned idx = it.get_local_id(0); idx < nx.y * nx.x; idx += it.get_local_range(0) ) {
                        auto const iy = idx / nx.x;
                        auto const ix = idx % nx.x;
                        auto const gix = tile_idx.x * nx.x + ix;
                        auto const giy = tile_idx.y * nx.y + iy;
                        auto const out_idx = giy * gnx.x + gix;
                        d_out[ out_idx ] = tile_data[ iy * ext_nx.x + ix ].z;
                    }

                });
                break;
            };
        });
        q.wait();

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
    void save( const enum fcomp::cart fc, zdf::grid_info &info, zdf::iteration &iter, std::string path ) {

        // Fill in grid dimensions
        info.ndims = 2;
        info.count[0] = gnx.x;
        info.count[1] = gnx.y;

        const std::size_t bsize = gnx.x * gnx.y;

        S * d_data = device::malloc<S>( bsize, q );
        S * h_data = host::malloc<S>( bsize, q );

        gather( fc, d_data );

        device::memcpy_tohost( h_data, d_data, bsize, q );

        zdf::save_grid( h_data, info, iter, path );

        host::free( h_data, q );
        device::free( d_data, q );
    }

    void save( const enum fcomp::cart fc, std::string path ) {
        
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

        std::string fcomp_name[] = {"x","y","z"};

        std::string grid_name = "sycl-vec3-" + fcomp_name[fc];
        std::string grid_label = "sycl vec3 test (" + fcomp_name[fc] + ")";

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

        save( fc, info, iter, path );
    }

};

#endif