#pragma once

#include <string>

#include "../cyl_modes/tiled.cuh"
#include "../cyl3/tiled_cyl3.cuh"

namespace cyl_modes {

/**
 * @brief Tiled grid of cylindrical modes of a vector field (z,r,θ)
 *
 * Mode 0 is stored as a grid::tiled_cyl3<T> and modes 1 ... nmodes-1 as
 * grid::tiled_cyl3< util::complex<T> >.
 *
 * @note Inside namespace cyl_modes an unqualified tiled_cyl3 refers to this
 *       class; always use grid::tiled_cyl3 for the grid type.
 *
 * @tparam T    Base type (float or double)
 */
template < class T >
class tiled_cyl3 : public cyl_modes::tiled< T, grid::tiled_cyl3 > {

    using base = cyl_modes::tiled< T, grid::tiled_cyl3 >;

    protected:

    using base::grid_0;
    using base::grid_m;
    using base::check_mode;

    public:

    using typename base::grid0_type;
    using typename base::gridm_type;
    using base::nmodes;

    /**
     * @brief Construct a new cyl_modes::tiled_cyl3 object
     *
     * @param nmodes            Number of modes (including fundamental mode), must be >= 1
     * @param global_ntiles     Global number of tiles (z,r)
     * @param tile_dims         Individual tile size (z,r)
     * @param gc                Number of guard cells
     * @param part              Parallel partition, must not be periodic along r
     */
    using base::base;

    /**
     * @brief Save a field component of mode m to disk
     *
     * @note Mode 0 is saved as real data, modes m > 0 as complex data. The
     *       dataset is named <name>-m<m>-<component>
     *
     * @param m             Cyl. mode (0 <= m < nmodes)
     * @param fc            Field component
     * @param filename      Output file name (includes path)
     */
    void save( const int m, const fcomp::cyl fc, const std::string & filename ) {
        check_mode( m );
        if ( m == 0 ) grid_0.save( fc, filename );
        else          grid_m[m-1].save( fc, filename );
    }

    /**
     * @brief Save a field component of mode m to disk with full metadata
     *
     * @param m             Cyl. mode (0 <= m < nmodes)
     * @param fc            Field component
     * @param info          Grid metadata
     * @param iter          Iteration metadata
     * @param path          File path
     */
    void save( const int m, const fcomp::cyl fc, zdf::grid_info & info,
        const zdf::iteration & iter, const std::string & path ) {
        check_mode( m );
        if ( m == 0 ) grid_0.save( fc, info, iter, path );
        else          grid_m[m-1].save( fc, info, iter, path );
    }
};

} // namespace cyl_modes
