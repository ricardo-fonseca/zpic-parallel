#pragma once

#include <cassert>
#include <ostream>
#include <string>
#include <vector>

#include "core/bounds.hpp"
#include "parallel/partition.hpp"
#include "util/complex.hpp"

#include "grid/tiled.cuh"

namespace cyl_modes {

/**
 * @brief Tiled grid of cylindrical modes
 *
 * Mode 0 is stored as a real grid, Grid<T>, and modes 1 ... nmodes-1 as
 * complex grids, Grid< util::complex<T> >.
 *
 * @note The object is move-only (the mode grids cannot be copied)
 *
 * @note Grid x is the z (axial) direction and grid y is the r (radial)
 *       direction. The parallel partition must not be periodic along r.
 *
 * @note This class does not apply any axis (r = 0) boundary condition: the
 *       lower r guard cells on the node holding the axis are left untouched
 *       by all guard cell operations. These are handled by the physics
 *       classes (charge, current, emf).
 *
 * @tparam T        Base type (float or double)
 * @tparam Grid     Mode grid template, defaults to grid::tiled
 */
template < class T, template< class > class Grid = grid::tiled >
class tiled {

    public:

    /// @brief Mode 0 (real) grid type
    using grid0_type = Grid< T >;

    /// @brief Mode m > 0 (complex) grid type
    using gridm_type = Grid< util::complex<T> >;

    /// @brief Number of cylindrical modes (including fundamental mode)
    const int nmodes;

    protected:

    /// @brief Object name
    std::string name;

    /// @brief Mode zero (real)
    grid0_type grid_0;

    /// @brief High order modes (complex), grid_m[m-1] holds mode m
    std::vector< gridm_type > grid_m;

    /**
     * @brief Check that the mode index is valid, aborts otherwise
     *
     * @param m     Mode index (0 <= m < nmodes)
     */
    void check_mode( const int m ) const {
        if ( m < 0 || m >= nmodes )
            mpi::fatal( "invalid mode (" + std::to_string(m) + ") selected, must be in [0," +
                        std::to_string(nmodes) + "[" );
    }

    private:

    /**
     * @brief Validate the number of modes (used in the member initializer list)
     *
     */
    static int validate_nmodes( const int nmodes ) {
        if ( nmodes < 1 )
            mpi::fatal( "Invalid number of cylindrical modes (" + std::to_string(nmodes) + "), must be >= 1" );
        return nmodes;
    }

    /**
     * @brief Validate the parallel partition (used in the member initializer list)
     *
     */
    static const mpi::cart2d & validate_part( const mpi::cart2d & part ) {
        if ( part.periodic.y )
            mpi::fatal( "Parallel partition must not be periodic along r" );
        return part;
    }

    public:

    /**
     * @brief Construct a new cyl_modes::tiled object
     *
     * @note Parameters are validated before any grid is allocated
     *
     * @param nmodes            Number of modes (including fundamental mode), must be >= 1
     * @param global_ntiles     Global number of tiles (z,r)
     * @param tile_dims         Individual tile size (z,r)
     * @param gc                Number of guard cells
     * @param part              Parallel partition, must not be periodic along r
     */
    tiled( const int nmodes, uint2 const global_ntiles, uint2 const tile_dims,
        bounds_2d<unsigned int> const gc, const mpi::cart2d & part ) :
        nmodes( validate_nmodes( nmodes ) ),
        name( "cyl modes grid" ),
        grid_0( global_ntiles, tile_dims, gc, validate_part( part ) )
    {
        grid_m.reserve( nmodes - 1 );
        for ( int m = 1; m < nmodes; m++ )
            grid_m.emplace_back( global_ntiles, tile_dims, gc, part );

        set_name( name );
    }

    // Move-only: copy is implicitly deleted (grid copy constructors are
    // deleted), move assignment is implicitly deleted (const members)
    tiled( tiled && ) = default;

    /**
     * @brief Set the object name, mode grids are named <name>-m<m>
     *
     * @param new_name  New name
     */
    void set_name( const std::string & new_name ) {
        name = new_name;
        grid_0.name = name + "-m0";
        for ( int m = 1; m < nmodes; m++ )
            grid_m[m-1].name = name + "-m" + std::to_string(m);
    }

    /**
     * @brief Get the object name
     *
     * @return const std::string&
     */
    const std::string & get_name() const noexcept { return name; }

    /**
     * @brief Get number of cylindrical modes
     *
     * @return int
     */
    int get_nmodes() const noexcept { return nmodes; }

    /**
     * @brief Get the local number of tiles
     *
     * @return uint2
     */
    uint2 get_local_ntiles() const noexcept { return grid_0.get_local_ntiles(); }

    /**
     * @brief Get the local tile offset in the global MPI tile grid
     *
     * @return uint2
     */
    uint2 get_local_tile_start() const noexcept { return grid_0.get_local_tile_start(); }

    /**
     * @brief Get the local grid size (all local tiles)
     *
     * @return uint2
     */
    uint2 get_local_dims() const noexcept { return grid_0.get_local_dims(); }

    /**
     * @brief Get the global grid size
     *
     * @return uint2
     */
    uint2 get_global_dims() const noexcept { return grid_0.get_global_dims(); }

    /**
     * @brief Get the tile dimensions
     *
     * @return uint2
     */
    uint2 get_tile_dims() const noexcept { return grid_0.tile_dims; }

    /**
     * @brief Get the parallel partition
     *
     * @return const mpi::cart2d&
     */
    const mpi::cart2d & get_part() const noexcept { return grid_0.get_part(); }

    /**
     * @brief Stream extraction
     *
     * @param os
     * @param obj
     * @return std::ostream&
     */
    friend std::ostream& operator<<( std::ostream& os, const tiled & obj ) {
        const auto ntiles    = obj.get_local_ntiles();
        const auto tile_dims = obj.get_tile_dims();
        os << obj.name << '{'
           << ' ' << obj.nmodes << " modes"
           << ", " << ntiles.x << "×" << ntiles.y << " local tiles"
           << ", " << tile_dims.x << "×" << tile_dims.y << " points/tile "
           << '}';
        return os;
    }

    /**
     * @brief Access mode 0
     *
     * @return grid0_type&  Mode 0 grid
     */
    grid0_type & mode0() noexcept { return grid_0; }
    const grid0_type & mode0() const noexcept { return grid_0; }

    /**
     * @brief Access mode m
     *
     * @param m             Mode index, must be in [1, nmodes[
     * @return gridm_type&  Mode m grid
     */
    gridm_type & mode( const int m ) {
        assert(( m >= 1 && m < nmodes ));
        return grid_m[m-1];
    }

    const gridm_type & mode( const int m ) const {
        assert(( m >= 1 && m < nmodes ));
        return grid_m[m-1];
    }

    /**
     * @brief Sets all values to 0
     *
     */
    void zero() {
        grid_0.zero();
        for ( auto & g : grid_m ) g.zero();
    }

    /**
     * @brief Add another cyl_modes::tiled object to this object
     *
     * @param rhs   Object to add, must have the same number of modes and
     *              grid layout
     */
    void add( const tiled & rhs ) {
        assert(( nmodes == rhs.nmodes ));
        grid_0.add( rhs.grid_0 );
        for ( int m = 1; m < nmodes; m++ )
            grid_m[m-1].add( rhs.grid_m[m-1] );
    }

    /**
     * @brief Operator +=
     *
     * @param rhs   Object to add
     * @return tiled&
     */
    tiled & operator+=( const tiled & rhs ) {
        add( rhs );
        return *this;
    }

    /**
     * @brief Copies edge values to z (x) neighboring guard cells
     *
     * @warning Collective operation, must be called by all parallel nodes.
     */
    void copy_to_gc_x() {
        grid_0.copy_to_gc_x();
        for ( auto & g : grid_m ) g.copy_to_gc_x();
    }

    /**
     * @brief Copies edge values to r (y) neighboring guard cells
     *
     * @note Axis guard cells are left untouched
     *
     * @warning Collective operation, must be called by all parallel nodes.
     */
    void copy_to_gc_y() {
        grid_0.copy_to_gc_y();
        for ( auto & g : grid_m ) g.copy_to_gc_y();
    }

    /**
     * @brief Copies edge values to neighboring guard cells
     *
     * @note Axis guard cells are left untouched
     *
     * @warning Collective operation, must be called by all parallel nodes.
     */
    void copy_to_gc() {
        grid_0.copy_to_gc();
        for ( auto & g : grid_m ) g.copy_to_gc();
    }

    /**
     * @brief Adds values from neighboring guard cells to local data
     *
     * @note Values in the axis guard cells are not folded back
     *
     * @warning Collective operation, must be called by all parallel nodes.
     */
    void add_from_gc() {
        grid_0.add_from_gc();
        for ( auto & g : grid_m ) g.add_from_gc();
    }

    /**
     * @brief Left shifts data for a specified amount
     *
     * @note This operation is only allowed if the number of upper x guard cells
     *       is greater or equal to the requested shift
     *
     * @param shift Number of cells to shift
     */
    void x_shift_left( unsigned int const shift ) {
        grid_0.x_shift_left( shift );
        for ( auto & g : grid_m ) g.x_shift_left( shift );
    }

    /**
     * @brief Perform a convolution with a 3 point kernel [a,b,c] along x
     *
     * @param a             Kernel value a
     * @param b             Kernel value b
     * @param c             Kernel value c
     * @param upper_ext     (optional) see grid::tiled::kernel3_x
     */
    template < typename S >
    void kernel3_x( S const a, S const b, S const c, bool const upper_ext = false ) {
        grid_0.kernel3_x( a, b, c, upper_ext );
        for ( auto & g : grid_m ) g.kernel3_x( a, b, c, upper_ext );
    }

    /**
     * @brief Perform a convolution with a 3 point kernel [a,b,c] along y
     *
     * @note Uses the lower r guard cells on the axis node, which this class
     *       does not set; the caller must fill them first
     *
     * @param a             Kernel value a
     * @param b             Kernel value b
     * @param c             Kernel value c
     * @param upper_ext     (optional) see grid::tiled::kernel3_y
     */
    template < typename S >
    void kernel3_y( S const a, S const b, S const c, bool const upper_ext = false ) {
        grid_0.kernel3_y( a, b, c, upper_ext );
        for ( auto & g : grid_m ) g.kernel3_y( a, b, c, upper_ext );
    }

    /**
     * @brief Perform a convolution with a 3 point kernel [a,b,c] along x and y
     *
     * @note Uses the lower r guard cells on the axis node, which this class
     *       does not set; the caller must fill them first
     *
     * @param a             Kernel value a
     * @param b             Kernel value b
     * @param c             Kernel value c
     * @param upper_ext     (optional) see grid::tiled::kernel3_xy
     */
    template < typename S >
    void kernel3_xy( S const a, S const b, S const c, bool const upper_ext = false ) {
        grid_0.kernel3_xy( a, b, c, upper_ext );
        for ( auto & g : grid_m ) g.kernel3_xy( a, b, c, upper_ext );
    }

    /**
     * @brief Save mode m grid values to disk
     *
     * @note Mode 0 is saved as real data, modes m > 0 as complex data
     *
     * @param m             Cyl. mode (0 <= m < nmodes)
     * @param filename      Output file name (includes path)
     */
    void save( const int m, const std::string & filename ) {
        check_mode( m );
        if ( m == 0 ) grid_0.save( filename );
        else          grid_m[m-1].save( filename );
    }

    /**
     * @brief Save mode m grid values to disk with full metadata
     *
     * @param m             Cyl. mode (0 <= m < nmodes)
     * @param info          Grid metadata
     * @param iter          Iteration metadata
     * @param path          File path
     */
    void save( const int m, zdf::grid_info & info, const zdf::iteration & iter, const std::string & path ) {
        check_mode( m );
        if ( m == 0 ) grid_0.save( info, iter, path );
        else          grid_m[m-1].save( info, iter, path );
    }
};

} // namespace cyl_modes
