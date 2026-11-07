#pragma once

#include <functional>

#include "cyl_modes/tiled_cyl3.cuh"

/**
 * @brief Digital filters for grid quantities (typically the current density)
 *
 * All filters are a sequence of 3-point kernel passes, applied with the
 * grid's kernel3_x() (z), kernel3_y() (r) and kernel3_xy() (z and r)
 * routines. Filtering along both z and r uses kernel3_xy(), a separable 3x3
 * kernel [a,b,c] ⊗ [a,b,c] applied with a single kernel launch and guard
 * cell update per pass.
 *
 * A filter is a plain value (copyable, no virtual functions), so the same
 * filter object can be applied to any grid type providing the kernel3_*
 * routines: grid::tiled<T>, grid::tiled_cyl3<S> for real or complex S, and
 * the cyl_modes classes, which filter every mode.
 *
 *      filter::digital f = filter::compensated( filter::axis::zr, 2 );
 *      f.apply( J );                       // J: grid::tiled_cyl3<float>
 *      f.apply( J_modes, false, fix_axis );
 *
 * @note The kernel3_* routines read the i±1 (and, for kernel3_xy, corner)
 *       neighbours and update guard cells on exit with copy_to_gc(), so
 *       consecutive passes are consistent across tile and node boundaries.
 *       copy_to_gc() does not touch guard cells on non-periodic physical
 *       boundaries, including the cylindrical axis; if those hold
 *       boundary-condition values (e.g. mirrored values at a reflecting
 *       wall, or across r = 0) pass a `post_pass` hook to apply() that
 *       refreshes them after every pass.
 *
 * @note Guard cells must be valid on entry: for the current density this
 *       means calling copy_to_gc() after add_from_gc(), and applying any
 *       physical boundary conditions, before applying the filter.
 *
 * @note When the grid has enough upper guard cells (see upper_ext) every
 *       pass is also computed on the first upper guard column / row. On a
 *       staggered grid the upper wall node of a PEC / PMC boundary sits on
 *       the first upper guard cell, and must be filtered like any interior
 *       node. On internal tile / node boundaries these values are simply
 *       overwritten by the guard cell update.
 */
namespace filter {

/// @brief Filtering direction(s)
enum class axis : unsigned int {
    z  = 1,     ///< Filter along z only
    r  = 2,     ///< Filter along r only
    zr = 3      ///< Filter along both z and r (separable 3x3 kernel)
};

namespace detail {

/// @brief Check if d is a valid filter axis
inline bool valid( const axis d ) {
    return d == axis::z || d == axis::r || d == axis::zr;
}

} // namespace detail

/**
 * @brief Digital filter, a sequence of 3-point kernel passes
 *
 * A default constructed filter does nothing (see filter::none). The
 * binomial and compensated classes only set the passes in their
 * constructors and add no data, so they can be freely converted to (stored
 * as) filter::digital.
 */
class digital {
    public:

    /**
     * @brief Optional hook called after every convolution pass
     *
     * Called after the grid's own guard cell update, typically to refresh
     * guard cells on physical boundaries.
     */
    using hook = std::function<void()>;

    /// @brief 3-point kernel [a,b,c]
    struct kernel3 {
        float a, b, c;
    };

    protected:

    /// @brief Filter direction(s)
    axis dirs = axis::zr;

    /// @brief Kernel passes, applied in order
    std::vector< kernel3 > passes;

    /// @brief Apply a 3-point kernel along all selected directions
    template < class G >
    void apply_kernel3( G & fld, const kernel3 & k, const bool upper_ext ) const {
        switch( dirs ) {
        case( axis::z ):  fld.kernel3_x ( k.a, k.b, k.c, upper_ext ); break;
        case( axis::r ):  fld.kernel3_y ( k.a, k.b, k.c, upper_ext ); break;
        case( axis::zr ): fld.kernel3_xy( k.a, k.b, k.c, upper_ext ); break;
        }
    }

    public:

    /**
     * @brief Construct an empty filter (no filtering)
     *
     */
    digital() = default;

    /**
     * @brief Construct a filter from a sequence of kernel passes
     *
     * @param dirs      Filter direction(s): axis::z, axis::r or axis::zr
     * @param passes    Kernel passes, applied in order
     */
    digital( const axis dirs, std::vector< kernel3 > passes ) :
        dirs( dirs ), passes( std::move( passes ) )
    {
        if ( ! detail::valid( dirs ) )
            mpi::fatal( "filter: invalid filter direction" );
    }

    /**
     * @brief Apply the filter
     *
     * @note The post_pass hook is the third parameter, after upper_ext, so
     *       that a lambda can never be mistaken for the bool argument
     *
     * @tparam G            Grid type, must provide kernel3_x(), kernel3_y()
     *                      and kernel3_xy()
     * @param fld           Grid to filter, guard cells must be valid on entry
     * @param upper_ext     (optional) Also process the first upper guard
     *                      column / row, defaults to false
     * @param post_pass     (optional) Called after every convolution pass
     */
    template < class G >
    void apply( G & fld, const bool upper_ext = false, const hook & post_pass = hook{} ) const {
        for( const auto & k : passes ) {
            apply_kernel3( fld, k, upper_ext );
            if ( post_pass ) post_pass();
        }
    }

    /**
     * @brief Returns true if the filter does nothing
     *
     */
    bool empty() const noexcept { return passes.empty(); }

    /**
     * @brief Get the filter direction(s)
     *
     */
    axis get_dirs() const noexcept { return dirs; }

    /**
     * @brief Get the kernel passes
     *
     */
    const std::vector< kernel3 > & get_passes() const noexcept { return passes; }
};

/**
 * @brief No filtering
 */
class none final : public digital {
    public:
    none() = default;
};

/**
 * @brief Binomial filter, (1/4, 1/2, 1/4) kernel applied `order` times
 *
 * Transfer function (per filtered direction) is cos^(2n)(k Δ / 2), with
 * n = order.
 */
class binomial : public digital {

    protected:

    /// @brief Binomial passes
    static std::vector< kernel3 > binomial_passes( const unsigned int order ) {
        return std::vector< kernel3 >( order, kernel3{ 0.25f, 0.5f, 0.25f } );
    }

    /// @brief Construct from explicit passes (used by derived filters)
    binomial( const axis dirs, std::vector< kernel3 > passes ) :
        digital( dirs, std::move( passes ) ) { }

    public:

    /**
     * @brief Construct a new binomial filter
     *
     * @param dirs     Filter direction(s): axis::z, axis::r or axis::zr
     * @param order    Filter order (number of passes), 0 is promoted to 1
     */
    binomial( const axis dirs, const unsigned int order = 1 ) :
        digital( dirs, binomial_passes( ( order > 0 ) ? order : 1 ) ) { }
};

/**
 * @brief Compensated binomial filter
 *
 * Applies a binomial filter of order n followed by a compensator kernel
 * (-n/4, 1 + n/2, -n/4) that cancels the O(k^2) attenuation of the binomial
 * passes. Transfer function (per filtered direction) is
 *
 *     cos^(2n)(k Δ / 2) * ( 1 + n sin^2(k Δ / 2) )
 *
 * which is flat to O(k^4), monotonically decreasing, and never exceeds 1.
 */
class compensated final : public binomial {

    static std::vector< kernel3 > compensated_passes( unsigned int order ) {
        if ( order == 0 ) order = 1;
        auto p = binomial_passes( order );

        // Compensator, normalized kernel (a, b, a) with 2a + b = 1
        const float a = -0.25f * order;
        const float b =  1.0f  + 0.5f * order;
        p.push_back( kernel3{ a, b, a } );
        return p;
    }

    public:

    /**
     * @brief Construct a new compensated binomial filter
     *
     * @param dirs     Filter direction(s): axis::z, axis::r or axis::zr
     * @param order    Binomial order (number of binomial passes), 0 is promoted to 1
     */
    compensated( const axis dirs, const unsigned int order = 1 ) :
        binomial( dirs, compensated_passes( order ) ) { }
};

} // namespace filter
