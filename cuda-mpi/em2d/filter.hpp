#pragma once

#include <memory>
#include <functional>

// Also provides coord::cart and mpi::fatal()
#include "grid/tiled_vec3.cuh"

/**
 * @brief Digital filters for grid quantities (typically the current density)
 *
 * All filters are built from 3-point kernels applied with the grid's
 * kernel3_x(), kernel3_y() and kernel3_xy() routines. Filtering along both
 * x and y uses kernel3_xy(), a separable 3x3 kernel [a,b,c] ⊗ [a,b,c]
 * applied with a single kernel launch and guard cell update per pass.
 *
 * @note The kernel3_* routines read the i±1 (and, for kernel3_xy, corner)
 *       neighbours and update guard cells on exit with copy_to_gc(), so
 *       consecutive passes are consistent across tile and node boundaries.
 *       copy_to_gc() does not touch guard cells on non-periodic physical
 *       boundaries; if those hold boundary-condition values (e.g. mirrored
 *       values at a reflecting wall) pass a `post_pass` hook to apply() that
 *       refreshes them after every pass.
 *
 * @note Guard cells must be valid on entry: for the current density this
 *       means calling copy_to_gc() after add_from_gc(), and applying any
 *       physical boundary conditions, before applying the filter.
 *
 * @note When the grid has enough upper guard cells (see upper_ext()) every
 *       pass is also computed on the first upper guard column / row. On a
 *       staggered grid the upper wall node of a PEC / PMC boundary sits on
 *       the first upper guard cell, and must be filtered like any interior
 *       node. On internal tile / node boundaries these values are simply
 *       overwritten by the guard cell update.
 */
namespace filter {

/// @brief Filtering direction(s)
enum class axis : unsigned int {
    x  = 1,     ///< Filter along x only
    y  = 2,     ///< Filter along y only
    xy = 3      ///< Filter along both x and y (separable 3x3 kernel)
};

namespace detail {

/// @brief Check if d is a valid filter axis
inline bool valid( axis d ) {
    return d == axis::x || d == axis::y || d == axis::xy;
}

} // namespace detail

/**
 * @brief Abstract digital filter
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

    virtual std::unique_ptr<digital> clone() const = 0;

    /**
     * @brief Apply the filter
     *
     * @param fld           Grid to filter, guard cells must be valid on entry
     * @param ext           Process additional external row/column
     * @param post_pass     (optional) Called after every convolution pass
     */
    void apply( grid::tiled_vec3<float> & fld, const bool ext, const hook & post_pass = hook{} ) const {
        do_apply( fld, ext, post_pass );
    }

    /**
     * @brief Apply the filter (interior only)
     *
     * @param fld           Grid to filter, guard cells must be valid on entry
     * @param post_pass     (optional) Called after every convolution pass
     */
    void apply( grid::tiled_vec3<float> & fld, const hook & post_pass = hook{} ) const {
        do_apply( fld, false, post_pass );
    }

    virtual ~digital() = default;

    protected:

    /**
     * @brief Filter implementation, overridden by derived classes
     *
     * @note Derived classes override this instead of apply(), so they never
     *       hide the apply() overloads
     */
    virtual void do_apply( grid::tiled_vec3<float> & fld, const bool ext, const hook & post_pass ) const = 0;
};

/**
 * @brief No filtering
 */
class none final : public digital {
    public:
    std::unique_ptr<digital> clone() const override {
        return std::make_unique<none>( *this );
    }

    protected:
    void do_apply( grid::tiled_vec3<float> &, const bool, const hook & ) const override { /* do nothing */ }
};

/**
 * @brief Binomial filter, (1/4, 1/2, 1/4) kernel applied `order` times
 *
 * Transfer function (per filtered direction) is cos^(2n)(k Δ / 2), with
 * n = order.
 */
class binomial : public digital {
    protected:

    /// @brief Filter direction(s)
    axis dirs;
    /// @brief Binomial filter order (number of passes)
    unsigned int order;

    /// @brief Apply a 3-point kernel along all selected directions
    void kernel3( grid::tiled_vec3<float> & fld, float a, float b, float c, const bool ext ) const {
        switch( dirs ) {
        case( axis::x ):  fld.kernel3_x ( a, b, c, ext ); break;
        case( axis::y ):  fld.kernel3_y ( a, b, c, ext ); break;
        case( axis::xy ): fld.kernel3_xy( a, b, c, ext ); break;
        }
    }

    void do_apply( grid::tiled_vec3<float> & fld, const bool ext, const hook & post_pass ) const override {
        for( unsigned int i = 0; i < order; i++ ) {
            kernel3( fld, 0.25f, 0.5f, 0.25f, ext );
            if ( post_pass ) post_pass();
        }
    }

    public:

    /**
     * @brief Construct a new binomial filter
     *
     * @param dirs     Filter direction(s): axis::x, axis::y or axis::xy
     * @param order    Filter order (number of passes), 0 is promoted to 1
     */
    binomial( axis dirs, unsigned int order = 1 ) :
        dirs( dirs ),
        order( ( order > 0 ) ? order : 1 )
    {
        if ( ! detail::valid( dirs ) )
            mpi::fatal( "filter: invalid filter direction" );
    }

    std::unique_ptr<digital> clone() const override {
        return std::make_unique<binomial>( *this );
    }
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
    public:

    compensated( axis dirs, unsigned int order = 1 ) :
        binomial( dirs, order ) { }

    std::unique_ptr<digital> clone() const override {
        return std::make_unique<compensated>( *this );
    }

    protected:

    void do_apply( grid::tiled_vec3<float> & fld, const bool ext, const hook & post_pass ) const override {
        // Binomial passes
        binomial::do_apply( fld, ext, post_pass );

        // Compensator, normalized kernel (a, b, a) with 2a + b = 1
        const float a = -0.25f * order;
        const float b =  1.0f  + 0.5f * order;
        kernel3( fld, a, b, a, ext );
        if ( post_pass ) post_pass();
    }
};

} // namespace filter
