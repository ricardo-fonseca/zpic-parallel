#pragma once

#include "util/error.hpp"

#include "grid/tiled_vec3.hpp"
#include "grid/fft.hpp"

#include "filter.hpp"

namespace em2ds {


class current {

    public:

    // Complex values to use
    using complex_t = util::complex<float>;

    enum class quantity { j = 0, fj };
    struct bc {
        enum type { none = 0, periodic, reflecting };
    };
    using bc_type = bounds_2d<bc::type>;

    private:

    /// @brief Simulation box size
    const float2 box;

    /// @brief cell size
    // const float2 dx;
    
    /// @brief time step
    float dt;

    /// @brief Boundary condition
    current::bc_type grid_bc;

    /// @brief Iteration number
    int iter;

    /// @brief FFT plan
    grid::fft::r2c_plan * fft_forward = nullptr;

    /**
     * @brief Process boundary conditions
     * 
     */
    void process_bc();

    /**
     * @brief Validate boundary conditions against the grid periodicity
     * 
     * @note Calls util::fatal() on invalid combinations
     * 
     * @param test_bc   Boundary conditions to validate
     */
    void validate_bc( const current::bc_type & test_bc ) const {

        // Periodic boundaries must be set on both sides
        if ( (test_bc.x.lower == current::bc::periodic) || (test_bc.x.upper == current::bc::periodic) ) {
            if ( test_bc.x.lower != test_bc.x.upper ) {
                util::fatal( "Current boundary type mismatch along x."
                            " When choosing periodic boundaries both lower and upper types"
                            " must be set to current::bc::periodic." );
            }
        }

        if ( (test_bc.y.lower == current::bc::periodic) || (test_bc.y.upper == current::bc::periodic) ) {
            if ( test_bc.y.lower != test_bc.y.upper ) {
                util::fatal( "Current boundary type mismatch along y."
                            " When choosing periodic boundaries both lower and upper types"
                            " must be set to current::bc::periodic." );
            }
        }

        // Boundary periodicity must match grid periodicity (both ways)
        const auto periodic = J -> get_grid_periodic();

        if ( periodic.x && test_bc.x.lower != current::bc::periodic ) {
            util::fatal( "Only periodic x boundaries are supported with periodic x grids.");
        }

        if ( periodic.y && test_bc.y.lower != current::bc::periodic ) {
            util::fatal( "Only periodic y boundaries are supported with periodic y grids.");
        }

        if ( ! periodic.x && test_bc.x.lower == current::bc::periodic ) {
            util::fatal( "Periodic x boundaries require a periodic x grid.");
        }

        if ( ! periodic.y && test_bc.y.lower == current::bc::periodic ) {
            util::fatal( "Periodic y boundaries require a periodic y grid.");
        }

        // Issue warning for reflecting boundaries
        if ( test_bc.x.lower == current::bc::reflecting || test_bc.x.upper == current::bc::reflecting ||
             test_bc.y.lower == current::bc::reflecting || test_bc.y.upper == current::bc::reflecting ) {
            std::cout << "(*warning*) Reflecting current bc only fold guard cell current,"
                         " the spectral field solve remains periodic\n";
        }
    }

    public:

    /**
     * @brief Current density
     * 
     * @note Owned by this object
     */
    grid::tiled_vec3<float> * J = nullptr;

    /**
     * @brief Current density k-space (filtered)
     * 
     * @note Owned by this object. Symmetric (NVSHMEM) allocation created by
     *       `fft_forward`
     */
    grid::flat3<complex_t> * fJ = nullptr;

    /**
     * @brief Filtering parameters
     * 
     * @note Owned by this object, use set_filter() to replace it
     */
    std::unique_ptr<filter::digital> filter;

    /**
     * @brief Construct a new Current object
     * 
     * @param grid_ntiles       Number of tiles
     * @param tile_dims         Individual tile size
     * @param box               Simulation box size
     * @param dt                Time step
     * @param periodic          Periodic boundaries (x,y)
     */
    current( uint2 const grid_ntiles, uint2 const tile_dims, float2 const box, float const dt, int2 const periodic ):
        box(box), 
        dt(dt)
    {
        // Guard cells (1 below, 2 above)
        bounds_2d<unsigned int> gc;
        gc.x = {1,2};
        gc.y = {1,2};

        J = new grid::tiled_vec3<float> ( grid_ntiles, tile_dims, gc, periodic );
        J -> name = "Current density";

        fft_forward = new grid::fft::r2c_plan( *J );

        fJ = fft_forward -> new_kspace_grid3();

        // Zero initial charge
        // This is only relevant for diagnostics, current is always zeroed before deposition
        J -> zero();

        // Set default boundary conditions to be none or periodic
        // according to grid periodicity
        current::bc_type default_bc = current::bc_type (current::bc::periodic);
        if ( ! periodic.x ) default_bc.x.lower = default_bc.x.upper = current::bc::none;
        if ( ! periodic.y ) default_bc.y.lower = default_bc.y.upper = current::bc::none;
        set_bc( default_bc );

        // Set default filtering
        filter = std::make_unique<filter::lowpass>( make_float2( 0.5, 0.5 ) );

        // Reset iteration number
        iter = 0;
    };

    current( const current& ) = delete;
    current& operator=( const current& ) = delete;

    /**
     * @brief Destroy the Current object
     * 
     */
    ~current() {
        delete (J);

        delete (fJ);
        delete( fft_forward );
    }

    /**
     * @brief Get the type of boundary conditions
     * 
     * @return current::bc_type
     */
    current::bc_type get_bc( ) const noexcept { return grid_bc; }

    /**
     * @brief Set the boundary conditions
     * 
     * @warning The field solve is spectral and therefore periodic. Reflecting
     *          boundaries only fold guard cell current back into the domain;
     *          the fields computed from `fJ` still see a periodic domain.
     * 
     * @param new_bc    New boundary condition values
     */
    void set_bc( current::bc_type new_bc ) {

        // Validate parameters
        validate_bc( new_bc );

        // Store new values
        grid_bc = new_bc;
    }

    /**
     * @brief Replace the digital filter
     * 
     * @note Stores a copy (via clone()) and deletes the previous filter; the
     *       caller keeps ownership of `new_filter`
     * 
     * @param new_filter    New filter object
     */
    void set_filter( const filter::digital & new_filter ) {
        filter = new_filter.clone();
    }

    /**
     * @brief Advances electric current density 1 time step
     * 
     * The routine will:
     * 1. Update the guard cells
     * 2. Get the Fourier transform of the current
     * 3. Apply spectral filtering
     * 
     */
    void advance();

    /**
     * @brief Zero electric current values
     * 
     */
    void zero() {
        J -> zero();
    }

    /**
     * @brief Save electric current data to diagnostic file
     * 
     * @param quant     Which field to save (J or fJ)
     * @param jc        Current component to save
     */
    void save( const quantity quant, const fcomp::cart jc );
};

} // namespace em2ds
