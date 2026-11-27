#pragma once

#include "util/error.hpp"

#include "grid/tiled.hpp"
#include "grid/fft.hpp"

#include "util/complex.hpp"
#include "filter.hpp"

namespace em2ds {

class charge {

    public:

    using complex_t = util::complex<float>;

    enum class quantity  { rho = 0, frho };

    struct bc {
        enum type { none = 0, periodic, reflecting };
    };
    using bc_type = bounds_2d<bc::type>;

    private:

    /// @brief Simulation box size
    float2 box;

    /// @brief cell size
    // float2 dx;
    
    /// @brief time step
    float dt;

    /// @brief Boundary condition
    charge::bc_type grid_bc;

    /// @brief Iteration number
    int iter;

    /// @brief FFT plan
    grid::fft::r2c_plan  * fft_forward;

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
    void validate_bc( const charge::bc_type & test_bc ) const {

        // Periodic boundaries must be set on both sides
        if ( (test_bc.x.lower == charge::bc::periodic) || (test_bc.x.upper == charge::bc::periodic) ) {
            if ( test_bc.x.lower != test_bc.x.upper ) {
                util::fatal( "Charge boundary type mismatch along x."
                            " When choosing periodic boundaries both lower and upper types"
                            " must be set to charge::bc::periodic." );
            }
        }

        if ( (test_bc.y.lower == charge::bc::periodic) || (test_bc.y.upper == charge::bc::periodic) ) {
            if ( test_bc.y.lower != test_bc.y.upper ) {
                util::fatal( "Charge boundary type mismatch along y."
                            " When choosing periodic boundaries both lower and upper types"
                            " must be set to charge::bc::periodic." );
            }
        }

        // Boundary periodicity must match grid periodicity (both ways)
        const auto periodic = rho -> get_grid_periodic();

        if ( periodic.x && test_bc.x.lower != charge::bc::periodic ) {
            util::fatal( "Only periodic x boundaries are supported with periodic x grids.");
        }

        if ( periodic.y && test_bc.y.lower != charge::bc::periodic ) {
            util::fatal( "Only periodic y boundaries are supported with periodic y grids.");
        }

        if ( ! periodic.x && test_bc.x.lower == charge::bc::periodic ) {
            util::fatal( "Periodic x boundaries require a periodic x grid.");
        }

        if ( ! periodic.y && test_bc.y.lower == charge::bc::periodic ) {
            util::fatal( "Periodic y boundaries require a periodic y grid.");
        }
    }

    public:

    /**
     * @brief Charge density
     * 
     * @note Owned by this object
     */
    grid::tiled<float> * rho;

    /**
     * @brief Charge density k-space (filtered)
     * 
     * @note Owned by this object. Symmetric (NVSHMEM) allocation created by
     *       `fft_forward`
     */
    grid::flat<complex_t> * frho;

    /**
     * @brief Neutralizing background
     * 
     * @note Defaults to `nullptr` (no background). If set, ownership is
     *       transferred to this object, which will delete it. Must have the
     *       same tiling as `rho`
     */
    grid::tiled<float> * neutral;

    /**
     * @brief Filtering parameters
     * 
     * @note Owned by this object, use set_filter() to replace it
     */
    std::unique_ptr<filter::digital> filter;

    /**
     * @brief Construct a new Charge object
     * 
     * @param grid_ntiles       Number of tiles
     * @param tile_dims         Individual tile size
     * @param box               Simulation box size
     * @param dt                Time step
     * @param periodic          Periodic boundaries (x,y)
     */
    charge( uint2 const grid_ntiles, uint2 const tile_dims, float2 const box, float const dt, int2 const periodic ):
        box(box), 
        // dx( { box.x / ( nx.x * ntiles.x ), box.y / ( nx.y * ntiles.y ) } ),
        dt(dt)
    {
        // Guard cells (1 below, 2 above)
        bounds_2d<unsigned int> gc;
        gc.x = {1,2};
        gc.y = {1,2};

        rho = new grid::tiled<float> ( grid_ntiles, tile_dims, gc, periodic );
        rho -> name = "Charge";

        fft_forward = new grid::fft::r2c_plan( *rho );
        frho = fft_forward -> new_kspace_grid();

        // Zero initial charge
        // This is only relevant for diagnostics, current should always zeroed before deposition
        rho -> zero();

        // Default boundary conditions follow the grid periodicity:
        // periodic where the grid is periodic, none otherwise
        charge::bc_type default_bc = charge::bc_type (charge::bc::periodic);
        if ( ! periodic.x ) default_bc.x.lower = default_bc.x.upper = charge::bc::none;
        if ( ! periodic.y ) default_bc.y.lower = default_bc.y.upper = charge::bc::none;
        set_bc( default_bc );

        // Set default filtering
        filter = std::make_unique<filter::lowpass>( make_float2( 0.5, 0.5 ) );

        // Reset iteration number
        iter = 0;

        // Default is not to have a neutralizing background
        neutral = nullptr;
    };
    
    charge( const charge& ) = delete;
    charge& operator=( const charge& ) = delete;

    /**
     * @brief Destroy the Charge object
     * 
     */
    ~charge() {
        delete (rho);
        delete (frho);

        delete (neutral);
        delete( fft_forward );
    }

    /**
     * @brief Get the type of boundary conditions
     * 
     * @return charge::bc_type
     */
    charge::bc_type get_bc( ) { return grid_bc; }

    /**
     * @brief Set the boundary conditions
     * 
     * @param new_bc    New boundary condition values
     */
    void set_bc( charge::bc_type new_bc ) {

        // Validate parameters
        validate_bc( new_bc );

        // Store new values
        grid_bc = new_bc;

        // Issue warning for reflecting boundaries
        if ( grid_bc.x.lower == charge::bc::reflecting || grid_bc.x.upper == charge::bc::reflecting ||
             grid_bc.y.lower == charge::bc::reflecting || grid_bc.y.upper == charge::bc::reflecting ) {
            std::cout << "(*warning*) Reflecting charge boundaries only fold guard cell charge,"
                         " the spectral field solve remains periodic\n";
        }
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
     * @brief Advance charge
     * 
     * @note This will i) add up charge deposited on guard cells, ii) apply
     *       physical boundary conditions, iii) add the neutralizing background,
     *       iv) Fourier transform and v) filter.
     * @note Filtering is applied to `frho` only, `rho` remains unfiltered
     */
    void advance();

    /**
     * @brief Zero charge density values
     * 
     */
    void zero() {
        rho -> zero();
    }

    /**
     * @brief Save charge density to disk
     * 
     * @note `rho` is saved unfiltered, `frho` is saved filtered, with ky
     *       rotated so that the ky axis is monotonic
     * 
     * @param quant     Which quantity to save (rho or frho)
     */
    void save( const quantity quant );
};

} // namespace em2ds
