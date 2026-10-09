#pragma once

#include "grid/tiled_vec3.cuh"
#include "grid/fft.cuh"

#include "filter.cuh"


class current {

    public:

    // Complex values to use
    using complex_t = util::complex64;

    enum class quantity { j = 0, fj };
    struct bc {
        enum type { none = 0, periodic, reflecting };
    };
    using bc_type = bounds_2d<bc::type>;

    private:

    /// @brief Simulation box size
    const float2 box;
    
    /// @brief time step
    float dt;

    /// @brief Boundary condition (global values)
    current::bc_type global_bc;

    /// @brief Boundary condition (local rank values)
    current::bc_type local_bc;

    /// @brief Iteration number
    int iter;

    /**
     * @brief FFT plan
     * 
     * @note `fJ` holds NVSHMEM symmetric memory and a reference to this
     *       plan's transposed partition, so it must be destroyed before the
     *       plan (see destructor)
     */
    grid::fft::r2c_plan * fft_forward = nullptr;

    /**
     * @brief Process boundary conditions
     * 
     */
    void process_bc();

    /**
     * @brief Validate boundary conditions against the parallel partition
     * 
     * @note Calls mpi::fatal() on invalid combinations
     * 
     * @param test_bc   Boundary conditions to validate
     */
    void validate_bc( const current::bc_type & test_bc ) const {

        // Periodic boundaries must be set on both sides
        if ( (test_bc.x.lower == current::bc::periodic) || (test_bc.x.upper == current::bc::periodic) ) {
            if ( test_bc.x.lower != test_bc.x.upper ) {
                mpi::fatal( "Current boundary type mismatch along x."
                            " When choosing periodic boundaries both lower and upper types"
                            " must be set to current::bc::periodic." );
            }
        }

        if ( (test_bc.y.lower == current::bc::periodic) || (test_bc.y.upper == current::bc::periodic) ) {
            if ( test_bc.y.lower != test_bc.y.upper ) {
                mpi::fatal( "Current boundary type mismatch along y."
                            " When choosing periodic boundaries both lower and upper types"
                            " must be set to current::bc::periodic." );
            }
        }

        // Boundary periodicity must match partition periodicity (both ways)
        const auto & part = J -> get_part();

        if ( part.periodic.x && test_bc.x.lower != current::bc::periodic ) {
            mpi::fatal( "Only periodic x boundaries are supported with periodic x parallel partitions.");
        }

        if ( part.periodic.y && test_bc.y.lower != current::bc::periodic ) {
            mpi::fatal( "Only periodic y boundaries are supported with periodic y parallel partitions.");
        }

        if ( ! part.periodic.x && test_bc.x.lower == current::bc::periodic ) {
            mpi::fatal( "Periodic x boundaries require a periodic x parallel partition.");
        }

        if ( ! part.periodic.y && test_bc.y.lower == current::bc::periodic ) {
            mpi::fatal( "Periodic y boundaries require a periodic y parallel partition.");
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
    filter::digital *filter = nullptr;

    /**
     * @brief Construct a new Current object
     * 
     * @note The FFT requires the parallel partition to be split along y only
     *       (`parallel.dims.x == 1`), with each rank holding a whole number
     *       of tile rows that matches the cuFFTMp slab distribution; in
     *       practice `global_ntiles.y` must be divisible by `parallel.dims.y`
     * @note Collective (creates a cuFFTMp plan and symmetric allocations)
     * 
     * @param global_ntiles     Global number of tiles
     * @param tile_dims         Individual tile size
     * @param box               Global simulation box size
     * @param dt                Time step
     * @param parallel          Parallel partition 
     */
    current( uint2 const global_ntiles, uint2 const tile_dims, float2 const box, float const dt, mpi::cart2d & parallel ):
        box(box), 
        dt(dt)
    {
        // Guard cells (1 below, 2 above)
        bounds_2d<unsigned int> gc;
        gc.x = {1,2};
        gc.y = {1,2};

        J = new grid::tiled_vec3<float> ( global_ntiles, tile_dims, gc, parallel );
        J -> name = "Current density";

        fft_forward = new grid::fft::r2c_plan( *J );

        fJ = fft_forward -> new_kspace_grid3();

        // Zero initial current
        // This is only relevant for diagnostics, current is always zeroed before deposition
        J -> zero();

        // Default boundary conditions follow the partition periodicity:
        // periodic where the partition is periodic, none otherwise
        current::bc_type default_bc = current::bc_type (current::bc::periodic);
        if ( ! parallel.periodic.x ) default_bc.x.lower = default_bc.x.upper = current::bc::none;
        if ( ! parallel.periodic.y ) default_bc.y.lower = default_bc.y.upper = current::bc::none;
        set_bc( default_bc );

        // Set default filtering
        filter = new filter::lowpass( make_float2( 0.5, 0.5 ) );

        // Reset iteration number
        iter = 0;
    };
    
    current( const current& ) = delete;
    current& operator=( const current& ) = delete;

    /**
     * @brief Destroy the Current object
     * 
     * @warning Collective: `fJ` is freed with nvshmem_free(), so all ranks
     *          must destroy their current objects together, and before
     *          grid::fft::cleanup() is called
     */
    ~current() {
        delete (filter);
        
        delete (J);

        // Order matters: fJ (symmetric memory, references the plan's
        // transposed partition) must be destroyed before the FFT plan
        delete (fJ);
        delete( fft_forward );
    }

    /**
     * @brief Get the type of global boundary conditions
     * 
     * @return current::bc_type
     */
    current::bc_type get_bc( ) const noexcept { return global_bc; }

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
        local_bc = global_bc = new_bc;

        // Correct local rank values
        const auto & part = J -> get_part();
        if ( ! part.on_edge( coord::x, edge::lower ) ) local_bc.x.lower = current::bc::none;
        if ( ! part.on_edge( coord::x, edge::upper ) ) local_bc.x.upper = current::bc::none;
        if ( ! part.on_edge( coord::y, edge::lower ) ) local_bc.y.lower = current::bc::none;
        if ( ! part.on_edge( coord::y, edge::upper ) ) local_bc.y.upper = current::bc::none;

        // Issue warning for reflecting boundaries
        if ( mpi::root() ) {
            if ( global_bc.x.lower == current::bc::reflecting || global_bc.x.upper == current::bc::reflecting ||
                 global_bc.y.lower == current::bc::reflecting || global_bc.y.upper == current::bc::reflecting ) {
                std::cout << "(*warning*) Reflecting current boundaries only fold guard cell current,"
                             " the spectral field solve remains periodic\n";
            }
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
        filter::digital * tmp = new_filter.clone();
        delete filter;
        filter = tmp;
    }

    /**
     * @brief Advances electric current density 1 time step
     * 
     * The routine will:
     * 1. Add up current deposited on guard cells
     * 2. Apply physical boundary conditions
     * 3. Get the Fourier transform of the current
     * 4. Apply spectral filtering
     * 
     * @note Filtering is applied to `fJ` only, `J` remains unfiltered
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
     * @note `J` is saved unfiltered, `fJ` is saved filtered, with ky
     *       rotated so that the ky axis is monotonic
     * 
     * @param quant     Which quantity to save (j or fj)
     * @param jc        Current component to save
     */
    void save( const quantity quant, const fcomp::cart jc );
};
