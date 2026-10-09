#pragma once

#include "core/vec_types.cuh"
#include "core/bounds.hpp"

#include "grid/tiled_vec3.cuh"
#include "grid/flat3.cuh"
#include "grid/fft.cuh"

#include "current.hpp"
#include "charge.hpp"
#include "util/complex.hpp"

#include <string>

/**
 * @brief Electro-magnetic fields (spectral PSATD solver)
 * 
 * The field state is kept in k-space (`fEt`, `fB`); the real space fields
 * `E` and `B` are recomputed from it on every advance. All fields are
 * initialized to zero; initial fields, if any, must be set directly in
 * k-space by the caller.
 */
class emf {

    public:

    using complex_t = util::complex64;

    enum class quantity {e = 0, b, fe, fet, fb};
    struct bc { 
        enum type { none = 0, periodic, pec, pmc };
    };
    using bc_type = bounds_2d<bc::type>;

    private:

    /// @brief Boundary condition (global values) - currently always periodic
    emf::bc_type global_bc;

    /// @brief cell size
    const float2 dx;

    /// @brief time step
    const double dt;

    /// @brief Iteration number
    int iter;

    /// @brief Device buffer for field energy calculations (6 values)
    double * d_energy = nullptr;

    /**
     * @brief Process boundary conditions
     * 
     * @note Placeholder, not yet implemented (only periodic boundaries are
     *       supported)
     */
    void process_bc( );

    /**
     * @brief Move simulation window if needed
     * 
     * @note Placeholder, not yet implemented
     */
    void move_window( );

    public:

    /// @brief Electric field
    grid::tiled_vec3<float> * E = nullptr;
    /// @brief Magnetic field
    grid::tiled_vec3<float> * B = nullptr;
    /// @brief Simulation box size
    const float2 box;

    /**
     * @brief Fourier transform of Electric field (transverse + longitudinal)
     * 
     * @note Only updated by advance( current, charge )
     */
    grid::flat3<complex_t> * fE = nullptr;
    /// @brief Fourier transform of transverse Electric field (field state)
    grid::flat3<complex_t> * fEt = nullptr;
    /// @brief Fourier transform of Magnetic field (field state)
    grid::flat3<complex_t> * fB = nullptr;

    /**
     * @brief FFT plan
     * 
     * @note `fE`, `fEt` and `fB` hold NVSHMEM symmetric memory and a reference
     *       to this plan's transposed partition, so they must be destroyed
     *       before the plan (see destructor)
     */
    grid::fft::c2r_plan * fft_backward = nullptr;

    /**
     * @brief Construct a new EMF object
     * 
     * @note The FFT requires the parallel partition to be split along y only
     *       (`parallel.dims.x == 1`), with each rank holding a whole number
     *       of tile rows that matches the cuFFTMp slab distribution; in
     *       practice `global_ntiles.y` must be divisible by `parallel.dims.y`.
     *       Only periodic partitions are currently supported.
     * @note Collective (creates a cuFFTMp plan and symmetric allocations)
     * 
     * @param global_ntiles     Global number of tiles
     * @param tile_dims         Individual tile size
     * @param box               Global simulation box size
     * @param dt                Time step
     * @param parallel          Parallel partition 
     */
    emf( uint2 const global_ntiles, uint2 const tile_dims, float2 const box, double const dt, mpi::cart2d & parallel );
    
    emf( const emf& ) = delete;
    emf& operator=( const emf& ) = delete;

    /**
     * @brief Destroy the EMF object
     * 
     * @warning Collective: the k-space grids are freed with nvshmem_free(), so
     *          all ranks must destroy their emf objects together, and before
     *          grid::fft::cleanup() is called
     */
    ~emf() {
        delete (E);
        delete (B);

        // Order matters: the k-space grids (symmetric memory, referencing the
        // plan's transposed partition) must be destroyed before the FFT plan
        delete (fE);
        delete (fEt);
        delete (fB);

        delete( fft_backward );

        gpu::device::free( d_energy );
    }

    /**
     * @brief Stream extraction
     * 
     * @param os 
     * @param obj 
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, const emf & obj) {
        return os << "EMF object";
    }

    /**
     * @brief Get the iteration number
     * 
     * @return int
     */
    int get_iter() const noexcept { return iter; }

    /**
     * @brief Get the type of global boundary conditions
     * 
     * @return emf::bc_type 
     */
    emf::bc_type get_bc( ) const noexcept { return global_bc; }

    /**
     * @brief Advance EM field 1 iteration assuming no current or charge
     * 
     * @note Only `fEt` and `fB` are advanced; `fE` is not updated
     */
    void advance( );
    
    /**
     * @brief Advance EM field 1 iteration
     * 
     * @note `current` and `charge` must already have been advanced (i.e.
     *       `fJ` and `frho` must be up to date)
     * 
     * @param current   Electric current density
     * @param charge    Electric charge density
     */
    void advance( current & current, charge & charge );

    /**
     * @brief Save EM field component to file
     * 
     * @param quant     Which field to save (E, B, fE, fEt or fB)
     * @param fc        Which field component to save (x, y or z)
     */
    void save( quantity const quant, const fcomp::cart fc ) const;
    
    /**
     * @brief Get EM field energy
     * 
     * @note The energy is recalculated each time this routine is called
     * @warning Collective: values are summed over all parallel nodes
     * 
     * @param ene_E     Electric field energy (per component)
     * @param ene_B     Magnetic field energy (per component)
     */
    void get_energy( double3 & ene_E, double3 & ene_B ) const;
};
