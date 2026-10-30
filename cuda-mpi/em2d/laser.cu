#include "core/gpu.cuh"
#include "grid/tiled_vec3.cuh"
#include "laser.hpp"

#include <iostream>
#include <cassert>
#include <cmath>

#include "filter.hpp"

namespace laser {
namespace kernel {

// Note: Plain-data copies of the laser parameters are passed to the kernels
// because objects of classes with virtual functions may not be passed as
// arguments to __global__ functions.

/**
 * @brief Plane wave kernel parameters
 */
struct plane_wave_params {
    /// @brief Longitudinal envelope
    envelope env;
    /// @brief Wavenumber (= omega0, normalized units)
    float k;
    /// @brief Field amplitude (omega0 * a0)
    float amp;
    /// @brief Cosine of the polarization angle
    float cos_pol;
    /// @brief Sine of the polarization angle
    float sin_pol;
};

/**
 * @brief Gaussian beam kernel parameters
 */
struct gaussian_params {
    /// @brief Longitudinal envelope
    envelope env;
    /// @brief Laser frequency
    float omega0;
    /// @brief Beam waist
    float W0;
    /// @brief Focal plane position
    float focus;
    /// @brief Beam axis position
    float axis;
    /// @brief Field amplitude (omega0 * a0)
    float amp;
    /// @brief Cosine of the polarization angle
    float cos_pol;
    /// @brief Sine of the polarization angle
    float sin_pol;
};

/**
 * @brief Gets longitudinal laser envelope at a given position
 * 
 * @param env       Envelope parameters
 * @param z         Position
 * @return float    Laser envelope
 */
__device__
inline float lon_env( const laser::envelope & env, const float z ) {

    if ( z > env.start ) {
        // Ahead of laser
        return 0.0f;
    } else if ( z > env.start - env.rise ) {
        // Laser rise
        const float csi = z - env.start;
        const float e = sinf( static_cast<float>(M_PI_2) * csi / env.rise );
        return e*e;
    } else if ( z > env.start - (env.rise + env.flat) ) {
        // Flat-top
        return 1.0f;
    } else if ( z > env.start - (env.rise + env.flat + env.fall) ) {
        // Laser fall
        const float csi = z - (env.start - env.rise - env.flat - env.fall);
        const float e = sinf( static_cast<float>(M_PI_2) * csi / env.fall );
        return e*e;
    }

    // Before laser
    return 0.0f;
}

} // namespace kernel
} // namespace laser

namespace {

/**
 * @brief Get cell size from box size and global grid dimensions
 */
float2 cell_size( const grid::tiled_vec3<float> & E, const float2 box ) {
    return make_float2(
        box.x / E.get_global_dims().x,
        box.y / E.get_global_dims().y
    );
}

} // anonymous namespace

/**
 * @brief Validates laser parameters
 * 
 * @note Invalid parameters cause the simulation to stop (mpi::fatal)
 */
void laser::pulse::validate() const {

    if ( a0 <= 0 ) {
        mpi::fatal( "Invalid laser parameter (a0), must be > 0" );
    }    

    if ( omega0 <= 0 ) {
        mpi::fatal( "Invalid laser parameter (omega0), must be > 0" );
    }    

    // If fwhm > 0 it overrides the rise/flat/fall parameters, so these are
    // only checked when it is not set
    if ( fwhm <= 0 ) {
        if ( rise <= 0 ) {
            mpi::fatal( "Invalid laser parameter (rise), must be > 0" );
        }

        if ( flat < 0 ) {
            mpi::fatal( "Invalid laser parameter (flat), must be >= 0" );
        }

        if ( fall <= 0 ) {
            mpi::fatal( "Invalid laser parameter (fall), must be > 0" );
        }
    }
}

/**
 * @brief Get the longitudinal envelope
 * 
 * @note If fwhm > 0 this returns rise = fall = fwhm and flat = 0, otherwise
 *       the rise, flat and fall parameters are used
 * 
 * @return laser::envelope 
 */
laser::envelope laser::pulse::get_envelope() const {
    if ( fwhm > 0 ) {
        return laser::envelope{ start, fwhm, 0.0f, fwhm };
    }
    return laser::envelope{ start, rise, flat, fall };
}

/**
 * @brief Get the polarization direction
 * 
 * @note Uses cos_pol / sin_pol if either is set, otherwise these are
 *       calculated from the polarization angle
 * 
 * @return float2   { cos(polarization), sin(polarization) }
 */
float2 laser::pulse::get_polarization() const {
    if (( cos_pol == 0 ) && ( sin_pol == 0 )) {
        return make_float2( std::cos( polarization ), std::sin( polarization ) );
    }
    return make_float2( cos_pol, sin_pol );
}

/**
 * @brief Adds a new laser pulse onto an EMF object
 * 
 * @param emf   EMF object
 * @return      Returns 0 on success
 */
int laser::pulse::add( emf & emf ) const {

    grid::tiled_vec3<float> tmp_E( emf.E -> global_ntiles, emf.E -> tile_dims, emf.E -> gc, emf.E -> part );
    grid::tiled_vec3<float> tmp_B( emf.B -> global_ntiles, emf.B -> tile_dims, emf.B -> gc, emf.B -> part );

    // copy_to_gc() never writes guard cells on non-periodic global
    // boundaries, so these must start at zero
    tmp_E.zero();
    tmp_B.zero();

    // Get laser fields
    int ierr = launch( tmp_E, tmp_B, emf.box );

    // Add laser to simulation
    if ( ! ierr ) {
        if ( filter > 0 ) {
            // Filtering along x commutes with the discrete divergence, so
            // the filtered field remains divergence free
            filter::compensated fcomp( filter::axis::x, filter );
            fcomp.apply( tmp_E );
            fcomp.apply( tmp_B );
        }
        // add() includes guard cells, and tmp_E / tmp_B have valid guard
        // cells at this point, so no further guard cell update is required
        emf.E->add( tmp_E );
        emf.B->add( tmp_B );
    }

    return ierr;
}

namespace laser {
namespace kernel {

/**
 * @brief Sets plane wave laser fields
 * 
 * @note Fields are set on a Yee grid: E is evaluated at z and B at z + dx/2.
 *       Only the interior cells are set.
 * 
 * @param p                 Plane wave parameters
 * @param E_fld             E field grid
 * @param B_fld             B field grid
 * @param local_tile_start  Tile start position on global tile grid
 * @param dx                Cell size
 */
__global__
void plane_wave(
    plane_wave_params const p,
    grid::tiled_vec3_view<float> E_fld, grid::tiled_vec3_view<float> B_fld, 
    uint2 const local_tile_start, float2 const dx )
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };

    float3 * const __restrict__ E = E_fld.tile_data( tile_idx );
    float3 * const __restrict__ B = B_fld.tile_data( tile_idx );

    const int ix0 = ( local_tile_start.x + tile_idx.x ) * E_fld.tile_dims.x;
    const int ystride = E_fld.tile_ystride();

    const auto nx = E_fld.tile_dims;

    for( unsigned idx = gpu::block::thread_rank(); idx < nx.y * nx.x; idx += gpu::block::num_threads() ) {
        const int ix = idx % nx.x;
        const int iy = idx / nx.x; 

        const float z   = ( ix0 + ix ) * dx.x;
        const float z_2 = ( ix0 + ix + 0.5f ) * dx.x;

        const float fld   = p.amp * lon_env( p.env, z   ) * cosf( p.k * z   );
        const float fld_2 = p.amp * lon_env( p.env, z_2 ) * cosf( p.k * z_2 );

        E[ ix + iy * ystride ] = make_float3(
            0,
            +fld * p.cos_pol,
            +fld * p.sin_pol
        );

        B[ ix + iy * ystride ] = make_float3(
            0,
            -fld_2 * p.sin_pol,
            +fld_2 * p.cos_pol
        );
    }
}

} // namespace kernel
} // namespace laser

/**
 * @brief Launches a plane wave
 * 
 * The E and B tiled grids have the complete laser field.
 * 
 * @param E     Electric field
 * @param B     Magnetic field
 * @param box   Box size
 * @return      Returns 0 on success
 */
int laser::plane_wave::launch( grid::tiled_vec3<float>& E, grid::tiled_vec3<float>& B, const float2 box ) const {

    validate();

    const float2 dx  = cell_size( E, box );
    const float2 pol = get_polarization();

    const kernel::plane_wave_params p {
        get_envelope(), omega0, omega0 * a0, pol.x, pol.y
    };

    dim3 block( 64 );
    dim3 grid( E.get_local_ntiles().x, E.get_local_ntiles().y );

    kernel::plane_wave <<< grid, block >>> (
        p, E.view(), B.view(), E.get_local_tile_start(), dx
    );

    E.copy_to_gc();
    B.copy_to_gc();

    return 0;
}

/**
 * @brief Validate Gaussian laser parameters
 * 
 * @note Invalid parameters cause the simulation to stop (mpi::fatal)
 */
void laser::gaussian::validate() const {
    
    laser::pulse::validate();

    if ( W0 <= 0 ) {
        mpi::fatal( "Invalid laser parameter (W0), must be > 0" );
    }
}

namespace laser {
namespace kernel {

/**
 * @brief Gaussian beam transverse profile and phase
 * 
 * @note The amplitude factor (W0/W)^(1/2) corresponds to 2D slab geometry,
 *       but the Gouy phase atan(z/z0) is the 3D value (the 2D value is
 *       0.5 * atan(z/z0)). This is kept as in the original code.
 * 
 * @param p     Gaussian beam parameters
 * @param z     Distance from focal plane
 * @param r     Distance from beam axis
 * @return      Field profile (normalized)
 */
__device__
inline float gauss_phase( const gaussian_params & p, const float z, const float r ) {
    const float z0   = p.omega0 * ( p.W0 * p.W0 ) / 2;
    const float rho2 = r*r;
    const float curv = 0.5f * rho2 * z / ( z0*z0 + z*z );
    const float rWl2 = ( z0*z0 ) / ( z0*z0 + z*z );
    const float gouy_shift = 0.5f * atan2f( z, z0 );

    return sqrtf( sqrtf( rWl2 ) ) * 
        expf( - rho2 * rWl2 / ( p.W0 * p.W0 ) ) * 
        cosf( p.omega0 * ( z + curv ) - gouy_shift );
}

/**
 * @brief Sets Gaussian beam laser fields (transverse components)
 * 
 * @note Fields are set on a Yee grid:
 *       E.y at (z, r + dy/2), E.z at (z, r),
 *       B.y at (z + dx/2, r), B.z at (z + dx/2, r + dy/2).
 *       Only the interior cells are set.
 * 
 * @param p                 Gaussian beam parameters
 * @param E_fld             E field grid
 * @param B_fld             B field grid
 * @param local_tile_start  Tile start position on global tile grid
 * @param dx                Cell size
 */
__global__
void gaussian( 
    gaussian_params const p, 
    grid::tiled_vec3_view<float> E_fld, grid::tiled_vec3_view<float> B_fld, 
    uint2 const local_tile_start, float2 const dx )
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };

    float3 * const __restrict__ E = E_fld.tile_data( tile_idx );
    float3 * const __restrict__ B = B_fld.tile_data( tile_idx );

    const int ix0 = ( local_tile_start.x + tile_idx.x ) * E_fld.tile_dims.x;
    const int iy0 = ( local_tile_start.y + tile_idx.y ) * E_fld.tile_dims.y;
    const int ystride = E_fld.tile_ystride();

    const auto nx = E_fld.tile_dims;

    for( unsigned idx = gpu::block::thread_rank(); idx < nx.y * nx.x; idx += gpu::block::num_threads() ) {
        const int ix = idx % nx.x;
        const int iy = idx / nx.x; 

        const float z   = ( ix0 + ix ) * dx.x;
        const float z_2 = ( ix0 + ix + 0.5f ) * dx.x;

        const float r   = ( iy0 + iy ) * dx.y - p.axis;
        const float r_2 = ( iy0 + iy + 0.5f ) * dx.y - p.axis;

        const float lenv   = p.amp * lon_env( p.env, z );
        const float lenv_2 = p.amp * lon_env( p.env, z_2 );

        E[ ix + iy * ystride ] = make_float3(
            0,
            +lenv * gauss_phase( p, z - p.focus, r_2 ) * p.cos_pol,
            +lenv * gauss_phase( p, z - p.focus, r   ) * p.sin_pol
        );

        B[ ix + iy * ystride ] = make_float3(
            0,
            -lenv_2 * gauss_phase( p, z_2 - p.focus, r   ) * p.sin_pol,
            +lenv_2 * gauss_phase( p, z_2 - p.focus, r_2 ) * p.cos_pol
        );
    }
}

} // namespace kernel
} // namespace laser

/**
 * @brief Launches a Gaussian pulse
 * 
 * The E and B tiled grids have the complete laser field.
 * 
 * @param E     Electric field
 * @param B     Magnetic field
 * @param box   Box size
 * @return      Returns 0 on success
 */
int laser::gaussian::launch( grid::tiled_vec3<float>& E, grid::tiled_vec3<float>& B, const float2 box ) const {

    validate();

    const float2 dx  = cell_size( E, box );
    const float2 pol = get_polarization();

    const kernel::gaussian_params p {
        get_envelope(), omega0, W0, focus, axis, omega0 * a0, pol.x, pol.y
    };

    dim3 block( 64 );
    dim3 grid( E.get_local_ntiles().x, E.get_local_ntiles().y );

    kernel::gaussian <<< grid, block >>> (
        p, E.view(), B.view(), E.get_local_tile_start(), dx
    );

    // lon_x() reads neighboring cells across tile / node boundaries, so the
    // guard cells must be valid first
    E.copy_to_gc();
    B.copy_to_gc();

    // Set longitudinal components (also updates guard cells on exit)
    lon_x( E, B, dx );

    return 0;
}


namespace laser {
namespace kernel {

/**
 * @brief Divergence correction - Prefix scan local data to get longitudinal components
 * 
 * @note Must be called with grid( local_dims.y ) and block( gpu::warp_size ).
 *       Guard cells must be valid on entry.
 *
 * @param E_grid        E field grid
 * @param B_grid        B field grid
 * @param dx_dy         dx / dy
 * @param sendbuf       Message send buffer
 */
__global__
void lon_x_scan (
    grid::tiled_vec3_view<float> E_grid, 
    grid::tiled_vec3_view<float> B_grid,
    double const dx_dy, double * const __restrict__ sendbuf
)
{
    // Must be launched with grid(local_dims.y)
    const int local_dims_y = gridDim.x;

    // Row is block ID
    const int local_iy = blockIdx.x;

    // Get tile_idx.y
    const int tile_idx_y = local_iy / E_grid.tile_dims.y;
    // Get local tiles iy coordinate
    const int iy = local_iy - tile_idx_y * E_grid.tile_dims.y;

    const int ystride = E_grid.tile_ystride();
    const int nx      = E_grid.tile_dims.x;
    const int laneId  = gpu::block::thread_rank();

    // Running sums, kept in registers (same value on all lanes)
    double divEx = 0;
    double divBx = 0;

    // Start of the rightmost warp-sized chunk in a tile
    const int start = ( ( nx - 1 ) / gpu::warp_size ) * gpu::warp_size;
    
    // Process tiles right to left
    for( int tile_idx_x = E_grid.local_ntiles.x - 1; tile_idx_x >= 0; tile_idx_x-- ) {

        float3 * const __restrict__ tile_E = E_grid.tile_data( tile_idx_x, tile_idx_y );
        float3 * const __restrict__ tile_B = B_grid.tile_data( tile_idx_x, tile_idx_y );

        // All lanes leave this loop on the same iteration, so full mask
        // warp operations are safe
        for( int ix = start + laneId; ix >= 0; ix -= gpu::warp_size ) {

            double dEx = 0;
            double dBx = 0;

            // If inside tile read x divergence
            if ( ix < nx ) {
                dEx = dx_dy * (tile_E[ix+1 +     iy*ystride].y - tile_E[ix+1 + (iy-1)*ystride].y);
                dBx = dx_dy * (tile_B[ix   + (iy+1)*ystride].y - tile_B[ix   +     iy*ystride].y);
            }

            // Do a right to left inclusive scan
            #pragma unroll
            for( int i = 1; i < gpu::warp_size; i <<= 1 ) {
                const double tmpE = __shfl_down_sync( gpu::full_mask, dEx, i );
                const double tmpB = __shfl_down_sync( gpu::full_mask, dBx, i );
                if ( laneId < gpu::warp_size - i ) {
                    dEx += tmpE;
                    dBx += tmpB;
                }
            }

            // If inside tile store longitudinal components
            if ( ix < nx ) {
                tile_E[ix + iy * ystride].x = divEx + dEx;
                tile_B[ix + iy * ystride].x = divBx + dBx;
            }

            // Lane 0 holds the chunk total, broadcast it to all lanes
            divEx += __shfl_sync( gpu::full_mask, dEx, 0 );
            divBx += __shfl_sync( gpu::full_mask, dBx, 0 );
        }
    }
    
    if ( laneId == 0 ) {
        // Copy final divergence values to send buffer
        sendbuf[                local_iy ] = divEx;
        sendbuf[ local_dims_y + local_iy ] = divBx;
    }
}

/**
 * @brief Divergence correction - Add contribution from other parallel nodes
 * 
 * @note Must be called with grid( local_dims.y )
 * 
 * @param E_grid        E field grid
 * @param B_grid        B field grid
 * @param recvbuf       Message receive buffer
 */
__global__
void lon_x_sum (
    grid::tiled_vec3_view<float> E_grid, 
    grid::tiled_vec3_view<float> B_grid,
    double const * const __restrict__ recvbuf
)
{
    // Must be launched with grid(local_dims.y)
    const int local_dims_y = gridDim.x;

    // Row is block ID
    const int local_iy = blockIdx.x;

    const double divEx = recvbuf[                local_iy ];
    const double divBx = recvbuf[ local_dims_y + local_iy ];

    // Get tile_idx.y
    const int tile_idx_y = local_iy / E_grid.tile_dims.y;
    // Get local tiles iy coordinate
    const int iy = local_iy - tile_idx_y * E_grid.tile_dims.y;

    const int ystride = E_grid.tile_ystride();

    for( int tile_idx_x = 0; tile_idx_x < E_grid.local_ntiles.x; tile_idx_x++ ) {

        float3 * const __restrict__ tile_E = E_grid.tile_data( tile_idx_x, tile_idx_y );
        float3 * const __restrict__ tile_B = B_grid.tile_data( tile_idx_x, tile_idx_y );

        for( int ix = gpu::block::thread_rank(); ix < E_grid.tile_dims.x; ix += gpu::block::num_threads() ) {
            tile_E[ ix + iy * ystride ].x += divEx;
            tile_B[ ix + iy * ystride ].x += divBx;
        }
    }    
}

} // namespace kernel
} // namespace laser


/**
 * @brief Sets the longitudinal field components of E and B to ensure 0
 *        divergence
 * 
 * @note Guard cells must be valid on entry. Guard cells are updated on exit.
 * 
 * @warning The algorithm assumes 0 field at the right boundary of the box. If
 *          this is not true then the field values will be wrong.
 * 
 * @param E     E field
 * @param B     B field
 * @param dx    Cell size
 * @return      Returns 0 on success
 */
int laser::gaussian::lon_x( grid::tiled_vec3<float>& E, grid::tiled_vec3<float>& B, const float2 dx ) const {

    double * __restrict__ sendbuf = gpu::device::malloc<double>( 2 * E.local_dims.y );

    const double dx_dy = (double) dx.x / (double) dx.y;

    dim3 grid( E.local_dims.y );
    dim3 block( gpu::warp_size );

    laser::kernel::lon_x_scan <<< grid, block >>> ( 
        E.view(), B.view(), dx_dy, sendbuf
    );

    // If there is a parallel partition along x, add in contribution from nodes to the right
    if ( E.part.dims.x > 1 ) {
        double * __restrict__ recvbuf = gpu::device::malloc<double>( 2 * E.local_dims.y );
        
        // Create a communicator with nodes having the same y coordinate
        int color = E.part.get_coords().y;
        // Reorder ranks right to left
        int key   = E.part.dims.x - 1 - E.part.get_coords().x;
        MPI_Comm newcomm;
        MPI_Comm_split( E.part.get_comm(), color, key, &newcomm );
        
        // Ensure all device data is up to date
        gpu::device::sync();

        // Add contribution from all nodes to the right
        // Note: device buffers, requires CUDA-aware MPI
        MPI_Exscan( sendbuf, recvbuf, 2 * E.local_dims.y, MPI_DOUBLE, MPI_SUM, newcomm );

        // Add result to local grid
        // Rightmost node should not do this because the exscan result is undetermined
        // i.e. not necessarily 0
        if ( key > 0 ) {
            laser::kernel::lon_x_sum <<< grid, block >>> ( 
                E.view(), B.view(), recvbuf );
        }

        MPI_Comm_free( &newcomm );
        gpu::device::free( recvbuf );
    }

    // Free temporary memory
    gpu::device::free( sendbuf );

    // Correct longitudinal values on guard cells
    E.copy_to_gc();
    B.copy_to_gc();

    return 0;
}
