#include "emf.hpp"
#include "grid/fft.cuh"
#include "core/gpu.cuh"

#include <cmath>
#include <iostream>
#include <ostream>
#include <sstream>

/**
* @brief Construct a new EMF object
* 
* @param global_ntiles     Global number of tiles
* @param tile_dims         Individual tile size
* @param box               Global simulation box size
* @param dt                Time step
* @param parallel          Parallel partition 
*/
emf::emf( uint2 const global_ntiles, uint2 const tile_dims, float2 const box, double const dt, mpi::cart2d & parallel ) : 
    dx( float2{ box.x / ( tile_dims.x * global_ntiles.x ), box.y / ( tile_dims.y * global_ntiles.y ) } ),
    dt( dt ), box(box)
{
    // Verify Courant condition
    // The PSATD field solve itself has no CFL stability limit (it is exact in
    // vacuum for a current that is constant over the time step); this limit
    // is imposed by the particle push
    auto cour = std::sqrt( 1.0f/( 1.0f/(dx.x*dx.x) + 1.0f/(dx.y*dx.y) ) );
    if ( dt >= cour ){
        std::ostringstream msg;
        msg << "Invalid timestep, courant condition violation."
            << " For the current resolution " << dx
            << " the maximum timestep is dt = " << cour;
        mpi::fatal( msg.str() );
    }

    // Only periodic boundaries are currently supported
    if (( ! parallel.periodic.x ) || ( ! parallel.periodic.y ) ) {
        mpi::fatal("EMF algorithm only supports periodic boundaries and parallel partitions.");
    }
    global_bc = emf::bc_type( emf::bc::periodic );

    // Guard cells (1 below, 2 above)
    // These are required for field interpolation
    bounds_2d<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {1,2};

    E = new grid::tiled_vec3<float> ( global_ntiles, tile_dims, gc, parallel );
    E -> name = "Electric field";

    B = new grid::tiled_vec3<float> ( global_ntiles, tile_dims, gc, parallel );
    B -> name = "Magnetic field";

    // Check shared memory buffer
    // E and B tiles (including guard cells) are loaded into shared memory for
    // field interpolation
    auto local_mem_size = gpu::block::shared_mem_size();
    if ( local_mem_size < 2 * E->tile_vol * sizeof( float3 ) ) {
        std::ostringstream msg;
        msg << "Tile size too large " << tile_dims << " (plus guard cells),"
            << " insufficient local memory (" << local_mem_size << " B) for EMF object.";
        mpi::fatal( msg.str() );
    }

    // Create FFT plan
    fft_backward = new grid::fft::c2r_plan( *E );

    // Create complex grids for the Fourier transforms
    fE  = fft_backward->new_kspace_grid3();
    fEt = fft_backward->new_kspace_grid3();
    fB  = fft_backward->new_kspace_grid3();

    fE  -> name = "F(E)";
    fEt -> name = "F(Et)";
    fB  -> name = "F(B)";

    // Zero fields
    // Initial fields, if any, are set in k-space (fEt, fB) by the caller
    E -> zero();
    B -> zero();

    fE  -> zero();
    fEt -> zero();
    fB  -> zero();

    // Device buffer for energy calculations
    d_energy = gpu::device::malloc<double>( 6 );

    // Reset iteration number
    iter = 0;
}

namespace kernel {

/**
 * @brief Advance transverse component of E field and B field using PSATD
 *        algorithm without electric current
 * 
 * @note Launch with one block per local k-space row
 * 
 * @param fEt   Fourier transform of transverse component of E-field
 * @param fB    Fourier transform of magnetic field
 * @param dk    k-space cell size
 * @param dt    Time step
 */
__global__
void advance_psatd_nocurr( 
    grid::flat3_view<emf::complex_t> fEt, 
    grid::flat3_view<emf::complex_t> fB,
    const float2 dk, const float dt )
{
    static_assert( ! grid::fft::kspace_transposed, "Normal k-space layout expected" );

    emf::complex_t * const __restrict__ fEtx = fEt.x_buffer;
    emf::complex_t * const __restrict__ fEty = fEt.y_buffer;
    emf::complex_t * const __restrict__ fEtz = fEt.z_buffer;

    emf::complex_t * const __restrict__ fBx  = fB.x_buffer;
    emf::complex_t * const __restrict__ fBy  = fB.y_buffer;
    emf::complex_t * const __restrict__ fBz  = fB.z_buffer;

    constexpr emf::complex_t I{0,1};

    auto dims = fEt.local_dims;
    auto start = fEt.local_start;
    auto global = fEt.global_dims;
    const int ystride = fEt.local_dims.x;

    const int iy   = blockIdx.x; // one block per line
    const int giy  = start.y + iy;
    const float ky = grid::fft::k( make_int2( 0, giy ), global, dk ).y;

    for( auto ix = gpu::block::thread_rank(); ix < dims.x; ix += gpu::block::num_threads() ) {
        const int gix = start.x + ix;
        const float kx = gix * dk.x;

        const float k2 = kx*kx + ky*ky;
        const float k  = sqrtf( k2 );

        // PSATD Field advance equations
        const float C   = cos( k * dt );
        const float S_k = ( k > 0 ) ? sin( k * dt ) / k : dt;

        const int idx = iy * ystride + ix;

        auto Ex = fEtx[idx];
        auto Ey = fEty[idx];
        auto Ez = fEtz[idx];

        auto Bx = fBx[idx];
        auto By = fBy[idx];
        auto Bz = fBz[idx];

        Ex = C * Ex + S_k * ( I * (  ky *  fBz[idx]                  ) );
        Ey = C * Ey + S_k * ( I * ( -kx *  fBz[idx]                  ) );
        Ez = C * Ez + S_k * ( I * (  kx *  fBy[idx] - ky *  fBx[idx] ) );

        Bx = C * Bx - S_k * ( I * (  ky * fEtz[idx]                  ) );
        By = C * By - S_k * ( I * ( -kx * fEtz[idx]                  ) );
        Bz = C * Bz - S_k * ( I * (  kx * fEty[idx] - ky * fEtx[idx] ) );

        fEtx[idx] = Ex;
        fEty[idx] = Ey;
        fEtz[idx] = Ez;

        fBx[idx]  = Bx;
        fBy[idx]  = By;
        fBz[idx]  = Bz;
    }
}

/**
 * @brief Advance transverse component of E field and B field using PSATD
 *        algorithm
 * 
 * @note Launch with one block per local k-space row
 * 
 * @param fEt   Fourier transform of transverse component of E-field
 * @param fB    Fourier transform of magnetic field
 * @param fJ    Fourier transform of current density
 * @param dk    k-space cell size
 * @param dt    Time step
 */
__global__
void advance_psatd( 
    grid::flat3_view<emf::complex_t> fEt, 
    grid::flat3_view<emf::complex_t> fB,
    grid::flat3_view<emf::complex_t> fJ,
    const float2 dk, float const dt )
{
    static_assert( ! grid::fft::kspace_transposed, "Normal k-space layout expected" );

    emf::complex_t * const __restrict__ fEtx = fEt.x_buffer;
    emf::complex_t * const __restrict__ fEty = fEt.y_buffer;
    emf::complex_t * const __restrict__ fEtz = fEt.z_buffer;

    emf::complex_t * const __restrict__ fBx  = fB.x_buffer;
    emf::complex_t * const __restrict__ fBy  = fB.y_buffer;
    emf::complex_t * const __restrict__ fBz  = fB.z_buffer;

    emf::complex_t const * const __restrict__ fJx  = fJ.x_buffer;
    emf::complex_t const * const __restrict__ fJy  = fJ.y_buffer;
    emf::complex_t const * const __restrict__ fJz  = fJ.z_buffer;

    constexpr emf::complex_t I{0,1};

    auto dims   = fEt.local_dims;
    auto start  = fEt.local_start;
    auto global = fEt.global_dims;
    const int ystride = fEt.local_dims.x;

    const int iy   = blockIdx.x; // one block per line
    const int giy  = start.y + iy;
    const float ky = grid::fft::k( make_int2( 0, giy ), global, dk ).y;

    for( auto ix = gpu::block::thread_rank(); ix < dims.x; ix += gpu::block::num_threads() ) {
        const int gix = start.x + ix;
        const float kx = gix * dk.x;

        const float k2 = kx*kx + ky*ky;
        const float k  = sqrt( k2 );

        const int idx = iy * ystride + ix;

        // Compute transverse current
        const emf::complex_t kdJ  = (kx * fJx[idx] + ky * fJy[idx]);
        const emf::complex_t fJtx = (k2 > 0) ? fJx[idx] - kx * kdJ / k2 : 0;
        const emf::complex_t fJty = (k2 > 0) ? fJy[idx] - ky * kdJ / k2 : 0; 
        const emf::complex_t fJtz = fJz[idx];

        // PSATD Field advance equations
        const float C   = cos( k * dt );
        const float S_k = ( k > 0 ) ? sin( k * dt ) / k : dt;
        const emf::complex_t I1mC_k2 = ( k2 > 0 )? I * (1.0f - C) / k2 : 0;

        auto Ex = fEtx[idx];
        auto Ey = fEty[idx];
        auto Ez = fEtz[idx];

        auto Bx = fBx[idx];
        auto By = fBy[idx];
        auto Bz = fBz[idx];

        Ex = C * Ex + S_k * ( I * (  ky *  fBz[idx]                  ) - fJtx );
        Ey = C * Ey + S_k * ( I * ( -kx *  fBz[idx]                  ) - fJty );
        Ez = C * Ez + S_k * ( I * (  kx *  fBy[idx] - ky *  fBx[idx] ) - fJtz );

        Bx = C * Bx - S_k * ( I * (  ky * fEtz[idx]                  ) ) + I1mC_k2 * (  ky * fJz[idx]                 );
        By = C * By - S_k * ( I * ( -kx * fEtz[idx]                  ) ) + I1mC_k2 * ( -kx * fJz[idx]                 );
        Bz = C * Bz - S_k * ( I * (  kx * fEty[idx] - ky * fEtx[idx] ) ) + I1mC_k2 * (  kx * fJy[idx] - ky * fJx[idx] );

        fEtx[idx] = Ex;
        fEty[idx] = Ey;
        fEtz[idx] = Ez;

        fBx[idx]  = Bx;
        fBy[idx]  = By;
        fBz[idx]  = Bz;
    }
}

/**
 * @brief Update Electric field from charge density
 * 
 * @note Launch with one block per local k-space row
 * 
 * @param fE        Fourier transform of E-field (full)
 * @param fEt       Fourier transform of E-field (transverse)
 * @param frho_src  Fourier transform of charge density
 * @param dk        k-space cell size
 */
__global__
void update_fE( 
    grid::flat3_view<emf::complex_t> fE, 
    grid::flat3_view<emf::complex_t> fEt,
    grid::flat_view<emf::complex_t> frho_src,
    float2 const dk )
{
    static_assert( ! grid::fft::kspace_transposed, "Normal k-space layout expected" );

    emf::complex_t * const __restrict__ fEx = fE.x_buffer;
    emf::complex_t * const __restrict__ fEy = fE.y_buffer;
    emf::complex_t * const __restrict__ fEz = fE.z_buffer;

    emf::complex_t const * const __restrict__ fEtx = fEt.x_buffer;
    emf::complex_t const * const __restrict__ fEty = fEt.y_buffer;
    emf::complex_t const * const __restrict__ fEtz = fEt.z_buffer;

    emf::complex_t const * const __restrict__ frho = frho_src.d_buffer;

    constexpr emf::complex_t I{0,1};

    auto dims = fEt.local_dims;
    auto start = fEt.local_start;
    auto global = fEt.global_dims;
    const int ystride = fEt.local_dims.x;

    const int iy   = blockIdx.x; // one block per line
    const int giy  = start.y + iy;
    const float ky = grid::fft::k( make_int2( 0, giy ), global, dk ).y;

    for( auto ix = gpu::block::thread_rank(); ix < dims.x; ix += gpu::block::num_threads() ) {
        const int gix = start.x + ix;
        const float kx = gix * dk.x;

        // Longitudinal field from Gauss's law: E_L = -i k rho / k^2
        const float k2 = kx*kx + ky*ky;
        const float inv_k2 = ( k2 > 0 ) ? 1.f / k2 : 0;
        
        const int idx = iy * ystride + ix;

        fEx[idx] = -I * kx * frho[idx] * inv_k2 + fEtx[idx];
        fEy[idx] = -I * ky * frho[idx] * inv_k2 + fEty[idx];
        fEz[idx] =                                fEtz[idx] ;
    }
}

} // namespace kernel

/**
 * @brief Advance EM fields 1 time step (no current or charge)
 * 
 */
void emf::advance() {

    // Advance transverse fields
    kernel::advance_psatd_nocurr <<< fEt -> get_local_dims().y, 256 >>> ( 
        fEt -> view(), fB -> view(), grid::fft::dk( box ), dt );

    // Transform to real fields
    // This will already update halo values in the scatter stage
    fft_backward -> transform( *E, *fEt );
    fft_backward -> transform( *B, *fB );

    // Advance internal iteration number
    iter += 1;
}

/**
 * @brief Advance EM fields 1 time step including current
 * 
 * @param current   Electric current
 * @param charge    Electric charge
 */
void emf::advance( current & current, charge & charge ) {

    // The k-space grids come from different FFT plans, make sure they share
    // the same local layout
    if ( current.fJ -> get_local_dims()  != fEt -> get_local_dims()  ||
         current.fJ -> get_local_start() != fEt -> get_local_start() ||
         charge.frho -> get_local_dims()  != fEt -> get_local_dims()  ||
         charge.frho -> get_local_start() != fEt -> get_local_start() ) {
        mpi::fatal( "emf::advance(): k-space layout of current / charge does not match EMF grids" );
    }

    // Advance transverse fields
    kernel::advance_psatd <<< fEt -> get_local_dims().y, 256 >>> ( 
        fEt -> view() , fB  -> view(), current.fJ  -> view(),
        grid::fft::dk( box ), dt
    );

    // Update total E-field
    kernel::update_fE<<< fEt -> get_local_dims().y, 256 >>> ( 
        fE -> view(), fEt -> view(), charge.frho -> view(),
        grid::fft::dk( box )
    );

    // Transform to real fields
    // This will already update halo values in the scatter stage
    fft_backward -> transform( *E, *fE );
    fft_backward -> transform( *B, *fB );

    // Advance internal iteration number
    iter += 1;
}

/**
 * @brief Save EMF data to diagnostic file
 * 
 * @param quant     Field to save (E, B, fE, fEt or fB)
 * @param fc        Field component to save (x, y or z)
 */
void emf::save( const quantity quant, fcomp::cart const fc ) const {

    std::string vfname;  // Dataset name
    std::string vflabel; // Dataset label (for plots)

    grid::tiled_vec3<float> * f = nullptr;
    grid::flat3<emf::complex_t> * cf = nullptr;

    switch ( quant ) {
        case quantity::e :
            f = E;
            vfname = "E";
            vflabel = "E_";
            break;
        case quantity::b :
            f = B;
            vfname = "B";
            vflabel = "B_";
            break;
        case quantity::fe :
            cf = fE;
            vfname = "fE";
            vflabel = "\\mathcal{F}\\,E_";
            break;
        case quantity::fet :
            cf = fEt;
            vfname = "fEt";
            vflabel = "\\mathcal{F}\\,E^\\perp_";
            break;
        case quantity::fb :
            cf = fB;
            vfname = "fB";
            vflabel = "\\mathcal{F}\\,B_";
            break;
        default:
            mpi::fatal( "Invalid quantity selected" );
    }

    switch ( fc ) {
        case( fcomp::x ) :
            vfname  += 'x';
            vflabel += 'x';
            break;
        case( fcomp::y ) :
            vfname  += 'y';
            vflabel += 'y';
            break;
        case( fcomp::z ) :
            vfname  += 'z';
            vflabel += 'z';
            break;
        default:
            mpi::fatal( "Invalid field component (fc) selected" );
    }

    zdf::iteration iteration = {
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_n"
    };

    zdf::grid_info info = {
        .name = (char *) vfname.c_str(),
        .ndims = 2,
        .label = (char *) vflabel.c_str(),
        .units = (char *) "m_e c \\omega_n e^{-1}"
    };

    zdf::grid_axis axis[2];

    if ( f != nullptr ) {
        // Real field
        axis[0] = zdf::grid_axis {
            .name = (char *) "x",
            .min = 0.0,
            .max = box.x,
            .label = (char *) "x",
            .units = (char *) "c/\\omega_n"
        };

        axis[1] = zdf::grid_axis {
            .name = (char *) "y",
            .min = 0.0,
            .max = box.y,
            .label = (char *) "y",
            .units = (char *) "c/\\omega_n"
        };

        info.axis = axis;

        f -> save( fc, info, iteration, "EMF" );

    } else {
        // Complex (FFT) field
        // kx: 0 ... nx/2 (r2c half spectrum)
        // ky: kspace_save() rotates by ceil(ny/2) rows, so it runs from
        //     -floor(ny/2) to floor((ny-1)/2) modes
        const float2 dk = grid::fft::dk( box );
        const int nky = cf -> get_global_dims().y;

        axis[0] = zdf::grid_axis {
            .name = (char *) "kx",
            .min = 0.0,
            .max = (cf -> get_global_dims().x - 1) * dk.x,
            .label = (char *) "k_x",
            .units = (char *) "\\omega_n/c"
        };

        axis[1] = zdf::grid_axis {
            .name = (char *) "ky",
            .min = - dk.y * ( nky / 2 ),
            .max =   dk.y * ( (nky - 1) / 2 ),
            .label = (char *) "k_y",
            .units = (char *) "\\omega_n/c"
        };

        info.axis = axis;

        grid::fft::kspace_save( *cf, fc, info, iteration, "EMF" );
    }
}

namespace kernel {

/**
 * @brief Add up the local field energy (sum of squared field values)
 * 
 * @note Launch with one block per local tile; d_energy must be zeroed before
 *       the launch
 * 
 * @param E_fld     Electric field
 * @param B_fld     Magnetic field
 * @param d_energy  Output (6 values: Ex, Ey, Ez, Bx, By, Bz)
 */
__global__
void get_energy( 
    grid::tiled_vec3_view<float> E_fld, grid::tiled_vec3_view<float> B_fld,
    double * const __restrict__ d_energy ) 
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };

    float3 * const __restrict__ E = E_fld.tile_data( tile_idx );
    float3 * const __restrict__ B = B_fld.tile_data( tile_idx );

    const int ystride = E_fld.tile_ystride();

    double3 ene_E = double3{0};
    double3 ene_B = double3{0};

    auto nx = E_fld.tile_dims;

    for( int idx = gpu::block::thread_rank(); idx < nx.y * nx.x; idx += gpu::block::num_threads() ) {
        int const i = idx % nx.x;
        int const j = idx / nx.x;

        float3 const efld = E[ j * ystride + i ];
        float3 const bfld = B[ j * ystride + i ];

        ene_E.x += efld.x * efld.x;
        ene_E.y += efld.y * efld.y;
        ene_E.z += efld.z * efld.z;

        ene_B.x += bfld.x * bfld.x;
        ene_B.y += bfld.y * bfld.y;
        ene_B.z += bfld.z * bfld.z;
    }

    // Add up energy from in all warps
    ene_E.x = gpu::warp::reduce_add( ene_E.x );
    ene_E.y = gpu::warp::reduce_add( ene_E.y );
    ene_E.z = gpu::warp::reduce_add( ene_E.z );

    ene_B.x = gpu::warp::reduce_add( ene_B.x );
    ene_B.y = gpu::warp::reduce_add( ene_B.y );
    ene_B.z = gpu::warp::reduce_add( ene_B.z );

    // Add warp energy to global memory
    if ( gpu::warp::thread_rank() == 0 ) {
        gpu::device::atomic_fetch_add( &(d_energy[0]), ene_E.x );
        gpu::device::atomic_fetch_add( &(d_energy[1]), ene_E.y );
        gpu::device::atomic_fetch_add( &(d_energy[2]), ene_E.z );

        gpu::device::atomic_fetch_add( &(d_energy[3]), ene_B.x );
        gpu::device::atomic_fetch_add( &(d_energy[4]), ene_B.y );
        gpu::device::atomic_fetch_add( &(d_energy[5]), ene_B.z );
    }
}

} // namespace kernel

/**
 * @brief Get total field energy per field component
 * 
 * @warning This function will always recalculate the energy each time it is
 *          called. Collective: values are summed over all parallel nodes.
 * 
 * @param ene_E     Total E-field energy (per component)
 * @param ene_B     Total B-field energy (per component)
 */
void emf::get_energy( double3 & ene_E, double3 & ene_B ) const {

    // Zero energy values
    gpu::device::zero( d_energy, 6 );

    // Add up energy from all local cells
    dim3 grid( E->get_local_ntiles().x, E->get_local_ntiles().y );
    dim3 block( 1024 );
    kernel::get_energy <<< grid, block >>> ( 
        E->view(), B->view(), d_energy
    );

    // Copy results to host
    double h_energy[6];
    gpu::device::memcpy_tohost( h_energy, d_energy, 6 );

    // Sum over all parallel nodes
    E -> get_part().allreduce( h_energy, 6, MPI_SUM );

    // Normalize
    ene_E.x = 0.5 * dx.x * dx.y * h_energy[0];
    ene_E.y = 0.5 * dx.x * dx.y * h_energy[1];
    ene_E.z = 0.5 * dx.x * dx.y * h_energy[2];

    ene_B.x = 0.5 * dx.x * dx.y * h_energy[3];
    ene_B.y = 0.5 * dx.x * dx.y * h_energy[4];
    ene_B.z = 0.5 * dx.x * dx.y * h_energy[5];
}
