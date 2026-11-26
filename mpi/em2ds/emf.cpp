#include "emf.hpp"
#include "grid/fft.hpp"

#include <cmath>
#include <iostream>
#include <ostream>
#include <sstream>

namespace em2ds {

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
    // I believe the PSATD solver cannot work with larger time-steps, needs investigating
    // Regardless, this is required by the particle push
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
    // These are required for the Yee solver AND for field interpolation
    bounds_2d<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {1,2};

    E = new grid::tiled_vec3<float> ( global_ntiles, tile_dims, gc, parallel );
    E -> name = "Electric field";

    B = new grid::tiled_vec3<float> ( global_ntiles, tile_dims, gc, parallel );
    B -> name = "Magnetic field";

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

    // Reset iteration number
    iter = 0;
}

/**
 * @brief Advance transverse component of E field and B field using PSATD
 *        algorithm without electric current
 * 
 * @param fEt 
 * @param fB 
 * @param dims 
 * @param dk 
 * @param dt 
 */
void advance_psatd_nocurr( 
    grid::flat3<emf::complex_t> & fEt, 
    grid::flat3<emf::complex_t> & fB,
    const float2 dk, const float dt )
{

    static_assert( grid::fft::kspace_transposed, "Transposed k-space layout expected" );

    emf::complex_t * const __restrict__ fEtx = fEt.x();
    emf::complex_t * const __restrict__ fEty = fEt.y();
    emf::complex_t * const __restrict__ fEtz = fEt.z();

    emf::complex_t * const __restrict__ fBx  = fB.x();
    emf::complex_t * const __restrict__ fBy  = fB.y();
    emf::complex_t * const __restrict__ fBz  = fB.z();

    constexpr emf::complex_t I{0,1};

    auto dims = fEt.get_local_dims();
    auto start = fEt.get_local_start();
    auto global = fEt.get_global_dims();

    #pragma omp parallel for
    for( unsigned idx = 0; idx < dims.y * dims.x; idx++ ) {
        const int ix = start.x + idx % dims.x;
        const int iy = start.y + idx / dims.x;

        const float2 kvec = grid::fft::k( make_int2(ix,iy), global, dk );
        
        const float kx = kvec.x;
        const float ky = kvec.y;

        const float k2 = kx*kx + ky*ky;
        const float k  = sqrtf( k2 );

        // PSATD Field advance equations
        const float C   = std::cos( k * dt );
        const float S_k = ( k > 0 ) ? std::sin( k * dt ) / k : dt;

        emf::complex_t Ex = fEtx[idx];
        emf::complex_t Ey = fEty[idx];
        emf::complex_t Ez = fEtz[idx];

        emf::complex_t Bx = fBx[idx];
        emf::complex_t By = fBy[idx];
        emf::complex_t Bz = fBz[idx];

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
 * @param fEt   Fourier transform of transverse component of E-field
 * @param fB    Fourier transform of magnetic field
 * @param fJ    Fourier transform of current density
 * @param dims  Global grid dimensions
 * @param dk    k-space cell size
 * @param dt    Time step
 */
void advance_psatd( 
    grid::flat3<emf::complex_t> & fEt, 
    grid::flat3<emf::complex_t> & fB,
    grid::flat3<emf::complex_t> & fJ,
    float2 const dk, float const dt )
{

    emf::complex_t * const __restrict__ fEtx = fEt.x();
    emf::complex_t * const __restrict__ fEty = fEt.y();
    emf::complex_t * const __restrict__ fEtz = fEt.z();

    emf::complex_t * const __restrict__ fBx  = fB.x();
    emf::complex_t * const __restrict__ fBy  = fB.y();
    emf::complex_t * const __restrict__ fBz  = fB.z();

    emf::complex_t * const __restrict__ fJx = fJ.x();
    emf::complex_t * const __restrict__ fJy = fJ.y();
    emf::complex_t * const __restrict__ fJz = fJ.z();

    constexpr emf::complex_t I{0,1};

    auto dims = fEt.get_local_dims();
    auto start = fEt.get_local_start();
    auto global = fEt.get_global_dims();

    #pragma omp parallel for
    for( unsigned idx = 0; idx < dims.y * dims.x; idx++ ) {
        const int ix = start.x + idx % dims.x;
        const int iy = start.y + idx / dims.x;

        const float2 kvec = grid::fft::k( make_int2(ix,iy), global, dk );
        
        const float kx = kvec.x;
        const float ky = kvec.y;

        const float k2 = kx*kx + ky*ky;
        const float k  = sqrt( k2 );

        // Compute transverse current
        const emf::complex_t kdJ  = (kx * fJx[idx] + ky * fJy[idx]);
        const emf::complex_t fJtx = (k2 > 0) ? fJx[idx] - kx * kdJ / k2 : 0;
        const emf::complex_t fJty = (k2 > 0) ? fJy[idx] - ky * kdJ / k2 : 0; 
        const emf::complex_t fJtz = fJz[idx];

        // PSATD Field advance equations
        const float C   = cos( k * dt );
        const float S_k = ( k > 0 ) ? sin( k * dt ) / k : dt;
        const emf::complex_t I1mC_k2 = ( k2 > 0 )? I * (1.0f - C) / k2 : 0;

        emf::complex_t Ex = fEtx[idx];
        emf::complex_t Ey = fEty[idx];
        emf::complex_t Ez = fEtz[idx];

        emf::complex_t Bx = fBx[idx];
        emf::complex_t By = fBy[idx];
        emf::complex_t Bz = fBz[idx];

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
 * @param fE    Fourier transform of E-field (full)
 * @param fEt   Fourier transform of E-field (transverse)
 * @param frho  Fourier transform of charge density
 * @param dims  Global grid dimensions
 * @param dk    k-space cell size
 */
void update_fE( 
    grid::flat3<emf::complex_t> & fE, 
    grid::flat3<emf::complex_t> & fEt, 
    grid::flat<emf::complex_t>  & frho_src,
    float2 const dk )
{

    emf::complex_t * const __restrict__ fEx = fE.x();
    emf::complex_t * const __restrict__ fEy = fE.y();
    emf::complex_t * const __restrict__ fEz = fE.z();

    emf::complex_t const * const __restrict__ fEtx = fEt.x();
    emf::complex_t const * const __restrict__ fEty = fEt.y();
    emf::complex_t const * const __restrict__ fEtz = fEt.z();

    emf::complex_t const * const __restrict__ frho = frho_src.data();

    constexpr emf::complex_t I{0,1};

    auto dims = fEt.get_local_dims();
    auto start = fEt.get_local_start();
    auto global = fEt.get_global_dims();

    #pragma omp parallel for
    for( unsigned idx = 0; idx < dims.y * dims.x; idx++ ) {
        const int ix = start.x + idx % dims.x;
        const int iy = start.y + idx / dims.x;

        const float2 kvec = grid::fft::k( make_int2(ix,iy), global, dk );
        
        const float kx = kvec.x;
        const float ky = kvec.y;

        const float k2 = kx*kx + ky*ky;
        const float β = ( k2 > 0 ) ? 1.f / k2 : 0;
        
        fEx[idx] = -I * kx * frho[idx] * β + fEtx[idx];
        fEy[idx] = -I * ky * frho[idx] * β + fEty[idx];
        fEz[idx] =                           fEtz[idx] ;
    }
}

/**
 * @brief Advance EM fields 1 time step (no current or charge)
 * 
 */
void emf::advance() {

    // Advance transverse fields
    advance_psatd_nocurr ( *fEt, *fB, grid::fft::dk( box ), dt );

    // Transform to real fields
    // This will already update halo values
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

    // Advance transverse fields
    advance_psatd ( *fEt , *fB,*current.fJ,grid::fft::dk( box ), dt );

    // Update total E-field
    update_fE ( 
        *fE,
        *fEt,
        *charge.frho,
        grid::fft::dk( box )
    );

    // Transform to real fields
    fft_backward -> transform( *E, *fE );
    fft_backward -> transform( *B, *fB );

    // Advance internal iteration number
    iter += 1;
}

/**
 * @brief Save EMF data to diagnostic file
 * 
 * @param field     Field to save (0:E, 1:B)
 * @param fc        Field component to save (0, 1 or 2)
 */
void emf::save( const quantity quant, fcomp::cart const fc ) const {

    std::string vfname;  // Dataset name
    std::string vflabel; // Dataset label (for plots)

    grid::tiled_vec3<float> * f = nullptr;
    grid::flat3<complex_t> * cf = nullptr;

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

/**
 * @brief Get local field energy per field component
 * 
 * @warning This function will always recalculate the energy each time it is
 *          called.
 * 
 * @param ene_E     Total E-field energy (per component)
 * @param ene_B     Total B-field energy (per component)
 */
void emf::get_energy( double3 & ene_E, double3 & ene_B ) const {

    const uint2 ntiles          = E -> get_local_ntiles();
    const uint2 tile_dims       = E -> tile_dims;
    const unsigned int ystride  = E -> tile_ext_dims.x;

    double ex, ey, ez, bx, by, bz;
    ex = ey = ez = 0;
    bx = by = bz = 0;

    // Loop over all local tiles
    #pragma omp parallel for \
        reduction(+:ex,ey,ez) \
        reduction(+:bx,by,bz)
    for( unsigned tid = 0; tid < ntiles.y * ntiles.x; ++tid ) {
        float3 * const __restrict__ d_E = E -> tile_data(tid);
        float3 * const __restrict__ d_B = B -> tile_data(tid);

        double3 tile_ene_E = make_double3(0,0,0);
        double3 tile_ene_B = make_double3(0,0,0);

        // Loop over inner tile cells 
        for( unsigned iy = 0; iy < tile_dims.y; ++iy ) {
            for( unsigned ix = 0; ix < tile_dims.y; ++ix ) {
                float3 const efld = d_E[ iy * ystride + ix ];
                float3 const bfld = d_B[ iy * ystride + ix ];

                tile_ene_E.x += efld.x * efld.x;
                tile_ene_E.y += efld.y * efld.y;
                tile_ene_E.z += efld.z * efld.z;

                tile_ene_B.x += bfld.x * bfld.x;
                tile_ene_B.y += bfld.y * bfld.y;
                tile_ene_B.z += bfld.z * bfld.z;
            }
        }
        
        ex += tile_ene_E.x; ey += tile_ene_E.y; ez += tile_ene_E.z;
        bx += tile_ene_B.x; by += tile_ene_B.y; bz += tile_ene_B.z;
    }

    ene_E = make_double3(ex, ey, ez);
    ene_B = make_double3(bx, by, bz);

    ene_E.x *= 0.5 * dx.x * dx.y;
    ene_E.y *= 0.5 * dx.x * dx.y;
    ene_E.z *= 0.5 * dx.x * dx.y;

    ene_B.x *= 0.5 * dx.x * dx.y;
    ene_B.y *= 0.5 * dx.x * dx.y;
    ene_B.z *= 0.5 * dx.x * dx.y;

}

} // namespace em2ds
