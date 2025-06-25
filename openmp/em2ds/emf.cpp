#include "emf.h"

#include <iostream>
#include "zdf-cpp.h"

/**
 * @brief Construct a new EMF::EMF object
 * 
 * @param ntiles    Number of tiles
 * @param nx        Tile grid size
 * @param box       Simulation box size
 * @param dt        Time step
 * @param q         Sycl queue
 */
EMF::EMF( uint2 const ntiles, uint2 const nx, float2 const box,
    double const dt ) : 
    dx( float2{ box.x / ( nx.x * ntiles.x ), box.y / ( nx.y * ntiles.y ) } ),
    dt( dt ), box(box)
{
    // Verify Courant condition
    float cour = std::sqrt( 1.0f/( 1.0f/(dx.x*dx.x) + 1.0f/(dx.y*dx.y) ) );
    if ( dt >= cour ){
        std::cerr << "(*error*) Invalid timestep, courant condition violation.\n";
        std::cerr << "(*error*) For the current resolution [" << dx.x << "," << dx.y << "]\n";
        std::cerr << " the maximum timestep is dt = " << cour <<'\n';
        exit(-1);
    }

    // Guard cells (1 below, 2 above)
    // These are required for the Yee solver AND for field interpolation
    bnd<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {1,2};

    E = new vec3grid<float3> ( ntiles, nx, gc );
    E -> name = "Electric field";

    B = new vec3grid<float3> ( ntiles, nx, gc );
    B -> name = "Magnetic field";

    auto fdims = fft::fdims( E -> dims );

    fE  = new basic_grid3< std::complex<float> >( fdims );
    fEt = new basic_grid3< std::complex<float> >( fdims );
    fB  = new basic_grid3< std::complex<float> >( fdims );

    // Create FFT plan
    fft_backward = new fft::plan( *fE, *E );

    fE  -> name = "F(E)";
    fEt -> name = "F(Et)";
    fB  -> name = "F(B)";

    // Zero fields
    E -> zero();
    B -> zero();

    fE  -> zero();
    fEt -> zero();
    fB  -> zero();

    // Set default boundary conditions to periodic
    bc = emf::bc_type (emf::bc::periodic);

    // Reset iteration number
    iter = 0;

}

void advance_psatd_nocurr( 
    std::complex<float> * const __restrict__ fEt, 
    std::complex<float> * const __restrict__ fB,
    uint2 const dims, float2 const dk, float const dt )
{

    std::complex<float> * const __restrict__ fEtx = & fEt[                   0 ];
    std::complex<float> * const __restrict__ fEty = & fEt[     dims.x * dims.y ];
    std::complex<float> * const __restrict__ fEtz = & fEt[ 2 * dims.x * dims.y ];

    std::complex<float> * const __restrict__ fBx  = & fB[                   0 ];
    std::complex<float> * const __restrict__ fBy  = & fB[     dims.x * dims.y ];
    std::complex<float> * const __restrict__ fBz  = & fB[ 2 * dims.x * dims.y ];

    constexpr std::complex<float> I{0,1};

    #pragma omp parallel for
    for( unsigned idx = 0; idx < dims.y * dims.x; idx++ ) {
        int ix = idx % dims.x;
        int iy = idx / dims.x;

        const float ky = ((iy < int(dims.y)/2) ? iy : (iy - int(dims.y)) ) * dk.y;

        const float kx = ix * dk.x;
        const float k2 = kx*kx + ky*ky;
        const float k  = sqrtf( k2 );

        // PSATD Field advance equations
        const float C   = cosf( k * dt );
        const float S_k = ( k > 0 ) ? sinf( k * dt ) / k : dt;

        std::complex<float> Ex = fEtx[idx];
        std::complex<float> Ey = fEty[idx];
        std::complex<float> Ez = fEtz[idx];

        std::complex<float> Bx = fBx[idx];
        std::complex<float> By = fBy[idx];
        std::complex<float> Bz = fBz[idx];

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
void advance_psatd( std::complex<float> * const __restrict__ fEt, 
                    std::complex<float> * const __restrict__ fB,
                    std::complex<float> * const __restrict__ fJ,
                    uint2 const dims, float2 const dk, float const dt )
{

    std::complex<float> * const __restrict__ fEtx = & fEt[                   0 ];
    std::complex<float> * const __restrict__ fEty = & fEt[     dims.x * dims.y ];
    std::complex<float> * const __restrict__ fEtz = & fEt[ 2 * dims.x * dims.y ];

    std::complex<float> * const __restrict__ fBx = & fB[                   0 ];
    std::complex<float> * const __restrict__ fBy = & fB[     dims.x * dims.y ];
    std::complex<float> * const __restrict__ fBz = & fB[ 2 * dims.x * dims.y ];

    std::complex<float> * const __restrict__ fJx = & fJ[                   0 ];
    std::complex<float> * const __restrict__ fJy = & fJ[     dims.x * dims.y ];
    std::complex<float> * const __restrict__ fJz = & fJ[ 2 * dims.x * dims.y ];


    constexpr std::complex<float> I{0,1};

    #pragma omp parallel for
    for( unsigned idx = 0; idx < dims.y * dims.x; idx++ ) {
        int ix = idx % dims.x;
        int iy = idx / dims.x;

        const float kx = ix * dk.x;
        const float ky = ((iy < int(dims.y)/2) ? iy : (iy - int(dims.y)) ) * dk.y;

        const float k2 = kx*kx + ky*ky;
        const float k  = sqrt( k2 );

        // Calculate transverse current
        const std::complex<float> kdJ  = (kx * fJx[idx] + ky * fJy[idx]);
        const std::complex<float> fJtx = (k2 > 0) ? fJx[idx] - kx * kdJ / k2 : 0;
        const std::complex<float> fJty = (k2 > 0) ? fJy[idx] - ky * kdJ / k2 : 0; 
        const std::complex<float> fJtz = fJz[idx];

        // PSATD Field advance equations
        const float C   = cos( k * dt );
        const float S_k = ( k > 0 ) ? sin( k * dt ) / k : dt;
        const std::complex<float> I1mC_k2 = ( k2 > 0 )? I * (1.0f - C) / k2 : 0;

        std::complex<float> Ex = fEtx[idx];
        std::complex<float> Ey = fEty[idx];
        std::complex<float> Ez = fEtz[idx];

        std::complex<float> Bx = fBx[idx];
        std::complex<float> By = fBy[idx];
        std::complex<float> Bz = fBz[idx];

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
void update_fE( std::complex<float> * const __restrict__ d_fE, 
                std::complex<float> * const __restrict__ d_fEt, 
                std::complex<float> * const __restrict__ d_frho,
                uint2 const dims, float2 const dk )
{

    std::complex<float> * const __restrict__ fEx = & d_fE[                   0 ];
    std::complex<float> * const __restrict__ fEy = & d_fE[     dims.x * dims.y ];
    std::complex<float> * const __restrict__ fEz = & d_fE[ 2 * dims.x * dims.y ];

    std::complex<float> const * const __restrict__ fEtx = & d_fEt[                   0 ];
    std::complex<float> const * const __restrict__ fEty = & d_fEt[     dims.x * dims.y ];
    std::complex<float> const * const __restrict__ fEtz = & d_fEt[ 2 * dims.x * dims.y ];

    std::complex<float> const * const __restrict__ frho = & d_frho[                   0 ];

    constexpr std::complex<float> I{0,1};

    #pragma omp parallel for
    for( unsigned idx = 0; idx < dims.y * dims.x; idx++ ) {
        int ix = idx % dims.x;
        int iy = idx / dims.x;

        const float ky = ((iy < int(dims.y)/2) ? iy : (iy - int(dims.y)) ) * dk.y;

        const float kx = ix * dk.x;
        const float k2 = kx*kx + ky*ky;
        const float β = ( k2 > 0 ) ? 1.f / k2 : 0;
        
        fEx[idx] = -I * kx * frho[idx] * β + fEtx[idx];
		fEy[idx] = -I * ky * frho[idx] * β + fEty[idx];
		fEz[idx] =                                fEtz[idx] ;
    }
}

/**
 * @brief Advance EM fields 1 time step (no current or charge)
 * 
 */
void EMF::advance() {

    // Advance transverse fields
    advance_psatd_nocurr ( fEt -> d_buffer, fB -> d_buffer,  fEt -> dims, fft::dk( box ), dt );

    // Transform to real fields
    fft_backward -> transform( *fEt, *E );
    fft_backward -> transform( *fB, *B );

    // Update guard cell values
    E -> copy_to_gc();
    B -> copy_to_gc();

    // Advance internal iteration number
    iter += 1;
}

/**
 * @brief Advance EM fields 1 time step including current
 * 
 * @param current   Electric current
 * @param charge    Electric charge
 */
void EMF::advance( Current & current, Charge & charge ) {

    // Advance transverse fields
    advance_psatd ( 
        fEt -> d_buffer,
        fB -> d_buffer,
        current.fJ -> d_buffer,
        fEt -> dims, fft::dk( box ), dt
    );

    // Update total E-field
    update_fE ( 
        fE -> d_buffer,
        fEt -> d_buffer,
        charge.frho -> d_buffer,
        fE -> dims, fft::dk( box )
    );

    // Transform to real fields
    fft_backward -> transform( *fE, *E );
    fft_backward -> transform( *fB, *B );

    // Update guard cell values
    E -> copy_to_gc();
    B -> copy_to_gc();

    // Advance internal iteration number
    iter += 1;
}

/**
 * @brief Save EMF data to diagnostic file
 * 
 * @param field     Field to save (0:E, 1:B)
 * @param fc        Field component to save (0, 1 or 2)
 */
void EMF::save( const emf::field field, fcomp::cart const fc ) {

    std::string vfname;  // Dataset name
    std::string vflabel; // Dataset label (for plots)

    vec3grid<float3> * f = nullptr;
    basic_grid3<std::complex<float>> * cf = nullptr;

    switch (field ) {
        case emf::e :
            f = E;
            vfname = "E";
            vflabel = "E_";
            break;
        case emf::b :
            f = B;
            vfname = "B";
            vflabel = "B_";
            break;
        case emf::fe :
            cf = fE;
            vfname = "fE";
            vflabel = "\\mathcal{F} E_";
            break;
        case emf::fet :
            cf = fEt;
            vfname = "fEt";
            vflabel = "\\mathcal{F} E^\\perp_";
            break;
        case emf::fb :
            cf = fB;
            vfname = "fB";
            vflabel = "\\mathcal{F} B_";
            break;
        default:
            std::cerr << "Invalid field type selected, aborting\n";
            std::exit(1);
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
            std::cerr << "Invalid field component (fc) selected, aborting\n";
            std::exit(1);
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

    if ( field == emf::e || field == emf::b ) {
        axis[0] = (zdf::grid_axis) {
            .name = (char *) "x",
            .min = 0.0,
            .max = box.x,
            .label = (char *) "x",
            .units = (char *) "c/\\omega_n"
        };

        axis[1] = (zdf::grid_axis) {
            .name = (char *) "y",
            .min = 0.0,
            .max = box.y,
            .label = (char *) "y",
            .units = (char *) "c/\\omega_n"
        };

        info.axis = axis;
        info.count[0] = E -> dims.x;
        info.count[1] = E -> dims.y;

        f -> save( fc, info, iteration, "EMF" );

    } else {
        float2 dk = fft::dk( box );

        axis[0] = (zdf::grid_axis) {
            .name = (char *) "kx",
            .min = 0.0,
            .max = (fEt -> dims.x - 1) * dk.x,
            .label = (char *) "k_x"
        };

        axis[1] = (zdf::grid_axis) {
            .name = (char *) "ky",
            .min =  - dk.y * ( fEt -> dims.y / 2 ),
            .max =    dk.y * ( fEt -> dims.y / 2 - 1 ),
            .label = (char *) "k_y"
        };

        info.axis = axis;
        info.count[0] = fEt -> dims.x;
        info.count[1] = fEt -> dims.y;

        cf -> save( fc, info, iteration, "EMF" );
    }
}

/**
 * @brief Get total field energy per field component
 * 
 * @warning This function will always recalculate the energy each time it is
 *          called.
 * 
 * @param ene_E     Total E-field energy (per component)
 * @param ene_B     Total B-field energy (per component)
 */
void EMF::get_energy( double3 & ene_E, double3 & ene_B ) {

    ene_E = make_double3(0,0,0);
    ene_B = make_double3(0,0,0);

    const uint2 ntiles          = E -> ntiles;
    const unsigned int tile_vol = E -> tile_vol;
    const uint2 nx              = E -> nx;
    const unsigned int offset   = E -> offset;
    const unsigned int ystride  = E -> ext_nx.x;

    float3 * const __restrict__ d_E = E -> d_buffer;
    float3 * const __restrict__ d_B = B -> d_buffer;

    // Loop over tiles
    for( unsigned ty = 0; ty < ntiles.y; ++ty ) {
        for( unsigned tx = 0; tx < ntiles.x; ++tx ) {

            const auto tile_idx = make_uint2( tx, ty );
            const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
            const auto tile_off = tid * tile_vol + offset ;

            auto tile_ene_E = make_double3(0,0,0);
            auto tile_ene_B = make_double3(0,0,0);

            // Loop over cells
            for( unsigned iy = 0; iy < nx.y; ++iy ) {
                for( unsigned ix = 0; ix < nx.y; ++ix ) {
                    float3 const efld = d_E[ tile_off + iy * ystride + ix ];
                    float3 const bfld = d_B[ tile_off + iy * ystride + ix ];

                    tile_ene_E.x += efld.x * efld.x;
                    tile_ene_E.y += efld.y * efld.y;
                    tile_ene_E.z += efld.z * efld.z;

                    tile_ene_B.x += bfld.x * bfld.x;
                    tile_ene_B.y += bfld.y * bfld.y;
                    tile_ene_B.z += bfld.z * bfld.z;
                }
            }
            
            // reduce(add) data inside tile
            
            {   // Only 1 thread per tile does this
                // Atomic ops
                ene_E += tile_ene_E;
                ene_B += tile_ene_B;
            }

        }
    }

    ene_E.x *= 0.5 * dx.x * dx.y;
    ene_E.y *= 0.5 * dx.x * dx.y;
    ene_E.z *= 0.5 * dx.x * dx.y;

    ene_B.x *= 0.5 * dx.x * dx.y;
    ene_B.y *= 0.5 * dx.x * dx.y;
    ene_B.z *= 0.5 * dx.x * dx.y;

}