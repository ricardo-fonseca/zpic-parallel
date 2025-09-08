#include "emf.h"
#include "zpic.h"

#include <iostream>


/**
 * @brief B advance for Yee algorithm (mode 0)
 * 
 * @param E         Pointer to 0,0 coordinate of E field
 * @param B         Pointer to 0,0 coordinate of B field
 * @param nx        Internal tile size
 * @param jstride   Stride for j coordinate
 * @param dt        Time step
 * @param dx        Cell size (z,r)
 * @param ir0       Coordinate of lower tile corner in grid
 */
void yee0_b( 
    cyl_float3 const * const __restrict__ E, 
    cyl_float3 * const __restrict__ B, 
    uint2 const nx, int const jstride, 
    const double dt, float2 const dx, const int ir0 )
{
    // cylindrical cell sizes
    const auto dz    = dx.x;
    const auto dr    = dx.y;

    const auto dt_dz = dt / dz;
    const auto dt_dr = dt / dr;

    // Standard advance away from axial boundary
    const int j0 = ( ir0 > 0 ) ? -1 : 1;
    
    for( int j = j0; j < static_cast<int>(nx.y) + 1; j++ ) {
        
        /// @brief r at lower edge of j cell, normalized to Δr
        float rm  = ir0 + j - 0.5f;
        /// @brief r at upper edge of j cell, normalized to Δr
        float rp  = ir0 + j + 0.5f;
        /// @brief Δt/r at the center of j cell
        float t   = dt / ( ( ir0 + j ) * dr );

        for( int i = -1; i < static_cast<int>(nx.x) + 1; i++) {
            B[ i + j*jstride ].r += (   dt_dz * ( E[(i+1) + j*jstride].θ - E[i + j*jstride].θ ) );  

            B[ i + j*jstride ].θ += ( - dt_dz * ( E[(i+1) + j*jstride].r - E[i + j*jstride].r ) + 
                                        dt_dr * ( E[i + (j+1)*jstride].z - E[i + j*jstride].z ) );  

            B[ i + j*jstride ].z += ( - t * ( rp * E[i + (j+1)*jstride].θ - rm * E[i + j*jstride].θ ) );  
        }
    }

    // Solve for axial boundary if needed
    if ( ir0 == 0 ) {
        for( int i = -1; i < static_cast<int>(nx.x) + 1; i++) {

            B[ i +   0 *jstride ].r = - B[ i + 1*jstride ].r;  
            // B[ i + (-1)*jstride ].r = - B[ i + 2*jstride ].r;  // not used

            B[ i +   0  *jstride ].θ = 0;
            // B[ i +  (-1)*jstride ].θ = - B[ i + 1*jstride ].θ; // not used

            B[ i +   0 *jstride ].z += - 4 * dt_dr * E[ i + 1*jstride ].θ;  
            // B[ i + (-1)*jstride ].z  = B[ i + 1 *jstride ].z; // not used

        }
    }

}

/**
 * @brief E advance for Yee algorithm ( no current, mode 0 )
 * 
 * @param E         Pointer to 0,0 coordinate of E field
 * @param B         Pointer to 0,0 coordinate of B field
 * @param nx        Internal tile size
 * @param jstride   Stride for j coordinate
 * @param dt        Time step
 * @param dx        Cell size (z,r)
 * @param ir0       Coordinate of lower tile corner in grid
 */
void yee0_e( 
    cyl_float3 * const __restrict__ E, 
    cyl_float3 const * const __restrict__ B, 
    uint2 const nx, int const jstride, 
    const double dt, float2 const dx, const int ir0 )
{
    // cylindrical cell sizes
    const auto dz    = dx.x;
    const auto dr    = dx.y;

    const auto dt_dz = dt / dz;
    const auto dt_dr = dt / dr;

    // Standard advance away from axial boundary
    const int j0 = ( ir0 > 0 ) ? 0 : 1;

    for( int j = j0; j < static_cast<int>(nx.y) + 2; j ++ ) {

        /// @brief r at center of j-1 cell, normalized to Δr
        float rcm  = ir0 + j - 1;
        /// @brief r at center of j cell, normalized to Δr
        float rc   = ir0 + j;
        /// @brief Δt/r at the lower edge of j cell
        float dt_rm = dt / ( ( ir0 + j - 0.5 ) * dr );

        for( int i = 0; i < static_cast<int>(nx.x) + 2; i++ ) {

            E[i + j*jstride].r += ( - dt_dz * ( B[i + j*jstride].θ - B[(i-1) + j*jstride].θ) );

            E[i + j*jstride].θ += ( + dt_dz * ( B[i + j*jstride].r - B[(i-1) + j*jstride].r) - 
                                      dt_dr * ( B[i + j*jstride].z - B[i + (j-1)*jstride].z) );

            E[i + j*jstride].z += ( + dt_rm * ( rc * B[i + j*jstride].θ - rcm * B[i + (i-1)*jstride].θ) );
        }
    }

    // Solve for axial boundary if needed
    if ( ir0 == 0 ) {
        for( int i = 0; i < static_cast<int>(nx.x) + 2; i++) {
            E[i +   0 *jstride].r = 0;
            // E[i + (-1)*jstride].r = -E[i + 1*jstride].r; // not used

            E[i +   0 *jstride].θ = -E[i + 1*jstride].θ;
            // E[i + (-1)*jstride].θ = -E[i + 2*jstride].θ; // not used

            E[i +   0 *jstride].z = E[i + 1*jstride].z;
            // E[i + (-1)*jstride].z = E[i + 2*jstride].z; // not used
        }
    }
}

/**
 * @brief B advance for Yee algorithm (mode m)
 * 
 * @param m         Mode
 * @param E         Pointer to 0,0 coordinate of E field
 * @param B         Pointer to 0,0 coordinate of B field
 * @param nx        Internal tile size
 * @param jstride   Stride for j coordinate
 * @param dt        Time step
 * @param dx        Cell size (z,r)
 * @param ir0       Coordinate of lower tile corner in grid
 */
void yeem_b(
    const int m, 
    cyl_cfloat3 const * const __restrict__ E, 
    cyl_cfloat3 * const __restrict__ B, 
    uint2 const nx, int const jstride, 
    const double dt, float2 const dx, const int ir0 )
{
    // cylindrical cell sizes
    const auto dz    = dx.x;
    const auto dr    = dx.y;

    const float dt_dz = dt / dz;
    const float dt_dr = dt / dr;

    // Standard advance away from axial boundary
    const int j0 = ( ir0 > 0 ) ? -1 : 1;

    ///@brief m * imaginary unit
    const std::complex<float> mI{0,static_cast<float>(m)};
    ///@brief imaginary unit
    constexpr std::complex<float> I{0,1};

    for( int j = j0; j < static_cast<int>(nx.y) + 1; j++ ) {
        
        /// @brief r/Δr at lower edge of j cell
        float rm  = ir0 + j - 0.5f;
        /// @brief r/Δr at upper edge of j cell
        float rp  = ir0 + j + 0.5f;
        /// @brief Δt/r at the center of j cell
        float dt_rc = dt / ( ( ir0 + j ) * dr );
        /// @brief Δt/r at the lower edge of j cell
        float dt_rm = dt / ( ( ir0 + j - 0.5 ) * dr );

        for( int i = -1; i < static_cast<int>(nx.x) + 1; i++) {

            B[ i + j*jstride ].r += (   
                + dt_dz * ( E[(i+1) + j*jstride].θ - E[i + j*jstride].θ )   // Δt ∂Eθ/∂z
                - dt_rm * mI * E[ i + j * jstride ].z                       // (Δt/r) m I Ez
            );  

            B[ i + j*jstride ].θ += ( 
                - dt_dz * ( E[(i+1) + j*jstride].r - E[i + j*jstride].r )   // Δt ∂Er/∂z
                + dt_dr * ( E[i + (j+1)*jstride].z - E[i + j*jstride].z )   // Δt ∂Ez/∂r
            );  

            B[ i + j*jstride ].z += - dt_rc * (                                // Δt/r
                + ( rp * E[i + (j+1)*jstride].θ - rm * E[i + j*jstride].θ )    // ∂(r Eθ)/∂r 
                - mI * E[ i + j * jstride ].r                                  // m I Er
            );  
        }
    }

    // Solve for axial boundary if needed
    if ( ir0 == 0 ) {
        if ( m == 1 ) {
            // Mode m = 1 is a special case
            for( int i = -1; i < static_cast<int>(nx.x) + 1; i++) {
                B[ i + 0 *jstride ].r = + B[ i + 1*jstride ].r;  
                B[ i + 0 *jstride ].θ = - 0.125f * I * (9.f * B[ i + 1*jstride ].r - B[ i + 2*jstride ].r );
                B[ i + 0 *jstride ].z = 0;  
            }
        } else {
            for( int i = -1; i < static_cast<int>(nx.x) + 1; i++) {
                B[ i + 0 *jstride ].r = - B[ i + 1*jstride ].r;  // Br(r=0) = 0
                B[ i + 0 *jstride ].θ = 0;
                B[ i + 0 *jstride ].z = 0;  
            }
        }
    }
}

/**
 * @brief E advance for Yee algorithm ( no current, mode m )
 * 
 * @param m         Mode
 * @param E         Pointer to 0,0 coordinate of E field
 * @param B         Pointer to 0,0 coordinate of B field
 * @param nx        Internal tile size
 * @param jstride   Stride for j coordinate
 * @param dt        Time step
 * @param dx        Cell size (z,r)
 * @param ir0       Coordinate of lower tile corner in grid
 */
void yeem_e( 
    const int m,
    cyl_cfloat3 * const __restrict__ E, 
    cyl_cfloat3 const * const __restrict__ B, 
    uint2 const nx, int const jstride, 
    const double dt, float2 const dx, const int ir0 )
{
    // cylindrical cell sizes
    const auto dz    = dx.x;
    const auto dr    = dx.y;

    const float dt_dz = dt / dz;
    const float dt_dr = dt / dr;

    // Standard advance away from axial boundary
    const int j0 = ( ir0 > 0 ) ? 0 : 1;

    /// @brief m * imaginary unit
    const std::complex<float> mI{0,static_cast<float>(m)};

    for( int j = j0; j < static_cast<int>(nx.y) + 2; j ++ ) {

        /// @brief r/Δr at center of j-1 cell
        float rcm  = ir0 + j - 1;
        /// @brief rΔr at center of j cell
        float rc   = ir0 + j;
        /// @brief Δt/r at the lower edge of j cell
        float dt_rm = dt / ( ( ir0 + j - 0.5 ) * dr );
        /// @brief Δt/r at the center of j cell
        float dt_rc = dt / ( ( ir0 + j ) * dr );

        for( int i = 0; i < static_cast<int>(nx.x) + 2; i++ ) {           
            E[i + j*jstride].r += ( 
                - dt_dz * ( B[i + j*jstride].θ - B[(i-1) + j*jstride].θ)   // Δt ∂Bθ/∂z 
                + dt_rc * mI * B[ i + j * jstride ].z                      // (Δt/r) m I Bz
            );

            E[i + j*jstride].θ += ( + dt_dz * ( B[i + j*jstride].r - B[(i-1) + j*jstride].r)
                                    - dt_dr * ( B[i + j*jstride].z - B[i + (j-1)*jstride].z) );

            E[i + j*jstride].z +=  dt_rm * (                                // Δt/r
                + rc * B[i + j * jstride].θ - rcm * B[i + (j-1)*jstride].θ    // ∂(r Bθ)/∂r 
                - mI * B[i + j * jstride].r                               // m I Br
            );

        }
    }

    // Solve for axial boundary if needed
    if ( ir0 == 0 ) {
        if ( m == 1 ) {
            // Mode m = 1 is a special case
            for( int i = 0; i < static_cast<int>(nx.x) + 2; i++) {
                E[ i + 0 *jstride ].r = ( 4.f * E[ i + 1*jstride ].r - E[ i + 2*jstride ].r ) / 3.f;  
                E[ i + 0 *jstride ].θ =  E[ i + 1 * jstride ].θ;   // ∂Bθ/∂r(r=0) = 0
                E[ i + 0 *jstride ].z = -E[ i + 1 * jstride ].z;  // Ez(r=0) = 0
            }
        } else {
            for( int i = 0; i < static_cast<int>(nx.x) + 2; i++) {
                E[ i + 0 *jstride ].r = 0;
                E[ i + 0 *jstride ].θ = - E[ i + 1 *jstride ].θ;    // Eθ(r=0) = 0
                E[ i + 0 *jstride ].z = - E[ i + 1 *jstride ].z;    // Ez(r=0) = 0;  
            }
        }
    }
}

/**
 * @brief Construct a new EMF object
 * 
 * @param nmodes    Number of cylindrical modes (>= 1)
 * @param ntiles    Number of tiles in z,r direction
 * @param nx        Tile size (#cells)
 * @param box       Simulation box size (sim. units)
 * @param dt        Time step
 */
EMF::EMF( unsigned int nmodes, uint2 const ntiles, uint2 const nx, float2 const box,
     double const dt ) : 
    dx( make_float2( box.x / ( nx.x * ntiles.x ), box.y / ( nx.y * ntiles.y ) ) ),
    dt( dt ),
    nmodes( nmodes ),
    box(box)
{
    auto cour = zpic::courant( dx );
    if ( dt >= cour ){
        std::cerr << "(*error*) Invalid timestep, courant condition violation.\n";
        std::cerr << "(*error*) For the current resolution " << dx;
        std::cerr << " the maximum timestep is dt = " << cour <<'\n';
        exit(-1);
    }

    // Guard cells (1 below, 2 above)
    // These are required for the Yee solver AND for field interpolation
    bnd<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {1,2};

    E = new Cyl3CylGrid<float> ( nmodes, ntiles, nx, gc );
    E -> set_name( "E" );

    B = new Cyl3CylGrid<float> ( nmodes, ntiles, nx, gc );
    B -> set_name( "B" );

    // Zero fields
    E -> zero();
    B -> zero();

    // Set default boundary conditions
    bc.x.lower = bc.x.upper = emf::bc::periodic;
    bc.y.lower = emf::bc::axial; 
    bc.y.upper = emf::bc::none;

    // Reset iteration number
    iter = 0;

    std::cout << "completed EMF() constructor\n";
};


/**
 * @brief Advance EM fields 1 time step (no current)
 * 
 */
void EMF::advance() {

    // Get tile information from mode0
    auto & E0 = E -> mode0();
    auto & B0 = B -> mode0();

    const auto ntiles    = E0.ntiles;
    const auto field_vol = E0.tile_vol;
    const auto nx        = E0.nx;
    const auto offset    = E0.offset;
    const int  jstride   = E0.ext_nx.x;

    // Solve for mode 0
    #pragma omp parallel for
    for( unsigned tid = 0; tid < ntiles.x * ntiles.y; tid ++ ) {

        int ir0 = ( tid / ntiles.x ) * nx.y;
        const auto tile_off = tid * field_vol;

        // Copy E and B into shared memory
        cyl_float3 E_local[ field_vol ];
        cyl_float3 B_local[ field_vol ];
        for( unsigned i = 0; i < field_vol; i++ ) {
            E_local[i] = E0.d_buffer[ tile_off + i ];
            B_local[i] = B0.d_buffer[ tile_off + i ];
        }

        auto * const __restrict__ tile_E = & E_local[ offset ];
        auto * const __restrict__ tile_B = & B_local[ offset ];

        yee0_b( tile_E, tile_B, nx, jstride, dt/2, dx, ir0 );
        yee0_e( tile_E, tile_B, nx, jstride, dt,   dx, ir0 );
        yee0_b( tile_E, tile_B, nx, jstride, dt/2, dx, ir0 );

        // Copy data to global memory
        for( unsigned i = 0; i < field_vol; i++ ) {
            E0.d_buffer[ tile_off + i ] = E_local[i];
            B0.d_buffer[ tile_off + i ] = B_local[i];
        }
    }

    // Solve for high-order modes
    for( int m = 1; m < nmodes; m++ ) {
        auto & Em = E -> mode(m);
        auto & Bm = B -> mode(m);

        #pragma omp parallel for
        for( unsigned tid = 0; tid < ntiles.x * ntiles.y; tid ++ ) {

            const auto tile_off = tid * field_vol;
            int ir0 = ( tid / ntiles.x ) * nx.y;
 
            // Copy E and B into shared memory
            cyl_cfloat3 E_local[ field_vol ];
            cyl_cfloat3 B_local[ field_vol ];
            for( unsigned i = 0; i < field_vol; i++ ) {
                E_local[i] = Em.d_buffer[ tile_off + i ];
                B_local[i] = Bm.d_buffer[ tile_off + i ];
            }

            auto * const __restrict__ tile_E = & E_local[ offset ];
            auto * const __restrict__ tile_B = & B_local[ offset ];

            yeem_b( m, tile_E, tile_B, nx, jstride, dt/2, dx, ir0 );
            yeem_e( m, tile_E, tile_B, nx, jstride, dt,   dx, ir0 );
            yeem_b( m, tile_E, tile_B, nx, jstride, dt/2, dx, ir0 );

            // Copy data to global memory
            for( unsigned i = 0; i < field_vol; i++ ) {
                Em.d_buffer[ tile_off + i ] = E_local[i];
                Bm.d_buffer[ tile_off + i ] = B_local[i];
            }
        }        
    }

    // Update guard cells with new values
    E -> copy_to_gc();
    B -> copy_to_gc();

    // Do additional bc calculations if needed
    // process_bc();

    // Advance internal iteration number
    iter += 1;    

}

/**
 * @brief Save EM field component to file
 * 
 * @param field     Which field to save (E or B)
 * @param fc        Which field component to save (r, θ or z)
 * @param m         Mode
 */
void EMF::save( emf::field const field, const fcomp::cyl fc, unsigned m ) {
    std::string vfname;  // Dataset name
    std::string vflabel; // Dataset label (for plots)
    std::string path{"EMF"};

    switch (field ) {
        case emf::e :
            vfname = "E" + std::to_string(m);
            vflabel = "E^" + std::to_string(m) + "_";
            break;
        case emf::b :
            vfname = "B" + std::to_string(m);
            vflabel = "B^" + std::to_string(m) + "_";
            break;
        default:
            std::cerr << "(*error*) Invalid field type selected, returning..." << std::endl;
            return;
    }

    switch ( fc ) {
        case( fcomp::z ) :
            vfname  += "z";
            vflabel += "z";
            break;
        case( fcomp::r ) :
            vfname  += "r";
            vflabel += "r";
            break;
        case( fcomp::θ ) :
            vfname  += "θ";
            vflabel += "\\theta";
            break;
        default:
            std::cerr << "(*error*) Invalid field component (fc) selected, returning..." << std::endl;
            return;
    }

    zdf::grid_axis axis[2];
    axis[0] = (zdf::grid_axis) {
        .name = (char *) "z",
        .min = 0.0 + moving_window.motion(),
        .max = box.x + moving_window.motion(),
        .label = (char *) "z",
        .units = (char *) "c/\\omega_n"
    };

    axis[1] = (zdf::grid_axis) {
        .name = (char *) "r",
        .min = -dx.y/2,
        .max = box.y-dx.y/2,
        .label = (char *) "r",
        .units = (char *) "c/\\omega_n"
    };

    zdf::grid_info info = {
        .name = (char *) vfname.c_str(),
        .label = (char *) vflabel.c_str(),
        .units = (char *) "m_e c \\omega_n e^{-1}",
        .axis = axis
    };

    zdf::iteration iteration = {
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_n"
    };

    switch (field ) {
        case emf::e :
            E -> save( m, fc, info, iteration, path );
            break;
        case emf::b :
            B -> save( m, fc, info, iteration, path );
            break;
        default:
            std::cerr << "(*error*) Invalid field type selected, returning..." << std::endl;
            return;
    }
}

/**
 * @brief Get EM field energy
 * 
 * Energy is reported per mode
 * 
 * @note The energy will be recalculated each time this routine is called;
 * @note Axial cell values are currently being ignored
 * 
 * @param ene_E     Electric field energy
 * @param ene_b     Magnetic field energy
 */
void EMF::get_energy( cyl_double3 ene_E[], cyl_double3 ene_B[] ) {

    // Get tile information from mode0
    auto & E0 = E -> mode0();
    auto & B0 = B -> mode0();

    const auto ntiles    = E0.ntiles;
    const auto field_vol = E0.tile_vol;
    const auto nx        = E0.nx;
    const auto offset    = E0.offset;
    const int  jstride   = E0.ext_nx.x;

    const double dz = dx.x;
    const double dr = dx.y;

    for( int i = 0; i < nmodes; i++ ) {
        ene_E[i] = ene_B[i] = cyl_double3{0};
    }

    // Get energy for mode 0
    {
        double ene_Er{0}, ene_Eθ{0}, ene_Ez{0};
        double ene_Br{0}, ene_Bθ{0}, ene_Bz{0};

        #pragma omp parallel for \
            reduction(+:ene_Er,ene_Eθ,ene_Ez,ene_Br,ene_Bθ,ene_Bz)
        for( unsigned tid = 0; tid < ntiles.x * ntiles.y; tid ++ ) {    
            int ir0 = ( tid / ntiles.x ) * nx.y;
            const auto tile_off = tid * field_vol + offset;

            cyl_float3 * const __restrict__ tile_E = (cyl_float3 *) & E0.d_buffer[ tile_off ];
            cyl_float3 * const __restrict__ tile_B = (cyl_float3 *) & B0.d_buffer[ tile_off ];

            cyl_double3 tile_ene_E{0};
            cyl_double3 tile_ene_B{0};

            // Axial cells are not included
            int jstart = ( ir0 > 0 ) ? 0 : 1;

            for( unsigned j = jstart; j < nx.y; ++j ) {
                double rc = ( j + ir0       ) * dr;
                double rm = ( j + ir0 - 0.5 ) * dr;

                for( unsigned i = 0; i < nx.x; ++i ) {
                    auto const efld = tile_E[ i + j*jstride ];
                    auto const bfld = tile_B[ i + j*jstride ];

                    tile_ene_E.r += rc * efld.r * efld.r;
                    tile_ene_E.θ += rm * efld.θ * efld.θ;
                    tile_ene_E.z += rm * efld.z * efld.z;

                    tile_ene_B.r += rm * bfld.r * bfld.r;
                    tile_ene_B.θ += rc * bfld.θ * bfld.θ;
                    tile_ene_B.z += rc * bfld.z * bfld.z;
                }
            }

            // OpenMP reductions only work with basic scalar types
            ene_Er += tile_ene_E.r; ene_Eθ += tile_ene_E.θ; ene_Ez += tile_ene_E.z;
            ene_Br += tile_ene_B.r; ene_Bθ += tile_ene_B.θ; ene_Bz += tile_ene_B.z;
        }

        ene_E[0] = cyl_double3{ ene_Ez, ene_Er, ene_Eθ };
        ene_B[0] = cyl_double3{ ene_Bz, ene_Br, ene_Bθ };
    }

    // Get energy for high order modes
    for( int m = 1; m < nmodes; m++ ) {
        auto & Em = E -> mode(m);
        auto & Bm = B -> mode(m);

        double ene_Er{0}, ene_Eθ{0}, ene_Ez{0};
        double ene_Br{0}, ene_Bθ{0}, ene_Bz{0};

        #pragma omp parallel for \
            reduction(+:ene_Er,ene_Eθ,ene_Ez,ene_Br,ene_Bθ,ene_Bz)
        for( unsigned tid = 0; tid < ntiles.x * ntiles.y; tid ++ ) {    
            int ir0 = ( tid / ntiles.x ) * nx.y;
            const auto tile_off = tid * field_vol + offset;

            cyl_cfloat3 * const __restrict__ tile_E = (cyl_cfloat3 *) & Em.d_buffer[ tile_off ];
            cyl_cfloat3 * const __restrict__ tile_B = (cyl_cfloat3 *) & Bm.d_buffer[ tile_off ];

            cyl_double3 tile_ene_E{0};
            cyl_double3 tile_ene_B{0};

            // Axial cells are not included
            int jstart = ( ir0 > 0 ) ? 0 : 1;

            for( unsigned j = jstart; j < nx.y; ++j ) {
                double rc = ( j + ir0       ) * dr;
                double rm = ( j + ir0 - 0.5 ) * dr;

                for( unsigned i = 0; i < nx.x; ++i ) {
                    auto const efld = tile_E[ i + j*jstride ];
                    auto const bfld = tile_B[ i + j*jstride ];

                    // norm(z) = |z|^2
                    tile_ene_E.r += rc * norm( efld.r );
                    tile_ene_E.θ += rm * norm( efld.θ );
                    tile_ene_E.z += rm * norm( efld.z );

                    tile_ene_B.r += rm * norm( bfld.r );
                    tile_ene_B.θ += rc * norm( bfld.θ );
                    tile_ene_B.z += rc * norm( bfld.z );
                }
            }
            ene_Er += tile_ene_E.r; ene_Eθ += tile_ene_E.θ; ene_Ez += tile_ene_E.z;
            ene_Br += tile_ene_B.r; ene_Bθ += tile_ene_B.θ; ene_Bz += tile_ene_B.z;
        }

        ene_E[m] += cyl_double3{ ene_Ez, ene_Er, ene_Eθ };
        ene_B[m] += cyl_double3{ ene_Bz, ene_Br, ene_Bθ };
    }

    // Multiply by 0.5 * cell volume
    for( int m = 0; m < nmodes; m++ ) {
        const double norm = dz * dr * M_PI;
        ene_E[m] *= norm;
        ene_B[m] *= norm;
    }
}