#include "emf.h"

#include <iostream>
#include "bnd.h"

#include "zdf-cpp.h"

/**
 * @brief Construct a new EMF object
 * 
 * @param global_ntiles     Global number of tiles
 * @param nx                Individual tile size
 * @param box               Global simulation box size
 * @param dt                Time step
 * @param parallel          Parallel partition 
 */
EMF::EMF( uint2 const global_ntiles, uint2 const nx, float2 const box,
    double const dt, Partition & parallel ) : 
    dx( make_float2( box.x / ( nx.x * global_ntiles.x ), box.y / ( nx.y * global_ntiles.y ) ) ),
    dt( dt ), box(box)
{
    // Verify Courant condition
    auto cour = std::sqrt( 1.0f/( 1.0f/(dx.x*dx.x) + 1.0f/(dx.y*dx.y) ) );
    if ( dt >= cour ){
        if ( mpi::world_root() ) {
            std::cerr << "(*error*) Invalid timestep, courant condition violation.\n";
            std::cerr << "(*error*) For the current resolution " << dx;
            std::cerr << " the maximum timestep is dt = " << cour <<'\n';
        }
        mpi::abort(1);
    }

    // Guard cells (1 below, 2 above)
    // These are required for the Yee solver AND for field interpolation
    bnd<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {1,2};

    E = new vec3grid<float3> ( global_ntiles, nx, gc, parallel );
    E -> name = "Electric field";

    B = new vec3grid<float3> ( global_ntiles, nx, gc, parallel );
    B -> name = "Magnetic field";

    // Zero fields
    E -> zero();
    B -> zero();

    // Set boundary conditions to none
    bc = emf::bc_type (emf::bc::none);

    // Reset iteration number
    iter = 0;
}


/**
 * @brief B advance for Yee algorithm
 * 
 * @param E         Pointer to 0,0 coordinate of E field
 * @param B         Pointer to 0,0 coordinate of B field
 * @param nx        Internal tile size
 * @param ystride   Stride for y coordinate
 * @param dt_dx     \Delta t / \Delta x
 * @param dt_dy     \Delta t / \Delta y
 */
void yee_b( 
    float3 const * const __restrict__ E, 
    float3 * const __restrict__ B, 
    uint2 const nx, int const ystride, 
    float2 const dt_dx )
{
    for( int iy = -1; iy < static_cast<int>(nx.y) + 1; iy++ ) {
        for( int ix = -1; ix < static_cast<int>(nx.x) + 1; ix++) {
            B[ ix + iy*ystride ].x += ( - dt_dx.y * ( E[ix + (iy+1)*ystride].z - E[ix + iy*ystride].z ) );  
            B[ ix + iy*ystride ].y += (   dt_dx.x * ( E[(ix+1) + iy*ystride].z - E[ix + iy*ystride].z ) );  
            B[ ix + iy*ystride ].z += ( - dt_dx.x * ( E[(ix+1) + iy*ystride].y - E[ix + iy*ystride].y ) + 
                                          dt_dx.y * ( E[ix + (iy+1)*ystride].x - E[ix + iy*ystride].x ) );  
        }
    }
}

/**
 * @brief E advance for Yee algorithm ( no current )
 * 
 * @param E         Pointer to 0,0 coordinate of E field
 * @param B         Pointer to 0,0 coordinate of B field
 * @param nx        Internal tile size
 * @param ystride    Stride for y coordinate
 * @param dt_dx     \Delta t / \Delta x
 * @param dt_dy     \Delta t / \Delta y
 */
void yee_e( 
    float3 * const __restrict__ E, 
    float3 const * const __restrict__ B, 
    uint2 const nx, int const ystride, 
    float2 const dt_dx )
{
    for( int iy = 0; iy < static_cast<int>(nx.y) + 2; iy ++ ) {
        for( int ix = 0; ix < static_cast<int>(nx.x) + 2; ix++ ) {
            E[ix + iy*ystride].x += ( + dt_dx.y * ( B[ix + iy*ystride].z - B[ix + (iy-1)*ystride].z) );
            
            E[ix + iy*ystride].y += ( - dt_dx.x * ( B[ix + iy*ystride].z - B[(ix-1) + iy*ystride].z) );

            E[ix + iy*ystride].z += ( + dt_dx.x * ( B[ix + iy*ystride].y - B[(ix-1) + iy*ystride].y) - 
                                        dt_dx.y * ( B[ix + iy*ystride].x - B[ix + (iy-1)*ystride].x) );

        }
    }
}

/**
 * @brief Move simulation window
 * 
 * When using a moving simulation window checks if a window move is due
 * at the current iteration and if so shifts left the data, zeroing the
 * rightmost cells.
 * 
 */
void EMF::move_window() {

    if ( moving_window.needs_move( iter * dt ) ) {

        E->x_shift_left(1);
        B->x_shift_left(1);

        moving_window.advance();
    }
}

/**
 * @brief Advance EM fields 1 time step (no current)
 * 
 */
void EMF::advance() {

    const auto ntiles   = E -> get_ntiles();
    const auto field_vol = E -> tile_vol;
    const auto nx       = E -> nx;
    const auto offset   = E -> offset;
    const int  ystride  = E -> ext_nx.x;

    float2 const dt_dx   = make_float2( dt/dx.x, dt/dx.y );
    float2 const dt_dx_2 = make_float2( dt_dx.x/2, dt_dx.y/2 );

    // Loop over tiles
    #pragma omp parallel for
    for( int tid = 0; tid < ntiles.y * ntiles.x; tid++ ) {

        const auto tile_off = tid * field_vol;

        // Copy E and B into shared memory
        float3 E_local[ field_vol ];
        float3 B_local[ field_vol ];
        for( unsigned i = 0; i < field_vol; i++ ) {
            E_local[i] = E->d_buffer[ tile_off + i ];
            B_local[i] = B->d_buffer[ tile_off + i ];
        }

        float3 * const __restrict__ tile_E = & E_local[ offset ];
        float3 * const __restrict__ tile_B = & B_local[ offset ];

        // synchronize block (...)
        yee_b( tile_E, tile_B, nx, ystride, dt_dx_2 );
        // synchronize block (...)

        yee_e( tile_E, tile_B, nx, ystride, dt_dx );
        // synchronize block (...)

        yee_b( tile_E, tile_B, nx, ystride, dt_dx_2 );
        
        // synchronize block (...)

        // Copy data to global memory
        for( unsigned i = 0; i < field_vol; i++ ) {
            E->d_buffer[ tile_off + i ] = E_local[i];
            B->d_buffer[ tile_off + i ] = B_local[i];
        }
    }

    // Update guard cells with new values
    E -> copy_to_gc();
    B -> copy_to_gc();

    // Do additional bc calculations if needed
    // process_bc();

    // Advance internal iteration number
    iter += 1;

    // Move simulation window if needed
    if ( moving_window.active() ) move_window();

}

/**
 * @brief Physical boundary conditions for the x direction
 * 
 * @param tile_idx      Tile position on grid
 * @param E         Tile E field
 * @param B         Tile B field
 * @param nx        Number of cells
 * @param ext_nx    Number of cells including guard cells
 * @param bc        Boundary condition
 */
void emf_bcx( 
    const uint2 tile_idx,
    float3 * const __restrict__ E, float3 * const __restrict__ B,
    const uint2 nx, const uint2 ext_nx, emf::bc_type bc ) {

    const int ystride = ext_nx.x;

    if ( tile_idx.x == 0 ) {
        // Lower boundary
        switch( bc.x.lower ) {
        case( emf::bc::pmc) :
            for( int idx = 0; idx < static_cast<int>(ext_nx.y); idx ++ ) {
                // iy includes the y-stride
                const int iy = idx * ystride;

                E[ -1 + iy ].x = -E[ 0 + iy ].x;
                E[ -1 + iy ].y =  E[ 1 + iy ].y;
                E[ -1 + iy ].z =  E[ 1 + iy ].z;

                B[ -1 + iy ].x =  B[ 1 + iy ].x;
                B[ -1 + iy ].y = -B[ 0 + iy ].y;
                B[ -1 + iy ].z = -B[ 0 + iy ].z;

            }
            break;

        case( emf::bc::pec ) :
            for( int idx = 0; idx < static_cast<int>(ext_nx.y); idx ++ ) {
                const int iy = idx * ystride;

                E[ -1 + iy ].x =  E[ 0 + iy ].x;
                E[ -1 + iy ].y = -E[ 1 + iy ].y;
                E[ -1 + iy ].z = -E[ 1 + iy ].z;

                E[  0 + iy ].y = 0;
                E[  0 + iy ].z = 0;

                B[ -1 + iy ].x = -B[ 1 + iy ].x;
                B[ -1 + iy ].y =  B[ 0 + iy ].y;
                B[ -1 + iy ].z =  B[ 0 + iy ].z;

                B[  0 + iy ].x = 0;
            }
            break;
        default:
            break;
        } 
    } else {
        // Upper boundary
        switch( bc.x.upper ) {
        case( emf::bc::pmc) :
            for( int idx = 0; idx < static_cast<int>(ext_nx.y); idx ++ ) {
                const int iy = idx * ystride;

                E[ nx.x + iy ].x = -E[ nx.x-1 + iy ].x;
                //E[ nx.x + iy ].y =  E[ nx.x + iy ].y;
                //E[ nx.x + iy ].z =  E[ nx.x + iy ].z;

                E[ nx.x+1 + iy ].x = -E[ nx.x-2 + iy ].x;
                E[ nx.x+1 + iy ].y =  E[ nx.x-1 + iy ].y;
                E[ nx.x+1 + iy ].z =  E[ nx.x-1 + iy ].z;

                // B[ nx.x + iy ].x = -B[ nx.x + iy ].x;
                B[ nx.x + iy ].y = -B[ nx.x-1 + iy ].y;
                B[ nx.x + iy ].z = -B[ nx.x-1 + iy ].z;

                B[ nx.x+1 + iy ].x =  B[ nx.x-1 + iy ].x;
                B[ nx.x+1 + iy ].y = -B[ nx.x-2 + iy ].y;
                B[ nx.x+1 + iy ].z = -B[ nx.x-2 + iy ].z;
            }
            break;

        case( emf::bc::pec) :
            for( int idx = 0; idx < static_cast<int>(ext_nx.y); idx ++ ) {
                const int iy = idx * ystride;

                E[ nx.x + iy ].x =  E[ nx.x-1 + iy ].x;
                E[ nx.x + iy ].y =  0;
                E[ nx.x + iy ].z =  0;

                E[ nx.x+1 + iy ].x =  E[ nx.x-2 + iy ].x;
                E[ nx.x+1 + iy ].y = -E[ nx.x-1 + iy ].y;
                E[ nx.x+1 + iy ].z = -E[ nx.x-1 + iy ].z;

                B[ nx.x + iy ].x =  0;
                B[ nx.x + iy ].y =  B[ nx.x-1 + iy ].y;
                B[ nx.x + iy ].z =  B[ nx.x-1 + iy ].z;

                B[ nx.x+1 + iy ].x = -B[ nx.x-1 + iy ].x;
                B[ nx.x+1 + iy ].y =  B[ nx.x-2 + iy ].y;
                B[ nx.x+1 + iy ].z =  B[ nx.x-2 + iy ].z;
            }
            break;
        default:
            break;
        }
    }

}

/**
 * @brief Physical boundary conditions for the y direction
 * 
 * @param tile      Tile position on grid
 * @param E         Tile E field
 * @param B         Tile B field
 * @param nx        Number of cells
 * @param ext_nx    Number of cells including guard cells
 * @param bc        Boundary condition
 */
void emf_bcy( 
    const uint2 tile_idx,
    float3 * const __restrict__ E, float3 * const __restrict__ B,
    const uint2 nx, const uint2 ext_nx, emf::bc_type bc ) {

    const int ystride = ext_nx.x;

    if ( tile_idx.y == 0 ) {
        // Lower boundary
        switch( bc.y.lower ) {
        case( emf::bc::pmc) :
            for( int idx = 0; idx < static_cast<int>(ext_nx.x); idx ++ ) {
                const int ix = idx;

                E[ ix - ystride ].x =  E[ ix + ystride ].x;
                E[ ix - ystride ].y = -E[ ix +       0 ].y;
                E[ ix - ystride ].z =  E[ ix + ystride ].z;

                B[ ix - ystride ].x = -B[ ix +       0 ].x;
                B[ ix - ystride ].y =  B[ ix + ystride ].y;
                B[ ix - ystride ].z = -B[ ix +       0 ].z;
            }
            break;

        case( emf::bc::pec ) :
            for( int idx = 0; idx < static_cast<int>(ext_nx.x); idx ++ ) {
                int ix = idx;

                E[ ix - ystride ].x = -E[ ix + ystride ].x;
                E[ ix - ystride ].y =  E[ ix +       0 ].y;
                E[ ix - ystride ].z = -E[ ix + ystride ].z;

                E[ ix +       0 ].x = 0;
                E[ ix +       0 ].z = 0;
                
                B[ ix - ystride ].x =  B[ ix +       0 ].x;
                B[ ix - ystride ].y = -B[ ix + ystride ].y;
                B[ ix - ystride ].z =  B[ ix +       0 ].z;

                B[ ix +       0 ].y = 0;
            }
            break;
        default:
            break;
        }
    } else {
        // Upper boundary
        switch( bc.y.upper ) {
        case( emf::bc::pmc) :
            for( int idx = 0; idx < static_cast<int>(ext_nx.x); idx ++ ) {
                int ix = idx;

                E[ ix + nx.y * ystride ].y = -E[ ix + (nx.y-1) * ystride ].y;

                E[ ix + (nx.y+1) * ystride ].x =  E[ ix + (nx.y-1) * ystride ].x;
                E[ ix + (nx.y+1) * ystride ].y = -E[ ix + (nx.y-2) * ystride ].y;
                E[ ix + (nx.y+1) * ystride ].z =  E[ ix + (nx.y-1) * ystride ].z;

                B[ ix + (nx.y) * ystride ].x = -B[ ix + (nx.y-1)*ystride ].x;
                B[ ix + (nx.y) * ystride ].z = -B[ ix + (nx.y-1)*ystride ].z;

                B[ ix + (nx.y+1) * ystride ].x = -B[ ix + (nx.x-2) * ystride ].x;
                B[ ix + (nx.y+1) * ystride ].y =  B[ ix + (nx.x-1) * ystride ].y;
                B[ ix + (nx.y+1) * ystride ].z = -B[ ix + (nx.x-2) * ystride ].z;
            }
            break;

        case( emf::bc::pec) :
            for( int idx = 0; idx < static_cast<int>(ext_nx.x); idx ++ ) {
                const int ix = idx;

                E[ ix + (nx.y)*ystride ].x =  0;
                E[ ix + (nx.y)*ystride ].y =  E[ ix + (nx.y-1)*ystride ].y;
                E[ ix + (nx.y)*ystride ].z =  0;

                E[ ix + (nx.y+1)*ystride ].x = -E[ ix + (nx.x-1) * ystride ].x;
                E[ ix + (nx.y+1)*ystride ].y =  E[ ix + (nx.x-2) * ystride ].y;
                E[ ix + (nx.y+1)*ystride ].z = -E[ ix + (nx.x-1) * ystride ].z;

                B[ ix + (nx.y)*ystride ].x =  B[ ix + (nx.y-1) * ystride ].x;
                B[ ix + (nx.y)*ystride ].y =  0;
                B[ ix + (nx.y)*ystride ].z =  B[ ix + (nx.y-1) * ystride ].z;


                B[ ix + (nx.y+1) * ystride ].x =  B[ ix + (nx.y-2) * ystride ].x;
                B[ ix + (nx.y+1) * ystride ].y = -B[ ix + (nx.y-1) * ystride ].y;
                B[ ix + (nx.y+1) * ystride ].z =  B[ ix + (nx.y-2) * ystride ].z;
            }
            break;
        default:
            break;
        }
    }
}


/**
 * @brief Processes "physical" boundary conditions
 * 
 */
void EMF::process_bc() {

    std::cout << "(*error*) EMF::process_bc() have not been implemented yet,"
              << " aborting.\n";
    exit(1);

    const auto ntiles   = E -> get_ntiles();
    const auto tile_vol = E -> tile_vol;
    const auto nx       = E -> nx;
    const auto ext_nx   = E -> ext_nx;

    // x boundaries
    if ( bc.x.lower > emf::bc::periodic || bc.x.upper > emf::bc::periodic ) {

        // Loop over tiles
        for( unsigned ty = 0; ty < ntiles.y; ++ty ) {
            
            //  Only lower (0) and upper ( ntiles.x - 1 ) tiles have physical x boundaries
            for( unsigned tx : { 0u, ntiles.x-1 } ) {

                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                // Start at x cell 0
                const auto x_offset = E -> gc.x.lower;

                float3 * const __restrict__ tile_E = & E->d_buffer[ tile_off + x_offset ];
                float3 * const __restrict__ tile_B = & B->d_buffer[ tile_off + x_offset ];

                emf_bcx( tile_idx, tile_E, tile_B, nx, ext_nx, bc );
            }
        }
    }

    // y boundaries
    if ( bc.y.lower > emf::bc::periodic || bc.y.upper > emf::bc::periodic ) {

        // Loop over tiles

        //  Only lower (0) and upper ( ntiles.y - 1 ) tiles have physical y boundaries
        for( unsigned ty : { 0u, ntiles.y-1 } ) {
            for( unsigned tx = 0; tx < ntiles.x; ++tx ) {

                const auto tile_idx = make_uint2( tx, ty );
                const auto tid = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                // Start at y cell 0
                const auto y_offset = E -> gc.y.lower * ext_nx.x;

                float3 * const __restrict__ tile_E = & E->d_buffer[ tile_off + y_offset ];
                float3 * const __restrict__ tile_B = & B->d_buffer[ tile_off + y_offset ];

                emf_bcy( tile_idx, tile_E, tile_B, nx, ext_nx, bc );
            }
        }
    }
}

/**
 * @brief E advance for Yee algorithm including
 * 
 * @param E         Pointer to 0,0 coordinate of E field
 * @param B         Pointer to 0,0 coordinate of B field
 * @param nx        Internal tile size
 * @param ystride   Stride for y coordinate
 * @param J         Pointer to 0,0 coordinate of J field
 * @param J_ystride Stride for y coordinate of J field
 * @param dt_dx     \Delta t / \Delta x
 * @param dt_dy     \Delta t / \Delta y
 * @param dt        \Delta t
 */
void yeeJ_e( 
    float3 * const __restrict__ E, 
    float3 const * const __restrict__ B, 
    uint2 const nx, int const ystride, 
    float3 const * const __restrict__ J, int const J_ystride, 
    float2 const dt_dx, float const dt )
{
    for( int iy = 0; iy < static_cast<int>(nx.y) + 2; iy ++ ) {
        for( int ix = 0; ix < static_cast<int>(nx.x) + 2; ix++ ) {
            E[ix + iy*ystride].x += ( + dt_dx.y * ( B[ix + iy*ystride].z - B[ix + (iy-1)*ystride].z) )
                                    - dt * J[ix + iy*J_ystride].x;

            E[ix + iy*ystride].y += ( - dt_dx.x * ( B[ix + iy*ystride].z - B[(ix-1) + iy*ystride].z) )
                                    - dt * J[ix + iy*J_ystride].y;

            E[ix + iy*ystride].z += ( + dt_dx.x * ( B[ix + iy*ystride].y - B[(ix-1) + iy*ystride].y) - 
                                        dt_dx.y * ( B[ix + iy*ystride].x - B[ix + (iy-1)*ystride].x) )
                                    - dt * J[ix + iy*J_ystride ].z;
        }
    }
}

/**
 * @brief Advance EM fields 1 time step including current
 * 
 */
void EMF::advance( Current & current ) {

    const auto ntiles    = E -> get_ntiles();
    const auto field_vol = E -> tile_vol;
    const auto nx        = E -> nx;
    const auto offset    = E -> offset;
    const int  ystride   = E -> ext_nx.x;

    auto * J              = current.J->d_buffer;
    const auto J_tile_vol = current.J->tile_vol;
    const auto J_offset   = current.J->offset;
    const int  J_ystride  = current.J->ext_nx.x;

    float2 const dt_dx   = make_float2( dt/dx.x, dt/dx.y );
    float2 const dt_dx_2 = make_float2( dt_dx.x/2, dt_dx.y/2 );

    #pragma omp parallel for
    for( unsigned tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

        const auto tile_off = tid * field_vol;

        // Copy E and B into shared memory
        float3 E_local[ field_vol ];
        float3 B_local[ field_vol ];
        for( unsigned i = 0; i < field_vol; i++ ) {
            E_local[i] = E->d_buffer[ tile_off + i ];
            B_local[i] = B->d_buffer[ tile_off + i ];
        }

        float3 * const __restrict__ tile_E = & E_local[ offset ];
        float3 * const __restrict__ tile_B = & B_local[ offset ];
        
        // Use J from global memory
        const auto J_tile_off = tid * J_tile_vol + J_offset;
        float3 * const __restrict__ tile_J = & J[ J_tile_off ];

        // synchronize block (...)

        yee_b( tile_E, tile_B, nx, ystride, dt_dx_2 );
        // synchronize block (...)

        yeeJ_e( tile_E, tile_B, nx, ystride, tile_J, J_ystride, dt_dx, dt );
        // synchronize block (...)

        yee_b( tile_E, tile_B, nx, ystride, dt_dx_2 );
        
        // synchronize block (...)

        // Copy data to global memory
        for( unsigned i = 0; i < field_vol; i++ ) {
            E->d_buffer[ tile_off + i ] = E_local[i];
            B->d_buffer[ tile_off + i ] = B_local[i];
        }
    }

    // Update guard cells with new values
    E -> copy_to_gc();
    B -> copy_to_gc();

    // Do additional bc calculations if needed
    // process_bc();

    // Advance internal iteration number
    iter += 1;

    // Move simulation window if needed
    if ( moving_window.active() ) move_window();
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

    vec3grid<float3> * f;

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
        default:
            std::cerr << "(*error*) Invalid field type selected, returning..." << std::endl;
            return;
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
            std::cerr << "(*error*) Invalid field component (fc) selected, returning..." << std::endl;
            return;
    }

    zdf::grid_axis axis[2];
    axis[0] = (zdf::grid_axis) {
        .name = (char *) "x",
        .min = 0.0 + moving_window.motion(),
        .max = box.x + moving_window.motion(),
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

    f -> save( fc, info, iteration, "EMF" );
}

/**
 * @brief Get total field energy per field component
 * 
 * @param energy    Array that will hold energy values
 */
void EMF::get_energy( double3 & ene_E, double3 & ene_B ) {

    ene_E = make_double3(0,0,0);
    ene_B = make_double3(0,0,0);

    const uint2 ntiles          = E -> get_ntiles();
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
                for( unsigned ix = 0; ix < nx.x; ++ix ) {
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
}