#include "emf.h"

#include <iostream>
#include "zdf-cpp.h"

/**
 * This is required for using bnd<emf::bc::type> inside a Sycl kernel
 */
template<>
struct sycl::is_device_copyable<bnd<emf::bc::type>> : std::true_type {};

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
    double const dt, sycl::queue & q ) : 
    dx( make_float2( box.x / ( nx.x * ntiles.x ), box.y / ( nx.y * ntiles.y ) ) ),
    dt( dt ), q(q), box(box)
{
    // Verify Courant condition
    float cour = sqrtf( 1.0f/( 1.0f/(dx.x*dx.x) + 1.0f/(dx.y*dx.y) ) );
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

    E = new vec3grid<float3> ( ntiles, nx, gc, q );
    E -> name = "Electric field";

    B = new vec3grid<float3> ( ntiles, nx, gc, q );
    B -> name = "Magnetic field";

    // Check that local memory can hold up to 2 times the tile buffer
    auto local_mem_size = q.get_device().get_info<sycl::info::device::local_mem_size>();
    if ( local_mem_size < 2 * E->tile_vol * sizeof( float3 ) ) {
        std::cerr << "(*error*) Tile size too large " << nx << " (plus guard cells)\n";
        std::cerr << "(*error*) Insufficient local memory (" << local_mem_size << " B) for EMF object.\n";
        abort();
    }

    // Zero fields
    E -> zero( );
    B -> zero( );

    // Reserve device memory for energy diagnostic
    d_energy = device::malloc<double>( 6, q );

    // Set default boundary conditions to periodic
    bc = emf::bc_type (emf::bc::periodic);

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
inline void yee_b( 
    sycl::nd_item<2> & it, 
    float3 const * const __restrict__ E, 
    float3 * const __restrict__ B, 
    uint2 const nx, int const ystride, 
    float2 const dt_dx )
{
    int const range_x = nx.x + 2;
    int const range_y = nx.y + 2;
    
    for( int idx = it.get_local_id(0); idx < range_y * range_x; idx += it.get_local_range(0) ) {
        const int ix = -1 + idx % range_x;
        const int iy = -1 + idx / range_x;
   
        B[ ix + iy*ystride ].x += ( - dt_dx.y * ( E[ix + (iy+1)*ystride].z - E[ix + iy*ystride].z ) );  
        B[ ix + iy*ystride ].y += (   dt_dx.x * ( E[(ix+1) + iy*ystride].z - E[ix + iy*ystride].z ) );  
        B[ ix + iy*ystride ].z += ( - dt_dx.x * ( E[(ix+1) + iy*ystride].y - E[ix + iy*ystride].y ) + 
                                        dt_dx.y * ( E[ix + (iy+1)*ystride].x - E[ix + iy*ystride].x ) );  
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
inline void yee_e( 
    sycl::nd_item<2> & it, 
    float3 * const __restrict__ E, 
    float3 const * const __restrict__ B, 
    uint2 const nx, int const ystride, 
    float2 const dt_dx )
{
    int const range_x = nx.x + 2;
    int const range_y = nx.y + 2;

    for( int idx = it.get_local_id(0); idx < range_y * range_x; idx += it.get_local_range(0) ) {
        const int ix = idx % range_x;
        const int iy = idx / range_x;

        E[ix + iy*ystride].x += ( + dt_dx.y * ( B[ix + iy*ystride].z - B[ix + (iy-1)*ystride].z) );
        
        E[ix + iy*ystride].y += ( - dt_dx.x * ( B[ix + iy*ystride].z - B[(ix-1) + iy*ystride].z) );

        E[ix + iy*ystride].z += ( + dt_dx.x * ( B[ix + iy*ystride].y - B[(ix-1) + iy*ystride].y) - 
                                    dt_dx.y * ( B[ix + iy*ystride].x - B[ix + (iy-1)*ystride].x) );
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

    const auto dt = this -> dt;

    const auto ntiles   = E -> ntiles;
    const auto field_vol = E -> tile_vol;
    const auto nx       = E -> nx;
    const auto offset   = E -> offset;
    const int  ystride  = E -> ext_nx.x;

    auto * E_buffer = E -> d_buffer;
    auto * B_buffer = B -> d_buffer;

    float2 const dt_dx   = make_float2( dt/dx.x, dt/dx.y );
    float2 const dt_dx_2 = make_float2( dt_dx.x/2, dt_dx.y/2 );

    q.submit([&](sycl::handler &h) {

        /// @brief [shared] Local copy of E-field
        auto E_local = sycl::local_accessor< float3, 1 > ( field_vol, h );
        /// @brief [shared] Local copy of B-field
        auto B_local = sycl::local_accessor< float3, 1 > ( field_vol, h );

        // 8×1 work items per group
        sycl::range<2> local{ 8, 1 };

        // ntiles.x × ntiles.y groups
        sycl::range<2> global{ ntiles.x, ntiles.y };

        h.parallel_for( 
            sycl::nd_range{ global * local, local },
            [=](sycl::nd_item<2> it) { 

            const auto tile_idx = make_uint2( it.get_group(0), it.get_group(1) );
            const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
            const auto tile_off = tid * field_vol;

            // Copy E and B into shared memory
            for( unsigned i = it.get_local_id(0); i < field_vol; i += it.get_local_range(0) ) {
                E_local[i] = E_buffer[ tile_off + i ];
                B_local[i] = B_buffer[ tile_off + i ];
            }

            float3 * const __restrict__ tile_E = & E_local[ offset ];
            float3 * const __restrict__ tile_B = & B_local[ offset ];

            it.barrier();

            yee_b( it, tile_E, tile_B, nx, ystride, dt_dx_2 );
            it.barrier();

            yee_e( it, tile_E, tile_B, nx, ystride, dt_dx );
            it.barrier();

            yee_b( it, tile_E, tile_B, nx, ystride, dt_dx_2 );
            it.barrier();

            // Copy data to global memory
            for( unsigned i = it.get_local_id(0); i < field_vol; i += it.get_local_range(0) ) {
                E_buffer[ tile_off + i ] = E_local[i];
                B_buffer[ tile_off + i ] = B_local[i];
            }
        });
    });
    q.wait();

    // Update guard cells with new values
    E -> copy_to_gc();
    B -> copy_to_gc();

    // Do additional bc calculations if needed
    process_bc();

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
inline void emf_bcx( 
    sycl::nd_item<2> & it,
    float3 * const __restrict__ E, float3 * const __restrict__ B,
    const uint2 nx, const uint2 ext_nx, emf::bc_type bc ) {

    const auto tile_idx_x = it.get_group(0);
    const int ystride = ext_nx.x;

    if ( tile_idx_x == 0 ) {
        // Lower boundary
        switch( bc.x.lower ) {
        case( emf::bc::pmc) :
            for( int idx = it.get_local_id(0); idx < static_cast<int>(ext_nx.y); idx += it.get_local_range(0) ) {
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
            for( int idx = it.get_local_id(0); idx < static_cast<int>(ext_nx.y); idx += it.get_local_range(0) ) {
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
            for( int idx = it.get_local_id(0); idx < static_cast<int>(ext_nx.y); idx += it.get_local_range(0) ) {
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
            for( int idx = it.get_local_id(0); idx < static_cast<int>(ext_nx.y); idx += it.get_local_range(0) ) {
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
    sycl::nd_item<2> & it,
    float3 * const __restrict__ E, float3 * const __restrict__ B,
    const uint2 nx, const uint2 ext_nx, emf::bc_type bc ) {

    const auto tile_idx_y = it.get_group(1);
    const int ystride = ext_nx.x;

    if ( tile_idx_y == 0 ) {
        // Lower boundary
        switch( bc.y.lower ) {
        case( emf::bc::pmc) :
            for( int idx = it.get_local_id(0); idx < static_cast<int>(ext_nx.x); idx += it.get_local_range(0) ) {
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
            for( int idx = it.get_local_id(0); idx < static_cast<int>(ext_nx.x); idx += it.get_local_range(0) ) {
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
            for( int idx = it.get_local_id(0); idx < static_cast<int>(ext_nx.x); idx += it.get_local_range(0) ) {
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
            for( int idx = it.get_local_id(0); idx < static_cast<int>(ext_nx.x); idx += it.get_local_range(0) ) {
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

    const auto ntiles    = E -> ntiles;
    const auto field_vol = E -> tile_vol;
    const auto nx        = E -> nx;
    const auto ext_nx    = E -> ext_nx;
    const auto gc        = E -> gc;
    const auto bc        = this -> bc;
    
    auto * E_buffer = E -> d_buffer;
    auto * B_buffer = B -> d_buffer;

    // x boundaries
    if ( bc.x.lower > emf::bc::periodic || bc.x.upper > emf::bc::periodic ) {

        q.submit([&](sycl::handler &h) {

            // 8×1 work items per group
            sycl::range<2> local{ 8, 1 };

            // 2 × ntiles.y groups
            sycl::range<2> global{ 2, ntiles.y };

            h.parallel_for( 
                sycl::nd_range{ global * local, local },
                [=](sycl::nd_item<2> it) { 

                const auto tile_idx = make_uint2( 
                    it.get_group(0) * (ntiles.x-1),
                    it.get_group(1)
                );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * field_vol;

                // Start at x cell 0
                const auto x_offset = gc.x.lower;

                float3 * const __restrict__ tile_E = & E_buffer[ tile_off + x_offset ];
                float3 * const __restrict__ tile_B = & B_buffer[ tile_off + x_offset ];

                emf_bcx( it, tile_E, tile_B, nx, ext_nx, bc );
            });
        });
        q.wait();
    }

    // y boundaries
    if ( bc.y.lower > emf::bc::periodic || bc.y.upper > emf::bc::periodic ) {

        q.submit([&](sycl::handler &h) {

            // 8×1 work items per group
            sycl::range<2> local{ 8, 1 };

            // ntiles.x × 2 groups
            sycl::range<2> global{ ntiles.x, 2 };

            h.parallel_for( 
                sycl::nd_range{ global * local, local },
                [=](sycl::nd_item<2> it) { 

                const auto tile_idx = make_uint2( 
                    it.get_group(0) ,
                    it.get_group(1) * (ntiles.y-1)
                );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * field_vol;

                // Start at y cell 0
                const auto y_offset = gc.y.lower * ext_nx.x;

                float3 * const __restrict__ tile_E = & E_buffer[ tile_off + y_offset ];
                float3 * const __restrict__ tile_B = & B_buffer[ tile_off + y_offset ];

                emf_bcy( it, tile_E, tile_B, nx, ext_nx, bc );
            });
        });
        q.wait();
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
inline void yeeJ_e( 
    sycl::nd_item<2> & it, 
    float3 * const __restrict__ E, 
    float3 const * const __restrict__ B, 
    uint2 const nx, int const ystride, 
    float3 const * const __restrict__ J, int const J_ystride, 
    float2 const dt_dx, float const dt )
{
    int const range_x = nx.x + 2;
    int const range_y = nx.y + 2;

    for( int idx = it.get_local_id(0); idx < range_y * range_x; idx += it.get_local_range(0) ) {
        const int ix = idx % range_x;
        const int iy = idx / range_x;

        E[ix + iy*ystride].x += ( + dt_dx.y * ( B[ix + iy*ystride].z - B[ix + (iy-1)*ystride].z) )
                                - dt * J[ix + iy*J_ystride].x;

        E[ix + iy*ystride].y += ( - dt_dx.x * ( B[ix + iy*ystride].z - B[(ix-1) + iy*ystride].z) )
                                - dt * J[ix + iy*J_ystride].y;

        E[ix + iy*ystride].z += ( + dt_dx.x * ( B[ix + iy*ystride].y - B[(ix-1) + iy*ystride].y) - 
                                    dt_dx.y * ( B[ix + iy*ystride].x - B[ix + (iy-1)*ystride].x) )
                                - dt * J[ix + iy*J_ystride ].z;
    }
}

/**
 * @brief Advance EM fields 1 time step including current
 * 
 * @param current   Electric current
 */
void EMF::advance( Current & current ) {

    const auto dt = this -> dt;

    const auto ntiles    = E -> ntiles;
    const auto field_vol = E -> tile_vol;
    const auto nx        = E -> nx;
    const auto offset    = E -> offset;
    const int  ystride   = E -> ext_nx.x;

    auto * E_buffer = E -> d_buffer;
    auto * B_buffer = B -> d_buffer;

    auto * J              = current.J->d_buffer;
    const auto J_tile_vol = current.J->tile_vol;
    const auto J_offset   = current.J->offset;
    const int  J_ystride  = current.J->ext_nx.x;

    float2 const dt_dx   = make_float2( dt/dx.x, dt/dx.y );
    float2 const dt_dx_2 = make_float2( dt_dx.x/2, dt_dx.y/2 );

    q.submit([&](sycl::handler &h) {

        /// @brief [shared] Local copy of E-field
        auto E_local = sycl::local_accessor< float3, 1 > ( field_vol, h );
        /// @brief [shared] Local copy of B-field
        auto B_local = sycl::local_accessor< float3, 1 > ( field_vol, h );

        // 128×1 work items per group
        sycl::range<2> local{ 128, 1 };

        // ntiles.x × ntiles.y groups
        sycl::range<2> global{ ntiles.x, ntiles.y };

        h.parallel_for( 
            sycl::nd_range{ global * local, local },
            [=](sycl::nd_item<2> it) { 

            const auto tile_idx = make_uint2( it.get_group(0), it.get_group(1) );
            const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
            const auto tile_off = tid * field_vol;

            // Copy E and B into shared memory
            for( unsigned i = it.get_local_id(0); i < field_vol; i += it.get_local_range(0) ) {
                E_local[i] = E_buffer[ tile_off + i ];
                B_local[i] = B_buffer[ tile_off + i ];
            }

            float3 * const __restrict__ tile_E = & E_local[ offset ];
            float3 * const __restrict__ tile_B = & B_local[ offset ];
            
            // Use J from global memory
            const auto J_tile_off = tid * J_tile_vol + J_offset;
            float3 * const __restrict__ tile_J = & J[ J_tile_off ];

            it.barrier();

            yee_b( it, tile_E, tile_B, nx, ystride, dt_dx_2 );
            it.barrier();

            yeeJ_e( it, tile_E, tile_B, nx, ystride, tile_J, J_ystride, dt_dx, dt );
            it.barrier();

            yee_b( it, tile_E, tile_B, nx, ystride, dt_dx_2 );
            it.barrier();

            // Copy data to global memory
            for( unsigned i = it.get_local_id(0); i < field_vol; i += it.get_local_range(0) ) {
                E_buffer[ tile_off + i ] = E_local[i];
                B_buffer[ tile_off + i ] = B_local[i];
            }
        });
    });
    q.wait();

    // Update guard cells with new values
    E -> copy_to_gc( );
    B -> copy_to_gc( );

    // Do additional bc calculations if needed
    process_bc( );

    // Advance internal iteration number
    iter += 1;

    // Move simulation window if needed
    if ( moving_window.active() ) move_window( );
}


/**
 * @brief Save EMF data to diagnostic file
 * 
 * @param field     Field to save (0:E, 1:B)
 * @param fc        Field component to save (0, 1 or 2)
 */
void EMF::save( const emf::field field, fcomp::cart const fc ) {

    char vfname[16];	// Dataset name
    char vflabel[16];	// Dataset label (for plots)

    char comp[] = {'x','y','z'};

    if ( fc < 0 || fc > 2 ) {
        std::cerr << "(*error*) Invalid field component (fc) selected, returning" << std::endl;
        return;
    }

    // Choose field to save
    vec3grid<float3> * f;
    switch (field) {
        case emf::e :
            f = E;
            snprintf(vfname,16,"E%c",comp[fc]);
            snprintf(vflabel,16,"E_%c",comp[fc]);
            break;
        case emf::b :
            f = B;
            snprintf(vfname,16,"B%1c",comp[fc]);
            snprintf(vflabel,16,"B_%c",comp[fc]);
            break;
        default:
        std::cerr << "(*error*) Invalid field type selected, returning..." << std::endl;
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
        .name = vfname,
        .ndims = 2,
        .label = vflabel,
        .units = (char *) "m_e c \\omega_n e^{-1}",
        .axis = axis
    };

    info.count[0] = E -> gnx.x;
    info.count[1] = E -> gnx.y;


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
 * @warning This function will always recalculate the energy each time it is
 *          called.
 * 
 * @param ene_E     Total E-field energy (per component)
 * @param ene_B     Total B-field energy (per component)
 */
void EMF::get_energy( double3 & ene_E, double3 & ene_B ) {

    // Calculate energy on device memory

    const auto ntiles   = E -> ntiles;
    const auto tile_vol = E -> tile_vol;
    const auto nx       = E -> nx;
    const auto offset   = E -> offset;
    const auto ystride  = E -> ext_nx.x;
    const auto d_energy = this -> d_energy;

    float3 * const __restrict__ d_E = E -> d_buffer;
    float3 * const __restrict__ d_B = B -> d_buffer;

    // 8×1 work items per group
    sycl::range<2> local{ 8, 1 };

    // ntiles.x × ntiles.y groups
    sycl::range<2> global{ ntiles.x, ntiles.y };

    device::zero( d_energy, 6, q );

    q.submit([&](sycl::handler &h) {
        h.parallel_for( 
            sycl::nd_range{ global * local, local },
            [=](sycl::nd_item<2> it) {

            const int2 tile_idx = make_int2( it.get_group(0), it.get_group(1));
            const int tile_id = tile_idx.y * ntiles.x + tile_idx.x;
            const int tile_off = tile_id * tile_vol + offset;

            double3 ene_E = make_double3(0,0,0);
            double3 ene_B = make_double3(0,0,0);

            for( int idx = it.get_local_id(0); idx < nx.y * nx.x; idx += it.get_local_range(0) ) {
                int const i = idx % nx.x;
                int const j = idx / nx.x;

                float3 const efld = d_E[ tile_off + j * ystride + i ];
                float3 const bfld = d_B[ tile_off + j * ystride + i ];

                ene_E.x += efld.x * efld.x;
                ene_E.y += efld.y * efld.y;
                ene_E.z += efld.z * efld.z;

                ene_B.x += bfld.x * bfld.x;
                ene_B.y += bfld.y * bfld.y;
                ene_B.z += bfld.z * bfld.z;
            }

            // Add up energy from all warps
            auto sg = it.get_sub_group();
            ene_E.x = device::subgroup::reduce_add( sg, ene_E.x );
            ene_E.y = device::subgroup::reduce_add( sg, ene_E.y );
            ene_E.z = device::subgroup::reduce_add( sg, ene_E.z );

            ene_B.x = device::subgroup::reduce_add( sg, ene_B.x );
            ene_B.y = device::subgroup::reduce_add( sg, ene_B.y );
            ene_B.z = device::subgroup::reduce_add( sg, ene_B.z );

            if ( sg.get_local_linear_id() == 0 ) {
                device::global::atomicAdd( &(d_energy[0]), ene_E.x );
                device::global::atomicAdd( &(d_energy[1]), ene_E.y );
                device::global::atomicAdd( &(d_energy[2]), ene_E.z );

                device::global::atomicAdd( &(d_energy[3]), ene_B.x );
                device::global::atomicAdd( &(d_energy[4]), ene_B.y );
                device::global::atomicAdd( &(d_energy[5]), ene_B.z );
            }
        });
    });
    q.wait();

    // Copy results to host and normalize
    double h_energy[6];
    device::memcpy_tohost( h_energy, d_energy, 6, q );

    ene_E.x = 0.5 * dx.x * dx.y * h_energy[0];
    ene_E.y = 0.5 * dx.x * dx.y * h_energy[1];
    ene_E.z = 0.5 * dx.x * dx.y * h_energy[2];

    ene_B.x = 0.5 * dx.x * dx.y * h_energy[3];
    ene_B.y = 0.5 * dx.x * dx.y * h_energy[4];
    ene_B.z = 0.5 * dx.x * dx.y * h_energy[5];
}