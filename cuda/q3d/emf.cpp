#include "emf.h"

#include <iostream>
#include "zdf-cpp.h"

#define opt_yee_block 256

#if 0
namespace kernel {

__global__
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
    float3 * const __restrict__ E_buffer, float3 * const __restrict__ B_buffer,
    uint2 const ntiles, const uint2 nx, const uint2 ext_nx, bnd<unsigned int> gc, 
    emf::bc_type bc ) {

    const uint2  tile_idx = { blockIdx.x * ( ntiles.x - 1 ), blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    const int ystride = ext_nx.x;
    // Start at x cell 0
    const auto x_offset = gc.x.lower;

    float3 * const __restrict__ E = & E_buffer[ tile_off + x_offset ];
    float3 * const __restrict__ B = & B_buffer[ tile_off + x_offset ];

    if ( tile_idx.x == 0 ) {
        // Lower boundary
        switch( bc.x.lower ) {
        case( emf::bc::pmc) :
            for( int idx = block_thread_rank(); idx < static_cast<int>(ext_nx.y); idx += block_num_threads() ) {
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
            for( int idx = block_thread_rank(); idx < static_cast<int>(ext_nx.y); idx += block_num_threads() ) {
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
            for( int idx = block_thread_rank(); idx < static_cast<int>(ext_nx.y); idx += block_num_threads() ) {
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
            for( int idx = block_num_threads(); idx < static_cast<int>(ext_nx.y); idx += block_num_threads() ) {
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

__global__
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
    float3 * const __restrict__ E_buffer, float3 * const __restrict__ B_buffer,
    uint2 const ntiles, const uint2 nx, const uint2 ext_nx, bnd<unsigned int> gc, 
    emf::bc_type bc )
{

    const auto tile_idx = uint2{ 
        blockIdx.x ,
        blockIdx.y * (ntiles.y-1)
    };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    const int ystride = ext_nx.x;
    // Start at y cell 0
    const auto y_offset = gc.y.lower * ext_nx.x;

    float3 * const __restrict__ E = & E_buffer[ tile_off + y_offset ];
    float3 * const __restrict__ B = & B_buffer[ tile_off + y_offset ];

    if ( tile_idx.y == 0 ) {
        // Lower boundary
        switch( bc.y.lower ) {
        case( emf::bc::pmc) :
            for( int idx = block_thread_rank(); idx < static_cast<int>(ext_nx.x); idx += block_num_threads() ) {
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
            for( int idx = block_thread_rank(); idx < static_cast<int>(ext_nx.x); idx += block_num_threads() ) {
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
            for( int idx = block_thread_rank(); idx < static_cast<int>(ext_nx.x); idx += block_num_threads() ) {
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
            for( int idx = block_thread_rank(); idx < static_cast<int>(ext_nx.x); idx += block_num_threads() ) {
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

}

/**
 * @brief Processes "physical" boundary conditions
 * 
 */
void EMF::process_bc() {

    dim3 block( 64 );

    // x boundaries
    if ( bc.x.lower > emf::bc::periodic || bc.x.upper > emf::bc::periodic ) {
        dim3 grid( 2, E->ntiles.y );
        kernel::emf_bcx <<< grid, block >>> (
            E -> d_buffer, B -> d_buffer, E -> ntiles, E -> nx, E -> ext_nx, E -> gc, 
            bc
        );
    }

    // y boundaries
    if ( bc.y.lower > emf::bc::periodic || bc.y.upper > emf::bc::periodic ) {
        dim3 grid( E->ntiles.x, 2 );
        kernel::emf_bcx <<< grid, block >>> (
            E -> d_buffer, B -> d_buffer, E -> ntiles, E -> nx, E -> ext_nx, E -> gc, 
            bc
        );
    }
}

#endif


/**
 * @brief Construct a new EMF object
 * 
 * @param nmodes    Number of cylindrical modes (>= 1)
 * @param ntiles    Number of tiles in z,r direction
 * @param nx        Tile size (#cells)
 * @param box       Simulation box size (sim. units)
 * @param dt        Time step
 */
EMF::EMF( int nmodes, uint2 const ntiles, uint2 const nx, float2 const box,
     double const dt ) : 
    dx( make_float2( box.x / ( nx.x * ntiles.x ), box.y / ( nx.y * ntiles.y ) ) ),
    dt( dt ),
    nmodes( nmodes ),
    box(box)
{
    auto cour = zpic::courant( nmodes, dx );
    if ( dt >= cour ){
        std::cerr << "(*error*) EMF(): Invalid timestep, courant condition violation.\n"
                  << "(*error*) For the no. modes / current resolution " << nmodes << " / " << dx
                  << " the maximum timestep is dt = " << cour << '\n';
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

    auto & E0 = E -> mode0();

    // Check that local memory can hold up to all fields / modes for 1 tile
    auto req_mem_size = 2 * E0.tile_vol * 2 * 
        (( nmodes == 1 ) ? sizeof( cyl_float3 ) : sizeof( cyl_cfloat3 ));

    if ( block::shared_mem_size() < req_mem_size ) {
        std::cerr << "(*error*) Tile size too large " << nx << " (plus guard cells)\n";
        std::cerr << "(*error*) Insufficient local memory (" << block::shared_mem_size() << " B) for EMF object.\n";
        abort();
    }

    // Zero fields
    E -> zero( );
    B -> zero( );

    // Reserve device memory for energy diagnostic
    d_energy = device::malloc<double>( 6 );

    // Set default boundary conditions
    bc.x.lower = bc.x.upper = emf::bc::periodic;
    bc.y.lower = emf::bc::axial; 
    bc.y.upper = emf::bc::none;

    // Reset iteration number
    iter = 0;
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

namespace kernel {

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
__device__ void yee0_b( 
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

    int const range_z = nx.x + 2;
    int const range_r = nx.y + 1 - j0;
    
    for( int idx = block_thread_rank(); idx < range_r * range_z; idx += block_num_threads() ) {
        const int i = -1 + idx % range_z;
        const int j = j0 + idx / range_z;

        /// @brief r at lower edge of j cell, normalized to Δr
        float rm  = ir0 + j - 0.5f;
        /// @brief r at upper edge of j cell, normalized to Δr
        float rp  = ir0 + j + 0.5f;
        /// @brief Δt/r at the center of j cell
        float dt_rc   = dt / ( ( ir0 + j ) * dr );

        B[ i + j*jstride ].r += (   dt_dz * ( E[(i+1) + j*jstride].θ - E[i + j*jstride].θ ) );  

        B[ i + j*jstride ].θ += ( - dt_dz * ( E[(i+1) + j*jstride].r - E[i + j*jstride].r ) + 
                                    dt_dr * ( E[i + (j+1)*jstride].z - E[i + j*jstride].z ) );  

        B[ i + j*jstride ].z += ( - dt_rc * ( rp * E[i + (j+1)*jstride].θ - rm * E[i + j*jstride].θ ) );  
    }

    if ( ir0 == 0 ) {
        block_sync();
        for( int i = block_thread_rank() - 1; i < static_cast<int>(nx.x) + 1; i += block_num_threads() ) {
            B[ i +   0 * jstride ].r = - B[ i + 1*jstride ].r;  
            B[ i +   0 * jstride ].θ = 0;
            B[ i +   0 * jstride ].z += - 4 * dt_dr * E[ i + 1*jstride ].θ;  
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
__device__ void yee0_e( 
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

    int const range_z = nx.x + 2;      // [0, nx.x + 2[
    int const range_r = nx.y + 2 - j0; // [j0, nx.y + 2[

    for( int idx = block_thread_rank(); idx < range_z * range_r; idx += block_num_threads() ) {
        const int i = idx % range_z;
        const int j = idx / range_z + j0;

        /// @brief r at center of j-1 cell, normalized to Δr
        float rcm  = ir0 + j - 1;
        /// @brief r at center of j cell, normalized to Δr
        float rc   = ir0 + j;
        /// @brief Δt/r at the lower edge of j cell
        float dt_rm = dt / ( ( ir0 + j - 0.5 ) * dr );

        E[i + j*jstride].r += ( - dt_dz * ( B[i + j*jstride].θ - B[(i-1) + j*jstride].θ) );

        E[i + j*jstride].θ += ( + dt_dz * ( B[i + j*jstride].r - B[(i-1) + j*jstride].r) - 
                                    dt_dr * ( B[i + j*jstride].z - B[i + (j-1)*jstride].z) );

        E[i + j*jstride].z += ( + dt_rm * ( rc * B[i + j*jstride].θ - rcm * B[i + (i-1)*jstride].θ) );

    }

    if ( ir0 == 0 ) {
        block_sync();
        for( int i = block_thread_rank(); i < static_cast<int>(nx.x) + 2; i += block_num_threads() ) {
            E[i +   0 *jstride].r = 0;
            E[i +   0 *jstride].θ = -E[i + 1*jstride].θ;
            E[i +   0 *jstride].z = E[i + 1*jstride].z;
        }
    }
}

__global__ __launch_bounds__(opt_yee_block) void yee0( 
    cyl_float3 * const __restrict__ E, 
    cyl_float3 * const __restrict__ B, 
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx, unsigned int const offset, 
    const double dt, float2 const dx ) {

    auto * shm = block::shared_mem<cyl_float3>();

    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    const int jstride = ext_nx.x;
    const int ir0 = tile_idx.y * nx.y;

    auto * const __restrict__ E_local = & shm[ 0 ];
    auto * const __restrict__ B_local = & shm[ tile_vol ];

    // Copy E and B into shared memory
    for( unsigned i = block_thread_rank(); i < tile_vol; i += block_num_threads() ) {
        E_local[i] = E[ tile_off + i ];
        B_local[i] = B[ tile_off + i ];
    }

    auto * const __restrict__ tile_E = & E_local[ offset ];
    auto * const __restrict__ tile_B = & B_local[ offset ];

    block_sync();

    yee0_b( tile_E, tile_B, nx, jstride, dt/2, dx, ir0 );
    block_sync();

    yee0_e( tile_E, tile_B, nx, jstride, dt,   dx, ir0 );
    block_sync();

    yee0_b( tile_E, tile_B, nx, jstride, dt/2, dx, ir0 );
    block_sync();

    // Copy data to global memory
    for( unsigned i = block_thread_rank(); i < tile_vol; i += block_num_threads() ) {
        E[ tile_off + i ] = E_local[i];
        B[ tile_off + i ] = B_local[i];
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
__device__ void yeem_b( 
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
    const ops::complex<float> mI{0,static_cast<float>(m)};
    ///@brief imaginary unit
    const ops::complex<float> I{0,1};

    int const range_z = nx.x + 2;
    int const range_r = nx.y + 1 - j0;
    
    for( int idx = block_thread_rank(); idx < range_r * range_z; idx += block_num_threads() ) {

        const int i = -1 + idx % range_z;
        const int j = j0 + idx / range_z;

        /// @brief r/Δr at lower edge of j cell
        float rl  = ir0 + j - 0.5f;
        /// @brief r/Δr at upper edge of j cell
        float ru  = ir0 + j + 0.5f;
        /// @brief Δt/r at the center of j cell
        float dt_rc = dt / ( ( ir0 + j ) * dr );
        /// @brief Δt/r at the lower edge of j cell
        float dt_rl = dt / ( ( ir0 + j - 0.5 ) * dr );

        B[ i + j*jstride ].r += 
            - dt_rl * mI * E[ i + j * jstride ].z                       // (Δt/r) m I Ez
            + dt_dz * ( E[(i+1) + j*jstride].θ - E[i + j*jstride].θ );  // Δt ∂Eθ/∂z

        B[ i + j*jstride ].θ += 
            - dt_dz * ( E[(i+1) + j*jstride].r - E[i + j*jstride].r )   // Δt ∂Er/∂z
            + dt_dr * ( E[i + (j+1)*jstride].z - E[i + j*jstride].z );  // Δt ∂Ez/∂r

        B[ i + j*jstride ].z += - dt_rc * (                             // Δt/r
            + ( ru * E[i + (j+1)*jstride].θ - rl * E[i + j*jstride].θ ) // ∂(r Eθ)/∂r 
            - mI * E[ i + j * jstride ].r                               // m I Er
        );  
    }

    if ( ir0 == 0 ) {
        block_sync();
        if ( m == 1 ) {
            // Mode m = 1 is a special case
            for( int i = block_thread_rank() - 1; i < static_cast<int>(nx.x) + 1; i += block_num_threads() ) {
                B[ i + 0 *jstride ].r = B[ i + 1*jstride ].r;
                B[ i + 0 *jstride ].θ = 0.125f * I * (9.f * B[ i + 1*jstride ].r - B[ i + 2*jstride ].r );
                B[ i + 0 *jstride ].z = 0;
            }
        } else {
            for( int i = block_thread_rank() - 1; i < static_cast<int>(nx.x) + 1; i += block_num_threads() ) {
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
__device__ void yeem_e( 
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
    const ops::complex<float> mI{0,static_cast<float>(m)};

    int const range_z = nx.x + 2;      // [0, nx.x + 2[
    int const range_r = nx.y + 2 - j0; // [j0, nx.y + 2[

    for( int idx = block_thread_rank(); idx < range_z * range_r; idx += block_num_threads() ) {
        const int i = idx % range_z;
        const int j = idx / range_z + j0;

        /// @brief r/Δr at center of j-1 cell
        float rcm  = ir0 + j - 1;
        /// @brief rΔr at center of j cell
        float rc   = ir0 + j;
        /// @brief Δt/r at the lower edge of j cell
        float dt_rl = dt / ( ( ir0 + j - 0.5 ) * dr );
        /// @brief Δt/r at the center of j cell
        float dt_rc = dt / ( ( ir0 + j ) * dr );

        E[i + j*jstride].r += ( 
            - dt_dz * ( B[i + j*jstride].θ - B[(i-1) + j*jstride].θ)   // Δt ∂Bθ/∂z 
            + dt_rc * mI * B[ i + j * jstride ].z                      // (Δt/r) m I Bz
        );

        E[i + j*jstride].θ += ( + dt_dz * ( B[i + j*jstride].r - B[(i-1) + j*jstride].r)
                                - dt_dr * ( B[i + j*jstride].z - B[i + (j-1)*jstride].z) );

        E[i + j*jstride].z +=  dt_rl * (                                // Δt/r
            + rc * B[i + j * jstride].θ - rcm * B[i + (j-1)*jstride].θ  // ∂(r Bθ)/∂r 
            - mI * B[i + j * jstride].r                                 // - m I Br
        );

    }

    if ( ir0 == 0 ) {
        block_sync();
        if ( m == 1 ) {
            // Mode m = 1 is a special case
            for( int i = block_thread_rank(); i < static_cast<int>(nx.x) + 2; i += block_num_threads() ) {
                E[ i + 0 *jstride ].r = ( 4.f * E[ i + 1*jstride ].r - E[ i + 2*jstride ].r ) / 3.f;  
                E[ i + 0 *jstride ].θ =  E[ i + 1 * jstride ].θ;  // ∂Bθ/∂r(r=0) = 0
                E[ i + 0 *jstride ].z = -E[ i + 1 * jstride ].z;  // Ez(r=0) = 0
            }
        } else {
            for( int i = block_thread_rank(); i < static_cast<int>(nx.x) + 2; i += block_num_threads() ) {
                E[ i + 0 *jstride ].r = 0;
                E[ i + 0 *jstride ].θ = - E[ i + 1 *jstride ].θ;    // Eθ(r=0) = 0
                E[ i + 0 *jstride ].z = - E[ i + 1 *jstride ].z;    // Ez(r=0) = 0;  
            }
        }
    }
}

__global__ __launch_bounds__(opt_yee_block) void yeem( 
    const int m,
    cyl_cfloat3 * const __restrict__ E, 
    cyl_cfloat3 * const __restrict__ B, 
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx, unsigned int const offset, 
    const double dt, float2 const dx ) {

    auto * shm = block::shared_mem<cyl_cfloat3>();

    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    const int jstride = ext_nx.x;
    const int ir0 = tile_idx.y * nx.y;

    cyl_cfloat3 * const __restrict__ E_local = & shm[ 0 ];
    cyl_cfloat3 * const __restrict__ B_local = & shm[ tile_vol ];

    // Copy E and B into shared memory
    for( unsigned i = block_thread_rank(); i < tile_vol; i += block_num_threads() ) {
        E_local[i] = E[ tile_off + i ];
        B_local[i] = B[ tile_off + i ];
    }

    auto * const __restrict__ tile_E = & E_local[ offset ];
    auto * const __restrict__ tile_B = & B_local[ offset ];

    block_sync();

    yeem_b( m, tile_E, tile_B, nx, jstride, dt/2, dx, ir0 );
    block_sync();

    yeem_e( m, tile_E, tile_B, nx, jstride, dt,   dx, ir0 );
    block_sync();

    yeem_b( m, tile_E, tile_B, nx, jstride, dt/2, dx, ir0 );
    block_sync();

    // Copy data to global memory
    for( unsigned i = block_thread_rank(); i < tile_vol; i += block_num_threads() ) {
        E[ tile_off + i ] = E_local[i];
        B[ tile_off + i ] = B_local[i];
    }
}

}

/**
 * @brief Advance EM fields 1 time step (no current)
 * 
 */
void EMF::advance() {

    // Get tile information from mode0
    auto & E0 = E -> mode0();
    auto & B0 = B -> mode0();

    const auto field_vol = E0.tile_vol;

    dim3 grid( E0.ntiles.x, E0.ntiles.y );
    auto block = opt_yee_block;

    // Solve for mode 0
    size_t shm_size = 2 * field_vol * sizeof(cyl_float3);
    block::set_shmem_size( kernel::yee0, shm_size );
    kernel::yee0 <<< grid, block, shm_size >>> (
        E0.d_buffer, B0.d_buffer, 
        E0.ntiles, E0.nx, E0.ext_nx, E0.offset,
        dt, dx
    );

    // Solve for high-order modes
    for( int m = 1; m < nmodes; m++ ) { 
        auto & Em = E -> mode( m );
        auto & Bm = B -> mode( m );

        size_t shm_size = 2 * field_vol * sizeof(cyl_cfloat3);
        block::set_shmem_size( kernel::yeem, shm_size );
        kernel::yeem <<< grid, block, shm_size >>> (
            m, Em.d_buffer, Bm.d_buffer, 
            Em.ntiles, Em.nx, Em.ext_nx, Em.offset,
            dt, dx
        );
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



namespace kernel {

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
__device__ void yee0J_e( 
    cyl_float3 * const __restrict__ E, 
    cyl_float3 const * const __restrict__ B, 
    uint2 const nx, int const jstride, 
    cyl_float3 const * const __restrict__ J, int const J_jstride, 
    float const dt, float2 const dx, const int ir0 )
{
    // cylindrical cell sizes
    const auto dz    = dx.x;
    const auto dr    = dx.y;

    const auto dt_dz = dt / dz;
    const auto dt_dr = dt / dr;

    // Standard advance away from axial boundary
    const int j0 = ( ir0 > 0 ) ? 0 : 1;

    int const range_z = nx.x + 2;      // [0, nx.x + 2[
    int const range_r = nx.y + 2 - j0; // [j0, nx.y + 2[

    for( int idx = block_thread_rank(); idx < range_z * range_r; idx += block_num_threads() ) {
        const int i = idx % range_z;
        const int j = idx / range_z + j0;

        /// @brief r at center of j-1 cell, normalized to Δr
        float rcm  = ir0 + j - 1;
        /// @brief r at center of j cell, normalized to Δr
        float rc   = ir0 + j;
        /// @brief Δt/r at the lower edge of j cell
        float dt_rm = dt / ( ( ir0 + j - 0.5 ) * dr );

        E[i + j*jstride].r += - dt_dz * ( B[i + j*jstride].θ - B[(i-1) + j*jstride].θ)
                                - dt * J[i + j*J_jstride].r;

        E[i + j*jstride].θ += ( + dt_dz * ( B[i + j*jstride].r - B[(i-1) + j*jstride].r) - 
                                    dt_dr * ( B[i + j*jstride].z - B[i + (j-1)*jstride].z) )
                                - dt * J[i + j*J_jstride].θ;

        E[i + j*jstride].z += ( + dt_rm * ( rc * B[i + j*jstride].θ - rcm * B[i + (j-1)*jstride].θ) )
                                - dt * J[i + j*J_jstride].z;
    }

    if ( ir0 == 0 ) {
        block_sync();
        for( int i = block_thread_rank(); i < static_cast<int>(nx.x) + 2; i += block_num_threads() ) {
            E[i +   0 *jstride].r = 0;
            E[i +   0 *jstride].θ = -E[i + 1*jstride].θ;
            E[i +   0 *jstride].z = E[i + 1*jstride].z;
        }
    }
}

__global__ __launch_bounds__(opt_yee_block) void yee0J( 
    cyl_float3 * const __restrict__ E,
    cyl_float3 * const __restrict__ B,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx, unsigned int const offset, 
    cyl_float3 * const __restrict__ J_buffer,
    uint2 const J_ext_nx, unsigned int const J_offset, 
    float2 const dx, float const dt ) {

    auto * shm = block::shared_mem<cyl_float3>();

    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    const int jstride = ext_nx.x;
    const int ir0 = tile_idx.y * nx.y;

    auto * const __restrict__ E_local = & shm[ 0 ];
    auto * const __restrict__ B_local = & shm[ tile_vol ];

    // Copy E and B into shared memory
    for( unsigned i = block_thread_rank(); i < tile_vol; i += block_num_threads() ) {
        E_local[i] = E[ tile_off + i ];
        B_local[i] = B[ tile_off + i ];
    }

    auto * const __restrict__ tile_E = & E_local[ offset ];
    auto * const __restrict__ tile_B = & B_local[ offset ];

    const int    J_tile_vol = roundup4( J_ext_nx.x * J_ext_nx.y );
    const int    J_jstride  = J_ext_nx.x;
    cyl_float3 * const __restrict__ tile_J = & J_buffer[ tile_id * J_tile_vol + J_offset ];

    block_sync();

    yee0_b( tile_E, tile_B, nx, jstride, dt/2, dx, ir0 );
    block_sync();

    yee0J_e( tile_E, tile_B, nx, jstride, tile_J, J_jstride, dt, dx, ir0 );
    block_sync();

    yee0_b( tile_E, tile_B, nx, jstride, dt/2, dx, ir0 );
    block_sync();

    // Copy data to global memory
    for( unsigned i = block_thread_rank(); i < tile_vol; i += block_num_threads() ) {
        E[ tile_off + i ] = E_local[i];
        B[ tile_off + i ] = B_local[i];
    }

}

__device__ void yeemJ_e( 
    const int m,
    cyl_cfloat3 * const __restrict__ E, 
    cyl_cfloat3 const * const __restrict__ B, 
    uint2 const nx, int const jstride, 
    cyl_cfloat3 const * const __restrict__ J, int const J_jstride, 
    float const dt, float2 const dx, const int ir0 )
{
    // cylindrical cell sizes
    const auto dz    = dx.x;
    const auto dr    = dx.y;

    const auto dt_dz = dt / dz;
    const auto dt_dr = dt / dr;

    // Standard advance away from axial boundary
    const int j0 = ( ir0 > 0 ) ? 0 : 1;

    /// @brief m * imaginary unit
    const ops::complex<float> mI{0,static_cast<float>(m)};

    int const range_z = nx.x + 2;      // [0, nx.x + 2[
    int const range_r = nx.y + 2 - j0; // [j0, nx.y + 2[

    for( int idx = block_thread_rank(); idx < range_z * range_r; idx += block_num_threads() ) {
        const int i = idx % range_z;
        const int j = idx / range_z + j0;

        /// @brief r/Δr at center of j-1 cell
        float rcm  = ir0 + j - 1;
        /// @brief rΔr at center of j cell
        float rc   = ir0 + j;
        /// @brief Δt/r at the lower edge of j cell
        float dt_rl = dt / ( ( ir0 + j - 0.5 ) * dr );
        /// @brief Δt/r at the center of j cell
        float dt_rc = dt / ( ( ir0 + j ) * dr );

        E[i + j*jstride].r +=  
            + dt_rc * mI * B[ i + j * jstride ].z                      // (Δt/r) m I Bz
            - dt_dz * ( B[i + j*jstride].θ - B[(i-1) + j*jstride].θ)   // Δt ∂Bθ/∂z 
            - dt * J[i + j*J_jstride].r;

        E[i + j*jstride].θ += 
            + dt_dz * ( B[i + j*jstride].r - B[(i-1) + j*jstride].r)
            - dt_dr * ( B[i + j*jstride].z - B[i + (j-1)*jstride].z) 
            - dt * J[i + j*J_jstride].θ;

        E[i + j*jstride].z +=  dt_rl * (                                // Δt/r
            + rc * B[i + j * jstride].θ - rcm * B[i + (j-1)*jstride].θ  // ∂(r Bθ)/∂r 
            - mI * B[i + j * jstride].r                                 // m I Br
        ) - dt * J[i + j*J_jstride].z;
    }

    if ( ir0 == 0 ) {
        block_sync();
        for( int i = block_thread_rank(); i < static_cast<int>(nx.x) + 2; i += block_num_threads() ) {
            E[i +   0 *jstride].r = 0;
            E[i +   0 *jstride].θ = -E[i + 1*jstride].θ;
            E[i +   0 *jstride].z = E[i + 1*jstride].z;
        }
        if ( m == 1 ) {
            // Mode m = 1 is a special case
            for( int i = block_thread_rank(); i < static_cast<int>(nx.x) + 2; i += block_num_threads() ) {
                E[ i + 0 *jstride ].r = ( 4.f * E[ i + 1*jstride ].r - E[ i + 2*jstride ].r ) / 3.f;
                E[ i + 0 *jstride ].θ =  E[ i + 1 * jstride ].θ;   // ∂Bθ/∂r(r=0) = 0
                E[ i + 0 *jstride ].z = -E[ i + 1 * jstride ].z;  // Ez(r=0) = 0
            }
        } else {
            for( int i = block_thread_rank(); i < static_cast<int>(nx.x) + 2; i += block_num_threads() ) {
                E[ i + 0 *jstride ].r = 0;
                E[ i + 0 *jstride ].θ = - E[ i + 1 *jstride ].θ;    // Eθ(r=0) = 0
                E[ i + 0 *jstride ].z = - E[ i + 1 *jstride ].z;    // Ez(r=0) = 0;  
            }
        }

    }
}

__global__ __launch_bounds__(opt_yee_block) void yeemJ( 
    const int m,
    cyl_cfloat3 * const __restrict__ E, 
    cyl_cfloat3 * const __restrict__ B, 
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx, unsigned int const offset, 
    cyl_cfloat3 * const __restrict__ J_buffer,
    uint2 const J_ext_nx, unsigned int const J_offset, 
    float2 const dx, float const dt ) {

    auto * shm = block::shared_mem<cyl_cfloat3>();

    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    const int jstride = ext_nx.x;
    const int ir0 = tile_idx.y * nx.y;

    cyl_cfloat3 * const __restrict__ E_local = & shm[ 0 ];
    cyl_cfloat3 * const __restrict__ B_local = & shm[ tile_vol ];

    // Copy E and B into shared memory
    for( unsigned i = block_thread_rank(); i < tile_vol; i += block_num_threads() ) {
        E_local[i] = E[ tile_off + i ];
        B_local[i] = B[ tile_off + i ];
    }

    auto * const __restrict__ tile_E = & E_local[ offset ];
    auto * const __restrict__ tile_B = & B_local[ offset ];

    const int    J_tile_vol = roundup4( J_ext_nx.x * J_ext_nx.y );
    const int    J_jstride  = J_ext_nx.x;
    cyl_cfloat3 * const __restrict__ tile_J = & J_buffer[ tile_id * J_tile_vol + J_offset ];

    block_sync();

    yeem_b( m, tile_E, tile_B, nx, jstride, dt/2, dx, ir0 );
    block_sync();

    yeemJ_e( m, tile_E, tile_B, nx, jstride, tile_J, J_jstride, dt, dx, ir0 );
    block_sync();

    yeem_b( m, tile_E, tile_B, nx, jstride, dt/2, dx, ir0 );
    block_sync();

    // Copy data to global memory
    for( unsigned i = block_thread_rank(); i < tile_vol; i += block_num_threads() ) {
        E[ tile_off + i ] = E_local[i];
        B[ tile_off + i ] = B_local[i];
    }
}

}

/**
 * @brief Advance EM fields 1 time step including current
 * 
 * @param current   Electric current
 */
void EMF::advance( Current & current ) {

    // Get tile information from mode0
    auto & E0 = E -> mode0();
    auto & B0 = B -> mode0();

    const auto field_vol = E0.tile_vol;

    auto & J0 = current.mode0();

    dim3 grid( E0.ntiles.x, E0.ntiles.y );
    auto block = opt_yee_block;
    
    // Solve for mode 0
    size_t shm_size = 2 * field_vol * sizeof(float3);
    block::set_shmem_size( kernel::yee0J, shm_size );
    kernel::yee0J <<< grid, block, shm_size >>> (
        E0.d_buffer, B0.d_buffer, 
        E0.ntiles, E0.nx, E0.ext_nx, E0.offset,
        J0.d_buffer, J0.ext_nx, J0.offset,
        dx, static_cast<float>(dt)
    );

    // Solve for high-order modes
    for( int m = 1; m < nmodes; m++ ) { 
        auto & Em = E -> mode( m );
        auto & Bm = B -> mode( m );
        auto & Jm = current.mode( m );

        size_t shm_size = 2 * field_vol * sizeof(cyl_cfloat3);
        block::set_shmem_size( kernel::yeem, shm_size );
        kernel::yeemJ <<< grid, block, shm_size >>> (
            m, Em.d_buffer, Bm.d_buffer, 
            Em.ntiles, Em.nx, Em.ext_nx, Em.offset,
            Jm.d_buffer, Jm.ext_nx, Jm.offset,
            dx, static_cast<float>(dt)
        );
    }

    // Update guard cells with new values
    E -> copy_to_gc( );
    B -> copy_to_gc( );

    // Do additional bc calculations if needed
    // process_bc();

    // Advance internal iteration number
    iter += 1;

    // Move simulation window if needed
    if ( moving_window.active() ) move_window( );
}

/**
 * @brief Save EM field component to file
 * 
 * @param field     Which field to save (E or B)
 * @param fc        Which field component to save (r, θ or z)
 * @param m         Mode
 */
void EMF::save( emf::field const field, const fcomp::cyl fc, const int m ) {
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

namespace kernel {

__global__
void get_energy( 
    cyl_float3 * const __restrict__ E,
    cyl_float3 * const __restrict__ B,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx, unsigned int const offset, 
    double dz, double dr,
    double * const __restrict__ d_energy ) {

    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    cyl_float3 * const __restrict__ tile_E = & E[ tile_off + offset ];
    cyl_float3 * const __restrict__ tile_B = & B[ tile_off + offset ];

    const int jstride = ext_nx.x;
    int ir0 = tile_idx.y * nx.y;

    cyl_double3 tile_ene_E{0};
    cyl_double3 tile_ene_B{0};

    // Axial cells are not included
    int const j0 = ( ir0 > 0 ) ? 0 : 1;
    int const range_z = nx.x;
    int const range_r = nx.y - j0;

    for( int idx = block_thread_rank(); idx < range_z * range_r; idx += block_num_threads() ) {
        int const i = idx % nx.x;
        int const j = idx / nx.x + j0;

        double rc = ( j + ir0       ) * dr;
        double rm = ( j + ir0 - 0.5 ) * dr;

        auto efld = tile_E[ j * jstride + i ];
        auto bfld = tile_B[ j * jstride + i ];

        tile_ene_E.r += rc * efld.r * efld.r;
        tile_ene_E.θ += rm * efld.θ * efld.θ;
        tile_ene_E.z += rm * efld.z * efld.z;

        tile_ene_B.r += rm * bfld.r * bfld.r;
        tile_ene_B.θ += rc * bfld.θ * bfld.θ;
        tile_ene_B.z += rc * bfld.z * bfld.z;
    }

    // Add up energy from all warps
    tile_ene_E.r = warp::reduce_add( tile_ene_E.r );
    tile_ene_E.θ = warp::reduce_add( tile_ene_E.θ );
    tile_ene_E.z = warp::reduce_add( tile_ene_E.z );

    tile_ene_B.r = warp::reduce_add( tile_ene_B.r );
    tile_ene_B.θ = warp::reduce_add( tile_ene_B.θ );
    tile_ene_B.z = warp::reduce_add( tile_ene_B.z );

    if ( warp::thread_rank() == 0 ) {
        device::atomic_fetch_add( &(d_energy[0]), tile_ene_E.r );
        device::atomic_fetch_add( &(d_energy[1]), tile_ene_E.θ );
        device::atomic_fetch_add( &(d_energy[2]), tile_ene_E.z );

        device::atomic_fetch_add( &(d_energy[3]), tile_ene_B.r );
        device::atomic_fetch_add( &(d_energy[4]), tile_ene_B.θ );
        device::atomic_fetch_add( &(d_energy[5]), tile_ene_B.z );
    }
}

__global__
void get_energy( 
    cyl_cfloat3 * const __restrict__ E_buffer,
    cyl_cfloat3 * const __restrict__ B_buffer,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx, unsigned int const offset, 
    double dz, double dr,
    double * const __restrict__ d_energy ) {

    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    cyl_cfloat3 * const __restrict__ tile_E = & E_buffer[ tile_off + offset ];
    cyl_cfloat3 * const __restrict__ tile_B = & B_buffer[ tile_off + offset ];

    const int jstride = ext_nx.x;
    int ir0 = tile_idx.y * nx.y;

    cyl_double3 tile_ene_E{0};
    cyl_double3 tile_ene_B{0};

    // Axial cells are not included
    int const j0 = ( ir0 > 0 ) ? 0 : 1;
    int const range_z = nx.x;
    int const range_r = nx.y - j0;

    for( int idx = block_thread_rank(); idx < range_z * range_r; idx += block_num_threads() ) {
        int const i = idx % nx.x;
        int const j = idx / nx.x + j0;

        double rc = ( j + ir0       ) * dr;
        double rm = ( j + ir0 - 0.5 ) * dr;

        auto efld = tile_E[ j * jstride + i ];
        auto bfld = tile_B[ j * jstride + i ];

        tile_ene_E.r += rc * norm( efld.r );
        tile_ene_E.θ += rm * norm( efld.θ );
        tile_ene_E.z += rm * norm( efld.z );

        tile_ene_B.r += rm * norm( bfld.r );
        tile_ene_B.θ += rc * norm( bfld.θ );
        tile_ene_B.z += rc * norm( bfld.z );
    }

    // Add up energy from all warps
    tile_ene_E.r = warp::reduce_add( tile_ene_E.r );
    tile_ene_E.θ = warp::reduce_add( tile_ene_E.θ );
    tile_ene_E.z = warp::reduce_add( tile_ene_E.z );

    tile_ene_B.r = warp::reduce_add( tile_ene_B.r );
    tile_ene_B.θ = warp::reduce_add( tile_ene_B.θ );
    tile_ene_B.z = warp::reduce_add( tile_ene_B.z );

    if ( warp::thread_rank() == 0 ) {
        device::atomic_fetch_add( &(d_energy[0]), tile_ene_E.r );
        device::atomic_fetch_add( &(d_energy[1]), tile_ene_E.θ );
        device::atomic_fetch_add( &(d_energy[2]), tile_ene_E.z );

        device::atomic_fetch_add( &(d_energy[3]), tile_ene_B.r );
        device::atomic_fetch_add( &(d_energy[4]), tile_ene_B.θ );
        device::atomic_fetch_add( &(d_energy[5]), tile_ene_B.z );
    }
}

}


/**
 * @brief Get EM field energy
 * 
 * @note The energy will be recalculated each time this routine is called;
 * @note Axial cell values are currently being ignored
 * 
* @param ene_E     Electric field energy
* @param ene_b     Magnetic field energy
* @param m         Mode
*/
void EMF::get_energy( cyl_double3 & ene_E, cyl_double3 & ene_B, const int m ) {

    // Get tile information from mode0
    auto & E0 = E -> mode0();
    auto & B0 = B -> mode0();

    // Zero energy values
    device::zero( d_energy, 6 );

    dim3 grid( E0.ntiles.x, E0.ntiles.y );
    dim3 block( 1024 );

    if ( m == 0 ) {
        kernel::get_energy <<< grid, block >>> ( 
            E0.d_buffer, B0.d_buffer,
            E0.ntiles, E0.nx, E0.ext_nx, E0.offset,
            dx.x, dx.y, d_energy
        );
    } else {
        if ( m > nmodes ) {
            std::cerr << "(*error*) Invalid mode (" << m << ") requested, aborting...\n";
            std::exit(1);
        }

        auto & Em = E -> mode(m);
        auto & Bm = B -> mode(m);

        kernel::get_energy <<< grid, block >>> ( 
            Em.d_buffer, Bm.d_buffer,
            Em.ntiles, Em.nx, Em.ext_nx, Em.offset,
            dx.x, dx.y, d_energy
        );

    }

    // Copy results to host and normalize
    double h_energy[6];
    device::memcpy_tohost( h_energy, d_energy, 6 );

    ene_E = cyl_double3{ h_energy[0], h_energy[1], h_energy[2] };
    ene_B = cyl_double3{ h_energy[3], h_energy[4], h_energy[5] };

    const double dz = dx.x;
    const double dr = dx.y;
    const double norm = dz * dr * M_PI;
    ene_E *= norm;
    ene_B *= norm;
}