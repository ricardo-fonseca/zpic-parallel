#include "laser.h"

#include <iostream>
#include <cassert>

#include "filter.h"

namespace kernel {

/**
 * @brief Gets longitudinal laser envelope a given position
 * 
 * @param z         position
 * @param start     Start position
 * @param rise      Rise length
 * @param flat      Flat length
 * @param fall      Fall length
 * @return float    laser envelope
 */
__device__ __inline__
float lon_env( Laser::Pulse & laser, float z ) {

    if ( z > laser.start ) {
        // Ahead of laser
        return 0.0;
    } else if ( z > laser.start - laser.rise ) {
        // Laser rise
        float csi = z - laser.start;
        float e = sin( M_PI_2 * csi / laser.rise );
        return e*e;
    } else if ( z > laser.start - (laser.rise + laser.flat) ) {
        // Flat-top
        return 1.0;
    } else if ( z > laser.start - (laser.rise + laser.flat + laser.fall) ) {
        // Laser fall
        float csi = z - (laser.start - laser.rise - laser.flat - laser.fall);
        float e = sin( M_PI_2 * csi / laser.fall );
        return e*e;
    }

    // Before laser
    return 0.0;
}

/**
 * @brief Divergence correction - Prefix scan local data to get longitudinal components
 * 
 * @note Must be called with grid( local_nx.y ) and block( WARP_SIZE )
 * 
 * @param E_buffer      E field tile buffer (including offset to cell [0,0])
 * @param B_buffer      B field tile buffer (including offset to cell [0,0])
 * @param ntiles        Number of tiles in local node (x,y)
 * @param nx            Individual tile size
 * @param ext_nx        Individual tile size including guard cells
 * @param recvbuf       Message send buffer
 */
__global__
void div_corr_x_scan (
    float3 * const __restrict__ E_buffer, 
    float3 * const __restrict__ B_buffer,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx,
    double const dx_dy, double * const __restrict__ sendbuf
)
{
    int local_nx_y = gridDim.x;
    int local_iy = blockIdx.x;

    const int tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const int ystride = ext_nx.x;

    int ty = local_iy / nx.y;
    int iy = local_iy - ty * nx.y;

    __shared__ double divEx; divEx = 0;
    __shared__ double divBx; divBx = 0;

    block_sync();

    // Process tiles right to left
    for( int tx = ntiles.x-1; tx >=0; tx-- ) {

        const auto tile_idx = make_uint2( tx, ty );
        const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
        const auto tile_off = tid * tile_vol;

        float3 * const __restrict__ tile_E = & E_buffer[ tile_off ];
        float3 * const __restrict__ tile_B = & B_buffer[ tile_off ];

        int start = nx.x / WARP_SIZE;
        for( int ix = start + block_thread_rank(); ix >= 0; ix -= WARP_SIZE ) {

            double dEx = 0;
            double dBx = 0;

            // If inside tile read x divergence
            if ( ix < nx.x ) {
                dEx = dx_dy * (tile_E[ix+1 +     iy*ystride].y - tile_E[ix+1 + (iy-1)*ystride].y);
                dBx = dx_dy * (tile_B[ix   + (iy+1)*ystride].y - tile_B[ix   +     iy*ystride].y);
            }

            // Do a right to left inclusive scan
            {
                const int laneId = threadIdx.x & ( WARP_SIZE - 1 );
                #pragma unroll
                for( int i = 1; i < WARP_SIZE; i <<= 1 ) {
                    double tmp1 = __shfl_down_sync( 0xffffffff, dEx, i );
                    double tmp2 = __shfl_down_sync( 0xffffffff, dBx, i );
                    if ( laneId < WARP_SIZE - i ) {
                        dEx += tmp1;
                        dBx += tmp2;
                    }
                }
            }

            // If inside tile store longitudinal components
            if ( ix < nx.x ) {
                tile_E[ix + iy * ystride].x = divEx + dEx;
                tile_B[ix + iy * ystride].x = divBx + dBx;
            }

            // Accumulate results for next loop
            if ( block_thread_rank() == 0 ) {
                divEx += dEx;
                divBx += dBx;
            }
            block_sync();
        }
    }
    
    if ( block_thread_rank() == 0 ) {
        // Copy final divergence values to send buffer
        sendbuf[              local_iy ] = divEx;
        sendbuf[ local_nx_y + local_iy ] = divBx;
    }

}

/**
 * @brief Divergence correction - Add contribution from other parallel nodes
 * 
 * @note Must be called with grid( local_nx.y )
 * 
 * @param E_buffer      E field tile buffer (including offset to cell [0,0])
 * @param B_buffer      B field tile buffer (including offset to cell [0,0])
 * @param ntiles        Number of tiles in local node (x,y)
 * @param nx            Individual tile size
 * @param ext_nx        Individual tile size including guard cells
 * @param recvbuf       Message receive buffer
 */
__global__
void div_corr_x_sum (
    float3 * const __restrict__ E_buffer, 
    float3 * const __restrict__ B_buffer,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx,
    double * const __restrict__ recvbuf
)
{
    int local_nx_y = gridDim.x;
    int local_iy = blockIdx.x;

    const int tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const int ystride = ext_nx.x;

    double divEx = recvbuf[              local_iy ];
    double divBx = recvbuf[ local_nx_y + local_iy ];

    int ty = local_iy / nx.y;
    int iy = local_iy - ty * nx.y;

    for( int tx = 0; tx < ntiles.x; tx++ ) {

        const auto tile_idx = make_uint2( tx, ty );
        const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
        const auto tile_off = tid * tile_vol;

        float3 * const __restrict__ tile_E = & E_buffer[ tile_off ];
        float3 * const __restrict__ tile_B = & B_buffer[ tile_off ];

        for( int ix = block_thread_rank(); ix < nx.x; ix+= block_num_threads() ) {
            tile_E[ ix + iy * ystride ].x += divEx;
            tile_B[ ix + iy * ystride ].x += divBx;
        }
    }    

}

}

/**
 * @brief Sets the longitudinal field components of E and B to ensure 0
 *        divergence
 * 
 * @warning The algorithm assumes 0 field at the right boundary of the box. If
 *          this is not true then the field values will be wrong.
 * 
 * @param E     E field
 * @param B     B field
 * @param dx    Cell size
 */
void div_corr_x( vec3grid<float3>& E, vec3grid<float3>& B, const float2 dx ) {

    double  * __restrict__ sendbuf = managed::malloc<double>( 2 * E.local_nx.y );

    const double dx_dy = (double) dx.x / (double) dx.y;

    dim3 grid( E.local_nx.y );
    dim3 block( WARP_SIZE );

    kernel::div_corr_x_scan <<< grid, block >>> ( 
        & E.d_buffer[ E.offset ], & B.d_buffer[ B.offset ], 
        E.get_ntiles(), E.nx, E.ext_nx, dx_dy, 
        sendbuf
    );


    // If there is a parallel partition along x, add in contribution from nodes to the right
    if ( E.part.dims.x > 1 ) {
        double * __restrict__ recvbuf = device::malloc<double>( 2 * E.local_nx.y );
        
        // Create a communicator with nodes having the same y coordinate
        int color = E.part.get_coords().y;
        // Reorder ranks right to left
        int key   = E.part.dims.x - E.part.get_coords().x;
        MPI_Comm newcomm;
        MPI_Comm_split( E.part.get_comm(), color, key, &newcomm );
        
        // Add contribution from all nodes to the right
        MPI_Exscan( sendbuf, recvbuf, 2 * E.local_nx.y, MPI_DOUBLE, MPI_SUM, newcomm );

        // Add result to local grid
        // Rightmost node does not need to do this
        if ( key > 0 ) {
            kernel::div_corr_x_sum <<< grid, block >>> ( 
                & E.d_buffer[ E.offset ], & B.d_buffer[ B.offset ], 
                E.get_ntiles(), E.nx, E.ext_nx, 
                recvbuf );
        }

        MPI_Comm_free( &newcomm );
        device::free( recvbuf );
    }

    // Free temporary memory
    device::free( sendbuf );    

    // Correct longitudinal values on guard cells
    E.copy_to_gc();
    B.copy_to_gc();
}


/**
 * @brief Validates laser parameters
 * 
 * @return      0 on success, -1 on error
 */
int Laser::Pulse::validate() {

    if ( a0 <= 0 ) {
        std::cerr << "(*error*) Invalid laser a0, must be > 0\n";
        return -1;
    }    

    if ( omega0 <= 0 ) {
        std::cerr << "(*error*) Invalid laser OMEGA0, must be > 0\n";
        return -1;
    }    

    if ( fwhm > 0 ) {
        // The fwhm parameter overrides the rise/flat/fall parameters
        rise = fwhm;
        fall = fwhm;
        flat = 0.;
    } else {
        if ( rise <= 0 ) {
            std::cerr << "(*error*) Invalid laser RISE, must be > 0\n";
            return (-1);
        }

        if ( flat < 0 ) {
            std::cerr << "(*error*) Invalid laser FLAT, must be >= 0\n";
            return (-1);
        }

        if ( fall <= 0 ) {
            std::cerr << "(*error*) Invalid laser FALL, must be > 0\n";
            return (-1);
        }
    }

    return 0;
}


namespace kernel {

__global__
void plane_wave(
    Laser::PlaneWave laser, 
    float3 * __restrict__ E_buffer, float3 * __restrict__ B_buffer,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx, uint2 const  global_tile_off,
    float2 const dx )
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    float3 * const __restrict__ tile_E = & E_buffer[ tile_off ];
    float3 * const __restrict__ tile_B = & B_buffer[ tile_off ];

    const int ix0 = ( global_tile_off.x + tile_idx.x ) * nx.x;
    const float k = laser.omega0;
    const float amp = laser.omega0 * laser.a0;
    const int ystride = ext_nx.x;

    for( unsigned idx = block_thread_rank(); idx < nx.y * nx.x; idx += block_num_threads() ) {
        const auto ix = idx % nx.x;
        const auto iy = idx / nx.x; 

        const float z   = ( ix0 + ix ) * dx.x;
        const float z_2 = ( ix0 + ix + 0.5 ) * dx.x;

        float lenv   = amp * lon_env( laser, z );
        float lenv_2 = amp * lon_env( laser, z_2 );

        tile_E[ ix + iy * ystride ] = make_float3(
            0,
            +lenv * cos( k * z ) * laser.cos_pol,
            +lenv * cos( k * z ) * laser.sin_pol
        );

        tile_B[ ix + iy * ystride ] = make_float3(
            0,
            -lenv_2 * cos( k * z_2 ) * laser.sin_pol,
            +lenv_2 * cos( k * z_2 ) * laser.cos_pol
        );
    }
}


}

/**
 * @brief Launches a plane wave
 * 
 * The E and B tiled grids have the complete laser field.
 * 
 * @param E     Electric field
 * @param B     Magnetic field
 * @param box   Box size
 * @return      Returns 0 on success, -1 on error (invalid laser parameters)
 */
int Laser::PlaneWave::launch( vec3grid<float3>& E, vec3grid<float3>& B, float2 box ) {

    if ( validate() < 0 ) return -1;

    if (( cos_pol == 0 ) && ( sin_pol == 0 )) {
        cos_pol = cos( polarization );
        sin_pol = sin( polarization );
    }

    uint2 g_nx = E.get_global_nx();

    float2 dx = make_float2(
        box.x / g_nx.x,
        box.y / g_nx.y
    );

    const auto ntiles     = E.get_ntiles();
    dim3 block( 64 );
    dim3 grid( ntiles.x, ntiles.y );

    kernel::plane_wave<<<grid, block>>> (
        *this, & E.d_buffer[ E.offset ], & B.d_buffer[ B.offset ],
        ntiles, E.nx, E.ext_nx, E.get_tile_off(), dx
    );

    E.copy_to_gc( );
    B.copy_to_gc( );

    if ( filter > 0 ) {

        Filter::Compensated fcomp( coord::x, filter);
        fcomp.apply( E );
        fcomp.apply( B );
    }

    return 0;
}

/**
 * @brief Validate Gaussian laser parameters
 * 
 * @return      0 on success, -1 on error
 */
int Laser::Gaussian::validate() {
    
    if ( Laser::Pulse::validate() < 0 ) {
        return -1;
    }

    if ( W0 <= 0 ) {
        std::cerr << "(*error*) Invalid laser W0, must be > 0\n";
        return (-1);
    }

    return 0;
}

namespace kernel {

/**
 * @brief Returns local phase for a gaussian beamn
 * 
 * @param omega0    Beam frequency
 * @param W0        Beam waist
 * @param z         Position along focal line (focal plane at z = 0)
 * @param r         Position transverse to focal line (focal line at r = 0)
 * @return          Local field value
 */
__device__ float gauss_phase( const float omega0, const float W0, const float z, const float r ) {
    const float z0   = omega0 * ( W0 * W0 ) / 2;
    const float rho2 = r*r;
    const float curv = 0.5 * rho2 * z / (z0*z0 + z*z);
    const float rWl2 = (z0*z0)/(z0*z0 + z*z);
    const float gouy_shift = atan2( z, z0 );

    return sqrt( sqrt(rWl2) ) * 
           exp( - rho2 * rWl2/( W0 * W0 ) ) * 
           cos( omega0*( z + curv ) - gouy_shift );
}

__global__
void gaussian( 
    Laser::Gaussian beam, 
    float3 * __restrict__ E_buffer, float3 * __restrict__ B_buffer,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx, uint2 const  global_tile_off,
    float2 const dx
) {
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    float3 * const __restrict__ tile_E = & E_buffer[ tile_off ];
    float3 * const __restrict__ tile_B = & B_buffer[ tile_off ];

    const int ix0 = ( global_tile_off.x + tile_idx.x ) * nx.x;
    const int iy0 = ( global_tile_off.y + tile_idx.y ) * nx.y;

    const int ystride = ext_nx.x;

    const float amp = beam.omega0 * beam.a0;

    for( unsigned idx = block_thread_rank(); idx < nx.y * nx.x; idx += block_num_threads() ) {
        const auto ix = idx % nx.x;
        const auto iy = idx / nx.x; 

        const float z   = ( ix0 + ix ) * dx.x;
        const float z_2 = ( ix0 + ix + 0.5 ) * dx.x;

        const float r   = (iy0 + iy ) * dx.y - beam.axis;
        const float r_2 = (iy0 + iy + 0.5 ) * dx.y - beam.axis;

        const float lenv   = amp * lon_env( beam, z );
        const float lenv_2 = amp * lon_env( beam, z_2 );

        tile_E[ ix + iy * ystride ] = make_float3(
            0,
            +lenv * gauss_phase( beam.omega0, beam.W0, z - beam.focus, r_2 ) * beam.cos_pol,
            +lenv * gauss_phase( beam.omega0, beam.W0, z - beam.focus, r   ) * beam.sin_pol
        );
        tile_B[ ix + iy * ystride ] = make_float3(
            0,
            -lenv_2 * gauss_phase( beam.omega0, beam.W0, z_2 - beam.focus, r   ) * beam.sin_pol,
            +lenv_2 * gauss_phase( beam.omega0, beam.W0, z_2 - beam.focus, r_2 ) * beam.cos_pol
        );
    }
}

}

/**
 * @brief Launches a Gaussian pulse
 * 
 * The E and B tiled grids have the complete laser field.
 * 
 * @param E     Electric field
 * @param B     Magnetic field
 * @param dx    Cell size
 * @return      Returns 0 on success, -1 on error (invalid laser parameters)
 */
int Laser::Gaussian::launch(vec3grid<float3>& E, vec3grid<float3>& B, float2 const box ) {

    // std::cout << "Launching gaussian pulse...\n";

    if ( validate() < 0 ) return -1;

    if (( cos_pol == 0 ) && ( sin_pol == 0 )) {
        cos_pol = cos( polarization );
        sin_pol = sin( polarization );
    }

    uint2 g_nx = E.get_global_nx();

    float2 dx = make_float2(
        box.x / g_nx.x,
        box.y / g_nx.y
    );

    const auto ntiles     = E.get_ntiles();
    dim3 block( 64 );
    dim3 grid( ntiles.x, ntiles.y );

    kernel::gaussian<<<grid, block>>> (
        *this, & E.d_buffer[ E.offset ], & B.d_buffer[ B.offset ],
        ntiles, E.nx, E.ext_nx, E.get_tile_off(), dx
    );

    E.copy_to_gc( );
    B.copy_to_gc( );

    // Get longitudinal field components by enforcing div = 0
    div_corr_x( E, B, dx );

    // Apply filtering if required
    if ( filter > 0 ) {
        Filter::Compensated fcomp( coord::x, filter);
        fcomp.apply( E );
        fcomp.apply( B );
    }
    
    return 0;
}