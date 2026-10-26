#include "core/gpu.cuh"
#include "species.hpp"
#include <vector>
#include <cstddef>
#include <iostream>
#include <string>


#include "grid/tiled.cuh"
#include "parallel/partition.hpp"
#include "util/memory.cuh"

/**
 * The following values were determined experimentally using a single
 * NVIDIA A100 80GB PCIe board
 * 
 */

/// @brief Optimal block size for push kernel
int constexpr opt_push_block = 1024;

/// @brief Optimal block size for move kernel
int constexpr opt_move_block = 256;

/// @brief Optimal minimum number of blocks for move/push kernels
int constexpr opt_min_blocks = 2048;


namespace kernel {

    /**
 * @brief Returns reciprocal Lorentz gamma factor
 * 
 * $ \frac{1}{\sqrt{u_x^2 + u_y^2 + u_z^2 + 1 }} $
 * 
 * @param u         Generalized momentum in units of c
 * @return float    Reciprocal Lorentz gamma factor
 */
__device__
inline float rgamma( const float3 u ) {
    return rsqrt( fma( u.z, u.z, 
                fma( u.y, u.y, 
                fma( u.x, u.x, 1.0f ) ) ) );
}

/**
 * @brief Interpolate EM field values at particle position using linear 
 * (1st order) interpolation.
 * 
 * @note The EM fields are assumed to be organized according to the Yee scheme with
 * the charge defined at lower left corner of the cell
 * 
 * @param E         Pointer to position (0,0) of E field grid
 * @param B         Pointer to position (0,0) of B field grid
 * @param ystride   E and B grids y stride (must be signed)
 * @param ix        Particle cell index
 * @param x         Particle postion inside cell
 * @param e[out]    E field at particle position
 * @param b[out]    B field at particle position
 */
__device__
void interpolate_fld( 
    float3 const * const __restrict__ E, 
    float3 const * const __restrict__ B, 
    const int ystride,
    const int2 ix, const float2 x, float3 & e, float3 & b)
{
    const int i = ix.x;
    const int j = ix.y;

    const float s0x = 0.5f - x.x;
    const float s1x = 0.5f + x.x;

    const float s0y = 0.5f - x.y;
    const float s1y = 0.5f + x.y;


    // Interpolate E field


    e.x = ( E[i +    j *ystride].x * s0x + E[i+1 +    j *ystride].x * s1x ) * s0y +
          ( E[i + (j+1)*ystride].x * s0x + E[i+1 + (j+1)*ystride].x * s1x ) * s1y;

    e.y = ( E[i +    j *ystride].y * s0x + E[i+1 +    j *ystride].y * s1x ) * s0y +
          ( E[i + (j+1)*ystride].y * s0x + E[i+1 + (j+1)*ystride].y * s1x ) * s1y;

    e.z = ( E[i +    j *ystride].z * s0x + E[i+1 +    j *ystride].z * s1x ) * s0y +
          ( E[i + (j+1)*ystride].z * s0x + E[i+1 + (j+1)*ystride].z * s1x ) * s1y;

    // Interpolate B field
    b.x = ( B[i +    j *ystride].x * s0x + B[i+1 +    j *ystride].x * s1x ) * s0y +
          ( B[i + (j+1)*ystride].x * s0x + B[i+1 + (j+1)*ystride].x * s1x ) * s1y;

    b.y = ( B[i +    j *ystride].y * s0x + B[i+1 +    j *ystride].y * s1x ) * s0y +
          ( B[i + (j+1)*ystride].y * s0x + B[i+1 + (j+1)*ystride].y * s1x ) * s1y;

    b.z = ( B[i +    j *ystride].z * s0x + B[i+1 +    j *ystride].z * s1x ) * s0y +
          ( B[i + (j+1)*ystride].z * s0x + B[i+1 + (j+1)*ystride].z * s1x ) * s1y;

}

/**
 * @brief Advance momentum using a relativistic Boris pusher.
 * 
 * The momentum advance in this method is split into 3 parts:
 * 1. Perform half of E-field acceleration
 * 2. Perform full B-field rotation
 * 3. Perform half of E-field acceleration
 * 
 * Note that this implementation (as it is usual in textbooks) uses a
 * linearization of a tangent calculation in the rotation, which may lead
 * to issues for high magnetic fields.
 * 
 * For the future, other, more accurate, rotation algorithms should be used
 * instead, such as employing the full Euler-Rodrigues formula.
 * 
 * @param alpha     Normalization 
 * @param e         E-field interpolated at particle position
 * @param b         B-field interpolated at particle position
 * @param u         Initial particle momentum
 * @param energy    Particle energy (time centered)
 * @return float3   Final particle momentum
 */
__device__
float3 dudt_boris( const float alpha, float3 e, float3 b, float3 u, double & energy )
{

    // First half of acceleration
    e.x *= alpha;
    e.y *= alpha;
    e.z *= alpha;

    float3 ut = make_float3( 
        u.x + e.x,
        u.y + e.y,
        u.z + e.z
    );

    {
        const float utsq = fma( ut.z, ut.z, fma( ut.y, ut.y, ut.x * ut.x ) );
        const float gamma = sqrt( 1.0f + utsq );
        
        // Get time centered energy
        energy += utsq / (gamma + 1.0f);

        // Time centered \alpha / \gamma
        const float alpha_gamma = alpha / gamma;

        // Rotation
        b.x *= alpha_gamma;
        b.y *= alpha_gamma;
        b.z *= alpha_gamma;
    }

    u.x = fma( b.z, ut.y, ut.x );
    u.y = fma( b.x, ut.z, ut.y );
    u.z = fma( b.y, ut.x, ut.z );

    u.x = fma( -b.y, ut.z, u.x );
    u.y = fma( -b.z, ut.x, u.y );
    u.z = fma( -b.x, ut.y, u.z );

    {
        const float otsq = 2.0f / 
            fma( b.z, b.z, fma( b.y, b.y, fma( b.x, b.x, 1.0f ) ) );
        
        b.x *= otsq;
        b.y *= otsq;
        b.z *= otsq;
    }

    ut.x = fma( b.z, u.y, ut.x );
    ut.y = fma( b.x, u.z, ut.y );
    ut.z = fma( b.y, u.x, ut.z );

    ut.x = fma( -b.y, u.z, ut.x );
    ut.y = fma( -b.z, u.x, ut.y );
    ut.z = fma( -b.x, u.y, ut.z );

    // Second half of acceleration
    ut.x += e.x;
    ut.y += e.y;
    ut.z += e.z;

    return ut;
}

/**
 * @brief Advance momentum using a relativistic Boris pusher for high magnetic fields
 * 
 * This is similar to the dudt_boris method above, but the rotation is done using
 * using an exact Euler-Rodriguez method.2
 * 
 * @param tem 
 * @param e 
 * @param b 
 * @param u 
 * @return float3 
 */
__device__
inline float3 dudt_boris_euler( const float alpha, float3 e, float3 b, float3 u, double & energy )
{

    // First half of acceleration
    e.x *= alpha;
    e.y *= alpha;
    e.z *= alpha;

    float3 ut = make_float3( 
        u.x + e.x,
        u.y + e.y,
        u.z + e.z
    );

    {
        const float utsq = fma( ut.z, ut.z, fma( ut.y, ut.y, ut.x * ut.x ) );
        const float gamma = sqrt( 1.0f + utsq );
        
        // Get time centered energy
        energy += utsq / (gamma + 1.0f);
        
        // Time centered 2 * \alpha / \gamma
        float const alpha2_gamma = ( alpha * 2 ) / gamma ;

        b.x *= alpha2_gamma;
        b.y *= alpha2_gamma;
        b.z *= alpha2_gamma;
    }

    {
        float const bnorm = sqrt(fma( b.x, b.x, fma( b.y, b.y, b.z * b.z ) ));
        float const s = -(( bnorm > 0 ) ? sin( bnorm / 2 ) / bnorm : 0.5 );

        float const ra = cos( bnorm / 2 );
        float const rb = b.x * s;
        float const rc = b.y * s;
        float const rd = b.z * s;

        float const r11 =   fma(ra,ra,rb*rb)-fma(rc,rc,rd*rd);
        float const r12 = 2*fma(rb,rc,ra*rd);
        float const r13 = 2*fma(rb,rd,-ra*rc);

        float const r21 = 2*fma(rb,rc,-ra*rd);
        float const r22 =   fma(ra,ra,rc*rc)-fma(rb,rb,rd*rd);
        float const r23 = 2*fma(rc,rd,ra*rb);

        float const r31 = 2*fma(rb,rd,ra*rc);
        float const r32 = 2*fma(rc,rd,-ra*rb);
        float const r33 =   fma(ra,ra,rd*rd)-fma(rb,rb,rc*rc);

        u.x = fma( r11, ut.x, fma( r21, ut.y , r31 * ut.z ));
        u.y = fma( r12, ut.x, fma( r22, ut.y , r32 * ut.z ));
        u.z = fma( r13, ut.x, fma( r23, ut.y , r33 * ut.z ));
    }

    // Second half of acceleration
    u.x += e.x;
    u.y += e.y;
    u.z += e.z;

    return u;
}

/**
 * @brief Deposit current from single particle
 * 
 * @param J         Electric current buffer
 * @param ystride   y-stride for J
 * @param ix0       Initial position of particle (cell)
 * @param x0        Initial position of particle (position inside cell)
 * @param u         Particle momenta
 * @param rg        1 / Lorentz γ
 * @param dx        Particle motion normalized to cell size
 * @param q         Particle charge
 */
__device__
inline void dep_current( 
    float3 * const __restrict__ J, const int ystride, 
    int2 ix0, float2 x0, float3 u, 
    float rg, float2 dx, float q ) {

    // Find position time centered with velocity
    float2 x1 = make_float2( x0.x + 0.5f * dx.x, x0.y + 0.5f * dx.y );
    
    int2 const deltai = make_int2(
        ((x1.x >= 0.5f) - (x1.x < -0.5f)),
        ((x1.y >= 0.5f) - (x1.y < -0.5f))
    );

    int2 ix  = make_int2( ix0.x + deltai.x, ix0.y + deltai.y );
    float2 x = make_float2( x1.x - deltai.x, x1.y - deltai.y );

    const float S0x = 0.5f - x.x;
    const float S1x = 0.5f + x.x;

    const float S0y = 0.5f - x.y;
    const float S1y = 0.5f + x.y;

    const float jx = q * u.x * rg;
    const float jy = q * u.y * rg;
    const float jz = q * u.z * rg;

    int idx = ix.y * ystride + ix.x;

    gpu::block::atomic_fetch_add( &J[ idx               ].x,  S0y * S0x * jx );
    gpu::block::atomic_fetch_add( &J[ idx               ].y,  S0y * S0x * jy );
    gpu::block::atomic_fetch_add( &J[ idx               ].z,  S0y * S0x * jz );

    gpu::block::atomic_fetch_add( &J[ idx + 1           ].x,  S0y * S1x * jx );
    gpu::block::atomic_fetch_add( &J[ idx + 1           ].y,  S0y * S1x * jy );
    gpu::block::atomic_fetch_add( &J[ idx + 1           ].z,  S0y * S1x * jz );

    gpu::block::atomic_fetch_add( &J[ idx + ystride     ].x,  S1y * S0x * jx );
    gpu::block::atomic_fetch_add( &J[ idx + ystride     ].y,  S1y * S0x * jy );
    gpu::block::atomic_fetch_add( &J[ idx + ystride     ].z,  S1y * S0x * jz );

    gpu::block::atomic_fetch_add( &J[ idx + ystride + 1 ].x,  S1y * S1x * jx );
    gpu::block::atomic_fetch_add( &J[ idx + ystride + 1 ].y,  S1y * S1x * jy );
    gpu::block::atomic_fetch_add( &J[ idx + ystride + 1 ].z,  S1y * S1x * jz );
}

/**
 * @brief Deposit charge from single particle
 * 
 * @param rho       Charge density buffer
 * @param ystride   y-stride for rho
 * @param ix        Particle position (cell)
 * @param x         Particle position inside cell
 * @param q         Particle charge
 */
__device__
inline void dep_charge( float * const __restrict__ rho, const int ystride, int2 ix, float2 x, float q )
{

    const float S0x = 0.5f - x.x;
    const float S1x = 0.5f + x.x;

    const float S0y = 0.5f - x.y;
    const float S1y = 0.5f + x.y;

    int idx = ix.y * ystride + ix.x;

    gpu::block::atomic_fetch_add( & rho[ idx               ], S0y * S0x * q );
    gpu::block::atomic_fetch_add( & rho[ idx + 1           ], S0y * S1x * q );
    gpu::block::atomic_fetch_add( & rho[ idx + ystride     ], S1y * S0x * q );
    gpu::block::atomic_fetch_add( & rho[ idx + ystride + 1 ], S1y * S1x * q );
}

/**
 * @brief Move particles and deposit current
 * 
 * @param tile_idx          Tile index
 * @param part              Particle data
 * @param d_current         current grid (global)
 * @param current_offset    Offset to position [0,0] of the current grid
 * @param ext_nx            current grid size (external)
 * @param dt_dx             Ratio between time step and cell size
 * @param q                 Particle charge
 * @param d_nmove           Pointer to total number of particles moved
 */
__global__
void __launch_bounds__(opt_move_block) move_deposit(
    part::particles_view const part,
    grid::tiled_view<float3> current,
    grid::tiled_view<float> charge,
    float2 const dt_dx, float const q,
    unsigned long long * const __restrict__ d_nmove ) 
{
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id   = tile_idx.y * part.local_ntiles.x + tile_idx.x;
    const int tile_np   = part.tile_np[ tile_id ];

   // Select subset of particles to move (if gridDim.z > 1)
    // We are using multiple blocks per tile (blockIdx.z)
    const int set_size = roundup<gpu::warp_size>(( tile_np + gridDim.z - 1 ) / gridDim.z);
    const int begin = blockIdx.z * set_size;

    // If the block does not have any particles to process return
    if ( begin >= tile_np ) return;

    // Local current / charge buffers
    extern __shared__ char block_shm[];
    float  * __restrict__ _charge_buffer   = reinterpret_cast< float * >  ( & block_shm[0] );
    float3 * __restrict__ _current_buffer  = reinterpret_cast< float3 * > ( & block_shm[charge.tile_vol * sizeof(float)] );

    // Zero local buffers
    for( int i = gpu::block::thread_rank(); i < charge.tile_vol; i += gpu::block::num_threads() )
        _charge_buffer[i] = 0.f;
    for( int i = gpu::block::thread_rank(); i < current.tile_vol; i += gpu::block::num_threads() )
        _current_buffer[i] = make_float3(0,0,0);

    gpu::block::sync();

    // Move particles and deposit current
    const int end   = ( begin + set_size < tile_np ) ? begin + set_size : tile_np;

    float3 * __restrict__ J   = & _current_buffer[ current.inner_offset ];
    float  * __restrict__ rho = & _charge_buffer[ charge.inner_offset ];

    const int part_offset    = part.tile_offset[ tile_id ];
    int2   * __restrict__ ix = &part.ix[ part_offset ];
    float2 * __restrict__ x  = &part.x[ part_offset ];
    float3 * __restrict__ u  = &part.u[ part_offset ];

    for( auto i = begin + gpu::block::thread_rank(); i < end; i+= gpu::block::num_threads() ) {
        float3 pu = u[i];
        float2 const x0 = x[i];
        int2   const ix0 =ix[i];

        // Get 1 / Lorentz gamma
        float const rg = rgamma( pu );

        // Get particle motion
        float2 const delta = make_float2(
            dt_dx.x * rg * pu.x,
            dt_dx.y * rg * pu.y
        );

        // Deposit current
        dep_current( J, current.tile_ystride(), ix0, x0, pu, rg, delta, q );

        // Advance position
        float2 x1 = make_float2(
            x0.x + delta.x,
            x0.y + delta.y
        );

        // Check for cell crossings
        int2 const deltai = make_int2(
            ((x1.x >= 0.5f) - (x1.x < -0.5f)),
            ((x1.y >= 0.5f) - (x1.y < -0.5f))
        );

        // Correct position
        x1.x -= deltai.x;
        x1.y -= deltai.y;

        // Modify cell
        int2 ix1 = make_int2(
            ix0.x + deltai.x,
            ix0.y + deltai.y
        );

        // Deposit charge
        dep_charge( rho, charge.tile_ystride(), ix1, x1, q );

        // Store result
        x[i] = x1;
        ix[i] = ix1;
    }

    // sync
    gpu::block::sync();

    // Add current and charge to global buffers
    float3 * __restrict__ J_buffer = current.tile_buffer( tile_id );
    float  * __restrict__ rho_buffer = charge.tile_buffer( tile_id );
    if ( gridDim.z > 1 ) {
        // Multiple blocks deposit to the same buffers
        for( auto i =  gpu::block::thread_rank(); i < current.tile_vol; i+= gpu::block::num_threads() ) {
            gpu::device::atomic_fetch_add( & J_buffer[i].x, _current_buffer[i].x );
            gpu::device::atomic_fetch_add( & J_buffer[i].y, _current_buffer[i].y );
            gpu::device::atomic_fetch_add( & J_buffer[i].z, _current_buffer[i].z );
        }
        for( auto i =  gpu::block::thread_rank(); i < charge.tile_vol; i+= gpu::block::num_threads() ) {
            gpu::device::atomic_fetch_add( & rho_buffer[i], _charge_buffer[i] );
        }
    } else {
        // only 1 block deposits to each buffer, no need for atomic ops
        for( auto i =  gpu::block::thread_rank(); i < current.tile_vol; i+= gpu::block::num_threads() )
            J_buffer[i] += _current_buffer[i];
        for( auto i =  gpu::block::thread_rank(); i < charge.tile_vol; i+= gpu::block::num_threads() )
            rho_buffer[i] += _charge_buffer[i];
    }

    if ( gpu::block::thread_rank() == 0 && blockIdx.z == 0 ) {
        // Update total particle pushes counter (for performance metrics)
        unsigned long long np64 = tile_np;
        gpu::device::atomic_fetch_add( d_nmove, np64 );
    }

}

/**
 * @brief Advance particle velocities
 * 
 * @tparam type         Pusher type
 * @param tile_idx      Tile index
 * @param part          Particle data view
 * @param E_grid        E field tiled grid view
 * @param B_grid        B fiels tiled grid view
 * @param alpha         Normalization parameter
 * @param d_energy      Total particle energy (if using OpenMP this must be a reduction variable)
 */
template < species::pusher type >
__global__
void push ( 
    part::particles_view const part,
    grid::tiled_view<float3> const E_grid, grid::tiled_view<float3> const B_grid,
    float const alpha, double * __restrict__ d_energy )
{
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id   = tile_idx.y * part.local_ntiles.x + tile_idx.x;
    const int tile_np   = part.tile_np[ tile_id ];

    // Select subset of particles to move
    // We are using multiple blocks per tile (blockIdx.z)
    const int set_size = roundup<gpu::warp_size>(( tile_np + gridDim.z - 1 ) / gridDim.z);
    const int begin = blockIdx.z * set_size;

    // If the block does not have any particles to process then return
    if ( begin >= tile_np ) return;

    // Copy E and B into shared memory
    extern __shared__ float3 local[];
    float3 * __restrict__ E_local = & local[0];
    float3 * __restrict__ B_local = & local[ E_grid.tile_vol ];

    gpu::block::memcpy(E_local, E_grid.tile_buffer(tile_id), E_grid.tile_vol);
    gpu::block::memcpy(B_local, B_grid.tile_buffer(tile_id), B_grid.tile_vol);

    gpu::block::sync();

    float3 const * const __restrict__ E = & E_local[ E_grid.inner_offset ];
    float3 const * const __restrict__ B = & B_local[ B_grid.inner_offset ];

    // Push particles
    const int end   = ( begin + set_size < tile_np ) ? begin + set_size : tile_np;
    const int part_offset = part.tile_offset[ tile_id ];
    int2   * __restrict__ ix = &part.ix[ part_offset ];
    float2 * __restrict__ x  = &part.x[ part_offset ];
    float3 * __restrict__ u  = &part.u[ part_offset ];

    double energy = 0;

    const int ystride = E_grid.tile_ystride();

    for( auto i = begin + gpu::block::thread_rank(); i < end; i+= gpu::block::num_threads() ) {

        // Interpolate field
        float3 e, b;
        interpolate_fld( E, B, ystride, ix[i], x[i], e, b );
        
        // Advance momentum
        float3 pu = u[i];
        
        if constexpr ( type == species::pusher::boris ) u[i] = dudt_boris( alpha, e, b, pu, energy );
        if constexpr ( type == species::pusher::euler ) u[i] = dudt_boris_euler( alpha, e, b, pu, energy );
    }

    // Add up energy from all particles
    energy = gpu::warp::reduce_add( energy );
    if ( gpu::warp::thread_rank() == 0 ) { 
        gpu::device::atomic_fetch_add( d_energy, energy );
    }
}

} // namespace kernel


/**
 * @brief Initialize data structures and inject initial particle distribution
 * 
 * @param box_              Global simulation box size
 * @param global_ntiles     Global number of tiles
 * @param tile_dims         Individutal tile grid size
 * @param dt_               Time step
 * @param id_               Species unique identifier
 * @param parallel          Parallel configuration
 */
void species::initialize( float2 const box_, uint2 const global_ntiles, uint2 const tile_dims,
    float const dt_, int const id_, mpi::cart2d & parallel ) {
    
    // Store simulation box size
    box = box_;

    // Store simulation time step
    dt = dt_;

    // Store species id (used by RNG)
    id = id_;

    // Set charge normalization factor
    q = copysign( density->n0 , m_q ) / (ppc.x * ppc.y);
       
    // Get global grid size
    auto  global_dims = tile_dims * global_ntiles;

    // Set cell size
    dx.x = box.x / (global_dims.x);
    dx.y = box.y / (global_dims.y);

    /// @brief Number of tiles in local parallel node
    uint2 local_ntiles = parallel.grid_size( global_ntiles );

    /// @brief Local parallel node grid size
    uint2  local_dims = tile_dims * local_ntiles;

    // Reference number maximum number of particles
    unsigned int max_part = 1.2 * local_dims.x * local_dims.y * ppc.x * ppc.y;

    // Validate boundary conditions against the parallel partition
    validate_bc( global_bc, parallel );

    // Compute local_bc
    // (part, tmp and sort are still null here, they get the periodic flags on construction)
    update_bc( parallel );

    // Create particle data structures 
    int2 const periodic = global_periodic();
    particles = new part::particles( global_ntiles, tile_dims, max_part, parallel, periodic );
    tmp = new part::particles( global_ntiles, tile_dims, max_part, parallel, periodic );
    sort = new part::particle_sort( local_ntiles, max_part, parallel, periodic );
    np_inj = gpu::device::malloc<int>( local_ntiles.x * local_ntiles.y );

    // Initialize energy diagnostic
    d_energy = gpu::device::malloc<double>(1);
    gpu::device::zero( d_energy, 1 );

    // Initialize particle move counter
    d_nmove = gpu::device::malloc< unsigned long long >( 1 );
    gpu::device::zero( d_nmove, 1 );

    // Reset iteration numbers
    iter = 0;

    // Inject initial distribution

    // Count particles to inject and store in np_inj
    np_inject( particles -> local_range(), np_inj );

    // Do an exclusive scan to get the required offsets
    auto num_part = gpu::device::exscan_reduce_add( particles -> get_tile_offset(), np_inj, local_ntiles.x * local_ntiles.y );

    // In case the reference maximum number of particles was too small, grow
    // the buffers. Note: this is just a safeguard, it should never happen
    if ( static_cast<unsigned int>(num_part) > max_part ) {
        mpi::cout << "(*warning*) Particle buffer too small for initial injection,"
                     " growing buffer\n";
        particles -> grow_buffer(num_part);
        tmp -> grow_buffer(num_part );
        sort -> check_buffer(num_part);
    }

    // Inject the particles
    inject( particles -> local_range() );

    // Set inital velocity distribution
    udist -> set( *particles, id );
}

namespace kernel {
/**
 * @brief Physical boundary conditions for the x direction 
 * 
 * @note Launch with grid( 2, local_ntiles.y ): blockIdx.x selects the
 *       boundary (0 - lower, 1 - upper), blockIdx.y the tile row
 * 
 * @param part          Particle
 * @param local_bc      Local rank boundary condition
 */
__global__
void species_bcx(
    part::particles_view const part,
    species::bc_type const local_bc ) 
{
    int bnd_x = blockIdx.x;
    int tile_idx_y = blockIdx.y;

    const uint2 ntiles  = part.local_ntiles;
    const int nx = part.tile_dims.x;
        
    const int tile_idx_x = (bnd_x==0)?0 : ntiles.x - 1;
    const int tid = tile_idx_y * ntiles.x + tile_idx_x;

    const int part_offset    = part.tile_offset[ tid ];
    const int np             = part.tile_np[ tid ];
    int2   * __restrict__ ix = &part.ix[ part_offset ];
    float2 * __restrict__ x  = &part.x[ part_offset ];
    float3 * __restrict__ u  = &part.u[ part_offset ];

    if ( bnd_x == 0 ) {
        // Lower boundary
        switch( local_bc.x.lower ) {
        case( species::bc::reflecting ) :
            for( int i = gpu::block::thread_rank(); i < np; i+= gpu::block::num_threads() ) {
                if( ix[i].x < 0 ) {
                    ix[i].x += 1;
                    x[i].x = -x[i].x;
                    u[i].x = -u[i].x;
                }
            }
            break;
        default:
            break;
        }
    } else {
        // Upper boundary
        switch( local_bc.x.upper ) {
        case( species::bc::reflecting ) :
            for( int i = gpu::block::thread_rank(); i < np; i+= gpu::block::num_threads() ) {
                if( ix[i].x >=  nx ) {
                    ix[i].x -= 1;
                    // Protects against x[i].x exactly == -0.5
                    x[i].x = fminf( -x[i].x, nextafterf( 0.5f, 0.f ) );
                    u[i].x = -u[i].x;
                }
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
 * @note Launch with grid( local_ntiles.x, 2 ): blockIdx.x selects the tile
 *       column, blockIdx.y the boundary (0 - lower, 1 - upper)
 *
 * @param part          Particle
 * @param local_bc      Local rank boundary conditions
 */
__global__
void species_bcy(
    part::particles_view const part,
    species::bc_type const local_bc ) {

    int tile_idx_x = blockIdx.x;
    int bnd_y = blockIdx.y;

    const uint2 ntiles  = part.local_ntiles;
    const int ny = part.tile_dims.y;

    const int tile_idx_y = (bnd_y==0)? 0 : ntiles.y - 1;
    const int tid = tile_idx_y * ntiles.x + tile_idx_x;

    const int part_offset    = part.tile_offset[ tid ];
    const int np             = part.tile_np[ tid ];
    int2   * __restrict__ ix = &part.ix[ part_offset ];
    float2 * __restrict__ x  = &part.x[ part_offset ];
    float3 * __restrict__ u  = &part.u[ part_offset ];

    if ( bnd_y == 0 ) {
        // Lower boundary
        switch( local_bc.y.lower ) {
        case( species::bc::reflecting ) :
            for( int i = gpu::block::thread_rank(); i < np; i+= gpu::block::num_threads() ) {
                if( ix[i].y < 0 ) {
                    ix[i].y += 1;
                    x[i].y = -x[i].y;
                    u[i].y = -u[i].y;
                }
            }
            break;
        default:
            break;
        }
    } else {
        // Upper boundary
        switch( local_bc.y.upper ) {
        case( species::bc::reflecting ) :
            for( int i = gpu::block::thread_rank(); i < np; i+= gpu::block::num_threads() ) {
                if( ix[i].y >=  ny ) {
                    ix[i].y -= 1;
                    // Protects against x[i].y exactly == -0.5
                    x[i].y = fminf( -x[i].y, nextafterf( 0.5f, 0.f ) );
                    u[i].y = -u[i].y;
                }
            }
            break;
        default:
            break;
        }
    }
}

} // namespace kernel

/**
 * @brief Processes "physical" boundary conditions
 * 
 * @note open and periodic boundaries are processed elsehwere
 */
void species::process_bc() {

    // x boundaries
    if ( local_bc.x.lower > species::bc::periodic || local_bc.x.upper > species::bc::periodic ) {
        dim3 block( 1024 );
        dim3 grid( 2, particles->get_local_ntiles().y );
        kernel::species_bcx <<< grid, block >>> ( particles -> view(), local_bc );
    }

    // y boundaries
    if ( local_bc.y.lower > species::bc::periodic || local_bc.y.upper > species::bc::periodic ) {
        dim3 block( 1024 );
        dim3 grid( particles->get_local_ntiles().x, 2 );
        kernel::species_bcy <<< grid, block >>> ( particles -> view(), local_bc );

    }
}


/**
 * @brief Moves particles and deposit current
 * 
 * current will be accumulated on existing data
 * 
 * @param current   current grid
 */
void species::move( grid::tiled_vec3<float> & J, grid::tiled<float> & rho )
{
    const float2 dt_dx = make_float2(
        dt / dx.x,
        dt / dx.y
    );

    auto ntiles = particles -> get_local_ntiles();

    // If number of tiles is small use multiple blocks per tile
    int tile_blocks = opt_min_blocks / ( ntiles.x * ntiles.y );
    if ( tile_blocks < 1 ) tile_blocks = 1;
    dim3 grid( ntiles.x, ntiles.y, tile_blocks );

    size_t shm_size = rho.tile_vol * sizeof(float) + J.tile_vol * sizeof(float3);

    auto block = opt_move_block;
    gpu::block::set_shmem_size( kernel::move_deposit, shm_size );
    kernel::move_deposit <<< grid, block, shm_size >>> ( 
        particles -> view(), J.view(), rho.view(), dt_dx, q, d_nmove
    );

}

namespace kernel{

/**
 * @brief kernel for moving particles (no current/charge deposition)
 * 
 * @param part      particle data
 * @param dt_dx     Time step over cell size
 * @param d_nmove   Number of particles moved
 */
__global__
void move(
    part::particles_view const part,
    float2 const dt_dx,
    unsigned long long * const __restrict__ d_nmove ) 
{
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const uint2 ntiles  = part.local_ntiles;

    // Move particles and deposit current
    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    const int part_offset    = part.tile_offset[ tid ];
    const int np             = part.tile_np[ tid ];
    int2   * __restrict__ ix = &part.ix[ part_offset ];
    float2 * __restrict__ x  = &part.x[ part_offset ];
    float3 * __restrict__ u  = &part.u[ part_offset ];

    for( int i = gpu::block::thread_rank(); i < np; i+= gpu::block::num_threads() ) {
        float3 pu = u[i];
        float2 x0 = x[i];
        int2 ix0 =ix[i];

        // Get 1 / Lorentz gamma
        float rg = rgamma( pu );

        // Get particle motion
        float2 delta = make_float2(
            dt_dx.x * rg * pu.x,
            dt_dx.y * rg * pu.y
        );

        // Advance position
        float2 x1 = make_float2(
            x0.x + delta.x,
            x0.y + delta.y
        );

        // Check for cell crossings
        int2 deltai = make_int2(
            ((x1.x >= 0.5f) - (x1.x < -0.5f)),
            ((x1.y >= 0.5f) - (x1.y < -0.5f))
        );

        // Correct position and store
        x1.x -= deltai.x;
        x1.y -= deltai.y;
                
        x[i] = x1;

        // Modify cell and store
        int2 ix1 = make_int2(
            ix0.x + deltai.x,
            ix0.y + deltai.y
        );
        ix[i] = ix1;
    }

    if ( gpu::block::thread_rank() == 0 ) { 
        // Update total particle pushes counter (for performance metrics)
        unsigned long long np64 = np;
        gpu::device::atomic_fetch_add( d_nmove, np64 );
    }
}

} // namespace kernel


/**
 * @brief Moves particles (no current/charge deposition)
 * 
 * This is usually used for test species: species that do not self-consistently
 * influence the simulation
 * 
 * @param current   current grid
 */
void species::move( )
{
    const float2 dt_dx = make_float2(
        dt / dx.x,
        dt / dx.y
    );

    dim3 grid( particles -> get_local_ntiles().x, particles -> get_local_ntiles().y );
    auto block = 1024;

    kernel::move <<< grid, block >>> ( 
        particles -> view(), dt_dx, d_nmove
    );
}

/**
 * @brief       Accelerates particles using a Boris pusher
 * 
 * @param E     Electric field
 * @param B     Magnetic field
 */
void species::push( grid::tiled_vec3<float> & E, grid::tiled_vec3<float> & B )
{
    const float alpha = 0.5 * dt / m_q;
    
    gpu::device::zero( d_energy, 1 );

    auto ntiles = particles -> get_local_ntiles();

    int tile_blocks = opt_min_blocks / ( ntiles.x * ntiles.y );
    if ( tile_blocks < 1 ) tile_blocks = 1;
    dim3 grid( ntiles.x, ntiles.y, tile_blocks );
    
    auto block = opt_push_block;
    size_t shm_size = 2 * ( E.tile_vol * sizeof(float3) );

    switch( push_type ) {
    case( species::pusher::euler ):
        gpu::block::set_shmem_size( kernel::push <species::pusher::euler>, shm_size );
        kernel::push <species::pusher::euler> <<< grid, block, shm_size >>> (
            particles -> view(), E.view(), B.view(), alpha,
            d_energy
        );
        break;

    case( species::pusher::boris ):
       gpu::block::set_shmem_size( kernel::push <species::pusher::boris>, shm_size );
        kernel::push <species::pusher::boris> <<< grid, block, shm_size >>> ( 
            particles -> view(), E.view(), B.view(), alpha,
            d_energy
        );
        break;
    }
}

namespace kernel {

/**
 * @brief kernel for depositing charge from all species particles
 * 
 * @param part      Particle data
 * @param q         Species charge per particle
 * @param charge    Charge density grid
 */
__global__
void dep_charge_all(
    part::particles_view const part, const float q, 
    grid::tiled_view<float> charge )
{
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const uint2 ntiles  = part.local_ntiles;
 
    extern __shared__ float _charge_buffer[];

    // Zero shared memory and sync.
    for( unsigned i = gpu::block::thread_rank(); i < charge.tile_vol; i += gpu::block::num_threads() ) {
        _charge_buffer[i] = 0;
    }

    float * __restrict__ rho = & _charge_buffer[ charge.inner_offset ];

    gpu::block::sync();

    const int tid      = tile_idx.y * ntiles.x + tile_idx.x;
    const int part_off = part.tile_offset[ tid ];
    const int np       = part.tile_np[ tid ];
    int2   const * __restrict__ const ix = &part.ix[ part_off ];
    float2 const * __restrict__ const x  = &part.x[ part_off ];
    const int ystride = charge.tile_ystride();

    for( int i = gpu::block::thread_rank(); i < np; i += gpu::block::num_threads() ) {
        const int idx = ix[i].y * ystride + ix[i].x;
        const float s0x = 0.5f - x[i].x;
        const float s1x = 0.5f + x[i].x;
        const float s0y = 0.5f - x[i].y;
        const float s1y = 0.5f + x[i].y;

        // Multiple threads per tile, these need to be atomic inside tile
        gpu::block::atomic_fetch_add( &rho[ idx               ], s0y * s0x * q );
        gpu::block::atomic_fetch_add( &rho[ idx + 1           ], s0y * s1x * q );
        gpu::block::atomic_fetch_add( &rho[ idx     + ystride ], s1y * s0x * q );
        gpu::block::atomic_fetch_add( &rho[ idx + 1 + ystride ], s1y * s1x * q );
    }

   gpu::block::sync();

    // Copy data to global memory
    float * __restrict__ tgt = charge.tile_buffer(tid);
    for( unsigned i = gpu::block::thread_rank(); i < charge.tile_vol; i += gpu::block::num_threads() )
        tgt[i] += _charge_buffer[i];
}

}

/**
 * @brief Deposit charge density
 * 
 * @param charge    Charge density grid
 */
void species::deposit_charge( grid::tiled<float> &charge ) const {

    dim3 grid( particles -> get_local_ntiles().x, particles -> get_local_ntiles().y );
    auto block = 64;
    size_t shm_size = charge.tile_vol * sizeof(float);

    gpu::block::set_shmem_size( kernel::dep_charge_all, shm_size );
    kernel::dep_charge_all <<< grid, block, shm_size >>> (
        particles -> view(), q, charge.view()
    );
}

/**
 * @brief Save particle data to file
 * 
 */
void species::save() const {

    const std::string path = "PARTICLES";

    const part::quantity quants[] = {
        part::quantity::x, part::quantity::y,
        part::quantity::ux, part::quantity::uy, part::quantity::uz
    };

    const char * qnames[] = {
        "x","y",
        "ux","uy","uz"
    };

    const char * qlabels[] = {
        "x","y",
        "u_x","u_y","u_z"
    };

    const char * qunits[] = {
        "c/\\omega_n", "c/\\omega_n",
        "c","c","c"
    };

    zdf::iteration iter_info = {
        .name = (char*) "ITERATION",
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_n"
    };

    // Omit number of particles, this will be filled in later
    zdf::part_info info = {
        .name = name.c_str(),
        .label = (char *) name.c_str(),
        .nquants = 5,
        .quants = (char **) qnames,
        .qlabels = (char **) qlabels,
        .qunits = (char **) qunits,
    };

    // Get total number of particles to save
    uint64_t local = local_np();
    uint64_t global = 0;

    particles -> parallel.allreduce( &local, &global, 1, mpi::sum );

    // Update metadata entry
    info.np = global;

    if ( global > 0 ) {
        // Create a communicator including only the nodes with local particles
        int color = ( local > 0 ) ? 1 : MPI_UNDEFINED;
        MPI_Comm comm;
        MPI_Comm_split( particles -> parallel.get_comm(), color, 0, & comm );

        // Only nodes with particles are involved in this section
        if ( local > 0 ) {

            // Get rank in new communicator
            int rank;
            MPI_Comm_rank( comm, & rank );

            // Open file
            zdf::par_file part_file;
            zdf::open_part_file( part_file, info, iter_info, path+"/"+info.name, comm );

            // create the datasets
            std::vector<zdf::dataset> dsets( info.nquants );

            for( uint32_t i = 0; i < info.nquants; i++ ) {
                dsets[i].name      = info.quants[i];
                dsets[i].data_type = zdf::data_type<float>();
                dsets[i].ndims     = 1;
                dsets[i].data      = nullptr;
                dsets[i].count[0]  = global; 

                if ( !zdf::start_cdset( part_file, dsets[i] ) ) {
                    mpi::fatal( 
                        "part::particles::save() - Unable to create chunked dataset " + 
                        std::string(dsets[i].name) );
                }
            }

            // Allocate buffer for gathering particle data
            float *d_data = gpu::device::malloc<float>( local );
            float *h_data = gpu::host::malloc<float>( local );

            // Get offsets - this avoids recalculating offsets for each quantity
            uint64_t file_off;
            MPI_Exscan( &local, &file_off, 1, MPI_UINT64_T, MPI_SUM, comm );
            if ( rank == 0 ) file_off = 0;

            // Local data chunk
            zdf::chunk chunk;
            chunk.count[0] = local;
            chunk.start[0] = file_off;
            chunk.stride[0] = 1;
            chunk.data = h_data;

            float2 scale;
            
            // x and y quantities are scaled before saving to file
            scale = make_float2( dx.x, dx.x/2 );
            particles -> gather( part::quantity::x, scale, d_data );
            gpu::device::memcpy_tohost( h_data, d_data, local );
            zdf::write_cdset( part_file, dsets[0], chunk, file_off );

            scale = make_float2( dx.y, dx.y/2 );
            particles -> gather( part::quantity::y, scale, d_data );
            gpu::device::memcpy_tohost( h_data, d_data, local );
            zdf::write_cdset( part_file, dsets[1], chunk, file_off );

            // Remaining quantities are saved "as is"
            for( uint32_t i = 2; i < info.nquants; i++ ) {
                particles -> gather( quants[i], d_data );
                gpu::device::memcpy_tohost( h_data, d_data, local );
                zdf::write_cdset( part_file, dsets[i], chunk, file_off );
            }

            // Free temporary memory
            gpu::device::free( d_data );
            gpu::host::free( h_data );

            // close the datasets
            for( uint32_t i = 0; i < info.nquants; i++ ) 
                zdf::end_cdset( part_file, dsets[i] );

            // Close the file
            zdf::close_file( part_file );

            MPI_Comm_free(&comm);
        }

    } else {
        // No particles - root node creates an empty file
        if ( particles -> parallel.root() ) {
            zdf::file part_file;
            zdf::open_part_file( part_file, info, iter_info, path+"/"+info.name );

            for ( uint32_t i = 0; i < info.nquants; i ++) {
                zdf::add_quant_part_file( part_file, info.quants[i],  nullptr, 0 );
            }

            zdf::close_file( part_file );
        }
    }
}


/**
 * @brief Saves charge density to file
 * 
 * The routine will create a new charge grid, deposit the charge and save the grid
 * 
 */
void species::save_charge() const {

    // For linear interpolation we only require 1 guard cell at the upper boundary
    bounds_2d<unsigned int> gc;
    gc.x = {0,1};
    gc.y = {0,1};

    // Deposit charge on device
    grid::tiled<float> charge( particles -> get_global_ntiles(), particles -> get_tile_dims(), gc, particles -> parallel );

    charge.zero();

    deposit_charge( charge );

    charge.add_from_gc();

    // Prepare file info
    zdf::grid_axis axis[2];
    axis[0] = zdf::grid_axis {
        .name = (char *) "x",
        .min = 0.,
        .max = box.x,
        .label = (char *) "x",
        .units = (char *) "c/\\omega_n"
    };

    axis[1] = zdf::grid_axis {
        .name = (char *) "y",
        .min = 0.,
        .max = box.y,
        .label = (char *) "y",
        .units = (char *) "c/\\omega_n"
    };

    std::string grid_name = name + "-charge";
    std::string grid_label = name + " \\rho";

    zdf::grid_info info = {
        .name = (char *) grid_name.c_str(),
        .label = (char *) grid_label.c_str(),
        .units = (char *) "n_e",
        .axis  = axis
    };

    zdf::iteration iter_info = {
        .name = (char *) "ITERATION",
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_n"
    };

    std::string path = "CHARGE/";
    path += name;
    
    charge.save( info, iter_info, path );
}

namespace phasespace {

template < phasespace::quantity quant >
__device__
float value ( int const i, int const shiftx, int const shifty,
    int2 const * const __restrict__ ix, float2 const * const __restrict__ x,
    float3 const * const __restrict__ u )
{
    if constexpr ( quant == quantity::x  ) return ( shiftx + ix[i].x ) + ( x[i].x + 0.5f );
    if constexpr ( quant == quantity::y  ) return ( shifty + ix[i].y ) + ( x[i].y + 0.5f );
    if constexpr ( quant == quantity::ux ) return u[i].x;
    if constexpr ( quant == quantity::uy ) return u[i].y;
    if constexpr ( quant == quantity::uz ) return u[i].z;
}

namespace kernel {

template < phasespace::quantity quant >
__global__
void dep_pha1(
    float * const __restrict__ d_data, float2 const range, int const size,
    float const norm, 
    part::particles_view const part )
{
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const uint2 ntiles    = part.local_ntiles;
    const uint2 tile_dims = part.tile_dims;

    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    const int part_offset = part.tile_offset[ tid ];
    const int np          = part.tile_np[ tid ];
    int2   * __restrict__ ix = &part.ix[ part_offset ];
    float2 * __restrict__ x  = &part.x[ part_offset ];
    float3 * __restrict__ u  = &part.u[ part_offset ];

    float const pha_rdx = size / (range.y - range.x);

    const int shiftx = (part.local_tile_start.x + tile_idx.x) * tile_dims.x;
    const int shifty = (part.local_tile_start.y + tile_idx.y) * tile_dims.y;

    for( int i = gpu::block::thread_rank(); i < np; i += gpu::block::num_threads() ) {
        float d = phasespace::value<quant>( i, shiftx, shifty, ix, x, u );

        float n =  (d - range.x ) * pha_rdx - 0.5f;
        int   k = floorf( n );
        float w = n - k;

        // Atomic across threads and tiles
        if ((k   >= 0) && (k   < size)) gpu::device::atomic_fetch_add( &d_data[k  ], (1-w) * norm );
        if ((k+1 >= 0) && (k+1 < size)) gpu::device::atomic_fetch_add( &d_data[k+1],    w  * norm );
    }
}

template < phasespace::quantity quant0, phasespace::quantity quant1 >
__global__
void dep_pha2(
    float * const __restrict__ d_data, 
    float2 const range0, int const size0,
    float2 const range1, int const size1,
    float const norm, 
    part::particles_view const part )
{
    static_assert( quant1 > quant0, "quant1 must be > quant0" );

    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const uint2 ntiles  = part.local_ntiles;
    const auto tile_dims  = part.tile_dims;

    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    const int part_offset = part.tile_offset[ tid ];
    const int np          = part.tile_np[ tid ];
    int2   * __restrict__ ix  = &part.ix[ part_offset ];
    float2 * __restrict__ x   = &part.x[ part_offset ];
    float3 * __restrict__ u   = &part.u[ part_offset ];

    float const pha_rdx0 = size0 / (range0.y - range0.x);
    float const pha_rdx1 = size1 / (range1.y - range1.x);

    const int shiftx = (part.local_tile_start.x + tile_idx.x) * tile_dims.x;
    const int shifty = (part.local_tile_start.y + tile_idx.y) * tile_dims.y;

    for( int i = gpu::block::thread_rank(); i < np; i += gpu::block::num_threads() ) {
        float d0 = phasespace::value<quant0>( i, shiftx, shifty, ix, x, u );;

        float n0 =  (d0 - range0.x ) * pha_rdx0 - 0.5f;
        int   k0 = floorf( n0 );
        float w0 = n0 - k0;

        float d1 = phasespace::value<quant1>( i, shiftx, shifty, ix, x, u );;

        float n1 =  (d1 - range1.x ) * pha_rdx1 - 0.5f;
        int   k1 = floorf( n1 );
        float w1 = n1 - k1;

        // Most common case, particle inside grid for all deposits
        if ( k0 >= 0 && k0+1 < size0 && k1 >= 0 && k1+1 < size1 ) {
            gpu::device::atomic_fetch_add( &d_data[(k1  )*size0 + k0  ], (1-w0) * (1-w1) * norm );
            gpu::device::atomic_fetch_add( &d_data[(k1  )*size0 + k0+1],    w0  * (1-w1) * norm );
            gpu::device::atomic_fetch_add( &d_data[(k1+1)*size0 + k0  ], (1-w0) *    w1  * norm );
            gpu::device::atomic_fetch_add( &d_data[(k1+1)*size0 + k0+1],    w0  *    w1  * norm );
        } else {
            // corner cases
            if ((k1   >= 0) && (k1   < size1)) {
                if ((k0   >= 0) && (k0   < size0)) 
                    gpu::device::atomic_fetch_add( &d_data[(k1  )*size0 + k0  ], (1-w0) * (1-w1) * norm );
                if ((k0+1 >= 0) && (k0+1 < size0)) 
                    gpu::device::atomic_fetch_add( &d_data[(k1  )*size0 + k0+1],    w0  * (1-w1) * norm );
            }
            if ((k1+1 >= 0) && (k1+1 < size1)) {
                if ((k0   >= 0) && (k0   < size0)) 
                    gpu::device::atomic_fetch_add( &d_data[(k1+1)*size0 + k0  ], (1-w0) *    w1  * norm );
                if ((k0+1 >= 0) && (k0+1 < size0)) 
                    gpu::device::atomic_fetch_add( &d_data[(k1+1)*size0 + k0+1],    w0  *    w1  * norm );
            }
        }
    }
}

} // namespace kernel


/**
 * @brief Helper function to dispatch the appropriate template phasespace deposition
 * 
 * @note 
 * The lambda function should be declared as [&] ( auto qc ), and start with
 *     constexpr auto quant = decltype(qc)::value;
 * The dep_pha kernel can then be called using:
 *     dep_pha1<quant>(...)
 *
 * @tparam F    Generic lambda function type
 * @param q     Phasespace quantity to deposit
 * @param f     lambda function that calls the dep_pha* kernel
 */
template < typename F >
inline void dispatch( quantity q, F && f ) {
    switch( q ) {
    case quantity::x : f( std::integral_constant< quantity, quantity::x >{} ); break;
    case quantity::y : f( std::integral_constant< quantity, quantity::y >{} ); break;
    case quantity::ux: f( std::integral_constant< quantity, quantity::ux>{} ); break;
    case quantity::uy: f( std::integral_constant< quantity, quantity::uy>{} ); break;
    case quantity::uz: f( std::integral_constant< quantity, quantity::uz>{} ); break;
    }
}

} // namespace phasespace

/**
 * @brief Deposit 1D phasespace
 * 
 * Output data will be zeroed before deposition
 * 
 * @param d_data    Output (device) data
 * @param quant     Phasespace quantity
 * @param range     Phasespace value range
 * @param size      Phasespace grid size
 */
void species::dep_phasespace( float * const d_data, const phasespace::quantity quant, 
    float2 range, const int size ) const
{
    // Zero device memory
    gpu::device::zero( d_data, size );
    
    // In OSIRIS we don't take the absolute value of q
    float norm = fabs(q) * ( dx.x * dx.y ) *
                 size / (range.y - range.x) ;

    dim3 grid( particles -> get_local_ntiles().x, particles -> get_local_ntiles().y );
    auto block = 64;

    // Launch the appropriate phasespace deposition kernel
    phasespace::dispatch(quant, [&] ( auto quant_c ) {
        constexpr auto quant = decltype(quant_c)::value;
        
        // Normalize range to cell sizes, if required
        if constexpr ( quant == phasespace::quantity::x ) { range.x /= dx.x; range.y /= dx.x; }
        if constexpr ( quant == phasespace::quantity::y ) { range.x /= dx.y; range.y /= dx.y; }
        
        phasespace::kernel::dep_pha1<quant> <<< grid, block >>> (
            d_data, range, size, norm, 
            particles -> view()
        );
    });
}

/**
 * @brief Deposits a 2D phasespace in a device buffer
 * 
 * @param d_data    Pointer to device buffer
 * @param quant0    Quantity 0
 * @param range0    Range of values of quantity 0
 * @param size0     Phasespace grid size for quantity 0
 * @param quant0    Quantity 1
 * @param range1    Range of values of quantity 1
 * @param size1     Phasespace grid size for quantity 1
 */
void species::dep_phasespace( 
    float * const d_data,
    const phasespace::quantity quant0, float2 range0, const int size0,
    const phasespace::quantity quant1, float2 range1, const int size1 ) const
{
    // Zero device memory
    gpu::device::zero( d_data, size0 * size1 );

    // In OSIRIS we don't take the absolute value of q
    float norm = fabs(q) * ( dx.x * dx.y ) *
                          ( size0 / (range0.y - range0.x) ) *
                          ( size1 / (range1.y - range1.x) );

    dim3 grid( particles -> get_local_ntiles().x, particles -> get_local_ntiles().y );
    auto block = 64;

    // Launch the appropriate phasespace deposition kernel
    phasespace::dispatch(quant0, [&] ( auto qc0 ) {
    phasespace::dispatch(quant1, [&] ( auto qc1 ) {

        constexpr auto quant0 = decltype(qc0)::value;
        constexpr auto quant1 = decltype(qc1)::value;
        
        // Don't generate code for quant1 <= quant0
        if constexpr ( quant1 > quant0 ) {

            // Normalize range to cell sizes, if required
            if constexpr ( quant0 == phasespace::quantity::x ) { range0.x /= dx.x; range0.y /= dx.x; }
            if constexpr ( quant0 == phasespace::quantity::y ) { range0.x /= dx.y; range0.y /= dx.y; }
            if constexpr ( quant1 == phasespace::quantity::x ) { range1.x /= dx.x; range1.y /= dx.x; }
            if constexpr ( quant1 == phasespace::quantity::y ) { range1.x /= dx.y; range1.y /= dx.y; }
            
            phasespace::kernel::dep_pha2<quant0,quant1> <<< grid, block >>> (
                d_data, range0, size0, range1, size1, norm, 
                particles -> view()
            );
        }
    });
    });
}


/**
 * @brief Save 1D phasespace
 * 
 * @param q         Phasespace quantity
 * @param range     Phasespace range
 * @param size      Phasespace grid size
 */
void species::save_phasespace( phasespace::quantity quant, float2 const range, 
    int const size ) const
{
    // Get metadata for quantity
    std::string qname, qlabel, qunits;
    phasespace::qinfo( quant, qname, qlabel, qunits );
    
    // Prepare file info
    zdf::grid_axis axis = {
        .name = (char *) qname.c_str(),
        .min = range.x,
        .max = range.y,
        .label = (char *) qlabel.c_str(),
        .units = (char *) qunits.c_str()
    };

    std::string pha_name  = name + "-" + qname;
    std::string pha_label = name + "\\,(" + qlabel+")";

    zdf::grid_info info = {
        .name = (char *) pha_name.c_str(),
        .ndims = 1,
        .label = (char *) pha_label.c_str(),
        .units = (char *) "n_e",
        .axis  = &axis
    };

    info.count[0] = size;

    zdf::iteration iter_info = {
        .name = (char *) "ITERATION",
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_n"
    };

    // Deposit local 1D phasespace
    float * d_data = gpu::device::malloc<float>( size );

    dep_phasespace( d_data, quant, range, size );

    // Add contributions from all parallel nodes to root node
    particles -> parallel.reduce( d_data, size, mpi::sum );

    // Save file (data is on root node)
    if ( particles -> parallel.root() ) {
        float * h_data = gpu::host::malloc<float>( size );
        gpu::device::memcpy_tohost(  h_data, d_data, size );
        zdf::save_grid( h_data, info, iter_info, "PHASESPACE/" + name );
        gpu::host::free( h_data );
    }

    gpu::device::free( d_data );
}

/**
 * @brief Save 2D phasespace
 * 
 * @param quant0    Quantity 0
 * @param range0    Range of values of quantity 0
 * @param size0     Phasespace grid size for quantity 0
 * @param quant1    Quantity 1
 * @param range1    Range of values of quantity 1
 * @param size1     Phasespace grid size for quantity 0
 */
void species::save_phasespace( 
    phasespace::quantity quant0, float2 const range0, int const size0,
    phasespace::quantity quant1, float2 const range1, int const size1 )
    const
{

    if ( quant0 >= quant1 ) {
        std::cerr << "(*error*) for 2D phasespaces, the 2nd quantity must be indexed higher than the first one\n";
        return;
    }

    std::string qname0, qlabel0, qunits0;
    std::string qname1, qlabel1, qunits1;

    phasespace::qinfo( quant0, qname0, qlabel0, qunits0 );
    phasespace::qinfo( quant1, qname1, qlabel1, qunits1 );
    
    // Prepare file info
    zdf::grid_axis axis[2] = {
        zdf::grid_axis {
            .name = (char *) qname0.c_str(),
            .min = range0.x,
            .max = range0.y,
            .label = (char *) qlabel0.c_str(),
            .units = (char *) qunits0.c_str()
        },
        zdf::grid_axis {
            .name = (char *) qname1.c_str(),
            .min = range1.x,
            .max = range1.y,
            .label = (char *) qlabel1.c_str(),
            .units = (char *) qunits1.c_str()
        }
    };

    std::string pha_name  = name + "-" + qname0 + qname1;
    std::string pha_label = name + " \\,(" + qlabel0 + "\\rm{-}" + qlabel1+")";

    zdf::grid_info info = {
        .name = (char *) pha_name.c_str(),
        .ndims = 2,
        .count = { static_cast<unsigned>(size0), static_cast<unsigned>(size1), 0 },
        .label = (char *) pha_label.c_str(),
        .units = (char *) "n_e",
        .axis  = axis
    };

    zdf::iteration iter_info = {
        .name = (char *) "ITERATION",
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_n"
    };

    // Deposit local 2D phasespace
    float * d_data = gpu::device::malloc<float>( size0 * size1 );
    dep_phasespace( d_data, quant0, range0, size0, quant1, range1, size1 );

    // Add contributions from all parallel nodes to root node
    particles -> parallel.reduce( d_data, size0 * size1, mpi::sum );

    // Save file (data is on root node)
    if ( particles -> parallel.root() ) {
        float * h_data = gpu::host::malloc<float>( size0 * size1 );
        gpu::device::memcpy_tohost( h_data, d_data, size0 * size1 );
        zdf::save_grid( h_data, info, iter_info, "PHASESPACE/" + name );
        gpu::host::free( h_data );
    }

    gpu::device::free( d_data );
}

