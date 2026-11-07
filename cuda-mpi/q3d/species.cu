#include "cyl3/tiled_cyl3.cuh"
#include "species.hpp"
#include <type_traits>

using complex_t = species::complex_t;

/**
 * The following values were determined experimentally using a single
 * NVIDIA A100 80GB PCIe board
 * 
 */

/// @brief Optimal block size for push kernel
int constexpr opt_push_block = 256;

/// @brief Optimal block size for move kernel
int constexpr opt_move_block = 256;

/// @brief Optimal minimum number of blocks for move/push kernels
int constexpr opt_min_blocks = 4096;


/**
 * @brief Initialize data structures and inject initial particle distribution
 * 
 * @param nmodes_           Number of cylindrical modes (including fundamental mode)
 * @param box_              Global simulation box size
 * @param global_ntiles     Global number of tiles
 * @param nx                Individual tile grid size
 * @param dt_               Time step
 * @param id_               Species unique identifier
 * @param parallel          Parallel partition
 */
void species::initialize( const int nmodes_, float2 const box_, uint2 const global_ntiles, uint2 const tile_dims,
    double const dt_, int const id_, mpi::cart2d & parallel ) {
    
    // Store number of cylindrical modes
    if ( nmodes_ < 1 || nmodes_ > 2 ) {
        mpi::fatal("Unsupported number of modes, must be 1 or 2");
    }

    // The axial cell needs at least 1 particle per cell off axis
    if ( ppc.y < 2 ) {
        mpi::fatal("The number of particles per cell along r must be >= 2");
    }

    // Avoid aliasing of the highest mode, ppc.z >= 2 m_max + 1
    if ( nmodes_ > 1 && ppc.z < 2u * ( nmodes_ - 1 ) + 1 ) {
        mpi::fatal("The number of particles per cell along θ must be >= 2 m_max + 1 = " +
                   std::to_string( 2 * ( nmodes_ - 1 ) + 1 ) );
    }

    if ( nmodes_ == 1 && ppc.z > 1 ) {
        mpi::fatal("When using only 1 mode, the number of particles per cell along θ should be 1");
    }

    nmodes = nmodes_;

    // Store simulation box size
    box = box_;

    // Store simulation time step
    dt = dt_;

    // Store species id (used by RNG)
    id = id_;

    // Set charge normalization factor
    q_ref = copysign( density->n0 , m_q ) / (ppc.x * ppc.y * ppc.z);
       
    // Get global grid size
    auto  global_dims = tile_dims * global_ntiles;

    // Set cell size
    dx.x = box.x / (global_dims.x);
    dx.y = box.y / (global_dims.y);

    /// @brief Number of tiles in local parallel node
    uint2 local_ntiles = parallel.grid_size( global_ntiles );

    /// @brief Local parallel node grid size
    auto  local_dims = tile_dims * local_ntiles;

    // Reference maximum number of local particles
    unsigned int max_part = 1.2 * local_dims.x * local_dims.y * ppc.x * ppc.y * ppc.z;

    // Periodic parallel domains override open (default) boundaries
    if ( parallel.periodic.x ) {
        if ( global_bc.x.lower == bc::open ) global_bc.x.lower = bc::periodic;
        if ( global_bc.x.upper == bc::open ) global_bc.x.upper = bc::periodic;
    }

    // Validate boundary conditions against the parallel partition
    validate_bc( global_bc, parallel );

    // Compute local_bc
    // (part, tmp and sort are still null here, they get the periodic flags on construction)
    update_bc( parallel );

    // Create particle data structures 
    int2 const periodic = global_periodic();
    particles = new cyl_part::particles( global_ntiles, tile_dims, max_part, parallel, periodic );
    tmp = new cyl_part::particles( global_ntiles, tile_dims, max_part, parallel, periodic );
    sort = new cyl_part::particle_sort( local_ntiles, max_part, parallel, periodic );
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
 * (1st order) interpolation (purely real)
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
template< typename T >
__device__ void interpolate_fld( 
    cyl3<T> const * const __restrict__ E, 
    cyl3<T> const * const __restrict__ B, 
    const int jstride,
    const int2 ix, const float2 x, cyl3<T> & e, cyl3<T> & b)
{
    const int i = ix.x;
    const int j = ix.y;

    const auto z = x.x;
    const auto r = x.y;

    const auto s0z = 0.5f - z;
    const auto s1z = 0.5f + z;

    const auto s0r = 0.5f - r;
    const auto s1r = 0.5f + r;

    const int hz = z < 0;
    const int hr = r < 0;

    const int ih = i - hz;
    const int jh = j - hr;

    const auto s0zh = (1-hz) - z;
    const auto s1zh = (  hz) + z;

    const auto s0rh = (1-hr) - r;
    const auto s1rh = (  hr) + r;

    // Interpolate E field
    e.z = ( E[ih +     j *jstride].z * s0zh + E[ih+1 +     j *jstride].z * s1zh ) * s0r +
          ( E[ih + (j +1)*jstride].z * s0zh + E[ih+1 + (j +1)*jstride].z * s1zh ) * s1r;

    e.r = ( E[i  +     jh*jstride].r * s0z  + E[i+1  +     jh*jstride].r * s1z ) * s0rh +
          ( E[i  + (jh+1)*jstride].r * s0z  + E[i+1  + (jh+1)*jstride].r * s1z ) * s1rh;

    e.th = ( E[i  +     j *jstride].th * s0z  + E[i+1  +     j *jstride].th * s1z ) * s0r +
          ( E[i  + (j +1)*jstride].th * s0z  + E[i+1  + (j +1)*jstride].th * s1z ) * s1r;

    // Interpolate B fieldj
    b.z = ( B[i  +     jh*jstride].z * s0z  + B[i+1  +     jh*jstride].z * s1z ) * s0rh +
          ( B[i  + (jh+1)*jstride].z * s0z  + B[i+1  + (jh+1)*jstride].z * s1z ) * s1rh;

    b.r = ( B[ih +      j*jstride].r * s0zh + B[ih+1 +      j*jstride].r * s1zh ) * s0r +
          ( B[ih + (j +1)*jstride].r * s0zh + B[ih+1 + (j +1)*jstride].r * s1zh ) * s1r;

    b.th = ( B[ih +     jh*jstride].th * s0zh + B[ih+1 +     jh*jstride].th * s1zh ) * s0rh +
          ( B[ih + (jh+1)*jstride].th * s0zh + B[ih+1 + (jh+1)*jstride].th * s1zh ) * s1rh;

}

/**
 * @brief Advance momentum using a relativistic Boris pusher.
 * 
 * @note
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
 * Uses CUDA intrinsic fma() functions
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
        const float gamma = std::sqrt( 1.0f + utsq );
        
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
 * @note This is similar to the dudt_boris method above, but the rotation is done using
 * using an exact Euler-Rodriguez method.
 * 
 * @param tem 
 * @param e 
 * @param b 
 * @param u 
 * @return float3 
 */
__device__ 
float3 dudt_boris_euler( const float alpha, float3 e, float3 b, float3 u, double & energy )
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
        const float gamma = std::sqrt( 1.0f + utsq );
        
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
        float const s = -(( bnorm > 0 ) ? std::sin( bnorm / 2 ) / bnorm : 0.5f );

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
 * @brief Deposit (charge conserving) current for 1 segment inside a cell (mode m = 0)
 * 
 * @param ix        Particle cell
 * @param x0        Initial particle position (z,r)
 * @param x1        Final particle position (z,r)
 * @param t0        Initial particle transverse position (x,y)
 * @param t1        Final particle  transverse position (x,y)
 * @param q         Particle charge
 * @param J         current(J) grid (should be in shared memory)
 * @param stride    current(J) grid stride
 */
__device__ 
inline void dep_current_seg_0(
    const int2 ix, const float2 x0, const float2 x1, 
    const float2 t0, const float2 t1,
    float q,
    cyl3<float> * __restrict__ J, const int stride )
{
    const auto z0 = x0.x;
    const auto z1 = x1.x;
    const auto r0 = x0.y;
    const auto r1 = x1.y;
    
    const auto S0z0 = 0.5f - z0;
    const auto S0z1 = 0.5f + z0;

    const auto S1z0 = 0.5f - z1;
    const auto S1z1 = 0.5f + z1;

    const auto S0r0 = 0.5f - r0;
    const auto S0r1 = 0.5f + r0;

    const auto S1r0 = 0.5f - r1;
    const auto S1r1 = 0.5f + r1;

    const auto wl1 = q * (z1 - z0);
    const auto wl2 = q * (r1 - r0);
    
    const auto wp10 = 0.5f*(S0r0 + S1r0);
    const auto wp11 = 0.5f*(S0r1 + S1r1);
    
    const auto wp20 = 0.5f*(S0z0 + S1z0);
    const auto wp21 = 0.5f*(S0z1 + S1z1);

    const auto xif = t0.x + t1.x;
    const auto yif = t0.y + t1.y;
    const auto Δx  = t1.x - t0.x;
    const auto Δy  = t1.y - t0.y;

    const auto jth = q * ( fma( - Δx , yif , Δy * xif ) ) / std::sqrt( fma( xif, xif, yif*yif) );

    int i = ix.x;
    int j = ix.y;

    gpu::block::atomic_fetch_add( &J[ stride * j     + i     ].z, wl1 * wp10 );
    gpu::block::atomic_fetch_add( &J[ stride * (j+1) + i     ].z, wl1 * wp11 );

    gpu::block::atomic_fetch_add( &J[ stride * j     + i     ].r, wl2 * wp20 );
    gpu::block::atomic_fetch_add( &J[ stride * j     + (i+1) ].r, wl2 * wp21 );

    gpu::block::atomic_fetch_add( &J[ stride * j     + i     ].th, jth * ( S0z0 * S0r0 + S1z0 * S1r0 + (S0z0 * S1r0 - S1z0 * S0r0)/2.0f ) );
    gpu::block::atomic_fetch_add( &J[ stride * j     + (i+1) ].th, jth * ( S0z1 * S0r0 + S1z1 * S1r0 + (S0z1 * S1r0 - S1z1 * S0r0)/2.0f ) );
    gpu::block::atomic_fetch_add( &J[ stride * (j+1) + i     ].th, jth * ( S0z0 * S0r1 + S1z0 * S1r1 + (S0z0 * S1r1 - S1z0 * S0r1)/2.0f ) );
    gpu::block::atomic_fetch_add( &J[ stride * (j+1) + (i+1) ].th, jth * ( S0z1 * S0r1 + S1z1 * S1r1 + (S0z1 * S1r1 - S1z1 * S0r1)/2.0f ) );
}

/**
 * @brief Deposit (charge conserving) current for 1 segment inside a cell
 * 
 * @tparam m        Azymuthal mode (> 0)
 * @param ix        Initial position (cell index)
 * @param x0        Initial position (z,r)
 * @param x1        Final position (z,r)
 * @param t0        Initial angular position (cos,sin)
 * @param tm        Mid-point angular position (cos,sin)
 * @param t1        Final angular position (cos,sin)
 * @param q         Charge
 * @param vt        Angular velocity (not momentum)
 * @param J         Current buffer
 * @param stride    j stride for current buffer
 */
template< int m >
__device__ 
inline void dep_current_seg(
    const int ir0, const int2 ix, const float2 x0, const float2 x1,
    const float2 t0, const float2 t1, 
    const float q,
    cyl3<complex_t> * __restrict__ J, const int stride )
{
    static_assert( m > 0, "only modes m > 0 are supported");

    // Initial and final grid positions
    // We rename the variables for clarity
    int i = ix.x;
    int j = ix.y;
    const auto z0 = x0.x;
    const auto z1 = x1.x;
    const auto r0 = x0.y;
    const auto r1 = x1.y;

    //const auto cr0 = (ir0 + j) + r0;
    //const auto cr1 = (ir0 + j) + r1;

    const auto rcr0 = 1.0f / sqrt( fma( t0.x, t0.x, t0.y*t0.y ) );
    const auto rcr1 = 1.0f / sqrt( fma( t1.x, t1.x, t1.y*t1.y ) );

    /// @brief Initial θ
    const auto th0 = make_float2( t0.x * rcr0, t0.y * rcr0 );
    /// @brief Final θ
    const auto th1 = make_float2( t1.x * rcr1, t1.y * rcr1 );

    const auto xif = t0.x + t1.x;
    const auto yif = t0.y + t1.y;

    const auto rrm2 = 1.0f / sqrt( fma( xif, xif, yif*yif ) );
    const auto thm = float2{ xif * rrm2, yif * rrm2 };

    // Complex coefficients for initial, mid and final angular positions
    
/*
    auto cm = expimt<m>( thm );
    auto c0 = expimt<m>( th0 ) - cm;
    auto c1 = expimt<m>( th1 ) - cm;
*/

    static_assert( m == 1, "only mode m = 1 is currently supported" );
    auto cm = complex_t{ thm.x, -thm.y };
    auto c0 = complex_t{ th0.x, -th0.y } - cm;
    auto c1 = complex_t{ th1.x, -th1.y } - cm;


    const auto S0z0 = 0.5f - z0;
    const auto S0z1 = 0.5f + z0;

    const auto S1z0 = 0.5f - z1;
    const auto S1z1 = 0.5f + z1;

    const auto S0r0 = 0.5f - r0;
    const auto S0r1 = 0.5f + r0;

    const auto S1r0 = 0.5f - r1;
    const auto S1r1 = 0.5f + r1;

    const auto wl1 = (z1 - z0) * cm;
    const auto wl2 = (r1 - r0) * cm;
    
    const auto wp10 = 0.5f*(S0r0 + S1r0);
    const auto wp11 = 0.5f*(S0r1 + S1r1);
    
    const auto wp20 = 0.5f*(S0z0 + S1z0);
    const auto wp21 = 0.5f*(S0z1 + S1z1);

    gpu::block::atomic_add( &J[ stride * j     + i     ].z, q * wl1 * wp10 );
    gpu::block::atomic_add( &J[ stride * (j+1) + i     ].z, q * wl1 * wp11 );

    gpu::block::atomic_add( &J[ stride * j     + i     ].r, q * wl2 * wp20 );
    gpu::block::atomic_add( &J[ stride * j     + (i+1) ].r, q * wl2 * wp21 );

    // Shape factors are S{time}{coord}{node}: e.g. S1z0 is the weight of node 0
    // along z at the final position (time 1)
    gpu::block::atomic_add( &J[ stride * j     + i     ].th, q * ( S1z0 * S1r0 * c1 - S0z0 * S0r0 * c0 ) );
    gpu::block::atomic_add( &J[ stride * j     + (i+1) ].th, q * ( S1z1 * S1r0 * c1 - S0z1 * S0r0 * c0 ) );
    gpu::block::atomic_add( &J[ stride * (j+1) + i     ].th, q * ( S1z0 * S1r1 * c1 - S0z0 * S0r1 * c0 ) );
    gpu::block::atomic_add( &J[ stride * (j+1) + (i+1) ].th, q * ( S1z1 * S1r1 * c1 - S0z1 * S0r1 * c0 ) );
}


/**
 * @brief Split particle trajectory into segments fitting in a single cell
 * 
 * The transverse motion is a straight line in (x,y), so r(ε) = |t0 + ε Δt| is
 * not linear in ε. The split points are placed on the actual (curved) path:
 * the r-split solves |t0 + ε Δt| = R for the cell edge R, and the radial
 * position at a z-split is r(εz).
 * 
 * @note If the particle does not move radially (Δr == 0), the radial position
 *       at a z-split is set to x0.y exactly. Otherwise roundoff in r(εz) gives
 *       the two segments small, opposite radial displacements, depositing a
 *       spurious Jr dipole on neighbouring z columns (charge is still
 *       conserved, but the spurious currents drive spurious fields).
 * 
 * @note The curved path may dip outside the cell between the endpoints
 *       without a crossing being detected (r(ε) is convex); the radial
 *       position at a z-split can then lie slightly outside [-0.5, 0.5].
 *       Charge conservation is not affected, since consecutive segments share
 *       their end points.
 * 
 * @param ix        Initial cell index (iz, ir)
 * @param x0        Initial position inside cell (z, r)
 * @param delta     Particle motion (Δz, Δr)
 * @param deltai    [out] Cell motion
 * @param t0        Initial transverse position (x, y)
 * @param tdelta    Transverse particle motion (Δx, Δy)
 * @param v0_ix     [out] 1st segment cell index (iz, ir)
 * @param v0_x0     [out] 1st segment initial position (z0, r0)
 * @param v0_x1     [out] 1st segment final position (z1, r1)
 * @param v0_t0     [out] 1st segment initial transverse position (x0, y0)
 * @param v0_t1     [out] 1st segment final transverse position (x1, y1)
 * @param v1_ix     [out] 2nd segment cell index (iz, ir) 
 * @param v1_x0     [out] 2nd segment initial position (z0, r0) 
 * @param v1_x1     [out] 2nd segment final position (z1, r1) 
 * @param v1_t0     [out] 2nd segment initial transverse position (x0, y0)
 * @param v1_t1     [out] 2nd segment final transverse position (x1, y1) 
 * @param v2_ix     [out] 3rd segment cell index (iz, ir) 
 * @param v2_x0     [out] 3rd segment initial position (z0, r0) 
 * @param v2_x1     [out] 3rd segment final position (z1, r1) 
 * @param v2_t0     [out] 3rd segment initial transverse position (x0, y0)
 * @param v2_t1     [out] 3rd segment final transverse position (x1, y1) 
 * @param cross     [out] Cell edge crossing
 */
__device__ 
inline void split2d_cyl( 
    const int ir0, const int2 ix, const float2 x0, const float2 delta, int2 & deltai,
    const float2 t0, const float2 tdelta,
    int2 & v0_ix, float2 & v0_x0, float2 & v0_x1, float2 & v0_t0, float2 & v0_t1,
    int2 & v1_ix, float2 & v1_x0, float2 & v1_x1, float2 & v1_t0, float2 & v1_t1,
    int2 & v2_ix, float2 & v2_x0, float2 & v2_x1, float2 & v2_t0, float2 & v2_t1,
    int2 & cross
) {
    /// @brief final (z,r) position, indexed to original cell
    float2 x1 = x0 + delta;

    /// @brief final (x,y) transverse position
    float2 t1 = t0 + tdelta;

    // Get cell (iz,ir) motion
    deltai = make_int2(
        ((x1.x >= 0.5f) - (x1.x < -0.5f)),
        ((x1.y >= 0.5f) - (x1.y < -0.5f))
    );

    // Get cell crossings
    cross = make_int2( deltai.x != 0, deltai.y != 0 );

    /// @brief cell position of z-split, if any
    float zs = 0.5f * deltai.x;
    /// @brief cell position of r-split, if any
    float rs = 0.5f * deltai.y;

    /// @brief z split fraction
    float εz;
    /// @brief r split fraction
    float εr;


#if 0
    // Original code - keep for reference
    #warning Using original split2d_cyl

    // z-split
    float xz, yz, rz;
    if ( cross.x ) {
        εz = (zs - x0.x) / delta.x;

        // z-split positions
        xz = t0.x + εz * tdelta.x;
        yz = t0.y + εz * tdelta.y;

        rz = ( delta.y == 0 ) ? x0.y : sqrt( fma( xz, xz, yz * yz ) ) - (ir0 + ix.y);
    }

    // r-split
    float xr, yr, zr;
    if ( cross.y ) {
        εr = (rs - x0.y) / delta.y;
        // r-split positions
        xr = t0.x + εr * tdelta.x;
        yr = t0.y + εr * tdelta.y;
        zr = x0.x + εr * delta.x;
    }

#else

    /// @brief |Δt|², the transverse path satisfies |t0 + ε Δt|² = |t0|² + 2 b ε + a ε²
    const float a = fma( tdelta.x, tdelta.x, tdelta.y * tdelta.y );
    /// @brief t0 · Δt
    const float b = fma( t0.x, tdelta.x, t0.y * tdelta.y );
    /// @brief initial radial position (cell based, as in the move kernels)
    const float r0 = ( ir0 + ix.y ) + x0.y;

    // z-split
    float xz, yz, rz;
    if ( cross.x ) {
        εz = (zs - x0.x) / delta.x;

        // z-split positions
        xz = fma( εz, tdelta.x, t0.x );
        yz = fma( εz, tdelta.y, t0.y );

        // Radial position at the z-split, relative to the initial cell. Uses
        // r(ε) - r0 = ε ( 2 b + ε a ) / ( r(ε) + r0 ), which avoids the
        // cancellation in sqrt( xz² + yz² ) - ( ir0 + ix.y ) at large radii.
        // - delta.y == 0: no radial motion, keep rz = x0.y exactly so that no
        //   (spurious) Jr is deposited
        // - den == 0: particle exactly on axis and εz == 0, avoids 0/0
        const float den = sqrt( fma( xz, xz, yz * yz ) ) + r0;
        rz = ( delta.y != 0 && den > 0 ) ? x0.y + εz * fma( εz, a, 2 * b ) / den : x0.y;
    }

    // r-split
    float xr, yr, zr;
    if ( cross.y ) {
        // Crossing of the curved path with the cell edge R = ir0 + ix.y + rs,
        // i.e. the root of a ε² + 2 b ε + c = 0 with c = r0² - R², written as
        // a product to avoid cancellation
        const float c = ( x0.y - rs ) * ( 2 * (ir0 + ix.y) + x0.y + rs );

        // Numerically stable roots q/a and c/q. Roundoff can make the
        // discriminant slightly negative near tangency, so it is clamped to 0
        const float q = - ( b + copysignf( sqrtf( fmaxf( fma( b, b, - a * c ), 0.f ) ), b ) );
        const float e1 = q / a;
        const float e2 = c / q;

        // First crossing: smallest root >= 0 (outward motion has one root of
        // each sign, inward motion two positive roots)
        const float lo = fminf( e1, e2 );
        const float hi = fmaxf( e1, e2 );

        // The crossing itself was decided from delta.y, so clamp to [0,1];
        // fmaxf / fminf also map a NaN (q == 0) to a valid value
        εr = fminf( fmaxf( ( lo >= 0 ) ? lo : hi, 0.f ), 1.f );

        // r-split positions
        xr = fma( εr, tdelta.x, t0.x );
        yr = fma( εr, tdelta.y, t0.y );
        zr = fma( εr, delta.x, x0.x );
    }

#endif

    // Set 1st segment initial positions
    // This will be the same for any particle
    v0_ix = ix; v0_x0 = x0; v0_t0 = t0;

    // assume no splits, fill in other options later
    v0_x1 = x1; v0_t1 = t1;

    // z-cross only
    if ( cross.x && ! cross.y ) {
        v0_x1 = make_float2( zs, rz );
        v0_t1 = make_float2( xz, yz );

        v1_ix = make_int2( ix.x + deltai.x, ix.y );
        v1_x0 = make_float2( -zs, rz );
        v1_x1 = make_float2( x1.x - deltai.x, x1.y );

        v1_t0 = v0_t1;
        v1_t1 = t1;
    }

    // r-cross only
    if ( ! cross.x && cross.y ) {
        v0_x1 = make_float2( zr, rs );
        v0_t1 = make_float2( xr, yr );

        v1_ix = make_int2( ix.x, ix.y + deltai.y );
        v1_x0 = make_float2( zr, -rs );
        v1_x1 = make_float2( x1.x, x1.y - deltai.y );

        v1_t0 = v0_t1;
        v1_t1 = t1;
    }

    // crossing on 2 directions
    if ( cross.x && cross.y ) {
        if ( εz < εr ) {
            // z-cross first
            v0_x1 = make_float2( zs, rz );
            v0_t1 = make_float2( xz, yz );

            v1_ix = make_int2( ix.x + deltai.x, ix.y );
            v1_x0 = make_float2( -zs, rz );
            v1_x1 = make_float2( zr - deltai.x, rs );
            v1_t0 = make_float2( xz, yz );
            v1_t1 = make_float2( xr, yr );

            v2_x0 = make_float2( zr - deltai.x, -rs );
            v2_t0 = make_float2( xr, yr );
        } else {
            // r-cross first
            v0_x1 = make_float2( zr, rs );
            v0_t1 = make_float2( xr, yr );

            v1_ix = make_int2( ix.x, ix.y + deltai.y );
            v1_x0 = make_float2( zr, -rs );
            v1_x1 = make_float2( zs, rz - deltai.y );
            v1_t0 = make_float2( xr, yr );
            v1_t1 = make_float2( xz, yz );

            v2_x0 = make_float2( -zs, rz - deltai.y );
            v2_t0 = make_float2( xz, yz );

        }

        v2_ix = make_int2( ix.x + deltai.x, ix.y + deltai.y );
        v2_x1 = make_float2( x1.x  - deltai.x, x1.y - deltai.y );
        v2_t1 = t1;
    }
}

__global__
/**
 * @brief Kernel for moving particles without depositing current
 * 
 * @param part          Particle data
 * @param dt_dx         Ratio between time step and cell size (x/y)
 * @param shift         Additional cell shift for particles
 * @param d_nmove       (out) Number of particles pushed (for performance metrics)
 */
void move(
    cyl_part::particles_view const part,
    float2 const dt_dx, int2 const shift,
    unsigned long long * const __restrict__ d_nmove
) {

    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const uint2 ntiles  = part.local_ntiles;

    // Move particles and deposit current
    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    const int part_offset    = part.tile_offset[ tid ];
    const int np             = part.tile_np[ tid ];
    int2 * __restrict__ ix   = &part.ix[ part_offset ];
    float2 * __restrict__ x    = &part.x[ part_offset ];
    float3 * __restrict__ u    = &part.u[ part_offset ];
    auto * __restrict__ th   = &part.th[ part_offset ];
    
    auto const dt_dz = dt_dx.x;
    auto const dt_dr = dt_dx.y;

    const int ir0 = (part.local_tile_start.y + tile_idx.y) * part.tile_dims.y;

    for( int i = gpu::block::thread_rank(); i < np; i+= gpu::block::num_threads() ) {
        auto pu = u[i];
        auto x0 = x[i];
        auto ix0 = ix[i];

        auto cos_th = th[i].x;
        auto sin_th = th[i].y; 

        // Get 1 / Lorentz gamma
        float rg = rgamma( pu );

        // Cartesian motion
        auto Δx = dt_dr * rg * pu.x;
        auto Δy = dt_dr * rg * pu.y;

        /// @brief initial radial position
        auto ri = (ir0 + ix0.y) + x0.y;

        // New cartesian positions
        auto xf = fma( ri, cos_th, Δx );
        auto yf = fma( ri, sin_th, Δy );

        /// @brief xi + xf
        auto xif = fma( ri, cos_th, xf );
        /// @brief yi + yf
        auto yif = fma( ri, sin_th, yf );

        // Final positions

        /// @brief New radial position
        auto rf = std::sqrt( fma( xf, xf, yf*yf ) );
        /// @brief radial motion
        auto Δr = fma( Δx , xif , Δy * yif ) / (rf + ri);
        // auto Δr = rf - ri;
    
    
        // Advance grid (z,r) position
        auto Δz = dt_dz * rg * pu.z;
        float2 delta = make_float2( Δz, Δr );

        auto x1 = x0 + delta;
        // Check for cell crossings
        auto deltai = make_int2(
            ((x1.x >= 0.5f) - (x1.x < -0.5f)),
            ((x1.y >= 0.5f) - (x1.y < -0.5f))
        );

        // Modify cell position and store
        x[i] = make_float2( 
            x1.x - deltai.x,
            x1.y - deltai.y
        );

        // Modify cell and store
        ix[i] = make_int2(
            ix0.x + deltai.x + shift.x,
            ix0.y + deltai.y + shift.y
        );

        /// @brief store new angular position
        th[i] = float2{ xf/rf, yf/rf };

    }

    if ( gpu::block::thread_rank() == 0 ) { 
        // Update total particle pushes counter (for performance metrics)
        unsigned long long np64 = np;
        gpu::device::atomic_fetch_add( (unsigned long long *) d_nmove, np64 );
    }
}

/**
 * @brief Kernel for moving particles and depositing current
 * 
 * @note This version allows the use of multiple blocks per tile. It must be
 *       launched with grid( ntiles.x, ntiles.y, blocks_per_tile )
 *       
 * 
 * @param part              Particle data
 * @param d_current         Electric current buffer
 * @param current_offset    Offset to cell 0,0 in electric current tile
 * @param nx                Current grid tile size (internal)
 * @param ext_nx            Current grid tile size (external, includes guard cells)
 * @param dt_dx             Ratio between time step and cell size (x/y)
 * @param shift             Cell shift
 * @param d_nmove           (out) Number of particles pushed (for performance metrics)
 */
__global__ __launch_bounds__(opt_move_block) 
void move_deposit_0(
    cyl_part::particles_view part,
    grid::tiled_cyl3_view<float> current_m0, 
    float2 const dt_dx, const int2 shift, 
    unsigned long long * const __restrict__ d_nmove
) {
    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int tile_id   = tile_idx.y * part.local_ntiles.x + tile_idx.x;
    const int tile_np   = part.tile_np[ tile_id ];

    // Select subset of particles to move (if gridDim.z > 1)
    // We are using multiple blocks per tile (blockIdx.z)
    const int set_size = roundup<gpu::warp_size>(( tile_np + gridDim.z - 1 ) / gridDim.z);
    const int begin = blockIdx.z * set_size;

    // If the block does not have any particles to process return
    if ( begin >= tile_np ) return;
    const int end   = ( begin + set_size < tile_np ) ? begin + set_size : tile_np;

    /// @brief [shared] Local copy of current density
    auto * _current_buffer_m0 = gpu::block::shared_mem< cyl3<float> >();

    // Zero local current buffer
    for( auto i = gpu::block::thread_rank(); i < current_m0.tile_vol; i+= gpu::block::num_threads() ) 
        _current_buffer_m0[i] = cyl3<float>{};

    gpu::block::sync();

    // Move particles and deposit current
    cyl3<float> * J0 = & _current_buffer_m0[ current_m0.inner_offset ];
    const int jstride = current_m0.tile_ystride();

    const int part_offset  = part.tile_offset[ tile_id ];
    auto * __restrict__ ix = &part.ix[ part_offset ];
    auto * __restrict__ x  = &part.x[ part_offset ];
    auto * __restrict__ u  = &part.u[ part_offset ];
    auto * __restrict__ th = &part.th[ part_offset ];
    auto * __restrict__ q  = &part.q[ part_offset ];
    
    auto const dt_dz = dt_dx.x;
    auto const dt_dr = dt_dx.y;

    const int ir0 = (part.local_tile_start.y + tile_idx.y) * part.tile_dims.y;

    // Move particles and deposit current
    for( auto i = begin + gpu::block::thread_rank(); i < end; i+= gpu::block::num_threads() ) {
        auto pu  = u[i];
        auto x0  = x[i];
        auto ix0 = ix[i];
        auto thi = th[i];
        auto pq  = q[i];

        // Get 1 / Lorentz gamma
        float const rg = rgamma( pu );

        // Cartesian motion
        auto Δx = dt_dr * rg * pu.x;
        auto Δy = dt_dr * rg * pu.y;

        /// @brief initial radial position
        auto ri = (ir0 + ix0.y) + x0.y;
        auto xi = ri * thi.x;
        auto yi = ri * thi.y;

        // New cartesian positions
        auto xf = fma( ri, thi.x, Δx );
        auto yf = fma( ri, thi.y, Δy );

        /// @brief xi + xf
        auto xif = fma( ri, thi.x, xf );
        /// @brief yi + yf
        auto yif = fma( ri, thi.y, yf );

        // Final positions

        /// @brief New radial position
        auto rf = sqrt( fma( xf, xf, yf*yf ) );
        /// @brief radial motion
        auto Δr = fma( Δx , xif , Δy * yif ) / (rf + ri);
        // auto Δr = rf - ri;

        // Advance grid (z,r) position
        auto Δz = dt_dz * rg * pu.z;
        float2 delta = make_float2( Δz, Δr );

        // Check for cell crossings and split trajectory
        int2 deltai, cross;

        int2 v0_ix; float2 v0_x0, v0_x1, v0_t0, v0_t1; 
        int2 v1_ix; float2 v1_x0, v1_x1, v1_t0, v1_t1; 
        int2 v2_ix; float2 v2_x0, v2_x1, v2_t0, v2_t1; 

        split2d_cyl( 
            ir0, ix0, x0, delta, deltai,
            make_float2( xi, yi ), make_float2( Δx, Δy ),
            v0_ix, v0_x0, v0_x1, v0_t0, v0_t1,
            v1_ix, v1_x0, v1_x1, v1_t0, v1_t1,
            v2_ix, v2_x0, v2_x1, v2_t0, v2_t1,
            cross
        );
        
        // Deposit current (mode 0)
                                  dep_current_seg_0( v0_ix, v0_x0, v0_x1, v0_t0, v0_t1, pq, J0, jstride );
        if ( cross.x || cross.y ) dep_current_seg_0( v1_ix, v1_x0, v1_x1, v1_t0, v1_t1, pq, J0, jstride );
        if ( cross.x && cross.y ) dep_current_seg_0( v2_ix, v2_x0, v2_x1, v2_t0, v2_t1, pq, J0, jstride );

        // Modify cell position and store
        x[i] = make_float2( 
            (x0.x + Δz ) - deltai.x,
            (x0.y + Δr ) - deltai.y
        );

        // Modify cell and store
        ix[i] = make_int2(
            ix0.x + deltai.x + shift.x,
            ix0.y + deltai.y + shift.y
        );

        /// @brief store new angular position
        th[i] = float2{ xf/rf, yf/rf };

    }

    gpu::block::sync();

    // Add current to global buffers
    float * dev_J0 = reinterpret_cast< float * > ( current_m0.tile_buffer(tile_id));
    float * shm_J0 = reinterpret_cast< float * > ( _current_buffer_m0 );

    if ( gridDim.z > 1 ) {
        // When using multiple blocks per tile we must add the current
        // using atomic ops
        for( auto i =  gpu::block::thread_rank(); i < 3 * current_m0.tile_vol; i+= gpu::block::num_threads() ) {
            gpu::device::atomic_fetch_add( & dev_J0[ i ], shm_J0[ i ] );
        }
    } else {
        for( auto i =  gpu::block::thread_rank(); i < 3 * current_m0.tile_vol; i+= gpu::block::num_threads() ) {
            dev_J0[ i ] += shm_J0[ i ]; 
        }
    }

    if ( gpu::block::thread_rank() == 0 && blockIdx.z == 0 ) {
        // Update total particle pushes counter (for performance metrics)
        unsigned long long np64 = tile_np;
        gpu::device::atomic_fetch_add( d_nmove, np64 );
    }
}


/**
 * @brief Kernel for moving particles and depositing current
 * 
 * @note This version allows the use of multiple blocks per tile. It must be
 *       launched with grid( ntiles.x, ntiles.y, blocks_per_tile )
 *       
 * 
 * @param part              Particle data
 * @param d_current         Electric current buffer
 * @param current_offset    Offset to cell 0,0 in electric current tile
 * @param ext_nx            External tile size (includes guard cells)
 * @param dt_dx             Ratio between time step and cell size (x/y)
 * @param q                 Particle charge
 * @param qnx               Normalization values for in plane current deposition
 * @param d_nmove           (out) Number of particles pushed (for performance metrics)
 */
__global__ __launch_bounds__(opt_move_block) 
void move_deposit_1(
    cyl_part::particles_view const part,
    grid::tiled_cyl3_view<float> current_m0, 
    grid::tiled_cyl3_view< complex_t > current_m1, 
    float2 const dt_dx, const int2 shift,
    unsigned long long * const __restrict__ d_nmove
) {
    using complex_t = complex_t;
    
    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int tile_id   = tile_idx.y * part.local_ntiles.x + tile_idx.x;
    const int tile_np   = part.tile_np[ tile_id ];

   // Select subset of particles to move (if gridDim.z > 1)
    // We are using multiple blocks per tile (blockIdx.z)
    const int set_size = roundup<gpu::warp_size>(( tile_np + gridDim.z - 1 ) / gridDim.z);
    const int begin = blockIdx.z * set_size;

    // If the block does not have any particles to process return
    if ( begin >= tile_np ) return;

    /// @brief [shared] Local copy of current density
    gpu::block::shared_buffer shm;
    auto * _current_buffer_m0 = shm.get< cyl3<float>  >( current_m0.tile_vol );
    auto * _current_buffer_m1 = shm.get< cyl3< complex_t > >( current_m1.tile_vol );

    // Zero local current buffers
    for( auto i = gpu::block::thread_rank(); i < current_m0.tile_vol; i+= gpu::block::num_threads() ) {
        _current_buffer_m0[i] = cyl3<float>{};
        _current_buffer_m1[i] = cyl3<complex_t>{};
    }

    gpu::block::sync();

    // Move particles and deposit current
    const int end   = ( begin + set_size < tile_np ) ? begin + set_size : tile_np;

    cyl3<float> * J0 = & _current_buffer_m0[ current_m0.inner_offset ];
    cyl3<complex_t> * J1 = & _current_buffer_m1[ current_m1.inner_offset ];

    const int jstride = current_m0.tile_ystride();

    const int part_offset  = part.tile_offset[ tile_id ];
    auto * __restrict__ ix = &part.ix[ part_offset ];
    auto * __restrict__ x  = &part.x[ part_offset ];
    auto * __restrict__ u  = &part.u[ part_offset ];
    auto * __restrict__ th = &part.th[ part_offset ];
    auto * __restrict__ q  = &part.q[ part_offset ];

    auto const dt_dz = dt_dx.x;
    auto const dt_dr = dt_dx.y;

    const int ir0 = (part.local_tile_start.y + tile_idx.y) * part.tile_dims.y;

    // Move particles and deposit current
    for( auto i = begin + gpu::block::thread_rank(); i < end; i+= gpu::block::num_threads() ) {

        auto pu  = u[i];
        auto x0  = x[i];
        auto ix0 = ix[i];
        auto thi = th[i];
        auto pq  = q[i];

        // Get 1 / Lorentz gamma
        float const rg = rgamma( pu );

        // Cartesian motion
        auto Δx = dt_dr * rg * pu.x;
        auto Δy = dt_dr * rg * pu.y;

        /// @brief initial radial position
        auto ri = ( ir0 + ix0.y ) + x0.y;
        auto xi = ri * thi.x;
        auto yi = ri * thi.y;

        // New cartesian positions
        auto xf = fma( ri, thi.x, Δx );
        auto yf = fma( ri, thi.y, Δy );

        /// @brief xi + xf
        auto xif = fma( ri, thi.x, xf );
        /// @brief yi + yf
        auto yif = fma( ri, thi.y, yf );

        // Final positions

        /// @brief New radial position
        auto rf = sqrt( fma( xf, xf, yf*yf ) );
        /// @brief radial motion
        auto Δr = fma( Δx , xif , Δy * yif ) / (rf + ri);
        // auto Δr = rf - ri;

        // Advance grid (z,r) position
        auto Δz = dt_dz * rg * pu.z;
        float2 delta = make_float2( Δz, Δr );

        // Check for cell crossings and split trajectory
        int2 deltai, cross;

        int2 v0_ix; float2 v0_x0, v0_x1, v0_t0, v0_t1; 
        int2 v1_ix; float2 v1_x0, v1_x1, v1_t0, v1_t1; 
        int2 v2_ix; float2 v2_x0, v2_x1, v2_t0, v2_t1; 

        split2d_cyl( 
            ir0, ix0, x0, delta, deltai,
            make_float2( xi, yi ), make_float2( Δx, Δy ),
            v0_ix, v0_x0, v0_x1, v0_t0, v0_t1,
            v1_ix, v1_x0, v1_x1, v1_t0, v1_t1,
            v2_ix, v2_x0, v2_x1, v2_t0, v2_t1,
            cross
        );
        
        // Deposit current (mode 0)
                                  dep_current_seg_0( v0_ix, v0_x0, v0_x1, v0_t0, v0_t1, pq, J0, jstride );
        if ( cross.x || cross.y ) dep_current_seg_0( v1_ix, v1_x0, v1_x1, v1_t0, v1_t1, pq, J0, jstride );
        if ( cross.x && cross.y ) dep_current_seg_0( v2_ix, v2_x0, v2_x1, v2_t0, v2_t1, pq, J0, jstride );

        // Deposit current (mode 1)
                                  dep_current_seg<1>( ir0, v0_ix, v0_x0, v0_x1, v0_t0, v0_t1, pq, J1, jstride );
        if ( cross.x || cross.y ) dep_current_seg<1>( ir0, v1_ix, v1_x0, v1_x1, v1_t0, v1_t1, pq, J1, jstride );
        if ( cross.x && cross.y ) dep_current_seg<1>( ir0, v2_ix, v2_x0, v2_x1, v2_t0, v2_t1, pq, J1, jstride );


        // Correct cell position and store
        x[i] = make_float2( 
            (x0.x + Δz ) - deltai.x,
            (x0.y + Δr ) - deltai.y
        );

        // Modify cell and store
        ix[i] = make_int2(
            ix0.x + deltai.x + shift.x,
            ix0.y + deltai.y + shift.y
        );

        /// @brief store new angular position
        th[i] = float2{ xf/rf, yf/rf };
    }

    gpu::block::sync();

    // Add current to global buffers
    float * dev_J0 = reinterpret_cast< float * > ( current_m0.tile_buffer(tile_id));
    float * shm_J0 = reinterpret_cast< float * > ( _current_buffer_m0 );
    float * dev_J1 = reinterpret_cast< float * > ( current_m1.tile_buffer(tile_id));
    float * shm_J1 = reinterpret_cast< float * > ( _current_buffer_m1 );

    if ( gridDim.z > 1 ) {
        // When using multiple blocks per tile we must add the current
        // using atomic ops
        for( auto i =  gpu::block::thread_rank(); i < 3 * current_m0.tile_vol; i+= gpu::block::num_threads() ) {
            gpu::device::atomic_fetch_add( & dev_J0[ i ], shm_J0[ i ] );
        }
        for( auto i =  gpu::block::thread_rank(); i < 3 * 2 * current_m1.tile_vol; i+= gpu::block::num_threads() ) {
            gpu::device::atomic_fetch_add( & dev_J1[ i ], shm_J1[ i ] );
        }
    } else {
        for( auto i =  gpu::block::thread_rank(); i < 3 * current_m0.tile_vol; i+= gpu::block::num_threads() ) {
            dev_J0[ i ] += shm_J0[ i ]; 
        }
        for( auto i =  gpu::block::thread_rank(); i < 3 * 2 * current_m1.tile_vol; i+= gpu::block::num_threads() ) {
            dev_J1[ i ] += shm_J1[ i ]; 
        }
    }

    if ( gpu::block::thread_rank() == 0 && blockIdx.z == 0 ) {
        // Update total particle pushes counter (for performance metrics)
        unsigned long long np64 = tile_np;
        gpu::device::atomic_fetch_add( d_nmove, np64 );
    }
}

/**
 * @brief Kernel for depositing current (mode 1). Particle motion is not stored.
 * 
 * @note This version allows the use of multiple blocks per tile. It must be
 *       launched with grid( ntiles.x, ntiles.y, blocks_per_tile )
 *       
 * 
 * @param part              Particle data
 * @param d_current         Electric current buffer
 * @param current_offset    Offset to cell 0,0 in electric current tile
 * @param ext_nx            External tile size (includes guard cells)
 * @param dt_dx             Ratio between time step and cell size (x/y)
 * @param q                 Particle charge
 * @param qnx               Normalization values for in plane current deposition
 * @param d_nmove           (out) Number of particles pushed (for performance metrics)
 */
__global__ __launch_bounds__(opt_move_block) 
void deposit_1(
    cyl_part::particles_view const part,
    grid::tiled_cyl3_view< complex_t > current_m1,
    float2 const dt_dx
) {
    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int tile_id   = tile_idx.y * part.local_ntiles.x + tile_idx.x;
    const int tile_np   = part.tile_np[ tile_id ];

   // Select subset of particles to move (if gridDim.z > 1)
    // We are using multiple blocks per tile (blockIdx.z)
    const int set_size = roundup<gpu::warp_size>(( tile_np + gridDim.z - 1 ) / gridDim.z);
    const int begin = blockIdx.z * set_size;

    // If the block does not have any particles to process return
    if ( begin >= tile_np ) return;

    /// @brief [shared] Local copy of current density
    auto * _current_buffer_m1 = gpu::block::shared_mem< cyl3<complex_t> >();

    // Zero local current buffers
    gpu::block::zero( _current_buffer_m1, current_m1.tile_vol );

    gpu::block::sync();

    // Move particles and deposit current
    const int end   = ( begin + set_size < tile_np ) ? begin + set_size : tile_np;

    auto * J1 = & _current_buffer_m1[ current_m1.inner_offset ];

    const int jstride = current_m1.tile_ystride();

    const int part_offset  = part.tile_offset[ tile_id ];
    auto * __restrict__ ix = &part.ix[ part_offset ];
    auto * __restrict__ x  = &part.x[ part_offset ];
    auto * __restrict__ u  = &part.u[ part_offset ];
    auto * __restrict__ th = &part.th[ part_offset ];
    auto * __restrict__ q  = &part.q[ part_offset ];

    auto const dt_dz = dt_dx.x;
    auto const dt_dr = dt_dx.y;

    const int ir0 = (part.local_tile_start.y + tile_idx.y) * part.tile_dims.y;

    // Move particles and deposit current
    for( auto i = begin + gpu::block::thread_rank(); i < end; i+= gpu::block::num_threads() ) {
        auto pu  = u[i];
        auto x0  = x[i];
        auto ix0 = ix[i];
        auto thi = th[i];
        auto pq  = q[i];

        // Get 1 / Lorentz gamma
        float const rg = rgamma( pu );

        // Cartesian motion
        auto Δx = dt_dr * rg * pu.x;
        auto Δy = dt_dr * rg * pu.y;

        /// @brief initial radial position
        auto ri = (ir0 + ix0.y) + x0.y;
        auto xi = ri * thi.x;
        auto yi = ri * thi.y;

        // New cartesian positions
        auto xf = fma( ri, thi.x, Δx );
        auto yf = fma( ri, thi.y, Δy );

        /// @brief xi + xf
        auto xif = fma( ri, thi.x, xf );
        /// @brief yi + yf
        auto yif = fma( ri, thi.y, yf );

        // Final positions

        /// @brief New radial position
        auto rf = sqrt( fma( xf, xf, yf*yf ) );
        /// @brief radial motion
        auto Δr = fma( Δx , xif , Δy * yif ) / (rf + ri);
        // auto Δr = rf - ri;

        // Advance grid (z,r) position
        auto Δz = dt_dz * rg * pu.z;
        float2 delta = make_float2( Δz, Δr );

        // Check for cell crossings and split trajectory
        int2 deltai, cross;

        int2 v0_ix; float2 v0_x0, v0_x1, v0_t0, v0_t1; 
        int2 v1_ix; float2 v1_x0, v1_x1, v1_t0, v1_t1; 
        int2 v2_ix; float2 v2_x0, v2_x1, v2_t0, v2_t1; 

        split2d_cyl( 
            ir0, ix0, x0, delta, deltai,
            make_float2( xi, yi ), make_float2( Δx, Δy ),
            v0_ix, v0_x0, v0_x1, v0_t0, v0_t1,
            v1_ix, v1_x0, v1_x1, v1_t0, v1_t1,
            v2_ix, v2_x0, v2_x1, v2_t0, v2_t1,
            cross
        );

        // Deposit current (mode 1)
                                  dep_current_seg<1>( ir0, v0_ix, v0_x0, v0_x1, v0_t0, v0_t1, pq, J1, jstride );
        if ( cross.x || cross.y ) dep_current_seg<1>( ir0, v1_ix, v1_x0, v1_x1, v1_t0, v1_t1, pq, J1, jstride );
        if ( cross.x && cross.y ) dep_current_seg<1>( ir0, v2_ix, v2_x0, v2_x1, v2_t0, v2_t1, pq, J1, jstride );
    }

    gpu::block::sync();


    float * dev_J1 = reinterpret_cast< float * > ( current_m1.tile_buffer(tile_id));
    float * shm_J1 = reinterpret_cast< float * > ( _current_buffer_m1 );

    if ( gridDim.z > 1 ) {
        // When using multiple blocks per tile we must add the current
        // using atomic ops
        for( auto i =  gpu::block::thread_rank(); i < 3 * 2 * current_m1.tile_vol; i+= gpu::block::num_threads() ) {
            gpu::device::atomic_fetch_add( & dev_J1[ i ], shm_J1[ i ] );
        }
    } else {
        for( auto i =  gpu::block::thread_rank(); i < 3 * 2 * current_m1.tile_vol; i+= gpu::block::num_threads() ) {
            dev_J1[ i ] += shm_J1[ i ]; 
        }
    }
}

/**
 * @brief Kernel for accelerating particles
 * 
 * @tparam type         Template variable to choose pusher type. Must be
 *                      species::boris or species::euler
 * @param part          Particle data
 * @param E_buffer      E-field buffer
 * @param B_buffer      B-field buffer
 * @param ext_nx        External E/B tile size (includes guard cells)
 * @param field_offset  Offset to cell 0,0 in E/B field buffers
 * @param alpha         Pusher alpha parameter ( 0.5 * dt / m_q )
 * @param d_energy      (out) Total time-centered particle energy
 */
template < species::pusher type >
__global__ __launch_bounds__(opt_push_block)
void push_0 ( 
    cyl_part::particles_view part,
    grid::tiled_cyl3_view<float> E_grid, grid::tiled_cyl3_view<float> B_grid, 
    float const alpha, double * __restrict__ d_energy )
{
    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int tile_id   = tile_idx.y * part.local_ntiles.x + tile_idx.x;
    const int tile_np   = part.tile_np[ tile_id ];

   // Select subset of particles to move (if gridDim.z > 1)
    // We are using multiple blocks per tile (blockIdx.z)
    const int set_size = roundup<gpu::warp_size>(( tile_np + gridDim.z - 1 ) / gridDim.z);
    const int begin = blockIdx.z * set_size;

    // If the block does not have any particles to process return
    if ( begin >= tile_np ) return;
    const int end   = ( begin + set_size < tile_np ) ? begin + set_size : tile_np;

    // Copy E and B into shared memory (mode 0)
    gpu::block::shared_buffer shm;
    auto * __restrict__ E_local_m0 = shm.get<cyl3<float>>( E_grid.tile_vol );
    auto * __restrict__ B_local_m0 = shm.get<cyl3<float>>( B_grid.tile_vol );

    gpu::block::memcpy( E_local_m0, E_grid.tile_buffer( tile_id ), E_grid.tile_vol );
    gpu::block::memcpy( B_local_m0, B_grid.tile_buffer( tile_id ), B_grid.tile_vol );
    
    auto const * const __restrict__ E_m0 = & E_local_m0[ E_grid.inner_offset ];
    auto const * const __restrict__ B_m0 = & B_local_m0[ B_grid.inner_offset ];

    // Push particles
    const int part_offset  = part.tile_offset[ tile_id ];
    auto * __restrict__ ix = &part.ix[ part_offset ];
    auto * __restrict__ x  = &part.x[ part_offset ];
    auto * __restrict__ u  = &part.u[ part_offset ];
    auto * __restrict__ th = &part.th[ part_offset ];
    auto * __restrict__ q  = &part.q[ part_offset ];

    double energy = 0;
    const int jstride = E_grid.tile_ystride();

    gpu::block::sync();


    // Push particles
    for( auto i = begin + gpu::block::thread_rank(); i < end; i+= gpu::block::num_threads() ) {

        // Interpolate field
        cyl3<float> e, b;
        interpolate_fld( E_m0, B_m0, jstride, ix[i], x[i], e, b );

        // Convert to cartesian components
        auto cos_th = th[i].x;
        auto sin_th = th[i].y;
        
        float3 cart_e = make_float3(
            fma( e.r, cos_th, - e.th * sin_th ),
            fma( e.r, sin_th, + e.th * cos_th ),
            e.z
        );

        float3 cart_b = make_float3(
            fma( b.r, cos_th, - b.th * sin_th ),
            fma( b.r, sin_th, + b.th * cos_th ),
            b.z
        );
        
        // Advance momentum
        float3 pu = u[i];

        /// @brief Particle kinetic energy (γ - 1), time centered
        double ekin = 0;
        if constexpr ( type == species::pusher::boris ) u[i] = dudt_boris( alpha, cart_e, cart_b, pu, ekin );
        if constexpr ( type == species::pusher::euler ) u[i] = dudt_boris_euler( alpha, cart_e, cart_b, pu, ekin );

        // Particles have different weights (charge q ∝ r)
        energy += q[i] * ekin;
    }

    // Add up energy from all threads
    energy = gpu::warp::reduce_add( energy );
    if ( gpu::warp::thread_rank() == 0 ) { 
        gpu::device::atomic_fetch_add( d_energy, energy );
    }
}


/**
 * @brief Advance particle velocities (modes 0 and 1)
 * 
 * @tparam type 
 * @param tile_idx      Tile index
 * @param part          Particle data
 * @param d_E           E-field grid (global)
 * @param d_B           B-field grid (global)
 * @param field_offset  Offset to position [0,0] of field grids
 * @param ext_nx        Field grid size (external)
 * @param alpha         Normalization parameter
 * @param d_energy      Total particle energy (if using OpenMP this must be a reduction variable)
 */
template < species::pusher type >
__global__ __launch_bounds__(opt_push_block)
void  push_1 ( 
    cyl_part::particles_view part,
    grid::tiled_cyl3_view<float> E_grid_m0, grid::tiled_cyl3_view<float> B_grid_m0,
    grid::tiled_cyl3_view<complex_t> E_grid_m1, grid::tiled_cyl3_view<complex_t> B_grid_m1, 
    float const alpha, double * const __restrict__ d_energy )
{
    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int tile_id   = tile_idx.y * part.local_ntiles.x + tile_idx.x;
    const int tile_np   = part.tile_np[ tile_id ];

   // Select subset of particles to move (if gridDim.z > 1)
    // We are using multiple blocks per tile (blockIdx.z)
    const int set_size = roundup<gpu::warp_size>(( tile_np + gridDim.z - 1 ) / gridDim.z);
    const int begin = blockIdx.z * set_size;

    // If the block does not have any particles to process return
    if ( begin >= tile_np ) return;
    const int end   = ( begin + set_size < tile_np ) ? begin + set_size : tile_np;

    /// @brief Shared memory block
    gpu::block::shared_buffer shm;

    // Copy E and B into shared memory (mode 0)
    auto * __restrict__ E_local_m0 = shm.get< cyl3<float> >( E_grid_m0.tile_vol );
    auto * __restrict__ B_local_m0 = shm.get< cyl3<float> >( B_grid_m0.tile_vol );

    gpu::block::memcpy( E_local_m0, E_grid_m0.tile_buffer(tile_id), E_grid_m0.tile_vol );
    gpu::block::memcpy( B_local_m0, B_grid_m0.tile_buffer(tile_id), B_grid_m0.tile_vol );

    auto const * const __restrict__ E_m0 = & E_local_m0[ E_grid_m0.inner_offset ];
    auto const * const __restrict__ B_m0 = & B_local_m0[ B_grid_m0.inner_offset ];

    // Copy E and B into shared memory (mode 1)
    auto * __restrict__ E_local_m1 = shm.get< cyl3<complex_t> >( E_grid_m1.tile_vol );
    auto * __restrict__ B_local_m1 = shm.get< cyl3<complex_t> >( B_grid_m1.tile_vol );

    gpu::block::memcpy( E_local_m1, E_grid_m1.tile_buffer(tile_id), E_grid_m1.tile_vol );
    gpu::block::memcpy( B_local_m1, B_grid_m1.tile_buffer(tile_id), B_grid_m1.tile_vol );

    auto const * const __restrict__ E_m1 = & E_local_m1[ E_grid_m1.inner_offset ];
    auto const * const __restrict__ B_m1 = & B_local_m1[ B_grid_m1.inner_offset ];

    // Push particles
    const int part_offset  = part.tile_offset[ tile_id ];
    auto * __restrict__ ix = &part.ix[ part_offset ];
    auto * __restrict__ x  = &part.x[ part_offset ];
    auto * __restrict__ u  = &part.u[ part_offset ];
    auto * __restrict__ th = &part.th[ part_offset ];
    auto * __restrict__ q  = &part.q[ part_offset ];

    double energy = 0;
    const int jstride = E_grid_m0.tile_ystride();

    gpu::block::sync();

    for( auto i = begin + gpu::block::thread_rank(); i < end; i+= gpu::block::num_threads() ) {

        // Interpolate field
        cyl3<float> e, b;
        interpolate_fld( E_m0, B_m0, jstride, ix[i], x[i], e, b );

        // Interpolate field - mode 1
        cyl3<complex_t> e1, b1;
        interpolate_fld( E_m1, B_m1, jstride, ix[i], x[i], e1, b1 );

        // Get full field
        const auto cos_th = th[i].x;
        const auto sin_th = th[i].y;

        e.z += fma( cos_th, e1.z.real( ), -sin_th * e1.z.imag( ) );
        e.r += fma( cos_th, e1.r.real( ), -sin_th * e1.r.imag( ) );
        e.th += fma( cos_th, e1.th.real( ), -sin_th * e1.th.imag( ) );

        b.z += fma( cos_th, b1.z.real( ), -sin_th * b1.z.imag( ) );
        b.r += fma( cos_th, b1.r.real( ), -sin_th * b1.r.imag( ) );
        b.th += fma( cos_th, b1.th.real( ), -sin_th * b1.th.imag( ) );

        // Convert to cartesian components       
        float3 cart_e = make_float3(
            fma( e.r, cos_th, - e.th * sin_th ),
            fma( e.r, sin_th, + e.th * cos_th ),
            e.z
        );

        float3 cart_b = make_float3(
            fma( b.r, cos_th, - b.th * sin_th ),
            fma( b.r, sin_th, + b.th * cos_th ),
            b.z
        );
        
        // Advance momentum
        float3 pu = u[i];

        /// @brief Particle kinetic energy (γ - 1), time centered
        double ekin = 0;
        if constexpr ( type == species::pusher::boris ) u[i] = dudt_boris( alpha, cart_e, cart_b, pu, ekin );
        if constexpr ( type == species::pusher::euler ) u[i] = dudt_boris_euler( alpha, cart_e, cart_b, pu, ekin );

        // Particles have different weights (charge q ∝ r)
        energy += q[i] * ekin;
    }

    // Add up energy from all threads
    energy = gpu::warp::reduce_add( energy );
    if ( gpu::warp::thread_rank() == 0 ) { 
        gpu::device::atomic_fetch_add( d_energy, energy );
    }
}

__global__
/**
 * @brief Physical boundary conditions for the z direction 
 * 
 * @param part      Particle data
 * @param bc        Boundary condition
 */
void species_bcz(
    cyl_part::particles_view const part,
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
                    // Protects against x[i].x exactly == -0.5
                    x[i].x = fminf( -x[i].x, nextafterf( 0.5f, 0.f ) );
                    // Reverse longitudinal momentum
                    u[i].z = -u[i].z;
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
                    // Reverse longitudinal momentum
                    u[i].z = -u[i].z;
                }
            }
            break;
        default:
            break;
        }
    }
}

__global__
/**
 * @brief Physical boundary conditions for the r direction 
 * 
 * @param part      Particle data
 * @param bc        Boundary condition
 */
void species_bcr(
    cyl_part::particles_view const part,
    species::bc_type const local_bc ) 
{
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
    float2 * __restrict__ th = &part.th[ part_offset ];

    if ( bnd_y == 1 ) {
        // Upper boundary
        switch( local_bc.y.upper ) {
        case( species::bc::reflecting ) :
            for( int i = gpu::block::thread_rank(); i < np; i+= gpu::block::num_threads() ) {
                if( ix[i].y >=  ny ) {
                // Correct radial position
                ix[i].y -= 1;
                // Protects against x[i].y exactly == -0.5
                x[i].y = fminf( -x[i].y, nextafterf( 0.5f, 0.f ) );

                // Correct radial velocity
                const auto costh = th[i].x;
                const auto sinth = th[i].y;
                auto       ur  = u[i].x * costh + u[i].y * sinth;
                const auto uth = u[i].y * costh - u[i].x * sinth;

                ur = -ur;
                u[i].x = ur * costh - uth * sinth;
                u[i].y = ur * sinth + uth * costh;
                }
            }
            break;
        default:
            break;
        }
    }
}

/**
 * @brief Kernel for charge density deposition
 * 
 * @param part              Particle data
 * @param q                 Particle charge
 * @param charge_buffer     Charge buffer
 * @param charge_offset     Offset to cell 0,0 in charge tile
 * @param ext_nx            External tile size (includes guard cells)
 */
__global__
void dep_charge_0(
    cyl_part::particles_view const part,
    grid::tiled_view<float> charge_m0 )
{
    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int tile_id   = tile_idx.y * part.local_ntiles.x + tile_idx.x;
    const int tile_np   = part.tile_np[ tile_id ];

    // Select subset of particles to move (if gridDim.z > 1)
    // We are using multiple blocks per tile (blockIdx.z)
    const int set_size = roundup<gpu::warp_size>(( tile_np + gridDim.z - 1 ) / gridDim.z);
    const int begin = blockIdx.z * set_size;

    // If the block does not have any particles to process return
    if ( begin >= tile_np ) return;
    const int end   = ( begin + set_size < tile_np ) ? begin + set_size : tile_np;

    /// @brief [shared] local copy of charge density
    auto * charge_local = gpu::block::shared_mem<float>();

    // Zero shared memory and sync.
    for( auto i = gpu::block::thread_rank(); i < charge_m0.tile_vol; i += gpu::block::num_threads() ) {
        charge_local[i] = 0;
    }

    float *charge = &charge_local[ charge_m0.inner_offset ];

    gpu::block::sync();

    const int part_off = part.tile_offset[ tile_id ];
    auto const * __restrict__ const ix = &part.ix[ part_off ];
    auto const * __restrict__ const x  = &part.x[ part_off ];
    auto const * __restrict__ const q  = &part.q[ part_off ];
    const int jstride = charge_m0.tile_ystride(); 

    for( auto i = begin + gpu::block::thread_rank(); i < end; i+= gpu::block::num_threads() ) {
        const float s0z = 0.5f - x[i].x;
        const float s1z = 0.5f + x[i].x;
        const float s0r = 0.5f - x[i].y;
        const float s1r = 0.5f + x[i].y;

        const int idx = ix[i].y * jstride + ix[i].x;

        gpu::block::atomic_fetch_add( & charge[ idx               ], s0r * s0z * q[i] );
        gpu::block::atomic_fetch_add( & charge[ idx + 1           ], s0r * s1z * q[i] );
        gpu::block::atomic_fetch_add( & charge[ idx     + jstride ], s1r * s0z * q[i] );
        gpu::block::atomic_fetch_add( & charge[ idx + 1 + jstride ], s1r * s1z * q[i] );
    }

    gpu::block::sync();

    // Add charge to global buffers
    float * dev_rho = reinterpret_cast< float * > ( charge_m0.tile_buffer(tile_id));
    float * shm_rho = reinterpret_cast< float * > ( charge );

    if ( gridDim.z > 1 ) {
        // When using multiple blocks per tile we must add the current
        // using atomic ops
        for( auto i =  gpu::block::thread_rank(); i < charge_m0.tile_vol; i+= gpu::block::num_threads() ) {
            gpu::device::atomic_fetch_add( & dev_rho[ i ], shm_rho[ i ] );
        }
    } else {
        for( auto i =  gpu::block::thread_rank(); i < charge_m0.tile_vol; i+= gpu::block::num_threads() ) {
            dev_rho[ i ] += shm_rho[ i ]; 
        }
    }
}

/**
 * @brief kernel for depositing mode m>0 charge
 * 
 * @param d_charge  Charge density grid (will be zeroed by this kernel)
 * @param offset    Offset to position (0,0) of grid
 * @param ext_nx    External tile size (i.e. including guard cells)
 * @param d_tile    Particle tiles information
 * @param d_ix      Particle buffer (cells)
 * @param d_x       Particle buffer (position)
 * @param q         Species charge per particle
 */
template< int m >
__global__ 
void dep_charge(
    cyl_part::particles_view const part,
    grid::tiled_view<complex_t> charge_m )
{
    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int tile_id   = tile_idx.y * part.local_ntiles.x + tile_idx.x;
    const int tile_np   = part.tile_np[ tile_id ];

    // Select subset of particles to move (if gridDim.z > 1)
    // We are using multiple blocks per tile (blockIdx.z)
    const int set_size = roundup<gpu::warp_size>(( tile_np + gridDim.z - 1 ) / gridDim.z);
    const int begin = blockIdx.z * set_size;

    // If the block does not have any particles to process return
    if ( begin >= tile_np ) return;
    const int end   = ( begin + set_size < tile_np ) ? begin + set_size : tile_np;

    /// @brief [shared] local copy of charge density
    auto * charge_local_m = gpu::block::shared_mem<complex_t>();

    // Zero shared memory and sync.
    for( auto i = gpu::block::thread_rank(); i < charge_m.tile_vol; i += gpu::block::num_threads() ) {
        charge_local_m[i] = 0;
    }

    auto *charge = &charge_local_m[ charge_m.inner_offset ];

    gpu::block::sync();

    const int part_off = part.tile_offset[ tile_id ];
    auto const * __restrict__ const ix = &part.ix[ part_off ];
    auto const * __restrict__ const x  = &part.x[ part_off ];
    auto const * __restrict__ const q  = &part.q[ part_off ];
    auto const * __restrict__ const th = &part.th[ part_off ];
    const int jstride = charge_m.tile_ystride(); 

    for( auto i = begin + gpu::block::thread_rank(); i < end; i+= gpu::block::num_threads() ) {
        const float s0z = 0.5f - x[i].x;
        const float s1z = 0.5f + x[i].x;
        const float s0r = 0.5f - x[i].y;
        const float s1r = 0.5f + x[i].y;

        const int idx = ix[i].y * jstride + ix[i].x;

        static_assert( m == 1, "only mode m = 1 is currently supported" );
        // auto qm = q[i] * expimt<m>( t[i] );
        auto qm = q[i] * complex_t{ th[i].x, -th[i].y };

        gpu::block::atomic_add( & charge[ idx               ], s0r * s0z * qm );
        gpu::block::atomic_add( & charge[ idx + 1           ], s0r * s1z * qm );
        gpu::block::atomic_add( & charge[ idx     + jstride ], s1r * s0z * qm );
        gpu::block::atomic_add( & charge[ idx + 1 + jstride ], s1r * s1z * qm );
    }

    gpu::block::sync();

    // Copy data to global memory
    float * dev_rho = reinterpret_cast< float * > ( charge_m.tile_buffer(tile_id));
    float * shm_rho = reinterpret_cast< float * > ( charge );

    if ( gridDim.z > 1 ) {
        // When using multiple blocks per tile we must add the current
        // using atomic ops
        for( auto i =  gpu::block::thread_rank(); i < 2 * charge_m.tile_vol; i+= gpu::block::num_threads() ) {
            gpu::device::atomic_fetch_add( & dev_rho[ i ], shm_rho[ i ] );
        }
    } else {
        for( auto i =  gpu::block::thread_rank(); i < 2 * charge_m.tile_vol; i+= gpu::block::num_threads() ) {
            dev_rho[ i ] += shm_rho[ i ]; 
        }
    }
}

/**
 * @brief Normalize charge grid for "ring" particles
 * 
 * @tparam T            Type ( real or complex, depending on the mode )
 * @param tile_idx      Tile index
 * @param d_charge      Pointer to charge grid
 * @param offset        Offset to position (0,0) on the grid
 * @param nx            Tile grid size
 * @param ext_nx        External tile grid size
 * @param dr            Radial cell size (in simulation units)
 * @param scale         Scale for normalization
 * @param parity        Parity of the mode across the axis, (-1)^m
 */
template< class T >
__global__ 
void charge_norm(
    grid::tiled_view<T> charge_grid, const float dr, const float scale = 1.0f,
    const float parity = 1.0f
) {
    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int jstride = charge_grid.tile_ystride();

    auto * __restrict__ charge = charge_grid.tile_data( tile_idx );
    auto tile_dims = charge_grid.tile_dims;

    int ir0 = ( charge_grid.local_tile_start.y + tile_idx.y ) * tile_dims.y;
    for( int k = gpu::block::thread_rank(); k < (tile_dims.y+1)*(tile_dims.x+1); k += gpu::block::num_threads() ) {
        int j = k / (tile_dims.x+1);
        int i = k % (tile_dims.x+1);
        auto norm = scale/(std::abs( ir0 + j - 0.5f) * dr);
        charge[ j * jstride +i ] *= norm;
    }

    // Axial boundary
    // Fold values for r < 0 back into simulation domain, row 0 is at
    // r = -Δr/2 and mirrors row 1 with the mode parity
    if ( ir0 == 0 ) {
        for( int i = gpu::block::thread_rank(); i < static_cast<int>(tile_dims.x+1); i+= gpu::block::num_threads() ){
            charge[ i + 1 * jstride ] += parity * charge[ i + 0 * jstride ];
            charge[ i + 0 * jstride ]  = parity * charge[ i + 1 * jstride ];
        }
    }
}

} // namespace kernel

/**
 * @brief Processes "physical" boundary conditions
 * 
 */
void species::process_bc() {

    // z boundaries
    if ( local_bc.x.lower == species::bc::reflecting || local_bc.x.upper == species::bc::reflecting ) {
        dim3 block( 1024 );
        dim3 grid( 2, particles->get_local_ntiles().y );
        kernel::species_bcz <<< grid, block >>> ( particles -> view(), local_bc );
    }

    // r boundaries
    if ( local_bc.y.upper == species::bc::reflecting ) {
        dim3 block( 1024 );
        dim3 grid( particles->get_local_ntiles().x, 2 );
        kernel::species_bcr <<< grid, block >>> ( particles -> view(), local_bc );
    }
}

#if 0
// I need to check which of these gives the best performance

/**
 * @brief Moves particles, deposit current and shift particle cell indices
 * 
 * current will be accumulated on existing data
 *
 * @note If nmodes = 2, both modes will be accumulated in the same kernel
 * 
 * @param J         Electric current density
 * @param shift     Cell index shift applied after the move
 */
void species::move( cyl_modes::tiled_cyl3<float> & current, const int2 shift )
{

    ///@brief timestep to cell size ratio
    const float2 dt_dx = make_float2(
        dt / dx.x,
        dt / dx.y
    );

    ///@brief Mode 0 current
    auto & J0 = current.mode0();

    auto ntiles = particles -> get_local_ntiles();

    // If number of tiles is small use multiple blocks per tile
    int tile_blocks = opt_min_blocks / ( ntiles.x * ntiles.y );
    if ( tile_blocks < 1 ) tile_blocks = 1;
    dim3 grid( ntiles.x, ntiles.y, tile_blocks );
    auto block = opt_move_block;

    size_t shm_size = J0.tile_vol * sizeof( cyl3<float> );

    switch (nmodes) {
    case 1: {
        
        gpu::block::set_shmem_size( kernel::move_deposit_0, shm_size );
        kernel::move_deposit_0 <<< grid, block, shm_size >>> ( 
            particles->view(), J0.view(), dt_dx, shift, d_nmove
        );
    } break;
    case 2: {
        auto & J1 = current.mode(1);
        shm_size += J1.tile_vol * sizeof( cyl3<complex_t> );
        gpu::block::set_shmem_size( kernel::move_deposit_1, shm_size );
        kernel::move_deposit_1 <<< grid, block, shm_size >>> ( 
            particles->view(), J0.view(), J1.view(), dt_dx, shift, d_nmove
        );
    } break;
    default:
        break;
    }
}

#else

/**
 * @brief Move particles (advance positions), deposit current and shift positions
 * 
 * @note For nmodes = 2, this version will:
 *        1. Deposit current for mode 1
 *        2. Move and deposit current for mode 0
 *      This version could easily be updated to an arbitrary number of modes 
 *
 * @param current   Electric current density
 * @param shift     Cell shift
 */
void species::move( cyl_modes::tiled_cyl3<float> & current, const int2 shift )
{

    ///@brief timestep to cell size ratio
    const float2 dt_dx = make_float2(
        dt / dx.x,
        dt / dx.y
    );

    ///@brief Mode 0 current
    auto & J0 = current.mode0();

    auto ntiles = particles -> get_local_ntiles();

    // If number of tiles is small use multiple blocks per tile
    int tile_blocks = opt_min_blocks / ( ntiles.x * ntiles.y );
    if ( tile_blocks < 1 ) tile_blocks = 1;
    dim3 grid( ntiles.x, ntiles.y, tile_blocks );
    auto block = opt_move_block;

    switch (nmodes) {
    case 1: {
        size_t shm_size = J0.tile_vol * sizeof( cyl3<float> );
        gpu::block::set_shmem_size( kernel::move_deposit_0, shm_size );
        kernel::move_deposit_0 <<< grid, block, shm_size >>> ( 
            particles -> view(), J0.view(),
            dt_dx, shift, d_nmove
        );
    } break;
    case 2: {
        // Deposit J1 current
        auto & J1 = current.mode(1);
        size_t shm_size = J1.tile_vol * sizeof( cyl3<complex_t> );
        gpu::block::set_shmem_size( kernel::deposit_1, shm_size );
        kernel::deposit_1 <<< grid, block, shm_size >>> ( 
            particles -> view(), J1.view(), dt_dx
        );

        // Deposit J0 current and move particles
        shm_size = J0.tile_vol * sizeof( cyl3<float> );
        gpu::block::set_shmem_size( kernel::move_deposit_0, shm_size );
        kernel::move_deposit_0 <<< grid, block, shm_size >>> ( 
            particles -> view(), J0.view(), dt_dx, shift, d_nmove
        );
    } break;
    default:
        break;
    }
}

#endif

/**
 * @brief Moves particles (no current deposition)
 * 
 * @note This is usually used for test species: species that do not
 *       self-consistently influence the simulation
 * 
 * @param shift     Additional cell shift for particles, defaults to 0
 */
void species::move( const int2 shift )
{
    const float2 dt_dx = make_float2(
        dt / dx.x,
        dt / dx.y
    );

    auto ntiles = particles -> get_local_ntiles();

    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 512 );

    kernel::move <<< grid, block >>> ( 
        particles -> view(), dt_dx, shift, d_nmove
    );
}

/**
 * @brief       Accelerates particles using a Boris pusher
 *
 * @note Only nmodes = 1, 2 is supported
 * 
 * @param emf     EMF field
 */
void species::push( cyl_modes::tiled_cyl3<float> & E, cyl_modes::tiled_cyl3<float> & B )
{
    const float alpha = 0.5 * dt / m_q;
    
    // Zero particle energy
    gpu::device::zero( d_energy, 1 );

    auto ntiles = particles -> get_local_ntiles();

    int tile_blocks = opt_min_blocks / ( ntiles.x * ntiles.y );
    if ( tile_blocks < 1 ) tile_blocks = 1;
    dim3 grid( ntiles.x, ntiles.y, tile_blocks );
    dim3 block( opt_push_block );

    auto & E0 = E.mode0();
    auto & B0 = B.mode0();
    size_t shm_size = 2 * ( E0.tile_vol * sizeof(cyl3<float>) );

    // Launch the push kernels for the selected pusher type
    auto launch = [&]( auto type_c ) {
        constexpr species::pusher type = decltype( type_c )::value;

        switch (nmodes) {
        case 1: {
            gpu::block::set_shmem_size( kernel::push_0 <type>, shm_size );
            kernel::push_0 <type> <<< grid, block, shm_size >>> ( 
                particles -> view(), E0.view(), B0.view(), alpha,
                d_energy
            );
        } break;
        case 2: {
            auto & E1 = E.mode(1);
            auto & B1 = B.mode(1);

            // Add shared memory size for mode 1 fields
            shm_size += 2 * ( E1.tile_vol * sizeof(cyl3<complex_t>) );

            gpu::block::set_shmem_size( kernel::push_1 <type>, shm_size );
            kernel::push_1 <type> <<< grid, block, shm_size >>> ( 
                particles -> view(), E0.view(), B0.view(),
                E1.view(), B1.view(), alpha, d_energy
            );
        } break;

        default:
            break;
        }
    };

    switch ( push_type ) {
    case species::pusher::euler:
        launch( std::integral_constant< species::pusher, species::pusher::euler >{} );
        break;
    case species::pusher::boris:
    default:
        launch( std::integral_constant< species::pusher, species::pusher::boris >{} );
        break;
    }
}

/**
 * @brief Deposit charge density (mode 0)
 * 
 * @param charge0    Charge density grid
 */
void species::deposit_charge0( grid::tiled<float> & charge0 ) const {

    auto ntiles = particles -> get_local_ntiles();
    int tile_blocks = opt_min_blocks / ( ntiles.x * ntiles.y );
    if ( tile_blocks < 1 ) tile_blocks = 1;
    dim3 grid( ntiles.x, ntiles.y, tile_blocks );
    dim3 block( opt_push_block );

    auto shm_size = charge0.tile_vol * sizeof(float);

    gpu::block::set_shmem_size( kernel::dep_charge_0, shm_size );
    kernel::dep_charge_0 <<< grid, block, shm_size >>> (
        particles -> view(), charge0.view()
    );

    // charge norm requires grid.z = 1
    dim3 grid_norm( ntiles.x, ntiles.y );
    kernel::charge_norm <<< grid_norm, block >>> (
         charge0.view(), dx.y
    );
}

/**
 * @brief Deposit charge density high order modes
 * 
 * @param m         Cylindrical mode to deposit (1 to 4)
 * @param charge    Charge density grid (complex)
 */
void species::deposit_charge( const unsigned m, grid::tiled<complex_t> &charge ) const {

    if ( m != 1  ) {
        mpi::fatal( "Only modes m = 1 is currently supported" );
    }

    auto ntiles = particles -> get_local_ntiles();
    int tile_blocks = opt_min_blocks / ( ntiles.x * ntiles.y );
    if ( tile_blocks < 1 ) tile_blocks = 1;
    dim3 grid( ntiles.x, ntiles.y, tile_blocks );
    dim3 block( opt_push_block );

    size_t shm_size = charge.tile_vol * sizeof( complex_t );

    gpu::block::set_shmem_size( kernel::dep_charge<1>, shm_size );
    kernel::dep_charge<1> <<< grid, block, shm_size >>> (
        particles -> view(), charge.view()
    );

    // High-order modes need an additional factor of 2, and fold across the
    // axis with parity (-1)^m
    dim3 grid_norm( ntiles.x, ntiles.y );
    kernel::charge_norm <<< grid_norm, block >>> (
        charge.view(), dx.y, 2.f, ( m % 2 ) ? -1.f : 1.f
    );
}

/**
 * @brief Saves charge density to file
 * 
 * The routine will create a new charge grid, deposit the charge and save the grid
 * 
 */
void species::save_charge( const unsigned m ) const {

    // For linear interpolation we only require 1 guard cell at the upper boundary
    bounds_2d<unsigned int> gc;
    gc.x = {0,1};
    gc.y = {0,1};

    // Prepare file info
    zdf::grid_axis axis[2];
    axis[0] = (zdf::grid_axis) {
        .name = (char *) "z",
        .min = mov_window.motion(),
        .max = box.x + mov_window.motion(),
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

    std::string grid_name = name + "-ρ" + std::to_string(m);
    std::string grid_label = name + " \\rho^" + std::to_string(m);

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

    if ( m == 0 ) {
        // Fundamental mode
        grid::tiled<float> charge( particles -> get_global_ntiles(), particles -> get_tile_dims(), gc, particles -> parallel );

        charge.zero();
        deposit_charge0( charge );
        charge.add_from_gc();
        charge.save( info, iter_info, path );
    } else {
        // High-order mode
        grid::tiled<complex_t> charge( particles -> get_global_ntiles(), particles -> get_tile_dims(), gc, particles -> parallel );

        charge.zero();
        deposit_charge( m, charge );
        charge.add_from_gc();
        charge.save( info, iter_info, path );
    }
}

/**
 * @brief save particle data to file
 * 
 */
void species::save() const {

    const std::string path = "PARTICLES";

    const cyl_part::quantity quants[] = {
        cyl_part::quantity::z, cyl_part::quantity::r, 
        cyl_part::quantity::q,
        cyl_part::quantity::cos_th, cyl_part::quantity::sin_th,
        cyl_part::quantity::ux, cyl_part::quantity::uy, cyl_part::quantity::uz 
    };

    const char * qnames[] = {
        "z","r",
        "q",
        "cosθ","sinθ",
        "ux","uy","uz"
    };

    const char * qlabels[] = {
        "z","r",
        "q",
        "\\cos \\theta", "\\sin \\theta",
        "u_x","u_y","u_z"
    };

    const char * qunits[] = {
        "c/\\omega_n", "c/\\omega_n",
        "e",
        "", "",
        "c","c","c"
    };

    zdf::iteration iter_info = {
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_n"
    };

    // Omit number of particles, this will be filled in later
    zdf::part_info info = {
        .name = (char *) name.c_str(),
        .label = (char *) name.c_str(),
        .nquants = 8,
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

            // Allocate device and host buffers for gathering particle data
            float * d_data = gpu::device::malloc<float>( local );
            float * h_data = gpu::host::malloc<float>( local );

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
            
            // z and r quantities are scaled before saving to file
            // z positions include the moving window motion (lab frame, as the grid diagnostics)
            scale = make_float2( dx.x, mov_window.motion() );
            particles -> gather( cyl_part::quantity::z, scale, d_data );
            gpu::device::memcpy_tohost( h_data, d_data, local );
            zdf::write_cdset( part_file, dsets[0], chunk, file_off );

            scale = make_float2( dx.y, 0 );
            particles -> gather( cyl_part::quantity::r, scale, d_data );
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

namespace phasespace {

template < phasespace::quantity quant >
__device__
float value ( int const i, int const shiftx, int const shifty,
    int2 const * const __restrict__ ix, float2 const * const __restrict__ x,
    float3 const * const __restrict__ u )
{
    if constexpr ( quant == quantity::z  ) return ( shiftx + ix[i].x ) + ( x[i].x + 0.5f );
    if constexpr ( quant == quantity::r  ) return ( shifty + ix[i].y ) + ( x[i].y );
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
    cyl_part::particles_view const part )
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
    float * __restrict__ q  = &part.q[ part_offset ];

    float const pha_rdx = size / (range.y - range.x);

    const int shiftx = (part.local_tile_start.x + tile_idx.x) * tile_dims.x;
    const int shifty = (part.local_tile_start.y + tile_idx.y) * tile_dims.y;

    for( int i = gpu::block::thread_rank(); i < np; i += gpu::block::num_threads() ) {
        float d = phasespace::value<quant>( i, shiftx, shifty, ix, x, u );

        float n =  (d - range.x ) * pha_rdx - 0.5f;
        int   k = floorf( n );
        float w = n - k;
        float qnorm = std::abs(q[i]) * norm;

        // Atomic across threads and tiles
        if ((k   >= 0) && (k   < size)) gpu::device::atomic_fetch_add( &d_data[k  ], (1-w) * qnorm);
        if ((k+1 >= 0) && (k+1 < size)) gpu::device::atomic_fetch_add( &d_data[k+1],    w  * qnorm );
    }
}

template < phasespace::quantity quant0, phasespace::quantity quant1 >
__global__
void dep_pha2(
    float * const __restrict__ d_data, 
    float2 const range0, int const size0,
    float2 const range1, int const size1,
    float const norm, 
    cyl_part::particles_view const part )
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
    float * __restrict__ q  = &part.q[ part_offset ];

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

        float qnorm = std::abs(q[i]) * norm;

        // Most common case, particle inside grid for all deposits
        if ( k0 >= 0 && k0+1 < size0 && k1 >= 0 && k1+1 < size1 ) {
            gpu::device::atomic_fetch_add( &d_data[(k1  )*size0 + k0  ], (1-w0) * (1-w1) * qnorm );
            gpu::device::atomic_fetch_add( &d_data[(k1  )*size0 + k0+1],    w0  * (1-w1) * qnorm );
            gpu::device::atomic_fetch_add( &d_data[(k1+1)*size0 + k0  ], (1-w0) *    w1  * qnorm );
            gpu::device::atomic_fetch_add( &d_data[(k1+1)*size0 + k0+1],    w0  *    w1  * qnorm );
        } else {
            // corner cases
            if ((k1   >= 0) && (k1   < size1)) {
                if ((k0   >= 0) && (k0   < size0)) 
                    gpu::device::atomic_fetch_add( &d_data[(k1  )*size0 + k0  ], (1-w0) * (1-w1) * qnorm );
                if ((k0+1 >= 0) && (k0+1 < size0)) 
                    gpu::device::atomic_fetch_add( &d_data[(k1  )*size0 + k0+1],    w0  * (1-w1) * qnorm );
            }
            if ((k1+1 >= 0) && (k1+1 < size1)) {
                if ((k0   >= 0) && (k0   < size0)) 
                    gpu::device::atomic_fetch_add( &d_data[(k1+1)*size0 + k0  ], (1-w0) *    w1  * qnorm );
                if ((k0+1 >= 0) && (k0+1 < size0)) 
                    gpu::device::atomic_fetch_add( &d_data[(k1+1)*size0 + k0+1],    w0  *    w1  * qnorm );
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
    case quantity::z : f( std::integral_constant< quantity, quantity::z >{} ); break;
    case quantity::r : f( std::integral_constant< quantity, quantity::r >{} ); break;
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
    float2 range, unsigned const size ) const
{
    // Zero device memory
    gpu::device::zero( d_data, size );
    
    float norm = ( dx.x * dx.y ) *
                 size / (range.y - range.x) ;

    dim3 grid( particles -> get_local_ntiles().x, particles -> get_local_ntiles().y );
    auto block = 64;

    // Launch the appropriate phasespace deposition kernel
    phasespace::dispatch(quant, [&] ( auto quant_c ) {
        constexpr auto quant = decltype(quant_c)::value;
        
        // Normalize range to cell sizes, if required. z ranges are given in
        // the lab frame and converted to the moving window frame
        if constexpr ( quant == phasespace::quantity::z ) {
            const float x0 = mov_window.motion();
            range.x = ( range.x - x0 ) / dx.x; range.y = ( range.y - x0 ) / dx.x;
        }
        if constexpr ( quant == phasespace::quantity::r ) { range.x /= dx.y; range.y /= dx.y; }
        
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
    const phasespace::quantity quant0, float2 range0, const unsigned int size0,
    const phasespace::quantity quant1, float2 range1, const unsigned int size1 ) const
{
    // Zero device memory
    gpu::device::zero( d_data, size0 * size1 );

    float norm = ( dx.x * dx.y ) *
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

            // Normalize range to cell sizes, if required. x ranges are given
            // in the lab frame and converted to the moving window frame
            // (quant1 > quant0, so only quant0 can be x)
            if constexpr ( quant0 == phasespace::quantity::z ) {
                const float x0 = mov_window.motion();
                range0.x = ( range0.x - x0 ) / dx.x; range0.y = ( range0.y - x0 ) / dx.x;
            }
            if constexpr ( quant0 == phasespace::quantity::r ) { range0.x /= dx.y; range0.y /= dx.y; }
            if constexpr ( quant1 == phasespace::quantity::r ) { range1.x /= dx.y; range1.y /= dx.y; }
            
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
        mpi::cout << "(*error*) for 2D phasespaces, the 2nd quantity must be indexed higher"
                     " than the first one, phasespace not saved\n";
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

