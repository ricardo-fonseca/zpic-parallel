#include "species.hpp"
#include <iostream>
#include <string>
#include <vector>

#include "simd/simd.hpp"
#include "simd/vec_complex.hpp"

using complex_t = species::complex_t;

/**
 * @brief Memory alignment of local buffers
 * 
 * @warning Must be >= 32 to avoid some compiler issues at high optimization
 *          levels
 */
constexpr int local_align = 64;

/**
 * @brief Force inlining of the small per-particle routines used inside the
 *        multi-mode loops
 * 
 * @note GCC does not inline dep_current_seg_n() / interpolate_fld_w() on its
 *       own (they contain loops / are called several times per particle), and
 *       the call overhead roughly doubles the cost of the single pass kernels
 */
#if defined(__GNUC__) || defined(__clang__)
#define Q3D_FORCE_INLINE __attribute__((always_inline)) inline
#else
#define Q3D_FORCE_INLINE inline
#endif

/**
 * @brief Trajectory splitter selection
 * 
 * 0 (default) - curved path splitter; 1 - original splitter (kept for
 * reference / comparison with other codes, see split2d_cyl()). The SIMD
 * move / deposit kernel implements the curved path splitter only, so the
 * scalar move / deposit kernels are used when the original splitter is
 * selected.
 */
#ifndef Q3D_ORIGINAL_SPLIT
#define Q3D_ORIGINAL_SPLIT 0
#endif

/// @brief Maximum number of cylindrical modes supported
/// @note The CPU code has no intrinsic limit, this matches the CUDA version
int constexpr max_push_modes = 16;

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
    if ( nmodes_ < 1 || nmodes_ > max_push_modes ) {
        mpi::fatal("Invalid number of modes, must be in [1, " + std::to_string( max_push_modes ) + "]");
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
    sort = new part::particle_sort( local_ntiles, max_part, parallel, periodic );
    np_inj = memory::malloc<int>( local_ntiles.x * local_ntiles.y );

    // Initialize energy diagnostic
    energy = 0;

    // Initialize particle move counter
    nmoved = 0;

    // Reset iteration numbers
    iter = 0;

    // Inject initial distribution

    // Count particles to inject and store in np_inj
    np_inject( particles -> local_range(), np_inj );

    // Do an exclusive scan to get the required offsets
    uint32_t num_part = 0;
    auto tile_offset = particles -> get_tile_offset();
    for( unsigned i = 0; i < local_ntiles.x * local_ntiles.y; i ++ ) {
        tile_offset[i] = num_part;
        num_part += np_inj[i];
    }

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

namespace kernel{

/**
 * @brief Returns reciprocal Lorentz gamma factor
 * 
 * $ \frac{1}{\sqrt{u_x^2 + u_y^2 + u_z^2 + 1 }} $
 * 
 * @param u         Generalized momentum in units of c
 * @return float    Reciprocal Lorentz gamma factor
 */
inline float rgamma( const float3 u ) {
    return 1.0f/std::sqrt( ops::fma( u.z, u.z, 
                           ops::fma( u.y, u.y, 
                           ops::fma( u.x, u.x, 1.0f ) ) ) );
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
void interpolate_fld( 
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
 * @param alpha     Normalization 
 * @param e         E-field interpolated at particle position
 * @param b         B-field interpolated at particle position
 * @param u         Initial particle momentum
 * @param energy    Particle energy (time centered)
 * @return float3   Final particle momentum
 */
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
        const float utsq = ops::fma( ut.z, ut.z, ops::fma( ut.y, ut.y, ut.x * ut.x ) );
        const float gamma = std::sqrt( 1.0f + utsq );
        
        // Get time centered energy
        energy = utsq / (gamma + 1.0f);

        // Time centered \alpha / \gamma
        const float alpha_gamma = alpha / gamma;

        // Rotation
        b.x *= alpha_gamma;
        b.y *= alpha_gamma;
        b.z *= alpha_gamma;
    }

    u.x = ops::fma( b.z, ut.y, ut.x );
    u.y = ops::fma( b.x, ut.z, ut.y );
    u.z = ops::fma( b.y, ut.x, ut.z );

    u.x = ops::fma( -b.y, ut.z, u.x );
    u.y = ops::fma( -b.z, ut.x, u.y );
    u.z = ops::fma( -b.x, ut.y, u.z );

    {
        const float otsq = 2.0f / 
            ops::fma( b.z, b.z, ops::fma( b.y, b.y, ops::fma( b.x, b.x, 1.0f ) ) );
        
        b.x *= otsq;
        b.y *= otsq;
        b.z *= otsq;
    }

    ut.x = ops::fma( b.z, u.y, ut.x );
    ut.y = ops::fma( b.x, u.z, ut.y );
    ut.z = ops::fma( b.y, u.x, ut.z );

    ut.x = ops::fma( -b.y, u.z, ut.x );
    ut.y = ops::fma( -b.z, u.x, ut.y );
    ut.z = ops::fma( -b.x, u.y, ut.z );

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
        const float utsq = ops::fma( ut.z, ut.z, ops::fma( ut.y, ut.y, ut.x * ut.x ) );
        const float gamma = std::sqrt( 1.0f + utsq );
        
        // Get time centered energy
        energy = utsq / (gamma + 1.0f);
        
        // Time centered 2 * \alpha / \gamma
        float const alpha2_gamma = ( alpha * 2 ) / gamma ;

        b.x *= alpha2_gamma;
        b.y *= alpha2_gamma;
        b.z *= alpha2_gamma;
    }

    {
        float const bnorm = std::sqrt(ops::fma( b.x, b.x, ops::fma( b.y, b.y, b.z * b.z ) ));
        float const s = -(( bnorm > 0 ) ? std::sin( bnorm / 2 ) / bnorm : 0.5f );

        float const ra = std::cos( bnorm / 2 );
        float const rb = b.x * s;
        float const rc = b.y * s;
        float const rd = b.z * s;

        float const r11 =   ops::fma(ra,ra,rb*rb)-ops::fma(rc,rc,rd*rd);
        float const r12 = 2*ops::fma(rb,rc,ra*rd);
        float const r13 = 2*ops::fma(rb,rd,-ra*rc);

        float const r21 = 2*ops::fma(rb,rc,-ra*rd);
        float const r22 =   ops::fma(ra,ra,rc*rc)-ops::fma(rb,rb,rd*rd);
        float const r23 = 2*ops::fma(rc,rd,ra*rb);

        float const r31 = 2*ops::fma(rb,rd,ra*rc);
        float const r32 = 2*ops::fma(rc,rd,-ra*rb);
        float const r33 =   ops::fma(ra,ra,rd*rd)-ops::fma(rb,rb,rc*rc);

        u.x = ops::fma( r11, ut.x, ops::fma( r21, ut.y , r31 * ut.z ));
        u.y = ops::fma( r12, ut.x, ops::fma( r22, ut.y , r32 * ut.z ));
        u.z = ops::fma( r13, ut.x, ops::fma( r23, ut.y , r33 * ut.z ));
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

    // Azimuthal displacement (r Δθ, cell units) times charge. The 1/3 factor
    // is the normalization of the Jθ weights below
    const auto jθ = q * ( ops::fma( - Δx , yif , Δy * xif ) ) / ( 3 * std::sqrt( ops::fma( xif, xif, yif*yif) ) );

    int i = ix.x;
    int j = ix.y;

    // When using more than 1 thread per tile all of these need to be atomic
    J[ stride * j     + i     ].z += wl1 * wp10;
    J[ stride * (j+1) + i     ].z += wl1 * wp11;

    J[ stride * j     + i     ].r += wl2 * wp20;
    J[ stride * j     + (i+1) ].r += wl2 * wp21;

    // Jθ weights are the time average of Sz(t) Sr(t) along the segment:
    //   (S0z S0r + S1z S1r)/3 + (S0z S1r + S1z S0r)/6
    // The factor 1/3 is included in jθ
    J[ stride * j     + i     ].th += jθ * ( S0z0 * S0r0 + S1z0 * S1r0 + (S0z0 * S1r0 + S1z0 * S0r0)/2.0f );
    J[ stride * j     + (i+1) ].th += jθ * ( S0z1 * S0r0 + S1z1 * S1r0 + (S0z1 * S1r0 + S1z1 * S0r0)/2.0f );
    J[ stride * (j+1) + i     ].th += jθ * ( S0z0 * S0r1 + S1z0 * S1r1 + (S0z0 * S1r1 + S1z0 * S0r1)/2.0f );
    J[ stride * (j+1) + (i+1) ].th += jθ * ( S0z1 * S0r1 + S1z1 * S1r1 + (S0z1 * S1r1 + S1z1 * S0r1)/2.0f );
}

/**
 * @brief Returns exp( -i m θ ) from the angular position (cos θ, sin θ)
 * 
 * @note Computed by repeated multiplication of (cos θ - i sin θ), which is
 *       exact for m = 1 and accurate to O(m) roundoff for m > 1. The same
 *       function must be used for the charge and the current deposits, so
 *       that the current deposit is charge conserving for every mode.
 * 
 * @param m     Mode number (m >= 1)
 * @param th    Angular position (cos θ, sin θ)
 * @return      exp( -i m θ )
 */ 
inline complex_t expmimt( const int m, const float2 th ) {
    // Repeated multiplication by (cos θ - i sin θ), using the same operation
    // order as util::complex::operator*= in the CUDA version (this also
    // avoids the NaN checks of the std::complex multiplication)
    float re = th.x, im = -th.y;
    for( int k = 1; k < m; k++ ) {
        const float re1 = ops::fma( re, th.x,  im * th.y );
        const float im1 = ops::fma( re, -th.y, im * th.x );
        re = re1; im = im1;
    }
    return complex_t{ re, im };
}

/**
 * @brief Deposit (charge conserving) current for 1 segment inside a cell,
 *        all modes m = 1 ... nmodes-1 at once
 * 
 * For every mode m > 0 the segment geometry and shape factors are computed
 * once, and the angular factors exp( -i m θ ) are updated incrementally from
 * one mode to the next, using the same operations as expmimt() (which is also
 * used by the charge deposit, so that the deposit is charge conserving for
 * every mode).
 * 
 * Jz and Jr use the mid-point factor exp(-i m θm); Jθ carries the difference
 * between the endpoint factors and exp(-i m θm), so that
 * Δρ_m + ΔJz + ΔJr - Jθ = 0 holds exactly (charge conservation)
 * 
 * @param nmodes    Number of modes (including mode 0)
 * @param ix        Particle cell
 * @param x0        Initial particle position (z,r)
 * @param x1        Final particle position (z,r)
 * @param t0        Initial particle transverse position (x,y)
 * @param t1        Final particle  transverse position (x,y)
 * @param q         Particle charge
 * @param J         Mode 1 current grid, pointing at cell (0,0); mode m is at
 *                  J + (m-1) * mstride
 * @param stride    Current grid y stride
 * @param mstride   Distance between the current grids of consecutive modes
 */
Q3D_FORCE_INLINE void dep_current_seg_n(
    const int nmodes,
    const int2 ix, const float2 x0, const float2 x1,
    const float2 t0, const float2 t1, 
    const float q,
    cyl3<complex_t> * __restrict__ J, const int stride, const int mstride )
{
    int i = ix.x;
    int j = ix.y;
    const auto z0 = x0.x;
    const auto z1 = x1.x;
    const auto r0 = x0.y;
    const auto r1 = x1.y;

    const auto rcr0 = 1.0f / std::sqrt( ops::fma( t0.x, t0.x, t0.y*t0.y ) );
    const auto rcr1 = 1.0f / std::sqrt( ops::fma( t1.x, t1.x, t1.y*t1.y ) );

    /// @brief Initial θ
    const auto th0 = make_float2( t0.x * rcr0, t0.y * rcr0 );
    /// @brief Final θ
    const auto th1 = make_float2( t1.x * rcr1, t1.y * rcr1 );

    // Not contracted, see dep_current_seg_0()
    const auto xif = t0.x + t1.x;
    const auto yif = t0.y + t1.y;

    const auto rrm2 = 1.0f / std::sqrt( ops::fma( xif, xif, yif*yif ) );
    const auto thm = float2{ xif * rrm2, yif * rrm2 };

    const auto S0z0 = 0.5f - z0;
    const auto S0z1 = 0.5f + z0;

    const auto S1z0 = 0.5f - z1;
    const auto S1z1 = 0.5f + z1;

    const auto S0r0 = 0.5f - r0;
    const auto S0r1 = 0.5f + r0;

    const auto S1r0 = 0.5f - r1;
    const auto S1r1 = 0.5f + r1;

    const auto dz = z1 - z0;
    const auto dr = r1 - r0;
    
    const auto wp10 = 0.5f*(S0r0 + S1r0);
    const auto wp11 = 0.5f*(S0r1 + S1r1);
    
    const auto wp20 = 0.5f*(S0z0 + S1z0);
    const auto wp21 = 0.5f*(S0z1 + S1z1);

    /// @brief exp( -i m θ ) for the initial, final and mid point angles
    float e0r = th0.x, e0i = -th0.y;
    float e1r = th1.x, e1i = -th1.y;
    float emr = thm.x, emi = -thm.y;

    for( int m = 1; m < nmodes; m++ ) {

        if ( m > 1 ) {
            // exp( -i m θ ) = exp( -i (m-1) θ ) exp( -i θ ), as in expmimt()
            float re, im;
            re = ops::fma( e0r, th0.x, e0i * th0.y ); im = ops::fma( e0r, -th0.y, e0i * th0.x ); e0r = re; e0i = im;
            re = ops::fma( e1r, th1.x, e1i * th1.y ); im = ops::fma( e1r, -th1.y, e1i * th1.x ); e1r = re; e1i = im;
            re = ops::fma( emr, thm.x, emi * thm.y ); im = ops::fma( emr, -thm.y, emi * thm.x ); emr = re; emi = im;
        }

        const complex_t cm{ emr, emi };
        const auto c0 = complex_t{ e0r, e0i } - cm;
        const auto c1 = complex_t{ e1r, e1i } - cm;

        const auto wl1 = dz * cm;
        const auto wl2 = dr * cm;

        cyl3<complex_t> * __restrict__ Jm = J + ( m - 1 ) * mstride;

        Jm[ stride * j     + i     ].z += q * wl1 * wp10;
        Jm[ stride * (j+1) + i     ].z += q * wl1 * wp11;

        Jm[ stride * j     + i     ].r += q * wl2 * wp20;
        Jm[ stride * j     + (i+1) ].r += q * wl2 * wp21;

        Jm[ stride * j     + i     ].th += q * ( S1z0 * S1r0 * c1 - S0z0 * S0r0 * c0 );
        Jm[ stride * j     + (i+1) ].th += q * ( S1z1 * S1r0 * c1 - S0z1 * S0r0 * c0 );
        Jm[ stride * (j+1) + i     ].th += q * ( S1z0 * S1r1 * c1 - S0z0 * S0r1 * c0 );
        Jm[ stride * (j+1) + (i+1) ].th += q * ( S1z1 * S1r1 * c1 - S0z1 * S0r1 * c0 );
    }
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
inline void split2d_cyl( 
    const int ir0, const int2 ix, const float2 x0, const float2 delta, int2 & deltai,
    const float2 t0, const float2 tdelta,
    int2 & v0_ix, float2 & v0_x0, float2 & v0_x1, float2 & v0_t0, float2 & v0_t1,
    int2 & v1_ix, float2 & v1_x0, float2 & v1_x1, float2 & v1_t0, float2 & v1_t1,
    int2 & v2_ix, float2 & v2_x0, float2 & v2_x1, float2 & v2_t0, float2 & v2_t1,
    int2 & cross
) {
    /// @brief final (z,r) position, indexed to original cell
    float2 x1 = make_float2( x0.x + delta.x, x0.y + delta.y );

    /// @brief final (x,y) transverse position
    float2 t1 = make_float2( t0.x + tdelta.x, t0.y + tdelta.y );

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


#if Q3D_ORIGINAL_SPLIT
    // Original code - keep for reference
    #warning Using original split2d_cyl

    // z-split
    float xz, yz, rz;
    if ( cross.x ) {
        εz = (zs - x0.x) / delta.x;

        // z-split positions
        xz = t0.x + εz * tdelta.x;
        yz = t0.y + εz * tdelta.y;

        rz = ( delta.y == 0 ) ? x0.y : std::sqrt( ops::fma( xz, xz, yz * yz ) ) - (ir0 + ix.y);
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
    const float a = ops::fma( tdelta.x, tdelta.x, tdelta.y * tdelta.y );
    /// @brief t0 · Δt
    const float b = ops::fma( t0.x, tdelta.x, t0.y * tdelta.y );
    /// @brief initial radial position (cell based, as in the move kernels)
    const float r0 = ( ir0 + ix.y ) + x0.y;

    // z-split
    float xz, yz, rz;
    if ( cross.x ) {
        εz = (zs - x0.x) / delta.x;

        // z-split positions
        xz = ops::fma( εz, tdelta.x, t0.x );
        yz = ops::fma( εz, tdelta.y, t0.y );

        // Radial position at the z-split, relative to the initial cell. Uses
        // r(ε) - r0 = ε ( 2 b + ε a ) / ( r(ε) + r0 ), which avoids the
        // cancellation in sqrt( xz² + yz² ) - ( ir0 + ix.y ) at large radii.
        // - delta.y == 0: no radial motion, keep rz = x0.y exactly so that no
        //   (spurious) Jr is deposited
        // - den == 0: particle exactly on axis and εz == 0, avoids 0/0
        const float den = std::sqrt( ops::fma( xz, xz, yz * yz ) ) + r0;
        rz = ( delta.y != 0 && den > 0 ) ? x0.y + εz * ops::fma( εz, a, 2 * b ) / den : x0.y;
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
        const float q = - ( b + std::copysign( std::sqrt( std::fmax( ops::fma( b, b, - a * c ), 0.f ) ), b ) );
        const float e1 = q / a;
        const float e2 = c / q;

        // First crossing: smallest root >= 0 (outward motion has one root of
        // each sign, inward motion two positive roots)
        const float lo = std::fmin( e1, e2 );
        const float hi = std::fmax( e1, e2 );

        // The crossing itself was decided from delta.y, so clamp to [0,1];
        // fmaxf / fminf also map a NaN (q == 0) to a valid value
        εr = std::fmin( std::fmax( ( lo >= 0 ) ? lo : hi, 0.f ), 1.f );

        // r-split positions
        xr = ops::fma( εr, tdelta.x, t0.x );
        yr = ops::fma( εr, tdelta.y, t0.y );
        zr = ops::fma( εr, delta.x, x0.x );
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

/**
 * @brief Kernel for moving particles without depositing current
 * 
 * @param part          Particle data
 * @param dt_dx         Ratio between time step and cell size (x/y)
 * @param shift         Additional cell shift for particles
 * @param d_nmove       (out) Number of particles pushed (for performance metrics)
 */
void move(
    const uint2 tile_idx,
    cyl_part::particles_view const part,
    float2 const dt_dx, int2 const shift
) {
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

    for( int i = 0; i < np; i++ ) {
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
        auto xf = ops::fma( ri, cos_th, Δx );
        auto yf = ops::fma( ri, sin_th, Δy );

        /// @brief xi + xf
        auto xif = ops::fma( ri, cos_th, xf );
        /// @brief yi + yf
        auto yif = ops::fma( ri, sin_th, yf );

        // Final positions

        /// @brief New radial position
        auto rf = std::sqrt( ops::fma( xf, xf, yf*yf ) );
        /// @brief radial motion
        auto Δr = ops::fma( Δx , xif , Δy * yif ) / (rf + ri);
        // auto Δr = rf - ri;
    
    
        // Advance grid (z,r) position
        auto Δz = dt_dz * rg * pu.z;
        float2 delta = make_float2( Δz, Δr );

        // Same rounding as split2d_cyl (no fma contraction)
        auto x1 = make_float2( x0.x + delta.x, x0.y + delta.y );
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
}


/**
 * @brief Move particles and deposit current (mode 0 only) and shift positions
 * 
 * @param tile_idx          Tile index
 * @param part              Particle data
 * @param current_m0        Current grid (m=0)
 * @param current_offset    Offset to position [0,0] of the current grid
 * @param nx                Current grid size (internal)
 * @param ext_nx            Current grid size (external)
 * @param dt_dx             Ratio between time step and cell size
 * @param shift             Cell shift
 */
void move_deposit_0(
    uint2 const tile_idx,
    cyl_part::particles_view part,
    grid::tiled_cyl3_view<float> current_m0, 
    float2 const dt_dx, const int2 shift ) 
{
    const int tile_id   = tile_idx.y * part.local_ntiles.x + tile_idx.x;
    const int tile_np   = part.tile_np[ tile_id ];

    // This is usually in block shared memory
    alignas(local_align) cyl3<float> _current_buffer_m0[ current_m0.tile_vol ];

    // Zero local current buffer
    memory::zero( _current_buffer_m0, current_m0.tile_vol );

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

    for( int i = 0; i < tile_np; i++ ) {
        auto pu  = u[i];
        auto x0  = x[i];
        auto ix0 = ix[i];
        auto θi  = th[i];
        auto pq  = q[i];

        // Get 1 / Lorentz gamma
        float const rg = rgamma( pu );

        // Cartesian motion
        auto Δx = dt_dr * rg * pu.x;
        auto Δy = dt_dr * rg * pu.y;

        /// @brief initial radial position
        auto ri = (ir0 + ix0.y) + x0.y;
        auto xi = ri * θi.x;
        auto yi = ri * θi.y;

        // New cartesian positions
        auto xf = ops::fma( ri, θi.x, Δx );
        auto yf = ops::fma( ri, θi.y, Δy );

        /// @brief xi + xf
        auto xif = ops::fma( ri, θi.x, xf );
        /// @brief yi + yf
        auto yif = ops::fma( ri, θi.y, yf );

        // Final positions

        /// @brief New radial position
        auto rf = std::sqrt( ops::fma( xf, xf, yf*yf ) );
        /// @brief radial motion
        auto Δr = ops::fma( Δx , xif , Δy * yif ) / (rf + ri);
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

    // Current normalization is done in Current::normalize()

    // Add current to global buffer
    float * dev_J0 = reinterpret_cast< float * > ( current_m0.tile_buffer(tile_id));
    float * shm_J0 = reinterpret_cast< float * > ( _current_buffer_m0 );

    for( auto i =  0; i < 3 * current_m0.tile_vol; i++ ) {
        dev_J0[ i ] += shm_J0[ i ]; 
    }
}

/**
 * @brief Move particles, deposit current (all modes, single pass) and
 *        shift positions
 * 
 * Single pass version for any number of modes: the trajectory of each
 * particle is computed and split once, and the current of all modes is
 * deposited from the same segments (see dep_current_seg_n()).
 * 
 * @param tile_idx          Tile index
 * @param part              Particle data
 * @param current_m0        Current grid (m=0)
 * @param current_m         Current grids (m > 0), indexed by m (entry 0 unused)
 * @param nmodes            Number of modes (including mode 0)
 * @param dt_dx             Ratio between time step and cell size
 * @param shift             Cell shift
 * @param buffer_m0         Scratch buffer for the mode 0 tile current,
 *                          current_m0.tile_vol elements (one per thread)
 * @param buffer_m          Scratch buffer for the tile current of the modes
 *                          m > 0, ( nmodes - 1 ) * tile_vol elements (one per
 *                          thread)
 */
void move_deposit_n(
    uint2 const tile_idx,
    cyl_part::particles_view const part,
    grid::tiled_cyl3_view<float> current_m0, 
    grid::tiled_cyl3_view<complex_t> const * const current_m, 
    const int nmodes,
    float2 const dt_dx, const int2 shift,
    cyl3<float> * const __restrict__ buffer_m0,
    cyl3<complex_t> * const __restrict__ buffer_m )
{
    const int tile_id   = tile_idx.y * part.local_ntiles.x + tile_idx.x;
    const int tile_np   = part.tile_np[ tile_id ];

    if ( tile_np == 0 ) return;

    const int tile_vol = current_m0.tile_vol;

    // Zero local current buffers
    memory::zero( buffer_m0, tile_vol );
    memory::zero( buffer_m, ( nmodes - 1 ) * tile_vol );

    cyl3<float>     * J0 = & buffer_m0[ current_m0.inner_offset ];
    cyl3<complex_t> * Jm = & buffer_m[ current_m0.inner_offset ];
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

    for( int i = 0; i < tile_np; i++ ) {
        auto pu  = u[i];
        auto x0  = x[i];
        auto ix0 = ix[i];
        auto θi  = th[i];
        auto pq  = q[i];

        // Get 1 / Lorentz gamma
        float const rg = rgamma( pu );

        // Cartesian motion
        auto Δx = dt_dr * rg * pu.x;
        auto Δy = dt_dr * rg * pu.y;

        /// @brief initial radial position
        auto ri = (ir0 + ix0.y) + x0.y;
        auto xi = ri * θi.x;
        auto yi = ri * θi.y;

        // New cartesian positions
        auto xf = ops::fma( ri, θi.x, Δx );
        auto yf = ops::fma( ri, θi.y, Δy );

        /// @brief xi + xf
        auto xif = ops::fma( ri, θi.x, xf );
        /// @brief yi + yf
        auto yif = ops::fma( ri, θi.y, yf );

        /// @brief New radial position
        auto rf = std::sqrt( ops::fma( xf, xf, yf*yf ) );
        /// @brief radial motion
        auto Δr = ops::fma( Δx , xif , Δy * yif ) / (rf + ri);

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

        // Deposit current (modes m > 0)
        dep_current_seg_n( nmodes, v0_ix, v0_x0, v0_x1, v0_t0, v0_t1, pq, Jm, jstride, tile_vol );
        if ( cross.x || cross.y ) dep_current_seg_n( nmodes, v1_ix, v1_x0, v1_x1, v1_t0, v1_t1, pq, Jm, jstride, tile_vol );
        if ( cross.x && cross.y ) dep_current_seg_n( nmodes, v2_ix, v2_x0, v2_x1, v2_t0, v2_t1, pq, Jm, jstride, tile_vol );

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

    // Add current to global buffers
    {
        float * dev_J = reinterpret_cast< float * > ( current_m0.tile_buffer(tile_id) );
        float * loc_J = reinterpret_cast< float * > ( buffer_m0 );
        for( int k = 0; k < 3 * tile_vol; k++ ) dev_J[ k ] += loc_J[ k ];
    }

    for( int m = 1; m < nmodes; m++ ) {
        float * dev_J = reinterpret_cast< float * > ( current_m[m].tile_buffer(tile_id) );
        float * loc_J = reinterpret_cast< float * > ( & buffer_m[ ( m - 1 ) * tile_vol ] );
        for( int k = 0; k < 3 * 2 * tile_vol; k++ ) dev_J[ k ] += loc_J[ k ];
    }
}

/**
 * @brief Advance particle velocities (mode 0 only)
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
void push_0 ( 
    uint2 const tile_idx,
    cyl_part::particles_view part,
    grid::tiled_cyl3_view<float> E_grid, grid::tiled_cyl3_view<float> B_grid, 
    float const alpha, double & tile_energy )
{
    const int tile_id   = tile_idx.y * part.local_ntiles.x + tile_idx.x;
    const int tile_np   = part.tile_np[ tile_id ];

    // Copy E and B into shared memory

    alignas(local_align) cyl3<float> E_local_m0[ E_grid.tile_vol ];
    alignas(local_align) cyl3<float> B_local_m0[ B_grid.tile_vol ];

    memory::memcpy( E_local_m0, E_grid.tile_buffer( tile_id ), E_grid.tile_vol );
    memory::memcpy( B_local_m0, B_grid.tile_buffer( tile_id ), B_grid.tile_vol );

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

    for( int i = 0; i < tile_np; i++ ) {

        // Interpolate field
        cyl3<float> e, b;
        interpolate_fld( E_m0, B_m0, jstride, ix[i], x[i], e, b );
        
        // Convert to cartesian components
        auto cosθ = th[i].x;
        auto sinθ = th[i].y;
        
        float3 cart_e = make_float3(
            ops::fma( e.r, cosθ, - e.th * sinθ ),
            ops::fma( e.r, sinθ, + e.th * cosθ ),
            e.z
        );

        float3 cart_b = make_float3(
            ops::fma( b.r, cosθ, - b.th * sinθ ),
            ops::fma( b.r, sinθ, + b.th * cosθ ),
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

    // Return energy from all particles in tile
    tile_energy = energy;
}

/**
 * @brief Linear interpolation weights for the staggered E / B grids
 * 
 * Holds the cell indices and shape factors used by interpolate_fld(), so
 * that they can be computed once per particle and reused for every mode.
 */
struct interp_weights {
    int i, j, ih, jh;
    float s0z, s1z, s0r, s1r;
    float s0zh, s1zh, s0rh, s1rh;

    interp_weights( const int2 ix, const float2 x ) {
        i = ix.x;
        j = ix.y;

        const auto z = x.x;
        const auto r = x.y;

        s0z = 0.5f - z;
        s1z = 0.5f + z;

        s0r = 0.5f - r;
        s1r = 0.5f + r;

        const int hz = z < 0;
        const int hr = r < 0;

        ih = i - hz;
        jh = j - hr;

        s0zh = (1-hz) - z;
        s1zh = (  hz) + z;

        s0rh = (1-hr) - r;
        s1rh = (  hr) + r;
    }
};

/**
 * @brief Interpolate E and B at the particle position using precomputed
 *        weights (same interpolation as interpolate_fld())
 */
template< typename T >
Q3D_FORCE_INLINE void interpolate_fld_w( 
    cyl3<T> const * const __restrict__ E, 
    cyl3<T> const * const __restrict__ B, 
    const int jstride, const interp_weights & w,
    cyl3<T> & e, cyl3<T> & b )
{
    const int i = w.i, j = w.j, ih = w.ih, jh = w.jh;

    e.z = ( E[ih +     j *jstride].z * w.s0zh + E[ih+1 +     j *jstride].z * w.s1zh ) * w.s0r +
          ( E[ih + (j +1)*jstride].z * w.s0zh + E[ih+1 + (j +1)*jstride].z * w.s1zh ) * w.s1r;

    e.r = ( E[i  +     jh*jstride].r * w.s0z  + E[i+1  +     jh*jstride].r * w.s1z ) * w.s0rh +
          ( E[i  + (jh+1)*jstride].r * w.s0z  + E[i+1  + (jh+1)*jstride].r * w.s1z ) * w.s1rh;

    e.th = ( E[i  +     j *jstride].th * w.s0z  + E[i+1  +     j *jstride].th * w.s1z ) * w.s0r +
           ( E[i  + (j +1)*jstride].th * w.s0z  + E[i+1  + (j +1)*jstride].th * w.s1z ) * w.s1r;

    b.z = ( B[i  +     jh*jstride].z * w.s0z  + B[i+1  +     jh*jstride].z * w.s1z ) * w.s0rh +
          ( B[i  + (jh+1)*jstride].z * w.s0z  + B[i+1  + (jh+1)*jstride].z * w.s1z ) * w.s1rh;

    b.r = ( B[ih +      j*jstride].r * w.s0zh + B[ih+1 +      j*jstride].r * w.s1zh ) * w.s0r +
          ( B[ih + (j +1)*jstride].r * w.s0zh + B[ih+1 + (j +1)*jstride].r * w.s1zh ) * w.s1r;

    b.th = ( B[ih +     jh*jstride].th * w.s0zh + B[ih+1 +     jh*jstride].th * w.s1zh ) * w.s0rh +
           ( B[ih + (jh+1)*jstride].th * w.s0zh + B[ih+1 + (jh+1)*jstride].th * w.s1zh ) * w.s1rh;
}

/**
 * @brief Advance particle velocities, any number of modes (single pass)
 * 
 * For each particle the interpolation weights are computed once, and the E
 * and B fields of all modes are gathered in one go, accumulating
 * Re( F_m exp( i m θ ) ). The field tiles are read in place (one thread per
 * tile, read only).
 * 
 * @tparam type         Pusher type
 * @param tile_idx      Tile index
 * @param part          Particle data
 * @param E_grid_m0     E field grid, mode 0
 * @param B_grid_m0     B field grid, mode 0
 * @param E_grid_m      E field grids, modes m > 0 (indexed by m, entry 0 unused)
 * @param B_grid_m      B field grids, modes m > 0 (indexed by m, entry 0 unused)
 * @param nmodes        Number of modes (including mode 0)
 * @param alpha         Normalization parameter
 * @param tile_energy   (out) Total particle energy in the tile
 */
template < species::pusher type >
void push_all ( 
    uint2 const tile_idx,
    cyl_part::particles_view part,
    grid::tiled_cyl3_view<float> E_grid_m0, grid::tiled_cyl3_view<float> B_grid_m0,
    grid::tiled_cyl3_view<complex_t> const * const E_grid_m,
    grid::tiled_cyl3_view<complex_t> const * const B_grid_m,
    const int nmodes,
    float const alpha, double & tile_energy )
{
    const int tile_id   = tile_idx.y * part.local_ntiles.x + tile_idx.x;
    const int tile_np   = part.tile_np[ tile_id ];

    const int jstride   = E_grid_m0.tile_ystride();

    const int part_offset  = part.tile_offset[ tile_id ];
    auto * __restrict__ ix = &part.ix[ part_offset ];
    auto * __restrict__ x  = &part.x[ part_offset ];
    auto * __restrict__ u  = &part.u[ part_offset ];
    auto * __restrict__ th = &part.th[ part_offset ];
    auto * __restrict__ q  = &part.q[ part_offset ];

    // Pointers to (0,0) of the field tiles of each mode
    cyl3<float> const * const E0 = E_grid_m0.tile_data( tile_id );
    cyl3<float> const * const B0 = B_grid_m0.tile_data( tile_id );

    constexpr int max_modes = max_push_modes;
    cyl3<complex_t> const * Em[ max_modes ];
    cyl3<complex_t> const * Bm[ max_modes ];
    for( int m = 1; m < nmodes; m++ ) {
        Em[m] = E_grid_m[m].tile_data( tile_id );
        Bm[m] = B_grid_m[m].tile_data( tile_id );
    }

    double energy = 0;

    for( int i = 0; i < tile_np; i++ ) {

        const interp_weights w( ix[i], x[i] );
        const float2 t = th[i];

        // Mode 0
        cyl3<float> e, b;
        interpolate_fld_w( E0, B0, jstride, w, e, b );

        // Modes m > 0
        float emr = 1, emi = 0;
        for( int m = 1; m < nmodes; m++ ) {
            // exp( i m θ ) = exp( i (m-1) θ ) * ( cos θ + i sin θ )
            const float re = ops::fma( emr, t.x, - emi * t.y );
            const float im = ops::fma( emr, t.y,   emi * t.x );
            emr = re; emi = im;

            cyl3<complex_t> ve, vb;
            interpolate_fld_w( Em[m], Bm[m], jstride, w, ve, vb );

            e.z  += ops::fma( emr, ve.z.real(),  -emi * ve.z.imag()  );
            e.r  += ops::fma( emr, ve.r.real(),  -emi * ve.r.imag()  );
            e.th += ops::fma( emr, ve.th.real(), -emi * ve.th.imag() );

            b.z  += ops::fma( emr, vb.z.real(),  -emi * vb.z.imag()  );
            b.r  += ops::fma( emr, vb.r.real(),  -emi * vb.r.imag()  );
            b.th += ops::fma( emr, vb.th.real(), -emi * vb.th.imag() );
        }

        const auto cosθ = t.x;
        const auto sinθ = t.y;

        // Convert to cartesian components
        float3 cart_e = make_float3(
            ops::fma( e.r, cosθ, - e.th * sinθ ),
            ops::fma( e.r, sinθ, + e.th * cosθ ),
            e.z
        );

        float3 cart_b = make_float3(
            ops::fma( b.r, cosθ, - b.th * sinθ ),
            ops::fma( b.r, sinθ, + b.th * cosθ ),
            b.z
        );

        float3 pu = u[i];

        /// @brief Particle kinetic energy (γ - 1), time centered
        double ekin = 0;
        if constexpr ( type == species::pusher::boris ) u[i] = dudt_boris( alpha, cart_e, cart_b, pu, ekin );
        if constexpr ( type == species::pusher::euler ) u[i] = dudt_boris_euler( alpha, cart_e, cart_b, pu, ekin );

        // Particles have different weights (charge q ∝ r)
        energy += q[i] * ekin;
    }

    tile_energy = energy;
}

/*
 * Vector (SIMD) optimized functions
 */

#ifdef SIMD

// SIMD types and vec_*() functions (see simd/simd.hpp, simd/vec_complex.hpp)
using namespace simd;

/**
 * @brief Returns reciprocal Lorentz gamma factor (SIMD version)
 * 
 * @param u         (simd vector) Generalized velocity in units of c
 * @return vfloat   (simd vector) Reciprocal Lorentz gamma factor
 */
inline vfloat rgamma( const vfloat3 u ) {
    vfloat c1 = vec_float(1);
    return vec_div( c1, vec_sqrt( vec_fmadd( u.z, u.z,
                                  vec_fmadd( u.y, u.y,
                                  vec_fmadd( u.x, u.x, c1 ) ) ) ) );
}

/**
 * @brief Advance momentum using a relativistic Boris pusher (SIMD version)
 * 
 * @note See dudt_boris() for details
 * 
 * @param alpha     (simd vector) Normalization 0.5 q dt / m
 * @param e         (simd vector) E-field at particle position (cartesian)
 * @param b         (simd vector) B-field at particle position (cartesian)
 * @param u         (simd vector) Initial generalized velocity
 * @param ekin[out] (simd vector) Particle kinetic energy (γ - 1), time centered
 * @return vfloat3  (simd vector) Final generalized velocity
 */
inline vfloat3 vdudt_boris( const vfloat alpha, vfloat3 e, vfloat3 b, vfloat3 u, vfloat & ekin )
{
    // First half of acceleration
    e.x = vec_mul( e.x, alpha );
    e.y = vec_mul( e.y, alpha );
    e.z = vec_mul( e.z, alpha );

    vfloat3 ut { 
        vec_add( u.x, e.x ),
        vec_add( u.y, e.y ),
        vec_add( u.z, e.z )
    };

    {
        const vfloat utsq = vec_fmadd( ut.z, ut.z, vec_fmadd( ut.y, ut.y, vec_mul( ut.x, ut.x ) ) );
        const vfloat gamma = vec_sqrt( vec_add( utsq, 1.0f ) );
        
        // Get time centered energy
        ekin = vec_div( utsq, vec_add( gamma, 1.0f ) );

        // Time centered \alpha / \gamma
        const vfloat alpha_gamma = vec_div( alpha, gamma );

        // Rotation
        b.x = vec_mul( b.x, alpha_gamma );
        b.y = vec_mul( b.y, alpha_gamma );
        b.z = vec_mul( b.z, alpha_gamma );
    }

    u.x = vec_fmadd( b.z, ut.y, ut.x );
    u.y = vec_fmadd( b.x, ut.z, ut.y );
    u.z = vec_fmadd( b.y, ut.x, ut.z );

    u.x = vec_fnmadd( b.y, ut.z, u.x );
    u.y = vec_fnmadd( b.z, ut.x, u.y );
    u.z = vec_fnmadd( b.x, ut.y, u.z );

    {
        const vfloat otsq = vec_div( 
            vec_float(2), 
            vec_fmadd( b.z, b.z, vec_fmadd( b.y, b.y, vec_fmadd( b.x, b.x, vec_float(1) ) ) )
        );
        
        b.x = vec_mul( b.x, otsq );
        b.y = vec_mul( b.y, otsq );
        b.z = vec_mul( b.z, otsq );
    }

    ut.x = vec_fmadd( b.z, u.y, ut.x );
    ut.y = vec_fmadd( b.x, u.z, ut.y );
    ut.z = vec_fmadd( b.y, u.x, ut.z );

    ut.x = vec_fnmadd( b.y, u.z, ut.x );
    ut.y = vec_fnmadd( b.z, u.x, ut.y );
    ut.z = vec_fnmadd( b.x, u.y, ut.z );

    // Second half of acceleration
    ut.x = vec_add( ut.x, e.x );
    ut.y = vec_add( ut.y, e.y );
    ut.z = vec_add( ut.z, e.z );

    return ut;
}

/**
 * @brief Advance momentum using a relativistic Boris pusher for high magnetic
 *        fields (Euler-Rodrigues rotation), SIMD version
 * 
 * @note See dudt_boris_euler() for details. As in the em2d SIMD version, the
 *       rotation quaternion is obtained using vec_sinc_cos(), so no test on
 *       |b| > 0 is required.
 * 
 * @param alpha     (simd vector) Normalization 0.5 q dt / m
 * @param e         (simd vector) E-field at particle position (cartesian)
 * @param b         (simd vector) B-field at particle position (cartesian)
 * @param u         (simd vector) Initial generalized velocity
 * @param ekin[out] (simd vector) Particle kinetic energy (γ - 1), time centered
 * @return vfloat3  (simd vector) Final generalized velocity
 */
inline vfloat3 vdudt_boris_euler( const vfloat alpha, vfloat3 e, vfloat3 b, vfloat3 u, vfloat & ekin )
{
    // First half of acceleration
    e.x = vec_mul( e.x, alpha );
    e.y = vec_mul( e.y, alpha );
    e.z = vec_mul( e.z, alpha );
 
    vfloat3 ut {
        vec_add( u.x, e.x ),
        vec_add( u.y, e.y ),
        vec_add( u.z, e.z )
    };
 
    {
        const vfloat utsq  = vec_fmadd( ut.z, ut.z, vec_fmadd( ut.y, ut.y, vec_mul( ut.x, ut.x ) ) );
        const vfloat gamma = vec_sqrt( vec_add( utsq, 1.0f ) );
 
        // Get time centered energy
        ekin = vec_div( utsq, vec_add( gamma, 1.0f ) );
 
        // Time centered 2 * \alpha / \gamma
        const vfloat alpha2_gamma = vec_div( vec_add( alpha, alpha ), gamma );
 
        b.x = vec_mul( b.x, alpha2_gamma );
        b.y = vec_mul( b.y, alpha2_gamma );
        b.z = vec_mul( b.z, alpha2_gamma );
    }
 
    {
        // Compute ||b||/2
        const vfloat bnorm_2 = vec_mul( vec_sqrt(
                vec_fmadd( b.z, b.z, vec_fmadd( b.y, b.y, vec_mul( b.x, b.x ) ) )
            ), .5f );

        // Compute cos( ||b||/2 ) and -sin( ||b||/2 ) / ||b||
        vfloat ra, sinc;
        vec_sinc_cos( bnorm_2, sinc, ra );
        const vfloat s  = vec_mul( sinc, -0.5f );
 
        const vfloat rb = vec_mul( b.x, s );
        const vfloat rc = vec_mul( b.y, s );
        const vfloat rd = vec_mul( b.z, s );
  
        const vfloat r11 = vec_sub( vec_fmadd( ra, ra, vec_mul( rb, rb ) ),
                                    vec_fmadd( rc, rc, vec_mul( rd, rd ) ) );
        const vfloat r12 = vec_mul( vec_fmadd(  rb, rc, vec_mul( ra, rd ) ), 2.f );
        const vfloat r13 = vec_mul( vec_fnmadd( ra, rc, vec_mul( rb, rd ) ), 2.f );
 
        const vfloat r21 = vec_mul( vec_fnmadd( ra, rd, vec_mul( rb, rc ) ), 2.f );
        const vfloat r22 = vec_sub( vec_fmadd( ra, ra, vec_mul( rc, rc ) ),
                                    vec_fmadd( rb, rb, vec_mul( rd, rd ) ) );
        const vfloat r23 = vec_mul( vec_fmadd(  rc, rd, vec_mul( ra, rb ) ), 2.f );
 
        const vfloat r31 = vec_mul( vec_fmadd(  rb, rd, vec_mul( ra, rc ) ), 2.f );
        const vfloat r32 = vec_mul( vec_fnmadd( ra, rb, vec_mul( rc, rd ) ), 2.f );
        const vfloat r33 = vec_sub( vec_fmadd( ra, ra, vec_mul( rd, rd ) ),
                                    vec_fmadd( rb, rb, vec_mul( rc, rc ) ) );
 
        u.x = vec_fmadd( r11, ut.x, vec_fmadd( r21, ut.y, vec_mul( r31, ut.z ) ) );
        u.y = vec_fmadd( r12, ut.x, vec_fmadd( r22, ut.y, vec_mul( r32, ut.z ) ) );
        u.z = vec_fmadd( r13, ut.x, vec_fmadd( r23, ut.y, vec_mul( r33, ut.z ) ) );
    }
 
    // Second half of acceleration
    u.x = vec_add( u.x, e.x );
    u.y = vec_add( u.y, e.y );
    u.z = vec_add( u.z, e.z );
 
    return u;
}

/**
 * @brief Linear interpolation weights for the staggered E / B grids (SIMD)
 * 
 * Same as interp_weights. Cell indices are linear (cell) indices into the
 * tile, i + j * jstride, for the 4 combinations of regular / half-shifted
 * cells used by the E and B components.
 */
struct vinterp_weights {
    vint i_j, i_jh, ih_j, ih_jh;
    vfloat s0z, s1z, s0r, s1r;
    vfloat s0zh, s1zh, s0rh, s1rh;

    vinterp_weights( const vint2 ix, const vfloat2 x, const int jstride ) {
        const vfloat c_1_2 = vec_float( 0.5f );

        s0z = vec_sub( c_1_2, x.x );
        s1z = vec_add( c_1_2, x.x );

        s0r = vec_sub( c_1_2, x.y );
        s1r = vec_add( c_1_2, x.y );

        const vint c_1i = vec_int( 1 );

        const vint hz = vec_lt( x.x, vec_zero_float(), c_1i );
        const vint hr = vec_lt( x.y, vec_zero_float(), c_1i );

        const vint ih = vec_sub( ix.x, hz );
        const vint jh = vec_sub( ix.y, hr );

        s0zh = vec_sub( vec_float( vec_sub( c_1i, hz ) ), x.x );
        s1zh = vec_add( vec_float(                hz  ), x.x );

        s0rh = vec_sub( vec_float( vec_sub( c_1i, hr ) ), x.y );
        s1rh = vec_add( vec_float(                hr  ), x.y );

        const vint j  = vec_mul( ix.y, jstride );
        const vint jhs = vec_mul( jh, jstride );

        i_j   = vec_add( ix.x, j );
        i_jh  = vec_add( ix.x, jhs );
        ih_j  = vec_add( ih, j );
        ih_jh = vec_add( ih, jhs );
    }
};

/**
 * @brief Bilinear interpolation of one field component (SIMD)
 * 
 * @param base      Address of the component of cell (0,0), as float *
 * @param idx       (simd vector) Linear index (in floats) of the lower cell
 * @param sx        Distance (in floats) between consecutive cells along z
 * @param sy        Distance (in floats) between consecutive cells along r
 * @param w0z, w1z  (simd vector) z weights
 * @param w0r, w1r  (simd vector) r weights
 * @return vfloat   (simd vector) Interpolated value
 */
inline vfloat vinterp( const float * __restrict__ base, const vint idx,
    const int sx, const int sy,
    const vfloat w0z, const vfloat w1z, const vfloat w0r, const vfloat w1r )
{
    vfloat f0 = vec_mul(   vec_gather( base,      idx ), w0z );
    f0 = vec_fmadd( vec_gather( base + sx,      idx ), w1z, f0 );
    vfloat f1 = vec_mul(   vec_gather( base + sy, idx ), w0z );
    f1 = vec_fmadd( vec_gather( base + sx + sy, idx ), w1z, f1 );
    return vec_fmadd( f1, w1r, vec_mul( f0, w0r ) );
}

/**
 * @brief Interpolate E and B at the particle positions (SIMD)
 * 
 * @tparam NF       Number of floats per field component (1 - real, 2 - complex)
 * @param E         E field tile, cell (0,0), as float *
 * @param B         B field tile, cell (0,0), as float *
 * @param jstride   Field tile y stride (in cells)
 * @param w         Interpolation weights
 * @param part      Real (0) or imaginary (1) part (complex fields only)
 * @param e[out]    (simd vector) E field (z, r, θ in x, y, z)
 * @param b[out]    (simd vector) B field (z, r, θ in x, y, z)
 */
template< int NF >
inline void vinterpolate_fld( const float * __restrict__ E, const float * __restrict__ B,
    const int jstride, const vinterp_weights & w, const int part,
    vfloat3 & e, vfloat3 & b )
{
    // Floats per cell and strides (in floats)
    constexpr int cell = 3 * NF;
    const int sy = cell * jstride;

    const vint i_j   = vec_mul( w.i_j,   cell );
    const vint i_jh  = vec_mul( w.i_jh,  cell );
    const vint ih_j  = vec_mul( w.ih_j,  cell );
    const vint ih_jh = vec_mul( w.ih_jh, cell );

    // Component offsets (z, r, θ)
    const int oz = 0 * NF + part, or_ = 1 * NF + part, ot = 2 * NF + part;

    e.x = vinterp( E + oz,  ih_j,  cell, sy, w.s0zh, w.s1zh, w.s0r,  w.s1r  );
    e.y = vinterp( E + or_, i_jh,  cell, sy, w.s0z,  w.s1z,  w.s0rh, w.s1rh );
    e.z = vinterp( E + ot,  i_j,   cell, sy, w.s0z,  w.s1z,  w.s0r,  w.s1r  );

    b.x = vinterp( B + oz,  i_jh,  cell, sy, w.s0z,  w.s1z,  w.s0rh, w.s1rh );
    b.y = vinterp( B + or_, ih_j,  cell, sy, w.s0zh, w.s1zh, w.s0r,  w.s1r  );
    b.z = vinterp( B + ot,  ih_jh, cell, sy, w.s0zh, w.s1zh, w.s0rh, w.s1rh );
}

/**
 * @brief Bilinear interpolation of one complex field component (SIMD)
 * 
 * The (real, imaginary) pairs are gathered together using vec_gather_c()
 * 
 * @param base      Address of the component of cell (0,0)
 * @param idx       (simd vector) Linear index (in complex values) of the lower cell
 * @param sx        Distance (in complex values) between consecutive cells along z
 * @param sy        Distance (in complex values) between consecutive cells along r
 * @param w0z, w1z  (simd vector) z weights
 * @param w0r, w1r  (simd vector) r weights
 * @return vcomplex (simd vector) Interpolated value
 */
inline vcomplex vinterp_c( const complex_t * __restrict__ base, const vint idx,
    const int sx, const int sy,
    const vfloat w0z, const vfloat w1z, const vfloat w0r, const vfloat w1r )
{
    vcomplex f0 = vec_mul(   vec_gather_c( base,           idx ), w0z );
    f0 = vec_fmadd( vec_gather_c( base + sx,      idx ), w1z, f0 );
    vcomplex f1 = vec_mul(   vec_gather_c( base + sy,      idx ), w0z );
    f1 = vec_fmadd( vec_gather_c( base + sx + sy, idx ), w1z, f1 );
    return vec_fmadd( f1, w1r, vec_mul( f0, w0r ) );
}

/**
 * @brief SIMD vector of cylindrical complex vectors
 */
struct vcomplex3 {
    vcomplex z, r, th;
};

/**
 * @brief Interpolate E and B (complex, mode m > 0) at the particle
 *        positions (SIMD)
 * 
 * @param E         E field tile, cell (0,0)
 * @param B         B field tile, cell (0,0)
 * @param jstride   Field tile y stride (in cells)
 * @param w         Interpolation weights
 * @param e[out]    (simd vector) E field
 * @param b[out]    (simd vector) B field
 */
inline void vinterpolate_fld_c( const cyl3<complex_t> * __restrict__ E, const cyl3<complex_t> * __restrict__ B,
    const int jstride, const vinterp_weights & w,
    vcomplex3 & e, vcomplex3 & b )
{
    // Complex values per cell and strides (in complex values)
    constexpr int cell = 3;
    const int sy = cell * jstride;

    const vint i_j   = vec_mul3( w.i_j   );
    const vint i_jh  = vec_mul3( w.i_jh  );
    const vint ih_j  = vec_mul3( w.ih_j  );
    const vint ih_jh = vec_mul3( w.ih_jh );

    const complex_t * __restrict__ Ec = reinterpret_cast< const complex_t * >( E );
    const complex_t * __restrict__ Bc = reinterpret_cast< const complex_t * >( B );

    e.z  = vinterp_c( Ec + 0, ih_j,  cell, sy, w.s0zh, w.s1zh, w.s0r,  w.s1r  );
    e.r  = vinterp_c( Ec + 1, i_jh,  cell, sy, w.s0z,  w.s1z,  w.s0rh, w.s1rh );
    e.th = vinterp_c( Ec + 2, i_j,   cell, sy, w.s0z,  w.s1z,  w.s0r,  w.s1r  );

    b.z  = vinterp_c( Bc + 0, i_jh,  cell, sy, w.s0z,  w.s1z,  w.s0rh, w.s1rh );
    b.r  = vinterp_c( Bc + 1, ih_j,  cell, sy, w.s0zh, w.s1zh, w.s0r,  w.s1r  );
    b.th = vinterp_c( Bc + 2, ih_jh, cell, sy, w.s0zh, w.s1zh, w.s0rh, w.s1rh );
}

/**
 * @brief Advance particle velocities, any number of modes (SIMD version)
 * 
 * Same algorithm as push_all() (also used for nmodes = 1), processing
 * simd::width particles at a time. The fields are gathered from the field
 * tiles (read in place); for modes m > 0 the complex values are gathered
 * as (real, imaginary) pairs (vec_gather_c) and combined with exp( i m θ )
 * (vcomplex).
 * Remaining particles are processed with the scalar code.
 * 
 * @tparam type         Pusher type
 * @param tile_idx      Tile index
 * @param part          Particle data
 * @param E_grid_m0     E field grid, mode 0
 * @param B_grid_m0     B field grid, mode 0
 * @param E_grid_m      E field grids, modes m > 0 (indexed by m, entry 0 unused)
 * @param B_grid_m      B field grids, modes m > 0 (indexed by m, entry 0 unused)
 * @param nmodes        Number of modes (including mode 0)
 * @param alpha         Normalization parameter
 * @param tile_energy   (out) Total particle energy in the tile
 */
template < species::pusher type >
void vpush_all ( 
    uint2 const tile_idx,
    cyl_part::particles_view part,
    grid::tiled_cyl3_view<float> E_grid_m0, grid::tiled_cyl3_view<float> B_grid_m0,
    grid::tiled_cyl3_view<complex_t> const * const E_grid_m,
    grid::tiled_cyl3_view<complex_t> const * const B_grid_m,
    const int nmodes,
    float const alpha, double & tile_energy )
{
    const int tile_id   = tile_idx.y * part.local_ntiles.x + tile_idx.x;
    const int tile_np   = part.tile_np[ tile_id ];

    const int jstride   = E_grid_m0.tile_ystride();

    const int part_offset  = part.tile_offset[ tile_id ];
    auto * __restrict__ ix = &part.ix[ part_offset ];
    auto * __restrict__ x  = &part.x[ part_offset ];
    auto * __restrict__ u  = &part.u[ part_offset ];
    auto * __restrict__ th = &part.th[ part_offset ];
    auto * __restrict__ q  = &part.q[ part_offset ];

    // Pointers to (0,0) of the field tiles of each mode
    cyl3<float> const * const E0 = E_grid_m0.tile_data( tile_id );
    cyl3<float> const * const B0 = B_grid_m0.tile_data( tile_id );

    constexpr int max_modes = max_push_modes;
    cyl3<complex_t> const * Em[ max_modes ];
    cyl3<complex_t> const * Bm[ max_modes ];
    for( int m = 1; m < nmodes; m++ ) {
        Em[m] = E_grid_m[m].tile_data( tile_id );
        Bm[m] = B_grid_m[m].tile_data( tile_id );
    }

    double energy = 0;

    const vfloat valpha = vec_float( alpha );
    const int np_vec = ( tile_np / simd::width ) * simd::width;

    for( int i = 0; i < np_vec; i += simd::width ) {

        const vint2   vix = vec_load_s2( (int *)   & ix[i] );
        const vfloat2 vx  = vec_load_s2( (float *) & x[i] );
        const vfloat2 vth = vec_load_s2( (float *) & th[i] );

        const vinterp_weights w( vix, vx, jstride );

        // Mode 0 (z, r, θ components in x, y, z)
        vfloat3 e, b;
        vinterpolate_fld<1>( (const float *) E0, (const float *) B0, jstride, w, 0, e, b );

        // Modes m > 0
        vcomplex em = vec_complex( 1.0f, 0.0f );
        const vcomplex eth = vec_complex( vth.x, vth.y );
        for( int m = 1; m < nmodes; m++ ) {
            // exp( i m θ ) = exp( i (m-1) θ ) * ( cos θ + i sin θ )
            em = vec_mul( em, eth );

            vcomplex3 ec, bc;
            vinterpolate_fld_c( Em[m], Bm[m], jstride, w, ec, bc );

            // Re( F_m exp( i m θ ) )
            e.x = vec_add( e.x, vec_real_mul( em, ec.z  ) );
            e.y = vec_add( e.y, vec_real_mul( em, ec.r  ) );
            e.z = vec_add( e.z, vec_real_mul( em, ec.th ) );

            b.x = vec_add( b.x, vec_real_mul( em, bc.z  ) );
            b.y = vec_add( b.y, vec_real_mul( em, bc.r  ) );
            b.z = vec_add( b.z, vec_real_mul( em, bc.th ) );
        }

        // Convert to cartesian components
        const vfloat3 cart_e {
            vec_fmsub( e.y, vth.x, vec_mul( e.z, vth.y ) ),
            vec_fmadd( e.y, vth.y, vec_mul( e.z, vth.x ) ),
            e.x
        };

        const vfloat3 cart_b {
            vec_fmsub( b.y, vth.x, vec_mul( b.z, vth.y ) ),
            vec_fmadd( b.y, vth.y, vec_mul( b.z, vth.x ) ),
            b.x
        };

        vfloat3 pu = vec_load_s3( (float *) & u[i] );

        vfloat ekin;
        if constexpr ( type == species::pusher::boris ) pu = vdudt_boris( valpha, cart_e, cart_b, pu, ekin );
        if constexpr ( type == species::pusher::euler ) pu = vdudt_boris_euler( valpha, cart_e, cart_b, pu, ekin );

        vec_store_s3( (float *) & u[i], pu );

        // Particles have different weights (charge q ∝ r)
        energy += vec_reduce_add( vec_mul( vec_loadu( & q[i] ), ekin ) );
    }

    // Remaining particles (scalar code, see push_all())
    for( int i = np_vec; i < tile_np; i++ ) {

        const interp_weights w( ix[i], x[i] );
        const float2 t = th[i];

        cyl3<float> e, b;
        interpolate_fld_w( E0, B0, jstride, w, e, b );

        float emr = 1, emi = 0;
        for( int m = 1; m < nmodes; m++ ) {
            const float re = ops::fma( emr, t.x, - emi * t.y );
            const float im = ops::fma( emr, t.y,   emi * t.x );
            emr = re; emi = im;

            cyl3<complex_t> ve, vb;
            interpolate_fld_w( Em[m], Bm[m], jstride, w, ve, vb );

            e.z  += ops::fma( emr, ve.z.real(),  -emi * ve.z.imag()  );
            e.r  += ops::fma( emr, ve.r.real(),  -emi * ve.r.imag()  );
            e.th += ops::fma( emr, ve.th.real(), -emi * ve.th.imag() );

            b.z  += ops::fma( emr, vb.z.real(),  -emi * vb.z.imag()  );
            b.r  += ops::fma( emr, vb.r.real(),  -emi * vb.r.imag()  );
            b.th += ops::fma( emr, vb.th.real(), -emi * vb.th.imag() );
        }

        float3 cart_e = make_float3(
            ops::fma( e.r, t.x, - e.th * t.y ),
            ops::fma( e.r, t.y, + e.th * t.x ),
            e.z
        );

        float3 cart_b = make_float3(
            ops::fma( b.r, t.x, - b.th * t.y ),
            ops::fma( b.r, t.y, + b.th * t.x ),
            b.z
        );

        float3 pu = u[i];

        double ekin = 0;
        if constexpr ( type == species::pusher::boris ) u[i] = dudt_boris( alpha, cart_e, cart_b, pu, ekin );
        if constexpr ( type == species::pusher::euler ) u[i] = dudt_boris_euler( alpha, cart_e, cart_b, pu, ekin );

        energy += q[i] * ekin;
    }

    tile_energy = energy;
}

/**
 * @brief Trajectory segment (SIMD)
 */
struct vsegment {
    vint2   ix;
    vfloat2 x0, x1;
    vfloat2 t0, t1;
};

/**
 * @brief Split particle trajectories into segments fully contained in a
 *        cell (SIMD version of split2d_cyl(), curved path splitter)
 * 
 * @note Same algorithm and operation order as the (new) split2d_cyl(). All
 *       possible split positions are computed for every particle and the
 *       segments are assembled using vec_select(). Values computed for lanes
 *       that do not use them (e.g. the z-split position of a particle that
 *       does not cross a z edge) may be inf / NaN, and are discarded.
 * 
 * @param ir0       Radial cell index of tile lower edge
 * @param ix        (simd vector) Initial particle cell
 * @param x0        (simd vector) Initial particle position (z,r)
 * @param delta     (simd vector) Particle motion (z,r)
 * @param deltai    (simd vector) [out] Particle cell motion
 * @param t0        (simd vector) Initial transverse position (x,y)
 * @param tdelta    (simd vector) Transverse motion (x,y)
 * @param v0        (simd vector) [out] 1st segment
 * @param v1        (simd vector) [out] 2nd segment
 * @param v2        (simd vector) [out] 3rd segment
 * @param cross_z   (simd vector) [out] Mask of particles crossing a z edge
 * @param cross_r   (simd vector) [out] Mask of particles crossing a r edge
 */
inline void vsplit2d_cyl( 
    const int ir0, const vint2 ix, const vfloat2 x0, const vfloat2 delta, vint2 & deltai,
    const vfloat2 t0, const vfloat2 tdelta,
    vsegment & v0, vsegment & v1, vsegment & v2,
    vmask & cross_z, vmask & cross_r )
{
    const vfloat c_1_2 = vec_float( 0.5f );
    const vint   c_1i  = vec_int( 1 );
    const vint   c_0i  = vec_int( 0 );

    /// @brief final (z,r) position, indexed to original cell
    const vfloat2 x1{ vec_add( x0.x, delta.x ), vec_add( x0.y, delta.y ) };

    /// @brief final (x,y) transverse position
    const vfloat2 t1{ vec_add( t0.x, tdelta.x ), vec_add( t0.y, tdelta.y ) };

    // Get cell (iz,ir) motion
    deltai.x = vec_sub( vec_ge( x1.x, c_1_2, c_1i ), vec_lt( x1.x, vec_neg( c_1_2 ), c_1i ) );
    deltai.y = vec_sub( vec_ge( x1.y, c_1_2, c_1i ), vec_lt( x1.y, vec_neg( c_1_2 ), c_1i ) );

    // Get cell crossings
    cross_z = vec_ne( deltai.x, c_0i );
    cross_r = vec_ne( deltai.y, c_0i );

    const vfloat fdeltai_x = vec_float( deltai.x );
    const vfloat fdeltai_y = vec_float( deltai.y );

    /// @brief cell position of z-split, if any
    const vfloat zs = vec_mul( c_1_2, fdeltai_x );
    /// @brief cell position of r-split, if any
    const vfloat rs = vec_mul( c_1_2, fdeltai_y );

    /// @brief |Δt|²
    const vfloat a = vec_fmadd( tdelta.x, tdelta.x, vec_mul( tdelta.y, tdelta.y ) );
    /// @brief t0 · Δt
    const vfloat b = vec_fmadd( t0.x, tdelta.x, vec_mul( t0.y, tdelta.y ) );

    /// @brief ir0 + ix.y
    const vint ir = vec_add( vec_int( ir0 ), ix.y );
    /// @brief initial radial position (cell based, as in the move kernels)
    const vfloat r0 = vec_add( vec_float( ir ), x0.y );

    // z-split
    const vfloat εz = vec_div( vec_sub( zs, x0.x ), delta.x );
    const vfloat xz = vec_fmadd( εz, tdelta.x, t0.x );
    const vfloat yz = vec_fmadd( εz, tdelta.y, t0.y );
    vfloat rz;
    {
        const vfloat den = vec_add( vec_sqrt( vec_fmadd( xz, xz, vec_mul( yz, yz ) ) ), r0 );
        const vfloat drz = vec_div( vec_mul( εz, vec_fmadd( εz, a, vec_add( b, b ) ) ), den );
        const vmask  use = vec_and( vec_ne( delta.y, vec_zero_float() ), vec_gt( den, vec_zero_float() ) );
        rz = vec_select( x0.y, vec_add( x0.y, drz ), use );
    }

    // r-split
    vfloat εr;
    {
        const vfloat c = vec_mul( vec_sub( x0.y, rs ),
            vec_add( vec_add( vec_float( vec_add( ir, ir ) ), x0.y ), rs ) );
        const vfloat disc = vec_max( vec_fmadd( b, b, vec_neg( vec_mul( a, c ) ) ), vec_zero_float() );
        const vfloat qq = vec_neg( vec_add( b, vec_copysign( vec_sqrt( disc ), b ) ) );
        const vfloat e1 = vec_div( qq, a );
        const vfloat e2 = vec_div( c, qq );
        const vfloat lo = vec_min( e1, e2 );
        const vfloat hi = vec_max( e1, e2 );
        const vfloat root = vec_select( hi, lo, vec_ge( lo, vec_zero_float() ) );
        εr = vec_min( vec_max( root, vec_zero_float() ), vec_float( 1.0f ) );
    }
    const vfloat xr = vec_fmadd( εr, tdelta.x, t0.x );
    const vfloat yr = vec_fmadd( εr, tdelta.y, t0.y );
    const vfloat zr = vec_fmadd( εr, delta.x, x0.x );

    // Crossing cases
    const vmask both   = vec_and( cross_z, cross_r );
    const vmask zfirst = vec_and( both, vec_lt( εz, εr ) );
    /// @brief z-split is the 1st split (z-cross only, or z-cross first)
    const vmask zsplit = vec_or( vec_and( cross_z, vec_not( cross_r ) ), zfirst );
    /// @brief r-split is the 1st split (r-cross only, or r-cross first)
    const vmask rsplit = vec_or( vec_and( cross_r, vec_not( cross_z ) ), vec_and( both, vec_not( zfirst ) ) );
    /// @brief z-cross only
    const vmask zonly  = vec_and( cross_z, vec_not( cross_r ) );
    /// @brief r-cross only
    const vmask ronly  = vec_and( cross_r, vec_not( cross_z ) );

    const vfloat x1z = vec_sub( x1.x, fdeltai_x );
    const vfloat x1r = vec_sub( x1.y, fdeltai_y );

    // 1st segment
    v0.ix = ix; v0.x0 = x0; v0.t0 = t0;
    v0.x1.x = vec_select( vec_select( x1.x, zr, rsplit ), zs, zsplit );
    v0.x1.y = vec_select( vec_select( x1.y, rs, rsplit ), rz, zsplit );
    v0.t1.x = vec_select( vec_select( t1.x, xr, rsplit ), xz, zsplit );
    v0.t1.y = vec_select( vec_select( t1.y, yr, rsplit ), yz, zsplit );

    // 2nd segment
    v1.ix.x = vec_select( ix.x, vec_add( ix.x, deltai.x ), zsplit );
    v1.ix.y = vec_select( ix.y, vec_add( ix.y, deltai.y ), rsplit );
    v1.x0.x = vec_select( zr,  vec_neg( zs ), zsplit );
    v1.x0.y = vec_select( vec_neg( rs ), rz, zsplit );
    v1.x1.x = vec_select( vec_select( vec_select( zs, vec_sub( zr, fdeltai_x ), zfirst ), x1.x, ronly ), x1z, zonly );
    v1.x1.y = vec_select( vec_select( vec_select( vec_sub( rz, fdeltai_y ), rs, zfirst ), x1r, ronly ), x1.y, zonly );
    v1.t0   = v0.t1;
    v1.t1.x = vec_select( vec_select( xz, xr, zfirst ), t1.x, vec_or( zonly, ronly ) );
    v1.t1.y = vec_select( vec_select( yz, yr, zfirst ), t1.y, vec_or( zonly, ronly ) );

    // 3rd segment
    v2.ix.x = vec_add( ix.x, deltai.x );
    v2.ix.y = vec_add( ix.y, deltai.y );
    v2.x0.x = vec_select( vec_neg( zs ), vec_sub( zr, fdeltai_x ), zfirst );
    v2.x0.y = vec_select( vec_sub( rz, fdeltai_y ), vec_neg( rs ), zfirst );
    v2.t0.x = vec_select( xz, xr, zfirst );
    v2.t0.y = vec_select( yz, yr, zfirst );
    v2.x1   = vfloat2{ x1z, x1r };
    v2.t1   = t1;
}

/**
 * @brief Mode 0 current weights of a segment (SIMD version of
 *        dep_current_seg_0(), weights only)
 * 
 * @param s         (simd vector) Segment
 * @param q         (simd vector) Particle charge
 * @param w[out]    (simd vector) Weights: Jz(i,j), Jz(i,j+1), Jr(i,j),
 *                  Jr(i+1,j), Jθ(i,j), Jθ(i+1,j), Jθ(i,j+1), Jθ(i+1,j+1)
 */
inline void vdep_weights_0( const vsegment & s, const vfloat q, vfloat w[8] )
{
    const vfloat c_1_2 = vec_float( 0.5f );

    const vfloat S0z0 = vec_sub( c_1_2, s.x0.x );
    const vfloat S0z1 = vec_add( c_1_2, s.x0.x );
    const vfloat S1z0 = vec_sub( c_1_2, s.x1.x );
    const vfloat S1z1 = vec_add( c_1_2, s.x1.x );

    const vfloat S0r0 = vec_sub( c_1_2, s.x0.y );
    const vfloat S0r1 = vec_add( c_1_2, s.x0.y );
    const vfloat S1r0 = vec_sub( c_1_2, s.x1.y );
    const vfloat S1r1 = vec_add( c_1_2, s.x1.y );

    const vfloat wl1 = vec_mul( q, vec_sub( s.x1.x, s.x0.x ) );
    const vfloat wl2 = vec_mul( q, vec_sub( s.x1.y, s.x0.y ) );

    const vfloat wp10 = vec_mul( c_1_2, vec_add( S0r0, S1r0 ) );
    const vfloat wp11 = vec_mul( c_1_2, vec_add( S0r1, S1r1 ) );
    const vfloat wp20 = vec_mul( c_1_2, vec_add( S0z0, S1z0 ) );
    const vfloat wp21 = vec_mul( c_1_2, vec_add( S0z1, S1z1 ) );

    const vfloat xif = vec_add( s.t0.x, s.t1.x );
    const vfloat yif = vec_add( s.t0.y, s.t1.y );
    const vfloat Δx  = vec_sub( s.t1.x, s.t0.x );
    const vfloat Δy  = vec_sub( s.t1.y, s.t0.y );

    // Azimuthal displacement times charge, including the 1/3 normalization
    const vfloat jθ = vec_div( 
        vec_mul( q, vec_fmadd( vec_neg( Δx ), yif, vec_mul( Δy, xif ) ) ),
        vec_mul( vec_float( 3.0f ), vec_sqrt( vec_fmadd( xif, xif, vec_mul( yif, yif ) ) ) ) );

    w[0] = vec_mul( wl1, wp10 );
    w[1] = vec_mul( wl1, wp11 );
    w[2] = vec_mul( wl2, wp20 );
    w[3] = vec_mul( wl2, wp21 );

    // Jθ weights, time average of Sz(t) Sr(t) (the 1/3 factor is in jθ)
    auto wth = [&]( vfloat S0z, vfloat S1z, vfloat S0r, vfloat S1r ) {
        return vec_mul( jθ, vec_add( vec_fmadd( S0z, S0r, vec_mul( S1z, S1r ) ),
            vec_mul( vec_fmadd( S0z, S1r, vec_mul( S1z, S0r ) ), c_1_2 ) ) );
    };
    w[4] = wth( S0z0, S1z0, S0r0, S1r0 );
    w[5] = wth( S0z1, S1z1, S0r0, S1r0 );
    w[6] = wth( S0z0, S1z0, S0r1, S1r1 );
    w[7] = wth( S0z1, S1z1, S0r1, S1r1 );
}

/**
 * @brief Mode n current weights of a segment (SIMD version of
 *.       dep_current_seg_n(), weights only)
 * 
 * @param s         (simd vector) Segment
 * @param e0        (simd vector) for the initial angle
 * @param e1        (simd vector) for the final angle
 * @param em        (simd vector) for the mid point angle
 * @param q         (simd vector) Particle charge
 * @param w[out]    (simd vector) Weights: Jz(i,j), Jz(i,j+1), Jr(i,j),
 *                  Jr(i+1,j), Jθ(i,j), Jθ(i+1,j), Jθ(i,j+1), Jθ(i+1,j+1)
 */
inline void vdep_weights_n( const vsegment & s, const vcomplex e0, const vcomplex e1, const vcomplex em, const vfloat q, vcomplex w[8] )
{
        const vfloat c_1_2 = vec_float( 0.5f );

        const vfloat S0z0 = vec_sub( c_1_2, s.x0.x );
        const vfloat S0z1 = vec_add( c_1_2, s.x0.x );
        const vfloat S1z0 = vec_sub( c_1_2, s.x1.x );
        const vfloat S1z1 = vec_add( c_1_2, s.x1.x );
        const vfloat S0r0 = vec_sub( c_1_2, s.x0.y );
        const vfloat S0r1 = vec_add( c_1_2, s.x0.y );
        const vfloat S1r0 = vec_sub( c_1_2, s.x1.y );
        const vfloat S1r1 = vec_add( c_1_2, s.x1.y );

        const vfloat dz = vec_sub( s.x1.x, s.x0.x );
        const vfloat dr = vec_sub( s.x1.y, s.x0.y );

        const vfloat wp10 = vec_mul( c_1_2, vec_add( S0r0, S1r0 ) );
        const vfloat wp11 = vec_mul( c_1_2, vec_add( S0r1, S1r1 ) );
        const vfloat wp20 = vec_mul( c_1_2, vec_add( S0z0, S1z0 ) );
        const vfloat wp21 = vec_mul( c_1_2, vec_add( S0z1, S1z1 ) );

        const vcomplex c0 = vec_sub( e0, em );
        const vcomplex c1 = vec_sub( e1, em );

        const vcomplex qwl1 = vec_mul( q, vec_mul( em, dz ) );
        const vcomplex qwl2 = vec_mul( q, vec_mul( em, dr ) );

        w[0] = vec_mul( qwl1, wp10 );
        w[1] = vec_mul( qwl1, wp11 );
        w[2] = vec_mul( qwl2, wp20 );
        w[3] = vec_mul( qwl2, wp21 );

        auto wth = [&]( vfloat S0z, vfloat S1z, vfloat S0r, vfloat S1r ) {
            return vec_mul( q, vec_sub( vec_mul( c1, vec_mul( S1z, S1r ) ), vec_mul( c0, vec_mul( S0z, S0r ) ) ) );
        };
        w[4] = wth( S0z0, S1z0, S0r0, S1r0 );
        w[5] = wth( S0z1, S1z1, S0r0, S1r0 );
        w[6] = wth( S0z0, S1z0, S0r1, S1r1 );
        w[7] = wth( S0z1, S1z1, S0r1, S1r1 );
}

/**
 * @brief Deposit the mode 0 current of a segment, masked SIMD version
 *
 * Same strategy as vdep_current_seg() in mpi/em2d: the current weights are
 * computed with SIMD instructions (vdep_weights_0()), only for the lanes
 * selected by the mask. The weights, cell indices and mask are stored to
 * small local (aligned) arrays, and then added to the tile current one
 * particle (lane) at a time.
 *
 * @param s         (simd vector) Segment
 * @param q         (simd vector) Particle charge
 * @param J         Mode 0 current grid, pointing at cell (0,0)
 * @param stride    Current grid y stride
 * @param mask      (simd vector) Particles to deposit
 */
inline void vdep_current_seg_0( const vsegment & s, const vfloat q,
    cyl3<float> * __restrict__ J, const int stride, const vmask mask )
{
    if ( vec_any( mask ) ) {
        constexpr int W = simd::width;

        alignas(local_align) float w_s[8][W];
        alignas(local_align) int   idx_s[W];
        alignas(local_align) int   act_s[W];

        {
            vfloat w[8];
            vdep_weights_0( s, q, w );
            for( int c = 0; c < 8; c++ ) vec_store( w_s[c], w[c] );
        }
        vec_store( idx_s, vec_add( s.ix.x, vec_mul( s.ix.y, stride ) ) );
        vec_store( act_s, vec_select( vec_int( 0 ), vec_int( 1 ), mask ) );

        for( int k = 0; k < W; k++ ) {
            // Only deposit particles with active mask
            if ( act_s[k] ) {
                cyl3<float> * __restrict__ const Js = & J[ idx_s[k] ];
                Js[ 0          ].z  += w_s[0][k];
                Js[ stride     ].z  += w_s[1][k];
                Js[ 0          ].r  += w_s[2][k];
                Js[ 1          ].r  += w_s[3][k];
                Js[ 0          ].th += w_s[4][k];
                Js[ 1          ].th += w_s[5][k];
                Js[ stride     ].th += w_s[6][k];
                Js[ stride + 1 ].th += w_s[7][k];
            }
        }
    }
}

/**
 * @brief Deposit the current of a segment for all modes m > 0, masked SIMD
 *        version
 *
 * For each mode the current weights are computed with SIMD instructions
 * (vseg_m), stored to small local (aligned) arrays, and then added to the
 * tile current one particle (lane) at a time, only for the lanes selected by
 * the mask (same strategy as vdep_current_seg() in mpi/em2d).
 *
 * @param nmodes    Number of modes (including mode 0)
 * @param s         (simd vector) Segment
 * @param q         (simd vector) Particle charge
 * @param J         Mode 1 current grid, pointing at cell (0,0); mode m is at
 *                  J + (m-1) * mstride
 * @param stride    Current grid y stride
 * @param mstride   Distance between the current grids of consecutive modes
 * @param mask      (simd vector) Particles to deposit
 */
inline void vdep_current_seg_n( const int nmodes, const vsegment & s, const vfloat q,
    cyl3<complex_t> * __restrict__ J, const int stride, const int mstride, const vmask mask )
{
    if ( vec_any( mask ) ) {
        constexpr int W = simd::width;

        // Memory buffers for accessing vector values
        alignas(local_align) float w_re[8][W];
        alignas(local_align) float w_im[8][W];
        alignas(local_align) int   idx[W];
        alignas(local_align) int   deposit[W];

        vec_store( idx, vec_add( s.ix.x, vec_mul( s.ix.y, stride ) ) );
        vec_store( deposit, vec_select( vec_int( 0 ), vec_int( 1 ), mask ) );

        const vfloat c_1   = vec_float( 1.0f );

        const vfloat rcr0 = vec_div( c_1, vec_sqrt( vec_fmadd( s.t0.x, s.t0.x, vec_mul( s.t0.y, s.t0.y ) ) ) );
        const vfloat rcr1 = vec_div( c_1, vec_sqrt( vec_fmadd( s.t1.x, s.t1.x, vec_mul( s.t1.y, s.t1.y ) ) ) );

        const vfloat xif = vec_add( s.t0.x, s.t1.x );
        const vfloat yif = vec_add( s.t0.y, s.t1.y );
        const vfloat rrm2 = vec_div( c_1, vec_sqrt( vec_fmadd( xif, xif, vec_mul( yif, yif ) ) ) );

        /// @brief exp( -i θ ) for the initial, final and mid point angles
        const vcomplex e0_1 = vec_complex( vec_mul( s.t0.x, rcr0 ), vec_neg( vec_mul( s.t0.y, rcr0 ) ) );
        const vcomplex e1_1 = vec_complex( vec_mul( s.t1.x, rcr1 ), vec_neg( vec_mul( s.t1.y, rcr1 ) ) );
        const vcomplex em_1 = vec_complex( vec_mul( xif, rrm2 ),    vec_neg( vec_mul( yif, rrm2 ) ) );

        vcomplex e0 = e0_1;
        vcomplex e1 = e1_1;
        vcomplex em = em_1;

        for( int m = 1; m < nmodes; m++ ) {
            
            /// @brief exp( -i m θ ) for the initial, final and mid point angles
            if ( m > 1 ) {
                e0 = vec_mul( e0, e0_1 );
                e1 = vec_mul( e1, e1_1 );
                em = vec_mul( em, em_1 );
            }

            {   // Get deposition weights for all 8 components
                vcomplex w[8];
                vdep_weights_n(s, e0, e1, em, q, w );
                
                // Store them in memory
                #pragma unroll 8
                for( int c = 0; c < 8; c++ ) {
                    vec_store( w_re[c], w[c].re );
                    vec_store( w_im[c], w[c].im );
                }
            }

            cyl3<complex_t> * __restrict__ const Jm = J + ( m - 1 ) * mstride;

            for( int k = 0; k < W; k++ ) {
                // Only deposit particles with active mask
                if ( deposit[k] ) {
                    cyl3<complex_t> * __restrict__ const Js = & Jm[ idx[k] ];
                    Js[ 0          ].z  += complex_t( w_re[0][k], w_im[0][k] );
                    Js[ stride     ].z  += complex_t( w_re[1][k], w_im[1][k] );
                    Js[ 0          ].r  += complex_t( w_re[2][k], w_im[2][k] );
                    Js[ 1          ].r  += complex_t( w_re[3][k], w_im[3][k] );
                    Js[ 0          ].th += complex_t( w_re[4][k], w_im[4][k] );
                    Js[ 1          ].th += complex_t( w_re[5][k], w_im[5][k] );
                    Js[ stride     ].th += complex_t( w_re[6][k], w_im[6][k] );
                    Js[ stride + 1 ].th += complex_t( w_re[7][k], w_im[7][k] );
                }
            }
        }
    }
}

/**
 * @brief Move particles, deposit current (all modes) and shift positions,
 *        SIMD version
 * 
 * Same algorithm as move_deposit_0() / move_deposit_n(), processing
 * simd::width particles at a time: the particle motion, the trajectory
 * split and the current weights (all modes) are computed with SIMD
 * instructions. The current is deposited segment by segment
 * (vdep_current_seg_0() / vdep_current_seg_n(), same strategy as mpi/em2d),
 * one particle at a time for the particles using each segment, since
 * different particles may update the same cells. Remaining particles are
 * processed with the scalar code.
 * 
 * @param tile_idx          Tile index
 * @param part              Particle data
 * @param current_m0        Current grid (m=0)
 * @param current_m         Current grids (m > 0), indexed by m (entry 0 unused)
 * @param nmodes            Number of modes (including mode 0)
 * @param dt_dx             Ratio between time step and cell size
 * @param shift             Cell shift
 * @param buffer_m0         Scratch buffer for the mode 0 tile current,
 *                          current_m0.tile_vol elements (one per thread)
 * @param buffer_m          Scratch buffer for the tile current of the modes
 *                          m > 0, ( nmodes - 1 ) * tile_vol elements (one per
 *                          thread)
 */
void vmove_deposit_n(
    uint2 const tile_idx,
    cyl_part::particles_view const part,
    grid::tiled_cyl3_view<float> current_m0, 
    grid::tiled_cyl3_view<complex_t> const * const current_m, 
    const int nmodes,
    float2 const dt_dx, const int2 shift,
    cyl3<float> * const __restrict__ buffer_m0,
    cyl3<complex_t> * const __restrict__ buffer_m )
{
    const int tile_id   = tile_idx.y * part.local_ntiles.x + tile_idx.x;
    const int tile_np   = part.tile_np[ tile_id ];

    if ( tile_np == 0 ) return;

    const int tile_vol = current_m0.tile_vol;

    // Zero local current buffers
    memory::zero( buffer_m0, tile_vol );
    if ( nmodes > 1 ) memory::zero( buffer_m, ( nmodes - 1 ) * tile_vol );

    cyl3<float>     * const J0 = & buffer_m0[ current_m0.inner_offset ];
    cyl3<complex_t> * const Jm = & buffer_m[ current_m0.inner_offset ];
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

    constexpr int W = simd::width;


    const vfloat vdt_dz = vec_float( dt_dz );
    const vfloat vdt_dr = vec_float( dt_dr );
    const vint   vshift_x = vec_int( shift.x );
    const vint   vshift_y = vec_int( shift.y );

    const int np_vec = ( tile_np / W ) * W;

    for( int i = 0; i < np_vec; i += W ) {
        const vfloat3 pu  = vec_load_s3( (float *) & u[i] );
        const vfloat2 x0  = vec_load_s2( (float *) & x[i] );
        const vint2   ix0 = vec_load_s2( (int *)   & ix[i] );
        const vfloat2 θi  = vec_load_s2( (float *) & th[i] );
        const vfloat  pq  = vec_loadu( & q[i] );

        // Get 1 / Lorentz gamma
        const vfloat rg = rgamma( pu );

        // Cartesian motion
        const vfloat Δx = vec_mul( vec_mul( vdt_dr, rg ), pu.x );
        const vfloat Δy = vec_mul( vec_mul( vdt_dr, rg ), pu.y );

        /// @brief initial radial position
        const vfloat ri = vec_add( vec_float( vec_add( vec_int( ir0 ), ix0.y ) ), x0.y );
        const vfloat xi = vec_mul( ri, θi.x );
        const vfloat yi = vec_mul( ri, θi.y );

        // New cartesian positions
        const vfloat xf = vec_fmadd( ri, θi.x, Δx );
        const vfloat yf = vec_fmadd( ri, θi.y, Δy );

        const vfloat xif = vec_fmadd( ri, θi.x, xf );
        const vfloat yif = vec_fmadd( ri, θi.y, yf );

        /// @brief New radial position
        const vfloat rf = vec_sqrt( vec_fmadd( xf, xf, vec_mul( yf, yf ) ) );
        /// @brief radial motion
        const vfloat Δr = vec_div( vec_fmadd( Δx, xif, vec_mul( Δy, yif ) ), vec_add( rf, ri ) );

        // Advance grid (z,r) position
        const vfloat Δz = vec_mul( vec_mul( vdt_dz, rg ), pu.z );

        // Split trajectory
        vint2 deltai;
        vsegment seg0, seg1, seg2;
        vmask cross_z, cross_r;
        vsplit2d_cyl( ir0, ix0, x0, vfloat2{ Δz, Δr }, deltai, vfloat2{ xi, yi }, vfloat2{ Δx, Δy },
            seg0, seg1, seg2, cross_z, cross_r );

        // Masked deposit (as in mpi/em2d): segment by segment, from the vector
        // registers, lanes selected by the segment masks
        {
            const vmask use_seg1 = vec_or( cross_z, cross_r );
            const vmask use_seg2 = vec_and( cross_z, cross_r );

            vdep_current_seg_0( seg0, pq, J0, jstride, vec_true() ); // all lanes
            vdep_current_seg_0( seg1, pq, J0, jstride, use_seg1 );
            vdep_current_seg_0( seg2, pq, J0, jstride, use_seg2 );

            if ( nmodes > 1 ) {
                vdep_current_seg_n( nmodes, seg0, pq, Jm, jstride, tile_vol, vec_true() );
                vdep_current_seg_n( nmodes, seg1, pq, Jm, jstride, tile_vol, use_seg1 );
                vdep_current_seg_n( nmodes, seg2, pq, Jm, jstride, tile_vol, use_seg2 );
            }
        }

        // Modify cell position and store
        const vfloat2 x1{ 
            vec_sub( vec_add( x0.x, Δz ), vec_float( deltai.x ) ),
            vec_sub( vec_add( x0.y, Δr ), vec_float( deltai.y ) )
        };
        vec_store_s2( (float *) & x[i], x1 );

        // Modify cell and store
        const vint2 ix1{
            vec_add( vec_add( ix0.x, deltai.x ), vshift_x ),
            vec_add( vec_add( ix0.y, deltai.y ), vshift_y )
        };
        vec_store_s2( (int *) & ix[i], ix1 );

        // Store new angular position
        const vfloat2 θf{ vec_div( xf, rf ), vec_div( yf, rf ) };
        vec_store_s2( (float *) & th[i], θf );
    }

    // Remaining particles (scalar code, see move_deposit_n())
    for( int i = np_vec; i < tile_np; i++ ) {
        auto pu  = u[i];
        auto x0  = x[i];
        auto ix0 = ix[i];
        auto θi  = th[i];
        auto pq  = q[i];

        float const rg = rgamma( pu );

        auto Δx = dt_dr * rg * pu.x;
        auto Δy = dt_dr * rg * pu.y;

        auto ri = (ir0 + ix0.y) + x0.y;
        auto xi = ri * θi.x;
        auto yi = ri * θi.y;

        auto xf = ops::fma( ri, θi.x, Δx );
        auto yf = ops::fma( ri, θi.y, Δy );

        auto xif = ops::fma( ri, θi.x, xf );
        auto yif = ops::fma( ri, θi.y, yf );

        auto rf = std::sqrt( ops::fma( xf, xf, yf*yf ) );
        auto Δr = ops::fma( Δx , xif , Δy * yif ) / (rf + ri);

        auto Δz = dt_dz * rg * pu.z;
        float2 delta = make_float2( Δz, Δr );

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
        
                                  dep_current_seg_0( v0_ix, v0_x0, v0_x1, v0_t0, v0_t1, pq, J0, jstride );
        if ( cross.x || cross.y ) dep_current_seg_0( v1_ix, v1_x0, v1_x1, v1_t0, v1_t1, pq, J0, jstride );
        if ( cross.x && cross.y ) dep_current_seg_0( v2_ix, v2_x0, v2_x1, v2_t0, v2_t1, pq, J0, jstride );

        if ( nmodes > 1 ) {
                                      dep_current_seg_n( nmodes, v0_ix, v0_x0, v0_x1, v0_t0, v0_t1, pq, Jm, jstride, tile_vol );
            if ( cross.x || cross.y ) dep_current_seg_n( nmodes, v1_ix, v1_x0, v1_x1, v1_t0, v1_t1, pq, Jm, jstride, tile_vol );
            if ( cross.x && cross.y ) dep_current_seg_n( nmodes, v2_ix, v2_x0, v2_x1, v2_t0, v2_t1, pq, Jm, jstride, tile_vol );
        }

        x[i] = make_float2( 
            (x0.x + Δz ) - deltai.x,
            (x0.y + Δr ) - deltai.y
        );

        ix[i] = make_int2(
            ix0.x + deltai.x + shift.x,
            ix0.y + deltai.y + shift.y
        );

        th[i] = float2{ xf/rf, yf/rf };
    }

    // Add current to global buffers
    {
        float * dev_J = reinterpret_cast< float * > ( current_m0.tile_buffer(tile_id) );
        float * loc_J = reinterpret_cast< float * > ( buffer_m0 );
        for( int k = 0; k < 3 * tile_vol; k++ ) dev_J[ k ] += loc_J[ k ];
    }

    for( int m = 1; m < nmodes; m++ ) {
        float * dev_J = reinterpret_cast< float * > ( current_m[m].tile_buffer(tile_id) );
        float * loc_J = reinterpret_cast< float * > ( & buffer_m[ ( m - 1 ) * tile_vol ] );
        for( int k = 0; k < 3 * 2 * tile_vol; k++ ) dev_J[ k ] += loc_J[ k ];
    }
}

#endif // SIMD

/**
 * @brief Physical boundary conditions for the x direction 
 * 
 * @param tile_idx  Tile index
 * @param part      Particle data
 * @param bc        Boundary condition
 */
void species_bcz(
    const int bnd_x, const int tile_idx_y,
    cyl_part::particles_view const part,
    species::bc_type const local_bc ) 
{
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
            for( int i = 0; i < np; i++ ) {
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
            for( int i = 0; i < np; i++ ) {
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

/**
 * @brief Physical boundary conditions for the r direction 
 * 
 * @param part      Particle data
 * @param bc        Boundary condition
 */
void species_bcr(
    const int tile_idx_x, const int bnd_y,
    cyl_part::particles_view const part,
    species::bc_type const local_bc ) 
{
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
            for( int i = 0; i < np; i++ ) {
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
void dep_charge_0(
    const uint2 tile_idx,
    cyl_part::particles_view const part,
    grid::tiled_view<float> charge_m0 )
{
    const int tile_id   = tile_idx.y * part.local_ntiles.x + tile_idx.x;
    const int tile_np   = part.tile_np[ tile_id ];

    // If the block does not have any particles to process return
    if ( tile_np == 0 ) return;

    /// local copy of charge density
    float charge_local[charge_m0.tile_vol];

    // Zero shared memory and sync.
    memory::zero( charge_local, charge_m0.tile_vol );

    float *charge = &charge_local[ charge_m0.inner_offset ];

    const int part_off = part.tile_offset[ tile_id ];
    auto const * __restrict__ const ix = &part.ix[ part_off ];
    auto const * __restrict__ const x  = &part.x[ part_off ];
    auto const * __restrict__ const q  = &part.q[ part_off ];
    const int jstride = charge_m0.tile_ystride(); 

    for( auto i = 0; i < tile_np; i++ ) {
        const float s0z = 0.5f - x[i].x;
        const float s1z = 0.5f + x[i].x;
        const float s0r = 0.5f - x[i].y;
        const float s1r = 0.5f + x[i].y;

        const int idx = ix[i].y * jstride + ix[i].x;

        charge[ idx               ] += s0r * s0z * q[i];
        charge[ idx + 1           ] += s0r * s1z * q[i];
        charge[ idx     + jstride ] += s1r * s0z * q[i];
        charge[ idx + 1 + jstride ] += s1r * s1z * q[i];
    }

    // Add charge to global buffers
    float * dev_rho = reinterpret_cast< float * > ( charge_m0.tile_buffer(tile_id));
    float * shm_rho = reinterpret_cast< float * > ( charge );
    for( auto i =  0; i < charge_m0.tile_vol; i++ ) {
        dev_rho[ i ] += shm_rho[ i ]; 
    }
}

/**
 * @brief kernel for depositing mode m>0 charge
 * 
 * @param m         Mode number (m >= 1)
 * @param d_charge  Charge density grid (will be zeroed by this kernel)
 * @param offset    Offset to position (0,0) of grid
 * @param ext_nx    External tile size (i.e. including guard cells)
 * @param d_tile    Particle tiles information
 * @param d_ix      Particle buffer (cells)
 * @param d_x       Particle buffer (position)
 * @param q         Species charge per particle
 */
void dep_charge(
    const uint2 tile_idx,
    const int m,
    cyl_part::particles_view const part,
    grid::tiled_view<complex_t> charge_m )
{
    const int tile_id   = tile_idx.y * part.local_ntiles.x + tile_idx.x;
    const int tile_np   = part.tile_np[ tile_id ];

    // If the block does not have any particles to process return
    if ( tile_np == 0 ) return;

    /// local copy of charge density
    complex_t charge_local_m[charge_m.tile_vol];

    // Zero shared memory
    memory::zero( charge_local_m, charge_m.tile_vol );

    auto *charge = &charge_local_m[ charge_m.inner_offset ];

    const int part_off = part.tile_offset[ tile_id ];
    auto const * __restrict__ const ix = &part.ix[ part_off ];
    auto const * __restrict__ const x  = &part.x[ part_off ];
    auto const * __restrict__ const q  = &part.q[ part_off ];
    auto const * __restrict__ const th = &part.th[ part_off ];
    const int jstride = charge_m.tile_ystride(); 

    for( auto i = 0; i < tile_np; i++ ) {
        const float s0z = 0.5f - x[i].x;
        const float s1z = 0.5f + x[i].x;
        const float s0r = 0.5f - x[i].y;
        const float s1r = 0.5f + x[i].y;

        const int idx = ix[i].y * jstride + ix[i].x;

        // Same angular factor as the current deposit (charge conservation)
        auto qm = q[i] * expmimt( m, th[i] );

        charge[ idx               ] += s0r * s0z * qm;
        charge[ idx + 1           ] += s0r * s1z * qm;
        charge[ idx     + jstride ] += s1r * s0z * qm;
        charge[ idx + 1 + jstride ] += s1r * s1z * qm;
    }

    // Copy data to global memory
    float * dev_rho = reinterpret_cast< float * > ( charge_m.tile_buffer(tile_id));
    float * shm_rho = reinterpret_cast< float * > ( charge );

    for( auto i =  0; i < 2 * charge_m.tile_vol; i++ ) {
        dev_rho[ i ] += shm_rho[ i ]; 
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
void charge_norm(
    const uint2 tile_idx,
    grid::tiled_view<T> charge_grid, const float dr, 
    const float scale = 1.0f,
    const float parity = 1.0f
) {
    const int jstride = charge_grid.tile_ystride();

    auto * __restrict__ charge = charge_grid.tile_data( tile_idx );
    auto tile_dims = charge_grid.tile_dims;

    int ir0 = ( charge_grid.local_tile_start.y + tile_idx.y ) * tile_dims.y;
    for( int k = 0; k < (tile_dims.y+1)*(tile_dims.x+1); k ++ ) {
        int j = k / (tile_dims.x+1);
        int i = k % (tile_dims.x+1);
        auto norm = scale/(std::abs( ir0 + j - 0.5f) * dr);
        charge[ j * jstride +i ] *= norm;
    }

    // Axial boundary
    // Fold values for r < 0 back into simulation domain, row 0 is at
    // r = -Δr/2 and mirrors row 1 with the mode parity
    if ( ir0 == 0 ) {
        for( int i = 0; i < static_cast<int>(tile_dims.x+1); i++ ){
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
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for( unsigned ty = 0; ty < particles -> get_local_ntiles().y; ty ++ ) {
            for( unsigned bnd_x : { 0, 1 } ) 
                kernel::species_bcz ( bnd_x, ty, particles -> view(), local_bc );
        }
    }

    // r boundaries
    if ( local_bc.y.upper == species::bc::reflecting ) {
        const unsigned bnd_y = 1;

        #pragma omp parallel for schedule(dynamic)
        for( unsigned tx = 0; tx < particles -> get_local_ntiles().x; tx ++ )
            kernel::species_bcr ( bnd_y, tx, particles -> view(), local_bc );
    }
}

/**
 * @brief Move particles (advance positions), deposit current and shift positions
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

    auto data = particles -> view();
    auto ntiles = data.local_ntiles;
    uint64_t _nmove = 0;

#if defined( SIMD ) && ! Q3D_ORIGINAL_SPLIT
    {
        // SIMD version, any number of modes
        std::vector< grid::tiled_cyl3_view<complex_t> > Jm;
        if ( nmodes > 1 ) {
            Jm.reserve( nmodes );
            Jm.push_back( current.mode(1).view() );     // entry 0 is unused
            for( int m = 1; m < nmodes; m++ ) Jm.push_back( current.mode(m).view() );
        }

        auto const tile_vol = J0.tile_vol;

        #pragma omp parallel reduction(+:_nmove)
        {
            // Per thread scratch buffers for the tile current
            std::vector< cyl3<float> >     buffer_m0( tile_vol );
            std::vector< cyl3<complex_t> > buffer_m( ( nmodes - 1 ) * tile_vol );

            #pragma omp for collapse(2) schedule( dynamic )
            for( int ty = 0; ty < ntiles.y; ty ++ ) {
                for( int tx = 0; tx < ntiles.x; tx ++ ) {
                    kernel::vmove_deposit_n( 
                        make_uint2( tx, ty ),
                        data, J0.view(), Jm.data(), nmodes,
                        dt_dx, shift, buffer_m0.data(), buffer_m.data() );
                    _nmove += data.tile_np[ ty * ntiles.x + tx ];
                }
            }
        }
    }
#else
    switch (nmodes) {
    case 1: {
        #pragma omp parallel for collapse(2) schedule( dynamic ) reduction(+:_nmove)
        for( int ty = 0; ty < ntiles.y; ty ++ ) {
            for( int tx = 0; tx < ntiles.x; tx ++ ) {
                kernel::move_deposit_0( 
                    make_uint2( tx, ty ),
                    data, J0.view(),
                    dt_dx, shift );
                _nmove += data.tile_np[ ty * ntiles.x + tx ];
            }
        }
    } break;

    // Any number of modes (single pass)
    default: {
        // Single pass: move particles and deposit the current of all modes
        std::vector< grid::tiled_cyl3_view<complex_t> > Jm;
        Jm.reserve( nmodes );
        Jm.push_back( current.mode(1).view() );     // entry 0 is unused
        for( int m = 1; m < nmodes; m++ ) Jm.push_back( current.mode(m).view() );

        auto const tile_vol = J0.tile_vol;

        #pragma omp parallel reduction(+:_nmove)
        {
            // Per thread scratch buffers for the tile current
            std::vector< cyl3<float> >     buffer_m0( tile_vol );
            std::vector< cyl3<complex_t> > buffer_m( ( nmodes - 1 ) * tile_vol );

            #pragma omp for collapse(2) schedule( dynamic )
            for( int ty = 0; ty < ntiles.y; ty ++ ) {
                for( int tx = 0; tx < ntiles.x; tx ++ ) {
                    kernel::move_deposit_n( 
                        make_uint2( tx, ty ),
                        data, J0.view(), Jm.data(), nmodes,
                        dt_dx, shift, buffer_m0.data(), buffer_m.data() );
                    _nmove += data.tile_np[ ty * ntiles.x + tx ];
                }
            }
        }
    } break;
    }
#endif

    nmoved += _nmove;
}

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

    auto data = particles -> view();
    auto ntiles = data.local_ntiles;
    uint64_t _nmove = 0;

    #pragma omp parallel for collapse(2) schedule( dynamic ) reduction(+:_nmove)
    for( int ty = 0; ty < ntiles.y; ty ++ ) {
        for( int tx = 0; tx < ntiles.x; tx ++ ) {
            kernel::move( 
                make_uint2( tx, ty ),
                data,
                dt_dx, shift );
            _nmove += data.tile_np[ ty * ntiles.x + tx ];
        }
    }

    nmoved += _nmove;
}

/**
 * @brief       Accelerates particles using a Boris/Euler pusher
 *
 * @note Also computes total time-centerered kinetic energy
 * 
 * @param emf     EMF field
 */
void species::push( cyl_modes::tiled_cyl3<float> & E, cyl_modes::tiled_cyl3<float> & B )
{
    const float alpha = 0.5 * dt / m_q;
    
    double _energy = 0;
    auto ntiles = particles -> get_local_ntiles();

    auto & E0 = E.mode0();
    auto & B0 = B.mode0();

    // Views of the modes m > 0, indexed by m (entry 0 is unused)
    std::vector< grid::tiled_cyl3_view<complex_t> > Em, Bm;
    if ( nmodes > 1 ) {
        Em.reserve( nmodes ); Bm.reserve( nmodes );
        Em.push_back( E.mode(1).view() ); Bm.push_back( B.mode(1).view() );
        for( int m = 1; m < nmodes; m++ ) {
            Em.push_back( E.mode(m).view() );
            Bm.push_back( B.mode(m).view() );
        }
    }

    // Launch the push kernels for the selected pusher type
    auto launch = [&]( auto type_c ) {
        constexpr species::pusher type = decltype( type_c )::value;

#ifdef SIMD
        // SIMD version, any number of modes
        #pragma omp parallel for collapse(2) schedule(dynamic) reduction(+:_energy)
        for( int ty = 0; ty < ntiles.y; ty++ ) {
            for( int tx = 0; tx < ntiles.x; tx++ ) {
                double tile_energy;
                kernel::vpush_all <type> ( make_uint2(tx,ty), 
                    particles -> view(),
                    E0.view(), B0.view(), 
                    Em.data(), Bm.data(), nmodes,
                    alpha, tile_energy
                );
                _energy += tile_energy;
            }
        }
#else
        switch (nmodes) {
        case 1: {
            #pragma omp parallel for collapse(2) schedule(dynamic) reduction(+:_energy)
            for( int ty = 0; ty < ntiles.y; ty++ ) {
                for( int tx = 0; tx < ntiles.x; tx++ ) {
                    double tile_energy;
                    kernel::push_0 <type> ( make_uint2(tx,ty), 
                        particles -> view(), E0.view(), B0.view(), alpha,
                        tile_energy
                    );
                    _energy += tile_energy;
                }
            }
        } break;

        // Any number of modes (single pass)
        default: {
            #pragma omp parallel for collapse(2) schedule(dynamic) reduction(+:_energy)
            for( int ty = 0; ty < ntiles.y; ty++ ) {
                for( int tx = 0; tx < ntiles.x; tx++ ) {
                    double tile_energy;
                    kernel::push_all <type> ( make_uint2(tx,ty), 
                        particles -> view(),
                        E0.view(), B0.view(), 
                        Em.data(), Bm.data(), nmodes,
                        alpha, tile_energy
                    );
                    _energy += tile_energy;
                }
            }
        } break;
        }
#endif
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

    energy = _energy;
}

/**
 * @brief Deposit charge density (mode 0)
 * 
 * @param charge0    Charge density grid
 */
void species::deposit_charge_0( grid::tiled<float> & charge0 ) const {

    auto ntiles = particles -> get_local_ntiles();
  
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for( int ty = 0; ty < ntiles.y; ty++ ) {
        for( int tx = 0; tx < ntiles.x; tx++ ) {
            uint2 tile_idx = make_uint2(tx,ty);
            kernel::dep_charge_0 (
                tile_idx, particles -> view(), charge0.view()
            );
            kernel::charge_norm(
                tile_idx, charge0.view(), dx.y
            );
        }
    }
}

/**
 * @brief Deposit charge density high order modes
 * 
 * @param m         Cylindrical mode to deposit (1 to nmodes-1)
 * @param charge    Charge density grid (complex)
 */
void species::deposit_charge_m( const int m, grid::tiled<complex_t> &charge ) const {

    if ( m < 1 || m >= nmodes ) {
        mpi::fatal( "Invalid mode m = " + std::to_string(m) + ", must be in [1, nmodes[" );
    }

    auto ntiles = particles -> get_local_ntiles();
    auto parity = ( m % 2 ) ? -1.f : 1.f;

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for( int ty = 0; ty < ntiles.y; ty++ ) {
        for( int tx = 0; tx < ntiles.x; tx++ ) {
            uint2 tile_idx = make_uint2(tx,ty);
            kernel::dep_charge (
                tile_idx, m, particles -> view(), charge.view()
            );
            kernel::charge_norm(
                tile_idx, charge.view(), dx.y, 2.f, parity
            );
        }
    }
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
        deposit_charge_0( charge );
        charge.add_from_gc();
        charge.save( info, iter_info, path );
    } else {
        // High-order mode
        grid::tiled<complex_t> charge( particles -> get_global_ntiles(), particles -> get_tile_dims(), gc, particles -> parallel );

        charge.zero();
        deposit_charge_m( m, charge );
        charge.add_from_gc();
        charge.save( info, iter_info, path );
    }
}

/**
 * @brief Gather particle data, scaling values
 * 
 * @warning This expects the particle buffer to be compact, it will fail if 
 *          called with a non-compact buffer.
 *
 * @note Data (val) will be returned as `scale.x * val + scale.y`
 * 
 * @tparam quant    Quantity to gather
 * @param part      Particle data
 * @param d_data    Scaled output data
 * @param scale     Scale factor for data
 */
template < cyl_part::quantity quant >
void gather_quant( 
    cyl_part::particles_view src,
    float * const __restrict__ d_data,
    const float2 scale )
{
    const int2 ntiles = make_int2( src.local_ntiles.x, src.local_ntiles.y );

    #pragma omp parallel for schedule(dynamic)
    for( auto tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

        auto tx = tid % ntiles.x;
        auto ty = tid / ntiles.x;

        // Global spatial offsets of local tile
        const int offz = (src.local_tile_start.x + tx) * src.tile_dims.x;
        const int offr = (src.local_tile_start.y + ty) * src.tile_dims.y;

        const auto offset = src.tile_offset[tid];
        const auto np     = src.tile_np[tid];

        int2   const * __restrict__ const ix = & src.ix[ offset ];
        float2 const * __restrict__ const x  = & src.x [ offset ];
        float3 const * __restrict__ const u  = & src.u [ offset ];
        float  const * __restrict__ const q  = & src.q [ offset ];
        float2 const * __restrict__ const th = & src.th[ offset ];
        
        for( int idx = 0; idx < np; idx ++ ) {
            float val;
            if constexpr( quant == cyl_part::quantity::z  ) val = ( offz + ix[idx].x ) + (0.5f + x[idx].x);
            if constexpr( quant == cyl_part::quantity::r  ) val = ( offr + ix[idx].y ) + ( x[idx].y);
            if constexpr( quant == cyl_part::quantity::q      ) val = q[idx];
            if constexpr( quant == cyl_part::quantity::cos_th ) val = th[idx].x;
            if constexpr( quant == cyl_part::quantity::sin_th ) val = th[idx].y;
            if constexpr( quant == cyl_part::quantity::ux ) val = u[idx].x;
            if constexpr( quant == cyl_part::quantity::uy ) val = u[idx].y;
            if constexpr( quant == cyl_part::quantity::uz ) val = u[idx].z;
            d_data[ offset + idx ] = ops::fma( scale.x, val, scale.y );
        }
    }
};

/**
 * @brief Gather data from a specific particle quantity in a device buffer, scaling values
 * 
 * @warning This expects the particle buffer to be compact, it will fail if 
 *          called with a non-compact buffer.
 *
 * @param quant     Quantity to gather
 * @param d_data    Output data buffer, assumed to have size >= np
 * @param scale     Scale factor for data
 */
void species::gather( cyl_part::quantity quant, float * const __restrict__ data, 
    const float2 scale ) const
{
    auto view = particles -> view();

    switch (quant) {
    case cyl_part::quantity::z : 
        gather_quant<cyl_part::quantity::z> ( view, data, scale );
        break; 
    case cyl_part::quantity::r: 
        gather_quant<cyl_part::quantity::r> ( view, data, scale );
        break; 
    case cyl_part::quantity::q: 
        gather_quant<cyl_part::quantity::q> ( view, data, scale );
        break;
    case cyl_part::quantity::sin_th: 
        gather_quant<cyl_part::quantity::sin_th> ( view, data, scale );
        break;
    case cyl_part::quantity::cos_th: 
        gather_quant<cyl_part::quantity::cos_th> ( view, data, scale );
        break;
    case cyl_part::quantity::ux: 
        gather_quant<cyl_part::quantity::ux>( view, data, scale );
        break; 
    case cyl_part::quantity::uy: 
        gather_quant<cyl_part::quantity::uy>( view, data, scale );
        break; 
    case cyl_part::quantity::uz: 
        gather_quant<cyl_part::quantity::uz>( view, data, scale );
        break;
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
            float * h_data = memory::malloc<float>( local );

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
            gather( cyl_part::quantity::z, h_data, scale );
            zdf::write_cdset( part_file, dsets[0], chunk, file_off );

            scale = make_float2( dx.y, 0 );
            gather( cyl_part::quantity::r, h_data, scale );
            zdf::write_cdset( part_file, dsets[1], chunk, file_off );

            // Remaining quantities are saved "as is"
            for( uint32_t i = 2; i < info.nquants; i++ ) {
                gather( quants[i], h_data );
                zdf::write_cdset( part_file, dsets[i], chunk, file_off );
            }

            // Free temporary memory
            memory::free( h_data );

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
void dep_pha1(
    const int2 tile_idx, 
    float * const __restrict__ d_data, float2 const range, int const size,
    float const norm, 
    cyl_part::particles_view const part )
{
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

    for( int i = 0; i < np; i ++ ) {
        float d = phasespace::value<quant>( i, shiftx, shifty, ix, x, u );

        float n =  (d - range.x ) * pha_rdx - 0.5f;
        int   k = floorf( n );
        float w = n - k;
        float qnorm = std::abs(q[i]) * norm;

        if ((k   >= 0) && (k   < size)) d_data[k  ] += (1-w) * qnorm;
        if ((k+1 >= 0) && (k+1 < size)) d_data[k+1] +=    w  * qnorm;
    }
}

template < phasespace::quantity quant0, phasespace::quantity quant1 >
void dep_pha2(
    const int2 tile_idx,
    float * const __restrict__ d_data, 
    float2 const range0, int const size0,
    float2 const range1, int const size1,
    float const norm, 
    cyl_part::particles_view const part )
{
    static_assert( quant1 > quant0, "quant1 must be > quant0" );

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

    for( int i = 0; i < np; i ++ ) {
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
            d_data[(k1  )*size0 + k0  ] += (1-w0) * (1-w1) * qnorm;
            d_data[(k1  )*size0 + k0+1] +=    w0  * (1-w1) * qnorm;
            d_data[(k1+1)*size0 + k0  ] += (1-w0) *    w1  * qnorm;
            d_data[(k1+1)*size0 + k0+1] +=    w0  *    w1  * qnorm;
        } else {
            // corner cases
            if ((k1   >= 0) && (k1   < size1)) {
                if ((k0   >= 0) && (k0   < size0)) 
                    d_data[(k1  )*size0 + k0  ] += (1-w0) * (1-w1) * qnorm;
                if ((k0+1 >= 0) && (k0+1 < size0)) 
                    d_data[(k1  )*size0 + k0+1] +=    w0  * (1-w1) * qnorm;
            }
            if ((k1+1 >= 0) && (k1+1 < size1)) {
                if ((k0   >= 0) && (k0   < size0)) 
                    d_data[(k1+1)*size0 + k0  ] += (1-w0) *    w1  * qnorm;
                if ((k0+1 >= 0) && (k0+1 < size0)) 
                    d_data[(k1+1)*size0 + k0+1] +=    w0  *    w1  * qnorm;
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
    memory::zero( d_data, size );
    
    float norm = ( dx.x * dx.y ) *
                 size / (range.y - range.x) ;

    auto ntiles = particles -> get_local_ntiles();

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
        
        // The current implementation does not support OpenMP parallelism here
        for( int ty = 0; ty < ntiles.y; ty ++ ) {
            for( int tx = 0; tx < ntiles.x; tx ++ ) {
                phasespace::kernel::dep_pha1<quant> (
                    make_int2( tx, ty ),
                    d_data, range, size, norm, 
                    particles -> view()
                );
            }
        }
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
    memory::zero( d_data, size0 * size1 );

    float norm = ( dx.x * dx.y ) *
                 ( size0 / (range0.y - range0.x) ) *
                 ( size1 / (range1.y - range1.x) );

    auto ntiles = particles -> get_local_ntiles();

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

            // The current implementation does not support OpenMP parallelism here
            for( int ty = 0; ty < ntiles.y; ty ++ ) {
                for( int tx = 0; tx < ntiles.x; tx ++ ) {
                    phasespace::kernel::dep_pha2<quant0,quant1> (
                        make_int2( tx, ty ),
                        d_data, range0, size0, range1, size1, norm, 
                        particles -> view()
                    );
                }
            }
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
    float * h_data = memory::malloc<float>( size );

    dep_phasespace( h_data, quant, range, size );

    // Add contributions from all parallel nodes to root node
    particles -> parallel.reduce( h_data, size, mpi::sum );

    // Save file (data is on root node)
    if ( particles -> parallel.root() ) {
        zdf::save_grid( h_data, info, iter_info, "PHASESPACE/" + name );
    }

    memory::free( h_data );
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
    float * h_data = memory::malloc<float>( size0 * size1 );
    dep_phasespace( h_data, quant0, range0, size0, quant1, range1, size1 );

    // Add contributions from all parallel nodes to root node
    particles -> parallel.reduce( h_data, size0 * size1, mpi::sum );

    // Save file (data is on root node)
    if ( particles -> parallel.root() ) {
        zdf::save_grid( h_data, info, iter_info, "PHASESPACE/" + name );
    }

    memory::free( h_data );
}
