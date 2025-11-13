#include "species.h"
#include <iostream>

// Not implemented yet
// #include "simd/simd.h"

/**
 * @brief Memory alignment of local buffers
 * 
 * @warning Must be >= 32 to avoid some compiler issues at high optimization
 *          levels
 */
constexpr int local_align = 64;

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

    e.θ = ( E[i  +     j *jstride].θ * s0z  + E[i+1  +     j *jstride].θ * s1z ) * s0r +
          ( E[i  + (j +1)*jstride].θ * s0z  + E[i+1  + (j +1)*jstride].θ * s1z ) * s1r;

    // Interpolate B fieldj
    b.z = ( B[i  +     jh*jstride].z * s0z  + B[i+1  +     jh*jstride].z * s1z ) * s0rh +
          ( B[i  + (jh+1)*jstride].z * s0z  + B[i+1  + (jh+1)*jstride].z * s1z ) * s1rh;

    b.r = ( B[ih +      j*jstride].r * s0zh + B[ih+1 +      j*jstride].r * s1zh ) * s0r +
          ( B[ih + (j +1)*jstride].r * s0zh + B[ih+1 + (j +1)*jstride].r * s1zh ) * s1r;

    b.θ = ( B[ih +     jh*jstride].θ * s0zh + B[ih+1 +     jh*jstride].θ * s1zh ) * s0rh +
          ( B[ih + (jh+1)*jstride].θ * s0zh + B[ih+1 + (jh+1)*jstride].θ * s1zh ) * s1rh;
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
        energy += utsq / (gamma + 1.0f);

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
 * This is similar to the dudt_boris method above, but the rotation is done using
 * using an exact Euler-Rodriguez method.2
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
        energy += utsq / (gamma + 1.0f);
        
        // Time centered 2 * \alpha / \gamma
        float const alpha2_gamma = ( alpha * 2 ) / gamma ;

        b.x *= alpha2_gamma;
        b.y *= alpha2_gamma;
        b.z *= alpha2_gamma;
    }

    {
        float const bnorm = std::sqrt(ops::fma( b.x, b.x, ops::fma( b.y, b.y, b.z * b.z ) ));
        float const s = -(( bnorm > 0 ) ? std::sin( bnorm / 2 ) / bnorm : 1 );

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
        float const r33 =   ops::fma(ra,ra,rd*rd)-ops::fma(rb,rb,-rc*rc);

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

    const auto jθ = q * ( ops::fma( - Δx , yif , Δy * xif ) ) / std::sqrt( ops::fma( xif, xif, yif*yif) );

    int i = ix.x;
    int j = ix.y;

    // When using more than 1 thread per tile all of these need to be atomic
    J[ stride * j     + i     ].z += wl1 * wp10;
    J[ stride * (j+1) + i     ].z += wl1 * wp11;

    J[ stride * j     + i     ].r += wl2 * wp20;
    J[ stride * j     + (i+1) ].r += wl2 * wp21;

    J[ stride * j     + i     ].θ += jθ * ( S0z0 * S0r0 + S1z0 * S1r0 + (S0z0 * S1r0 - S1z0 * S0r0)/2.0f );
    J[ stride * j     + (i+1) ].θ += jθ * ( S0z1 * S0r0 + S1z1 * S1r0 + (S0z1 * S1r0 - S1z1 * S0r0)/2.0f );
    J[ stride * (j+1) + i     ].θ += jθ * ( S0z0 * S0r1 + S1z0 * S1r1 + (S0z0 * S1r1 - S1z0 * S0r1)/2.0f );
    J[ stride * (j+1) + (i+1) ].θ += jθ * ( S0z1 * S0r1 + S1z1 * S1r1 + (S0z1 * S1r1 - S1z1 * S0r1)/2.0f );
}

/**
 * @brief Deposit (charge conserving) current for 1 segment inside a cell
 * 
 * @tparam m        Azymuthal mode (> 0)
 * @param ix        Initial position (cell index)
 * @param x0        Initial position (z,r)
 * @param x1        Final position (z,r)
 * @param θ0        Initial angular position (cos,sin)
 * @param θm        Mid-point angular position (cos,sin)
 * @param θ1        Final angular position (cos,sin)
 * @param q         Charge
 * @param vθ        Angular velocity (not momentum)
 * @param J         Current buffer
 * @param stride    j stride for current buffer
 */
template< int m >
inline void dep_current_seg(
    const int ir0, const int2 ix, const float2 x0, const float2 x1,
    const float2 t0, const float2 t1, 
    const float q,
    cyl3<std::complex<float>> * __restrict__ J, const int stride )
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

    const auto cr0 = std::sqrt( ops::fma( t0.x, t0.x, t0.y*t0.y ) );
    const auto cr1 = std::sqrt( ops::fma( t1.x, t1.x, t1.y*t1.y ) );

    const auto θ0 = make_float2( t0.x/cr0, t0.y/cr0 );
    const auto θ1 = make_float2( t1.x/cr1, t1.y/cr1 );

    const auto xif = t0.x + t1.x;
    const auto yif = t0.y + t1.y;

    const auto rm2 = std::sqrt( ops::fma( xif, xif, yif*yif ) );
    const auto θm = float2{ xif/rm2, yif/rm2 };

    // Complex coefficients for initial, mid and final angular positions
    
/*
    auto cm = expimθ<m>( θm );
    auto c0 = expimθ<m>( θ0 ) - cm;
    auto c1 = expimθ<m>( θ1 ) - cm;
*/

    static_assert( m == 1, "only mode m = 1 is currently supported" );
    auto cm = std::complex<float>{ θm.x, -θm.y };
    auto c0 = std::complex<float>{ θ0.x, -θ0.y } - cm;
    auto c1 = std::complex<float>{ θ1.x, -θ1.y } - cm;


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

    // When using more than 1 thread per tile all of these need to be atomic
    J[ stride * j     + i     ].z += q * wl1 * wp10;
    J[ stride * (j+1) + i     ].z += q * wl1 * wp11;

    J[ stride * j     + i     ].r += q * wl2 * wp20;
    J[ stride * j     + (i+1) ].r += q * wl2 * wp21;

    J[ stride * j     + i     ].θ += q * ( S0z1 * S0r1 * c1 - S0z0 * S0r0 * c0 );
    J[ stride * j     + (i+1) ].θ += q * ( S1z1 * S0r1 * c1 - S1z0 * S0r0 * c0 );
    J[ stride * (j+1) + i     ].θ += q * ( S0z1 * S1r1 * c1 - S0z0 * S1r0 * c0 );
    J[ stride * (j+1) + (i+1) ].θ += q * ( S1z1 * S1r1 * c1 - S1z0 * S1r0 * c0 );
}

/**
 * @brief Split particle trajectory into segments fitting in a single cell
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

    // z-splitx
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

#if 0
        // New splitter
        // This has some roundoff issues
        if ( x1.y == 0.5f ) { 
            εr = 1;
        } else {
            auto a = ops::fma( tdelta.x, tdelta.x, tdelta.y * tdelta.y );
            auto b = ops::fma( t0.x, tdelta.x, t0.y * tdelta.y );
            auto c = ( x0.y - rs ) * ( 2 * (ir0 + ix.y) + x0.y + rs );

            εr = - ( b + std::copysign( std::sqrt( ops::fma( b, b, - a*c )), b ) ) / a;
            if ( εr < 0 || εr > 1 ) εr = c / (a * εr);
        }
#else
        // Old splitter (OSIRIS)
        εr = (rs - x0.y) / delta.y;
#endif

/*
        if ( εr < 0 || εr > 1) {
            std::cerr << "(*error*) Invalid εr: "<< εr << '\n';
            std::exit(1);
        }
*/
        // r-split positions
        xr = t0.x + εr * tdelta.x;
        yr = t0.y + εr * tdelta.y;
        zr = x0.x + εr * delta.x;
    }

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
            // std::cout << "(*info*) z_cross 1st " << '\n';

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
            // std::cout << "(*info*) r_cross 1st " << '\n';

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
 * @brief Move particles and deposit current (mode 0 only) and shift positions
 * 
 * @param tile_idx          Tile index
 * @param part              Particle data
 * @param current_m0        Current grid (m=0)
 * @param current_offset    Offset to position [0,0] of the current grid
 * @param nx                Current grid size (internal)
 * @param ext_nx            Current grid size (external)
 * @param dt_dx             Ratio between time step and cell size
 */
void move_deposit_0(
    uint2 const tile_idx,
    ParticleData const part,
    cyl3<float> * const __restrict__ current_m0, 
    unsigned int const current_offset, uint2 const nx, uint2 const ext_nx,
    float2 const dt_dx ) 
{
    const uint2 ntiles  = part.ntiles;
    const int tile_size = roundup4( ext_nx.x * ext_nx.y );

    // This is usually in block shared memory
    alignas(local_align) cyl3<float> tile_buffer_0[tile_size];

    // Zero local current buffer
    for( auto i = 0; i < tile_size; i++ ) 
        tile_buffer_0[i] = cyl3<float>{0};

    // sync

    // Move particles and deposit current
    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    cyl3<float> * J0 = & tile_buffer_0[ current_offset ];
    const int jstride = ext_nx.x;

    const int part_offset    = part.offset[ tid ];
    const int np             = part.np[ tid ];
    auto * __restrict__ ix = &part.ix[ part_offset ];
    auto * __restrict__ x  = &part.x[ part_offset ];
    auto * __restrict__ u  = &part.u[ part_offset ];
    auto * __restrict__ θ  = &part.θ[ part_offset ];
    auto * __restrict__ q  = &part.q[ part_offset ];
    
    auto const dt_dz = dt_dx.x;
    auto const dt_dr = dt_dx.y;

    const int ir0 = tile_idx.y * nx.y;

    for( int i = 0; i < np; i++ ) {
        auto pu  = u[i];
        auto x0  = x[i];
        auto ix0 = ix[i];
        auto θi  = θ[i];
        auto pq  = q[i];

        // Get 1 / Lorentz gamma
        float const rg = rgamma( pu );

        // Cartesian motion
        auto Δx = dt_dr * rg * pu.x;
        auto Δy = dt_dr * rg * pu.y;

        /// @brief initial radial position
        auto ri = ( ir0 + ix0.y ) + x0.y;
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
            ix0.x + deltai.x,
            ix0.y + deltai.y
        );

        /// @brief store new angular position
        θ[i] = float2{ xf/rf, yf/rf };

    }

    // Current normalization is done in Current::normalize()

    // Add current to global buffer
    const int tile_off = tid * tile_size;

    for( unsigned i = 0; i < ext_nx.x * ext_nx.y; i++ ) {
        current_m0[tile_off + i] += tile_buffer_0[i];
    }
}

/**
 * @brief Move particles and deposit current (modes 0 and 1)
 * 
 * @param tile_idx          Tile index
 * @param part              Particle data
 * @param d_current         Current grid (global)
 * @param current_offset    Offset to position [0,0] of the current grid
 * @param nx                Current grid size (internal)
 * @param ext_nx            Current grid size (external)
 * @param dt_dx             Ratio between time step and cell size
 */
void move_deposit_1(
    uint2 const tile_idx,
    ParticleData const part,
    cyl3<float> * const __restrict__ current_m0, 
    cyl3<std::complex<float>> * const __restrict__ current_m1, 
    unsigned int const current_offset, uint2 const nx, uint2 const ext_nx,
    float2 const dt_dx ) 
{
    const uint2 ntiles  = part.ntiles;
    const int tile_size = roundup4( ext_nx.x * ext_nx.y );

    // This is usually in block shared memory
    alignas(local_align) cyl3<float>               tile_buffer_0[tile_size];
    alignas(local_align) cyl3<std::complex<float>> tile_buffer_1[tile_size];

    // Zero local current buffer
    for( auto i = 0; i < tile_size; i++ ) {
        tile_buffer_0[i] = cyl3<float>{0};
        tile_buffer_1[i] = cyl3<std::complex<float>>{0};
    }

    // sync

    // Move particles and deposit current
    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    cyl3<float> *               J0 = & tile_buffer_0[ current_offset ];
    cyl3<std::complex<float>> * J1 = & tile_buffer_1[ current_offset ];

    const int jstride = ext_nx.x;

    const int part_offset    = part.offset[ tid ];
    const int np             = part.np[ tid ];
    auto * __restrict__ ix = &part.ix[ part_offset ];
    auto * __restrict__ x  = &part.x[ part_offset ];
    auto * __restrict__ u  = &part.u[ part_offset ];
    auto * __restrict__ θ  = &part.θ[ part_offset ];
    auto * __restrict__ q  = &part.q[ part_offset ];
    
    auto const dt_dz = dt_dx.x;
    auto const dt_dr = dt_dx.y;

    const int ir0 = tile_idx.y * nx.y;

    for( int i = 0; i < np; i++ ) {
        auto pu  = u[i];
        auto x0  = x[i];
        auto ix0 = ix[i];
        auto θi  = θ[i];
        auto pq  = q[i];

        // Get 1 / Lorentz gamma
        float const rg = rgamma( pu );

        // Cartesian motion
        auto Δx = dt_dr * rg * pu.x;
        auto Δy = dt_dr * rg * pu.y;
        auto Δz = dt_dz * rg * pu.z;

        /// @brief initial radial position
        auto ri = ( ir0 + ix0.y ) + x0.y;
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
            ix0.x + deltai.x,
            ix0.y + deltai.y
        );

        /// @brief store new angular position
        θ[i] = float2{ xf/rf, yf/rf };
    }

    // Current normalization is done in Current::normalize()

    // Add current to global buffer
    const int tile_off = tid * tile_size;

    for( unsigned i = 0; i < ext_nx.x * ext_nx.y; i++ ) {
        current_m0[tile_off + i] += tile_buffer_0[i];
        current_m1[tile_off + i] += tile_buffer_1[i];
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
    ParticleData const part,
    cyl3<float> * const __restrict__ current_m0, 
    unsigned int const current_offset, uint2 const nx, uint2 const ext_nx,
    float2 const dt_dx, const int2 shift ) 
{
    const uint2 ntiles  = part.ntiles;
    const int tile_size = roundup4( ext_nx.x * ext_nx.y );

    // This is usually in block shared memory
    alignas(local_align) cyl3<float> tile_buffer_0[tile_size];

    // Zero local current buffer
    for( auto i = 0; i < tile_size; i++ ) 
        tile_buffer_0[i] = cyl3<float>{0};

    // sync

    // Move particles and deposit current
    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    cyl3<float> * J0 = & tile_buffer_0[ current_offset ];
    const int jstride = ext_nx.x;

    const int part_offset    = part.offset[ tid ];
    const int np             = part.np[ tid ];
    auto * __restrict__ ix = &part.ix[ part_offset ];
    auto * __restrict__ x  = &part.x[ part_offset ];
    auto * __restrict__ u  = &part.u[ part_offset ];
    auto * __restrict__ θ  = &part.θ[ part_offset ];
    auto * __restrict__ q  = &part.q[ part_offset ];
    
    auto const dt_dz = dt_dx.x;
    auto const dt_dr = dt_dx.y;

    const int ir0 = tile_idx.y * nx.y;

    for( int i = 0; i < np; i++ ) {
        auto pu  = u[i];
        auto x0  = x[i];
        auto ix0 = ix[i];
        auto θi  = θ[i];
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
        θ[i] = float2{ xf/rf, yf/rf };

    }

    // Current normalization is done in Current::normalize()

    // Add current to global buffer
    const int tile_off = tid * tile_size;

    for( unsigned i = 0; i < ext_nx.x * ext_nx.y; i++ ) {
        current_m0[tile_off + i] += tile_buffer_0[i];
    }
}

/**
 * @brief Move particles, deposit current (modes 0 and 1) and shift positions
 * 
 * @param tile_idx          Tile index
 * @param part              Particle data
 * @param current_m0        Current grid (m=0)
 * @param current_m1        Current grid (m=1)
 * @param current_offset    Offset to position [0,0] of the current grid
 * @param nx                Current grid size (internal)
 * @param ext_nx            Current grid size (external)
 * @param dt_dx             Ratio between time step and cell size
 * @param shift             Cell shift
 */
void move_deposit_1(
    uint2 const tile_idx,
    ParticleData const part,
    cyl3<float> * const __restrict__ current_m0, 
    cyl3<std::complex<float>> * const __restrict__ current_m1, 
    unsigned int const current_offset, uint2 const nx, uint2 const ext_nx,
    float2 const dt_dx, const int2 shift ) 
{
//    std::cout << "move_deposit_1(shift)\n";

    const uint2 ntiles  = part.ntiles;
    const int tile_size = roundup4( ext_nx.x * ext_nx.y );

    // This is usually in block shared memory
    alignas(local_align) cyl3<float>               tile_buffer_0[tile_size];
    alignas(local_align) cyl3<std::complex<float>> tile_buffer_1[tile_size];

    // Zero local current buffer
    for( auto i = 0; i < tile_size; i++ ) {
        tile_buffer_0[i] = cyl3<float>{0};
        tile_buffer_1[i] = cyl3<std::complex<float>>{0};
    }

    // sync

    // Move particles and deposit current
    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    cyl3<float> *               J0 = & tile_buffer_0[ current_offset ];
    cyl3<std::complex<float>> * J1 = & tile_buffer_1[ current_offset ];

    const int jstride = ext_nx.x;

    const int part_offset  = part.offset[ tid ];
    const int np           = part.np[ tid ];
    auto * __restrict__ ix = &part.ix[ part_offset ];
    auto * __restrict__ x  = &part.x[ part_offset ];
    auto * __restrict__ u  = &part.u[ part_offset ];
    auto * __restrict__ θ  = &part.θ[ part_offset ];
    auto * __restrict__ q  = &part.q[ part_offset ];
    
    auto const dt_dz = dt_dx.x;
    auto const dt_dr = dt_dx.y;

    const int ir0 = tile_idx.y * nx.y;

    for( int i = 0; i < np; i++ ) {
        auto pu  = u[i];
        auto x0  = x[i];
        auto ix0 = ix[i];
        auto θi  = θ[i];
        auto pq  = q[i];

        // Get 1 / Lorentz gamma
        float const rg = rgamma( pu );

        // Cartesian motion
        auto Δx = dt_dr * rg * pu.x;
        auto Δy = dt_dr * rg * pu.y;

        /// @brief initial radial position
        auto ri = ( ir0 + ix0.y ) + x0.y;
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
        θ[i] = float2{ xf/rf, yf/rf };
    }

    // Current normalization is done in Current::normalize()

    // Add current to global buffer
    const int tile_off = tid * tile_size;

    for( unsigned i = 0; i < ext_nx.x * ext_nx.y; i++ ) {
        current_m0[tile_off + i] += tile_buffer_0[i];
        current_m1[tile_off + i] += tile_buffer_1[i];
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
    ParticleData const part,
    cyl3<float> * __restrict__ d_E, cyl3<float> * __restrict__ d_B, 
    unsigned int const field_offset, uint2 const ext_nx,
    float const alpha, double * __restrict__ d_energy )
{
    const uint2 ntiles  = part.ntiles;

    // Tile ID
    const int tid =  tile_idx.y * ntiles.x + tile_idx.x;

    int const field_vol = roundup4( ext_nx.x * ext_nx.y );
    int const tile_off = tid * field_vol;

    // Copy E and B into shared memory

    alignas(local_align) cyl3<float> E_local_m0[ field_vol ];
    alignas(local_align) cyl3<float> B_local_m0[ field_vol ];

    for( auto i = 0; i < field_vol; i++ ) {
        E_local_m0[i] = d_E[tile_off + i];
        B_local_m0[i] = d_B[tile_off + i];
    }

    cyl3<float> const * const __restrict__ E_m0 = & E_local_m0[ field_offset ];
    cyl3<float> const * const __restrict__ B_m0 = & B_local_m0[ field_offset ];

    // Push particles
    const int part_offset = part.offset[ tid ];
    const int np          = part.np[ tid ];
    int2   * __restrict__ ix = &part.ix[ part_offset ];
    float2 * __restrict__ x  = &part.x[ part_offset ];
    float3 * __restrict__ u  = &part.u[ part_offset ];
    float2 * __restrict__ θ  = &part.θ[ part_offset ];

    double energy = 0;

    const int jstride = ext_nx.x;

    for( int i = 0; i < np; i++ ) {

        // Interpolate field
        cyl3<float> e, b;
        interpolate_fld( E_m0, B_m0, jstride, ix[i], x[i], e, b );
        
        // Convert to cartesian components
        auto cosθ = θ[i].x;
        auto sinθ = θ[i].y;
        
        float3 cart_e = make_float3(
            ops::fma( e.r, cosθ, - e.θ * sinθ ),
            ops::fma( e.r, sinθ, + e.θ * cosθ ),
            e.z
        );

        float3 cart_b = make_float3(
            ops::fma( b.r, cosθ, - b.θ * sinθ ),
            ops::fma( b.r, sinθ, + b.θ * cosθ ),
            b.z
        );

        // Advance momentum
        float3 pu = u[i];

        if constexpr ( type == species::boris ) u[i] = dudt_boris( alpha, cart_e, cart_b, pu, energy );
        if constexpr ( type == species::euler ) u[i] = dudt_boris_euler( alpha, cart_e, cart_b, pu, energy );
    }

    // Add up energy from all particles
    // In OpenMP, d_energy needs to be a reduction variable
    *d_energy += energy;
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
void push_1 ( 
    uint2 const tile_idx,
    ParticleData const part,
    cyl3<float>               * __restrict__ d_E_m0, cyl3<float>               * __restrict__ d_B_m0, 
    cyl3<std::complex<float>> * __restrict__ d_E_m1, cyl3<std::complex<float>> * __restrict__ d_B_m1, 
    unsigned int const field_offset, uint2 const ext_nx,
    float const alpha, double * __restrict__ d_energy )
{
//    std::cout << "push_1()\n";
    
    const uint2 ntiles  = part.ntiles;

    // Tile ID
    const int tid =  tile_idx.y * ntiles.x + tile_idx.x;

    int const field_vol = roundup4( ext_nx.x * ext_nx.y );
    int const tile_off = tid * field_vol;

    // Copy E and B into shared memory

    alignas(local_align) cyl3<float> E_local_m0[ field_vol ];
    alignas(local_align) cyl3<float> B_local_m0[ field_vol ];

    for( auto i = 0; i < field_vol; i++ ) {
        E_local_m0[i] = d_E_m0[tile_off + i];
        B_local_m0[i] = d_B_m0[tile_off + i];
    }

    cyl3<float> const * const __restrict__ E_m0 = & E_local_m0[ field_offset ];
    cyl3<float> const * const __restrict__ B_m0 = & B_local_m0[ field_offset ];

    alignas(local_align) cyl3<std::complex<float>> E_local_m1[ field_vol ];
    alignas(local_align) cyl3<std::complex<float>> B_local_m1[ field_vol ];

    for( auto i = 0; i < field_vol; i++ ) {
        E_local_m1[i] = d_E_m1[tile_off + i];
        B_local_m1[i] = d_B_m1[tile_off + i];
    }

    cyl3<std::complex<float>> const * const __restrict__ E_m1 = & E_local_m1[ field_offset ];
    cyl3<std::complex<float>> const * const __restrict__ B_m1 = & B_local_m1[ field_offset ];

    // Push particles
    const int part_offset = part.offset[ tid ];
    const int np          = part.np[ tid ];
    int2   * __restrict__ ix = &part.ix[ part_offset ];
    float2 * __restrict__ x  = &part.x[ part_offset ];
    float3 * __restrict__ u  = &part.u[ part_offset ];
    float2 * __restrict__ θ  = &part.θ[ part_offset ];

    double energy = 0;

    const int jstride = ext_nx.x;

    for( int i = 0; i < np; i++ ) {

        // Interpolate field - mode 0
        cyl3<float> e, b;
        interpolate_fld( E_m0, B_m0, jstride, ix[i], x[i], e, b );

        // Interpolate field - mode 1
        cyl3<std::complex<float>> e1, b1;
        interpolate_fld( E_m1, B_m1, jstride, ix[i], x[i], e1, b1 );

        // Get full field
        auto cosθ = θ[i].x;
        auto sinθ = θ[i].y;

        e.z += ops::fma( cosθ, e1.z.real( ), -sinθ * e1.z.imag( ) );
        e.r += ops::fma( cosθ, e1.r.real( ), -sinθ * e1.r.imag( ) );
        e.θ += ops::fma( cosθ, e1.θ.real( ), -sinθ * e1.θ.imag( ) );

        b.z += ops::fma( cosθ, b1.z.real( ), -sinθ * b1.z.imag( ) );
        b.r += ops::fma( cosθ, b1.r.real( ), -sinθ * b1.r.imag( ) );
        b.θ += ops::fma( cosθ, b1.θ.real( ), -sinθ * b1.θ.imag( ) );

        // Convert to cartesian components
        float3 cart_e = make_float3(
            ops::fma( e.r, cosθ, - e.θ * sinθ ),
            ops::fma( e.r, sinθ, + e.θ * cosθ ),
            e.z
        );

        float3 cart_b = make_float3(
            ops::fma( b.r, cosθ, - b.θ * sinθ ),
            ops::fma( b.r, sinθ, + b.θ * cosθ ),
            b.z
        );

        // Advance momentum
        float3 pu = u[i];

        if constexpr ( type == species::boris ) u[i] = dudt_boris( alpha, cart_e, cart_b, pu, energy );
        if constexpr ( type == species::euler ) u[i] = dudt_boris_euler( alpha, cart_e, cart_b, pu, energy );
    }

    // Add up energy from all particles
    // In OpenMP, d_energy needs to be a reduction variable
    *d_energy += energy;
}

/**
 * @brief Construct a new Species object
 * 
 * @param name  Name for the species object (used for diagnostics)
 * @param m_q   Mass over charge ratio
 * @param ppc   Number of particles per cell
 */
Species::Species( std::string const name, float const m_q, uint3 const ppc ):
    ppc(ppc), name(name), m_q(m_q)
{

    // Validate parameters
    if ( m_q == 0 ) {
        std::cerr << "(*error*) Invalid m_q value, must be not 0, aborting...\n";
        exit(1);
    }

    if ( ppc.x < 1 || ppc.y < 1 || ppc.z < 1 ) {
        std::cerr << "(*error*) Invalid ppc value, must be >= 1 in all directions\n";
        exit(1);
    }

    // Set default parameters
    density   = new Density::Uniform( 1.0 );
    udist     = new UDistribution::None();
    push_type = species::boris;

    bc.x.lower = bc.x.upper = species::bc::periodic;
    bc.y.lower = species::bc::axial; 
    bc.y.upper = species::bc::open;

    // Nullify pointers to data structures
    particles = nullptr;
    tmp = nullptr;
    sort = nullptr;

    // Set nmodes to invalid value, will be set by initialize
    nmodes = 0;
}


/**
 * @brief Initialize data structures and inject initial particle distribution
 * 
 * @param nmodes_            Number of cylindrical modes (including fundamental mode)
 * @param box_              Global simulation box size
 * @param ntiles            Number of tiles
 * @param nx                Individual tile grid size
 * @param dt_               Time step
 * @param id_               Species unique identifier
 */
void Species::initialize( int nmodes_, float2 const box_, uint2 const ntiles, uint2 const nx,
    float const dt_, int const id_ ) {
    
    // Store number of cylindrical modes
    nmodes = nmodes_;

    if ( nmodes < 1 || nmodes > 2 ) {
        std::cerr << "(*error*) Unsupported number of nodes, must be 1 or 2\n";
        std::exit(1);
    }

    // Store simulation box size
    box = box_;

    // Store simulation time step
    dt = dt_;

    // Store species id (used by RNG)
    id = id_;

    // Set charge normalization factor
    q_ref = copysign( density->n0 , m_q ) / (ppc.x * ppc.y * ppc.z);
       
    // Get global grid size
    auto  dims = nx * ntiles;

    // Set cell size
    dx.x = box.x / (dims.x);
    dx.y = box.y / (dims.y);

    // Reference number maximum number of particles
    unsigned int max_part = 1.2 * dims.x * dims.y * ppc.x * ppc.y * ppc.z;

    // Create particle data structure
    particles = new Particles( ntiles, nx, max_part );
    particles->periodic_z = ( bc.x.lower == species::bc::periodic );

    tmp = new Particles( ntiles, nx, max_part );
    sort = new ParticleSort( ntiles, max_part );
    np_inj = memory::malloc<int>( ntiles.x * ntiles.y );

    // Initialize energy diagnostic
    d_energy = 0;

    // Initialize particle move counter
    d_nmove = 0;

    // Reset iteration numbers
    iter = 0;

    // Inject initial distribution

    // Count particles to inject and store in np_inj
    np_inject( particles -> local_range(), np_inj );

    // Do an exclusive scan to get the required offsets
    uint32_t off = 0;
    for( unsigned i = 0; i < ntiles.x * ntiles.y; i ++ ) {
        particles -> offset[i] = off;
        off += np_inj[i];
    }

    // Inject the particles
    inject( particles -> local_range() );

    // Set inital velocity distribution
    udist -> set( *particles, id );
}

/**
 * @brief Destroy the Species object
 * 
 */
Species::~Species() {
    memory::free( np_inj );
    delete( tmp );
    delete( sort );
    delete( particles );
    delete( density );
    delete( udist );
};


/**
 * @brief Inject particles in the complete simulation box
 * 
 */
void Species::inject( ) {

    float2 ref{ static_cast<float>(moving_window.motion()), 0 };

    density -> inject( *particles, copysign( 1.0f, m_q ), ppc, dx, ref, particles -> local_range() );
}

/**
 * @brief Inject particles in a specific cell range
 * 
 */
void Species::inject( bnd<unsigned int> range ) {

    float2 ref{ static_cast<float>(moving_window.motion()), 0 };

    density -> inject( *particles, copysign( 1.0f, m_q ), ppc, dx, ref, range );
}

/**
 * @brief Gets the number of particles that would be injected in a specific cell range
 * 
 * Although the routine only considers injection in a specific range, the
 * number of particles to be injected is calculated on all tiles (returning
 * zero on those, as expected)
 * 
 * @param range 
 * @param np        (device pointer) Number of particles to inject in each tile
 */
void Species::np_inject( bnd<unsigned int> range, int * np ) {

    /// @brief position of lower corner of local grid in simulation units
    float2 ref{ static_cast<float>(moving_window.motion()), 0 };

    density -> np_inject( *particles, ppc, dx, ref, range, np );
}

/**
 * @brief Physical boundary conditions for the x direction 
 * 
 * @param ntiles    Number of tiles
 * @param tile_idx  Tile index
 * @param tiles     Particle tile information
 * @param data      Particle data
 * @param nx        Tile grid size
 * @param bc        Boundary condition
 */
void species_bcx(
    uint2 const tile_idx,
    ParticleData const part,
    species::bc_type const bc ) 
{
    const uint2 ntiles  = part.ntiles;
    const int nx = part.nx.x;
    
    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    const int part_offset    = part.offset[ tid ];
    const int np             = part.np[ tid ];
    int2   * __restrict__ ix = &part.ix[ part_offset ];
    float2 * __restrict__ x  = &part.x[ part_offset ];
    float3 * __restrict__ u  = &part.u[ part_offset ];

    if ( tile_idx.x == 0 ) {
        // Lower boundary
        switch( bc.x.lower ) {
        case( species::bc::reflecting ) :
            for( int i = 0; i < np; i++ ) {
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
        switch( bc.x.upper ) {
        case( species::bc::reflecting ) :
            for( int i = 0; i < np; i++ ) {
                if( ix[i].x >=  nx ) {
                    ix[i].x -= 1;
                    x[i].x = -x[i].x;
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
 * @brief Physical boundary conditions for the y direction (upper bound only)
 * 
 * @param ntiles    Number of tiles
 * @param tile_idx  Tile index
 * @param tiles     Particle tile information
 * @param data      Particle data
 * @param nx        Tile grid size
 * @param bc        Boundary condition
 */
void species_bcy_upper(
    uint2 const tile_idx, double dr,
    ParticleData const part,
    species::bc_type const bc ) 
{
    const uint2 ntiles  = part.ntiles;
    const int ny = part.nx.y;

    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    const int part_offset  = part.offset[ tid ];
    const int np           = part.np[ tid ];
    auto * __restrict__ ix = &part.ix[ part_offset ];
    auto * __restrict__ x  = &part.x[ part_offset ];
    auto * __restrict__ u  = &part.u[ part_offset ];
    auto * __restrict__ θ  = &part.θ[ part_offset ];


    // Upper radial boundary
    switch( bc.y.upper ) {
    case( species::bc::reflecting ) :
        for( int i = 0; i < np; i++ ) {
            if( ix[i].y >=  ny ) {
                // Correct radial position
                ix[i].y -= 1;
                x[i].y = -x[i].y;

                // Correct radial velocity
                const auto cosθ = θ[i].x;
                const auto sinθ = θ[i].y;
                auto       ur = u[i].x * cosθ + u[i].y * sinθ;
                const auto uθ = u[i].y * cosθ - u[i].x * sinθ;

                ur = -ur;
                u[i].x = ur * cosθ - uθ * sinθ;
                u[i].y = ur * sinθ + uθ * cosθ;
            }
        }
        break;
    default:
        break;
    }
}

/**
 * @brief Processes "physical" boundary conditions
 * 
 */
void Species::process_bc() {


    // x boundaries
    if ( bc.x.lower > species::bc::periodic || bc.x.upper > species::bc::periodic ) {
        
        for( unsigned ty = 0; ty < particles -> ntiles.y; ty ++ ) {
            for( unsigned tx : { 0u, particles -> ntiles.x-1 } ) {
                const auto tile_idx = make_uint2( tx, ty );
                species_bcx ( tile_idx, *particles, bc );
            }
        }
    }

    // y boundaries
    if ( bc.y.upper > species::bc::periodic ) {
        double dr = dx.y;
        auto ty = particles -> ntiles.y-1;
        for( unsigned tx = 0; tx < particles -> ntiles.x; tx ++ ) {
            const auto tile_idx = make_uint2( tx, ty );
            species_bcy_upper ( tile_idx, dr, *particles, bc );
        }
    }
}

/**
 * @brief Free stream particles 1 iteration
 * 
 * @note No acceleration or current deposition is performed. Used for debug purposes.
 * 
 */
void Species::advance( ) {

    // Advance positions
    move( );

    // Process physical boundary conditions
    // process_bc();
    
    // Sort particles according to tile
    particles -> tile_sort( *tmp, *sort );

    // Increase internal iteration number
    iter++;
}

/**
 * @brief Free-stream particles 1 iteration
 * 
 * This routine will:
 * 1. Advance positions and deposit current
 * 2. Process boundary conditions
 * 3. Sort particles according to tiles
 * 
 * @param emf       EM fields
 * @param current   Electric current density
 */
void Species::advance( Current & current ) {

    // Advance positions and deposit current
    move( current );

    // Process physical boundary conditions
    // process_bc();

    // Increase internal iteration number
    iter++;
    
    // Sort particles according to tile
    particles -> tile_sort( *tmp, *sort );

}

/**
 * @brief Advance particles 1 iteration
 * 
 * This routine will:
 * 1. Advance momenta
 * 2. Advance positions and deposit current
 * 3. Process boundary conditions
 * 4. Sort particles according to tiles
 * 
 * @param emf       EM fields
 * @param current   Electric current density
 */
void Species::advance( EMF const &emf, Current &current ) {

    // Advance momenta
    push( emf );

    // Advance positions and deposit current
    move( current );

    // Process physical boundary conditions
    // process_bc();

    // Increase internal iteration number
    iter++;
    
    // Sort particles according to tile
    particles -> tile_sort( *tmp, *sort );
}

void Species::advance_mov_window( Current &current ) {

    if ( moving_window.needs_move( (iter+1) * dt ) ) {

        // Advance positions, deposit current and shift particles
        move( current, make_int2(-1,0) );

        // Process boundary conditions
        process_bc();

        // Find range where new particles need to be injected
        uint2 g_nx = particles -> get_dims();
        bnd<unsigned int> range;
        range.x = { g_nx.x - 1, g_nx.x - 1 };
        range.y = {          0, g_nx.y - 1 };

        // Count new particles to be injected
        np_inject( range, np_inj );

        // Sort particles over tiles, leaving room for new particles to be injected
        particles -> tile_sort( *tmp, *sort, np_inj );

        // Inject new particles
        inject( range );

        // Advance moving window
        moving_window.advance();

    } else {
        
        // Advance positions and deposit current
        move( current );

        // Process boundary conditions
        process_bc();

        // Sort particles over tiles
        particles -> tile_sort( *tmp, *sort );
    }

    // Increase internal iteration number
    iter++;
}

void Species::advance_mov_window( EMF const &emf, Current &current ) {

    // Advance momenta
    push( emf );

    if ( moving_window.needs_move( (iter+1) * dt ) ) {

        // Advance positions, deposit current and shift particles
        move( current, make_int2(-1,0) );

        // Process boundary conditions
        process_bc();

        // Find range where new particles need to be injected
        uint2 g_nx = particles -> get_dims();
        bnd<unsigned int> range;
        range.x = { g_nx.x - 1, g_nx.x - 1 };
        range.y = {          0, g_nx.y - 1 };

        // Count new particles to be injected
        np_inject( range, np_inj );

        // Sort particles over tiles, leaving room for new particles to be injected
        particles -> tile_sort( *tmp, *sort, np_inj );

        // Inject new particles
        inject( range );

        // Advance moving window
        moving_window.advance();

    } else {
        
        // Advance positions and deposit current
        move( current );

        // Process boundary conditions
        process_bc();

        // Sort particles over tiles
        particles -> tile_sort( *tmp, *sort );
    }

    // Increase internal iteration number
    iter++;
}

/**
 * @brief Moves particles and deposit current
 * 
 * Current will be accumulated on existing data
 * 
 * @param current   Current grid
 */
void Species::move( Current & current )
{
    ///@brief timestep to cell size ratio
    const float2 dt_dx = make_float2(
        dt / dx.x,
        dt / dx.y
    );

    ///@brief Mode 0 current
    auto & J0 = current.mode0();

    switch (nmodes)
    {
    case 1: {
        #pragma omp parallel for schedule(dynamic)
        for( unsigned tid = 0; tid < particles -> ntiles.y * particles -> ntiles.x; tid ++ ) {
            
            const auto tile_idx = make_uint2( tid % particles -> ntiles.x, tid / particles -> ntiles.x );
            move_deposit_0(
                tile_idx, *particles,
                J0.d_buffer, J0.offset, J0.nx, J0.ext_nx, 
                dt_dx
            );
        }
    } break;

    case 2: {
        auto & J1 = current.mode(1);

        #pragma omp parallel for schedule(dynamic)
        for( unsigned tid = 0; tid < particles -> ntiles.y * particles -> ntiles.x; tid ++ ) {
            
            const auto tile_idx = make_uint2( tid % particles -> ntiles.x, tid / particles -> ntiles.x );
            move_deposit_1(
                tile_idx, *particles,
                J0.d_buffer, J1.d_buffer, J0.offset, J0.nx, J0.ext_nx, 
                dt_dx
            );
        }
    } break;

    default:
        break;
    }

    // This avoids the reduction overhead
    for( unsigned tid = 0; tid < particles -> ntiles.y * particles -> ntiles.x; tid ++ ) {
        d_nmove += particles -> np[tid];
    }
}

/**
 * @brief Move particles (advance positions), deposit current and shift positions
 * 
 * @param current   Electric current density
 * @param shift     Cell shift
 */
void Species::move( Current & current, const int2 shift )
{
    ///@brief timestep to cell size ratio
    const float2 dt_dx = make_float2(
        dt / dx.x,
        dt / dx.y
    );

    ///@brief Mode 0 current
    auto & J0 = current.mode0();

    switch (nmodes)
    {
    case 1: {
        #pragma omp parallel for schedule(dynamic)
        for( unsigned tid = 0; tid < particles -> ntiles.y * particles -> ntiles.x; tid ++ ) {
            
            const auto tile_idx = make_uint2( tid % particles -> ntiles.x, tid / particles -> ntiles.x );
            move_deposit_0(
                tile_idx, *particles,
                J0.d_buffer, J0.offset, J0.nx, J0.ext_nx, 
                dt_dx, shift
            );
        }
    } break;

    case 2: {
        auto & J1 = current.mode(1);

        #pragma omp parallel for schedule(dynamic)
        for( unsigned tid = 0; tid < particles -> ntiles.y * particles -> ntiles.x; tid ++ ) {
            
            const auto tile_idx = make_uint2( tid % particles -> ntiles.x, tid / particles -> ntiles.x );
            move_deposit_1(
                tile_idx, *particles,
                J0.d_buffer, J1.d_buffer, J0.offset, J0.nx, J0.ext_nx, 
                dt_dx, shift
            );
        }
    } break;

    default:
        break;
    }

    // This avoids the reduction overhead
    for( unsigned tid = 0; tid < particles -> ntiles.y * particles -> ntiles.x; tid ++ ) {
        d_nmove += particles -> np[tid];
    }
}


/**
 * @brief kernel for moving particles
 * 
 * @param d_tile            Particle tiles information
 * @param d_ix              Particle buffer (cells)
 * @param d_x               Particle buffer (positions)
 * @param d_u               Particle buffer (momenta)
 * @param dt_dx             Time step over cell size
 */
void move_kernel(
    uint2 const tile_idx,
    ParticleData const part,
    float2 const dt_dx ) 
{
    const uint2 ntiles  = part.ntiles;

    // Move particles and deposit current
    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    const int part_offset    = part.offset[ tid ];
    const int np             = part.np[ tid ];
    auto * __restrict__ ix = &part.ix[ part_offset ];
    auto * __restrict__ x  = &part.x[ part_offset ];
    auto * __restrict__ u  = &part.u[ part_offset ];
    auto * __restrict__ θ  = &part.θ[ part_offset ];
    
    auto const dt_dz = dt_dx.x;
    auto const dt_dr = dt_dx.y;

    for( int i = 0; i < np; i++ ) {
        auto pu  = u[i];
        auto x0  = x[i];
        auto ix0 = ix[i];

        auto cosθ = θ[i].x;
        auto sinθ = θ[i].y; 

        // Get 1 / Lorentz gamma
        float rg = rgamma( pu );

        // Cartesian motion
        auto Δx = dt_dr * rg * pu.x;
        auto Δy = dt_dr * rg * pu.y;
        auto Δz = dt_dz * rg * pu.z;

        // New cartesian positions
        auto ri = ix0.y + x0.y;
        auto xf = ops::fma( ri, cosθ, Δx );
        auto yf = ops::fma( ri, sinθ, Δy );

        // New radial position
        auto rf = std::sqrt( ops::fma( xf, xf, yf*yf ) );

        // Protection agains rf == 0
        // This is VERY unlikely
        float Δr;
        if ( rf > 0 ) {
            Δr = ops::fma( Δx , ops::fma( ri, cosθ, xf ) , Δy * ops::fma( ri, sinθ, yf ) ) / (rf + ri);
            cosθ = xf/rf;
            sinθ = yf/rf;
        } else {
            Δr   = -ri;
            cosθ = 1;
            sinθ = 0;
        }

        // Store new angular position
        θ[i] = float2{ cosθ, sinθ };

        // Advance grid (z,r) position
        float2 x1 = make_float2(
            x0.x + Δz,
            x0.y + Δr
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
}


/**
 * @brief Moves particles (no current/charge deposition)
 * 
 * This is usually used for test species: species that do not self-consistently
 * influence the simulation
 */
void Species::move( )
{


    const float2 dt_dx = make_float2(
        dt / dx.x,
        dt / dx.y
    );

    #pragma omp parallel for schedule(dynamic)
    for( unsigned tid = 0; tid < particles -> ntiles.y * particles -> ntiles.x; tid ++ ) {
        const auto tile_idx = make_uint2( tid % particles -> ntiles.x, tid / particles -> ntiles.x );
        move_kernel ( tile_idx, *particles, dt_dx );
    }

    // This avoids the reduction overhead
    for( unsigned tid = 0; tid < particles -> ntiles.y * particles -> ntiles.x; tid ++ ) {
        d_nmove += particles -> np[tid];
    }
}

/**
 * @brief       Accelerates particles using a Boris pusher
 * 
 * @param emf     EMF field
 */
void Species::push( EMF const &emf )
{
    // Currently only Boris pusher
    if ( push_type != species::boris ) {
        std::cerr << "(*error*) Only Boris pusher is currently available\n";
        std::exit(1);
    }

    const float alpha = 0.5 * dt / m_q;
    d_energy = 0;

    auto & E0 = emf.E -> mode0();
    auto & B0 = emf.B -> mode0();

    switch (nmodes)
    {
    case 1: {
        #pragma omp parallel for schedule(dynamic) reduction(+:d_energy)
        for( unsigned tid = 0; tid < particles -> ntiles.y * particles -> ntiles.x; tid ++ ) {    
            const uint2 tile_idx = make_uint2( tid % particles -> ntiles.x, tid / particles -> ntiles.x );
            push_0 <species::boris> (
                tile_idx, *particles,
                E0.d_buffer, B0.d_buffer, E0.offset, E0.ext_nx, alpha,
                &d_energy
            );
        }
    }
    break;

    case 2: {
        auto & E1 = emf.E -> mode(1);
        auto & B1 = emf.B -> mode(1);
        
        #pragma omp parallel for schedule(dynamic) reduction(+:d_energy)
        for( unsigned tid = 0; tid < particles -> ntiles.y * particles -> ntiles.x; tid ++ ) {    
            const uint2 tile_idx = make_uint2( tid % particles -> ntiles.x, tid / particles -> ntiles.x );
            push_1 <species::boris> (
                tile_idx, *particles,
                E0.d_buffer, B0.d_buffer, 
                E1.d_buffer, B1.d_buffer, 
                E0.offset, E0.ext_nx, alpha,
                &d_energy
            );
        }
    }
    break;

    default:
        break;
    }

}

/**
 * @brief kernel for depositing mode m=0 charge
 * 
 * @param d_charge  Charge density grid (will be zeroed by this kernel)
 * @param offset    Offset to position (0,0) of grid
 * @param ext_nx    External tile size (i.e. including guard cells)
 * @param d_tile    Particle tiles information
 * @param d_ix      Particle buffer (cells)
 * @param d_x       Particle buffer (position)
 * @param q         Species charge per particle
 */
void dep_charge_0(
    uint2 const tile_idx,
    ParticleData const part,
    float * const __restrict__ d_charge, int offset, uint2 ext_nx )
{
    const uint2 ntiles  = part.ntiles;
    const int tile_size = roundup4( ext_nx.x * ext_nx.y );
 
    float _dep_charge_buffer[tile_size];

    // Zero shared memory and sync.
    for( unsigned i = 0; i < ext_nx.x * ext_nx.y; i ++ ) {
        _dep_charge_buffer[i] = 0;
    }

    float *charge = &_dep_charge_buffer[ offset ];

    // sync;

    const int tid      = tile_idx.y * ntiles.x + tile_idx.x;
    const int part_off = part.offset[ tid ];
    const int np       = part.np[ tid ];
    auto const * __restrict__ const ix = &part.ix[ part_off ];
    auto const * __restrict__ const x  = &part.x[ part_off ];
    auto const * __restrict__ const q  = &part.q[ part_off ];
    const int ystride = ext_nx.x;

    for( int i = 0; i < np; i ++ ) {
        const int idx = ix[i].y * ystride + ix[i].x;
        const float s0x = 0.5f - x[i].x;
        const float s1x = 0.5f + x[i].x;
        const float s0y = 0.5f - x[i].y;
        const float s1y = 0.5f + x[i].y;

        // When use more than 1 thread per tile, these need to be atomic inside tile
        charge[ idx               ] += s0y * s0x * q[i];
        charge[ idx + 1           ] += s0y * s1x * q[i];
        charge[ idx     + ystride ] += s1y * s0x * q[i];
        charge[ idx + 1 + ystride ] += s1y * s1x * q[i];
    }

    // sync

    // Copy data to global memory
    const int tile_off = tid * roundup4( ext_nx.x * ext_nx.y );
    for( unsigned i = 0; i < ext_nx.x * ext_nx.y; i ++ ) {
        d_charge[tile_off + i] += _dep_charge_buffer[i];
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
void dep_charge(
    uint2 const tile_idx,
    ParticleData const part,
    std::complex<float> * const __restrict__ d_charge, int offset, uint2 ext_nx )
{
    const uint2 ntiles  = part.ntiles;
    const int tile_size = roundup4( ext_nx.x * ext_nx.y );
 
    std::complex<float> _dep_charge_buffer[tile_size];

    // Zero shared memory and sync.
    for( unsigned i = 0; i < ext_nx.x * ext_nx.y; i ++ ) {
        _dep_charge_buffer[i] = 0;
    }

    auto *charge = &_dep_charge_buffer[ offset ];

    // sync;

    const int tid      = tile_idx.y * ntiles.x + tile_idx.x;
    const int part_off = part.offset[ tid ];
    const int np       = part.np[ tid ];
    auto const * __restrict__ const ix = &part.ix[ part_off ];
    auto const * __restrict__ const x  = &part.x[ part_off ];
    auto const * __restrict__ const q  = &part.q[ part_off ];
    auto const * __restrict__ const θ  = &part.θ[ part_off ];
    const int ystride = ext_nx.x;

    for( int i = 0; i < np; i ++ ) {
        const int idx = ix[i].y * ystride + ix[i].x;
        const float s0x = 0.5f - x[i].x;
        const float s1x = 0.5f + x[i].x;
        const float s0y = 0.5f - x[i].y;
        const float s1y = 0.5f + x[i].y;

        static_assert( m == 1, "only mode m = 1 is currently supported" );
        // auto qm = q[i] * expimθ<m>( θ[i] );
        auto qm = q[i] * std::complex<float>{ θ[i].x, -θ[i].y };

        // When use more than 1 thread per tile, these need to be atomic inside tile
        charge[ idx               ] += s0y * s0x * qm;
        charge[ idx + 1           ] += s0y * s1x * qm;
        charge[ idx     + ystride ] += s1y * s0x * qm;
        charge[ idx + 1 + ystride ] += s1y * s1x * qm;
    }

    // sync

    // Copy data to global memory
    const int tile_off = tid * roundup4( ext_nx.x * ext_nx.y );
    for( unsigned i = 0; i < ext_nx.x * ext_nx.y; i ++ ) {
        d_charge[tile_off + i] += _dep_charge_buffer[i];
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
 */
template< class T >
void charge_norm(
    uint2 const tile_idx,
    uint2 const ntiles,
    T * const __restrict__ d_charge, int offset, 
    uint2 const nx, uint2 const ext_nx,
    const float dr, const float scale = 1.0f
) {

    auto tid = tile_idx.y * ntiles.x + tile_idx.x;
    const int tile_off = tid * roundup4( ext_nx.x * ext_nx.y );
    const int jstride = ext_nx.x;

    auto * __restrict__ charge = &  d_charge[ tile_off + offset ];

    int ir0 = tile_idx.y * nx.y;
    for( int j = 0; j < static_cast<int>(nx.y+1); j++ ){
        auto norm = scale/(std::abs( ir0 + j - 0.5f) * dr);
        for( int i = 0; i < static_cast<int>(nx.x+1); i++ ){
            charge[ j * jstride +i ] *= norm;
        }
    }

    // Axial boundary
    // Fold values for r < 0 back into simulation domain
    if ( ir0 == 0 ) {
        for( int i = 0; i < static_cast<int>(nx.x+1); i++ ){
            charge[ i + 1 * jstride ] += charge[ i + 0 * jstride ];
            charge[ i + 0 * jstride ]  = charge[ i + 1 * jstride ];
        }
    }
}

/**
 * @brief Deposit charge density (mode 0)
 * 
 * @param charge    Charge density grid
 */
void Species::deposit_charge0( grid<float> &charge0 ) const {

    #pragma omp parallel for schedule(dynamic)
    for( unsigned int tid = 0; tid <  particles -> ntiles.y * particles -> ntiles.x; tid++ ) {
        const auto tile_idx = make_uint2( 
            tid % particles -> ntiles.x,
            tid / particles -> ntiles.x
        );
       
        // Deposit mode 0
        dep_charge_0( tile_idx, *particles, charge0.d_buffer, charge0.offset, charge0.ext_nx );

        charge_norm( tile_idx, charge0.get_ntiles(), charge0.d_buffer, charge0.offset, 
                    charge0.nx, charge0.ext_nx, dx.y  );
    }
}

/**
 * @brief Deposit charge density high order modes
 * 
 * @param m         Cylindrical mode to deposit (1 to 4)
 * @param charge    Charge density grid (complex)
 */
void Species::deposit_charge( const unsigned m, grid<std::complex<float>> &charge ) const {

    if ( m < 1 || m > 4 ) {
        std::cerr << "(*error*) Only modes m = 1 to 4 are currently supported, aborting...\n";
        std::exit(1);
    }

    #pragma omp parallel for schedule(dynamic)
    for( unsigned int tid = 0; tid <  particles -> ntiles.y * particles -> ntiles.x; tid++ ) {
        const auto tile_idx = make_uint2( 
            tid % particles -> ntiles.x,
            tid / particles -> ntiles.x
        );

/*
        switch( m ) {
            case 4:
                dep_charge<4>( tile_idx, *particles, charge.d_buffer, charge.offset, charge.ext_nx );
                break;
            case 3:
                dep_charge<3>( tile_idx, *particles, charge.d_buffer, charge.offset, charge.ext_nx );
                break;
            case 2:
                dep_charge<2>( tile_idx, *particles, charge.d_buffer, charge.offset, charge.ext_nx );
                break;
            case 1:
                dep_charge<1>( tile_idx, *particles, charge.d_buffer, charge.offset, charge.ext_nx );
                break;
        }
*/
        dep_charge<1>( tile_idx, *particles, charge.d_buffer, charge.offset, charge.ext_nx );

        // High-order modes need an additional factor of 2
        charge_norm( tile_idx, charge.get_ntiles(), charge.d_buffer, charge.offset, 
                    charge.nx, charge.ext_nx, dx.y, 2.f);
    }
}

/**
 * @brief Saves charge density to file
 * 
 * The routine will create a new charge grid, deposit the charge and save the grid
 * 
 */
void Species::save_charge( const unsigned m ) const {

    // Prepare file info
    zdf::grid_axis axis[2];
    axis[0] = (zdf::grid_axis) {
        .name = (char *) "z",
        .min = 0. + moving_window.motion(),
        .max = box.x + moving_window.motion(),
        .label = (char *) "z",
        .units = (char *) "c/\\omega_n"
    };

    axis[1] = (zdf::grid_axis) {
        .name = (char *) "r",
        .min = -0.5,
        .max = box.y-.5,
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

    // For linear interpolation we only require 1 guard cell at the upper boundary
    bnd<unsigned int> gc;
    gc.x = {0,1};
    gc.y = {0,1};

    if ( m == 0 ) {
        // Fundamental mode
        grid<float> charge( particles -> ntiles, particles -> nx, gc );
        charge.set_periodic( int2{ particles->periodic_z, 0 } );
        
        charge.zero();
        deposit_charge0( charge );
        charge.add_from_gc();
        charge.save( info, iter_info, path );
    } else {
        // High-order mode
        grid<std::complex<float>> charge( particles -> ntiles, particles -> nx, gc );
        charge.set_periodic( int2{ particles->periodic_z, 0 } );

        charge.zero();
        deposit_charge( m, charge );
        charge.add_from_gc();
        charge.save( info, iter_info, path );
    }
}

/**
 * @brief Save particle data to file
 * 
 */
void Species::save() const {

    const std::string path = "PARTICLES";

    const char * qnames[] = {
        "z","r",
        "q",
        "cosθ","sinθ",
        "ux","uy","uz"
    };

    const char * qlabels[] = {
        "z","r",
        "q",
        "\\cos θ", "\\sin θ",
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
    uint32_t np = particles -> np_total();
    info.np = np;


    // Open file
    zdf::file part_file;
    zdf::open_part_file( part_file, info, iter_info, "PARTICLES/" + name );

    // Gather and save each quantity
    float *h_data = nullptr;
    if( np > 0 ) {
        h_data = memory::malloc<float>( np );
    }

    if ( np > 0 ) {
        float2 scale{ dx.x, static_cast<float>(moving_window.motion()) };
        particles -> gather( part::quant::z, scale, h_data );
    }
    zdf::add_quant_part_file( part_file, "z", h_data, np );

    if ( np > 0 ) {
        float2 scale{ dx.y, 0 };
        particles -> gather( part::quant::r, scale, h_data );
    }
    zdf::add_quant_part_file( part_file, "r", h_data, np );

    if ( np > 0 ) {
        particles -> gather( part::quant::q, h_data );
    }
    zdf::add_quant_part_file( part_file, "q", h_data, np );

    if ( np > 0 ) {
        particles -> gather( part::quant::cosθ, h_data );
    }
    zdf::add_quant_part_file( part_file, "cosθ", h_data, np );

    if ( np > 0 ) {
        particles -> gather( part::quant::sinθ, h_data );
    }
    zdf::add_quant_part_file( part_file, "sinθ", h_data, np );

    if ( np > 0 ) {
        particles -> gather( part::quant::ux, h_data );
    }
    zdf::add_quant_part_file( part_file, "ux", h_data, np );

    if ( np > 0 ) {
        particles -> gather( part::quant::uy, h_data );
    }
    zdf::add_quant_part_file( part_file, "uy", h_data, np );

    if ( np > 0 ) {
        particles -> gather( part::quant::uz, h_data );
    }
    zdf::add_quant_part_file( part_file, "uz", h_data, np );

    // Close the file
    zdf::close_file( part_file );

    // Cleanup
    if ( np > 0 ) {
        memory::free( h_data );
    }
}



/**
 * @brief kernel for depositing 1d phasespace
 * 
 * @tparam quant    Phasespace quantity
 * @param d_data    Output data
 * @param range     Phasespace value range
 * @param size      Phasespace grid size
 * @param tile_nx   Size of tile grid
 * @param norm      Normalization factor
 * @param part      Particle data
 */
template < phasespace::quant quant >
void dep_pha1_kernel(
    uint2 const tile_idx,
    float * const __restrict__ d_data, float2 const range, int const size,
    float const norm, 
    ParticleData const part )
{
    const uint2 ntiles  = part.ntiles;
    const uint2 tile_nx = part.nx;

    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    const int part_offset  = part.offset[ tid ];
    const int np           = part.np[ tid ];
    auto * __restrict__ ix = &part.ix[ part_offset ];
    auto * __restrict__ x  = &part.x[ part_offset ];
    auto * __restrict__ u  = &part.u[ part_offset ];
    auto * __restrict__ q  = &part.q[ part_offset ];

    float const pha_rdx = size / (range.y - range.x);

    const int shiftx = tile_idx.x * tile_nx.x;
    const int shifty = tile_idx.y * tile_nx.y;

    for( int i = 0; i < np; i++ ) {
        float d;
        if constexpr ( quant == phasespace:: z  ) d = ( shiftx + ix[i].x) + (x[i].x + 0.5f);
        if constexpr ( quant == phasespace:: r  ) d = ( shifty + ix[i].y) + 0.5f;
        if constexpr ( quant == phasespace:: ux ) d = u[i].x;
        if constexpr ( quant == phasespace:: uy ) d = u[i].y;
        if constexpr ( quant == phasespace:: uz ) d = u[i].z;

        float n =  (d - range.x ) * pha_rdx - 0.5f;
        int   k = int( n + 1 ) - 1;
        float w = n - k;

        // When using multi-threading these need to be atomic accross tiles
        if ((k   >= 0) && (k   < size-1)) d_data[k  ] += (1-w) * norm * q[i];
        if ((k+1 >= 0) && (k+1 < size-1)) d_data[k+1] +=    w  * norm * q[i];
    }
}


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
void Species::dep_phasespace( float * const d_data, phasespace::quant quant, 
    float2 range, unsigned const size ) const
{
    // Zero device memory
    memory::zero( d_data, size );
    
    float norm = std::abs(q_ref) * ( dx.x * dx.y ) *
                 size / (range.y - range.x) ;

    switch(quant) {
    case( phasespace::z ):
        range.y /= dx.x;
        range.x /= dx.x;
        for( unsigned ty = 0; ty < particles -> ntiles.y; ty ++ ) {
            for( unsigned tx = 0; tx < particles -> ntiles.x; tx ++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                dep_pha1_kernel<phasespace::z>  (
                    tile_idx, 
                    d_data, range, size, norm, 
                    *particles
                );
            }
        }

        break;
    case( phasespace:: r ):
        range.y /= dx.y;
        range.x /= dx.y;
        for( unsigned ty = 0; ty < particles -> ntiles.y; ty ++ ) {
            for( unsigned tx = 0; tx < particles -> ntiles.x; tx ++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                dep_pha1_kernel<phasespace::r>  (
                    tile_idx, 
                    d_data, range, size, norm, 
                    *particles
                );
            }
        }
        break;
    case( phasespace:: ux ):
        for( unsigned ty = 0; ty < particles -> ntiles.y; ty ++ ) {
            for( unsigned tx = 0; tx < particles -> ntiles.x; tx ++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                dep_pha1_kernel<phasespace::ux>  (
                    tile_idx, 
                    d_data, range, size, norm, 
                    *particles
                );
            }
        }
        break;
    case( phasespace:: uy ):
        for( unsigned ty = 0; ty < particles -> ntiles.y; ty ++ ) {
            for( unsigned tx = 0; tx < particles -> ntiles.x; tx ++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                dep_pha1_kernel<phasespace::uy>  (
                    tile_idx, 
                    d_data, range, size, norm, 
                    *particles
                );
            }
        }
        break;
    case( phasespace:: uz ):
        for( unsigned ty = 0; ty < particles -> ntiles.y; ty ++ ) {
            for( unsigned tx = 0; tx < particles -> ntiles.x; tx ++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                dep_pha1_kernel<phasespace::uz>  (
                    tile_idx, 
                    d_data, range, size, norm, 
                    *particles
                );
            }
        }
        break;
    };
}

/**
 * @brief Save 1D phasespace
 * 
 * @param q         Phasespace quantity
 * @param range     Phasespace range
 * @param size      Phasespace grid size
 */
void Species::save_phasespace( phasespace::quant quant, float2 const range, 
    int const size ) const
{
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

    if ( quant == phasespace::z ) {
        axis.min += moving_window.motion();
        axis.max += moving_window.motion();
    }

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
    float * d_data = memory::malloc<float>( size );

    dep_phasespace( d_data, quant, range, size );

    // Save file
    zdf::save_grid( d_data, info, iter_info, "PHASESPACE/" + name );

    memory::free( d_data );
}

/**
 * @brief kernel for depositing 2D phasespace
 * 
 * @tparam q0       Quantity 0
 * @tparam q1       Quantity 1
 * @param d_data    Ouput data
 * @param range0    Range of values of quantity 0
 * @param size0     Phasespace grid size for quantity 0
 * @param range1    Range of values of quantity 1
 * @param size1     Range of values of quantity 1
 * @param norm      Normalization factor
 * @param part      Particle data
 */
template < phasespace::quant quant0, phasespace::quant quant1 >
void dep_pha2_kernel(
    uint2 const tile_idx,
    float * const __restrict__ d_data, 
    float2 const range0, int const size0,
    float2 const range1, int const size1,
    float const norm, 
    ParticleData const part )
{
    static_assert( quant1 > quant0, "quant1 must be > quant0" );
    
    const uint2 ntiles  = part.ntiles;
    const auto tile_nx  = part.nx;

    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    const int part_offset = part.offset[ tid ];
    const int np          = part.np[ tid ];
    auto * __restrict__ ix  = &part.ix[ part_offset ];
    auto * __restrict__ x   = &part.x[ part_offset ];
    auto * __restrict__ u   = &part.u[ part_offset ];
    auto * __restrict__ q   = &part.q[ part_offset ];

    float const pha_rdx0 = size0 / (range0.y - range0.x);
    float const pha_rdx1 = size1 / (range1.y - range1.x);

    const int shiftx = tile_idx.x * tile_nx.x;
    const int shifty = tile_idx.y * tile_nx.y;

    for( int i = 0; i < np; i++ ) {
        float d0;
        if constexpr ( quant0 == phasespace:: z )  d0 = ( shiftx + ix[i].x) + (x[i].x + 0.5f);
        if constexpr ( quant0 == phasespace:: r )  d0 = ( shifty + ix[i].y) + x[i].y;
        if constexpr ( quant0 == phasespace:: ux ) d0 = u[i].x;
        if constexpr ( quant0 == phasespace:: uy ) d0 = u[i].y;
        if constexpr ( quant0 == phasespace:: uz ) d0 = u[i].z;

        float n0 =  (d0 - range0.x ) * pha_rdx0 - 0.5f;
        int   k0 = int( n0 + 1 ) - 1;
        float w0 = n0 - k0;

        float d1;
        // if constexpr ( quant1 == phasespace:: z )  d1 = ( shiftx + ix[i].x) + (x[i].x + 0.5f);
        if constexpr ( quant1 == phasespace:: r )  d1 = ( shifty + ix[i].y) + x[i].y;
        if constexpr ( quant1 == phasespace:: ux ) d1 = u[i].x;
        if constexpr ( quant1 == phasespace:: uy ) d1 = u[i].y;
        if constexpr ( quant1 == phasespace:: uz ) d1 = u[i].z;

        float n1 =  (d1 - range1.x ) * pha_rdx1 - 0.5f;
        int   k1 = int( n1 + 1 ) - 1;
        float w1 = n1 - k1;

        // When using multi-threading these need to atomic accross tiles
        if ((k0   >= 0) && (k0   < size0-1) && (k1   >= 0) && (k1   < size1-1))
            d_data[(k1  )*size0 + k0  ] += (1-w0) * (1-w1) * norm * q[i];
        if ((k0+1 >= 0) && (k0+1 < size0-1) && (k1   >= 0) && (k1   < size1-1))
            d_data[(k1  )*size0 + k0+1] +=    w0  * (1-w1) * norm * q[i];
        if ((k0   >= 0) && (k0   < size0-1) && (k1+1 >= 0) && (k1+1 < size1-1))
            d_data[(k1+1)*size0 + k0  ] += (1-w0) *    w1  * norm * q[i];
        if ((k0+1 >= 0) && (k0+1 < size0-1) && (k1+1 >= 0) && (k1+1 < size1-1))
            d_data[(k1+1)*size0 + k0+1] +=    w0  *    w1  * norm * q[i];
    }
}


/**
 * @brief Deposits a 2D phasespace
 * 
 * @param d_data    Pointer to buffer
 * @param quant0    Quantity 0
 * @param range0    Range of values of quantity 0
 * @param size0     Phasespace grid size for quantity 0
 * @param quant0    Quantity 1
 * @param range1    Range of values of quantity 1
 * @param size1     Phasespace grid size for quantity 1
 */
void Species::dep_phasespace( 
    float * const d_data,
    phasespace::quant quant0, float2 range0, unsigned const size0,
    phasespace::quant quant1, float2 range1, unsigned const size1 ) const
{

    // Zero device memory
    memory::zero( d_data, size0 * size1 );

    float norm = ( dx.x * dx.y ) *
                 ( size0 / (range0.y - range0.x) ) *
                 ( size1 / (range1.y - range1.x) );

    switch(quant0) {
    case( phasespace::z ):
        range0.y /= dx.x;
        range0.x /= dx.x;
        switch(quant1) {
        case( phasespace::r ):
            range1.y /= dx.y;
            range1.x /= dx.y;
            for( unsigned ty = 0; ty < particles -> ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < particles -> ntiles.x; tx ++ ) {
                    const auto tile_idx = make_uint2( tx, ty );
                    dep_pha2_kernel<phasespace::z,phasespace::r> (
                        tile_idx, 
                        d_data, range0, size0, range1, size1, norm, 
                        *particles
                    );
                }
            }
            break;
        case( phasespace::ux ):
            for( unsigned ty = 0; ty < particles -> ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < particles -> ntiles.x; tx ++ ) {
                    const auto tile_idx = make_uint2( tx, ty );
                    dep_pha2_kernel<phasespace::z,phasespace::ux> (
                        tile_idx, 
                        d_data, range0, size0, range1, size1, norm, 
                        *particles
                    );
                }
            }
            break;
        case( phasespace::uy ):
            for( unsigned ty = 0; ty < particles -> ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < particles -> ntiles.x; tx ++ ) {
                    const auto tile_idx = make_uint2( tx, ty );
                    dep_pha2_kernel<phasespace::z,phasespace::uy> (
                        tile_idx, 
                        d_data, range0, size0, range1, size1, norm, 
                        *particles
                    );
                }
            }
            break;
        case( phasespace::uz ):
            for( unsigned ty = 0; ty < particles -> ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < particles -> ntiles.x; tx ++ ) {
                    const auto tile_idx = make_uint2( tx, ty );
                    dep_pha2_kernel<phasespace::z,phasespace::uz> (
                        tile_idx, 
                        d_data, range0, size0, range1, size1, norm, 
                        *particles
                    );
                }
            }
            break;
        default:
            break;
        }
        break;
    case( phasespace:: r ):
        range0.y /= dx.y;
        range0.x /= dx.y;
        switch(quant1) {
        case( phasespace::ux ):
            for( unsigned ty = 0; ty < particles -> ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < particles -> ntiles.x; tx ++ ) {
                    const auto tile_idx = make_uint2( tx, ty );
                    dep_pha2_kernel<phasespace::r,phasespace::ux> (
                        tile_idx, 
                        d_data, range0, size0, range1, size1, norm, 
                        *particles
                    );
                }
            }
            break;
        case( phasespace::uy ):
            for( unsigned ty = 0; ty < particles -> ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < particles -> ntiles.x; tx ++ ) {
                    const auto tile_idx = make_uint2( tx, ty );
                    dep_pha2_kernel<phasespace::r,phasespace::uy> (
                        tile_idx, 
                        d_data, range0, size0, range1, size1, norm, 
                        *particles
                    );
                }
            }
            break;
        case( phasespace::uz ):
            for( unsigned ty = 0; ty < particles -> ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < particles -> ntiles.x; tx ++ ) {
                    const auto tile_idx = make_uint2( tx, ty );
                    dep_pha2_kernel<phasespace::r,phasespace::uz> (
                        tile_idx, 
                        d_data, range0, size0, range1, size1, norm, 
                        *particles
                    );
                }
            }
            break;
        default:
            break;
        }
        break;
    case( phasespace:: ux ):
        switch(quant1) {
        case( phasespace::uy ):
            for( unsigned ty = 0; ty < particles -> ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < particles -> ntiles.x; tx ++ ) {
                    const auto tile_idx = make_uint2( tx, ty );
                    dep_pha2_kernel<phasespace::ux,phasespace::uy> (
                        tile_idx, 
                        d_data, range0, size0, range1, size1, norm, 
                        *particles
                    );
                }
            }
            break;
        case( phasespace::uz ):
            for( unsigned ty = 0; ty < particles -> ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < particles -> ntiles.x; tx ++ ) {
                    const auto tile_idx = make_uint2( tx, ty );
                    dep_pha2_kernel<phasespace::ux,phasespace::uz> (
                        tile_idx, 
                        d_data, range0, size0, range1, size1, norm, 
                        *particles
                    );
                }
            }
            break;
        default:
            break;
        }
        break;
    case( phasespace:: uy ):
        for( unsigned ty = 0; ty < particles -> ntiles.y; ty ++ ) {
            for( unsigned tx = 0; tx < particles -> ntiles.x; tx ++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                dep_pha2_kernel<phasespace::uy,phasespace::uz> (
                    tile_idx, 
                    d_data, range0, size0, range1, size1, norm, 
                    *particles
                );
            }
        }
        break;
    default:
        break;
    };
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
void Species::save_phasespace( 
    phasespace::quant quant0, float2 const range0, int const size0,
    phasespace::quant quant1, float2 const range1, int const size1 )
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

    if ( quant0 == phasespace::z ) {
        axis[0].min += moving_window.motion();
        axis[0].max += moving_window.motion();
    }

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
    float * d_data = memory::malloc<float>( size0 * size1 );

    dep_phasespace( d_data, quant0, range0, size0, quant1, range1, size1 );

    // Save file
    zdf::save_grid( d_data, info, iter_info, "PHASESPACE/" + name );

    memory::free( d_data );
}


