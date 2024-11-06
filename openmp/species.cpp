#include "species.h"
#include <iostream>

#include "simd/simd.h"

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
    return 1.0f/std::sqrt( ops::fma( u.z, u.z, ops::fma( u.y, u.y, ops::fma( u.x, u.x, 1.0f ) ) ) );
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
 * @param b[out]    B field at particleposition
 */
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

    const int hx = x.x < 0;
    const int hy = x.y < 0;

    const int ih = i - hx;
    const int jh = j - hy;

    const float s0xh = (1-hx) - x.x;
    const float s1xh = (  hx) + x.x;

    const float s0yh = (1-hy) - x.y;
    const float s1yh = (  hy) + x.y;


    // Interpolate E field

    e.x = ( E[ih +     j *ystride].x * s0xh + E[ih+1 +     j*ystride].x * s1xh ) * s0y +
          ( E[ih + (j +1)*ystride].x * s0xh + E[ih+1 + (j+1)*ystride].x * s1xh ) * s1y;

    e.y = ( E[i  +     jh*ystride].y * s0x  + E[i+1  +     jh*ystride].y * s1x ) * s0yh +
          ( E[i  + (jh+1)*ystride].y * s0x  + E[i+1  + (jh+1)*ystride].y * s1x ) * s1yh;

    e.z = ( E[i  +     j *ystride].z * s0x  + E[i+1  +     j*ystride].z * s1x ) * s0y +
          ( E[i  + (j +1)*ystride].z * s0x  + E[i+1  + (j+1)*ystride].z * s1x ) * s1y;

    // Interpolate B field
    b.x = ( B[i  +     jh*ystride].x * s0x + B[i+1  +     jh*ystride].x * s1x ) * s0yh +
          ( B[i  + (jh+1)*ystride].x * s0x + B[i+1  + (jh+1)*ystride].x * s1x ) * s1yh;

    b.y = ( B[ih +      j*ystride].y * s0xh + B[ih+1 +      j*ystride].y * s1xh ) * s0y +
          ( B[ih + (j +1)*ystride].y * s0xh + B[ih+1 +  (j+1)*ystride].y * s1xh ) * s1y;

    b.z = ( B[ih +     jh*ystride].z * s0xh + B[ih+1 +     jh*ystride].z * s1xh ) * s0yh +
          ( B[ih + (jh+1)*ystride].z * s0xh + B[ih+1 + (jh+1)*ystride].z * s1xh ) * s1yh;

}


/**
 * @brief Advance momentum using a relativistic Boris pusher.
 * 
 * The momemtum advance in this method is split into 3 parts:
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
 * @brief Advance memntum using a relativistic Boris pusher for high magnetic fields
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
 * @brief Deposit (charge conserving) current for 1 segment inside a cell
 * 
 * @param ix        Particle cell
 * @param x0        Initial particle position
 * @param x1        Final particle position
 * @param qnx       Normalization values for in plane current deposition
 * @param qvz       Out of plane current
 * @param J         current(J) grid (should be in shared memory)
 * @param stride    current(J) grid stride
 */
inline void dep_current_seg(
    const int2 ix, const float2 x0, const float2 x1,
    const float2 qnx, const float qvz,
    float3 * __restrict__ J, const int stride )
{
    const float S0x0 = 0.5f - x0.x;
    const float S0x1 = 0.5f + x0.x;

    const float S1x0 = 0.5f - x1.x;
    const float S1x1 = 0.5f + x1.x;

    const float S0y0 = 0.5f - x0.y;
    const float S0y1 = 0.5f + x0.y;

    const float S1y0 = 0.5f - x1.y;
    const float S1y1 = 0.5f + x1.y;

    const float wl1 = qnx.x * (x1.x - x0.x);
    const float wl2 = qnx.y * (x1.y - x0.y);
    
    const float wp10 = 0.5f*(S0y0 + S1y0);
    const float wp11 = 0.5f*(S0y1 + S1y1);
    
    const float wp20 = 0.5f*(S0x0 + S1x0);
    const float wp21 = 0.5f*(S0x1 + S1x1);

    float * __restrict__ const Js = (float *) (&J[ix.x   + stride* ix.y]);
    int const stride3 = 3 * stride;

    // Reorder for linear access
    //                   y    x  fc

    // When using more than 1 thread per tile all of these need to be atomic
    Js[       0 + 0 + 0 ] += wl1 * wp10;
    Js[       0 + 0 + 1 ] += wl2 * wp20;
    Js[       0 + 0 + 2 ] += qvz * ( S0x0 * S0y0 + S1x0 * S1y0 + (S0x0 * S1y0 - S1x0 * S0y0)/2.0f );

    Js[       0 + 3 + 1 ] += wl2 * wp21;
    Js[       0 + 3 + 2 ] += qvz * ( S0x1 * S0y0 + S1x1 * S1y0 + (S0x1 * S1y0 - S1x1 * S0y0)/2.0f );

    Js[ stride3 + 0 + 0 ] += wl1 * wp11;
    Js[ stride3 + 0 + 2 ] += qvz * ( S0x0 * S0y1 + S1x0 * S1y1 + (S0x0 * S1y1 - S1x0 * S0y1)/2.0f );
    Js[ stride3 + 3 + 2 ] += qvz * ( S0x1 * S0y1 + S1x1 * S1y1 + (S0x1 * S1y1 - S1x1 * S0y1)/2.0f );
}


/**
 * @brief Split particle trajectory into segments fitting in a single cell
 * 
 * @param ix        Initial cell
 * @param x0        Initial position
 * @param x1        Final position
 * @param delta     Particle motion ( used to conserve precision, x1 = x0 + delta )
 * @param qvz       Charge times z velocity
 * @param deltai    [out] Cell motion
 * @param v0_ix     [out] 1st segment cell
 * @param v0_x0     [out] 1st segment initial position
 * @param v0_x1     [out] 1st segment final position
 * @param v0_qvz    [out] 1st segment charge times velocity
 * @param v1_ix     [out] 2nd segment cell
 * @param v1_x0     [out] 2nd segment initial position
 * @param v1_x1     [out] 2nd segment final position
 * @param v1_qvz    [out] 2nd segment charge times velocity
 * @param v2_ix     [out] 3rd segment cell
 * @param v2_x0     [out] 3rd segment initial position
 * @param v2_x1     [out] 3rd segment final position
 * @param v2_qvz    [out] 3rd segment charge times velocity
 * @param cross     [out] Cell edge crossing
 */
inline void split2d( 
    const int2 ix, const float2 x0, const float2 x1, const float2 delta, const float qvz,
    int2 & deltai, 
    int2 & v0_ix, float2 & v0_x0, float2 & v0_x1, float & v0_qvz,
    int2 & v1_ix, float2 & v1_x0, float2 & v1_x1, float & v1_qvz,
    int2 & v2_ix, float2 & v2_x0, float2 & v2_x1, float & v2_qvz,
    int2 & cross
) {
    deltai = make_int2(
        ((x1.x >= 0.5f) - (x1.x < -0.5f)),
        ((x1.y >= 0.5f) - (x1.y < -0.5f))
    );

    cross = make_int2( deltai.x != 0, deltai.y != 0 );

    // This will be the same for any particle
    v0_ix = ix; v0_x0 = x0;

    // assume no splits, fill in other options later
    v0_x1 = x1; v0_qvz = qvz;

    // Avoid "may be used uninitialized" warnings
    v1_ix.x = v1_ix.y = v2_ix.x = v2_ix.y = 0;
    v1_x0.x = v1_x0.y = v2_x0.x = v2_x0.y = 0;
    v1_x1.x = v1_x1.y = v2_x1.x = v2_x1.y = 0;
    v1_qvz = v2_qvz = 0;

    // y-cross only
    if ( cross.y && ! cross.x ) {
        float yint = 0.5f * deltai.y;
        float eps  = ( yint - x0.y ) / delta.y;
        float xint = x0.x + delta.x * eps;

        v0_x1  = make_float2(xint,yint);
        v0_qvz = qvz * eps;

        v1_ix = make_int2( ix.x, ix.y  + deltai.y );
        v1_x0 = make_float2(xint,-yint);
        v1_x1 = make_float2( x1.x, x1.y  - deltai.y );
        v1_qvz = qvz * (1-eps);
    }

    // x-cross
    if ( cross.x ) {
        float xint = 0.5f * deltai.x;
        float eps  = ( xint - x0.x ) / delta.x;
        float yint = x0.y + delta.y * eps;

        v0_x1 = make_float2(xint,yint);
        v0_qvz = qvz * eps;

        v1_ix = make_int2( ix.x + deltai.x, ix.y);
        v1_x0 = make_float2(-xint,yint);
        v1_x1 = make_float2( x1.x - deltai.x, x1.y );
        v1_qvz = qvz * (1-eps);

        // handle additional y-cross, if need be
        if ( cross.y ) {
            float yint2 = 0.5f * deltai.y;

            if ( yint >= -0.5f && yint < 0.5f ) {
                // y crosssing on 2nd vp
                eps   = (yint2 - yint) / (x1.y - yint );
                float xint2 = -xint + (x1.x - xint ) * eps;
                
                v2_ix = make_int2( v1_ix.x, v1_ix.y + deltai.y );
                v2_x0 = make_float2(xint2,-yint2);
                v2_x1 = make_float2( v1_x1.x, v1_x1.y - deltai.y );
                v2_qvz = v1_qvz * (1-eps);

                // Correct other particle
                v1_x1 = make_float2(xint2,yint2);
                v1_qvz *= eps;
            } else {
                // y crossing on 1st vp
                eps   = (yint2 - x0.y) / ( yint - x0.y );
                float xint2 = x0.x + ( xint - x0.x ) * eps;

                v2_ix = make_int2( v0_ix.x, v0_ix.y + deltai.y );
                v2_x0 = make_float2( xint2,-yint2);
                v2_x1 = make_float2( v0_x1.x, v0_x1.y - deltai.y );
                v2_qvz = v0_qvz * (1-eps);

                // Correct other particles
                v0_x1 = make_float2(xint2,yint2);
                v0_qvz *= eps;

                v1_ix.y += deltai.y;
                v1_x0.y -= deltai.y;
                v1_x1.y -= deltai.y;
            }
        }
    }
}


/*
 * Vector (SIMD) optimized functions
 */

#ifdef SIMD

/**
 * @brief Returns reciprocal Lorentz gamma factor
 * 
 * $ \frac{1}{\sqrt{u_x^2 + u_y^2 + u_z^2 + 1 }} $
 * 
 * @param u         Generalized momentum in units of c
 * @return float    Reciprocal Lorentz gamma factor
 */
inline vfloat rgamma( const vfloat3 u ) {
    vfloat c1 = vec_float(1);
    return vec_div( c1, vec_fmadd( u.z, u.z,
                        vec_fmadd( u.y, u.y,
                        vec_fmadd( u.x, u.x, c1 ) ) ) );
}

/**
 * @brief Advance momentum using a relativistic Boris pusher with SIMD
 *        operations.
 * 
 * @note For details on the algorithm check `dudt_boris(()`
 * 
 * @param alpha     (simd vector) 
 * @param e         (simd vector) E-field interpolated at the particle position
 * @param b         (simd vector) B-field interpolated at the particle position
 * @param u         (simd vector) generalized veloecity
 * @param energy    Total particle energy. The time centered energy of the particles
 *                  will be added to this variable.
 * @return vfloat3  (simd vector) Updated generalized velocity
 */
inline vfloat3 vdudt_boris( const vfloat alpha, vfloat3 e, vfloat3 b, vfloat3 u, double & energy )
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
        energy += vec_reduce_add( vec_div( utsq, vec_add (gamma , 1.0f) ) );

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
 * @brief Deposit (charge conserving) current for 1 segment inside a cell (vector version)
 * 
 * @note Current will be deposited for all vector elements
 * 
 * @param ix        Particle cell
 * @param x0        Initial particle position
 * @param x1        Final particle position
 * @param qnx       Normalization values for in plane current deposition
 * @param qvz       Out of plane current
 * @param J         current(J) grid (should be in shared memory)
 * @param stride    current(J) grid stride 
 */
void vdep_current_seg(
    const vint2 ix, const vfloat2 x0, const vfloat2 x1,
    const vfloat2 qnx, const vfloat qvz,
    float3 * __restrict__ J, const int stride )
{
    vfloat c1_2  = vec_float(.5);

    vfloat S0x0 = vec_sub( c1_2, x0.x );
    vfloat S0x1 = vec_add( c1_2, x0.x );

    vfloat S1x0 = vec_sub( c1_2, x1.x );
    vfloat S1x1 = vec_add( c1_2, x1.x );

    vfloat S0y0 = vec_sub( c1_2, x0.y );
    vfloat S0y1 = vec_add( c1_2, x0.y );

    vfloat S1y0 = vec_sub( c1_2, x1.y );
    vfloat S1y1 = vec_add( c1_2, x1.y );

    vfloat wl1 = vec_mul( qnx.x, vec_sub(x1.x, x0.x) );
    vfloat wl2 = vec_mul( qnx.y, vec_sub(x1.y, x0.y) );
    vfloat wp10 = vec_mul( c1_2, vec_add(S0y0, S1y0) );
    vfloat wp11 = vec_mul( c1_2, vec_add(S0y1, S1y1) );
    vfloat wp20 = vec_mul( c1_2, vec_add(S0x0, S1x0) );
    vfloat wp21 = vec_mul( c1_2, vec_add(S0x1, S1x1) );


    VecFloat_s a00x = vec_mul( wl1, wp10 );
    VecFloat_s a00y = vec_mul( wl2, wp20 );

    // a00z = qvz * ( S0x0 * S0y0 + S1x0 * S1y0 + (S0x0 * S1y0 - S1x0 * S0y0)/2.0f );
    VecFloat_s a00z = vec_mul( qvz, vec_fmadd( S0x0, S0y0, vec_fmadd( S1x0, S1y0, vec_mul( c1_2, vec_fmsub( S0x0, S1y0, vec_mul( S1x0, S0y0 ) ) ) ) ) );

    VecFloat_s a01y = vec_mul( wl2, wp21 );
    // a01z = qvz * ( S0x1 * S0y0 + S1x1 * S1y0 + (S0x1 * S1y0 - S1x1 * S0y0)/2.0f );
    VecFloat_s a01z = vec_mul( qvz, vec_fmadd( S0x1, S0y0, vec_fmadd( S1x1, S1y0, vec_mul( c1_2, vec_fmsub( S0x1, S1y0, vec_mul( S1x1, S0y0 ) ) ) ) ) );

    VecFloat_s a10x = vec_mul( wl1, wp11 );
    // a10z = qvz * ( S0x0 * S0y1 + S1x0 * S1y1 + (S0x0 * S1y1 - S1x0 * S0y1)/2.0f );
    VecFloat_s a10z = vec_mul( qvz, vec_fmadd( S0x0, S0y1, vec_fmadd( S1x0, S1y1, vec_mul( c1_2, vec_fmsub( S0x0, S1y1, vec_mul( S1x0, S0y1 ) ) ) ) ) );
    // a11z = qvz * ( S0x1 * S0y1 + S1x1 * S1y1 + (S0x1 * S1y1 - S1x1 * S0y1)/2.0f );
    VecFloat_s a11z = vec_mul( qvz, vec_fmadd( S0x1, S0y1, vec_fmadd( S1x1, S1y1, vec_mul( c1_2, vec_fmsub( S0x1, S1y1, vec_mul( S1x1, S0y1 ) ) ) ) ) );

    VecInt_s idx = vec_add( ix.x, vec_mul( ix.y, stride ) );

    for ( int i = 0; i < vecwidth; i++ ) {

        float * __restrict__ Js = (float *) (&J[idx.extract(i)]);
        int const stride3 = 3 * stride;

        Js[       0 + 0 + 0 ] += a00x.extract(i);
        Js[       0 + 0 + 1 ] += a00y.extract(i);
        Js[       0 + 0 + 2 ] += a00z.extract(i);

        Js[       0 + 3 + 1 ] += a01y.extract(i);
        Js[       0 + 3 + 2 ] += a01z.extract(i);

        Js[ stride3 + 0 + 0 ] += a10x.extract(i);
        Js[ stride3 + 0 + 2 ] += a10z.extract(i);

        Js[ stride3 + 3 + 2 ] += a11z.extract(i);

    }
}

/**
 * @brief Deposit (charge conserving) current for 1 segment inside a cell using a mask
 * 
 * @param ix        Particle cell
 * @param x0        Initial particle position
 * @param x1        Final particle position
 * @param qnx       Normalization values for in plane current deposition
 * @param qvz       Out of plane current
 * @param J         current(J) grid (should be in shared memory)
 * @param stride    current(J) grid stride
 * @param mask      Vector mask, current will only be deposited for mask elements equal true
 */
inline void vdep_current_seg(
    const vint2 ix, const vfloat2 x0, const vfloat2 x1,
    const vfloat2 qnx, const vfloat qvz,
    float3 * __restrict__ J, const int stride,
    const vmask mask )
{
    if ( vec_any( mask ) ) {
        vfloat c1_2  = vec_float(.5);

        vfloat S0x0 = vec_sub( c1_2, x0.x );
        vfloat S0x1 = vec_add( c1_2, x0.x );

        vfloat S1x0 = vec_sub( c1_2, x1.x );
        vfloat S1x1 = vec_add( c1_2, x1.x );

        vfloat S0y0 = vec_sub( c1_2, x0.y );
        vfloat S0y1 = vec_add( c1_2, x0.y );

        vfloat S1y0 = vec_sub( c1_2, x1.y );
        vfloat S1y1 = vec_add( c1_2, x1.y );

        vfloat wl1 = vec_mul( qnx.x, vec_sub(x1.x, x0.x) );
        vfloat wl2 = vec_mul( qnx.y, vec_sub(x1.y, x0.y) );
        vfloat wp10 = vec_mul( c1_2, vec_add(S0y0, S1y0) );
        vfloat wp11 = vec_mul( c1_2, vec_add(S0y1, S1y1) );
        vfloat wp20 = vec_mul( c1_2, vec_add(S0x0, S1x0) );
        vfloat wp21 = vec_mul( c1_2, vec_add(S0x1, S1x1) );

        VecFloat_s a00x = vec_mul( wl1, wp10 );
        VecFloat_s a00y = vec_mul( wl2, wp20 );

        // a00z = qvz * ( S0x0 * S0y0 + S1x0 * S1y0 + (S0x0 * S1y0 - S1x0 * S0y0)/2.0f );
        VecFloat_s a00z = vec_mul( qvz, vec_fmadd( S0x0, S0y0, vec_fmadd( S1x0, S1y0, vec_mul( c1_2, vec_fmsub( S0x0, S1y0, vec_mul( S1x0, S0y0 ) ) ) ) ) );

        VecFloat_s a01y = vec_mul( wl2, wp21 );
        // a01z = qvz * ( S0x1 * S0y0 + S1x1 * S1y0 + (S0x1 * S1y0 - S1x1 * S0y0)/2.0f );
        VecFloat_s a01z = vec_mul( qvz, vec_fmadd( S0x1, S0y0, vec_fmadd( S1x1, S1y0, vec_mul( c1_2, vec_fmsub( S0x1, S1y0, vec_mul( S1x1, S0y0 ) ) ) ) ) );

        VecFloat_s a10x = vec_mul( wl1, wp11 );
        // a10z = qvz * ( S0x0 * S0y1 + S1x0 * S1y1 + (S0x0 * S1y1 - S1x0 * S0y1)/2.0f );
        VecFloat_s a10z = vec_mul( qvz, vec_fmadd( S0x0, S0y1, vec_fmadd( S1x0, S1y1, vec_mul( c1_2, vec_fmsub( S0x0, S1y1, vec_mul( S1x0, S0y1 ) ) ) ) ) );
        // a11z = qvz * ( S0x1 * S0y1 + S1x1 * S1y1 + (S0x1 * S1y1 - S1x1 * S0y1)/2.0f );
        VecFloat_s a11z = vec_mul( qvz, vec_fmadd( S0x1, S0y1, vec_fmadd( S1x1, S1y1, vec_mul( c1_2, vec_fmsub( S0x1, S1y1, vec_mul( S1x1, S0y1 ) ) ) ) ) );

        VecInt_s idx = vec_add( ix.x, vec_mul( ix.y, stride) );
        VecMask_s deposit = mask;

        for ( int i = 0; i < vecwidth; i++ ) {
            // Only deposit particles with active mask
            if ( deposit.extract(i) ) {
                float * __restrict__ const Js = (float *) (&J[idx.extract(i)]);
                int const stride3 = 3 * stride;

                // Reorder for linear access
                //                   y    x  fc

                // When using more than 1 thread per tile all of these need to be atomic
                Js[       0 + 0 + 0 ] += a00x.extract(i);
                Js[       0 + 0 + 1 ] += a00y.extract(i);
                Js[       0 + 0 + 2 ] += a00z.extract(i);

                Js[       0 + 3 + 1 ] += a01y.extract(i);
                Js[       0 + 3 + 2 ] += a01z.extract(i);

                Js[ stride3 + 0 + 0 ] += a10x.extract(i);
                Js[ stride3 + 0 + 2 ] += a10z.extract(i);

                Js[ stride3 + 3 + 2 ] += a11z.extract(i);
            }
        }
    }
}


/**
 * @brief Split particle trajectory into segments fitting in a single cell (vector version)
 * 
 * @param ix        Initial cell
 * @param x0        Initial position
 * @param x1        Final position
 * @param delta     Particle motion ( used to conserve precision, x1 = x0 + delta )
 * @param qvz       Charge times z velocity
 * @param deltai    [out] Cell motion
 * @param v0_ix     [out] 1st segment cell
 * @param v0_x0     [out] 1st segment initial position
 * @param v0_x1     [out] 1st segment final position
 * @param v0_qvz    [out] 1st segment charge times velocity
 * @param v1_ix     [out] 2nd segment cell
 * @param v1_x0     [out] 2nd segment initial position
 * @param v1_x1     [out] 2nd segment final position
 * @param v1_qvz    [out] 2nd segment charge times velocity
 * @param v2_ix     [out] 3rd segment cell
 * @param v2_x0     [out] 3rd segment initial position
 * @param v2_x1     [out] 3rd segment final position
 * @param v2_qvz    [out] 3rd segment charge times velocity
 * @param cross     [out] Cell edge crossing
 */
inline void vsplit2d( 
    const vint2 ix, const vfloat2 x0, const vfloat2 x1, const vfloat2 delta, const vfloat qvz,
    vint2 & deltai,
    vint2 & v0_ix, vfloat2 & v0_x0, vfloat2 & v0_x1, vfloat & v0_qvz,
    vint2 & v1_ix, vfloat2 & v1_x0, vfloat2 & v1_x1, vfloat & v1_qvz,
    vint2 & v2_ix, vfloat2 & v2_x0, vfloat2 & v2_x1, vfloat & v2_qvz,
    vmask2 & cross
) {
    const vint   c1i   = vec_int(1);
    const vfloat c1_2  = vec_float(.5);
    const vfloat cm1_2 = vec_float(-.5);

    // Number of cells moved: -1, 0 or 1
    deltai = {
        vec_sub( vec_ge( x1.x, c1_2, c1i ), vec_lt( x1.x, cm1_2, c1i ) ),
        vec_sub( vec_ge( x1.y, c1_2, c1i ), vec_lt( x1.y, cm1_2, c1i ) )
    };
    cross.x = vec_ne( deltai.x, vec_zero_int() );
    cross.y = vec_ne( deltai.y, vec_zero_int() );

    // Convert to float for operations below
    const vfloat2 fdeltai = {
        vec_float( deltai.x ),
        vec_float( deltai.y )
    };

    // This will be the same for any particle
    v0_ix = ix; v0_x0 = x0;

    // assume no splits, fill in other options later
    v0_x1 = x1; v0_qvz = qvz;

    
    // Avoid "may be used uninitialized" warnings
    v1_ix.x = v1_ix.y = v2_ix.x = v2_ix.y = vec_zero_int();
    v1_x0.x = v1_x0.y = v2_x0.x = v2_x0.y = vec_zero_float();
    v1_x1.x = v1_x1.y = v2_x1.x = v2_x1.y = vec_zero_float();
    v1_qvz = v2_qvz = vec_zero_float();

    // y-cross only
    vmask y_only = vec_and( cross.y, vec_not( cross.x ));
    if ( vec_any( y_only ) ) {
        vfloat yint = vec_mul( c1_2, fdeltai.y );
        vfloat eps  = vec_div( vec_sub( yint, x0.y ), delta.y );
        vfloat xint = vec_fmadd( eps, delta.x, x0.x );

        v0_x1.x  = vec_select( v0_x1.x, xint, y_only );
        v0_x1.y  = vec_select( v0_x1.y, yint, y_only );
        v0_qvz = vec_select( v0_qvz, vec_mul( qvz, eps ), y_only );

        // No need to use select for v1
        v1_ix.x = ix.x;  v1_ix.y = vec_add( ix.y, deltai.y );
        v1_x0.x = xint;  v1_x0.y = vec_neg( yint );
        v1_x1.x = x1.x;  v1_x1.y = vec_sub( x1.y, fdeltai.y );
        v1_qvz = vec_fnmadd( eps, qvz, qvz );
    }

    // xcross
    if ( vec_any( cross.x ) ) {
        vfloat xint = vec_mul( c1_2, fdeltai.x );
        vfloat eps  = vec_div( vec_sub( xint, x0.x ), delta.x );
        vfloat yint = vec_fmadd( eps, delta.y, x0.y );

        v0_x1.x  = vec_select( v0_x1.x, xint, cross.x );
        v0_x1.y  = vec_select( v0_x1.y, yint, cross.x );
        v0_qvz = vec_select( v0_qvz, vec_mul( qvz, eps ), cross.x );

        v1_ix.x = vec_select( v1_ix.x, vec_add( ix.x, deltai.x ), cross.x ); 
        v1_ix.y = vec_select( v1_ix.y,                      ix.y, cross.x );

        v1_x0.x = vec_select( v1_x0.x, vec_neg(xint), cross.x );
        v1_x0.y = vec_select( v1_x0.y,          yint, cross.x );

        v1_x1.x = vec_select( v1_x1.x, vec_sub( x1.x, fdeltai.x ), cross.x );
        v1_x1.y = vec_select( v1_x1.y,                       x1.y, cross.x );

        v1_qvz = vec_select( v1_qvz, vec_fnmadd( eps, qvz, qvz ), cross.x );
    
        // additional ycross
        vmask xycross = vec_and( cross.x, cross.y );
        if ( vec_any(xycross) ) {
            vfloat yint2, xint2;
            yint2  = vec_mul( c1_2, fdeltai.y );

            vmask ycross2 = vec_and( vec_ge( yint, cm1_2 ), vec_lt( yint, c1_2 ) );

            // Assume the additional y-cross (if any) was on the 2nd vp
            // we store all values of v2 and correct later if need be
            vmask xycross2;
            xycross2 = vec_and( xycross, ycross2 );

            eps = vec_div( vec_sub( yint2, yint ), vec_sub( x1.y, yint ));
            xint2 = vec_fmsub( vec_sub( x1.x, xint ), eps, xint );
            v2_ix.x = v1_ix.x;
            v2_ix.y = vec_add( v1_ix.y, deltai.y );
            v2_x0.x = xint2;
            v2_x0.y = vec_neg( yint2 );
            v2_x1.x = v1_x1.x;
            v2_x1.y = vec_sub( v1_x1.y, fdeltai.y );
            v2_qvz = vec_fnmadd( eps, v1_qvz, v1_qvz );

            v1_x1.x  = vec_select( v1_x1.x, xint2, xycross2 );
            v1_x1.y  = vec_select( v1_x1.y, yint2, xycross2 );
            v1_qvz = vec_select( v1_qvz, vec_mul( v1_qvz, eps ), xycross2 );

            // Assume the additonal ycross (if any) was on the 1st vp
            xycross2 = vec_and( xycross, vec_not( ycross2 ));
            eps = vec_div( vec_sub( yint2, x0.y ), vec_sub( yint, x0.y ));
            xint2 = vec_fmadd( vec_sub( xint, x0.x ), eps, x0.x );

            v2_ix.x = vec_select( v2_ix.x, v0_ix.x,                      xycross2 );
            v2_ix.y = vec_select( v2_ix.y, vec_add( v0_ix.y, deltai.y ), xycross2 );

            v2_x0.x = vec_select( v2_x0.x, xint2,          xycross2 );
            v2_x0.y = vec_select( v2_x0.y, vec_neg(yint2), xycross2 );

            v2_x1.x = vec_select( v2_x1.x, v0_x1.x,                         xycross2 );
            v2_x1.y = vec_select( v2_x1.y, vec_sub( v0_x1.y, fdeltai.y ),   xycross2 );
            v2_qvz = vec_select( v2_qvz, vec_fnmadd( eps, v0_qvz, v0_qvz ), xycross2 );

            v0_x1.x = vec_select( v0_x1.x, xint2, xycross2 );
            v0_x1.y = vec_select( v0_x1.y, yint2, xycross2 );
            v0_qvz = vec_select( v0_qvz, vec_mul( v0_qvz, eps ), xycross2 );

            v1_ix.y = vec_select( v1_ix.y, vec_add( v1_ix.y, deltai.y  ), xycross2 );
            v1_x0.y = vec_select( v1_x0.y, vec_sub( v1_x0.y, fdeltai.y ), xycross2 );
            v1_x1.y = vec_select( v1_x1.y, vec_sub( v1_x1.y, fdeltai.y ), xycross2 );
        }
    }
}

/**
 * @brief Move particles and deposit current (vector version)
 * 
 * @param tile_idx          Tile index
 * @param part              Particle data
 * @param d_current         Current grid (global)
 * @param current_offset    Offset to position [0,0] of the current grid
 * @param ext_nx            Current grid size (external)
 * @param dt_dx             Ratio between time step and cell size
 * @param q                 Particle charge
 * @param qnx               Current normalization
 */
void move_deposit_kernel(
    uint2 const tile_idx,
    ParticleData const part,
    float3 * const __restrict__ d_current, unsigned int const current_offset, uint2 const ext_nx,
    float2 const dt_dx, float const q, float2 const qnx ) 
{
    const uint2 ntiles  = part.ntiles;

    const int tile_size = roundup4( ext_nx.x * ext_nx.y );

    // The alignment also avoids some optimization related segfaults
    alignas(local_align) float3 _move_deposit_buffer[ tile_size ];


    // Zero local current buffer
    for( auto i = 0; i < tile_size; i++ ) 
        _move_deposit_buffer[i] = make_float3(0,0,0);

    // sync

    // Move particles and deposit current
    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    float3 * J = _move_deposit_buffer + current_offset;
    const int ystride = ext_nx.x;

    const int part_offset    = part.offset[ tid ];
    const int np             = part.np[ tid ];
    int2   * __restrict__ ix = &part.ix[ part_offset ];
    float2 * __restrict__ x  = &part.x[ part_offset ];
    float3 * __restrict__ u  = &part.u[ part_offset ];

    const vfloat2 vqnx = {
        vec_float( qnx.x ),
        vec_float( qnx.y )
    };

    const int np_vec = (np/vecwidth) * vecwidth;
    for( int i = 0; i < np_vec; i+= vecwidth ) {
        vfloat3 pu        = vec_load_s3( (float *) & u[i] );
        vfloat2 const x0  = vec_load_s2( (float *) & x[i] );
        vint2   const ix0 = vec_load_s2( (int *) & ix[i] );

        // Get 1 / Lorentz gamma
        vfloat const rg = rgamma( pu );

        // Get particle motion
        vfloat2 const delta = {
            vec_mul( vec_mul( rg, pu.x ), dt_dx.x ),
            vec_mul( vec_mul( rg, pu.y ), dt_dx.y )
        };

        vfloat qvz = vec_mul( vec_mul( pu.z, rg ), q * 0.5f );

        // Advance position
        vfloat2 x1 = {
            vec_add( x0.x , delta.x ),
            vec_add( x0.y , delta.y )
        };

        // Check for cell crossings and split trajectory
        vint2 deltai; vmask2 cross;

        vint2 v0_ix; vfloat2 v0_x0, v0_x1; vfloat v0_qvz;
        vint2 v1_ix; vfloat2 v1_x0, v1_x1; vfloat v1_qvz;
        vint2 v2_ix; vfloat2 v2_x0, v2_x1; vfloat v2_qvz;

        vsplit2d( 
            ix0, x0, x1, delta, qvz, 
            deltai,
            v0_ix, v0_x0, v0_x1, v0_qvz,
            v1_ix, v1_x0, v1_x1, v1_qvz,
            v2_ix, v2_x0, v2_x1, v2_qvz,
            cross
        );
        
        // Deposit current
        vdep_current_seg( v0_ix, v0_x0, v0_x1, vqnx, v0_qvz, J, ystride );
        vdep_current_seg( v1_ix, v1_x0, v1_x1, vqnx, v1_qvz, J, ystride, vec_or(  cross.x, cross.y ) );
        vdep_current_seg( v2_ix, v2_x0, v2_x1, vqnx, v2_qvz, J, ystride, vec_and( cross.x, cross.y ) );

        // Correct cell position and store
        x1.x = vec_sub( x1.x, vec_float( deltai.x ) );
        x1.y = vec_sub( x1.y, vec_float( deltai.y ) );
        vec_store_s2( (float *) & x[i], x1 );

        // Modify cell and store
        vint2 ix1 = {
            vec_add( ix0.x, deltai.x ),
            vec_add( ix0.y, deltai.y )
        };
        vec_store_s2( (int *) & ix[i], ix1 );
    }

    // Process remaining particles using serial code
    for( int i = np_vec; i < np; i += 1 ) {
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

        float qvz = q * pu.z * rg * 0.5f;

        // Advance position
        float2 x1 = make_float2(
            x0.x + delta.x,
            x0.y + delta.y
        );

        // Check for cell crossings and split trajectory
        int2 deltai, cross;

        int2 v0_ix; float2 v0_x0, v0_x1; float v0_qvz;
        int2 v1_ix; float2 v1_x0, v1_x1; float v1_qvz;
        int2 v2_ix; float2 v2_x0, v2_x1; float v2_qvz;

        split2d( ix0, x0, x1, delta, qvz, deltai,
            v0_ix, v0_x0, v0_x1, v0_qvz,
            v1_ix, v1_x0, v1_x1, v1_qvz,
            v2_ix, v2_x0, v2_x1, v2_qvz,
            cross );
        
        // Deposit current
                                  dep_current_seg( v0_ix, v0_x0, v0_x1, qnx, v0_qvz, J, ystride );
        if ( cross.x || cross.y ) dep_current_seg( v1_ix, v1_x0, v1_x1, qnx, v1_qvz, J, ystride );
        if ( cross.x && cross.y ) dep_current_seg( v2_ix, v2_x0, v2_x1, qnx, v2_qvz, J, ystride );

        // Correct cell position and store
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

    // Add current to global buffer
    const int tile_off = tid * tile_size;

    for( unsigned i = 0; i < ext_nx.x * ext_nx.y; i++ ) {
        d_current[tile_off + i] += _move_deposit_buffer[i];
    }
}

/**
 * @brief Move particles, deposit current and shift positions
 * 
 * @param tile_idx          Tile index
 * @param part              Particle data
 * @param d_current         Current grid (global)
 * @param current_offset    Offset to position [0,0] of the current grid
 * @param ext_nx            Current grid size (external)
 * @param dt_dx             Ratio between time step and cell size
 * @param q                 Particle charge
 * @param qnx               Current normalization 
 * @param shift             Position shift
 */
void move_deposit_shift_kernel(
    uint2 const tile_idx,
    ParticleData const part,
    float3 * const __restrict__ d_current, unsigned int const current_offset, uint2 const ext_nx,
    float2 const dt_dx, float const q, float2 const qnx, int2 const shift ) 
{
    const uint2 ntiles  = part.ntiles;

    const int tile_size = roundup4( ext_nx.x * ext_nx.y );

    alignas(local_align) float3 _move_deposit_buffer[ tile_size ];

    // Zero local current buffer
    for( auto i = 0; i < tile_size; i++ ) 
        _move_deposit_buffer[i] = make_float3(0,0,0);

    // sync

    // Move particles and deposit current
    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    float3 * J = _move_deposit_buffer + current_offset;
    const int ystride = ext_nx.x;

    const int part_offset    = part.offset[ tid ];
    const int np             = part.np[ tid ];
    int2   * __restrict__ ix = &part.ix[ part_offset ];
    float2 * __restrict__ x  = &part.x[ part_offset ];
    float3 * __restrict__ u  = &part.u[ part_offset ];

    const vfloat2 vqnx = {
        vec_float( qnx.x ),
        vec_float( qnx.y )
    };

    const vint2 vshift = {
        vec_int( shift.x ),
        vec_int( shift.y )
    };

    const int np_vec = (np/vecwidth) * vecwidth;
    for( int i = 0; i < np_vec; i+= vecwidth ) {
        vfloat3 pu        = vec_load_s3( (float *) & u[i] );
        vfloat2 const x0  = vec_load_s2( (float *) & x[i] );
        vint2   const ix0 = vec_load_s2( (int *) & ix[i] );

        // Get 1 / Lorentz gamma
        vfloat const rg = rgamma( pu );

        // Get particle motion
        vfloat2 const delta = {
            vec_mul( vec_mul( rg, pu.x ), dt_dx.x ),
            vec_mul( vec_mul( rg, pu.y ), dt_dx.y )
        };

        vfloat qvz = vec_mul( vec_mul( pu.z, rg ), q * 0.5f );

        // Advance position
        vfloat2 x1 = {
            vec_add( x0.x , delta.x ),
            vec_add( x0.y , delta.y )
        };

        // Check for cell crossings and split trajectory
        vint2 deltai;
        vmask2 cross;

        vint2 v0_ix; vfloat2 v0_x0, v0_x1; vfloat v0_qvz;
        vint2 v1_ix; vfloat2 v1_x0, v1_x1; vfloat v1_qvz;
        vint2 v2_ix; vfloat2 v2_x0, v2_x1; vfloat v2_qvz;

        vsplit2d( 
            ix0, x0, x1, delta, qvz, 
            deltai,
            v0_ix, v0_x0, v0_x1, v0_qvz,
            v1_ix, v1_x0, v1_x1, v1_qvz,
            v2_ix, v2_x0, v2_x1, v2_qvz,
            cross
        );
        
        // Deposit current

        vdep_current_seg( v0_ix, v0_x0, v0_x1, vqnx, v0_qvz, J, ystride );
        vdep_current_seg( v1_ix, v1_x0, v1_x1, vqnx, v1_qvz, J, ystride, vec_or(  cross.x, cross.y ) );
        vdep_current_seg( v2_ix, v2_x0, v2_x1, vqnx, v2_qvz, J, ystride, vec_and( cross.x, cross.y ) );

        // Correct cell position and store
        x1.x = vec_sub( x1.x, vec_float( deltai.x ) );
        x1.y = vec_sub( x1.y, vec_float( deltai.y ) );
        vec_store_s2( (float *) & x[i], x1 );

        // Modify cell and store
        vint2 ix1 = {
            vec_add( vec_add( ix0.x, deltai.x ), vshift.x ),
            vec_add( vec_add( ix0.y, deltai.y ), vshift.y )
        };
        vec_store_s2( (int *) & ix[i], ix1 );
    }

    // Process remaining particles using serial code
    for( int i = np_vec; i < np; i += 1 ) {
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

        float qvz = q * pu.z * rg * 0.5f;

        // Advance position
        float2 x1 = make_float2(
            x0.x + delta.x,
            x0.y + delta.y
        );

        // Check for cell crossings and split trajectory
        int2 deltai, cross;

        int2 v0_ix; float2 v0_x0, v0_x1; float v0_qvz;
        int2 v1_ix; float2 v1_x0, v1_x1; float v1_qvz;
        int2 v2_ix; float2 v2_x0, v2_x1; float v2_qvz;

        split2d( ix0, x0, x1, delta, qvz, deltai,
            v0_ix, v0_x0, v0_x1, v0_qvz,
            v1_ix, v1_x0, v1_x1, v1_qvz,
            v2_ix, v2_x0, v2_x1, v2_qvz,
            cross );
        
        // Deposit current
                                  dep_current_seg( v0_ix, v0_x0, v0_x1, qnx, v0_qvz, J, ystride );
        if ( cross.x || cross.y ) dep_current_seg( v1_ix, v1_x0, v1_x1, qnx, v1_qvz, J, ystride );
        if ( cross.x && cross.y ) dep_current_seg( v2_ix, v2_x0, v2_x1, qnx, v2_qvz, J, ystride );

        // Correct cell position and store
        x1.x -= deltai.x;
        x1.y -= deltai.y;
        x[i] = x1;

        // Modify cell and store
        int2 ix1 = make_int2(
            ix0.x + deltai.x + shift.x,
            ix0.y + deltai.y + shift.x
        );
        ix[i] = ix1;
    }

    // Add current to global buffer
    const int tile_off = tid * tile_size;

    for( unsigned i = 0; i < ext_nx.x * ext_nx.y; i++ ) {
         d_current[tile_off + i] += _move_deposit_buffer[i];
    }
}

/**
 * @brief Interpolate EM field values at particle position using linear 
 * (1st order) interpolation using SIMD operations.
 * 
 * @note The EM fields are assumed to be organized according to the Yee scheme with
 * the charge defined at lower left corner of the cell
 * 
 * @param E         Pointer to position (0,0) of E field grid
 * @param B         Pointer to position (0,0) of B field grid
 * @param ystride   E and B grids y stride (must be signed)
 * @param ix        (simd vector) Particle cell index
 * @param x         (simd vector) Particle postion inside cell
 * @param e[out]    (simd vector) E field at particle position
 * @param b[out]    (simd vector) B field at particleposition
 */
inline void vinterpolate_fld( 
    float3 const * const __restrict__ E, 
    float3 const * const __restrict__ B, 
    const int ystride,
    const vint2 ix, const vfloat2 x, vfloat3 & e, vfloat3 & b)
{
    vint i = ix.x;
    vint j = ix.y;

    const vfloat c_1_2 = vec_float( 0.5 );

    const vfloat s0x = vec_sub( c_1_2, x.x );
    const vfloat s1x = vec_add( c_1_2, x.x );

    const vfloat s0y = vec_sub( c_1_2, x.y );
    const vfloat s1y = vec_add( c_1_2, x.y );

    const vint c_1i = vec_int( 1 );

    const vint hx = vec_lt( x.x, vec_zero_float(), c_1i );
    const vint hy = vec_lt( x.y, vec_zero_float(), c_1i );

    vint ih = vec_sub( i, hx );
    vint jh = vec_sub( j, hy );

    const vfloat s0xh = vec_sub( vec_float( vec_sub(c_1i, hx) ), x.x );
    const vfloat s1xh = vec_add( vec_float(               hx  ), x.x );

    const vfloat s0yh = vec_sub( vec_float( vec_sub(c_1i, hy) ), x.y );
    const vfloat s1yh = vec_add( vec_float(               hy  ), x.y );

    // Get linear indices
    int const ystride3 = 3 * ystride;

    i  = vec_mul3( i );
    ih = vec_mul3( ih );
    j  = vec_mul( j,  ystride3 );
    jh = vec_mul( jh, ystride3 );

    vint i_j   = vec_add( i, j );
    vint i_jh  = vec_add( i, jh );

    vint ih_j  = vec_add( ih, j );
    vint ih_jh = vec_add( ih, jh );

    {   // Interpolate E field
        const float * __restrict__ const Es = (float *) (&E[0]);

        vfloat fx, fy, fz;
        fx = vec_mul(   vec_gather( Es     + 0, ih_j ), s0xh );
        fy = vec_mul(   vec_gather( Es     + 1, i_jh ), s0x  );
        fz = vec_mul(   vec_gather( Es     + 2, i_j  ), s0x  );

        fx = vec_fmadd( vec_gather( Es + 3 + 0, ih_j ), s1xh, fx );
        fy = vec_fmadd( vec_gather( Es + 3 + 1, i_jh ), s1x,  fy );
        fz = vec_fmadd( vec_gather( Es + 3 + 2, i_j  ), s1x,  fz );

        e.x = vec_mul( fx, s0y );
        e.y = vec_mul( fy, s0yh );
        e.z = vec_mul( fz, s0y );

        fx = vec_mul(   vec_gather( Es + ystride3     + 0, ih_j ), s0xh );
        fy = vec_mul(   vec_gather( Es + ystride3     + 1, i_jh ), s0x  );
        fz = vec_mul(   vec_gather( Es + ystride3     + 2, i_j  ), s0x  );

        fx = vec_fmadd( vec_gather( Es + ystride3 + 3 + 0, ih_j ), s1xh, fx );
        fy = vec_fmadd( vec_gather( Es + ystride3 + 3 + 1, i_jh ), s1x,  fy );
        fz = vec_fmadd( vec_gather( Es + ystride3 + 3 + 2, i_j  ), s1x,  fz );

        e.x = vec_fmadd( fx, s1y,  e.x );
        e.y = vec_fmadd( fy, s1yh, e.y );
        e.z = vec_fmadd( fz, s1y,  e.z );
    }

    {   // Interpolate B field
        const float * __restrict__ const Bs = (float *) (&B[0]);

        vfloat fx, fy, fz;
        fx = vec_mul(   vec_gather( Bs     + 0, i_jh  ), s0x  );
        fy = vec_mul(   vec_gather( Bs     + 1, ih_j  ), s0xh );
        fz = vec_mul(   vec_gather( Bs     + 2, ih_jh ), s0xh );

        fx = vec_fmadd( vec_gather( Bs + 3 + 0, i_jh  ), s1x,  fx );
        fy = vec_fmadd( vec_gather( Bs + 3 + 1, ih_j  ), s1xh, fy );
        fz = vec_fmadd( vec_gather( Bs + 3 + 2, ih_jh ), s1xh, fz );

        b.x = vec_mul( fx, s0yh );
        b.y = vec_mul( fy, s0y  );
        b.z = vec_mul( fz, s0yh );

        fx = vec_mul(   vec_gather( Bs + ystride3     + 0, i_jh  ), s0x  );
        fy = vec_mul(   vec_gather( Bs + ystride3     + 1, ih_j  ), s0xh );
        fz = vec_mul(   vec_gather( Bs + ystride3     + 2, ih_jh ), s0xh );

        fx = vec_fmadd( vec_gather( Bs + ystride3 + 3 + 0, i_jh  ), s1x,  fx );
        fy = vec_fmadd( vec_gather( Bs + ystride3 + 3 + 1, ih_j  ), s1xh, fy );
        fz = vec_fmadd( vec_gather( Bs + ystride3 + 3 + 2, ih_jh ), s1xh, fz );

        b.x = vec_fmadd( fx, s1yh, b.x );
        b.y = vec_fmadd( fy, s1y,  b.y );
        b.z = vec_fmadd( fz, s1yh, b.z );
    }
}



template < species::pusher type >
void push_kernel ( 
    uint2 const tile_idx,
    ParticleData const part,
    float3 * __restrict__ d_E, float3 * __restrict__ d_B, 
    unsigned int const field_offset, uint2 const ext_nx,
    float const alpha, double * __restrict__ d_energy )
{
    const uint2 ntiles  = part.ntiles;

    // Tile ID
    const int tid =  tile_idx.y * ntiles.x + tile_idx.x;

    int const field_vol = roundup4( ext_nx.x * ext_nx.y );
    int const tile_off = tid * field_vol;

    // Copy E and B into shared memory
    alignas(local_align) float3 E_local[ field_vol ];
    alignas(local_align) float3 B_local[ field_vol ];

    for( auto i = 0; i < field_vol; i++ ) {
        E_local[i] = d_E[tile_off + i];
        B_local[i] = d_B[tile_off + i];
    }

    float3 const * const __restrict__ E = & E_local[ field_offset ];
    float3 const * const __restrict__ B = & B_local[ field_offset ];

    // Push particles
    const int part_offset = part.offset[ tid ];
    const int np          = part.np[ tid ];
    int2   * __restrict__ ix = &part.ix[ part_offset ];
    float2 * __restrict__ x  = &part.x[ part_offset ];
    float3 * __restrict__ u  = &part.u[ part_offset ];

    double energy = 0;

    const int ystride = ext_nx.x;

    const vfloat valpha = vec_float( alpha );

    const int np_vec = (np/vecwidth) * vecwidth;

    for( int i = 0; i < np_vec; i += vecwidth ) {
        vfloat3 e, b;

        const vint2   vix = vec_load_s2( (int *) & ix[i] );
        const vfloat2 vx  = vec_load_s2( (float *) & x[i] );

        vinterpolate_fld( E, B, ystride, vix, vx, e, b );

        vfloat3 pu = vec_load_s3( (float *) & u[i] );
        pu = vdudt_boris( valpha, e, b, pu, energy );
        vec_store_s3( (float *) & u[i], pu );
    }

    // Process remaining particles
    for( int i = np_vec; i < np; i += 1 ) {
        float3 e, b;
        
        int2   pix = ix[i];
        float2 px  = x[i];

        interpolate_fld( E, B, ystride, pix, px, e, b );

        float3 pu = u[i];
        pu = dudt_boris( alpha, e, b, pu, energy );
        u[i] = pu;
    }

    // Add up energy from all particles
    // In OpenMP, d_energy needs to be a reduction variable
    *d_energy += energy;
}

#else

/**
 * @brief Move particles and deposit current
 * 
 * @param tile_idx          Tile index
 * @param part              Particle data
 * @param d_current         Current grid (global)
 * @param current_offset    Offset to position [0,0] of the current grid
 * @param ext_nx            Current grid size (external)
 * @param dt_dx             Ratio between time step and cell size
 * @param q                 Particle charge
 * @param qnx               Current normalization
 */
void move_deposit_kernel(
    uint2 const tile_idx,
    ParticleData const part,
    float3 * const __restrict__ d_current, unsigned int const current_offset, uint2 const ext_nx,
    float2 const dt_dx, float const q, float2 const qnx ) 
{
    const uint2 ntiles  = part.ntiles;

    const int tile_size = roundup4( ext_nx.x * ext_nx.y );

    // This is usually in block shared memeory
    alignas(local_align) float3 _move_deposit_buffer[tile_size];

    // Zero local current buffer
    for( auto i = 0; i < tile_size; i++ ) 
        _move_deposit_buffer[i] = make_float3(0,0,0);

    // sync

    // Move particles and deposit current
    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    float3 * J = _move_deposit_buffer + current_offset;
    const int ystride = ext_nx.x;

    const int part_offset    = part.offset[ tid ];
    const int np             = part.np[ tid ];
    int2   * __restrict__ ix = &part.ix[ part_offset ];
    float2 * __restrict__ x  = &part.x[ part_offset ];
    float3 * __restrict__ u  = &part.u[ part_offset ];

    for( int i = 0; i < np; i++ ) {
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

        float qvz = q * pu.z * rg * 0.5f;

        // Advance position
        float2 x1 = make_float2(
            x0.x + delta.x,
            x0.y + delta.y
        );

        // Check for cell crossings and split trajectory
        int2 deltai, cross;

        int2 v0_ix; float2 v0_x0, v0_x1; float v0_qvz;
        int2 v1_ix; float2 v1_x0, v1_x1; float v1_qvz;
        int2 v2_ix; float2 v2_x0, v2_x1; float v2_qvz;

        split2d( ix0, x0, x1, delta, qvz, deltai,
            v0_ix, v0_x0, v0_x1, v0_qvz,
            v1_ix, v1_x0, v1_x1, v1_qvz,
            v2_ix, v2_x0, v2_x1, v2_qvz,
            cross );
        
        // Deposit current
                                  dep_current_seg( v0_ix, v0_x0, v0_x1, qnx, v0_qvz, J, ystride );
        if ( cross.x || cross.y ) dep_current_seg( v1_ix, v1_x0, v1_x1, qnx, v1_qvz, J, ystride );
        if ( cross.x && cross.y ) dep_current_seg( v2_ix, v2_x0, v2_x1, qnx, v2_qvz, J, ystride );

        // Correct cell position and store
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

    // Add current to global buffer
    const int tile_off = tid * tile_size;

    for( unsigned i = 0; i < ext_nx.x * ext_nx.y; i++ ) {
        d_current[tile_off + i] += _move_deposit_buffer[i];
    }
}

/**
 * @brief Move particles, deposit current and shift positions
 * 
 * @param tile_idx          Tile index
 * @param part              Particle data
 * @param d_current         Current grid (global)
 * @param current_offset    Offset to position [0,0] of the current grid
 * @param ext_nx            Current grid size (external)
 * @param dt_dx             Ratio between time step and cell size
 * @param q                 Particle charge
 * @param qnx               Current normalization 
 * @param shift             Position shift
 */
void move_deposit_shift_kernel(
    uint2 const tile_idx,
    ParticleData const part,
    float3 * const __restrict__ d_current, unsigned int const current_offset, uint2 const ext_nx,
    float2 const dt_dx, float const q, float2 const qnx, int2 const shift ) 
{
    const uint2 ntiles  = part.ntiles;
    const int tile_size = roundup4( ext_nx.x * ext_nx.y );

    // This is usually in block shared memeory
    alignas(local_align) float3 _move_deposit_buffer[tile_size];

    // Zero local current buffer
    for( auto i = 0; i < tile_size; i++ ) 
        _move_deposit_buffer[i] = make_float3(0,0,0);

    // Move particles and deposit current
    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    float3 * J = _move_deposit_buffer + current_offset;
    const int ystride = ext_nx.x;

    const int part_offset     = part.offset[ tid ];
    const int np              = part.np[ tid ];
    int2   * __restrict__ ix  = &part.ix[ part_offset ];
    float2 * __restrict__ x   = &part.x[ part_offset ];
    float3 * __restrict__ u   = &part.u[ part_offset ];

    for( int i = 0; i < np; i++ ) {
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

        float qvz = q * pu.z * rg * 0.5f;

        // Advance position
        float2 x1 = make_float2(
            x0.x + delta.x,
            x0.y + delta.y
        );

        // Check for cell crossings and split trajectory
        int2 deltai, cross;

        int2 v0_ix; float2 v0_x0, v0_x1; float v0_qvz;
        int2 v1_ix; float2 v1_x0, v1_x1; float v1_qvz;
        int2 v2_ix; float2 v2_x0, v2_x1; float v2_qvz;

        split2d( 
            ix0, x0, x1, delta, qvz, deltai,
            v0_ix, v0_x0, v0_x1, v0_qvz,
            v1_ix, v1_x0, v1_x1, v1_qvz,
            v2_ix, v2_x0, v2_x1, v2_qvz,
            cross
        );
        
        // Deposit current
                                  dep_current_seg( v0_ix, v0_x0, v0_x1, qnx, v0_qvz, J, ystride );
        if ( cross.x || cross.y ) dep_current_seg( v1_ix, v1_x0, v1_x1, qnx, v1_qvz, J, ystride );
        if ( cross.x && cross.y ) dep_current_seg( v2_ix, v2_x0, v2_x1, qnx, v2_qvz, J, ystride );

        // Correct cell position and store
        x1.x -= deltai.x;
        x1.y -= deltai.y;
        x[i] = x1;

        // Modify cell and store
        int2 ix1 = make_int2(
            ix0.x + deltai.x + shift.x,
            ix0.y + deltai.y + shift.x
        );
        ix[i] = ix1;
    }

    // Add current to global buffer
    const int tile_off = tid * tile_size;

    for( unsigned i = 0; i < ext_nx.x * ext_nx.y; i++ ) {
        d_current[tile_off + i] += _move_deposit_buffer[i];
    }
}

/**
 * @brief Kernel for pushing particles
 * 
 * @param d_tiles       Particle tile information
 * @param d_ix          Particle data (cells)
 * @param d_x           Particle data (positions)
 * @param d_u           Particle data (momenta)
 * @param d_E           E field grid
 * @param d_B           B field grid
 * @param field_offset  Tile offset to field position (0,0)
 * @param ext_nx        E,B tile grid external size
 * @param alpha         Force normalization ( 0.5 * q / m * dt )
 */

/**
 * @brief Advance particle velocities
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
void push_kernel ( 
    uint2 const tile_idx,
    ParticleData const part,
    float3 * __restrict__ d_E, float3 * __restrict__ d_B, 
    unsigned int const field_offset, uint2 const ext_nx,
    float const alpha, double * __restrict__ d_energy )
{
    const uint2 ntiles  = part.ntiles;

    // Tile ID
    const int tid =  tile_idx.y * ntiles.x + tile_idx.x;

    int const field_vol = roundup4( ext_nx.x * ext_nx.y );
    int const tile_off = tid * field_vol;

    // Copy E and B into shared memory

    alignas(local_align) float3 E_local[ field_vol ];
    alignas(local_align) float3 B_local[ field_vol ];

    for( auto i = 0; i < field_vol; i++ ) {
        E_local[i] = d_E[tile_off + i];
        B_local[i] = d_B[tile_off + i];
    }

    float3 const * const __restrict__ E = & E_local[ field_offset ];
    float3 const * const __restrict__ B = & B_local[ field_offset ];

    // Push particles
    const int part_offset = part.offset[ tid ];
    const int np          = part.np[ tid ];
    int2   * __restrict__ ix = &part.ix[ part_offset ];
    float2 * __restrict__ x  = &part.x[ part_offset ];
    float3 * __restrict__ u  = &part.u[ part_offset ];

    double energy = 0;

    const int ystride = ext_nx.x;

    for( int i = 0; i < np; i++ ) {

        // Interpolate field
        float3 e, b;
        interpolate_fld( E, B, ystride, ix[i], x[i], e, b );
        
        // Advance momentum
        float3 pu = u[i];
        
        if constexpr ( type == species::boris ) u[i] = dudt_boris( alpha, e, b, pu, energy );
        if constexpr ( type == species::euler ) u[i] = dudt_boris_euler( alpha, e, b, pu, energy );
    }

    // Add up energy from all particles
    // In OpenMP, d_energy needs to be a reduction variable
    *d_energy += energy;
}

#endif

/**
 * @brief Construct a new Species object
 * 
 * @param name  Name for the species object (used for diagnostics)
 * @param m_q   Mass over charge ratio
 * @param ppc   Number of particles per cell
 */
Species::Species( std::string const name, float const m_q, uint2 const ppc ):
    ppc(ppc), name(name), m_q(m_q)
{

    // Validate parameters
    if ( m_q == 0 ) {
        std::cerr << "(*error*) Invalid m_q value, must be not 0, aborting...\n";
        exit(1);
    }

    if ( ppc.x < 1 || ppc.y < 1 ) {
        std::cerr << "(*error*) Invalid ppc value, must be >= 1 in all directions\n";
        exit(1);
    }

    // Set default parameters
    density   = new Density::Uniform( 1.0 );
    udist     = new UDistribution::None();
    bc        = species::bc_type (species::bc::periodic);
    push_type = species::boris;

    // Nullify pointers to data structures
    particles = nullptr;
    tmp = nullptr;
    sort = nullptr;
}


/**
 * @brief Initialize data structures and inject particles
 * 
 * @param box_      Simulation global box size
 * @param ntiles    Number of tiles
 * @param nx        Grid size per tile
 * @param dt_       Time step
 * @param id_       Species unique id
 */
void Species::initialize( float2 const box_, uint2 const ntiles, uint2 const nx,
    float const dt_, int const id_ ) {
    
    // Store simulation box size
    box = box_;

    // Store simulation time step
    dt = dt_;

    // Store species id (used by RNG)
    id = id_;

    // Set charge normalization factor
    q = copysign( density->n0 , m_q ) / (ppc.x * ppc.y);
    
    float2 gnx = make_float2( nx.x * ntiles.x, nx.y * ntiles.y );

    // Set cell size
    dx.x = box.x / (gnx.x);
    dx.y = box.y / (gnx.y);

    // Reference number maximum number of particles
    unsigned int max_part = 1.2 * gnx.x * gnx.y * ppc.x * ppc.y;

    particles = new Particles( ntiles, nx, max_part );
    particles->periodic.x = ( bc.x.lower == species::bc::periodic );
    particles->periodic.y = ( bc.y.lower == species::bc::periodic );

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
    np_inject( particles -> g_range(), np_inj );

    // Do an exclusive scan to get the required offsets
    uint32_t off = 0;
    for( unsigned i = 0; i < ntiles.x * ntiles.y; i ++ ) {
        particles -> offset[i] = off;
        off += np_inj[i];
    }

    // Inject the particles
    inject( particles -> g_range() );

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

    float2 ref = make_float2( moving_window.motion(), 0 );

    density -> inject( *particles, ppc, dx, ref, particles -> g_range() );
}

/**
 * @brief Inject particles in a specific cell range
 * 
 */
void Species::inject( bnd<unsigned int> range ) {

    float2 ref = make_float2( moving_window.motion(), 0 );

    density -> inject( *particles, ppc, dx, ref, range );
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

    float2 ref = make_float2( moving_window.motion(), 0 );

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
 * @brief Physical boundary conditions for the y direction 
 * 
 * @param ntiles    Number of tiles
 * @param tile_idx  Tile index
 * @param tiles     Particle tile information
 * @param data      Particle data
 * @param nx        Tile grid size
 * @param bc        Boundary condition
 */
void species_bcy(
    uint2 const tile_idx,
    ParticleData const part,
    species::bc_type const bc ) 
{
    const uint2 ntiles  = part.ntiles;
    const int ny = part.nx.y;

    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    const int part_offset    = part.offset[ tid ];
    const int np             = part.np[ tid ];
    int2   * __restrict__ ix = &part.ix[ part_offset ];
    float2 * __restrict__ x  = &part.x[ part_offset ];
    float3 * __restrict__ u  = &part.u[ part_offset ];

    if ( tile_idx.y == 0 ) {
        // Lower boundary
        switch( bc.y.lower ) {
        case( species::bc::reflecting ) :
            for( int i = 0; i < np; i++ ) {
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
        switch( bc.y.upper ) {
        case( species::bc::reflecting ) :
            for( int i = 0; i < np; i++ ) {
                if( ix[i].y >=  ny ) {
                    ix[i].y -= 1;
                    x[i].y = -x[i].y;
                    u[i].y = -u[i].y;
                }
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
    if ( bc.y.lower > species::bc::periodic || bc.y.upper > species::bc::periodic ) {
        for( unsigned ty : { 0u, particles -> ntiles.y-1 } ) {
            for( unsigned tx = 0; tx < particles -> ntiles.x; tx ++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                species_bcy ( tile_idx, *particles, bc );
            }
        }
    }
}

/**
 * @brief Free stream particles 1 iteration
 * 
 * No acceleration or current deposition is performed. Used for debug purposes.
 * 
 */
void Species::advance( ) {

    // Advance positions
    move( );
    
    // Process physical boundary conditions
    process_bc();

    // Increase internal iteration number
    iter++;
    
    // Sort particles according to tile
    particles -> tile_sort( *tmp, *sort );

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
 * @param current   Electric durrent density
 */
void Species::advance( Current &current ) {

    // Advance positions and deposit current
    move( current.J );

    // Process physical boundary conditions
    process_bc();

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
 * @param current   Electric durrent density
 */
void Species::advance( EMF const &emf, Current &current ) {

    // Advance momenta
    push( emf.E, emf.B );

    // Advance positions and deposit current
    move( current.J );

    // Process physical boundary conditions
    process_bc();

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
 * 4. Handle moving window algorith,
 * 5. Sort particles according to tiles
 * 
 * @param emf       EM fields
 * @param current   Electric durrent density
 */
void Species::advance_mov_window( EMF const &emf, Current &current ) {

    // Advance momenta
    push( emf.E, emf.B );

    if ( moving_window.needs_move( (iter+1) * dt ) ) {

        // Advance positions, deposit current and shift particles
        move( current.J, make_int2(-1,0) );

        // Process boundary conditions
        process_bc();

        // Find range where new particles need to be injected
        uint2 g_nx = particles -> gnx;
        bnd<unsigned int> range;
        range.x = { .lower = g_nx.x - 1, .upper = g_nx.x - 1 };
        range.y = { .lower = 0, .upper = g_nx.y - 1 };

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
        move( current.J );

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
void Species::move( vec3grid<float3> * J )
{
    const float2 dt_dx = make_float2(
        dt / dx.x,
        dt / dx.y
    );

    const float2 qnx = make_float2(
        q * dx.x / dt,
        q * dx.y / dt
    );

    #pragma omp parallel for schedule(dynamic)
    for( unsigned tid = 0; tid < particles -> ntiles.y * particles -> ntiles.x; tid ++ ) {
        
        const auto tile_idx = make_uint2( tid % particles -> ntiles.x, tid / particles -> ntiles.x );
        move_deposit_kernel(
            tile_idx, *particles,
            J -> d_buffer, J -> offset, J -> ext_nx, dt_dx, q, qnx
            );
    }

    // This avoids the reduction overhead
    for( unsigned tid = 0; tid < particles -> ntiles.y * particles -> ntiles.x; tid ++ ) {
        d_nmove += particles -> np[tid];
    }
}

/**
 * @brief Moves particles and deposit current
 * 
 * Current will be accumulated on existing data
 * 
 * @param current   Current grid
 */
void Species::move( vec3grid<float3> * J, const int2 shift )
{
    const float2 dt_dx = make_float2(
        dt / dx.x,
        dt / dx.y
    );

    const float2 qnx = make_float2(
        q * dx.x / dt,
        q * dx.y / dt
    );

    #pragma omp parallel for schedule(dynamic)
    for( unsigned tid = 0; tid < particles -> ntiles.y * particles -> ntiles.x; tid ++ ) {
        const auto tile_idx = make_uint2( tid % particles -> ntiles.x, tid / particles -> ntiles.x );
        move_deposit_shift_kernel ( 
            tile_idx, *particles,
            J -> d_buffer, J -> offset, J -> ext_nx, dt_dx, q, qnx, shift
        );
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
    int2   * __restrict__ ix = &part.ix[ part_offset ];
    float2 * __restrict__ x  = &part.x[ part_offset ];
    float3 * __restrict__ u  = &part.u[ part_offset ];

    for( int i = 0; i < np; i++ ) {
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
}


/**
 * @brief Moves particles (no current deposition)
 * 
 * This is usually used for test species: species that do not self-consistently
 * influence the simulation
 * 
 * @param current   Current grid
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
 * @param E     Electric field
 * @param B     Magnetic field
 */
void Species::push( vec3grid<float3> * const E, vec3grid<float3> * const B )
{
    uint2 ext_nx = E -> ext_nx;
    const float alpha = 0.5 * dt / m_q;
    d_energy = 0;

    switch( push_type ) {
    case( species :: euler ):

        #pragma omp parallel for schedule(dynamic) reduction(+:d_energy)
        for( unsigned tid = 0; tid < particles -> ntiles.y * particles -> ntiles.x; tid ++ ) {    
            const uint2 tile_idx = make_uint2( tid % particles -> ntiles.x, tid / particles -> ntiles.x );
            push_kernel <species::euler> (
                tile_idx, *particles,
                E -> d_buffer, B -> d_buffer, E -> offset, ext_nx, alpha,
                &d_energy
            );
        }
        break;

    case( species :: boris ):

        #pragma omp parallel for schedule(dynamic) reduction(+:d_energy)
        for( unsigned tid = 0; tid < particles -> ntiles.y * particles -> ntiles.x; tid ++ ) {    
            const uint2 tile_idx = make_uint2( tid % particles -> ntiles.x, tid / particles -> ntiles.x );
            push_kernel <species::boris> (
                tile_idx, *particles,
                E -> d_buffer, B -> d_buffer, E -> offset, ext_nx, alpha,
                &d_energy
            );
        }
        break;
    }
}

/**
 * @brief kernel for depositing charge
 * 
 * @param d_charge  Charge density grid (will be zeroed by this kernel)
 * @param offset    Offset to position (0,0) of grid
 * @param ext_nx    External tile size (i.e. including guard cells)
 * @param d_tile    Particle tiles information
 * @param d_ix      Particle buffer (cells)
 * @param d_x       Particle buffer (position)
 * @param q         Species charge per particle
 */
void dep_charge_kernel(
    uint2 const tile_idx,
    ParticleData const part, const float q, 
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
    int2   const * __restrict__ const ix = &part.ix[ part_off ];
    float2 const * __restrict__ const x  = &part.x[ part_off ];
    const int ystride = ext_nx.x;

    for( int i = 0; i < np; i ++ ) {
        const int idx = ix[i].y * ystride + ix[i].x;
        const float s0x = 0.5f - x[i].x;
        const float s1x = 0.5f + x[i].x;
        const float s0y = 0.5f - x[i].y;
        const float s1y = 0.5f + x[i].y;

        // When use more thatn 1 thread per tile, these need to be atomic inside tile
        charge[ idx               ] += s0y * s0x * q;
        charge[ idx + 1           ] += s0y * s1x * q;
        charge[ idx     + ystride ] += s1y * s0x * q;
        charge[ idx + 1 + ystride ] += s1y * s1x * q;
    }

    // sync

    // Copy data to global memory
    const int tile_off = tid * roundup4( ext_nx.x * ext_nx.y );
    for( unsigned i = 0; i < ext_nx.x * ext_nx.y; i ++ ) {
        d_charge[tile_off + i] += _dep_charge_buffer[i];
    } 
}

/**
 * @brief Deposit charge density
 * 
 * @param charge    Charge density grid
 */
void Species::deposit_charge( grid<float> &charge ) const {

    for( unsigned ty = 0; ty < particles -> ntiles.y; ++ty ) {
        for( unsigned tx = 0; tx < particles -> ntiles.x; ++tx ) {
            const auto tile_idx = make_uint2( tx, ty );
            dep_charge_kernel ( tile_idx, *particles, q, charge.d_buffer, charge.offset, charge.ext_nx );
        }
    }
}


/**
 * @brief Save particle data to file
 * 
 */
void Species::save() const {

    const char * quants[] = {
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
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_n"
    };

    // Omit number of particles, this will be filled in later
    zdf::part_info info = {
        .name = (char *) name.c_str(),
        .label = (char *) name.c_str(),
        .nquants = 5,
        .quants = (char **) quants,
        .qlabels = (char **) qlabels,
        .qunits = (char **) qunits,
    };

    particles -> save( info, iter_info, "PARTICLES" );
}

/**
 * @brief Saves charge density to file
 * 
 * The routine will create a new charge grid, deposit the charge and save the grid
 * 
 */
void Species::save_charge() const {

    // For linear interpolation we only require 1 guard cell at the upper boundary
    bnd<unsigned int> gc;
    gc.x = {0,1};
    gc.y = {0,1};

    // Deposit charge on device
    grid<float> charge( particles -> ntiles, particles -> nx, gc );

    charge.zero();

    deposit_charge( charge );

    charge.add_from_gc();

    // Prepare file info
    zdf::grid_axis axis[2];
    axis[0] = (zdf::grid_axis) {
        .name = (char *) "x",
        .min = 0. + moving_window.motion(),
        .max = box.x + moving_window.motion(),
        .label = (char *) "x",
        .units = (char *) "c/\\omega_n"
    };

    axis[1] = (zdf::grid_axis) {
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
    
    charge.save( info, iter_info, path.c_str() );
}

/**
 * @brief kernel for depositing 1d phasespace
 * 
 * @tparam q        Phasespace quantity
 * @param d_data    Output data
 * @param range     Phasespace value range
 * @param size      Phasespace grid size
 * @param tile_nx   Size of tile grid
 * @param norm      Normalization factor
 * @param d_tiles   Particle tile information
 * @param d_ix      Particle data (cell)
 * @param d_x       Particle data (pos)
 * @param d_u       Particle data (generalized momenta)
 */
template < phasespace::quant q >
void dep_pha1_kernel(
    uint2 const tile_idx,
    float * const __restrict__ d_data, float2 const range, int const size,
    float const norm, 
    ParticleData const part )
{
    const uint2 ntiles  = part.ntiles;
    uint2 const tile_nx = part.nx;

    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    const int part_offset = part.offset[ tid ];
    const int np          = part.np[ tid ];
    int2   * __restrict__ ix = &part.ix[ part_offset ];
    float2 * __restrict__ x  = &part.x[ part_offset ];
    float3 * __restrict__ u  = &part.u[ part_offset ];

    float const pha_rdx = size / (range.y - range.x);

    for( int i = 0; i < np; i++ ) {
        float d;
        switch( q ) {
        case( phasespace:: x ): d = ( tile_idx.x * tile_nx.x + ix[i].x) + (x[i].x + 0.5f); break;
        case( phasespace:: y ): d = ( tile_idx.y * tile_nx.y + ix[i].y) + (x[i].y + 0.5f); break;
        case( phasespace:: ux ): d = u[i].x; break;
        case( phasespace:: uy ): d = u[i].y; break;
        case( phasespace:: uz ): d = u[i].z; break;
        }

        float n =  (d - range.x ) * pha_rdx - 0.5f;
        int   k = int( n + 1 ) - 1;
        float w = n - k;

        // When using multi-threading these need to atomic accross tiles
        if ((k   >= 0) && (k   < size-1)) d_data[k  ] += (1-w) * norm;
        if ((k+1 >= 0) && (k+1 < size-1)) d_data[k+1] +=    w  * norm;
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
    
    // In OSIRIS we don't take the absolute value of q
    float norm = fabs(q) * ( dx.x * dx.y ) *
                 size / (range.y - range.x) ;

    switch(quant) {
    case( phasespace::x ):
        range.y /= dx.x;
        range.x /= dx.x;
        for( unsigned ty = 0; ty < particles -> ntiles.y; ty ++ ) {
            for( unsigned tx = 0; tx < particles -> ntiles.x; tx ++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                dep_pha1_kernel<phasespace::x>  (
                    tile_idx, 
                    d_data, range, size, norm, 
                    *particles
                );
            }
        }

        break;
    case( phasespace:: y ):
        range.y /= dx.y;
        range.x /= dx.y;
        for( unsigned ty = 0; ty < particles -> ntiles.y; ty ++ ) {
            for( unsigned tx = 0; tx < particles -> ntiles.x; tx ++ ) {
                const auto tile_idx = make_uint2( tx, ty );
                dep_pha1_kernel<phasespace::y>  (
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

    if ( quant == phasespace::x ) {
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

    // Deposit 1D phasespace
    float * d_data = memory::malloc<float>( size );

    dep_phasespace( d_data, quant, range, size );

    // Save file
    zdf::save_grid( d_data, info, iter_info, "PHASESPACE/" + name );

    memory::free( d_data );
}

/**
 * @brief CUDA kernel for depositing 2D phasespace
 * 
 * @tparam q0       Quantity 0
 * @tparam q1       Quantity 1
 * @param d_data    Ouput data
 * @param range0    Range of values of quantity 0
 * @param size0     Phasespace grid size for quantity 0
 * @param range1    Range of values of quantity 1
 * @param size1     Range of values of quantity 1
 * @param tile_nx   Size of tile grid
 * @param norm      Normalization factor
 * @param d_tiles   Particle tile information
 * @param d_ix      Particle data (cell)
 * @param d_x       Particle data (pos)
 * @param d_u       Particle data (generalized momenta)
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
    int2   * __restrict__ ix  = &part.ix[ part_offset ];
    float2 * __restrict__ x   = &part.x[ part_offset ];
    float3 * __restrict__ u   = &part.u[ part_offset ];

    float const pha_rdx0 = size0 / (range0.y - range0.x);
    float const pha_rdx1 = size1 / (range1.y - range1.x);

    for( int i = 0; i < np; i++ ) {
        float d0;
        switch( quant0 ) {
        case( phasespace:: x ):  d0 = ( tile_idx.x * tile_nx.x + ix[i].x) + (x[i].x + 0.5f); break;
        case( phasespace:: y ):  d0 = ( tile_idx.y * tile_nx.y + ix[i].y) + (x[i].y + 0.5f); break;
        case( phasespace:: ux ): d0 = u[i].x; break;
        case( phasespace:: uy ): d0 = u[i].y; break;
        case( phasespace:: uz ): d0 = u[i].z; break;
        }

        float n0 =  (d0 - range0.x ) * pha_rdx0 - 0.5f;
        int   k0 = int( n0 + 1 ) - 1;
        float w0 = n0 - k0;

        float d1;
        switch( quant1 ) {
        //case( phasespace:: x ):  d1 = ( tile_idx.x * tile_nx.x + ix[i].x) + (x[i].x + 0.5f); break;
        case( phasespace:: y ):  d1 = ( tile_idx.y * tile_nx.y + ix[i].y) + (x[i].y + 0.5f); break;
        case( phasespace:: ux ): d1 = u[i].x; break;
        case( phasespace:: uy ): d1 = u[i].y; break;
        case( phasespace:: uz ): d1 = u[i].z; break;
        }

        float n1 =  (d1 - range1.x ) * pha_rdx1 - 0.5f;
        int   k1 = int( n1 + 1 ) - 1;
        float w1 = n1 - k1;

        // When using multi-threading these need to atomic accross tiles
        if ((k0   >= 0) && (k0   < size0-1) && (k1   >= 0) && (k1   < size1-1)) d_data[(k1  )*size0 + k0  ] += (1-w0) * (1-w1) * norm;
        if ((k0+1 >= 0) && (k0+1 < size0-1) && (k1   >= 0) && (k1   < size1-1)) d_data[(k1  )*size0 + k0+1] +=    w0  * (1-w1) * norm;
        if ((k0   >= 0) && (k0   < size0-1) && (k1+1 >= 0) && (k1+1 < size1-1)) d_data[(k1+1)*size0 + k0  ] += (1-w0) *    w1  * norm;
        if ((k0+1 >= 0) && (k0+1 < size0-1) && (k1+1 >= 0) && (k1+1 < size1-1)) d_data[(k1+1)*size0 + k0+1] +=    w0  *    w1  * norm;
    }
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
void Species::dep_phasespace( 
    float * const d_data,
    phasespace::quant quant0, float2 range0, unsigned const size0,
    phasespace::quant quant1, float2 range1, unsigned const size1 ) const
{

    // Zero device memory
    memory::zero( d_data, size0 * size1 );

    // In OSIRIS we don't take the absolute value of q
    float norm = fabs(q) * ( dx.x * dx.y ) *
                          ( size0 / (range0.y - range0.x) ) *
                          ( size1 / (range1.y - range1.x) );

    switch(quant0) {
    case( phasespace::x ):
        range0.y /= dx.x;
        range0.x /= dx.x;
        switch(quant1) {
        case( phasespace::y ):
            range1.y /= dx.y;
            range1.x /= dx.y;
            for( unsigned ty = 0; ty < particles -> ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < particles -> ntiles.x; tx ++ ) {
                    const auto tile_idx = make_uint2( tx, ty );
                    dep_pha2_kernel<phasespace::x,phasespace::y> (
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
                    dep_pha2_kernel<phasespace::x,phasespace::ux> (
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
                    dep_pha2_kernel<phasespace::x,phasespace::uy> (
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
                    dep_pha2_kernel<phasespace::x,phasespace::uz> (
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
    case( phasespace:: y ):
        range0.y /= dx.y;
        range0.x /= dx.y;
        switch(quant1) {
        case( phasespace::ux ):
            for( unsigned ty = 0; ty < particles -> ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < particles -> ntiles.x; tx ++ ) {
                    const auto tile_idx = make_uint2( tx, ty );
                    dep_pha2_kernel<phasespace::y,phasespace::ux> (
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
                    dep_pha2_kernel<phasespace::y,phasespace::uy> (
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
                    dep_pha2_kernel<phasespace::y,phasespace::uz> (
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

    if ( quant0 == phasespace::x ) {
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

    float * d_data = memory::malloc<float>( size0 * size1 );

    dep_phasespace( d_data, quant0, range0, size0, quant1, range1, size1 );

    zdf::save_grid( d_data, info, iter_info, "PHASESPACE/" + name );

    memory::free( d_data );
}


