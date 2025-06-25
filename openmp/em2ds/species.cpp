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
    return 1.0f/std::sqrt( ops::fma( u.z, u.z, 
                           ops::fma( u.y, u.y, 
                           ops::fma( u.x, u.x, 1.0f ) ) ) );
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
inline void dep_current( float3 * const __restrict__ J, const int ystride, 
    int2 ix0, float2 x0, float3 u, 
    float rg, float2 dx, float q ) {

    // Find position time centered with velocity
    float2 x1 = make_float2( x0.x + 0.5f * dx.x, x0.y + 0.5f * dx.y );
    
    int2 const deltai = make_int2(
        ((x1.x >= 0.5f) - (x1.x < -0.5f)),
        ((x1.y >= 0.5f) - (x1.y < -0.5f))
    );

    int2 ix  = make_int2( ix0.x + deltai.x, ix0.y + deltai.y );
    float2 x = make_float2( x0.x - deltai.x, x0.y - deltai.y );

    const float S0x = 0.5f - x.x;
    const float S1x = 0.5f + x.x;

    const float S0y = 0.5f - x.y;
    const float S1y = 0.5f + x.y;

    const float jx = q * u.x * rg;
    const float jy = q * u.y * rg;
    const float jz = q * u.z * rg;

    int idx = ix.y * ystride + ix.x;

    J[ idx               ].x += S0y * S0x * jx;
    J[ idx               ].y += S0y * S0x * jy;
    J[ idx               ].z += S0y * S0x * jz;

    J[ idx + 1           ].x += S0y * S1x * jx;
    J[ idx + 1           ].y += S0y * S1x * jy;
    J[ idx + 1           ].z += S0y * S1x * jz;

    J[ idx + ystride     ].x += S1y * S0x * jx;
    J[ idx + ystride     ].y += S1y * S0x * jy;
    J[ idx + ystride     ].z += S1y * S0x * jz;

    J[ idx + ystride + 1 ].x += S1y * S1x * jx;
    J[ idx + ystride + 1 ].y += S1y * S1x * jy;
    J[ idx + ystride + 1 ].z += S1y * S1x * jz;
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
inline void dep_charge( float * const __restrict__ rho, const int ystride, int2 ix, float2 x, float q )
{

    const float S0x = 0.5f - x.x;
    const float S1x = 0.5f + x.x;

    const float S0y = 0.5f - x.y;
    const float S1y = 0.5f + x.y;

    int idx = ix.y * ystride + ix.x;

    rho[ idx               ] += S0y * S0x * q;
    rho[ idx + 1           ] += S0y * S1x * q;
    rho[ idx + ystride     ] += S1y * S0x * q;
    rho[ idx + ystride + 1 ] += S1y * S1x * q;
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
    return vec_div( c1, vec_sqrt( vec_fmadd( u.z, u.z,
                                  vec_fmadd( u.y, u.y,
                                  vec_fmadd( u.x, u.x, c1 ) ) ) ) );
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
 * @brief Deposit current from single particle (vector version)
 * 
 * @note Current will be deposited for all vector elements
 * 
 * @param J         Electric current buffer
 * @param ystride   y-stride for J
 * @param ix0       Initial particle cell
 * @param x0        Initial particle position
 * @param u         Particle momenta
 * @param rg        1 / Lorentz γ
 * @param delta     Particle motion normalized to cell size
 * @param q         Particle charge
 */
void vdep_current(
    float3 * const __restrict__ J, const int ystride, 
    const vint2 ix0, const vfloat2 x0, const vfloat3 u,
    const vfloat rg, const vfloat2 delta, const float q )
{
    const vint   c1i   = vec_int(1);
    const vfloat c1_2  = vec_float(.5f);
    const vfloat cm1_2 = vec_float(-.5f);

    // Find position time centered with velocity
    vfloat2 x1 {
        vec_fmadd( c1_2, delta.x, x0.x ),
        vec_fmadd( c1_2, delta.y, x0.y )
    };

    vint2 deltai{
        vec_sub( vec_ge( x1.x, c1_2, c1i ), vec_lt( x1.x, cm1_2, c1i ) ),
        vec_sub( vec_ge( x1.y, c1_2, c1i ), vec_lt( x1.y, cm1_2, c1i ) )
    };

    vint2 ix{ vec_add( ix0.x, deltai.x ), vec_add( ix0.y, deltai.y ) };
    vfloat2 x {
        vec_sub( x0.x, vec_float(deltai.x ) ),
        vec_sub( x0.y, vec_float(deltai.y ) )
    };

    vfloat S0x = vec_sub( c1_2, x.x );
    vfloat S1x = vec_add( c1_2, x.x );

    vfloat S0y = vec_sub( c1_2, x.y );
    vfloat S1y = vec_add( c1_2, x.y );

    VecFloat_s jx = vec_mul( vec_mul( u.x, rg ), q );
    VecFloat_s jy = vec_mul( vec_mul( u.y, rg ), q );
    VecFloat_s jz = vec_mul( vec_mul( u.z, rg ), q );

    VecFloat_s a00 = vec_mul( S0y, S0x );
    VecFloat_s a01 = vec_mul( S0y, S1x );
    VecFloat_s a10 = vec_mul( S1y, S0x );
    VecFloat_s a11 = vec_mul( S1y, S1x );

    VecInt_s idx = vec_add( ix.x, vec_mul( ix.y, ystride ) );

    for ( int i = 0; i < vecwidth; i++ ) {

        float * __restrict__ Js = (float *) (&J[idx.extract(i)]);
        int const stride3 = 3 * ystride;

        float w00 = a00.extract(i);
        float w01 = a01.extract(i);
        float w10 = a10.extract(i);
        float w11 = a11.extract(i);

        float pjx = jx.extract(i);
        float pjy = jy.extract(i);
        float pjz = jz.extract(i);

        Js[       0 + 0 + 0 ] += w00 * pjx;
        Js[       0 + 0 + 1 ] += w00 * pjy;
        Js[       0 + 0 + 2 ] += w00 * pjz;

        Js[       0 + 3 + 0 ] += w01 * pjx;
        Js[       0 + 3 + 1 ] += w01 * pjy;
        Js[       0 + 3 + 2 ] += w01 * pjz;

        Js[ stride3 + 0 + 0 ] += w10 * pjx;
        Js[ stride3 + 0 + 1 ] += w10 * pjy;
        Js[ stride3 + 0 + 2 ] += w10 * pjz;

        Js[ stride3 + 3 + 0 ] += w11 * pjx;
        Js[ stride3 + 3 + 1 ] += w11 * pjy;
        Js[ stride3 + 3 + 2 ] += w11 * pjz;
    }
}


/**
 * @brief Deposit charge from single particle (vector version)
 * 
 * @param rho       Charge density buffer
 * @param ystride   y-stride for rho
 * @param ix        Particle position (cell)
 * @param x         Particle position inside cell
 * @param q         Particle charge
 */
inline void vdep_charge( 
    float * const __restrict__ rho, const int ystride, 
    const vint2 ix, const vfloat2 x, const float q )
{
    const vfloat c1_2  = vec_float(.5f);

    vfloat S0x = vec_sub( c1_2, x.x );
    vfloat S1x = vec_add( c1_2, x.x );

    vfloat S0y = vec_sub( c1_2, x.y );
    vfloat S1y = vec_add( c1_2, x.y );

    VecFloat_s rho00 = vec_mul( vec_mul( S0y, S0x ), q );
    VecFloat_s rho01 = vec_mul( vec_mul( S0y, S1x ), q );
    VecFloat_s rho10 = vec_mul( vec_mul( S1y, S0x ), q );
    VecFloat_s rho11 = vec_mul( vec_mul( S1y, S1x ), q );

    VecInt_s idx = vec_add( ix.x, vec_mul( ix.y, ystride ) );

    for ( int i = 0; i < vecwidth; i++ ) {
        float * __restrict__ rhos = & rho[ idx.extract(i) ];

        rhos[           0 ] += rho00.extract(i);
        rhos[           1 ] += rho01.extract(i);
        rhos[ ystride + 0 ] += rho10.extract(i);
        rhos[ ystride + 1 ] += rho11.extract(i);
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
    float3 * const __restrict__ d_current, unsigned int const current_offset, uint2 const current_ext_nx,
    float  * const __restrict__ d_charge, unsigned int const charge_offset, uint2 const charge_ext_nx,
    float2 const dt_dx, float const q, float2 const qnx ) 
{
    const uint2 ntiles  = part.ntiles;

    const int tile_vol = roundup4( current_ext_nx.x * current_ext_nx.y );

    // The alignment also avoids some optimization related segfaults
    alignas(local_align) float3 _current_buffer[ tile_vol ];
    alignas(local_align) float  _charge_buffer[ tile_vol ];

    // Zero local buffers
    for( auto i = 0; i < tile_vol; i++ ) {
        _current_buffer[i] = make_float3(0,0,0);
        _charge_buffer[i] = 0;
    }

    // sync

    // Move particles and deposit current
    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    float3 * J =  & _current_buffer[ current_offset ];
    float * rho = & _charge_buffer[ charge_offset ];

    const int part_offset    = part.offset[ tid ];
    const int np             = part.np[ tid ];
    int2   * __restrict__ ix = &part.ix[ part_offset ];
    float2 * __restrict__ x  = &part.x[ part_offset ];
    float3 * __restrict__ u  = &part.u[ part_offset ];

    const int np_vec = (np/vecwidth) * vecwidth;
    for( int i = 0; i < np_vec; i+= vecwidth ) {
        const vint   c1i   = vec_int(1);
        const vfloat c1_2  = vec_float( .5f);
        const vfloat cm1_2 = vec_float(-.5f);

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

        // Deposit current
        vdep_current( J, current_ext_nx.x, ix0, x0, pu, rg, delta, q );

        // Advance position
        vfloat2 x1 = {
            vec_add( x0.x , delta.x ),
            vec_add( x0.y , delta.y )
        };

        // Check for cell crossing
        vint2 deltai{
            vec_sub( vec_ge( x1.x, c1_2, c1i ), vec_lt( x1.x, cm1_2, c1i ) ),
            vec_sub( vec_ge( x1.y, c1_2, c1i ), vec_lt( x1.y, cm1_2, c1i ) )
        };


        // Correct cell position 
        x1.x = vec_sub( x1.x, vec_float( deltai.x ) );
        x1.y = vec_sub( x1.y, vec_float( deltai.y ) );

        // Modify cell 
        vint2 ix1 = {
            vec_add( ix0.x, deltai.x ),
            vec_add( ix0.y, deltai.y )
        };

        // Deposit charge
        vdep_charge( rho, charge_ext_nx.x, ix1, x1, q );

        // Store result
        vec_store_s2( (float *) & x[i], x1 );
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

        // Deposit current
        dep_current( J, current_ext_nx.x, ix0, x0, pu, rg, delta, q );

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
        dep_charge( rho, charge_ext_nx.x, ix1, x1, q );

        // Store result
        x[i] = x1;
        ix[i] = ix1;
    }

    // Add current to global buffer
    const int tile_off = tid * tile_vol;

    for( int i = 0; i < tile_vol; i++ ) {
        d_current[tile_off + i] += _current_buffer[i];
        d_charge[tile_off + i] += _charge_buffer[i];
    }
}


/**
 * @brief Interpolate EM field values at particle position using linear 
 * (1st order) interpolation using SIMD operations.
 * 
 * @note The EM fields are assumed to be defined on the lower left corner of the cell
 * 
 * @param E         Pointer to position (0,0) of E field grid
 * @param B         Pointer to position (0,0) of B field grid
 * @param ystride   E and B grids y stride (must be signed)
 * @param ix        (simd vector) Particle cell index
 * @param x         (simd vector) Particle postion inside cell
 * @param e[out]    (simd vector) E field at particle position
 * @param b[out]    (simd vector) B field at particle position
 */
inline void vinterpolate_fld( 
    float3 const * const __restrict__ E, 
    float3 const * const __restrict__ B, 
    const int ystride,
    const vint2 ix, const vfloat2 x, vfloat3 & e, vfloat3 & b)
{
    vint i = ix.x;
    vint j = ix.y;

    const vfloat c_1_2 = vec_float( 0.5f );

    const vfloat s0x = vec_sub( c_1_2, x.x );
    const vfloat s1x = vec_add( c_1_2, x.x );

    const vfloat s0y = vec_sub( c_1_2, x.y );
    const vfloat s1y = vec_add( c_1_2, x.y );

    // Get linear indices
    int const ystride3 = 3 * ystride;
    vint idx = vec_add( vec_mul( i, 3 ), vec_mul( j, ystride3 ) );

    {   // Interpolate E field
        const float * __restrict__ const Es = (float *) (&E[0]);

        vfloat fx, fy, fz;
        fx = vec_mul(   vec_gather( Es     + 0, idx ), s0x );
        fy = vec_mul(   vec_gather( Es     + 1, idx ), s0x );
        fz = vec_mul(   vec_gather( Es     + 2, idx ), s0x );

        fx = vec_fmadd( vec_gather( Es + 3 + 0, idx ), s1x, fx );
        fy = vec_fmadd( vec_gather( Es + 3 + 1, idx ), s1x, fy );
        fz = vec_fmadd( vec_gather( Es + 3 + 2, idx ), s1x, fz );

        e.x = vec_mul( fx, s0y );
        e.y = vec_mul( fy, s0y );
        e.z = vec_mul( fz, s0y );

        fx = vec_mul(   vec_gather( Es + ystride3     + 0, idx ), s0x );
        fy = vec_mul(   vec_gather( Es + ystride3     + 1, idx ), s0x );
        fz = vec_mul(   vec_gather( Es + ystride3     + 2, idx ), s0x );

        fx = vec_fmadd( vec_gather( Es + ystride3 + 3 + 0, idx ), s1x, fx );
        fy = vec_fmadd( vec_gather( Es + ystride3 + 3 + 1, idx ), s1x, fy );
        fz = vec_fmadd( vec_gather( Es + ystride3 + 3 + 2, idx ), s1x, fz );

        e.x = vec_fmadd( fx, s1y, e.x );
        e.y = vec_fmadd( fy, s1y, e.y );
        e.z = vec_fmadd( fz, s1y, e.z );
    }

    {   // Interpolate B field
        const float * __restrict__ const Bs = (float *) (&B[0]);

        vfloat fx, fy, fz;
        fx = vec_mul(   vec_gather( Bs     + 0, idx ), s0x );
        fy = vec_mul(   vec_gather( Bs     + 1, idx ), s0x );
        fz = vec_mul(   vec_gather( Bs     + 2, idx ), s0x );

        fx = vec_fmadd( vec_gather( Bs + 3 + 0, idx ), s1x, fx );
        fy = vec_fmadd( vec_gather( Bs + 3 + 1, idx ), s1x, fy );
        fz = vec_fmadd( vec_gather( Bs + 3 + 2, idx ), s1x, fz );

        b.x = vec_mul( fx, s0y );
        b.y = vec_mul( fy, s0y );
        b.z = vec_mul( fz, s0y );

        fx = vec_mul(   vec_gather( Bs + ystride3     + 0, idx ), s0x );
        fy = vec_mul(   vec_gather( Bs + ystride3     + 1, idx ), s0x );
        fz = vec_mul(   vec_gather( Bs + ystride3     + 2, idx ), s0x );

        fx = vec_fmadd( vec_gather( Bs + ystride3 + 3 + 0, idx ), s1x, fx );
        fy = vec_fmadd( vec_gather( Bs + ystride3 + 3 + 1, idx ), s1x, fy );
        fz = vec_fmadd( vec_gather( Bs + ystride3 + 3 + 2, idx ), s1x, fz );

        b.x = vec_fmadd( fx, s1y, b.x );
        b.y = vec_fmadd( fy, s1y, b.y );
        b.z = vec_fmadd( fz, s1y, b.z );
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
    float3 * const __restrict__ d_current, unsigned int const current_offset, uint2 const current_ext_nx,
    float  * const __restrict__ d_charge, unsigned int const charge_offset, uint2 const charge_ext_nx,
    float2 const dt_dx, float const q, float2 const qnx ) 
{
    const uint2 ntiles  = part.ntiles;
    const int tile_size = roundup4( current_ext_nx.x * current_ext_nx.y );

    // This is usually in block shared memeory
    alignas(local_align) float3 _current_buffer[ tile_size ];
    alignas(local_align) float  _charge_buffer[ tile_size ];

    // Zero local buffers
    for( auto i = 0; i < tile_size; i++ ) {
        _current_buffer[i] = make_float3(0,0,0);
        _charge_buffer[i] = 0;
    }

    // sync

    // Move particles and deposit current
    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    float3 * J   = & _current_buffer[ current_offset ];
    float  * rho = & _charge_buffer[ charge_offset ];

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

        // Deposit current
        dep_current( J, current_ext_nx.x, ix0, x0, pu, rg, delta, q );

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
        dep_charge( rho, charge_ext_nx.x, ix1, x1, q );

        // Store result
        x[i] = x1;
        ix[i] = ix1;
    }

    // Add current to global buffer
    const int tile_off = tid * tile_size;

    for( unsigned i = 0; i < current_ext_nx.x * current_ext_nx.y; i++ ) {
        d_current[tile_off + i] += _current_buffer[i];
        d_charge[tile_off + i] += _charge_buffer[i];
    }
}

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
 * @brief Initialize data structures and inject initial particle distribution
 * 
 * @param box_              Global simulation box size
 * @param global_ntiles     Global number of tiles
 * @param nx                Individutal tile grid size
 * @param dt_               Time step
 * @param id_               Species unique identifier
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
       
    // Get global grid size
    auto  dims = nx * ntiles;

    // Set cell size
    dx.x = box.x / (dims.x);
    dx.y = box.y / (dims.y);

    // Reference number maximum number of particles
    unsigned int max_part = 1.2 * dims.x * dims.y * ppc.x * ppc.y;

    // Create particle data structure
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

    float2 ref{0};

    density -> inject( *particles, ppc, dx, ref, particles -> local_range() );
}

/**
 * @brief Inject particles in a specific cell range
 * 
 */
void Species::inject( bnd<unsigned int> range ) {

    float2 ref{0};

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

    /// @brief position of lower corner of local grid in simulation units
    float2 ref{0};

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

    std::cout << "(*error*) Species::process_bc() have not been implemented yet,"
              << " aborting.\n";
    exit(1);
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
 * @param current   Electric durrent density
 */
void Species::advance( Current &current, Charge &charge ) {

    // Advance positions and deposit current
    move( current.J, charge.rho );

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
void Species::advance( EMF const &emf, Current &current, Charge & charge ) {

    // Advance momenta
    push( emf.E, emf.B );

    // Advance positions and deposit current/charge
    move( current.J, charge.rho );

    // Process physical boundary conditions
    // process_bc();

    // Increase internal iteration number
    iter++;
    
    // Sort particles according to tile
    particles -> tile_sort( *tmp, *sort );

}


/**
 * @brief Moves particles and deposit current
 * 
 * Current will be accumulated on existing data
 * 
 * @param current   Current grid
 */
void Species::move( vec3grid<float3> * J, grid<float> * rho )
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
            J -> d_buffer, J -> offset, J -> ext_nx, 
            rho -> d_buffer, rho -> offset, rho -> ext_nx,
            dt_dx, q, qnx
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
 * @brief Moves particles (no current/charge deposition)
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

    const std::string path = "PARTICLES";

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
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_n"
    };

    // Omit number of particles, this will be filled in later
    zdf::part_info info = {
        .name = (char *) name.c_str(),
        .label = (char *) name.c_str(),
        .nquants = 5,
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
        float2 scale{ dx.x, 0 };
        particles -> gather( part::quant::x, scale, h_data );
    }
    zdf::add_quant_part_file( part_file, "x", h_data, np );

    if ( np > 0 ) {
        float2 scale{ dx.y, 0 };
        particles -> gather( part::quant::y, scale, h_data );
    }
    zdf::add_quant_part_file( part_file, "y", h_data, np );

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
        .min = 0.,
        .max = box.x,
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
    
    charge.save( info, iter_info, path );
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

    const int part_offset = part.offset[ tid ];
    const int np          = part.np[ tid ];
    int2   * __restrict__ ix = &part.ix[ part_offset ];
    float2 * __restrict__ x  = &part.x[ part_offset ];
    float3 * __restrict__ u  = &part.u[ part_offset ];

    float const pha_rdx = size / (range.y - range.x);

    const int shiftx = tile_idx.x * tile_nx.x;
    const int shifty = tile_idx.y * tile_nx.y;

    for( int i = 0; i < np; i++ ) {
        float d;
        if constexpr ( quant == phasespace:: x  ) d = ( shiftx + ix[i].x) + (x[i].x + 0.5f);
        if constexpr ( quant == phasespace:: y  ) d = ( shifty + ix[i].y) + (x[i].y + 0.5f);
        if constexpr ( quant == phasespace:: ux ) d = u[i].x;
        if constexpr ( quant == phasespace:: uy ) d = u[i].y;
        if constexpr ( quant == phasespace:: uz ) d = u[i].z;

        float n =  (d - range.x ) * pha_rdx - 0.5f;
        int   k = int( n + 1 ) - 1;
        float w = n - k;

        // When using multi-threading these need to be atomic accross tiles
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
    int2   * __restrict__ ix  = &part.ix[ part_offset ];
    float2 * __restrict__ x   = &part.x[ part_offset ];
    float3 * __restrict__ u   = &part.u[ part_offset ];

    float const pha_rdx0 = size0 / (range0.y - range0.x);
    float const pha_rdx1 = size1 / (range1.y - range1.x);

    const int shiftx = tile_idx.x * tile_nx.x;
    const int shifty = tile_idx.y * tile_nx.y;

    for( int i = 0; i < np; i++ ) {
        float d0;
        if constexpr ( quant0 == phasespace:: x )  d0 = ( shiftx + ix[i].x) + (x[i].x + 0.5f);
        if constexpr ( quant0 == phasespace:: y )  d0 = ( shifty + ix[i].y) + (x[i].y + 0.5f);
        if constexpr ( quant0 == phasespace:: ux ) d0 = u[i].x;
        if constexpr ( quant0 == phasespace:: uy ) d0 = u[i].y;
        if constexpr ( quant0 == phasespace:: uz ) d0 = u[i].z;

        float n0 =  (d0 - range0.x ) * pha_rdx0 - 0.5f;
        int   k0 = int( n0 + 1 ) - 1;
        float w0 = n0 - k0;

        float d1;
        // if constexpr ( quant1 == phasespace:: x )  d1 = ( shiftx + ix[i].x) + (x[i].x + 0.5f);
        if constexpr ( quant1 == phasespace:: y )  d1 = ( shifty + ix[i].y) + (x[i].y + 0.5f);
        if constexpr ( quant1 == phasespace:: ux ) d1 = u[i].x;
        if constexpr ( quant1 == phasespace:: uy ) d1 = u[i].y;
        if constexpr ( quant1 == phasespace:: uz ) d1 = u[i].z;

        float n1 =  (d1 - range1.x ) * pha_rdx1 - 0.5f;
        int   k1 = int( n1 + 1 ) - 1;
        float w1 = n1 - k1;

        // When using multi-threading these need to atomic accross tiles
        if ((k0   >= 0) && (k0   < size0-1) && (k1   >= 0) && (k1   < size1-1))
            d_data[(k1  )*size0 + k0  ] += (1-w0) * (1-w1) * norm;
        if ((k0+1 >= 0) && (k0+1 < size0-1) && (k1   >= 0) && (k1   < size1-1))
            d_data[(k1  )*size0 + k0+1] +=    w0  * (1-w1) * norm;
        if ((k0   >= 0) && (k0   < size0-1) && (k1+1 >= 0) && (k1+1 < size1-1))
            d_data[(k1+1)*size0 + k0  ] += (1-w0) *    w1  * norm;
        if ((k0+1 >= 0) && (k0+1 < size0-1) && (k1+1 >= 0) && (k1+1 < size1-1))
            d_data[(k1+1)*size0 + k0+1] +=    w0  *    w1  * norm;
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


