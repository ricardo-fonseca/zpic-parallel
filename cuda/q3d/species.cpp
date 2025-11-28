#include "species.h"
#include <iostream>

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

/**
 * @brief Returns reciprocal Lorentz gamma factor: $ \frac{1}{\sqrt{u_x^2 + u_y^2 + u_z^2 + 1 }} $
 * 
 * @note Uses CUDA intrinsic fma() and rsqrt() functions
 * 
 * @param u         Generalized momentum in units of c
 * @return float    Reciprocal Lorentz gamma factor
 */
__host__ __device__ __inline__
float rgamma( const float3 u ) {
    return rsqrt( fma( u.z, u.z, fma( u.y, u.y, fma( u.x, u.x, 1.0f ) ) ) );
}

namespace block {

/**
 * @brief Simple class for managing dynamic block shared memory
 * 
 */
struct shared_buffer {
    private:

    /// @brief  Current offset inside shared memory region (in bytes)
    size_t offset;

    /// @brief  Base address of shared memory region
    uint8_t * shm;

    public:

    __device__
    /**
     * @brief Construct a new shared buffer object
     * @note If two or more objects are created inside the same kernel block
     *       they will share the same memory
     */
    shared_buffer() : shm( block::shared_mem<uint8_t>() ), offset(0) { }
    
    /**
     * @brief Reserve memory in the shared memory space
     * 
     * @note 
     * 
     * @tparam T        Type of data
     * @param count     Number of elements (not byte size)
     * @return T*       Pointer to shared memory space
     */
    template< class T >
    __device__
    inline T * get( size_t count ) {
        T * r = reinterpret_cast< T * > ( & shm [ offset ] );
        offset += sizeof( T ) * count;
        return r;
    }
};

}

namespace kernel {

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
template< typename T >
__device__ void interpolate_fld( 
    cyl3<T>  const * const __restrict__ E, 
    cyl3<T>  const * const __restrict__ B, 
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
 * Uses CUDA intrinsic fma() functions
 * 
 * @param tem 
 * @param e 
 * @param b 
 * @param u 
 * @return float3 
 */
__device__ float3 dudt_boris( const float alpha, float3 e, float3 b, float3 u, double & energy )
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
 * @brief Advance memntum using a relativistic Boris pusher for high magnetic fields
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
__device__ float3 dudt_boris_euler( const float alpha, float3 e, float3 b, float3 u, double & energy )
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
        float const bnorm = std::sqrt(fma( b.x, b.x, fma( b.y, b.y, b.z * b.z ) ));
        float const s = -(( bnorm > 0 ) ? std::sin( bnorm / 2 ) / bnorm : 1 );

        float const ra = std::cos( bnorm / 2 );
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
        float const r33 =   fma(ra,ra,rd*rd)-fma(rb,rb,-rc*rc);

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
__device__ __inline__ void dep_current_seg_0(
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

    block::atomic_fetch_add( &J[ stride * j     + i     ].z, wl1 * wp10 );
    block::atomic_fetch_add( &J[ stride * (j+1) + i     ].z, wl1 * wp11 );

    block::atomic_fetch_add( &J[ stride * j     + i     ].r, wl2 * wp20 );
    block::atomic_fetch_add( &J[ stride * j     + (i+1) ].r, wl2 * wp21 );

    block::atomic_fetch_add( &J[ stride * j     + i     ].th, jth * ( S0z0 * S0r0 + S1z0 * S1r0 + (S0z0 * S1r0 - S1z0 * S0r0)/2.0f ) );
    block::atomic_fetch_add( &J[ stride * j     + (i+1) ].th, jth * ( S0z1 * S0r0 + S1z1 * S1r0 + (S0z1 * S1r0 - S1z1 * S0r0)/2.0f ) );
    block::atomic_fetch_add( &J[ stride * (j+1) + i     ].th, jth * ( S0z0 * S0r1 + S1z0 * S1r1 + (S0z0 * S1r1 - S1z0 * S0r1)/2.0f ) );
    block::atomic_fetch_add( &J[ stride * (j+1) + (i+1) ].th, jth * ( S0z1 * S0r1 + S1z1 * S1r1 + (S0z1 * S1r1 - S1z1 * S0r1)/2.0f ) );
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
__device__ __inline__ void dep_current_seg(
    const int ir0, const int2 ix, const float2 x0, const float2 x1,
    const float2 t0, const float2 t1, 
    const float q,
    cyl_cfloat3 * __restrict__ J, const int stride )
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

    const auto cr0 = std::sqrt( fma( t0.x, t0.x, t0.y*t0.y ) );
    const auto cr1 = std::sqrt( fma( t1.x, t1.x, t1.y*t1.y ) );

    /// @brief Initial θ
    const auto th0 = make_float2( t0.x/cr0, t0.y/cr0 );
    /// @brief Final θ
    const auto th1 = make_float2( t1.x/cr1, t1.y/cr1 );

    const auto xif = t0.x + t1.x;
    const auto yif = t0.y + t1.y;

    const auto rm2 = std::sqrt( fma( xif, xif, yif*yif ) );
    const auto thm = float2{ xif/rm2, yif/rm2 };

    // Complex coefficients for initial, mid and final angular positions
    
/*
    auto cm = expimt<m>( thm );
    auto c0 = expimt<m>( th0 ) - cm;
    auto c1 = expimt<m>( th1 ) - cm;
*/

    static_assert( m == 1, "only mode m = 1 is currently supported" );
    auto cm = ops::complex<float>{ thm.x, -thm.y };
    auto c0 = ops::complex<float>{ th0.x, -th0.y } - cm;
    auto c1 = ops::complex<float>{ th1.x, -th1.y } - cm;


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

    ops::block::atomic_add( &J[ stride * j     + i     ].z, q * wl1 * wp10 );
    ops::block::atomic_add( &J[ stride * (j+1) + i     ].z, q * wl1 * wp11 );

    ops::block::atomic_add( &J[ stride * j     + i     ].r, q * wl2 * wp20 );
    ops::block::atomic_add( &J[ stride * j     + (i+1) ].r, q * wl2 * wp21 );

    ops::block::atomic_add( &J[ stride * j     + i     ].th, q * ( S0z1 * S0r1 * c1 - S0z0 * S0r0 * c0 ) );
    ops::block::atomic_add( &J[ stride * j     + (i+1) ].th, q * ( S1z1 * S0r1 * c1 - S1z0 * S0r0 * c0 ) );
    ops::block::atomic_add( &J[ stride * (j+1) + i     ].th, q * ( S0z1 * S1r1 * c1 - S0z0 * S1r0 * c0 ) );
    ops::block::atomic_add( &J[ stride * (j+1) + (i+1) ].th, q * ( S1z1 * S1r1 * c1 - S1z0 * S1r0 * c0 ) );
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
__device__ __inline__ void split2d_cyl( 
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

    // z-split
    float xz, yz, rz;
    if ( cross.x ) {
        εz = (zs - x0.x) / delta.x;

        // z-split positions
        xz = t0.x + εz * tdelta.x;
        yz = t0.y + εz * tdelta.y;
        rz = ( delta.y == 0 ) ? x0.y : std::sqrt( fma( xz, xz, yz * yz ) ) - (ir0 + ix.y);
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
            auto a = fma( tdelta.x, tdelta.x, tdelta.y * tdelta.y );
            auto b = fma( t0.x, tdelta.x, t0.y * tdelta.y );
            auto c = ( x0.y - rs ) * ( 2 * (ir0 + ix.y) + x0.y + rs );

            εr = - ( b + std::copysign( std::sqrt( fma( b, b, - a*c )), b ) ) / a;
            if ( εr < 0 || εr > 1 ) εr = c / (a * εr);
        }
#else
        // Old splitter (OSIRIS)
        εr = (rs - x0.y) / delta.y;
#endif

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


__global__
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
void __launch_bounds__(opt_move_block) move_deposit_0(
    ParticleData const part,
    cyl3<float> * const __restrict__ current_m0, 
    unsigned int const current_offset, uint2 const nx, uint2 const ext_nx,
    float2 const dt_dx, const int2 shift, 
    unsigned long long * const __restrict__ d_nmove
) {
    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const auto ntiles    = part.ntiles;
    const auto tile_size = roundup4( ext_nx.x * ext_nx.y );

    /// @brief [shared] Local copy of current density
    auto * tile_buffer_0 = block::shared_mem< cyl3<float> >();

    // Zero local current buffer
    for( auto i = block_thread_rank(); i < tile_size; i+= block_num_threads() ) 
        tile_buffer_0[i] = cyl3<float>{0};

    block_sync();

    // Move particles and deposit current
    const int tid  = tile_idx.y * ntiles.x + tile_idx.x;

    cyl3<float> * J0 = & tile_buffer_0[ current_offset ];
    const int jstride = ext_nx.x;

    const int part_offset  = part.offset[ tid ];
    const int np           = part.np[ tid ];
    auto * __restrict__ ix = &part.ix[ part_offset ];
    auto * __restrict__ x  = &part.x[ part_offset ];
    auto * __restrict__ u  = &part.u[ part_offset ];
    auto * __restrict__ th = &part.th[ part_offset ];
    auto * __restrict__ q  = &part.q[ part_offset ];

    auto const dt_dz = dt_dx.x;
    auto const dt_dr = dt_dx.y;

    const int ir0 = tile_idx.y * nx.y;

    // Get range of particles to process in block
    const int min_size = 4 * block_num_threads();
    int chunk_size = ( np + gridDim.z - 1 ) / gridDim.z;
    if ( chunk_size < min_size ) chunk_size = min_size;

    int begin = blockIdx.z * chunk_size;
    int end   = begin + chunk_size;

    if ( end > np ) end = np;
    if ( begin > np ) { begin = end = 0; }

    // Move particles and deposit current
    for( auto i = begin + block_thread_rank(); i < end; i+= block_num_threads() ) {
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
        auto rf = std::sqrt( fma( xf, xf, yf*yf ) );
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

    block_sync();

    // If any particles in block, add current to global buffer
    if ( end > begin ) {
        // Add current to global buffer
        const int tile_off = tid * tile_size;

        if ( gridDim.z > 1 ) {
            // When using multiple blocks per tile we must add the current
            // using atomic ops
            for( auto i =  block_thread_rank(); i < tile_size; i+= block_num_threads() ) {
                device::atomic_fetch_add( & current_m0[tile_off + i].z, tile_buffer_0[i].z );
                device::atomic_fetch_add( & current_m0[tile_off + i].r, tile_buffer_0[i].r );
                device::atomic_fetch_add( & current_m0[tile_off + i].th, tile_buffer_0[i].th );
            }
        } else {
            for( auto i =  block_thread_rank(); i < tile_size; i+= block_num_threads() ) {
                current_m0[tile_off + i] += tile_buffer_0[i];
            }
        }
    }

    // Update total particle pushes counter (for performance metrics)
    if ( block_thread_rank() == 0 && blockIdx.z == 0 ) {
        unsigned long long np64 = np;
        device::atomic_fetch_add( d_nmove, np64 );
    }
}

__global__
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
void __launch_bounds__(opt_move_block) move_deposit_1(
    ParticleData const part,
    cyl_float3 * const __restrict__ current_m0, 
    cyl_cfloat3 * const __restrict__ current_m1, 
    unsigned int const current_offset, uint2 const nx, uint2 const ext_nx,
    float2 const dt_dx, const int2 shift,
    unsigned long long * const __restrict__ d_nmove
) {
    const int2 tile_idx  = make_int2( blockIdx.x, blockIdx.y );
    const auto tile_size = roundup4( ext_nx.x * ext_nx.y );

    /// @brief [shared] Local copy of current density
    block::shared_buffer shm;
    auto * tile_buffer_0 = shm.get< cyl_float3  >( tile_size );
    auto * tile_buffer_1 = shm.get< cyl_cfloat3 >( tile_size );

    // Zero local current buffers
    for( auto i = block_thread_rank(); i < tile_size; i+= block_num_threads() ) {
        tile_buffer_0[i] = {0};
        tile_buffer_1[i] = {0};
    }

    block_sync();

    // Move particles and deposit current
    const int tid  = tile_idx.y * part.ntiles.x + tile_idx.x;

    auto * J0 = & tile_buffer_0[ current_offset ];
    auto * J1 = & tile_buffer_1[ current_offset ];

    const int jstride = ext_nx.x;

    const int part_offset  = part.offset[ tid ];
    const int np           = part.np[ tid ];
    auto * __restrict__ ix = &part.ix[ part_offset ];
    auto * __restrict__ x  = &part.x[ part_offset ];
    auto * __restrict__ u  = &part.u[ part_offset ];
    auto * __restrict__ th = &part.th[ part_offset ];
    auto * __restrict__ q  = &part.q[ part_offset ];

    auto const dt_dz = dt_dx.x;
    auto const dt_dr = dt_dx.y;

    const int ir0 = tile_idx.y * nx.y;

    // Get range of particles to process in block
    const int min_size = 4 * block_num_threads();
    int chunk_size = ( np + gridDim.z - 1 ) / gridDim.z;
    if ( chunk_size < min_size ) chunk_size = min_size;

    int begin = blockIdx.z * chunk_size;
    int end   = begin + chunk_size;

    if ( end   > np ) end = np;
    if ( begin > np ) { begin = end = 0; }

    // Move particles and deposit current
    for( auto i = begin + block_thread_rank(); i < end; i+= block_num_threads() ) {
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
        auto rf = std::sqrt( fma( xf, xf, yf*yf ) );
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

    block_sync();

    // If any particles in block, add current to global buffer
    if ( end > begin ) {
        // Add current to global buffer
        const int tile_off = tid * tile_size;

        if ( gridDim.z > 1 ) {
            // When using multiple blocks per tile we must add the current
            // using atomic ops
            for( auto i =  block_thread_rank(); i < tile_size; i+= block_num_threads() ) {
                device::atomic_fetch_add( & current_m0[tile_off + i].z, tile_buffer_0[i].z );
                device::atomic_fetch_add( & current_m0[tile_off + i].r, tile_buffer_0[i].r );
                device::atomic_fetch_add( & current_m0[tile_off + i].th, tile_buffer_0[i].th );
            }

            for( auto i =  block_thread_rank(); i < tile_size; i+= block_num_threads() ) {
                ops::device::atomic_add( & current_m1[tile_off + i].z, tile_buffer_1[i].z );
                ops::device::atomic_add( & current_m1[tile_off + i].r, tile_buffer_1[i].r );
                ops::device::atomic_add( & current_m1[tile_off + i].th, tile_buffer_1[i].th );
            }

        } else {
            for( auto i =  block_thread_rank(); i < tile_size; i+= block_num_threads() )
                current_m0[tile_off + i] += tile_buffer_0[i];

            for( auto i =  block_thread_rank(); i < tile_size; i+= block_num_threads() )
                current_m1[tile_off + i] += tile_buffer_1[i];
        }
    }

    // Update total particle pushes counter (for performance metrics)
    if ( block_thread_rank() == 0 && blockIdx.z == 0 ) {
        unsigned long long np64 = np;
        device::atomic_fetch_add( d_nmove, np64 );
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
__global__
void __launch_bounds__(opt_push_block) push_0 ( 
    ParticleData const part,
    cyl3<float> * __restrict__ d_E, cyl3<float> * __restrict__ d_B, 
    unsigned int const field_offset, uint2 const ext_nx,
    float const alpha, double * __restrict__ d_energy )
{
    const uint2 tile_idx{ blockIdx.x, blockIdx.y };
    const int tid =  tile_idx.y * part.ntiles.x + tile_idx.x;

    int const field_vol = roundup4( ext_nx.x * ext_nx.y );
    int const tile_off = tid * field_vol;

    // Copy E and B into shared memory (mode 0)
    block::shared_buffer shm;
    auto * __restrict__ E_local_m0 = shm.get<cyl3<float>>( field_vol );
    auto * __restrict__ B_local_m0 = shm.get<cyl3<float>>( field_vol );

    block::memcpy( E_local_m0, & d_E[ tile_off ], field_vol );
    block::memcpy( B_local_m0, & d_B[ tile_off ], field_vol );
    
    auto const * const __restrict__ E_m0 = & E_local_m0[ field_offset ];
    auto const * const __restrict__ B_m0 = & B_local_m0[ field_offset ];

    // Push particles
    const int part_offset  = part.offset[ tid ];
    const int np           = part.np[ tid ];
    auto * __restrict__ ix = &part.ix[ part_offset ];
    auto * __restrict__ x  = &part.x[ part_offset ];
    auto * __restrict__ u  = &part.u[ part_offset ];
    auto * __restrict__ th = &part.th[ part_offset ];

    double energy = 0;
    const int jstride = ext_nx.x;

    block_sync();

    const int min_size = 4 * block_num_threads();
    int chunk_size = ( np + gridDim.z - 1 ) / gridDim.z;
    if ( chunk_size < min_size ) chunk_size = min_size;

    int begin = blockIdx.z * chunk_size;
    int end   = begin + chunk_size;

    if ( end   > np ) end = np;
    if ( begin > np ) { begin = end = 0; }

    for( auto i = begin + block_thread_rank(); i < end; i+= block_num_threads() ) {

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
        
        if constexpr ( type == species::boris ) u[i] = dudt_boris( alpha, cart_e, cart_b, pu, energy );
        if constexpr ( type == species::euler ) u[i] = dudt_boris_euler( alpha, cart_e, cart_b, pu, energy );
    }

    // Add up energy from all threads
    energy = warp::reduce_add( energy );
    if ( warp::thread_rank() == 0 ) { 
        device::atomic_fetch_add( d_energy, energy );
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
__global__
void __launch_bounds__(opt_push_block)  push_1 ( 
    ParticleData const part,
    cyl_float3 * __restrict__ d_E_m0, cyl_float3 * __restrict__ d_B_m0, 
    cyl_cfloat3 * __restrict__ d_E_m1, cyl_cfloat3 * __restrict__ d_B_m1, 
    unsigned int const field_offset, uint2 const ext_nx,
    float const alpha, double * const __restrict__ d_energy )
{
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tid =  tile_idx.y * part.ntiles.x + tile_idx.x;

    int const field_vol = roundup4( ext_nx.x * ext_nx.y );
    int const tile_off = tid * field_vol;

    /// @brief Shared memory block
    block::shared_buffer shm;

    // Copy E and B into shared memory (mode 0)
    auto * __restrict__ E_local_m0 = shm.get< cyl_float3 >( field_vol );
    auto * __restrict__ B_local_m0 = shm.get< cyl_float3 >( field_vol );

    block::memcpy( E_local_m0, & d_E_m0[ tile_off ], field_vol );
    block::memcpy( B_local_m0, & d_B_m0[ tile_off ], field_vol );

    auto const * const __restrict__ E_m0 = & E_local_m0[ field_offset ];
    auto const * const __restrict__ B_m0 = & B_local_m0[ field_offset ];

    // Copy E and B into shared memory (mode 1)
    auto * __restrict__ E_local_m1 = shm.get< cyl_cfloat3 >( field_vol );
    auto * __restrict__ B_local_m1 = shm.get< cyl_cfloat3 >( field_vol );

    block::memcpy( E_local_m1, & d_E_m1[ tile_off ], field_vol );
    block::memcpy( B_local_m1, & d_B_m1[ tile_off ], field_vol );

    auto const * const __restrict__ E_m1 = & E_local_m1[ field_offset ];
    auto const * const __restrict__ B_m1 = & B_local_m1[ field_offset ];

    // Push particles
    const int part_offset  = part.offset[ tid ];
    const int np           = part.np[ tid ];
    auto * __restrict__ ix = &part.ix[ part_offset ];
    auto * __restrict__ x  = &part.x[ part_offset ];
    auto * __restrict__ u  = &part.u[ part_offset ];
    auto * __restrict__ th = &part.th[ part_offset ];

    double energy = 0;
    const int jstride = ext_nx.x;

    block_sync();

    const int min_size = 4 * block_num_threads();
    int chunk_size = ( np + gridDim.z - 1 ) / gridDim.z;
    if ( chunk_size < min_size ) chunk_size = min_size;

    int begin = blockIdx.z * chunk_size;
    int end   = begin + chunk_size;

    if ( end   > np ) end = np;
    if ( begin > np ) { begin = end = 0; }

    for( auto i = begin + block_thread_rank(); i < end; i+= block_num_threads() ) {

        // Interpolate field
        cyl3<float> e, b;
        interpolate_fld( E_m0, B_m0, jstride, ix[i], x[i], e, b );

        // Interpolate field - mode 1
        cyl3<ops::complex<float>> e1, b1;
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
        
        if constexpr ( type == species::boris ) u[i] = dudt_boris( alpha, cart_e, cart_b, pu, energy );
        if constexpr ( type == species::euler ) u[i] = dudt_boris_euler( alpha, cart_e, cart_b, pu, energy );
    }

    // Add up energy from all threads
    energy = warp::reduce_add( energy );
    if ( warp::thread_rank() == 0 ) { 
        device::atomic_fetch_add( d_energy, energy );
    }
}

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
    np_inj = nullptr;

    // Set nmodes to invalid value, will be set by initialize
    nmodes = 0;
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

    /// @brief Local parallel node grid size
    auto dims = nx * ntiles;

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
    np_inj = device::malloc<int>( ntiles.x * ntiles.y );

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
    device::exscan_add( particles -> offset, np_inj, ntiles.x * ntiles.y );

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
    device::free( np_inj );

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
 * @param range     Cell range in which to inject
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

namespace kernel {

__global__
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
    ParticleData const part,
    species::bc_type const bc ) 
{
    const auto ntiles  = part.ntiles;
    const int nx = part.nx.x;
    const int2 tile_idx = make_int2( blockIdx.x * ( ntiles.x - 1 ), blockIdx.y );
    
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
            for( int i = block_thread_rank(); i < np; i+= block_num_threads() ) {
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
            for( int i = block_thread_rank(); i < np; i+= block_num_threads() ) {
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

__global__
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
    ParticleData const part,
    species::bc_type const bc ) 
{
    const auto ntiles  = part.ntiles;
    const auto tile_idx = make_int2( blockIdx.x, blockIdx.y * (ntiles.y-1) );
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
            for( int i = block_thread_rank(); i < np; i+= block_num_threads() ) {
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
            for( int i = block_thread_rank(); i < np; i+= block_num_threads() ) {
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

}

/**
 * @brief Processes "physical" boundary conditions
 * 
 */
void Species::process_bc() {

    std::cout << "(*error*) Species::process_bc() have not been implemented yet,"
              << " aborting.\n";
    exit(1);

    dim3 block( 1024 );

    // x boundaries
    if ( bc.x.lower > species::bc::periodic || bc.x.upper > species::bc::periodic ) {
        dim3 grid( 2, particles->ntiles.y );
        kernel::species_bcx <<< grid, block >>> ( *particles, bc );
    }

    // y boundaries
    if ( bc.y.lower > species::bc::periodic || bc.y.upper > species::bc::periodic ) {
        dim3 grid( particles->ntiles.x, 2 );
        kernel::species_bcx <<< grid, block >>> ( *particles, bc );
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
    move();
    
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
void Species::advance( Current &current ) {

    // Advance positions and deposit current
    move( current );

    // Process physical boundary conditions
    // process_bc();
    
    // Sort particles according to tile
    particles -> tile_sort( *tmp, *sort );

    // Increase internal iteration number
    iter++;
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
    push( emf );

    // Advance positions and deposit current
    move( current );

    // Process physical boundary conditions
    // process_bc();
    
    // Sort particles according to tile
    particles -> tile_sort( *tmp, *sort );

    // Increase internal iteration number
    iter++;
}

void Species::advance_mov_window( Current &current ) {

    if ( moving_window.needs_move( (iter+1) * dt ) ) {

        // Advance positions, deposit current and shift particles
        move( current, make_int2(-1,0) );

        // Process boundary conditions
        // process_bc();

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
        // process_bc();

        // Sort particles over tiles
        particles -> tile_sort( *tmp, *sort );
    }

    // Increase internal iteration number
    iter++;
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
    push( emf );

    if ( moving_window.needs_move( (iter+1) * dt ) ) {

        // Advance positions, deposit current and shift particles
        move( current, make_int2(-1,0) );

        // Process boundary conditions
        // process_bc();

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
        // process_bc();

        // Sort particles over tiles
        particles -> tile_sort( *tmp, *sort );
    }

    // Increase internal iteration number
    iter++;
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

#if 0
    int tile_blocks = opt_min_blocks / (particles -> ntiles.x * particles -> ntiles.y);
    if ( tile_blocks < 1 ) tile_blocks = 1;   
    dim3 grid( particles -> ntiles.x, particles -> ntiles.y, tile_blocks );
    auto block = opt_move_block;
#else
    dim3 grid( particles -> ntiles.x, particles -> ntiles.y );
    auto block = opt_move_block;
#endif

    size_t shm_size = J0.tile_vol * sizeof( cyl_float3 );

    switch (nmodes) {
    case 1: {
        
        block::set_shmem_size( kernel::move_deposit_0, shm_size );
        kernel::move_deposit_0 <<< grid, block, shm_size >>> ( 
            *particles,
            J0.d_buffer, J0.offset, J0.nx, J0.ext_nx, 
            dt_dx, shift, d_nmove.ptr()
        );
    } break;
    case 2: {
        auto & J1 = current.mode(1);
        shm_size += J1.tile_vol * sizeof( cyl_cfloat3 );
        block::set_shmem_size( kernel::move_deposit_1, shm_size );
        kernel::move_deposit_1 <<< grid, block, shm_size >>> ( 
            *particles,
            J0.d_buffer, J1.d_buffer, J0.offset, J0.nx, J0.ext_nx, 
            dt_dx, shift, d_nmove.ptr()
        );
    }
    default:
        break;
    }
}

namespace kernel {

__global__
/**
 * @brief Kernel for moving particles without depositing current
 * 
 * @param part          Particle data
 * @param dt_dx         Ratio between time step and cell size (x/y)
 * @param d_nmove       (out) Number of particles pushed (for performance metrics)
 */
void move(
    ParticleData part,
    float2 const dt_dx, int2 const shift,
    unsigned long long * const __restrict__ d_nmove
) {

    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tid = tile_idx.y * part.ntiles.x + tile_idx.x;

    const int part_offset    = part.offset[ tid ];
    const int np             = part.np[ tid ];
    auto * __restrict__ ix   = &part.ix[ part_offset ];
    auto * __restrict__ x    = &part.x[ part_offset ];
    auto * __restrict__ u    = &part.u[ part_offset ];
    auto * __restrict__ th   = &part.th[ part_offset ];
    
    auto const dt_dz = dt_dx.x;
    auto const dt_dr = dt_dx.y;

    for( int i = block_thread_rank(); i < np; i+= block_num_threads() ) {
        float3 pu = u[i];
        float2 x0 = x[i];
        int2 ix0 = ix[i];

        auto cos_th = th[i].x;
        auto sin_th = th[i].y; 

        // Get 1 / Lorentz gamma
        float rg = rgamma( pu );

        // Cartesian motion
        auto Δx = dt_dr * rg * pu.x;
        auto Δy = dt_dr * rg * pu.y;
        auto Δz = dt_dz * rg * pu.z;

        // New cartesian positions
        auto ri = ix0.y + x0.y;
        auto xf = fma( ri, cos_th, Δx );
        auto yf = fma( ri, sin_th, Δy );

        // New radial position
        auto rf = std::sqrt( fma( xf, xf, yf*yf ) );

        // Protection agains rf == 0
        // This is VERY unlikely
        float Δr;
        if ( rf > 0 ) {
            Δr = fma( Δx , fma( ri, cos_th, xf ) , Δy * fma( ri, sin_th, yf ) ) / (rf + ri);
            cos_th = xf/rf;
            sin_th = yf/rf;
        } else {
            Δr   = -ri;
            cos_th = 1;
            sin_th = 0;
        }

        // Store new angular position
        th[i] = float2{ cos_th, sin_th };

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
            ix0.x + deltai.x + shift.x,
            ix0.y + deltai.y + shift.y
        );
        ix[i] = ix1;
    }

    if ( block_thread_rank() == 0 ) { 
        // Update total particle pushes counter (for performance metrics)
        unsigned long long np64 = np;
        device::atomic_fetch_add( (unsigned long long *) d_nmove, np64 );
    }
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
void Species::move( const int2 shift )
{
    const float2 dt_dx = make_float2(
        dt / dx.x,
        dt / dx.y
    );

    dim3 grid( particles -> ntiles.x, particles -> ntiles.y );
    auto block = 512;

    kernel::move <<< grid, block >>> ( 
        *particles, dt_dx, shift, d_nmove.ptr()
    );
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
    d_energy = 0.;

    auto & E0 = emf.E -> mode0();
    auto & B0 = emf.B -> mode0();
    size_t shm_size = 2 * ( E0.tile_vol * sizeof(cyl3<float>) );

#if 1
    int tile_blocks = opt_min_blocks / (particles -> ntiles.x * particles -> ntiles.y);
    if ( tile_blocks < 1 ) tile_blocks = 1;
    dim3 grid( particles -> ntiles.x, particles -> ntiles.y, tile_blocks );
    auto block = opt_push_block;
#else
    dim3 grid( particles -> ntiles.x, particles -> ntiles.y );
    auto block = opt_push_block;
#endif

    switch (nmodes) {
    case 1: {
        block::set_shmem_size( kernel::push_0 <species::boris>, shm_size );
        kernel::push_0 <species::boris> <<< grid, block, shm_size >>> ( 
            *particles,
            E0.d_buffer, B0.d_buffer, E0.offset, E0.ext_nx, alpha,
            d_energy.ptr()
        );
    } break;
    case 2: {
        auto & E1 = emf.E -> mode(1);
        auto & B1 = emf.B -> mode(1);

        // Add shared memory size for mode 1 fields
        shm_size += 2 * ( E1.tile_vol * sizeof(cyl3<ops::complex<float>>) );

        block::set_shmem_size( kernel::push_1 <species::boris>, shm_size );
        kernel::push_1 <species::boris> <<< grid, block, shm_size >>> ( 
            *particles,
            E0.d_buffer, B0.d_buffer, 
            E1.d_buffer, B1.d_buffer, 
            E0.offset, E0.ext_nx, alpha,
            d_energy.ptr()
        );
    } break;

    default:
        break;
    }
}

namespace kernel {
__global__
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
    ParticleData const part,
    float * const __restrict__ d_charge, int offset, uint2 ext_nx )
{
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const auto tile_vol = roundup4( ext_nx.x * ext_nx.y );

    auto * charge_local = block::shared_mem<float>();

    // Zero shared memory and sync.
    for( auto i = block_thread_rank(); i < tile_vol; i += block_num_threads() ) {
        charge_local[i] = 0;
    }

    float *charge = &charge_local[ offset ];

    block_sync();

    const int tid      = tile_idx.y * part.ntiles.x + tile_idx.x;
    const int part_off = part.offset[ tid ];
    const int np       = part.np[ tid ];
    auto const * __restrict__ const ix = &part.ix[ part_off ];
    auto const * __restrict__ const x  = &part.x[ part_off ];
    auto const * __restrict__ const q  = &part.q[ part_off ];
    const int jstride = ext_nx.x; 

    for( int i = block_thread_rank(); i < np; i += block_num_threads() ) {
        const int idx = ix[i].y * jstride + ix[i].x;
        const float s0z = 0.5f - x[i].x;
        const float s1z = 0.5f + x[i].x;
        const float s0r = 0.5f - x[i].y;
        const float s1r = 0.5f + x[i].y;

        block::atomic_fetch_add( & charge[ idx               ], s0r * s0z * q[i] );
        block::atomic_fetch_add( & charge[ idx + 1           ], s0r * s1z * q[i] );
        block::atomic_fetch_add( & charge[ idx     + jstride ], s1r * s0z * q[i] );
        block::atomic_fetch_add( & charge[ idx + 1 + jstride ], s1r * s1z * q[i] );
    }

    block_sync();


    // Copy data to global memory
    const int tile_off = tid * tile_vol;
    for( auto i = block_thread_rank(); i < tile_vol; i += block_num_threads() ) {
        d_charge[ tile_off + i ] += charge_local[i];
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
__global__ void dep_charge(
    ParticleData const part,
    ops::complex<float> * const __restrict__ d_charge, int offset, uint2 ext_nx )
{
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_size = roundup4( ext_nx.x * ext_nx.y );
 
    extern __shared__ ops::complex<float> charge_local_m[];

    // Zero shared memory and sync.
    for( unsigned i = 0; i < ext_nx.x * ext_nx.y; i ++ ) {
        charge_local_m[i] = 0;
    }

    auto *charge = &charge_local_m[ offset ];

    block_sync();

    const int tid      = tile_idx.y * part.ntiles.x + tile_idx.x;
    const int part_off = part.offset[ tid ];
    const int np       = part.np[ tid ];
    auto const * __restrict__ const ix = &part.ix[ part_off ];
    auto const * __restrict__ const x  = &part.x[ part_off ];
    auto const * __restrict__ const q  = &part.q[ part_off ];
    auto const * __restrict__ const th = &part.th[ part_off ];
    const int jstride = ext_nx.x;

    for( int i = block_thread_rank(); i < np; i += block_num_threads() ) {
        const int idx = ix[i].y * jstride + ix[i].x;
        const float s0z = 0.5f - x[i].x;
        const float s1z = 0.5f + x[i].x;
        const float s0r = 0.5f - x[i].y;
        const float s1r = 0.5f + x[i].y;

        static_assert( m == 1, "only mode m = 1 is currently supported" );
        // auto qm = q[i] * expimt<m>( t[i] );
        auto qm = q[i] * ops::complex<float>{ th[i].x, -th[i].y };

        ops::block::atomic_add( & charge[ idx               ], s0r * s0z * qm );
        ops::block::atomic_add( & charge[ idx + 1           ], s0r * s1z * qm );
        ops::block::atomic_add( & charge[ idx     + jstride ], s1r * s0z * qm );
        ops::block::atomic_add( & charge[ idx + 1 + jstride ], s1r * s1z * qm );
    }

    block_sync();

    // Copy data to global memory
    const int tile_off = tid * roundup4( ext_nx.x * ext_nx.y );
    for( unsigned i = 0; i < ext_nx.x * ext_nx.y; i ++ ) {
        d_charge[tile_off + i] += charge_local_m[i];
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
__global__ void charge_norm(
    uint2 const ntiles,
    T * const __restrict__ d_charge, int offset, 
    uint2 const nx, uint2 const ext_nx,
    const float dr, const float scale = 1.0f
) {
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    auto tid = tile_idx.y * ntiles.x + tile_idx.x;
    const int tile_off = tid * roundup4( ext_nx.x * ext_nx.y );
    const int jstride = ext_nx.x;

    auto * __restrict__ charge = &  d_charge[ tile_off + offset ];

    int ir0 = tile_idx.y * nx.y;
    for( int k = block_thread_rank(); k < (nx.y+1)*(nx.x+1); k += block_num_threads() ) {
        int j = k / (nx.x+1);
        int i = k % (nx.x+1);
        auto norm = scale/(std::abs( ir0 + j - 0.5f) * dr);
        charge[ j * jstride +i ] *= norm;
    }

    // Axial boundary
    // Fold values for r < 0 back into simulation domain
    if ( ir0 == 0 ) {
        for( int i = block_thread_rank(); i < static_cast<int>(nx.x+1); i+= block_num_threads() ){
            charge[ i + 1 * jstride ] += charge[ i + 0 * jstride ];
            charge[ i + 0 * jstride ]  = charge[ i + 1 * jstride ];
        }
    }
}

}

/**
 * @brief Deposit charge density (mode 0)
 * 
 * @param charge0    Charge density grid
 */
void Species::deposit_charge0( grid<float> & charge0 ) const {

    dim3 grid( particles -> ntiles.x, particles -> ntiles.y );
    auto block = 64;
    auto shm_size = charge0.tile_vol * sizeof(float);

    block::set_shmem_size( kernel::dep_charge_0, shm_size );
    kernel::dep_charge_0 <<< grid, block, shm_size >>> (
        *particles,
        charge0.d_buffer, charge0.offset, charge0.ext_nx
    );

    kernel::charge_norm <<< grid, block >>> (
        charge0.get_ntiles(), charge0.d_buffer, charge0.offset, 
        charge0.nx, charge0.ext_nx, dx.y
    );
}

/**
 * @brief Deposit charge density high order modes
 * 
 * @param m         Cylindrical mode to deposit (1 to 4)
 * @param charge    Charge density grid (complex)
 */
void Species::deposit_charge( const unsigned m, grid<ops::complex<float>> &charge ) const {

    if ( m != 1  ) {
        std::cerr << "(*error*) Only modes m = 1 is currently supported, aborting...\n";
        std::exit(1);
    }

    dim3 grid( particles -> ntiles.x, particles -> ntiles.y );
    auto block = 64;
    size_t shm_size = charge.tile_vol * sizeof(ops::complex<float>);

    block::set_shmem_size( kernel::dep_charge<1>, shm_size );
    kernel::dep_charge<1> <<< grid, block, shm_size >>> (
        *particles,
        charge.d_buffer, charge.offset, charge.ext_nx
    );

    // High-order modes need an additional factor of 2
    kernel::charge_norm <<< grid, block >>> (
        charge.get_ntiles(), charge.d_buffer, charge.offset, 
        charge.nx, charge.ext_nx, dx.y, 2.f
    );
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
        grid<ops::complex<float>> charge( particles -> ntiles, particles -> nx, gc );
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

    const part::quant quants[] = {
        part::quant::z, part::quant::r,
        part::quant::q,
        part::quant::cos_th, part::quant::sin_th,
        part::quant::ux, part::quant::uy, part::quant::uz 
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
    uint32_t np = particles -> np_total();
    info.np = np;

    // Open file
    zdf::file part_file;
    zdf::open_part_file( part_file, info, iter_info, path + '/' + name );

    if ( np > 0 ) {

        float * d_data = device::malloc<float>( np );
        float * h_data = host::malloc<float>( np );

        // z and r quantities are scaled before saving to file
        particles -> gather( part::quant::z, make_float2( dx.x, moving_window.motion() ), d_data );
        device::memcpy_tohost( h_data, d_data, np );
        zdf::add_quant_part_file( part_file, "z", h_data, np );

        particles -> gather( part::quant::r, make_float2( dx.y, 0 ), d_data );
        device::memcpy_tohost( h_data, d_data, np );
        zdf::add_quant_part_file( part_file, "r", h_data, np );

        // Remaining quantities are saved "as is"
        for( auto i = 2; i < info.nquants; i++ ) {
            particles -> gather( quants[i], d_data );
            device::memcpy_tohost( h_data, d_data, np );
            zdf::add_quant_part_file( part_file, qnames[i], h_data, np );
        }

        // Free temporary memory
        host::free( h_data );
        device::free( d_data );

    } else {
        // No particles - root node creates an empty file
        for ( auto i = 0; i < info.nquants; i ++) {
            zdf::add_quant_part_file( part_file, info.quants[i],  nullptr, 0 );
        }
    }

    // Close the file
    zdf::close_file( part_file );
}

namespace kernel {

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
template < phasespace::quant quant >
__global__
void dep_pha1(
    float * const __restrict__ d_data, float2 const range, int const size,
    float const norm, 
    ParticleData const part )
{
    uint2 const tile_nx = part.nx;

    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id  = tile_idx.y * part.ntiles.x + tile_idx.x;

    const int part_offset  = part.offset[ tile_id ];
    const int np           = part.np[ tile_id ];
    auto * __restrict__ ix = &part.ix[ part_offset ];
    auto * __restrict__ x  = &part.x[ part_offset ];
    auto * __restrict__ u  = &part.u[ part_offset ];
    auto * __restrict__ q  = &part.q[ part_offset ];

    float const pha_rdx = size / (range.y - range.x);

    const int shiftx = tile_idx.x * tile_nx.x;
    const int shifty = tile_idx.y * tile_nx.y;

    for( int i = block_thread_rank(); i < np; i += block_num_threads() ) {
        float d;
        if constexpr ( quant == phasespace:: z  ) d = ( shiftx + ix[i].x) + (x[i].x + 0.5f);
        if constexpr ( quant == phasespace:: r  ) d = ( shifty + ix[i].y) + 0.5f;
        if constexpr ( quant == phasespace:: ux ) d = u[i].x;
        if constexpr ( quant == phasespace:: uy ) d = u[i].y;
        if constexpr ( quant == phasespace:: uz ) d = u[i].z;

        float n =  (d - range.x ) * pha_rdx - 0.5f;
        int   k = static_cast<int>( n + 1 ) - 1;
        float w = n - k;

        if ((k   >= 0) && (k   < size-1)) device::atomic_fetch_add( &d_data[k  ], (1-w) * norm * q[i] );
        if ((k+1 >= 0) && (k+1 < size-1)) device::atomic_fetch_add( &d_data[k+1],    w  * norm * q[i] );
    }
}

}

/**
 * @brief Deposit 1D phasespace
 * 
 * @note Output data will be zeroed before deposition
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
    device::zero( d_data, size );
    
    float norm = q_ref * ( dx.x * dx.y ) *
                 size / (range.y - range.x) ;

    dim3 grid( particles -> ntiles.x, particles -> ntiles.y );
    auto block = 64;

    switch(quant) {
    case( phasespace::z ):
        range.y /= dx.x;
        range.x /= dx.x;
        kernel::dep_pha1<phasespace::z> <<< grid, block >>> 
            ( d_data, range, size, norm, *particles );
        break;
    case( phasespace:: r ):
        range.y /= dx.y;
        range.x /= dx.y;
        kernel::dep_pha1<phasespace::r> <<< grid, block >>> 
            ( d_data, range, size, norm, *particles );
        break;
    case( phasespace:: ux ):
        kernel::dep_pha1<phasespace::ux> <<< grid, block >>> 
            ( d_data, range, size, norm, *particles );
        break;
    case( phasespace:: uy ):
        kernel::dep_pha1<phasespace::uy> <<< grid, block >>> 
            ( d_data, range, size, norm, *particles );
        break;
    case( phasespace:: uz ):
        kernel::dep_pha1<phasespace::uz> <<< grid, block >>> 
            ( d_data, range, size, norm, *particles );
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

    // Deposit 1D phasespace
    float * d_data = device::malloc<float>( size );
    float * h_data = host::malloc<float>( size );
    
    dep_phasespace( d_data, quant, range, size );
    device::memcpy_tohost(  h_data, d_data, size );

    // Save file
    zdf::save_grid( h_data, info, iter_info, "PHASESPACE/" + name );

    host::free( h_data );
    device::free( d_data );
}

namespace kernel {

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
__global__
void dep_pha2(
    float * const __restrict__ d_data, 
    float2 const range0, int const size0,
    float2 const range1, int const size1,
    float const norm, 
    ParticleData const part
) {
    static_assert( quant1 > quant0, "quant1 must be > quant0" );
    
    const auto tile_nx  = part.nx;

    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id  = tile_idx.y * part.ntiles.x + tile_idx.x;

    const int part_offset = part.offset[ tile_id ];
    const int np          = part.np[ tile_id ];
    auto * __restrict__ ix  = &part.ix[ part_offset ];
    auto * __restrict__ x   = &part.x[ part_offset ];
    auto * __restrict__ u   = &part.u[ part_offset ];
    auto * __restrict__ q   = &part.q[ part_offset ];

    float const pha_rdx0 = size0 / (range0.y - range0.x);
    float const pha_rdx1 = size1 / (range1.y - range1.x);

    const int shiftx = tile_idx.x * tile_nx.x;
    const int shifty = tile_idx.y * tile_nx.y;

    for( int i = block_thread_rank(); i < np; i += block_num_threads() ) {
        float d0;
        if constexpr ( quant0 == phasespace:: z )  d0 = ( shiftx + ix[i].x) + (x[i].x + 0.5f);
        if constexpr ( quant0 == phasespace:: r )  d0 = ( shifty + ix[i].y) + x[i].y;
        if constexpr ( quant0 == phasespace:: ux ) d0 = u[i].x;
        if constexpr ( quant0 == phasespace:: uy ) d0 = u[i].y;
        if constexpr ( quant0 == phasespace:: uz ) d0 = u[i].z;

        float n0 =  (d0 - range0.x ) * pha_rdx0 - 0.5f;
        int   k0 = static_cast<int>( n0 + 1 ) - 1;
        float w0 = n0 - k0;

        float d1;
        // if constexpr ( quant1 == phasespace:: z )  d1 = ( shiftx + ix[i].x) + (x[i].x + 0.5f);
        if constexpr ( quant1 == phasespace:: r )  d1 = ( shifty + ix[i].y) + x[i].y;
        if constexpr ( quant1 == phasespace:: ux ) d1 = u[i].x;
        if constexpr ( quant1 == phasespace:: uy ) d1 = u[i].y;
        if constexpr ( quant1 == phasespace:: uz ) d1 = u[i].z;

        float n1 =  (d1 - range1.x ) * pha_rdx1 - 0.5f;
        int   k1 = static_cast<int>( n1 + 1 ) - 1;
        float w1 = n1 - k1;

        if ((k0   >= 0) && (k0   < size0-1) && (k1   >= 0) && (k1   < size1-1))
            device::atomic_fetch_add( &d_data[(k1  )*size0 + k0  ] , (1-w0) * (1-w1) * norm * q[i] );
        if ((k0+1 >= 0) && (k0+1 < size0-1) && (k1   >= 0) && (k1   < size1-1))
            device::atomic_fetch_add( &d_data[(k1  )*size0 + k0+1] ,    w0  * (1-w1) * norm * q[i] );
        if ((k0   >= 0) && (k0   < size0-1) && (k1+1 >= 0) && (k1+1 < size1-1))
            device::atomic_fetch_add( &d_data[(k1+1)*size0 + k0  ] , (1-w0) *    w1  * norm * q[i] );
        if ((k0+1 >= 0) && (k0+1 < size0-1) && (k1+1 >= 0) && (k1+1 < size1-1))
            device::atomic_fetch_add( &d_data[(k1+1)*size0 + k0+1] ,    w0  *    w1  * norm * q[i] );
    }
}

}

/**
 * @brief Deposits a 2D phasespace in a device buffer
 * 
 * @note Output data will be zeroed before deposition
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
    device::zero( d_data, size0 * size1 );

    float norm = q_ref * ( dx.x * dx.y ) *
                 ( size0 / (range0.y - range0.x) ) *
                 ( size1 / (range1.y - range1.x) );

    dim3 grid( particles -> ntiles.x, particles -> ntiles.y );
    auto block = 64;

    switch(quant0) {
    case( phasespace::z ):
        range0.y /= dx.x;
        range0.x /= dx.x;
        switch(quant1) {
        case( phasespace::r ):
            range1.y /= dx.y;
            range1.x /= dx.y;
            kernel::dep_pha2 <phasespace::z,phasespace::r> <<< grid, block >>> (
                d_data, range0, size0, range1, size1, norm, *particles
            );
            break;
        case( phasespace::ux ):
            kernel::dep_pha2 <phasespace::z,phasespace::ux> <<< grid, block >>> (
                d_data, range0, size0, range1, size1, norm, *particles
            );
            break;
        case( phasespace::uy ):
            kernel::dep_pha2 <phasespace::z,phasespace::uy> <<< grid, block >>> (
                d_data, range0, size0, range1, size1, norm, *particles
            );
            break;
        case( phasespace::uz ):
            kernel::dep_pha2 <phasespace::z,phasespace::uz> <<< grid, block >>> (
                d_data, range0, size0, range1, size1, norm, *particles
            );
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
            kernel::dep_pha2 <phasespace::r,phasespace::ux> <<< grid, block >>> (
                d_data, range0, size0, range1, size1, norm, *particles
            );
            break;
        case( phasespace::uy ):
            kernel::dep_pha2 <phasespace::r,phasespace::uy> <<< grid, block >>> (
                d_data, range0, size0, range1, size1, norm, *particles
            );
            break;
        case( phasespace::uz ):
            kernel::dep_pha2 <phasespace::r,phasespace::uz> <<< grid, block >>> (
                d_data, range0, size0, range1, size1, norm, *particles
            );
            break;
        default:
            break;
        }
        break;
    case( phasespace:: ux ):
        switch(quant1) {
        case( phasespace::uy ):
            kernel::dep_pha2 <phasespace::ux,phasespace::uy> <<< grid, block >>> (
                d_data, range0, size0, range1, size1, norm, *particles
            );
            break;
        case( phasespace::uz ):
            kernel::dep_pha2 <phasespace::ux,phasespace::uz> <<< grid, block >>> (
                d_data, range0, size0, range1, size1, norm, *particles
            );
            break;
        default:
            break;
        }
        break;
    case( phasespace:: uy ):
        kernel::dep_pha2 <phasespace::uy,phasespace::uz> <<< grid, block >>> (
            d_data, range0, size0, range1, size1, norm, *particles
        );
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

    float * d_data = device::malloc<float>( size0 * size1 );
    float * h_data = host::malloc<float>( size0 * size1 );

    dep_phasespace( d_data, quant0, range0, size0, quant1, range1, size1 );
    device::memcpy_tohost(  h_data, d_data, size0 * size1 );

    zdf::save_grid( h_data, info, iter_info, "PHASESPACE/" + name );

    host::free( h_data );
    device::free( d_data );
}

