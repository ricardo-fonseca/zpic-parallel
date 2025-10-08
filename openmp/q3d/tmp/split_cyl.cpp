#include "../vec_types.h"
#include "../utils.h"

/**
 * @file split_cyl.cpp
 * @author Ricardo Fonseca
 * @brief Test the new cylindrical splitter
 * @version 0.1
 * @date 2025-09-26
 * 
 * @copyright Copyright (c) 2025
 * 
 */


inline void split2d_cyl( 
    const int2 ix, const float2 x0, const float2 delta, int2 & deltai,
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
        rz = sqrt( ops::fma( xz, xz, yz * yz ) ) - ix.y;
    }

    // r-split
    float xr, yr, zr;
    if ( cross.y ) {
        auto a = ops::fma( tdelta.x, tdelta.x, tdelta.y * tdelta.y );
        auto b = ops::fma( t0.x, tdelta.x, t0.y * tdelta.y );
        auto c = ( x0.y - rs ) * ( 2 * ix.y + x0.y + rs );

        εr = - ( b + std::copysign( sqrt( ops::fma( b, b, - a*c )), b ) ) / a;
        if ( εr < 0 || εr >= 1 ) εr = c / (a * εr);

        std::cout << "(*info*) εr: " << εr << '\n';
        if ( εr <= 0 || εr >= 1) {
            std::cerr << "(*error*) Invalid εr: "<< εr << '\n'
                      << "ti: " << t0 << ", tdelta: " << tdelta << '\n'
                      << "a: " << a << ", b: " << b << ", c: " << c << '\n';
            std::exit(1);
        }

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
            std::cout << "(*info*) z_cross 1st " << '\n';

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
            std::cout << "(*info*) r_cross 1st " << '\n';

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

void split_info( int2 cross, int2 & v0_ix, float2 & v0_x0, float2 & v0_x1, float2 & v0_t0, float2 & v0_t1,
    int2 & v1_ix, float2 & v1_x0, float2 & v1_x1, float2 & v1_t0, float2 & v1_t1,
    int2 & v2_ix, float2 & v2_x0, float2 & v2_x1, float2 & v2_t0, float2 & v2_t1 ) {

    int nsplit = cross.x + cross.y;
    std::cout << "no. cell crossings: " << nsplit << '\n';


    std::cout << "seg. 0\n";
    std::cout << "ix: " << v0_ix << ", " << v0_x0 << "→" << v0_x1 << ", "
              << v0_t0 << "→" << v0_t1 << '\n';

    float2 delta = make_float2( v0_x1.x - v0_x0.x, v0_x1.y - v0_x0.y );
    float2 tdelta = make_float2( v0_t1.x - v0_t0.x, v0_t1.y - v0_t0.y );


    if ( nsplit > 0 ) {
        std::cout << "seg. 1\n";
        std::cout << "ix: " << v1_ix << ", " << v1_x0 << "→" << v1_x1 << ", "
                << v1_t0 << "→" << v1_t1 << '\n';

        delta += make_float2( v1_x1.x - v1_x0.x, v1_x1.y - v1_x0.y );
        tdelta += make_float2( v1_t1.x - v1_t0.x, v1_t1.y - v1_t0.y );
    }

    if ( nsplit > 1 ) {
        std::cout << "seg. 2\n";
        std::cout << "ix: " << v2_ix << ", " << v2_x0 << "→" << v2_x1 << ", "
                << v2_t0 << "→" << v2_t1 << '\n';
        delta += make_float2( v2_x1.x - v2_x0.x, v2_x1.y - v2_x0.y );
        tdelta += make_float2( v2_t1.x - v2_t0.x, v2_t1.y - v2_t0.y );
    }

    std::cout << "delta: " << delta << ", " << tdelta << '\n';
}

int main( void ) {

    int2 ix; float2 x0; float2 delta;
    int2 deltai;
    
    float2 t0; float2 tdelta;

    int2 v0_ix; float2 v0_x0, v0_x1, v0_t0, v0_t1;
    int2 v1_ix; float2 v1_x0, v1_x1, v1_t0, v1_t1;
    int2 v2_ix; float2 v2_x0, v2_x1, v2_t0, v2_t1;

    int2 cross;

    // Initial positions
    // These must be self consistent i.e.
    // ir + r = x^2 + y^2
    // ang( x / (ir+r), y / (ir+r))
    t0 = make_float2( 4, 3 );
    ix = make_int2( 3, 5 );
    x0 = make_float2( 0.1, 0.0 );
    float2 ang = make_float2( 4./5., 3./5.);

    std::cout << ansi::bold << "Initial positions\n" << ansi::reset;
    std::cout << "ix: " << ix << ", x0: " << x0 << ", t0: " << t0 << '\n';

    // --- no split

    std::cout << ansi::bold << "\nNo split\n" << ansi::reset;

    delta = make_float2( 0.1, 0.1 );
    // No angular motion
    tdelta = make_float2( delta.y * ang.x, delta.y * ang.y );
    std::cout << "delta: " << delta << ", " << tdelta << '\n';

    split2d_cyl( ix, x0, delta, deltai, t0, tdelta,
                v0_ix, v0_x0, v0_x1, v0_t0, v0_t1,
                v1_ix, v1_x0, v1_x1, v1_t0, v1_t1,
                v2_ix, v2_x0, v2_x1, v2_t0, v2_t1,
                cross );

    split_info( cross, 
                v0_ix, v0_x0, v0_x1, v0_t0, v0_t1,
                v1_ix, v1_x0, v1_x1, v1_t0, v1_t1,
                v2_ix, v2_x0, v2_x1, v2_t0, v2_t1 );

    // --- z split

    std::cout << ansi::bold << "\nz-split only\n" << ansi::reset;

    delta = make_float2( 0.5, 0.1 );

    // No angular motion
    tdelta = make_float2( delta.y * ang.x, delta.y * ang.y );

    std::cout << "delta: " << delta << ", " << tdelta << '\n';

    split2d_cyl( ix, x0, delta, deltai, t0, tdelta,
                v0_ix, v0_x0, v0_x1, v0_t0, v0_t1,
                v1_ix, v1_x0, v1_x1, v1_t0, v1_t1,
                v2_ix, v2_x0, v2_x1, v2_t0, v2_t1,
                cross );

    split_info( cross, 
                v0_ix, v0_x0, v0_x1, v0_t0, v0_t1,
                v1_ix, v1_x0, v1_x1, v1_t0, v1_t1,
                v2_ix, v2_x0, v2_x1, v2_t0, v2_t1 );

    // --- r split

    std::cout << ansi::bold << "\nr-split only\n" << ansi::reset;

    delta = make_float2( 0.1, 0.6 );

    // No angular motion
    tdelta = make_float2( delta.y * ang.x, delta.y * ang.y );
    std::cout << "delta: " << delta << ", " << tdelta << '\n';

    split2d_cyl( ix, x0, delta, deltai, t0, tdelta,
                v0_ix, v0_x0, v0_x1, v0_t0, v0_t1,
                v1_ix, v1_x0, v1_x1, v1_t0, v1_t1,
                v2_ix, v2_x0, v2_x1, v2_t0, v2_t1,
                cross );

    split_info( cross, 
                v0_ix, v0_x0, v0_x1, v0_t0, v0_t1,
                v1_ix, v1_x0, v1_x1, v1_t0, v1_t1,
                v2_ix, v2_x0, v2_x1, v2_t0, v2_t1 );

    // --- zr split A

    std::cout << ansi::bold << "\nzr-split A\n" << ansi::reset;

    delta = make_float2( 0.9, 0.6 );

    // No angular motion
    tdelta = make_float2( delta.y * ang.x, delta.y * ang.y );
    std::cout << "delta: " << delta << ", " << tdelta << '\n';

    split2d_cyl( ix, x0, delta, deltai, t0, tdelta,
                v0_ix, v0_x0, v0_x1, v0_t0, v0_t1,
                v1_ix, v1_x0, v1_x1, v1_t0, v1_t1,
                v2_ix, v2_x0, v2_x1, v2_t0, v2_t1,
                cross );

    split_info( cross, 
                v0_ix, v0_x0, v0_x1, v0_t0, v0_t1,
                v1_ix, v1_x0, v1_x1, v1_t0, v1_t1,
                v2_ix, v2_x0, v2_x1, v2_t0, v2_t1 );

    // --- zr split B

    std::cout << ansi::bold << "\nzr-split B\n" << ansi::reset;

    delta = make_float2( 0.6, 0.9 );

    // No angular motion
    tdelta = make_float2( delta.y * ang.x, delta.y * ang.y );
    std::cout << "delta: " << delta << ", " << tdelta << '\n';

    split2d_cyl( ix, x0, delta, deltai, t0, tdelta,
                v0_ix, v0_x0, v0_x1, v0_t0, v0_t1,
                v1_ix, v1_x0, v1_x1, v1_t0, v1_t1,
                v2_ix, v2_x0, v2_x1, v2_t0, v2_t1,
                cross );

    split_info( cross, 
                v0_ix, v0_x0, v0_x1, v0_t0, v0_t1,
                v1_ix, v1_x0, v1_x1, v1_t0, v1_t1,
                v2_ix, v2_x0, v2_x1, v2_t0, v2_t1 );

}
