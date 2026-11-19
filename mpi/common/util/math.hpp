#pragma once

#include <cmath>

/**
 * @brief Rounds up to a multiple of N (where N is a power of 2)
 * 
 * @tparam N    Value will be rounded to a multiple of N. Must be a power of 2.
 * @tparam T    Value type. Must be an integer type (int, long, unsigned, int64_t, etc.)
 * @param a     Value to round up
 * @return T    Value rounded up to a multiple of N
 */
template < int N, typename T >
constexpr T roundup( T a ) noexcept {
    static_assert( std::is_integral_v<T> && !std::is_same_v<T,bool>,
                   "T must be an integer type" );
    static_assert( N > 0 && !(N & (N-1)), "N must be a positive power of 2" );
    return ( a + (N-1) ) & static_cast<T>(-N);
}

namespace ops {

/**
 * @brief
 * Multiply-add operation: f = (x * y) + z
 * 
 * @note
 * If the `FP_FAST_FMA` macro is defined then the routine will call `std::fma()`
 * which is supposed to implement a (faster) fused multiply-add operation.
 * Otherwise, we just do the normal operation to avoid calling the much slower
 * `fma` operation in `libm`.
 * 
 * @tparam T 
 * @param x     x value
 * @param y     y value
 * @param z     z value
 * @return T 
 */
template < typename T >
constexpr T fma( T const x, T const y, T const z ) noexcept {
    if constexpr ( std::is_same_v<T,float> ) {
#ifdef FP_FAST_FMAF
        return std::fma( x, y, z );
#else
        return (x*y)+z;
#endif
    } else if constexpr ( std::is_same_v<T,double> ) {
#ifdef FP_FAST_FMA
        return std::fma( x, y, z );
#else
        return (x*y)+z;
#endif
    } else {
        return (x*y)+z;
    }
}

}

