#pragma once

#include <ostream>

/**
 * @brief Boundary information (lower upper)
 * 
 * @tparam T 
 */
template<typename T>
class bounds {
public:
    T lower;
    T upper;

    bounds() : lower(T{}), upper(T{}) {}
    bounds(T val) : lower(val), upper(val) {}
    bounds(T lower, T upper) : lower(lower), upper(upper) {}

    friend bool operator==(const bounds&, const bounds&) = default;

    friend std::ostream& operator<<(std::ostream& os, const bounds& obj) {
        return os << '(' << obj.lower << ", " << obj.upper << ')';
    }
};


/**
 * @brief 2D boundary information x (lower|upper) and y (lower|upper)
 * 
 * @tparam T 
 */
template<typename T>
class bounds_2d {
public:
    bounds<T> x;
    bounds<T> y;

    bounds_2d() = default;
    bounds_2d(T val) : x( val ), y( val ) {}
    bounds_2d(bounds<T> x, bounds<T> y) : x(x), y(y) {}

    friend bool operator==(const bounds_2d&, const bounds_2d&) = default;

    friend std::ostream& operator<<(std::ostream& os, const bounds_2d& obj) {
        return os
           << "{ x:" << obj.x
           << ", y:" << obj.y
           << " }";
    }
};
