#pragma once

#include <string>

namespace coord {
    enum class cart : int { x = 0, y };

    inline constexpr cart x = cart::x;
    inline constexpr cart y = cart::y;

    inline constexpr cart all[] = { cart::x, cart::y };

    inline constexpr std::string name( coord::cart fc ) {
        switch (fc) {
            case coord::cart::x : return "x";
            case coord::cart::y : return "y";
        }
    }
}

namespace edge {
    enum class pos : int { lower = 0, upper };

    inline constexpr pos lower = pos::lower;
    inline constexpr pos upper = pos::upper;

    inline constexpr pos all[] = { pos::lower, pos::upper };

    inline constexpr std::string name( edge::pos p ) {
        switch (p) {
            case edge::pos::lower : return "lower";
            case edge::pos::upper : return "upper";
        }
    }

}

/**
 * @brief Field components (x,y,z)
 * 
 */
namespace fcomp {
    enum class cart : int { x = 0, y, z };

    inline constexpr cart x = cart::x;
    inline constexpr cart y = cart::y;
    inline constexpr cart z = cart::z;

    inline constexpr cart all[] = { cart::x, cart::y, cart:: z };

    inline constexpr std::string name( fcomp::cart fc ) {
        switch (fc) {
            case fcomp::cart::x : return "x";
            case fcomp::cart::y : return "y";
            case fcomp::cart::z : return "z";
        }
    }
}