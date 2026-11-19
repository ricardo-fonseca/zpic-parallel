#pragma once

#include <string>
#include "../core/bounds.hpp"

namespace part {

struct bnd_t {
    enum type { none = 0, periodic, comm };

    /**
     * @brief Name of boundary type
     * 
     * @param t                 Boundary type
     * @return std::string      Name of boundary type
     */
    static std::string name( const type t ) {
        switch( t ) {
            case none:      return "none";
            case periodic:  return "periodic";
            case comm:      return "comm";
        }
        return "-";
    }
};

/**
 * @brief Local boundary type
 * 
 */
using bnd_type = bounds_2d<bnd_t::type>;

}