#pragma once

#include "core/bounds.hpp"

namespace cyl_part {

    struct bnd_t {
    enum type { none = 0, axial, periodic, comm };
};

/**
 * @brief Local boundary type
 * 
 */
using bnd_type = bounds_2d<bnd_t::type>;

}