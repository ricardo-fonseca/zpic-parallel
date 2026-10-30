#pragma once

#include <string_view>
#include <cstdint>

namespace hash {

constexpr uint32_t fnv1a_32(std::string_view str) {
    uint32_t hash = 2166136261u;
    for (char c : str) {
        hash ^= static_cast<uint32_t>(c);
        hash *= 16777619u;
    }
    return hash;
}

}

constexpr uint32_t operator""_hash(const char* str, std::size_t len) {
    return hash::fnv1a_32(std::string_view(str, len));
}
