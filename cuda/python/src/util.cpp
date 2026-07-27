#include "em2d/zpic.h"
#include "build_info.h"


/**
 * @brief Output system information
 * 
 */
void util_sys_info() {
    std::cout << ansi::bold;
    std::cout << "System information\n";
    std::cout << ansi::reset;
    print_gpu_info();
}

/**
 * @brief Output build information
 * 
 */
void util_build_info() {
    std::cout << ansi::bold;
    std::cout << "Build options\n";
    std::cout << ansi::reset;

    std::cout
       << "Build type      : " << ZPIC_BUILD_TYPE << '\n'
       << "Main compiler   : " << ZPIC_COMPILER_ID
                               << " " << ZPIC_COMPILER_VERSION << '\n'
       << "C flags         : " << ZPIC_C_FLAGS << '\n'
       << "C++ flags       : " << ZPIC_CXX_FLAGS << '\n'
       << "CUDA compiler.  : " << ZPIC_CUDA_COMPILER_ID
                               << " " << ZPIC_CUDA_COMPILER_VERSION << '\n'
       << "CUDA flags      : " << ZPIC_CUDA_FLAGS << '\n'
       << "CUDA archs.     : " << ZPIC_CUDA_ARCHITECTURES << '\n';
}