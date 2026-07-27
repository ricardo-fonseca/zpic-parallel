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
    zpic::sys_info();
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
       << "Compiler        : " << ZPIC_COMPILER_ID
                            << " " << ZPIC_COMPILER_VERSION << "\n"
       << "Options         : " << ZPIC_COMPILE_OPTIONS << "\n"
       << "Defines         : " << ZPIC_COMPILE_DEFS << "\n";
}