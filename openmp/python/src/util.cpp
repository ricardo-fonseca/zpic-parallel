#include "em2d/zpic.h"
#include "build_info.h"



void util_sys_info() {
    zpic::sys_info();
}

void util_build_info() {
    std::cout << "zpic "    << ZPIC_VERSION << "\n"
       << "  build type : " << ZPIC_BUILD_TYPE << "\n"
       << "  compiler   : " << ZPIC_COMPILER_ID
                            << " " << ZPIC_COMPILER_VERSION << "\n"
       << "  C++ std    : " << ZPIC_CXX_STANDARD << "\n"
       << "  CXX flags  : " << ZPIC_CXX_FLAGS << "\n"
       << "  options    : " << ZPIC_COMPILE_OPTIONS << "\n"
       << "  defines    : " << ZPIC_COMPILE_DEFS << "\n";
}