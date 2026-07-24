cmake_minimum_required(VERSION 3.20)

project(zpic LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Find Python + nanobind ---
if (NOT SKBUILD)
    message(WARNING
        "This CMakeLists.txt is intended to be built via scikit-build-core "
        "(i.e. `pip install .`). Direct cmake invocation may work but is not the "
        "supported path.")
endif()

# Find Python and nanobind
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

# Detect the installed nanobind package and import it into CMake
# Assume nanobind installed via pip
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_DIR)
find_package(nanobind CONFIG REQUIRED)

# The ZPIC code stays as a separete lib
add_library(zpic_em2d STATIC 
  src/em2d/particles.cpp 
  src/em2d/emf.cpp 
  src/em2d/laser.cpp 
  src/em2d/current.cpp 
  src/em2d/udist.cpp 
  src/em2d/density.cpp 
  src/em2d/species.cpp 
  src/em2d/cathode.cpp 
  src/em2d/zdf.c
  src/util.cpp
)
target_include_directories(zpic_em2d PUBLIC src/em2d)
target_compile_options(zpic_em2d PRIVATE -O3)

# ARM Neon optimizations
if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    target_compile_definitions(zpic_em2d PRIVATE USE_NEON)
endif()

# x86_64 AVX2 optimizations
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    target_compile_definitions(zpic_em2d PRIVATE USE_AVX2)
    target_compile_options(zpic_em2d PRIVATE -mavx2 -mfma)
endif()


# Find OpenMP installation - In the future this may be made optional
find_package(OpenMP REQUIRED)
target_link_libraries(zpic_em2d PRIVATE OpenMP::OpenMP_CXX)

set_property(TARGET zpic_em2d PROPERTY POSITION_INDEPENDENT_CODE ON)

get_target_property(opts zpic_em2d COMPILE_OPTIONS)
get_target_property(defs zpic_em2d COMPILE_DEFINITIONS)

# Create file with compile options
file(GENERATE
    OUTPUT  "${CMAKE_CURRENT_BINARY_DIR}/build_info.h"
    CONTENT
"// Auto-generated, do not edit.
#pragma once
#define ZPIC_BUILD_TYPE       \"$<CONFIG>\"
#define ZPIC_COMPILER_ID      \"${CMAKE_CXX_COMPILER_ID}\"
#define ZPIC_COMPILER_VERSION \"${CMAKE_CXX_COMPILER_VERSION}\"
#define ZPIC_CXX_STANDARD     \"$<TARGET_PROPERTY:zpic_em2d,CXX_STANDARD>\"
#define ZPIC_CXX_FLAGS        \"${CMAKE_CXX_FLAGS}\"
#define ZPIC_VERSION          \"${PROJECT_VERSION}\"
#define ZPIC_COMPILE_OPTIONS  \"$<JOIN:${opts}, >\"
#define ZPIC_COMPILE_DEFS     \"${defs}\"
"
    TARGET zpic_em2d
)

target_include_directories(zpic_em2d PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")


# The Python extension module
nanobind_add_module(
  _core                       # extension name -> zpic._core
  STABLE_ABI                  # remove this line if you don't want stable ABI
  NB_STATIC                   # link nanobind statically into the extension
  src/bindings.cpp
)
target_link_libraries(_core PRIVATE zpic_em2d)

# --- Install the extension into the wheel under zpic/ ---
install(TARGETS _core LIBRARY DESTINATION em2d)
