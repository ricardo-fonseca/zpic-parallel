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

###
# Include build information

# Grab target properties
get_target_property(_zpic_opts zpic_em2d COMPILE_OPTIONS)
if(NOT _zpic_opts)
    set(_zpic_opts "")
endif()
string(REPLACE ";" " " ZPIC_COMPILE_OPTIONS    "${_zpic_opts}")

# Get target defines. This may be undefined.
get_target_property(_zpic_defs zpic_em2d COMPILE_DEFINITIONS)
if(NOT _zpic_defs)
    set(_zpic_defs "")
endif()
string(REPLACE ";" " " ZPIC_COMPILE_DEFS    "${_zpic_defs}")

set(ZPIC_BUILD_TYPE            "${CMAKE_BUILD_TYPE}")
set(ZPIC_COMPILER_ID           "${CMAKE_CXX_COMPILER_ID}")
set(ZPIC_COMPILER_VERSION      "${CMAKE_CXX_COMPILER_VERSION}")

get_target_property(_zpic_cxx_std zpic_em2d CXX_STANDARD)
set(ZPIC_CXX_STANDARD          "${_zpic_cxx_std}")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/build_info.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/build_info.h"
    @ONLY
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
