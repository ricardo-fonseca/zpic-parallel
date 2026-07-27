cmake_minimum_required(VERSION 3.20)

project(zpic LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

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

# --- CUDA toolkit (for cudart, cuda_runtime.h, etc.) ---
find_package(CUDAToolkit REQUIRED)

set_source_files_properties(src/em2d/emf.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/em2d/laser.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/em2d/particles.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/em2d/current.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/em2d/udist.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/em2d/density.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/em2d/species.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/util.cpp PROPERTIES LANGUAGE CUDA)

# The ZPIC code stays as a separete lib
add_library(zpic_em2d STATIC 
  src/em2d/emf.cpp 
  src/em2d/laser.cpp 
  src/em2d/particles.cpp 
  src/em2d/current.cpp 
  src/em2d/udist.cpp 
  src/em2d/density.cpp 
  src/em2d/species.cpp 
  src/em2d/zdf.c
  src/util.cpp
)
target_include_directories(zpic_em2d PUBLIC src/em2d)

# Host-side optimization
target_compile_options(zpic_em2d PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:C>:-O3>
)

# Device-side optimization + forward host flags via -Xcompiler
target_compile_options(zpic_em2d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    # Uncomment for extra debug info in device code:
    # $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

set_target_properties(zpic_em2d PROPERTIES
    CUDA_ARCHITECTURES "80-virtual"
)

set_property(TARGET zpic_em2d PROPERTY POSITION_INDEPENDENT_CODE ON)

# Link the CUDA runtime
target_link_libraries(zpic_em2d PRIVATE CUDA::cudart)

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
"
    TARGET zpic_em2d
)

target_include_directories(zpic_em2d PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")


# The Python extension module
set_source_files_properties(src/bindings.cpp PROPERTIES LANGUAGE CUDA)

nanobind_add_module(
  _core                       # extension name -> zpic._core
  STABLE_ABI                  # remove this line if you don't want stable ABI
  NB_STATIC                   # link nanobind statically into the extension
  src/bindings.cpp
)
target_link_libraries(_core PRIVATE zpic_em2d CUDA::cudart)
set_target_properties(_core PROPERTIES
    CUDA_ARCHITECTURES "80-virtual"
)

# --- Install the extension into the wheel under zpic/ ---
install(TARGETS _core LIBRARY DESTINATION em2d)
