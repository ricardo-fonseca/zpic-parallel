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

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Since we don't use .cu for CUDA files we need to set this manually
set_source_files_properties(src/em2d/emf.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/em2d/laser.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/em2d/particles.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/em2d/current.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/em2d/udist.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/em2d/density.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/em2d/species.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/util.cpp PROPERTIES LANGUAGE CUDA)

# The ZPIC code stays as a separate lib
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

###
# Include build information

# Grab target properties
get_target_property(_zpic_opts zpic_em2d COMPILE_OPTIONS)

set(_zpic_c_flags "")
set(_zpic_cxx_flags "")
set(_zpic_cuda_flags "")
foreach( opt IN LISTS _zpic_opts )
    if(opt MATCHES "\\$<\\$<COMPILE_LANGUAGE:C>:([^>]+)>")
        list(APPEND _zpic_c_flags "${CMAKE_MATCH_1}")
    elseif(opt MATCHES "\\$<\\$<COMPILE_LANGUAGE:CXX>:([^>]+)>")
        list(APPEND _zpic_cxx_flags "${CMAKE_MATCH_1}")
    elseif(opt MATCHES "\\$<\\$<COMPILE_LANGUAGE:CUDA>:([^>]+)>")
        list(APPEND _zpic_cuda_flags "${CMAKE_MATCH_1}")
    endif()
endforeach()

string(REPLACE ";" " " ZPIC_C_FLAGS_STR    "${_zpic_c_flags}")
string(REPLACE ";" " " ZPIC_CXX_FLAGS_STR    "${_zpic_cxx_flags}")
string(REPLACE ";" " " ZPIC_CUDA_FLAGS_STR    "${_zpic_cuda_flags}")

# Get target defines. This may be undefined.
get_target_property(_zpic_defs zpic_em2d COMPILE_DEFINITIONS)
if(NOT _zpic_defs)
    set(_zpic_defs "")
endif()
string(REPLACE ";" " " ZPIC_COMPILE_DEFS_STR    "${_zpic_defs}")

set(ZPIC_BUILD_TYPE            "${CMAKE_BUILD_TYPE}")
set(ZPIC_COMPILER_ID           "${CMAKE_CXX_COMPILER_ID}")
set(ZPIC_COMPILER_VERSION      "${CMAKE_CXX_COMPILER_VERSION}")
set(ZPIC_CUDA_COMPILER_ID      "${CMAKE_CUDA_COMPILER_ID}")
set(ZPIC_CUDA_COMPILER_VERSION "${CMAKE_CUDA_COMPILER_VERSION}")
get_target_property(_zpic_cuda_arch zpic_em2d CUDA_ARCHITECTURES)
set(ZPIC_CUDA_ARCHITECTURES "${_zpic_cuda_arch}")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/build_info.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/build_info.h"
    @ONLY
)

target_include_directories(zpic_em2d PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

##
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
