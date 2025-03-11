# ZPIC MPI / OpenMP / SIMD

This version implements an MPI/OpenMP/SIMD optimized of the ZPIC algorithm.

__WARNING : This version is not yet operational__ 

## MPI

The MPI parallelization is done using a coarse spatial domain decomposition across parallel nodes, while using a fine domain decomposition (i.e. micro-spatial decomposition, see below) inside each parallel node. It is fully compatible with the OpenMP and SIMD optimizations.

For this version, MPI support is mandatory. If you don't have (or don't want) MPI support, you should use the OpenMP / SIMD version instead.

### Compiling with MPI

The supplied `Makefile` uses MPI wrapper compilers for MPI support. The code was tested with the Intel OneAPI toolkit and with MPICH from `brew`.

## OpenMP

The OpenMP implements (exactly) the same type of micro-spatial domain decomposition (tiling) used in the GPU versions, assigning 1 OpenMP thread per tile. For the routines where the load per tile may vary (namely because the number of particles per tile can be very different) thread assignment is done dynamically.

Also, while explicitly defining a "fast/shared" memory is not possible, grid quantities (E,B,J) are copied from global memory to local arrays (defined as variable lenght arrays), and all interpolation/current deposit is done on these arrays.

Note that OpenMP support is not mandatory; this code will compile and run without problems without OpenMP support.

### Enabling OpenMP support

OpenMP support is enabled by setting the appropriate compiler flag, namely:

+ `-fopenmp` for __GCC__ and __CLANG__ compilers
+ `-qopenmp` for __Intel oneApi__ compilers (2024)

For other compilers you will need to check your documentation.

## SIMD optimiztions

The code also allows to (optionally) use different versions of the particle advance that have been optimized for single-instruction-multiple-data (SIMD) architectures, such as the ones available on modern x86 and ARM cpus.

When using these versions, in each tile particles will be processed in sets `nV` particles, where `nV` correponds to the SIMD vector width (e.g. for x86 AVX2 we have 8 wide float vectors). Any remaining particles will be processed using the standard serial code.

Again, just as it was the case for `OpenMP`, SIMD support is not required and the code can be compiled without it.

### Enabling SIMD support 

To enable SIMD support you must i) define an appropriate compile time macro specifying the required architecture, and ii) add the appropriate compiler flags to support the instruction set.

Currently suported architectures:

+ __x86 AVX2__
    + Compile with `-DUSE_AVX2 -mavx2 -mfma`
+ __x86 AVX512__
    + Compile with `-DUSE_AVX512 -mavx512f`

ARM Neon and SVE(2) support is currently under development and will be published soon.
