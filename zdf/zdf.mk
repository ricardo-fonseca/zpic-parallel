# --------------------------------------------------------------------------- #
# zdf.mk
#
# Sources and include paths for the ZDF library (C/C++ interface). Include it
# from a project Makefile with:
#
#   ZDF_DIR ?= ../../zdf
#   include $(ZDF_DIR)/zdf.mk
#
# ZDF_DIR is then reset to the directory holding this file, so the including
# Makefile only needs a valid path to reach it.
#
# Sources are listed relative to ZDF_DIR. Each project compiles them into its
# own object directory, with its own flags:
#
#   ZDF_SRC     - serial I/O, no MPI dependency (all codes)
#   ZDF_PAR_SRC - parallel MPI I/O, requires mpi.h (MPI codes only)
#
# A serial code compiles $(ZDF_SRC); an MPI code compiles
# $(ZDF_SRC) $(ZDF_PAR_SRC).
#
# The Fortran interface files (zdf-interface.f03, zdf-par-interface.f03,
# zdf-parallel-f.c) are not needed for C/C++/CUDA codes and are not listed here.
# --------------------------------------------------------------------------- #

# Must be evaluated before any other include, hence :=
ZDF_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))

# Compiled sources (headers are tracked by automatic dependencies)
ZDF_SRC     := zdf.c
ZDF_PAR_SRC := zdf-parallel.c

# Headers are included without a prefix, e.g.
#   #include "zdf.hpp"           (serial)
#   #include "zdf-parallel.hpp"  (MPI, also includes zdf.hpp)
ZDF_INC := -I$(ZDF_DIR)