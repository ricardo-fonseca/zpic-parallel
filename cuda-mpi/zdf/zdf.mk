# --------------------------------------------------------------------------- #
# zdf.mk
#
# Sources and include paths for the ZDF library (serial + MPI parallel I/O,
# C/C++ interface). Include it from a project Makefile with:
#
#   include ../zdf/zdf.mk
#
# ZDF_DIR is set automatically to the directory holding this file, so the
# including Makefile does not need to know where zdf/ lives.
#
# Sources are listed relative to ZDF_DIR. Each project compiles them into its
# own object directory, with its own flags. The Fortran interface files
# (zdf-interface.f03, zdf-par-interface.f03, zdf-parallel-f.c) are not needed
# for C/C++/CUDA codes and are not listed here.
# --------------------------------------------------------------------------- #

# Must be evaluated before any other include, hence :=
ZDF_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))

# Compiled sources (headers are tracked by automatic dependencies)
ZDF_SRC := zdf.c zdf-parallel.c

# Headers are included without a prefix, e.g.
#   #include "zdf-parallel.hpp"
ZDF_INC := -I$(ZDF_DIR)
