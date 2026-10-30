# --------------------------------------------------------------------------- #
# common.mk
#
# Sources and include paths shared by cuda-mpi codes. Include it from a project
# Makefile with:
#
#   include ../common/common.mk
#
# COMMON_DIR is set automatically to the directory holding this file, so the
# including Makefile does not need to know where common/ lives.
#
# Sources are listed relative to COMMON_DIR. Each project compiles them into
# its own object directory, with its own flags.
# --------------------------------------------------------------------------- #

# Must be evaluated before any other include, hence :=
COMMON_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))

# Shared compiled sources (headers are tracked by automatic dependencies)
COMMON_SRC := part/particles.cu part/sort.cu

# Headers are included with their subdirectory prefix, e.g.
#   #include "grid/grid.cuh"
COMMON_INC := -I$(COMMON_DIR)
