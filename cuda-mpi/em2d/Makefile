# Makefile for GPU MPI version
#
# We assume a CUDA-aware MPICH or OpenMPI installation and use the compiler wrappers
# to get MPI compilation options

#--------------------- MPICH + NVIDIA Compilers --------------------

# CUDA-aware MPICH base installation

# CC = mpicc -cc=nvcc
# CFLAGS = -O3

# CXX = mpic++ -CC=nvcc
# CXXFLAGS = -O3

# LDFLAGS  = -forward-unknown-to-host-compiler

#--------------------- OpenMPI + NVIDIA Compilers --------------------

# CUDA-aware OpenMPI base installation
# Also works for NVIDIA HPC-X

CC = nvcc
CFLAGS = -O3
CXX = nvcc
CXXFLAGS = -O3

CFLAGS   += -forward-unknown-to-host-compiler $(shell mpicc  --showme:compile)
CXXFLAGS += -forward-unknown-to-host-compiler $(shell mpic++ --showme:compile)
LDFLAGS  += -forward-unknown-to-host-compiler $(shell mpic++ --showme:link)

# --------------------------------------------------------------------------- #

# Generate code for compute 8.0 architectue
CXXFLAGS += --gpu-architecture=compute_80
CFLAGS += --gpu-architecture=compute_80

# c11 compliance and full warnings
CFLAGS += -Xcompiler -Wall -Xcompiler -std=c11

# c++17 compliance and full warnings (except for VLA extension)
CXXFLAGS += -Xcompiler -Wall -std=c++20 -Wno-unknown-pragmas -Wsign-compare

# Debug
# CXXFLAGS += -Xcompiler "-fno-omit-frame-pointer -g -Og"

# --------------------------------------------------------------------------- #
# Shared code locations

# Both locations can be overridden from the command line, e.g.
#   make ZDF_DIR=/path/to/zdf

# Shared classes (core, grid, parallel, part, util)
COMMON_DIR ?= ../common
include $(COMMON_DIR)/common.mk

# ZDF library (separate project, alongside common/)
ZDF_DIR ?= ../zdf
include $(ZDF_DIR)/zdf.mk

# Include search order: this project first (so shared code can pick up
# project-level headers such as zpic.hpp), then common/, then ZDF
INCLUDES := -I. $(COMMON_INC) $(ZDF_INC)

CXXFLAGS += $(INCLUDES)
CFLAGS   += $(INCLUDES)

# --------------------------------------------------------------------------- #

SOURCE = emf.cu laser.cu current.cu udist.cu density.cu species.cu

#current.cu \
#         udist.cu density.cu species.cu
# cathode.cu

SOURCE += main.cu

TARGET = zpic

# --------------------------------------------------------------------------- #

OBJDIR := .objs
DEPDIR := .deps

# Objects for each source tree go in their own subdirectory:
#   local   -> $(OBJDIR)/emf.o
#   common/ -> $(OBJDIR)/common/part/particles.o
#   zdf/    -> $(OBJDIR)/zdf/zdf.o
OBJS := $(patsubst %,$(OBJDIR)/%.o,$(basename $(SOURCE))) \
        $(patsubst %,$(OBJDIR)/common/%.o,$(basename $(COMMON_SRC))) \
        $(patsubst %,$(OBJDIR)/zdf/%.o,$(basename $(ZDF_SRC)))

# Set automatic dependency generation
# The dependency file mirrors the object path, e.g.
#   .objs/common/part/sort.o -> .deps/common/part/sort.d
DEPFILE  = $(patsubst $(OBJDIR)/%.o,$(DEPDIR)/%.d,$@)
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPFILE)
DEPS := $(patsubst $(OBJDIR)/%.o,$(DEPDIR)/%.d,$(OBJS))

CMSG    := $(shell tput bold && tput setaf 3 && echo "[c]" && tput sgr0)
CPPMSG  := $(shell tput bold && tput setaf 4 && echo "[c++]" && tput sgr0)
CUDAMSG := $(shell tput bold && tput setaf 5 && echo "[cuda]" && tput sgr0)
LINKMSG := $(shell tput bold && tput setaf 2 && echo "[link]" && tput sgr0)

.PHONY: all
all: $(TARGET)

.PHONY: info
info:
	@ tput bold && echo "Compile options" && tput sgr0
	@ echo "[c]     $(CC) $(CFLAGS)"
	@ echo "[c++]   $(CXX) $(CXXFLAGS)"
	@ echo "[ld]    $(CXX) $(CXXFLAGS) $(LDFLAGS)"
	@ tput bold && echo "Shared code" && tput sgr0
	@ echo "common: $(COMMON_DIR)"
	@ echo "zdf:    $(ZDF_DIR)"

$(TARGET) : $(OBJS)
	@echo "$(LINKMSG) $(@F)"
	@$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

# --------------------------------------------------------------------------- #
# Compilation rules
#
# The same set of rules is generated for each source tree:
#   $(1) - source directory prefix (empty for the local project)
#   $(2) - object / dependency subdirectory prefix
#
# Object and dependency subdirectories are created as needed.

define compile_rules

$(OBJDIR)/$(2)%.o : $(1)%.cpp $(DEPDIR)/$(2)%.d
	@echo "$$(CPPMSG) $$(<F)"
	@mkdir -p $$(@D) $$(dir $$(DEPFILE))
	@$$(CXX) $$(DEPFLAGS) $$(CXXFLAGS) --x cu -c -o $$@ $$<

$(OBJDIR)/$(2)%.o : $(1)%.cu $(DEPDIR)/$(2)%.d
	@echo "$$(CUDAMSG) $$(<F)"
	@mkdir -p $$(@D) $$(dir $$(DEPFILE))
	@$$(CXX) $$(DEPFLAGS) $$(CXXFLAGS) -c -o $$@ $$<

$(OBJDIR)/$(2)%.o : $(1)%.c $(DEPDIR)/$(2)%.d
	@echo "$$(CMSG) $$(<F)"
	@mkdir -p $$(@D) $$(dir $$(DEPFILE))
	@$$(CC) $$(DEPFLAGS) $$(CFLAGS) -c -o $$@ $$<

endef

$(eval $(call compile_rules,,))
$(eval $(call compile_rules,$(COMMON_DIR)/,common/))
$(eval $(call compile_rules,$(ZDF_DIR)/,zdf/))

# --------------------------------------------------------------------------- #

.PHONY: clean
clean :
	@echo "[clean] Removing build directory"
	@rm -rf $(OBJDIR)
	@echo "[clean] Removing dependencies directory"
	@rm -rf $(DEPDIR)
	@echo "[clean] Removing build target"
	@rm -f $(TARGET)

$(DEPS):

include $(wildcard $(DEPS))
