# Makefile for cpu MPI version
#
# This makefile uses compiler wrappers for enabling MPI support
#

#--------------------- GCC Compilers --------------------
# CC = mpigcc
# CFLAGS = -O3 -march=native

# CXX = mpigxx
# CXXFLAGS = -O3 -march=native

# OpenMP support
# CXXFLAGS += -fopenmp

#--------------------- Clang Compilers --------------------
CC = mpicc
CFLAGS = -O3 -march=native

CXX = mpic++
CXXFLAGS = -O3 -march=native

# OpenMP support
CXXFLAGS += -fopenmp

#------------------ Intel oneApi Compilers -----------------
# CC = mpiicx
# CFLAGS = -O3

# CXX = mpiicpx
# CXXFLAGS = -O3

# OpenMP support
# CXXFLAGS += -qopenmp

# --------------------------------------------------------------------------- #

# Enable AVX2 extensions with FMA support
# CXXFLAGS += -DUSE_AVX2 -mavx2 -mfma

# Enable AVX512f extensions
# CXXFLAGS += -DUSE_AVX512 -mavx512f

# Enable ARM Neon extensions
CXXFLAGS += -DUSE_NEON

# Enable ARM SVE extensions (512 bits)
# CXXFLAGS += -DUSE_SVE -msve-vector-bits=512

# c11 compliance and full warnings
CFLAGS += -Wall -std=c11

# c++17 compliance and full warnings (except for VLA extension)
CXXFLAGS += -Wall -std=c++20 -Wno-unknown-pragmas 

# Disable VLA extension warnings (clang compilers)
# CXXFLAGS += -Wno-vla-cxx-extension
CXXFLAGS += -Wno-vla-extension

# Debug with sanitize=address (debugging only!)
# CXXFLAGS += -fsanitize=address,undefined -fno-omit-frame-pointer -g -Og

# In case of errors, output each process log to a different file using
# ASAN_OPTIONS=log_path=asan.log mpirun -np N ./zpic

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

SOURCE = emf.cpp laser.cpp current.cpp \
         udist.cpp density.cpp species.cpp

SOURCE += main.cpp

# Support classes
SOURCE += part/particles.cpp

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
	@echo "$(LINKMSG)$(@F)"
	@$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(OBJS)

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
