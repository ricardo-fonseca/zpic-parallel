# Makefile for cpu OpenMP version
#
# 

#--------------------- GCC Compilers --------------------
CC = gcc-15
CFLAGS = -O3 -march=native

CXX = g++-15
CXXFLAGS = -O3 -march=native

# OpenMP support
CXXFLAGS += -fopenmp

#--------------------- Clang Compilers --------------------
# CC = clang
# CFLAGS = -O3 -march=native

# CXX = clang++
# CXXFLAGS = -O3 -march=native

# OpenMP support
# CXXFLAGS += -fopenmp

#------------------ Intel oneApi Compilers -----------------
# CC = icx
# CFLAGS = -O3

# CXX = icpx
# CXXFLAGS = -O3

# OpenMP support
# CXXFLAGS += -qopenmp

#--------------------- FFT Library --------------------

# Add FFTW library (single precision)

CXXFLAGS += -I/opt/homebrew/include
LDFLAGS = -L/opt/homebrew/lib -lfftw3f_omp -lfftw3f

# LDFLAGS = -lfftw3f_omp -lfftw3f -lm

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
CXXFLAGS += -Wall -std=c++17 -Wno-unknown-pragmas 

# Disable VLA extension warnings (clang compilers)
# CXXFLAGS += -Wno-vla-cxx-extension
# CXXFLAGS += -Wno-vla-extension

# --------------------------------------------------------------------------- #

SOURCE = particles.cpp emf.cpp laser.cpp current.cpp charge.cpp \
         udist.cpp density.cpp species.cpp

SOURCE += main.cpp

# Add ZDF library
SOURCE += zdf.c

TARGET = zpic

# --------------------------------------------------------------------------- #

OBJDIR := .objs
OBJS := $(patsubst %,$(OBJDIR)/%.o,$(basename $(SOURCE)))

# Set automatic dependency generation
DEPDIR := .deps
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d
DEPS := $(patsubst %,$(DEPDIR)/%.d,$(basename $(SOURCE)))

CMSG    := $(shell tput bold && tput setaf 3 && echo "[c]   " && tput sgr0)
CPPMSG  := $(shell tput bold && tput setaf 4 && echo "[c++] " && tput sgr0)
LINKMSG := $(shell tput bold && tput setaf 2 && echo "[link]" && tput sgr0)


.PHONY: dirs
.PHONE: info

.PHONY: all
all: dirs $(SOURCE) $(TARGET)

dirs: $(OBJDIR) $(DEPDIR)

info:
	@ tput bold && echo "Compile options" && tput sgr0
	@ echo "[c]     $(CC) $(CFLAGS)"
	@ echo "[c++]   $(CXX) $(CXXFLAGS)"
	@ echo "[ld]    $(CXX) $(CXXFLAGS) $(LDFLAGS)"

$(TARGET) : $(OBJS)
	@echo "$(LINKMSG)$(@F)"
	@$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o : %.cpp $(DEPDIR)/%.d | $(DEPDIR)
	@echo "$(CPPMSG)$(<F)"
	@$(CXX) $(DEPFLAGS) $(CXXFLAGS) -c -o $@ $<

$(OBJDIR)/%.o : %.c $(DEPDIR)/%.d | $(DEPDIR)
	@echo "$(CMSG)$(<F)"
	@$(CC) $(DEPFLAGS) $(CFLAGS) -c -o $@ $<

.PHONY: clean
clean :
	@echo "[clean] Removing build directory"
	@rm -rf $(OBJS)
	@echo "[clean] Removing dependencies directory"
	@rm -rf $(DEPS)
	@echo "[clean] Removing build target"
	@rm -f $(TARGET)

$(OBJDIR): ; @ test -d $@ || mkdir -p $@

$(DEPDIR): ; @ test -d $@ || mkdir -p $@

$(DEPS):

include $(wildcard $(DEPS))
