# Makefile for GPU MPI version
#
# We assume a CUDA-aware OpenMPI installation and use the compiler wrappers
# to get MPI compilation options

#--------------------- NVIDIA Compilers --------------------
CC = nvcc
CFLAGS = -O3

CXX = nvcc
CXXFLAGS = -O3

#--------------------- MPI Configuration -------------------
# Assumes a CUDA-aware OpenMPI base installation

CFLAGS   += $(shell mpicc  --showme:compile)
CXXFLAGS += $(shell mpic++ --showme:compile)

LDFLAGS  += -forward-unknown-to-host-compiler $(shell mpic++ --showme:link)

# --------------------------------------------------------------------------- #

# Generate code for compute 8.0 architectue
CXXFLAGS += --gpu-architecture=compute_80
CFLAGS += --gpu-architecture=compute_80

# c11 compliance and full warnings
CFLAGS += -Xcompiler -Wall -Xcompiler -std=c11

# c++17 compliance and full warnings (except for VLA extension)
CXXFLAGS += -Xcompiler -Wall -std=c++17

# --------------------------------------------------------------------------- #

#SOURCE = emf.cpp laser.cpp particles.cpp current.cpp \
#         udist.cpp density.cpp species.cpp cathode.cpp

SOURCE = parallel.cpp emf.cpp laser.cpp particles.cpp current.cpp \
         udist.cpp density.cpp species.cpp

SOURCE += main.cpp

# Add ZDF library
SOURCE += zdf.c zdf-parallel.c

TARGET = zpic

# --------------------------------------------------------------------------- #

OBJDIR := .objs
OBJS := $(patsubst %,$(OBJDIR)/%.o,$(basename $(SOURCE)))

# Set automatic dependency generation
DEPDIR := .deps
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d
DEPS := $(patsubst %,$(DEPDIR)/%.d,$(basename $(SOURCE)))

CMSG    := $(shell tput bold && tput setaf 3 && echo "[c]" && tput sgr0)
CPPMSG  := $(shell tput bold && tput setaf 4 && echo "[c++]" && tput sgr0)
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

$(TARGET) : $(OBJS) | $(OBJDIR)
	@echo "$(LINKMSG) $(@F)"
	@$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(OBJS)

$(OBJDIR)/%.o : %.cpp $(DEPDIR)/%.d | $(DEPDIR)
	@echo "$(CPPMSG) $(<F)"
	@$(CXX) $(DEPFLAGS) $(CXXFLAGS) --x cu -c -o $@ $<

# $(OBJDIR)/%.o : %.c
$(OBJDIR)/%.o : %.c $(DEPDIR)/%.d | $(DEPDIR)
	@echo "$(CMSG) $(<F)"
	@$(CC) $(DEPFLAGS) $(CFLAGS) -c -o $@ $<

.PHONY: clean
clean :
	@echo "[clean] Removing build directory"
	@rm -rf $(OBJS)
	@echo "[clean] Removing dependencies directory"
	@rm -rf $(DEPS)
	@echo "[clean] Removing build target"
	@rm -f $(TARGET)

$(OBJDIR): ; @mkdir -p $@

$(DEPDIR): ; @mkdir -p $@

$(DEPS):

include $(wildcard $(DEPS))
