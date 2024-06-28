CC = gcc-14
# CC = clang
CFLAGS = -O3 -Wall -std=c11

CXX = g++-14
# CXX = clang++

# Debug
# CXXFLAGS = -O3 -g

# Profile
# CXXFLAGS = -O2 -g -debug inline-debug-info

# Production
CXXFLAGS = -O3 -march=native
# CXXFLAGS = -Ofast

# c++17 compliance and full warnings (except for VLA extension)
CXXFLAGS +=  -Wall -std=c++17 -Wno-unknown-pragmas

# Only needed by intel compilers
# -Wno-vla-extension

# OpenMP support
# CXXFLAGS += -fopenmp

CXXFLAGS += -mavx2 -mfma

SOURCE = particles.cpp emf.cpp laser.cpp current.cpp \
         udist.cpp density.cpp species.cpp cathode.cpp

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

CMSG    := $(shell tput bold && tput setaf 3 && echo "[c]" && tput sgr0)
CPPMSG  := $(shell tput bold && tput setaf 4 && echo "[c++]" && tput sgr0)
LINKMSG := $(shell tput bold && tput setaf 2 && echo "[link]" && tput sgr0)

.PHONY: all
all: $(SOURCE) $(TARGET)

$(TARGET) : $(OBJS) | $(OBJDIR)
	@echo "$(LINKMSG) $(@F)"
	@$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(OBJS)

$(OBJDIR)/%.o : %.cpp $(DEPDIR)/%.d | $(DEPDIR)
	@echo "$(CPPMSG) $(<F)"
	@$(CXX) $(DEPFLAGS) $(CXXFLAGS) -c -o $@ $<

$(OBJDIR)/%.o : %.c $(DEPDIR)/%.d | $(DEPDIR)
	@echo "$(CMSG) $(<F)"
	@$(CC) $(DEPFLAGS) $(CFLAGS) -c -o $@ $<

.PHONY: clean
clean :
	@echo "[clean] Removing build directory"
	@rm -rf $(OBJS)
	@echo "[clean] Removing dependencies directory"
	@rm -rf $(DEPS)
	@rm -f $(TARGET)

$(OBJDIR): ; @mkdir -p $@

$(DEPDIR): ; @mkdir -p $@

$(DEPS):

include $(wildcard $(DEPS))
