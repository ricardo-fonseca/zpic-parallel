CC = hipcc
CFLAGS = -O3


CXX = hipcc
CXXFLAGS = -O3

# c++17 compliance
CXXFLAGS += -std=c++17

# For NIVIDIA A100
CXXFLAGS += --gpu-architecture=compute_80

#SOURCE = emf.cpp laser.cpp particles.cpp current.cpp \
#         udist.cpp density.cpp species.cpp cathode.cpp

SOURCE = emf.cpp laser.cpp particles.cpp current.cpp \
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
