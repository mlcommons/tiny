TARGET_TOOLCHAIN_ROOT := 
TARGET_TOOLCHAIN_PREFIX := 

# These are microcontroller-specific rules for converting the ELF output
# of the linker into a binary image that can be loaded directly.
CXX             := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)g++'
CC              := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)gcc'
AS              := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)as'
AR              := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)ar' 
LD              := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)ld'
NM              := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)nm'
OBJDUMP         := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)objdump'
OBJCOPY         := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)objcopy'
SIZE            := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)size'

RM = rm -f
ARFLAGS := -csr

include example_submission/Makefile.inc

SRCS += \
device_harness/main.cc

OBJS := \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(SRCS)))

CXXFLAGS += -std=c++11
CCFLAGS +=  -std=c11

LDFLAGS +=  -lm

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

tinyml_perf_runner : $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

all: tinyml_perf_runner

clean:
	-$(RM) $(OBJS)
	-$(RM) tinyml_perf_runner
