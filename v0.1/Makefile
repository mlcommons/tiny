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

VWW_SRCS := $(SRCS) \
vww/vww_model_data.cc \
vww/vww_model_settings.cc \
vww/vww_inputs.cc 

VWW_OBJS += \
$(patsubst %.cc,%.o,$(VWW_SRCS))

OBJS += $(VWW_OBJS)

CXXFLAGS += -std=c++11 -I.
CCFLAGS +=  -std=c11 -I.

LDFLAGS +=  -lm

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

all: vww

vww: vww_perf_runner

vww_perf_runner : download_deps $(VWW_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(VWW_OBJS) $(LDFLAGS)

download_deps:
	$(DOWNLOAD_DEPS)

clean:
	-$(RM) $(OBJS)
	-$(RM) vww_perf_runner

clean_downloads:
	$(RM) -r $(DOWNLOADS)
