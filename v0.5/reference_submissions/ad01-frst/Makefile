TARGET_TOOLCHAIN_ROOT := 
TARGET_TOOLCHAIN_PREFIX := 

# These are microcontroller-specific rules for converting the ELF output
# of the linker into a binary image that can be loaded directly.
CXX             := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)clang++'
CC              := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)clang'
AS              := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)as'
AR              := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)ar' 
LD              := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)ld'
NM              := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)nm'
OBJDUMP         := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)objdump'
OBJCOPY         := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)objcopy'
SIZE            := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)size'

RM = rm -f
ARFLAGS := -csr

SRCS += \
main.cc \
submitter_implemented.cc \
api/internally_implemented.cc


OBJS += \
$(patsubst %.cc,%.o,$(SRCS))

CXXFLAGS += -std=c++11 -I.
CCFLAGS +=  -std=c11 -I.

LDFLAGS +=  -lm

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

all: runner

runner : download_deps $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

runner_msan : download_deps $(OBJS)
	$(CXX) $(CXXFLAGS) -fsanitize=memory -o $@ $(OBJS) $(LDFLAGS)

runner_asan : download_deps $(OBJS)
	$(CXX) $(CXXFLAGS) -fsanitize=address -o $@ $(OBJS) $(LDFLAGS)

runner_ubsan : download_deps $(OBJS)
	$(CXX) $(CXXFLAGS) -fsanitize=undefined -o $@ $(OBJS) $(LDFLAGS)

download_deps:
	$(DOWNLOAD_DEPS)

clean:
	-$(RM) $(OBJS)
	-$(RM) runner

clean_downloads:
	$(RM) -r $(DOWNLOADS)
