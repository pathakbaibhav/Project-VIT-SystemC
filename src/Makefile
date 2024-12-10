# If SYSTEMC is installed in a non-standard location (and environment var set)
ifneq ($(SYSTEMC),)
	# If SystemC is manually built, must match C++std
	SYSTEMC_LIBS ?= lib64
	SYSTEMC_CSTD ?= ${shell nm $(SYSTEMC)/$(SYSTEMC_LIBS)/libsystemc.so | grep cxx20 | head -n 1 | sed -E 's/.*cxx20([0-9]{2}).*/\1/'}
	SYSTEMC_CSTD ?= 17
	CFLAGS += -I$(SYSTEMC)/include -std=c$(SYSTEMC_CSTD)
	CPPFLAGS += -I$(SYSTEMC)/include -std=c++$(SYSTEMC_CSTD)
	LDFLAGS += -L$(SYSTEMC)/$(SYSTEMC_LIBS) -lsystemc
else
	# Ask the system where it's installed
	LDFLAGS += $(shell pkg-config --libs systemc)
endif

# OpenCV setup
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4) # Or opencv depending on version
OPENCV_LDFLAGS := $(shell pkg-config --libs opencv4)

# Eigen library
CXXFLAGS += -I/usr/include/eigen3

# Debug symbols
CPPFLAGS += -g

# Define source directories relative to the Makefile location
SRC_DIRS := hw testbench

# Find all hardware source files
HW_SRCS := $(wildcard hw/*.cpp)

# Find all testbench source files
TESTBENCH_SRCS := $(wildcard testbench/test_*.cpp)

# Find all testbench executables
TESTBENCH_EXECUTABLES := $(patsubst testbench/test_%.cpp, test_%,$(TESTBENCH_SRCS))

# Object files
HW_OBJS := $(HW_SRCS:.cpp=.o)
TESTBENCH_OBJS := $(TESTBENCH_SRCS:.cpp=.o)

# Auto generate dependencies
# See: http://www.microhowto.info/howto/automatically_generate_makefile_dependencies.html
CPPFLAGS += -MMD -MP

# Combine all flags
CXXFLAGS += $(OPENCV_CFLAGS)
LDFLAGS += $(OPENCV_LDFLAGS)

# Define a static library for hardware modules
LIBHW := libhw.a

# Default target
all: $(LIBHW) $(TESTBENCH_EXECUTABLES)

# Rule to build static library
$(LIBHW): $(HW_OBJS)
	ar rcs $@ $^

# Rule to build testbench executables
test_%: testbench/test_%.o $(LIBHW)
	$(CXX) $< $(LIBHW) $(LDFLAGS) -o $@

# Compile hardware object files
hw/%.o: hw/%.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(CPPFLAGS)

# Compile testbench object files
testbench/%.o: testbench/test_%.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(CPPFLAGS)

# Execute all testbench executables
test: $(TESTBENCH_EXECUTABLES)
	# Example: run all testbench executables
	@for exe in $(TESTBENCH_EXECUTABLES); do \
		echo "Running $$exe"; \
		./$$exe; \
	done

# Clean everything 
clean:
	rm -rf hw/*.o testbench/*.o $(LIBHW) $(TESTBENCH_EXECUTABLES) trace.vcd
	# rm -rf tmp/*

.PHONY: all test clean

# Include generated dependencies (created upon first build)
-include hw/*.d testbench/*.d
