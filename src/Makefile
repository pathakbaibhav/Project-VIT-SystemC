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

# Default target 
TARGET := main
SRC_DIRS := testbench hw
SRCS := $(wildcard $(addsuffix /*.cpp, $(SRC_DIRS)))
OBJS := $(SRCS:.cpp=.o)

# Auto generate dependencies
# See: http://www.microhowto.info/howto/automatically_generate_makefile_dependencies.html
CPPFLAGS += -MMD -MP

# Combine all flags
CXXFLAGS += $(OPENCV_CFLAGS)
LDFLAGS += $(OPENCV_LDFLAGS)

# Default target
all: $(TARGET)

# Link executable 
$(TARGET): $(OBJS)
	$(CXX) $^ $(LDFLAGS) -o $@

# Make weights
weights:
	python3 scripts/weights/readweights_multiple_files.py tmp/weights

# Execute test
test: $(TARGET)
	./$(TARGET) ../eval/n01491361_tiger_shark.JPEG tmp/weights

# Clean everything 
clean:
	rm -rf $(SRCS:%.cpp=%.o) $(SRCS:%.cpp=%.d) $(TARGET) trace.vcd
	# rm -rf tmp/*

.PHONY: clean

# Include generated dependencies (created upon first build)
-include $(SRCS:%.cpp=%.d)
