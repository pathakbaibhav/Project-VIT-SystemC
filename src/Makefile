# if SYSTEMC is installed in non-standard location (and environment var set)
ifneq ($(SYSTEMC),)
	# If SystemC is manually built, must match C++std
	SYSTEMC_LIBS ?= lib64
	SYSTEMC_CSTD ?= ${shell nm $(SYSTEMC)/$(SYSTEMC_LIBS)/libsystemc.so | grep cxx20 | head -n 1 | sed -E 's/.*cxx20([0-9]{2}).*/\1/'}
	SYSTEMC_CSTD ?= 17
	CFLAGS += -I$(SYSTEMC)/include -std=c$(SYSTEMC_CSTD)
	CPPFLAGS += -I$(SYSTEMC)/include -std=c++$(SYSTEMC_CSTD)
	LDFLAGS += -L$(SYSTEMC)/$(SYSTEMC_LIBS) -lsystemc
else
	# ask the system where installed
	LDFLAGS += $(shell pkg-config --libs systemc)
endif

TARGET := main
# SRC_DIRS := hw sw testbench
SRC_DIRS := testbench include hw
SRCS := $(wildcard $(addsuffix /*.cpp, $(SRC_DIRS)))
OBJS := $(SRCS:.cpp=.o)

# auto generate dependenices
#see: http://www.microhowto.info/howto/automatically_generate_makefile_dependencies.html
CPPFLAGS += -MMD -MP

# debug symbols 
CPPFLAGS += -g

# default target 
all: $(TARGET)

# link executable 
$(TARGET): $(OBJS)
	$(CXX) $^ $(LDFLAGS) -o $@

# execute test
test: $(TARGET)
	# python3 scripts/imgLoadAndPreprocess.py ../eval/JPEG tmp/img.csv
	# python3 scripts/weights/readweights_multiple_files.py tmp/weights
	./$(TARGET) tmp/img.csv tmp/weights

# clean everyting 
clean:
	rm -rf $(SRCS:%.cpp=%.o) $(SRCS:%.cpp=%.d) $(TARGET) trace.vcd
	# rm -rf tmp/*

.PHONY: clean

# include generated dependencies (created upon first build)
-include $(SRCS:%.cpp=%.d)