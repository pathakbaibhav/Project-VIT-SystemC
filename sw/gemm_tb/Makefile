CXXFLAGS += -O3 -std=c++11
TARGET=gemm_tb
SOURCE=gemm_tb.cc
DEBUG_PORT?=10024
QEMU_HOSTNAME?=qemu
QEMU_DEBUG_PORT_FILE:=$(HOME)/.qemu-dbg-port
AUTO_DEBUG_PORT=$(shell [ -f $(QEMU_DEBUG_PORT_FILE) ] && cat $(QEMU_DEBUG_PORT_FILE))
ifneq ($(AUTO_DEBUG_PORT),)
  DEBUG_PORT:=$(AUTO_DEBUG_PORT)
endif

ifeq ($(OECORE_SDK_VERSION),)
  $(error Will not build SW when SDK is not activated)
endif

$(TARGET): $(SOURCE)
		$(CXX) $(CXXFLAGS) -o $@ $^

all: $(TARGET)

upload: $(TARGET)
	scp $^ $(QEMU_HOSTNAME):/home/root

test: upload
	ssh $(QEMU_HOSTNAME) /home/root/$(TARGET)

debug: upload
	@echo "Connect to remote target by:"
	@echo "target remote localhost:$(DEBUG_PORT)"
	ssh $(QEMU_HOSTNAME) "gdbserver localhost:10024 /home/root/$(TARGET)"
	
clean:
	rm -rf *.o $(TARGET)

.PHONY: clean upload run kill
