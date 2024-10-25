
all: hw sw

sw: sw/vector_processor_tb sw/gemm_tb 

hw:
ifneq ($(OECORE_SDK_VERSION),)
	$(info Will not build SystemC HW while SDK is activated)
else
	$(MAKE) -C $@ 
endif

# we don't have a middle level makefile so just call 
# directly into the SW dirs
sw/vector_processor_tb sw/gemm_tb:
ifeq ($(OECORE_SDK_VERSION),)
	$(info Will not build SW when SDK is not activated)
else
	$(MAKE) -C $@
endif

# convenience targets pointing into hw
launch_cosim launch_cosim_debug testbench zynq_demo:
	$(MAKE) -C hw $@

# clean whatever we can based on environment
clean: 
ifeq ($(OECORE_SDK_VERSION),)
	$(MAKE) -C hw $@
else
	$(MAKE) -C sw/vector_processor_tb $@
	$(MAKE) -C sw/gemm_tb $@
endif

.PHONY: hw sw sw/vector_processor_tb sw/gemm_tb 
