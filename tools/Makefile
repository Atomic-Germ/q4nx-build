# Build the LD_PRELOAD shim that captures FastFlowLM's DPU instruction
# buffers (see header of ff_instr_dump.c for usage).
#
#   make            -> ff_instr_dump.so
#   make PREFIX=/opt/xilinx/xrt   (only needed if XRT headers are required)

CC     ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -shared -fPIC
LDLIBS += -ldl

ff_instr_dump.so: ff_instr_dump.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f ff_instr_dump.so

.PHONY: clean
