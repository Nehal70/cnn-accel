SIM ?= verilator
TOPLEVEL_LANG = verilog
ENGINE ?= conv

PWD := $(shell pwd)
RTL := $(PWD)/../rtl

VERILATOR_ARGS += --trace --timescale 1ns/1ns -Wno-fatal
EXTRA_ARGS += --trace --timescale 1ns/1ns -Wno-fatal

ifeq ($(ENGINE),conv)
  VERILOG_SOURCES := $(RTL)/cnn_pkg.sv $(RTL)/conv_engine.sv $(RTL)/conv_tb_top.sv
  TOPLEVEL := conv_tb_top
  MODULE := test_conv
endif
ifeq ($(ENGINE),pool)
  VERILOG_SOURCES := $(RTL)/cnn_pkg.sv $(RTL)/pool_engine.sv $(RTL)/pool_tb_top.sv
  TOPLEVEL := pool_tb_top
  MODULE := test_pool
endif
ifeq ($(ENGINE),fc)
  VERILOG_SOURCES := $(RTL)/cnn_pkg.sv $(RTL)/fc_engine.sv $(RTL)/fc_tb_top.sv
  TOPLEVEL := fc_tb_top
  MODULE := test_fc
endif
ifeq ($(ENGINE),add)
  VERILOG_SOURCES := $(RTL)/cnn_pkg.sv $(RTL)/eltwise.sv $(RTL)/add_tb_top.sv
  TOPLEVEL := add_tb_top
  MODULE := test_add
endif
ifeq ($(ENGINE),chip)
  VERILOG_SOURCES := \
	$(RTL)/cnn_pkg.sv \
	$(RTL)/conv_engine.sv \
	$(RTL)/pool_engine.sv \
	$(RTL)/fc_engine.sv \
	$(RTL)/eltwise.sv \
	$(RTL)/dma.sv \
	$(RTL)/netparser.sv \
	$(RTL)/cnn_accel_top.sv \
	$(RTL)/chip_tb_top.sv
  TOPLEVEL := chip_tb_top
  MODULE := test_chip
endif

COCOTB_TOPLEVEL := $(TOPLEVEL)
COCOTB_TEST_MODULES := $(MODULE)

include $(shell cocotb-config --makefiles)/Makefile.sim
