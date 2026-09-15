from __future__ import annotations

import os
import sys

import cocotb
import numpy as np
from cocotb.clock import Clock

sys.path.insert(0, os.path.dirname(__file__))
from ref import fc_int
from tbutil import CLK_NS, host_read, host_write, pulse_start, reset


@cocotb.test()
async def test_fc_small(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset(dut)
    rng = np.random.default_rng(1)
    vec = rng.integers(-40, 40, size=(16,), dtype=np.int8)
    wgt = rng.integers(-8, 8, size=(5, 16), dtype=np.int8)
    bias = rng.integers(-20, 20, size=(5,), dtype=np.int32)
    await host_write(dut, 0, vec.tobytes())
    await host_write(dut, 64, np.ascontiguousarray(wgt).tobytes())
    await host_write(dut, 64 + wgt.nbytes, np.ascontiguousarray(bias, dtype="<i4").tobytes())
    dut.relu.value = 1
    dut.c_in.value = 16
    dut.c_out.value = 5
    dut.src_base.value = 0
    dut.wgt_base.value = 64
    dut.bias_base.value = 64 + wgt.nbytes
    dut.dst_base.value = 400
    await pulse_start(dut)
    raw = await host_read(dut, 400, 5)
    got = np.frombuffer(raw, dtype=np.int8)
    exp = fc_int(vec, wgt, bias, True)
    np.testing.assert_array_equal(got, exp)


@cocotb.test()
async def test_fc_sat(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset(dut)
    vec = np.array([127, 127], dtype=np.int8)
    wgt = np.array([[127, 127], [-127, -127]], dtype=np.int8)
    bias = np.array([0, 0], dtype=np.int32)
    await host_write(dut, 0, vec.tobytes())
    await host_write(dut, 16, wgt.tobytes())
    await host_write(dut, 32, bias.astype("<i4").tobytes())
    dut.relu.value = 0
    dut.c_in.value = 2
    dut.c_out.value = 2
    dut.src_base.value = 0
    dut.wgt_base.value = 16
    dut.bias_base.value = 32
    dut.dst_base.value = 48
    await pulse_start(dut)
    raw = await host_read(dut, 48, 2)
    got = np.frombuffer(raw, dtype=np.int8)
    exp = fc_int(vec, wgt, bias, False)
    np.testing.assert_array_equal(got, exp)
