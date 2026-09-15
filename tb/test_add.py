from __future__ import annotations

import os
import sys

import cocotb
import numpy as np
from cocotb.clock import Clock

sys.path.insert(0, os.path.dirname(__file__))
from ref import add_int
from tbutil import CLK_NS, host_read, host_write, pulse_start, reset


@cocotb.test()
async def test_add(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset(dut)
    rng = np.random.default_rng(2)
    a = rng.integers(-100, 100, size=(4, 4, 3), dtype=np.int8)
    b = rng.integers(-100, 100, size=(4, 4, 3), dtype=np.int8)
    await host_write(dut, 0, a.tobytes())
    await host_write(dut, 128, b.tobytes())
    dut.relu.value = 1
    dut.c.value = 3
    dut.h.value = 4
    dut.w.value = 4
    dut.src0_base.value = 0
    dut.src1_base.value = 128
    dut.dst_base.value = 256
    await pulse_start(dut)
    raw = await host_read(dut, 256, a.size)
    got = np.frombuffer(raw, dtype=np.int8).reshape(4, 4, 3)
    exp = add_int(a, b, True)
    np.testing.assert_array_equal(got, exp)


@cocotb.test()
async def test_add_no_relu_sat(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset(dut)
    a = np.full((2, 2, 1), 100, dtype=np.int8)
    b = np.full((2, 2, 1), 100, dtype=np.int8)
    await host_write(dut, 0, a.tobytes())
    await host_write(dut, 16, b.tobytes())
    dut.relu.value = 0
    dut.c.value = 1
    dut.h.value = 2
    dut.w.value = 2
    dut.src0_base.value = 0
    dut.src1_base.value = 16
    dut.dst_base.value = 32
    await pulse_start(dut)
    raw = await host_read(dut, 32, 4)
    got = np.frombuffer(raw, dtype=np.int8)
    assert np.all(got == 127)
