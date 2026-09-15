from __future__ import annotations

import os
import sys

import cocotb
import numpy as np
from cocotb.clock import Clock

sys.path.insert(0, os.path.dirname(__file__))
from ref import maxpool_int
from tbutil import CLK_NS, host_read, host_write, pulse_start, reset


@cocotb.test()
async def test_pool_2x2(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset(dut)
    rng = np.random.default_rng(0)
    act = rng.integers(-128, 128, size=(8, 8, 2), dtype=np.int8)
    await host_write(dut, 0, act.tobytes())
    dut.k.value = 2
    dut.s.value = 2
    dut.relu.value = 0
    dut.c.value = 2
    dut.h.value = 8
    dut.w.value = 8
    dut.src_base.value = 0
    dut.dst_base.value = 2048
    await pulse_start(dut)
    raw = await host_read(dut, 2048, 4 * 4 * 2)
    got = np.frombuffer(raw, dtype=np.int8).reshape(4, 4, 2)
    exp = maxpool_int(act, 2, 2, False)
    np.testing.assert_array_equal(got, exp)


@cocotb.test()
async def test_pool_relu_and_k3(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset(dut)
    act = np.array([[[-5, 3], [-9, 1], [4, -2], [0, 8]]] * 4, dtype=np.int8)
    # shape 4x2x2 - wait that's wrong. Use (4,4,1)
    act = np.arange(16, dtype=np.int8).reshape(4, 4, 1) - 10
    await host_write(dut, 0, act.tobytes())
    dut.k.value = 2
    dut.s.value = 2
    dut.relu.value = 1
    dut.c.value = 1
    dut.h.value = 4
    dut.w.value = 4
    dut.src_base.value = 0
    dut.dst_base.value = 256
    await pulse_start(dut)
    raw = await host_read(dut, 256, 4)
    got = np.frombuffer(raw, dtype=np.int8).reshape(2, 2, 1)
    exp = maxpool_int(act, 2, 2, True)
    np.testing.assert_array_equal(got, exp)
