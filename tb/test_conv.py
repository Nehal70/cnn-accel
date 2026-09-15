from __future__ import annotations

import os
import sys

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, os.path.dirname(__file__))
from ref import conv2d_int

CLK_NS = 10


async def reset(dut):
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.host_we.value = 0
    dut.k.value = 0
    dut.s.value = 0
    dut.pad.value = 0
    dut.relu.value = 0
    dut.c_in.value = 0
    dut.c_out.value = 0
    dut.h.value = 0
    dut.w.value = 0
    dut.dst_base.value = 0
    dut.src_base.value = 0
    dut.wgt_base.value = 0
    dut.bias_base.value = 0
    dut.host_addr.value = 0
    dut.host_wdata.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def host_write(dut, addr: int, data: bytes):
    for i, b in enumerate(data):
        dut.host_addr.value = addr + i
        dut.host_wdata.value = int(b) & 0xFF
        dut.host_we.value = 1
        await RisingEdge(dut.clk)
    dut.host_we.value = 0
    await RisingEdge(dut.clk)


async def host_read(dut, addr: int, n: int) -> bytes:
    out = bytearray(n)
    for i in range(n):
        dut.host_addr.value = addr + i
        await RisingEdge(dut.clk)
        out[i] = int(dut.host_rdata.value) & 0xFF
    return bytes(out)


async def run_conv(dut, act, wgt, bias, stride, pad, relu, src=0, wbase=None, bbase=None, dst=None):
    h, w, cin = act.shape
    cout, _, k, _ = wgt.shape
    if wbase is None:
        wbase = src + act.nbytes
    if bbase is None:
        bbase = wbase + wgt.nbytes
    if dst is None:
        dst = bbase + bias.nbytes

    await host_write(dut, src, act.tobytes(order="C"))
    await host_write(dut, wbase, np.ascontiguousarray(wgt).tobytes(order="C"))
    await host_write(dut, bbase, np.ascontiguousarray(bias, dtype="<i4").tobytes())

    dut.k.value = k
    dut.s.value = stride
    dut.pad.value = pad
    dut.relu.value = int(relu)
    dut.c_in.value = cin
    dut.c_out.value = cout
    dut.h.value = h
    dut.w.value = w
    dut.src_base.value = src
    dut.wgt_base.value = wbase
    dut.bias_base.value = bbase
    dut.dst_base.value = dst

    await RisingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    cycles = 0
    timeout = 2_000_000
    while int(dut.done.value) == 0:
        await RisingEdge(dut.clk)
        cycles += 1
        if cycles > timeout:
            raise TimeoutError("conv did not finish")
    assert int(dut.error.value) == 0, "conv engine error"

    ho = (h + 2 * pad - k) // stride + 1
    wo = (w + 2 * pad - k) // stride + 1
    raw = await host_read(dut, dst, ho * wo * cout)
    got = np.frombuffer(raw, dtype=np.int8).reshape(ho, wo, cout)
    exp = conv2d_int(act, wgt, bias, stride, pad, relu)
    np.testing.assert_array_equal(got, exp)
    return got, cycles


def rng_i8(rng, shape):
    return rng.integers(-128, 128, size=shape, dtype=np.int8)


@cocotb.test()
async def test_tiny_3x3_pad1(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset(dut)
    rng = np.random.default_rng(0)
    act = rng_i8(rng, (8, 8, 1))
    wgt = rng_i8(rng, (8, 1, 3, 3))
    bias = rng.integers(-32, 32, size=(8,), dtype=np.int32)
    await run_conv(dut, act, wgt, bias, stride=1, pad=1, relu=True)


@cocotb.test()
async def test_valid_7x7_to_5x5(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset(dut)
    rng = np.random.default_rng(1)
    act = rng_i8(rng, (7, 7, 2))
    wgt = rng_i8(rng, (3, 2, 3, 3))
    bias = rng.integers(-1000, 1000, size=(3,), dtype=np.int32)
    await run_conv(dut, act, wgt, bias, stride=1, pad=0, relu=False)


@cocotb.test()
async def test_stride2(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset(dut)
    rng = np.random.default_rng(2)
    act = rng_i8(rng, (8, 8, 2))
    wgt = rng_i8(rng, (4, 2, 3, 3))
    bias = np.zeros(4, dtype=np.int32)
    await run_conv(dut, act, wgt, bias, stride=2, pad=0, relu=True)


@cocotb.test()
async def test_pointwise_1x1(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset(dut)
    rng = np.random.default_rng(3)
    act = rng_i8(rng, (4, 4, 3))
    wgt = rng_i8(rng, (5, 3, 1, 1))
    bias = rng.integers(-50, 50, size=(5,), dtype=np.int32)
    await run_conv(dut, act, wgt, bias, stride=1, pad=0, relu=False)


@cocotb.test()
async def test_relu_zeros_negatives(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset(dut)
    act = np.full((3, 3, 1), -5, dtype=np.int8)
    wgt = np.ones((1, 1, 3, 3), dtype=np.int8)
    bias = np.array([0], dtype=np.int32)
    got, _ = await run_conv(dut, act, wgt, bias, stride=1, pad=0, relu=True)
    assert got.shape == (1, 1, 1)
    assert got[0, 0, 0] == 0


@cocotb.test()
async def test_sat_high(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset(dut)
    act = np.full((1, 1, 1), 127, dtype=np.int8)
    wgt = np.array([[[[127]]]], dtype=np.int8)
    bias = np.array([1000], dtype=np.int32)
    got, _ = await run_conv(dut, act, wgt, bias, stride=1, pad=0, relu=False)
    assert got[0, 0, 0] == 127


@cocotb.test()
async def test_sat_low(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset(dut)
    act = np.full((1, 1, 1), -128, dtype=np.int8)
    wgt = np.array([[[[127]]]], dtype=np.int8)
    bias = np.array([-1000], dtype=np.int32)
    got, _ = await run_conv(dut, act, wgt, bias, stride=1, pad=0, relu=False)
    assert got[0, 0, 0] == -128


@cocotb.test()
async def test_k5_and_k8(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset(dut)
    rng = np.random.default_rng(4)
    act = rng_i8(rng, (8, 8, 1))
    wgt = rng_i8(rng, (2, 1, 5, 5))
    bias = rng.integers(-10, 10, size=(2,), dtype=np.int32)
    await run_conv(dut, act, wgt, bias, stride=1, pad=2, relu=True)
    await reset(dut)
    wgt8 = rng_i8(rng, (2, 1, 8, 8))
    await run_conv(dut, act, wgt8, bias, stride=1, pad=0, relu=False)


@cocotb.test()
async def test_random_configs(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset(dut)
    rng = np.random.default_rng(99)
    configs = 0
    while configs < 12:
        k = int(rng.choice([1, 3, 5]))
        s = int(rng.choice([1, 2]))
        p = int(rng.integers(0, min(3, k)))
        cin = int(rng.integers(1, 4))
        cout = int(rng.integers(1, 4))
        h = int(rng.choice([5, 6, 7, 8]))
        w = int(rng.choice([5, 6, 7, 8]))
        if (h + 2 * p - k) < 0 or (w + 2 * p - k) < 0:
            continue
        if (h + 2 * p - k) % s != 0 or (w + 2 * p - k) % s != 0:
            continue
        act = rng_i8(rng, (h, w, cin))
        wgt = rng_i8(rng, (cout, cin, k, k))
        bias = rng.integers(-64, 64, size=(cout,), dtype=np.int32)
        await reset(dut)
        await run_conv(dut, act, wgt, bias, stride=s, pad=p, relu=bool(rng.integers(0, 2)))
        configs += 1
