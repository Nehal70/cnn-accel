from __future__ import annotations

import os
import sys

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

sys.path.insert(0, os.path.dirname(__file__))
from isa import OP_ADD, OP_CONV, OP_FC, OP_LOAD, OP_POOL, OP_STOP, OP_STORE, pack, program
from ref import add_int, conv2d_int, fc_int, maxpool_int

CLK_NS = 10


async def reset_chip(dut):
    dut.rst_n.value = 0
    dut.go.value = 0
    dut.program_base.value = 0
    dut.host_we.value = 0
    dut.dram_we.value = 0
    dut.host_addr.value = 0
    dut.host_wdata.value = 0
    dut.dram_host_addr.value = 0
    dut.dram_host_wdata.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def sram_write(dut, addr, data: bytes):
    for i, b in enumerate(data):
        dut.host_addr.value = addr + i
        dut.host_wdata.value = int(b) & 0xFF
        dut.host_we.value = 1
        await RisingEdge(dut.clk)
    dut.host_we.value = 0
    await RisingEdge(dut.clk)


async def sram_read(dut, addr, n: int) -> bytes:
    out = bytearray(n)
    for i in range(n):
        dut.host_addr.value = addr + i
        await RisingEdge(dut.clk)
        out[i] = int(dut.host_rdata.value) & 0xFF
    return bytes(out)


async def dram_write(dut, addr, data: bytes):
    for i, b in enumerate(data):
        dut.dram_host_addr.value = addr + i
        dut.dram_host_wdata.value = int(b) & 0xFF
        dut.dram_we.value = 1
        await RisingEdge(dut.clk)
    dut.dram_we.value = 0
    await RisingEdge(dut.clk)


async def dram_read(dut, addr, n: int) -> bytes:
    out = bytearray(n)
    for i in range(n):
        dut.dram_host_addr.value = addr + i
        await RisingEdge(dut.clk)
        out[i] = int(dut.dram_host_rdata.value) & 0xFF
    return bytes(out)


async def run_prog(dut, prog: bytes, timeout=5_000_000):
    await sram_write(dut, 0, prog)
    dut.program_base.value = 0
    await RisingEdge(dut.clk)
    dut.go.value = 1
    await RisingEdge(dut.clk)
    dut.go.value = 0
    n = 0
    while int(dut.done.value) == 0:
        await RisingEdge(dut.clk)
        n += 1
        if n > timeout:
            raise TimeoutError("chip timeout")
    assert int(dut.error.value) == 0, "chip error"


@cocotb.test()
async def test_stop_only(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset_chip(dut)
    await run_prog(dut, pack(OP_STOP), timeout=2000)


@cocotb.test()
async def test_load_store(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset_chip(dut)
    payload = bytes(range(64))
    await dram_write(dut, 0x100, payload)
    prog = program(
        [
            pack(OP_LOAD, addr0=0x1000, addr1=0x100, addr3=64),
            pack(OP_STORE, addr0=0x1000, addr1=0x200, addr3=64),
            pack(OP_STOP),
        ]
    )
    await run_prog(dut, prog)
    got = await dram_read(dut, 0x200, 64)
    assert got == payload


@cocotb.test()
async def test_tinyconv_program(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset_chip(dut)
    rng = np.random.default_rng(7)
    act = rng.integers(-40, 40, size=(8, 8, 1), dtype=np.int8)
    wgt = rng.integers(-8, 8, size=(4, 1, 3, 3), dtype=np.int8)
    bias = rng.integers(-10, 10, size=(4,), dtype=np.int32)
    exp = conv2d_int(act, wgt, bias, 1, 1, True)

    await dram_write(dut, 0x0000, act.tobytes())
    await dram_write(dut, 0x0100, np.ascontiguousarray(wgt).tobytes())
    await dram_write(dut, 0x0200, np.ascontiguousarray(bias, dtype="<i4").tobytes())

    s_x, s_w, s_b, s_y = 0x2000, 0x2100, 0x2200, 0x2300
    prog = program(
        [
            pack(OP_LOAD, addr0=s_x, addr1=0x0000, addr3=act.nbytes),
            pack(OP_LOAD, addr0=s_w, addr1=0x0100, addr3=wgt.nbytes),
            pack(OP_LOAD, addr0=s_b, addr1=0x0200, addr3=bias.nbytes),
            pack(
                OP_CONV,
                relu=1,
                k=3,
                s=1,
                pad=1,
                c_in=1,
                c_out=4,
                h=8,
                w=8,
                addr0=s_y,
                addr1=s_x,
                addr2=s_w,
                addr3=s_b,
            ),
            pack(OP_STORE, addr0=s_y, addr1=0x0400, addr3=exp.nbytes),
            pack(OP_STOP),
        ]
    )
    await run_prog(dut, prog)
    raw = await dram_read(dut, 0x0400, exp.nbytes)
    got = np.frombuffer(raw, dtype=np.int8).reshape(exp.shape)
    np.testing.assert_array_equal(got, exp)


@cocotb.test()
async def test_pool_fc_add_program(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset_chip(dut)
    rng = np.random.default_rng(8)
    act = rng.integers(-30, 30, size=(4, 4, 2), dtype=np.int8)
    pooled = maxpool_int(act, 2, 2, False)
    vec = pooled.reshape(-1)
    wgt = rng.integers(-5, 5, size=(3, vec.size), dtype=np.int8)
    bias = np.zeros(3, dtype=np.int32)
    logits = fc_int(vec, wgt, bias, True)
    skip = add_int(pooled, pooled, True)

    await sram_write(dut, 0x1000, act.tobytes())
    await sram_write(dut, 0x1400, np.ascontiguousarray(wgt).tobytes())
    await sram_write(dut, 0x1600, np.ascontiguousarray(bias, dtype="<i4").tobytes())

    prog = program(
        [
            pack(OP_POOL, k=2, s=2, c_in=2, c_out=2, h=4, w=4, addr0=0x1100, addr1=0x1000),
            pack(
                OP_FC,
                relu=1,
                c_in=vec.size,
                c_out=3,
                h=1,
                w=1,
                addr0=0x1800,
                addr1=0x1100,
                addr2=0x1400,
                addr3=0x1600,
            ),
            pack(OP_ADD, relu=1, c_in=2, h=2, w=2, addr0=0x1900, addr1=0x1100, addr2=0x1100),
            pack(OP_STOP),
        ]
    )
    await run_prog(dut, prog)
    got_fc = np.frombuffer(await sram_read(dut, 0x1800, 3), dtype=np.int8)
    got_add = np.frombuffer(await sram_read(dut, 0x1900, skip.size), dtype=np.int8).reshape(skip.shape)
    np.testing.assert_array_equal(got_fc, logits)
    np.testing.assert_array_equal(got_add, skip)
