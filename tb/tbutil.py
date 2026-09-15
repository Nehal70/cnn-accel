from __future__ import annotations

from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

CLK_NS = 10


async def reset(dut):
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.host_we.value = 0
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


async def pulse_start(dut, timeout=500_000):
    await RisingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    n = 0
    while int(dut.done.value) == 0:
        await RisingEdge(dut.clk)
        n += 1
        if n > timeout:
            raise TimeoutError("engine timeout")
    assert int(dut.error.value) == 0
