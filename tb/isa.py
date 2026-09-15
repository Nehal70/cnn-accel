"""Pack 32-byte ISA instructions for tests."""

from __future__ import annotations

import struct

OP_STOP = 0
OP_LOAD = 1
OP_STORE = 2
OP_CONV = 3
OP_POOL = 4
OP_FC = 5
OP_ADD = 6


def pack(
    opcode: int,
    *,
    relu: int = 0,
    k: int = 0,
    s: int = 0,
    pad: int = 0,
    c_in: int = 0,
    c_out: int = 0,
    h: int = 0,
    w: int = 0,
    addr0: int = 0,
    addr1: int = 0,
    addr2: int = 0,
    addr3: int = 0,
) -> bytes:
    blob = struct.pack(
        "<BBBBBBHHHHHIIII",
        opcode & 0xFF,
        relu & 1,
        k & 0xFF,
        s & 0xFF,
        pad & 0xFF,
        0,
        c_in & 0xFFFF,
        c_out & 0xFFFF,
        h & 0xFFFF,
        w & 0xFFFF,
        0,
        addr0 & 0xFFFFFFFF,
        addr1 & 0xFFFFFFFF,
        addr2 & 0xFFFFFFFF,
        addr3 & 0xFFFFFFFF,
    )
    if len(blob) != 32:
        raise RuntimeError(f"instruction is {len(blob)} bytes, want 32")
    return blob


def program(instrs: list[bytes]) -> bytes:
    return b"".join(instrs)
