"""Bitwise-accurate int8 reference kernels matching the RTL."""

from __future__ import annotations

import numpy as np


def sat_i8(x: np.ndarray | int) -> np.int8 | np.ndarray:
    return np.clip(x, -128, 127).astype(np.int8)


def conv2d_int(
    act: np.ndarray,
    wgt: np.ndarray,
    bias: np.ndarray,
    stride: int,
    pad: int,
    relu: bool,
) -> np.ndarray:
    """act (H,W,Cin) int8, wgt (Cout,Cin,K,K) OIHW int8, bias (Cout,) int32."""
    h, w, cin = act.shape
    cout, cin_w, k, k2 = wgt.shape
    assert cin == cin_w and k == k2
    ho = (h + 2 * pad - k) // stride + 1
    wo = (w + 2 * pad - k) // stride + 1
    out = np.empty((ho, wo, cout), dtype=np.int8)
    a32 = act.astype(np.int32)
    w32 = wgt.astype(np.int32)
    b32 = bias.astype(np.int32)
    for oc in range(cout):
        for oy in range(ho):
            for ox in range(wo):
                acc = np.int32(0)
                for ic in range(cin):
                    for ky in range(k):
                        for kx in range(k):
                            iy = oy * stride - pad + ky
                            ix = ox * stride - pad + kx
                            if 0 <= iy < h and 0 <= ix < w:
                                acc += a32[iy, ix, ic] * w32[oc, ic, ky, kx]
                acc = acc + b32[oc]
                if relu and acc < 0:
                    acc = np.int32(0)
                out[oy, ox, oc] = sat_i8(acc)
    return out


def maxpool_int(act: np.ndarray, k: int, stride: int, relu: bool) -> np.ndarray:
    h, w, c = act.shape
    ho = (h - k) // stride + 1
    wo = (w - k) // stride + 1
    out = np.empty((ho, wo, c), dtype=np.int8)
    a32 = act.astype(np.int32)
    for oc in range(c):
        for oy in range(ho):
            for ox in range(wo):
                m = np.int32(-10_000)
                for ky in range(k):
                    for kx in range(k):
                        iy = oy * stride + ky
                        ix = ox * stride + kx
                        v = a32[iy, ix, oc]
                        if v > m:
                            m = v
                if relu and m < 0:
                    m = np.int32(0)
                out[oy, ox, oc] = sat_i8(m)
    return out


def fc_int(vec: np.ndarray, wgt: np.ndarray, bias: np.ndarray, relu: bool) -> np.ndarray:
    """vec (Cin,) int8, wgt (Cout,Cin) int8, bias (Cout,) int32."""
    cout, cin = wgt.shape
    acc = wgt.astype(np.int32) @ vec.astype(np.int32) + bias.astype(np.int32)
    if relu:
        acc = np.maximum(acc, 0)
    return sat_i8(acc)


def add_int(a: np.ndarray, b: np.ndarray, relu: bool) -> np.ndarray:
    acc = a.astype(np.int32) + b.astype(np.int32)
    if relu:
        acc = np.maximum(acc, 0)
    return sat_i8(acc)
