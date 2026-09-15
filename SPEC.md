# CNN Accelerator Spec (v0.3)

Shared contract for the compiler, Python ISA simulator, Verilog DUT, and testbench.
If this file and an implementation disagree, **this file wins** until we change it on purpose.

Status: decisions in **Agreed** are locked for v1. **Proposed** means pick these unless we hate them. **Open** must be filled before RTL freeze.

---

## 1. What this spec must contain

A useful spec for this project is not a CNN tutorial. It is every number and layout that two people (or a compiler and a parser) could silently disagree on:

| Section | Why it exists |
|---|---|
| Goals / non-goals | Stops scope creep (no training, no tiling, no softmax) |
| Legal CNN (“the box”) | Compiler reject rules; sixth network contract |
| ISA ops + exact encoding | `netparser` and the packer must bitwise-match |
| Per-op field semantics | What `addr0` means on `LOAD` vs `CONV` |
| Tensor layout + dtypes | SRAM byte math |
| Numerics (accum, bias, ReLU, sat) | PyTorch vs RTL golden |
| SRAM map + alignment | Allocator and preload |
| DRAM model + YAML schema | TB blobs |
| DMA rules | Burst size, outstanding reqs, who stalls |
| Module list + interfaces | Who talks to SRAM |
| CSRs | How simulation/CPU starts a run |
| Compiler contract | Passes, fail-closed behavior |
| Reset, errors, done | Hang vs clean halt |
| Golden-model rules | What “correct” means |

Do **not** put MLIR ODS, Verilog FSMs, or model zoo writeups here. Those are implementation.

---

## 2. Goals and non-goals

**Goals**

- Inference of small CNNs that fit the box, including networks we have not written yet.
- One program in SRAM, fetched instruction-by-instruction by `netparser`.
- Five acceptance nets: TinyConv, ConvPoolNet, LeNet-5 (ReLU), CIFAR-Tiny, MiniResNet.

**Non-goals (v1)**

- Training, backprop, dynamic shapes, batch > 1.
- Compiler tiling of fat layers.
- Softmax / argmax on-chip (host after `STORE`).
- `Flatten` as an opcode (view / addressing only).
- Groups, dilation, `Conv3d`, BatchNorm-as-a-layer, dropout.
- Cycle-accurate real DDR PHY.

---

## 3. The box (legal CNN)

A sixth network must compile **iff** it satisfies all of this. The five named nets are tests, not a whitelist.

| Constraint | v1 rule |
|---|---|
| Batch | Always 1 |
| Layout | NCHW, contiguous, `N=1` |
| Activation rank | `C × H × W` with each dim in `1..=64` |
| Dtype in SRAM | `int8` activations and weights |
| `Conv2d` | Square `K ∈ {1,3,5}`, stride `S ∈ 1..=8`, pad `P ∈ 0..=2` |
| `MaxPool2d` | Square `K ∈ 1..=8`, stride `S ∈ 1..=8`, pad `0` |
| `Linear` / FC | `in_features ∈ 1..=1024`, `out_features ∈ 1..=256` |
| Residual | Same-shape `int8` add, optional fused ReLU |
| Activation | ReLU only, fused as `flags.relu` on **every compute op**: `CONV`, `FC`, `ADD`, `POOL` |
| Scratchpad | Unified **512 KiB** for program + activations + weights |
| DRAM | Unbounded *capacity* in sim; still has latency |

One `64×64×64` int8 map is 256 KiB, so it fits. Two of them (a full-size skip) are 512 KiB with no room for program or weights — legal *shapes* can still **fail allocate**. Flattening a `64×64×64` cube into `FC` is illegal (`262144 > 1024`); pool down first.

Illegal at compile time: unknown ops, dynamic `H/W`, live set that does not fit 512 KiB, tiling required.

**Fixed spatial size** means `H` and `W` are constants when we compile. Different programs may use 8, 28, 32, 64, … as long as each is ≤ 64.

**`K` / `S` / `pad`:** `K` is the square window size in pixels (`K×K` filter for `CONV`, `K×K` window for `POOL`) — not channel count and not a learned pool kernel. `S` is how many pixels that window steps. `pad` is how many zeros `addr_gen` pretends exist beyond each spatial edge (symmetric). For `CONV`, `P_max = (K_max-1)/2 = 2` covers PyTorch `valid` (`P=0`) and `same` at stride 1 (`K=3 → P=1`, `K=5 → P=2`). Extra pad is legal in PyTorch but we do not need it. Pool pad stays 0. `S ≤ 8` is fine (same counter width as pool); compiler still requires `(H + 2P - K)` divisible by `S` and `H_out ≥ 1`.

---

## 4. ISA

### 4.1 Opcodes

| Code | Mnemonic | Class | Hardware block |
|------|----------|--------|----------------|
| `0x00` | `HALT` | Control | `netparser` |
| `0x01` | `LOAD` | Memory | `dma` |
| `0x02` | `STORE` | Memory | `dma` |
| `0x03` | `CONV` | Compute | `mac_array` + `addr_gen` |
| `0x04` | `POOL` | Compute | `pool_unit` + `addr_gen` |
| `0x05` | `FC` | Compute | `mac_array` + `addr_gen` |
| `0x06` | `ADD` | Compute | `eltwise` |
| `0x07–0xFF` | reserved | — | Must `error` |

One PyTorch `Conv2d` / `Linear` / `MaxPool2d` / residual `add` → **one** ISA op after ReLU fusion. `LOAD`/`STORE` are extra, inserted by the allocator.

### 4.2 Instruction size (Proposed)

- Fixed **32-byte** little-endian instruction.
- `PC` is a **byte** address in SRAM, **64-byte? no — 32-byte** aligned.
- After each op, `PC += 32` unless `HALT`.

Parser reads 32 bytes from `scratchpad[PC +: 32]`, then dispatches and waits for `engine_done`.

### 4.3 Byte layout (Proposed)

All multi-byte fields little-endian. Unused fields must be written **0** by the compiler (parser may ignore them).

| Bytes | Name | Type | Meaning |
|------:|------|------|---------|
| 0 | `opcode` | u8 | §4.1 |
| 1 | `flags` | u8 | bit 0 = `relu` (`CONV`/`FC`/`ADD`/`POOL`) |
| 2 | `k` | u8 | Kernel / pool window |
| 3 | `s` | u8 | Stride |
| 4 | `pad` | u8 | Spatial pad (CONV) |
| 5 | `shift` | u8 | v1: **0**. Reserved for requant shift |
| 6–7 | `c_in` | u16 | Input channels or `in_features` |
| 8–9 | `c_out` | u16 | Output channels or `out_features` |
| 10–11 | `h` | u16 | Input height (CONV/POOL) or `1` (FC) |
| 12–13 | `w` | u16 | Input width (CONV/POOL) or `1` (FC) |
| 14–15 | `_res` | u16 | 0 |
| 16–19 | `addr0` | u32 | See per-op table |
| 20–23 | `addr1` | u32 | |
| 24–27 | `addr2` | u32 | |
| 28–31 | `addr3` | u32 | |

`relu=1` means writeback `max(0, x)` after the op’s int32 result (after bias on `CONV`/`FC`, after the add on `ADD`, after the max on `POOL`), still before `sat_int8`. `LOAD` / `STORE` / `STOP` ignore the bit.

### 4.4 Per-op addresses

| Op | addr0 | addr1 | addr2 | addr3 |
|----|-------|-------|-------|-------|
| `HALT` | 0 | 0 | 0 | 0 |
| `LOAD` | SRAM dest | DRAM src | 0 | `nbytes` |
| `STORE` | SRAM src | DRAM dest | 0 | `nbytes` |
| `CONV` | SRAM dst act | SRAM src act | SRAM weights | SRAM bias |
| `POOL` | SRAM dst | SRAM src | 0 | 0 |
| `FC` | SRAM dst | SRAM src vec | SRAM weights | SRAM bias |
| `ADD` | SRAM dst | SRAM src0 | SRAM src1 | 0 |

SRAM addresses are **byte offsets** `0 .. 524287`. DRAM addresses are **byte offsets** in the TB array (32-bit is enough in sim).

### 4.5 Shape math (parser and compiler must match)

Let input be `C_in × H × W` (batch omitted).

**CONV** (`K,S,P` as in the instruction):

```
H_out = (H + 2P - K) / S + 1
W_out = (W + 2P - K) / S + 1
```

Must divide evenly (integer, no leftover). Output tensor `C_out × H_out × W_out`.

Weights: `C_out × C_in × K × K` int8, NCHW.
Bias: `C_out` **int32** (4 bytes each). See §6.

**POOL:**

```
H_out = (H - K) / S + 1     # pad = 0
W_out = (W - K) / S + 1
```

Must divide evenly. `C_out` must equal `C_in`. No weight tensor.

**FC:** Treat `src` as a vector of length `c_in`. Weights `c_out × c_in` int8 row-major (`out` is the slowest dim). Bias `c_out` int32. Output `c_out` int8. `h=w=1`, `k=s=pad=0`.

**ADD:** `src0` and `src1` and `dst` all `c_in × h × w` (use `c_out = c_in`). Elementwise.

**LOAD/STORE:** `nbytes` must be > 0 and `addr + nbytes` must not overflow that memory. No overlap rules in v1 except: a running compute op’s buffers must not be written by DMA (parser is in-order, one op at a time, so this is free).

---

## 5. Tensor layout in SRAM

- NCHW, `N=1` omitted from addressing.
- Index: `off = ((c * H + y) * W + x) * sizeof(int8)` for activations.
- Weights CONV: `off = ((oc * C_in + ic) * K + ky) * K + kx`.
- Weights FC: `off = oc * c_in + ic`.
- **Alignment (Proposed):** tensor bases 16-byte aligned. Program base 32-byte aligned. `nbytes` on DMA 16-byte aligned (compiler pads).

---

## 6. Numerics (Agreed direction, Proposed details)

| Item | Rule |
|---|---|
| Activations / weights in SRAM | `int8` two’s complement |
| MAC accumulator | `int32`, one acc per output element |
| Bias | `int32`, added after the MAC reduction |
| ReLU | If `flags.relu`: `x < 0 ? 0 : x` on the int32 result of `CONV`/`FC`/`ADD`/`POOL` |
| Writeback | `sat_int8(x)` = clamp to `[-128, 127]`, store int8 |
| `shift` | 0 in v1 (no explicit requant scale). PyTorch graphs must be **pre-quantized** or simulated with the same sat rule |
| Overflow of int32 acc | Open — treat as wrap (two’s complement) until we say otherwise |

v1 correctness story: golden Python uses **the same** int8 MAC + int32 acc + sat, not full f32 PyTorch, unless we add a quantize spec later.

---

## 7. Memory

### 7.1 SRAM (DUT)

- One unified array, **512 KiB** = `524288` bytes.
- Holds program, activations, weights, bias.
- Model in Verilog as an array (or dual-port array).
- **Proposed map:**
  - `[0x00000, 0x01000)` — program (4 KiB, 128 instructions).
  - `[0x01000, 0x80000)` — tensors (compiler bump allocator).
- Preload: TB writes the program (and optional constants) into SRAM **before** `go`. `netparser` is given `program_base` (default `0`).

### 7.2 DRAM (simulation only)

- Not an RTL macro. DUT has a **DMA port**; TB has a big byte array.
- Capacity: treat as unbounded for allocation (32-bit address in the instruction).
- Latency: **Proposed** fixed `32` cycles from request handshake to first beat, then one beat/cycle. Adjust later; do not model banks/refresh.
- Host/TB fills DRAM from YAML blobs at time 0.

### 7.3 Live set / allocator

Compiler tracks live SRAM buffers. Peak sum of live tensors + program region ≤ 512 KiB. Exceed → **compile error**, not silent wrap.

Skip `ADD` needs two input maps live at once; `dst` may reuse a dead buffer, not `src0`/`src1` unless we later allow in-place (v1: **dst distinct**, Proposed).

---

## 8. DMA

- In-order. One outstanding DRAM request at a time (Proposed).
- Beat width: **Proposed** 32-bit (4 bytes/cycle) on `dram_if`.
- `LOAD`: DRAM → SRAM. `STORE`: SRAM → DRAM.
- Parser stalls until DMA `done`.
- No compute/DMA overlap in v1.

---

## 9. Hardware modules (DUT)

Names are part of the spec so RTL files and the compiler README match.

| Module | Role |
|---|---|
| `host_csr` | `program_base`, `go`, `done`, `error` |
| `netparser` | Fetch 32 B, decode, dispatch, `PC += 32`, `HALT` → `done` |
| `scratchpad` | 512 KiB array |
| `sram_arb` | Parser I-fetch vs DMA vs datapath. Proposed: dual-port (A=parser, B=data) |
| `dma` | `LOAD`/`STORE` engine |
| `dram_if` | Request/response pins (addr, wen, wdata, rdata, valid/ready, len) |
| `addr_gen` | Walk `c,y,x,ic,ky,kx` (or FC `oc,ic`) from instruction fields |
| `mac_array` | `CONV` and `FC` MACs, bias, relu, sat |
| `pool_unit` | Max over `K×K`, stride `S`, optional relu + sat |
| `eltwise` | `ADD` + optional relu + sat |

No Flatten unit. No standalone ReLU unit.

**Control (Agreed):** single-issue, in-order, one engine at a time. Parser does not fetch the next instruction until `engine_done`.

---

## 10. CSRs (Proposed)

Byte offsets on a small CSR space (TB/CPU view):

| Offset | Name | Access | Meaning |
|--------|------|--------|---------|
| `0x00` | `GO` | W1 | Write 1 to start; stays 0 while running |
| `0x04` | `PROGRAM_BASE` | RW | SRAM byte address of first instruction |
| `0x08` | `DONE` | R | 1 after `HALT` |
| `0x0C` | `ERROR` | R | 0 ok; nonzero sticky fault (bad opcode, OOB addr) |
| `0x10` | `PC` | R | Current PC (debug) |

On reset: `DONE=0`, `ERROR=0`, `PC=PROGRAM_BASE`, SRAM/DRAM contents **not** cleared (TB preloads).

---

## 11. DRAM YAML blobs (Proposed)

Compiler and TB share this schema. Blobs are host-side; the Verilog DRAM is the flattened byte array.

```yaml
arch: cnn-accel-v0
entry: 0x00000000          # SRAM program_base
program:
  sram_base: 0x00000000
  file: program.bin        # packed 32-byte instructions
blobs:
  - name: input
    dram_addr: 0x00000000
    sram_addr: null        # filled if preloaded to SRAM; else LOAD
    shape: [1, 1, 28, 28]  # NCHW
    dtype: int8            # or int32 for bias
    file: input.bin
  - name: logits
    dram_addr: 0x00010000
    shape: [1, 10]
    dtype: int8
    file: null             # STORE target, empty at start
```

`file` bytes must equal `prod(shape) * sizeof(dtype)`.

---

## 12. Compiler contract

Pipeline (implementation can be Python first, MLIR later; **behavior** is spec):

1. Import FX / `torch.export`. Unknown op → error.
2. Fuse `relu` into the preceding `conv` / `linear` / `add` / `max_pool2d` (set `flags.relu`). A `relu` with no such producer is illegal in v1.
3. Lower `flatten`/`view` to layout (no op).
4. Check the box (§3).
5. Allocate SRAM; assign DRAM addresses for input, parameters, output.
6. Emit `LOAD`s, compute ops, `STORE` of outputs, `HALT`.
7. Pack 32-byte instructions + YAML + `.bin` blobs.

Must **not** special-case the five model class names.

---

## 13. Correctness

A run is correct when DRAM logits (or TinyConv’s stored feature map) **byte-match** the Python ISA simulator. The simulator is the golden model. PyTorch f32 is a reference for *structure* and, after quantization, for “close enough”; it is not the v1 RTL bitwise golden unless we say so in a later quant spec.

Acceptance order: TinyConv → DMA-only tests → CONV → ConvPoolNet → LeNet → CIFAR-Tiny → MiniResNet.

---

## 14. Open items (fill before RTL freeze)

- [ ] Confirm 32-byte instruction vs 64-byte (parser simplicity vs density).
- [ ] Dual-port SRAM vs single-port arbiter + instruction buffer.
- [ ] DMA beat width (32 vs 64 bit) and DRAM latency cycles.
- [ ] In-place `ADD` (`dst == src0`) allowed or not.
- [ ] Pool padding (currently 0 only).
- [ ] Bias in SRAM as int32 vs int8-upcast.
- [ ] int32 accumulator overflow policy.
- [ ] Quantization: pre-quantized graphs only vs scales in `shift`/per-channel.
- [ ] Exact `dram_if` valid/ready timing diagram.
- [ ] Max program size (4 KiB Proposed).

---

## 15. Document history

| Ver | Notes |
|-----|--------|
| 0.1 | Initial lock of box, 7 ops, 512 KiB unified SRAM, YAML DRAM, module names, proposed 32-byte encoding |
| 0.2 | Activation cap `64×64×64`; CONV stride `1..=8`; document `K`/`S`/`pad` |
| 0.3 | `flags.relu` on `CONV`, `FC`, `ADD`, and `POOL` |
