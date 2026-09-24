CNN Accel

Decided to codesign an accelerator for fundamental CNN workloads that use convolution, ReLU, linearize, FC layers. 
- Implemented a 8x8 Broadcast Mac Array
- Used 512KiB SRAM
- Defined 2 MLIR dialects, including an ISA, and an intermediate layer between PyTorch and ISA.
- Used NHWC activations, OIHW weights, int32 bias/accumulators
- Built 3 passes to lower IRs, fuse ReLU, etc.
- Tested over 15 PyTorch networks using verilator, cocotb and lowering the pytorch using my custom MLIR compiler into ISA that is fed into the DUT. 
