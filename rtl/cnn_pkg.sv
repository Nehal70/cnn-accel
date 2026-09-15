`ifndef CNN_PKG_SV
`define CNN_PKG_SV

package cnn_pkg;
  localparam int SRAM_BYTES = 524288;
  localparam int SRAM_AW    = 19;
  localparam int INSTR_BYTES = 32;

  localparam logic [7:0] OP_STOP  = 8'h00;
  localparam logic [7:0] OP_LOAD  = 8'h01;
  localparam logic [7:0] OP_STORE = 8'h02;
  localparam logic [7:0] OP_CONV  = 8'h03;
  localparam logic [7:0] OP_POOL  = 8'h04;
  localparam logic [7:0] OP_FC    = 8'h05;
  localparam logic [7:0] OP_ADD   = 8'h06;

  // Activations: NHWC with N=1, byte offset (y*W + x)*C + c
  // CONV weights: OIHW (PyTorch), (oc*Cin + ic)*K*K + ky*K + kx
  // FC weights:   (oc * Cin + ic)

  function automatic logic signed [7:0] sat_i8(input logic signed [31:0] x);
    if (x > 32'sd127)  return 8'sd127;
    if (x < -32'sd128) return -8'sd128;
    return x[7:0];
  endfunction

  function automatic logic signed [31:0] maybe_relu(
      input logic signed [31:0] x,
      input logic relu
  );
    if (relu && x[31]) return 32'sd0;
    return x;
  endfunction
endpackage

`endif
