module scratchpad #(
  parameter int BYTES = cnn_pkg::SRAM_BYTES
) (
  input  logic        clk,
  input  logic        req,
  input  logic        we,
  input  logic [18:0] addr,
  input  logic [7:0]  wdata,
  output logic [7:0]  rdata
);
  import cnn_pkg::*;
  logic [7:0] mem [0:BYTES-1];

  assign rdata = mem[addr];

  always_ff @(posedge clk) begin
    if (req && we)
      mem[addr] <= wdata;
  end
endmodule
