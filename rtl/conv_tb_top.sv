// Unit-test wrapper: 64KiB SRAM + conv_engine + host poke port
module conv_tb_top (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  output logic        busy,
  output logic        done,
  output logic        error,
  input  logic [7:0]  k,
  input  logic [7:0]  s,
  input  logic [7:0]  pad,
  input  logic        relu,
  input  logic [15:0] c_in,
  input  logic [15:0] c_out,
  input  logic [15:0] h,
  input  logic [15:0] w,
  input  logic [31:0] dst_base,
  input  logic [31:0] src_base,
  input  logic [31:0] wgt_base,
  input  logic [31:0] bias_base,
  input  logic        host_we,
  input  logic [15:0] host_addr,
  input  logic [7:0]  host_wdata,
  output logic [7:0]  host_rdata
);
  localparam int BYTES = 65536;

  logic [7:0] mem [0:BYTES-1];

  logic        mem_req, mem_we;
  logic [18:0] mem_addr;
  logic [7:0]  mem_wdata, mem_rdata;

  assign mem_rdata  = mem[mem_addr[15:0]];
  assign host_rdata = mem[host_addr];

  always_ff @(posedge clk) begin
    if (host_we)
      mem[host_addr] <= host_wdata;
    else if (mem_req && mem_we)
      mem[mem_addr[15:0]] <= mem_wdata;
  end

  conv_engine u_conv (
    .clk, .rst_n, .start, .busy, .done, .error,
    .k, .s, .pad, .relu, .c_in, .c_out, .h, .w,
    .dst_base, .src_base, .wgt_base, .bias_base,
    .mem_req, .mem_we, .mem_addr, .mem_wdata, .mem_rdata
  );
endmodule
