module pool_tb_top (
  input  logic        clk, rst_n, start,
  output logic        busy, done, error,
  input  logic [7:0]  k, s,
  input  logic        relu,
  input  logic [15:0] c, h, w,
  input  logic [31:0] dst_base, src_base,
  input  logic        host_we,
  input  logic [15:0] host_addr,
  input  logic [7:0]  host_wdata,
  output logic [7:0]  host_rdata
);
  localparam int BYTES = 65536;
  logic [7:0] mem [0:BYTES-1];
  logic mem_req, mem_we;
  logic [18:0] mem_addr;
  logic [7:0] mem_wdata, mem_rdata;
  assign mem_rdata = mem[mem_addr[15:0]];
  assign host_rdata = mem[host_addr];
  always_ff @(posedge clk) begin
    if (host_we) mem[host_addr] <= host_wdata;
    else if (mem_req && mem_we) mem[mem_addr[15:0]] <= mem_wdata;
  end
  pool_engine u (
    .clk, .rst_n, .start, .busy, .done, .error,
    .k, .s, .relu, .c, .h, .w, .dst_base, .src_base,
    .mem_req, .mem_we, .mem_addr, .mem_wdata, .mem_rdata
  );
endmodule
