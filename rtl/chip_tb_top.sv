module chip_tb_top (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        go,
  input  logic [31:0] program_base,
  output logic        done,
  output logic        error,
  output logic [31:0] pc,
  input  logic        host_we,
  input  logic [18:0] host_addr,
  input  logic [7:0]  host_wdata,
  output logic [7:0]  host_rdata,
  input  logic        dram_we,
  input  logic [31:0] dram_host_addr,
  input  logic [7:0]  dram_host_wdata,
  output logic [7:0]  dram_host_rdata
);
  localparam int DRAM_BYTES = 262144;
  logic [7:0] dram [0:DRAM_BYTES-1];

  logic        dram_req, dram_we_m;
  logic [31:0] dram_addr;
  logic [7:0]  dram_wdata, dram_rdata;

  assign dram_rdata      = dram[dram_addr[17:0]];
  assign dram_host_rdata = dram[dram_host_addr[17:0]];

  always_ff @(posedge clk) begin
    if (dram_we)
      dram[dram_host_addr[17:0]] <= dram_host_wdata;
    else if (dram_req && dram_we_m)
      dram[dram_addr[17:0]] <= dram_wdata;
  end

  cnn_accel_top u_dut (
    .clk, .rst_n, .go, .program_base, .done, .error, .pc,
    .host_we, .host_addr, .host_wdata, .host_rdata,
    .dram_req, .dram_we(dram_we_m), .dram_addr, .dram_wdata, .dram_rdata
  );
endmodule
