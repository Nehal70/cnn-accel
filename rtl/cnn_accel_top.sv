module cnn_accel_top (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        go,
  input  logic [31:0] program_base,
  output logic        done,
  output logic        error,
  output logic [31:0] pc,

  // host SRAM access when idle
  input  logic        host_we,
  input  logic [18:0] host_addr,
  input  logic [7:0]  host_wdata,
  output logic [7:0]  host_rdata,

  output logic        dram_req,
  output logic        dram_we,
  output logic [31:0] dram_addr,
  output logic [7:0]  dram_wdata,
  input  logic [7:0]  dram_rdata
);
  import cnn_pkg::*;

  logic [7:0] sram [0:SRAM_BYTES-1];

  logic p_req, c_req, o_req, f_req, a_req, d_req;
  logic p_we,  c_we,  o_we,  f_we,  a_we,  d_we;
  logic [18:0] p_addr, c_addr, o_addr, f_addr, a_addr, d_addr;
  logic [7:0]  p_wdata, c_wdata, o_wdata, f_wdata, a_wdata, d_wdata;
  logic [7:0]  sram_rdata;

  logic        mem_req, mem_we;
  logic [18:0] mem_addr;
  logic [7:0]  mem_wdata;

  logic conv_start, pool_start, fc_start, add_start, dma_start, dma_store;
  logic conv_done, pool_done, fc_done, add_done, dma_done;
  logic conv_err, pool_err, fc_err, add_err, dma_err;
  logic conv_busy, pool_busy, fc_busy, add_busy, dma_busy;

  logic [7:0]  k, s, pad;
  logic        relu;
  logic [15:0] c_in, c_out, h, w;
  logic [31:0] addr0, addr1, addr2, addr3;

  logic running;
  assign running = conv_busy | pool_busy | fc_busy | add_busy | dma_busy;

  always_comb begin
    mem_req   = 1'b0;
    mem_we    = 1'b0;
    mem_addr  = '0;
    mem_wdata = '0;
    if (p_req) begin
      mem_req = p_req; mem_we = p_we; mem_addr = p_addr; mem_wdata = p_wdata;
    end else if (c_req) begin
      mem_req = c_req; mem_we = c_we; mem_addr = c_addr; mem_wdata = c_wdata;
    end else if (o_req) begin
      mem_req = o_req; mem_we = o_we; mem_addr = o_addr; mem_wdata = o_wdata;
    end else if (f_req) begin
      mem_req = f_req; mem_we = f_we; mem_addr = f_addr; mem_wdata = f_wdata;
    end else if (a_req) begin
      mem_req = a_req; mem_we = a_we; mem_addr = a_addr; mem_wdata = a_wdata;
    end else if (d_req) begin
      mem_req = d_req; mem_we = d_we; mem_addr = d_addr; mem_wdata = d_wdata;
    end
  end

  assign sram_rdata = sram[mem_addr];
  assign host_rdata = sram[host_addr];

  always_ff @(posedge clk) begin
    if (host_we && !mem_req)
      sram[host_addr] <= host_wdata;
    else if (mem_req && mem_we)
      sram[mem_addr] <= mem_wdata;
  end

  netparser u_par (
    .clk, .rst_n, .go, .program_base, .done, .error, .pc,
    .conv_start, .pool_start, .fc_start, .add_start, .dma_start, .dma_store,
    .conv_done, .pool_done, .fc_done, .add_done, .dma_done,
    .conv_err, .pool_err, .fc_err, .add_err, .dma_err,
    .k, .s, .pad, .relu, .c_in, .c_out, .h, .w,
    .addr0, .addr1, .addr2, .addr3,
    .mem_req(p_req), .mem_we(p_we), .mem_addr(p_addr), .mem_wdata(p_wdata), .mem_rdata(sram_rdata)
  );

  conv_engine u_conv (
    .clk, .rst_n, .start(conv_start), .busy(conv_busy), .done(conv_done), .error(conv_err),
    .k, .s, .pad, .relu, .c_in, .c_out, .h, .w,
    .dst_base(addr0), .src_base(addr1), .wgt_base(addr2), .bias_base(addr3),
    .mem_req(c_req), .mem_we(c_we), .mem_addr(c_addr), .mem_wdata(c_wdata), .mem_rdata(sram_rdata)
  );

  pool_engine u_pool (
    .clk, .rst_n, .start(pool_start), .busy(pool_busy), .done(pool_done), .error(pool_err),
    .k, .s, .relu, .c(c_in), .h, .w,
    .dst_base(addr0), .src_base(addr1),
    .mem_req(o_req), .mem_we(o_we), .mem_addr(o_addr), .mem_wdata(o_wdata), .mem_rdata(sram_rdata)
  );

  fc_engine u_fc (
    .clk, .rst_n, .start(fc_start), .busy(fc_busy), .done(fc_done), .error(fc_err),
    .relu, .c_in, .c_out,
    .dst_base(addr0), .src_base(addr1), .wgt_base(addr2), .bias_base(addr3),
    .mem_req(f_req), .mem_we(f_we), .mem_addr(f_addr), .mem_wdata(f_wdata), .mem_rdata(sram_rdata)
  );

  eltwise u_add (
    .clk, .rst_n, .start(add_start), .busy(add_busy), .done(add_done), .error(add_err),
    .relu, .c(c_in), .h, .w,
    .dst_base(addr0), .src0_base(addr1), .src1_base(addr2),
    .mem_req(a_req), .mem_we(a_we), .mem_addr(a_addr), .mem_wdata(a_wdata), .mem_rdata(sram_rdata)
  );

  dma u_dma (
    .clk, .rst_n, .start(dma_start), .is_store(dma_store),
    .busy(dma_busy), .done(dma_done), .error(dma_err),
    .sram_addr(addr0), .dram_addr(addr1), .nbytes(addr3),
    .mem_req(d_req), .mem_we(d_we), .mem_addr(d_addr), .mem_wdata(d_wdata), .mem_rdata(sram_rdata),
    .dram_req, .dram_we, .dram_a(dram_addr), .dram_wdata, .dram_rdata
  );
endmodule
