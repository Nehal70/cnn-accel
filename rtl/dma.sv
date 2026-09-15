module dma (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         start,
  input  logic         is_store,  // 0=LOAD DRAM->SRAM, 1=STORE SRAM->DRAM
  output logic         busy,
  output logic         done,
  output logic         error,
  input  logic [31:0]  sram_addr,
  input  logic [31:0]  dram_addr,
  input  logic [31:0]  nbytes,
  output logic         mem_req,
  output logic         mem_we,
  output logic [18:0]  mem_addr,
  output logic [7:0]   mem_wdata,
  input  logic [7:0]   mem_rdata,
  output logic         dram_req,
  output logic         dram_we,
  output logic [31:0]  dram_a,
  output logic [7:0]   dram_wdata,
  input  logic [7:0]   dram_rdata
);
  import cnn_pkg::*;

  typedef enum logic [3:0] {
    ST_IDLE, ST_INIT, ST_RD, ST_RD_W, ST_WR, ST_NEXT, ST_DONE, ST_ERR
  } state_t;

  state_t state;
  logic [31:0] i;
  logic [7:0]  tmp;

  assign busy = (state != ST_IDLE) && (state != ST_DONE) && (state != ST_ERR);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= ST_IDLE; done <= 0; error <= 0;
      mem_req <= 0; mem_we <= 0; mem_addr <= '0; mem_wdata <= '0;
      dram_req <= 0; dram_we <= 0; dram_a <= '0; dram_wdata <= '0;
      i <= 0; tmp <= 0;
    end else begin
      done <= 0; mem_req <= 0; mem_we <= 0; dram_req <= 0; dram_we <= 0;
      unique case (state)
        ST_IDLE: if (start) begin error <= 0; state <= ST_INIT; end
        ST_INIT: begin
          if (nbytes == 0) state <= ST_ERR;
          else begin i <= 0; state <= ST_RD; end
        end
        ST_RD: begin
          if (is_store) begin
            mem_req <= 1; mem_addr <= 19'(sram_addr + i);
          end else begin
            dram_req <= 1; dram_a <= dram_addr + i;
          end
          state <= ST_RD_W;
        end
        ST_RD_W: begin
          tmp   <= is_store ? mem_rdata : dram_rdata;
          state <= ST_WR;
        end
        ST_WR: begin
          if (is_store) begin
            dram_req <= 1; dram_we <= 1; dram_a <= dram_addr + i; dram_wdata <= tmp;
          end else begin
            mem_req <= 1; mem_we <= 1; mem_addr <= 19'(sram_addr + i); mem_wdata <= tmp;
          end
          state <= ST_NEXT;
        end
        ST_NEXT: begin
          if (i + 1 < nbytes) begin i <= i + 1; state <= ST_RD; end
          else state <= ST_DONE;
        end
        ST_DONE: begin done <= 1; state <= ST_IDLE; end
        ST_ERR: begin error <= 1; done <= 1; state <= ST_IDLE; end
        default: state <= ST_IDLE;
      endcase
    end
  end
endmodule
