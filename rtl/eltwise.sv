module eltwise (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         start,
  output logic         busy,
  output logic         done,
  output logic         error,
  input  logic         relu,
  input  logic [15:0]  c,
  input  logic [15:0]  h,
  input  logic [15:0]  w,
  input  logic [31:0]  dst_base,
  input  logic [31:0]  src0_base,
  input  logic [31:0]  src1_base,
  output logic         mem_req,
  output logic         mem_we,
  output logic [18:0]  mem_addr,
  output logic [7:0]   mem_wdata,
  input  logic [7:0]   mem_rdata
);
  import cnn_pkg::*;

  typedef enum logic [3:0] {
    ST_IDLE, ST_INIT, ST_RD0, ST_RD0_W, ST_RD1, ST_RD1_W, ST_WR, ST_NEXT, ST_DONE, ST_ERR
  } state_t;

  state_t state;
  logic [31:0] i, n;
  logic signed [7:0] a_s, b_s;

  assign busy = (state != ST_IDLE) && (state != ST_DONE) && (state != ST_ERR);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= ST_IDLE; done <= 0; error <= 0;
      mem_req <= 0; mem_we <= 0; mem_addr <= '0; mem_wdata <= '0;
      i <= 0; n <= 0; a_s <= 0; b_s <= 0;
    end else begin
      done <= 0; mem_req <= 0; mem_we <= 0;
      unique case (state)
        ST_IDLE: if (start) begin error <= 0; state <= ST_INIT; end
        ST_INIT: begin
          n <= 32'(h) * 32'(w) * 32'(c);
          i <= 0;
          if (h == 0 || w == 0 || c == 0) state <= ST_ERR;
          else state <= ST_RD0;
        end
        ST_RD0: begin
          mem_req <= 1; mem_addr <= 19'(src0_base + i); state <= ST_RD0_W;
        end
        ST_RD0_W: begin a_s <= $signed(mem_rdata); state <= ST_RD1; end
        ST_RD1: begin
          mem_req <= 1; mem_addr <= 19'(src1_base + i); state <= ST_RD1_W;
        end
        ST_RD1_W: begin b_s <= $signed(mem_rdata); state <= ST_WR; end
        ST_WR: begin
          mem_req <= 1; mem_we <= 1;
          mem_addr <= 19'(dst_base + i);
          mem_wdata <= sat_i8(maybe_relu(32'(a_s) + 32'(b_s), relu));
          state <= ST_NEXT;
        end
        ST_NEXT: begin
          if (i + 1 < n) begin i <= i + 1; state <= ST_RD0; end
          else state <= ST_DONE;
        end
        ST_DONE: begin done <= 1; state <= ST_IDLE; end
        ST_ERR: begin error <= 1; done <= 1; state <= ST_IDLE; end
        default: state <= ST_IDLE;
      endcase
    end
  end
endmodule
