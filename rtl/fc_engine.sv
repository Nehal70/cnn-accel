module fc_engine (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         start,
  output logic         busy,
  output logic         done,
  output logic         error,
  input  logic         relu,
  input  logic [15:0]  c_in,
  input  logic [15:0]  c_out,
  input  logic [31:0]  dst_base,
  input  logic [31:0]  src_base,
  input  logic [31:0]  wgt_base,
  input  logic [31:0]  bias_base,
  output logic         mem_req,
  output logic         mem_we,
  output logic [18:0]  mem_addr,
  output logic [7:0]   mem_wdata,
  input  logic [7:0]   mem_rdata
);
  import cnn_pkg::*;

  typedef enum logic [4:0] {
    ST_IDLE, ST_INIT, ST_ACC, ST_RD_X, ST_RD_X_W, ST_RD_W, ST_RD_W_W, ST_MAC,
    ST_NEXT_IN, ST_RD_B, ST_RD_B_W, ST_BIAS, ST_WR, ST_NEXT, ST_DONE, ST_ERR
  } state_t;

  state_t state;
  logic [15:0] oc, ic;
  logic [1:0]  bidx;
  logic signed [31:0] acc;
  logic signed [7:0] x_s, w_s;
  logic [31:0] bias_word;

  assign busy = (state != ST_IDLE) && (state != ST_DONE) && (state != ST_ERR);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= ST_IDLE; done <= 0; error <= 0;
      mem_req <= 0; mem_we <= 0; mem_addr <= '0; mem_wdata <= '0;
      oc <= 0; ic <= 0; bidx <= 0; acc <= 0; x_s <= 0; w_s <= 0; bias_word <= 0;
    end else begin
      done <= 0; mem_req <= 0; mem_we <= 0;
      unique case (state)
        ST_IDLE: if (start) begin error <= 0; state <= ST_INIT; end
        ST_INIT: begin
          if (c_in == 0 || c_out == 0) state <= ST_ERR;
          else begin oc <= 0; state <= ST_ACC; end
        end
        ST_ACC: begin acc <= 0; ic <= 0; state <= ST_RD_X; end
        ST_RD_X: begin
          if (src_base + 32'(ic) >= SRAM_BYTES) state <= ST_ERR;
          else begin mem_req <= 1; mem_addr <= 19'(src_base + 32'(ic)); state <= ST_RD_X_W; end
        end
        ST_RD_X_W: begin x_s <= $signed(mem_rdata); state <= ST_RD_W; end
        ST_RD_W: begin
          if (wgt_base + 32'(oc) * 32'(c_in) + 32'(ic) >= SRAM_BYTES) state <= ST_ERR;
          else begin
            mem_req <= 1;
            mem_addr <= 19'(wgt_base + 32'(oc) * 32'(c_in) + 32'(ic));
            state <= ST_RD_W_W;
          end
        end
        ST_RD_W_W: begin w_s <= $signed(mem_rdata); state <= ST_MAC; end
        ST_MAC: begin acc <= acc + 32'(x_s) * 32'(w_s); state <= ST_NEXT_IN; end
        ST_NEXT_IN: begin
          if (ic + 1 < c_in) begin ic <= ic + 1; state <= ST_RD_X; end
          else begin bidx <= 0; bias_word <= 0; state <= ST_RD_B; end
        end
        ST_RD_B: begin
          mem_req <= 1;
          mem_addr <= 19'(bias_base + 32'(oc)*32'd4 + 32'(bidx));
          state <= ST_RD_B_W;
        end
        ST_RD_B_W: begin
          bias_word[{bidx, 3'b000} +: 8] <= mem_rdata;
          if (bidx == 2'd3) state <= ST_BIAS;
          else begin bidx <= bidx + 1; state <= ST_RD_B; end
        end
        ST_BIAS: begin acc <= maybe_relu(acc + $signed(bias_word), relu); state <= ST_WR; end
        ST_WR: begin
          mem_req <= 1; mem_we <= 1;
          mem_addr <= 19'(dst_base + 32'(oc));
          mem_wdata <= sat_i8(acc);
          state <= ST_NEXT;
        end
        ST_NEXT: begin
          if (oc + 1 < c_out) begin oc <= oc + 1; state <= ST_ACC; end
          else state <= ST_DONE;
        end
        ST_DONE: begin done <= 1; state <= ST_IDLE; end
        ST_ERR: begin error <= 1; done <= 1; state <= ST_IDLE; end
        default: state <= ST_IDLE;
      endcase
    end
  end
endmodule
