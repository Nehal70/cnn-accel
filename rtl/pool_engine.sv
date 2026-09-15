module pool_engine (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         start,
  output logic         busy,
  output logic         done,
  output logic         error,
  input  logic [7:0]   k,
  input  logic [7:0]   s,
  input  logic         relu,
  input  logic [15:0]  c,
  input  logic [15:0]  h,
  input  logic [15:0]  w,
  input  logic [31:0]  dst_base,
  input  logic [31:0]  src_base,
  output logic         mem_req,
  output logic         mem_we,
  output logic [18:0]  mem_addr,
  output logic [7:0]   mem_wdata,
  input  logic [7:0]   mem_rdata
);
  import cnn_pkg::*;

  typedef enum logic [3:0] {
    ST_IDLE, ST_INIT, ST_PIXEL, ST_TAP, ST_RD, ST_RD_W, ST_MAX,
    ST_NEXT_TAP, ST_WR, ST_NEXT, ST_DONE, ST_ERR
  } state_t;

  state_t state;
  logic [15:0] oc, oy, ox, ky, kx, h_out, w_out;
  logic signed [31:0] mval;
  logic signed [7:0]  v_s;
  logic first;

  assign busy = (state != ST_IDLE) && (state != ST_DONE) && (state != ST_ERR);

  function automatic logic [31:0] src_a(input logic [15:0] yy, xx, cc);
    return src_base + ((32'(yy) * 32'(w) + 32'(xx)) * 32'(c) + 32'(cc));
  endfunction
  function automatic logic [31:0] dst_a();
    return dst_base + ((32'(oy) * 32'(w_out) + 32'(ox)) * 32'(c) + 32'(oc));
  endfunction

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= ST_IDLE; done <= 0; error <= 0;
      mem_req <= 0; mem_we <= 0; mem_addr <= '0; mem_wdata <= '0;
      oc <= 0; oy <= 0; ox <= 0; ky <= 0; kx <= 0; h_out <= 0; w_out <= 0;
      mval <= 0; v_s <= 0; first <= 0;
    end else begin
      done <= 0; mem_req <= 0; mem_we <= 0;
      unique case (state)
        ST_IDLE: if (start) begin error <= 0; state <= ST_INIT; end
        ST_INIT: begin
          if (k == 0 || s == 0 || c == 0 || h == 0 || w == 0 || 16'(h) < 16'(k) || 16'(w) < 16'(k))
            state <= ST_ERR;
          else begin
            h_out <= (16'(h) - 16'(k)) / 16'(s) + 16'd1;
            w_out <= (16'(w) - 16'(k)) / 16'(s) + 16'd1;
            oc <= 0; oy <= 0; ox <= 0;
            state <= ST_PIXEL;
          end
        end
        ST_PIXEL: begin
          ky <= 0; kx <= 0; first <= 1; mval <= 0;
          state <= ST_TAP;
        end
        ST_TAP: begin
          if (src_a(16'(32'(oy)*32'(s)+32'(ky)), 16'(32'(ox)*32'(s)+32'(kx)), oc) >= SRAM_BYTES)
            state <= ST_ERR;
          else begin
            mem_req  <= 1;
            mem_addr <= src_a(16'(32'(oy)*32'(s)+32'(ky)), 16'(32'(ox)*32'(s)+32'(kx)), oc)[18:0];
            state    <= ST_RD_W;
          end
        end
        ST_RD_W: begin
          v_s   <= $signed(mem_rdata);
          state <= ST_MAX;
        end
        ST_MAX: begin
          if (first || 32'(v_s) > mval)
            mval <= 32'(v_s);
          first <= 0;
          state <= ST_NEXT_TAP;
        end
        ST_NEXT_TAP: begin
          if (kx + 1 < {8'd0, k}) begin kx <= kx + 1; state <= ST_TAP; end
          else if (ky + 1 < {8'd0, k}) begin kx <= 0; ky <= ky + 1; state <= ST_TAP; end
          else state <= ST_WR;
        end
        ST_WR: begin
          if (dst_a() >= SRAM_BYTES) state <= ST_ERR;
          else begin
            mem_req   <= 1; mem_we <= 1;
            mem_addr  <= dst_a()[18:0];
            mem_wdata <= sat_i8(maybe_relu(mval, relu));
            state     <= ST_NEXT;
          end
        end
        ST_NEXT: begin
          if (ox + 1 < w_out) begin ox <= ox + 1; state <= ST_PIXEL; end
          else if (oy + 1 < h_out) begin ox <= 0; oy <= oy + 1; state <= ST_PIXEL; end
          else if (oc + 1 < c) begin ox <= 0; oy <= 0; oc <= oc + 1; state <= ST_PIXEL; end
          else state <= ST_DONE;
        end
        ST_DONE: begin done <= 1; state <= ST_IDLE; end
        ST_ERR:  begin error <= 1; done <= 1; state <= ST_IDLE; end
        default: state <= ST_IDLE;
      endcase
    end
  end
endmodule
