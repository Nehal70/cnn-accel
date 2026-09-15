module conv_engine (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         start,
  output logic         busy,
  output logic         done,
  output logic         error,

  input  logic [7:0]   k,
  input  logic [7:0]   s,
  input  logic [7:0]   pad,
  input  logic         relu,
  input  logic [15:0]  c_in,
  input  logic [15:0]  c_out,
  input  logic [15:0]  h,
  input  logic [15:0]  w,
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
    ST_IDLE,
    ST_INIT,
    ST_PIXEL,
    ST_TAP_CHECK,
    ST_RD_ACT,
    ST_RD_ACT_W,
    ST_RD_WGT,
    ST_RD_WGT_W,
    ST_MAC,
    ST_NEXT_TAP,
    ST_RD_BIAS,
    ST_RD_BIAS_W,
    ST_BIAS_ACC,
    ST_WR_OUT,
    ST_NEXT_PIX,
    ST_DONE,
    ST_ERR
  } state_t;

  state_t state;

  logic [15:0] oc, oy, ox, ic, ky, kx;
  logic [15:0] h_out, w_out;
  logic [1:0]  bidx;
  logic signed [31:0] acc;
  logic signed [31:0] iy, ix;
  logic signed [7:0]  a_s, w_s;
  logic [31:0] bias_word;

  assign busy = (state != ST_IDLE) && (state != ST_DONE) && (state != ST_ERR);

  function automatic logic [31:0] act_addr(input logic [15:0] yy, xx, cc);
    return src_base + ((32'(yy) * 32'(w) + 32'(xx)) * 32'(c_in) + 32'(cc));
  endfunction

  function automatic logic [31:0] wgt_addr();
    return wgt_base
      + (((32'(oc) * 32'(c_in) + 32'(ic)) * 32'(k) + 32'(ky)) * 32'(k) + 32'(kx));
  endfunction

  function automatic logic [31:0] dst_addr();
    return dst_base + ((32'(oy) * 32'(w_out) + 32'(ox)) * 32'(c_out) + 32'(oc));
  endfunction

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state     <= ST_IDLE;
      done      <= 1'b0;
      error     <= 1'b0;
      mem_req   <= 1'b0;
      mem_we    <= 1'b0;
      mem_addr  <= '0;
      mem_wdata <= '0;
      oc <= '0; oy <= '0; ox <= '0; ic <= '0; ky <= '0; kx <= '0;
      h_out <= '0; w_out <= '0;
      acc <= '0; iy <= '0; ix <= '0;
      bidx <= '0; bias_word <= '0;
      a_s <= '0; w_s <= '0;
    end else begin
      done    <= 1'b0;
      mem_req <= 1'b0;
      mem_we  <= 1'b0;

      unique case (state)
        ST_IDLE: begin
          error <= 1'b0;
          if (start) state <= ST_INIT;
        end

        ST_INIT: begin
          if (k == 0 || s == 0 || c_in == 0 || c_out == 0 || h == 0 || w == 0)
            state <= ST_ERR;
          else if (16'(h) + 16'(pad) * 16'd2 < 16'(k))
            state <= ST_ERR;
          else begin
            h_out <= (16'(h) + 16'(pad) * 16'd2 - 16'(k)) / 16'(s) + 16'd1;
            w_out <= (16'(w) + 16'(pad) * 16'd2 - 16'(k)) / 16'(s) + 16'd1;
            oc <= '0; oy <= '0; ox <= '0;
            state <= ST_PIXEL;
          end
        end

        ST_PIXEL: begin
          acc <= 32'sd0;
          ic <= '0; ky <= '0; kx <= '0;
          state <= ST_TAP_CHECK;
        end

        ST_TAP_CHECK: begin
          iy <= $signed(32'(oy)) * $signed(32'(s)) - $signed(32'(pad)) + $signed(32'(ky));
          ix <= $signed(32'(ox)) * $signed(32'(s)) - $signed(32'(pad)) + $signed(32'(kx));
          state <= ST_RD_ACT;
        end

        ST_RD_ACT: begin
          if (iy < 0 || ix < 0 || iy >= $signed(32'(h)) || ix >= $signed(32'(w)))
            state <= ST_NEXT_TAP;
          else if (act_addr(iy[15:0], ix[15:0], ic) >= SRAM_BYTES)
            state <= ST_ERR;
          else begin
            mem_req  <= 1'b1;
            mem_addr <= act_addr(iy[15:0], ix[15:0], ic)[18:0];
            state    <= ST_RD_ACT_W;
          end
        end

        ST_RD_ACT_W: begin
          a_s   <= $signed(mem_rdata);
          state <= ST_RD_WGT;
        end

        ST_RD_WGT: begin
          if (wgt_addr() >= SRAM_BYTES) state <= ST_ERR;
          else begin
            mem_req  <= 1'b1;
            mem_addr <= wgt_addr()[18:0];
            state    <= ST_RD_WGT_W;
          end
        end

        ST_RD_WGT_W: begin
          w_s   <= $signed(mem_rdata);
          state <= ST_MAC;
        end

        ST_MAC: begin
          acc   <= acc + 32'(a_s) * 32'(w_s);
          state <= ST_NEXT_TAP;
        end

        ST_NEXT_TAP: begin
          if (kx + 16'd1 < {8'd0, k}) begin
            kx    <= kx + 16'd1;
            state <= ST_TAP_CHECK;
          end else if (ky + 16'd1 < {8'd0, k}) begin
            kx    <= '0;
            ky    <= ky + 16'd1;
            state <= ST_TAP_CHECK;
          end else if (ic + 16'd1 < c_in) begin
            kx    <= '0;
            ky    <= '0;
            ic    <= ic + 16'd1;
            state <= ST_TAP_CHECK;
          end else begin
            bidx      <= 2'd0;
            bias_word <= '0;
            state     <= ST_RD_BIAS;
          end
        end

        ST_RD_BIAS: begin
          if (bias_base + 32'(oc) * 32'd4 + 32'(bidx) >= SRAM_BYTES) state <= ST_ERR;
          else begin
            mem_req  <= 1'b1;
            mem_addr <= 19'(bias_base + 32'(oc) * 32'd4 + 32'(bidx));
            state    <= ST_RD_BIAS_W;
          end
        end

        ST_RD_BIAS_W: begin
          bias_word[{bidx, 3'b000} +: 8] <= mem_rdata;
          if (bidx == 2'd3) state <= ST_BIAS_ACC;
          else begin
            bidx  <= bidx + 2'd1;
            state <= ST_RD_BIAS;
          end
        end

        ST_BIAS_ACC: begin
          acc   <= maybe_relu(acc + $signed(bias_word), relu);
          state <= ST_WR_OUT;
        end

        ST_WR_OUT: begin
          if (dst_addr() >= SRAM_BYTES) state <= ST_ERR;
          else begin
            mem_req   <= 1'b1;
            mem_we    <= 1'b1;
            mem_addr  <= dst_addr()[18:0];
            mem_wdata <= sat_i8(acc);
            state     <= ST_NEXT_PIX;
          end
        end

        ST_NEXT_PIX: begin
          if (ox + 16'd1 < w_out) begin
            ox    <= ox + 16'd1;
            state <= ST_PIXEL;
          end else if (oy + 16'd1 < h_out) begin
            ox    <= '0;
            oy    <= oy + 16'd1;
            state <= ST_PIXEL;
          end else if (oc + 16'd1 < c_out) begin
            ox    <= '0;
            oy    <= '0;
            oc    <= oc + 16'd1;
            state <= ST_PIXEL;
          end else state <= ST_DONE;
        end

        ST_DONE: begin
          done  <= 1'b1;
          state <= ST_IDLE;
        end

        ST_ERR: begin
          error <= 1'b1;
          done  <= 1'b1;
          state <= ST_IDLE;
        end

        default: state <= ST_IDLE;
      endcase
    end
  end
endmodule
