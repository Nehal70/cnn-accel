module netparser (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         go,
  input  logic [31:0]  program_base,
  output logic         done,
  output logic         error,
  output logic [31:0]  pc,

  output logic         conv_start,
  output logic         pool_start,
  output logic         fc_start,
  output logic         add_start,
  output logic         dma_start,
  output logic         dma_store,
  input  logic         conv_done,
  input  logic         pool_done,
  input  logic         fc_done,
  input  logic         add_done,
  input  logic         dma_done,
  input  logic         conv_err,
  input  logic         pool_err,
  input  logic         fc_err,
  input  logic         add_err,
  input  logic         dma_err,

  output logic [7:0]   k, s, pad,
  output logic         relu,
  output logic [15:0]  c_in, c_out, h, w,
  output logic [31:0]  addr0, addr1, addr2, addr3,

  output logic         mem_req,
  output logic         mem_we,
  output logic [18:0]  mem_addr,
  output logic [7:0]   mem_wdata,
  input  logic [7:0]   mem_rdata
);
  import cnn_pkg::*;

  typedef enum logic [3:0] {
    ST_IDLE, ST_FETCH, ST_FETCH_W, ST_DECODE, ST_DISPATCH, ST_WAIT, ST_DONE, ST_ERR
  } state_t;

  state_t state;
  logic [5:0]  bi;
  logic [7:0]  instr [0:31];
  logic [7:0]  opcode;
  logic        started;

  assign mem_we    = 1'b0;
  assign mem_wdata = 8'd0;

  function automatic logic [15:0] u16(input int lo);
    return {instr[lo+1], instr[lo]};
  endfunction
  function automatic logic [31:0] u32(input int lo);
    return {instr[lo+3], instr[lo+2], instr[lo+1], instr[lo]};
  endfunction

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= ST_IDLE; done <= 0; error <= 0; pc <= 0;
      mem_req <= 0; mem_addr <= '0; bi <= 0; opcode <= 0; started <= 0;
      conv_start <= 0; pool_start <= 0; fc_start <= 0; add_start <= 0;
      dma_start <= 0; dma_store <= 0;
      k <= 0; s <= 0; pad <= 0; relu <= 0;
      c_in <= 0; c_out <= 0; h <= 0; w <= 0;
      addr0 <= 0; addr1 <= 0; addr2 <= 0; addr3 <= 0;
    end else begin
      done <= 0;
      mem_req <= 0;
      conv_start <= 0; pool_start <= 0; fc_start <= 0; add_start <= 0; dma_start <= 0;

      unique case (state)
        ST_IDLE: if (go) begin
          error <= 0; pc <= program_base; bi <= 0; state <= ST_FETCH;
        end
        ST_FETCH: begin
          mem_req  <= 1;
          mem_addr <= 19'(pc + 32'(bi));
          state    <= ST_FETCH_W;
        end
        ST_FETCH_W: begin
          instr[bi] <= mem_rdata;
          if (bi == 6'd31) state <= ST_DECODE;
          else begin bi <= bi + 1; state <= ST_FETCH; end
        end
        ST_DECODE: begin
          opcode <= instr[0];
          relu   <= instr[1][0];
          k      <= instr[2];
          s      <= instr[3];
          pad    <= instr[4];
          c_in   <= {instr[7], instr[6]};
          c_out  <= {instr[9], instr[8]};
          h      <= {instr[11], instr[10]};
          w      <= {instr[13], instr[12]};
          addr0  <= {instr[19], instr[18], instr[17], instr[16]};
          addr1  <= {instr[23], instr[22], instr[21], instr[20]};
          addr2  <= {instr[27], instr[26], instr[25], instr[24]};
          addr3  <= {instr[31], instr[30], instr[29], instr[28]};
          state  <= ST_DISPATCH;
        end
        ST_DISPATCH: begin
          started <= 1;
          unique case (opcode)
            OP_STOP:  state <= ST_DONE;
            OP_LOAD:  begin dma_store <= 0; dma_start <= 1; state <= ST_WAIT; end
            OP_STORE: begin dma_store <= 1; dma_start <= 1; state <= ST_WAIT; end
            OP_CONV:  begin conv_start <= 1; state <= ST_WAIT; end
            OP_POOL:  begin pool_start <= 1; state <= ST_WAIT; end
            OP_FC:    begin fc_start <= 1; state <= ST_WAIT; end
            OP_ADD:   begin add_start <= 1; state <= ST_WAIT; end
            default:  state <= ST_ERR;
          endcase
        end
        ST_WAIT: begin
          started <= 0;
          if (conv_err | pool_err | fc_err | add_err | dma_err) state <= ST_ERR;
          else if (conv_done | pool_done | fc_done | add_done | dma_done) begin
            pc    <= pc + 32'd32;
            bi    <= 0;
            state <= ST_FETCH;
          end
        end
        ST_DONE: begin done <= 1; state <= ST_IDLE; end
        ST_ERR:  begin error <= 1; done <= 1; state <= ST_IDLE; end
        default: state <= ST_IDLE;
      endcase
    end
  end
endmodule
