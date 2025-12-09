// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission
module mac_row (clk, out_s, in_w, in_n, valid, inst_w, reset, mode);

  parameter bw = 4;
  parameter psum_bw = 16;
  parameter col = 8;

  input  clk, reset;
  input mode;
  output [psum_bw*col-1:0] out_s;
  output [col-1:0] valid;
  input  [bw-1:0] in_w; // inst[1]:execute, inst[0]: kernel loading
  input  [1:0] inst_w;
  input  [psum_bw*col-1:0] in_n;

  // activation propagation (left to right)
  wire [(col+1)*bw-1:0] temp;   	 // bus is connection between PEs
  assign temp[bw-1:0] = in_w;

  // instruction propagation (left to right)
  wire [2*(col+1)-1:0] inst_temp;    // bus is connection between PEs (2 bits per tile)
  assign inst_temp[1:0] = inst_w;

  // for each row, instantiate all tiles (PEs)
  genvar i;
  generate     // wrap in generate block
    for (i=1; i < col+1 ; i=i+1) begin : col_num
      mac_tile #(.bw(bw), .psum_bw(psum_bw)) mac_tile_instance (
        .clk(clk),   				 // global clock
        .reset(reset),   				 // reset signal
        .mode(mode),        // mode signal
      .in_w( temp[bw*i-1:bw*(i-1)]),   		 // activation input (from west)
      .out_e(temp[bw*(i+1)-1:bw*i]),   		 // activation output (to east)
      .inst_w(inst_temp[2*i-1:2*(i-1)]),   	 // input instruction (from west)
      .inst_e(inst_temp[2*(i+1)-1:2*i]),   	 // output instruction (to east)
      .in_n(in_n[psum_bw*i-1:psum_bw*(i-1)]),    // partial sum input (from north)
      .out_s(out_s[psum_bw*i-1:psum_bw*(i-1)])    // partial sum output (to south)
      );

      // set valid bit (output data is ready to be sent, avoid sending every clock cycle when not ready)
      // 2*i is inst_e[0]
      // 2*i+1 is inst_e[1]
      assign valid[i-1] = inst_temp[2*i+1];
    end
  endgenerate

endmodule
