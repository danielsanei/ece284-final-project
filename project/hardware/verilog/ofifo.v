// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission 
module ofifo (clk, in, out, rd, wr, o_full, reset, o_ready, o_valid);

  parameter col  = 8;
  parameter bw = 4;

  input  clk;
  input  [col-1:0] wr;		// 8-bit vector, per column (write enable)
  input  rd;			// read request (read from all FIFOs at once)
  input  reset;
  input  [col*bw-1:0] in;	// 32-bit input (8 columns * 4 bits each)
  output [col*bw-1:0] out;	// 32-bit output
  output o_full;		// if any FIFO is full
  output o_ready;		// if all FIFOs have room
  output o_valid;		// all columns have at least 1 data word (ready to read)

  wire [col-1:0] empty;		// 8-bit vector for each column
  wire [col-1:0] full;		// 8-bit vector for each column
  reg  rd_en;			// internal register (read all columns)
  
  genvar i;

  assign o_ready = |(~full);	// not all columns full (at least 1 has room)
  assign o_full  = |full;	// at least 1 column is full
  assign o_valid = ~empty[col-1];	// at least one full vector is ready (not all empty)

  generate     // wrap in generate block
    for (i=0; i<col ; i=i+1) begin : col_num
        fifo_depth64 #(.bw(bw)) fifo_instance (
    .rd_clk(clk),
    .wr_clk(clk),
    .rd(rd_en),
    .wr(wr[i]),		// each col has its own write enable
          .o_empty(empty[i]),
          .o_full(full[i]),
    .in(in[bw*(i+1)-1:bw*i]),
    .out(out[bw*(i+1)-1:bw*i]),
          .reset(reset));
    end
  endgenerate


  always @ (posedge clk) begin
   if (reset) begin
      rd_en <= 0;
   end
   else
      
     // read all columns at once
     if (rd) begin
	     rd_en <= 1'b1;
     end else begin
	     rd_en <= 1'b0;
     end
 
  end


 

endmodule
