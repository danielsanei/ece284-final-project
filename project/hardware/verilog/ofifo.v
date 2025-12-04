// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission 
module ofifo (clk, in, out, rd, wr, o_full, reset, o_ready, o_valid);

  parameter col  = 8;
  parameter bw = 4;

  input  clk;
  input  [col-1:0] wr;
  input  rd;
  input  reset;
  input  [col*bw-1:0] in;
  output [col*bw-1:0] out;
  output o_full;
  output o_ready;
  output o_valid;

  wire [col-1:0] empty;
  wire [col-1:0] full;
  reg [col-1:0] rd_en = 0;
  
  genvar i;

  assign o_ready = ~(full[0] | full[1] | full[2] | full[3] | full[4] | full[5] | full[6] | full[7]) ;
  assign o_full  = full[0] | full[1] | full[2] | full[3] | full[4] | full[5] | full[6] | full[7];
  assign o_valid = ~(empty[0] | empty[1] | empty[2] | empty[3] | empty[4] | empty[5] | empty[6] | empty[7]);

  generate
  for (i=0; i<col ; i=i+1) begin : col_num
      fifo_depth64 #(.bw(bw)) fifo_instance (
     .rd_clk(clk),
     .wr_clk(clk),
     .rd(rd_en[i]),
     .wr(wr[i]),
         .o_empty(empty[i]),
         .o_full(full[i]),
     .in(in[(i+1)*bw-1:i*bw]),
     .out(out[(i+1)*bw-1:i*bw]),
         .reset(reset));
  end
  endgenerate


  always @ (posedge clk) begin
   if (reset) begin
      rd_en <= 8'b00000000;
   end
   else begin
      if(rd==1) begin
        rd_en <= 8'b11111111;
      end
      else begin
        rd_en <= 8'b00000000;
      end
   end
  end


 

endmodule



