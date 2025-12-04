// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission 
module l0 (clk, in, out, rd, wr, o_full, reset, o_ready);

  parameter row  = 8;
  parameter bw = 4;

  input  clk;
  input  wr;
  input  rd;
  input  reset;
  input  [row*bw-1:0] in;
  output [row*bw-1:0] out;
  output o_full;
  output o_ready;

  wire [row-1:0] empty;
  wire [row-1:0] full;
  reg [row-1:0] rd_en;

  assign o_ready =  !(full[0] | full[1] | full[2] | full[3] | full[4] | full[5] | full[6] | full[7]);
  assign o_full  =  full[0] | full[1] | full[2] | full[3] | full[4] | full[5] | full[6] | full[7];

  genvar i;
  generate
  for (i=0; i<row ; i=i+1) begin : row_num
      fifo_depth64 #(.bw(bw)) fifo_instance (
     .rd_clk(clk),
     .wr_clk(clk),
     .rd(rd_en[i]),
     .wr(wr),
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
   else
      if(rd==1) begin
        case(rd_en)
          8'b00000000: rd_en<=8'b00000001;
          8'b00000001: rd_en<=8'b00000011;
          8'b00000011: rd_en<=8'b00000111;
          8'b00000111: rd_en<=8'b00001111;
          8'b00001111: rd_en<=8'b00011111;
          8'b00011111: rd_en<=8'b00111111;
          8'b00111111: rd_en<=8'b01111111;
          8'b01111111: rd_en<=8'b11111111;
          8'b11111111: rd_en<=8'b11111111;
          default : rd_en<=8'b00000000;
        endcase
      end
      else begin
        case(rd_en)
          8'b11111111: rd_en<=8'b11111110;
          8'b11111110: rd_en<=8'b11111100;
          8'b11111100: rd_en<=8'b11111000;
          8'b11111000: rd_en<=8'b11110000;
          8'b11110000: rd_en<=8'b11100000;
          8'b11100000: rd_en<=8'b11000000;
          8'b11000000: rd_en<=8'b10000000;
          8'b10000000: rd_en<=8'b00000000;
          default : rd_en<=8'b00000000;
        endcase
      end
  end

endmodule


