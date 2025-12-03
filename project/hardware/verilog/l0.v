// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission 
module l0 (clk, in, out, rd, wr, o_full, reset, o_ready, cascade);

  parameter row  = 8;
  parameter bw = 4;

  input  clk;
  input  wr;
  input  rd;
  input  reset;
  input cascade;
  input  [row*bw-1:0] in;
  output [row*bw-1:0] out;
  output o_full;
  output o_ready;

  wire [row-1:0] empty;
  wire [row-1:0] full;
  reg [row-1:0] rd_en;
  
  genvar i;
 
  assign o_full  = |full;	// at least 1 row is full
  assign o_ready = ~(&full);	// room to write a new vector, if at least one not full

  // instantiate 8 FIFO rows
  generate     // wrap in generate block
    for (i=0; i<row ; i=i+1) begin : row_num
        fifo_depth64 #(.bw(bw)) fifo_instance (
    .rd_clk(clk),
    .wr_clk(clk),
    .rd((cascade ? rd & rd_en[i] : rd)),
    .wr(wr),
          .o_empty(empty[i]),
          .o_full(full[i]),
    .in(in[bw*(i+1)-1 : bw*i]),	// high bit index = bw*(i+1)-1
    .out(out[bw*(i+1)-1 : bw*i]),	// low bit index = bw*i
          .reset(reset));
    end
  endgenerate

/*
  always @ (posedge clk) begin
   if (reset) begin
      rd_en <= 8'b00000000;
   end
   else begin
*/
      /////////////// version1: read all row at a time ////////////////
      /*
      if (rd) begin   		 // read request signal
      	rd_en <= 8'b11111111;    // read all 8 rows
      end else begin
      	rd_en <= 8'b00000000;
      end
      */
      ///////////////////////////////////////////////////////

      //////////////// version2: read 1 row at a time /////////////////
      
      // version2: read 1 row at a time by rotating a one-hot rd_en
      always @ (posedge clk) begin
        if (reset) begin
      	  rd_en <= 8'b00000001;                  // start from row 0
        end else begin
          if (cascade) begin
            if (rd_en == 8'b00000000 || rd_en == 8'b11111111) begin // check for non-cascade residual
              if (rd) rd_en <= 8'b00000010;
              else rd_en <= 8'b00000001;
            end
            else begin
              if (rd) begin                          // on a read request
                rd_en <= {rd_en[6:0], rd_en[7]};     // rotate one-hot left
              end else begin
                rd_en <= rd_en;                      // keep last active row
              end
            end
          end
          else begin
            if (rd) begin   		 // read request signal
              rd_en <= {row{1'b1}};    // read all 8 rows
            end else begin
              rd_en <= 8'b00000000;
            end
          end
        end
       end
            
      ///////////////////////////////////////////////////////
  /* end
  end
*/
endmodule
