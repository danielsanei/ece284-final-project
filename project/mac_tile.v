// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission
module mac_tile (clk, out_s, in_w, out_e, in_n, inst_w, inst_e, reset);

parameter bw = 4;
parameter psum_bw = 16;

// inst[1]: execute, inst[0]: kernel loading
output [psum_bw-1:0] out_s;    // activation output to south
input  [bw-1:0] in_w;    	 // west input (weight)
output [bw-1:0] out_e;    	 // activation output to east
input  [1:0] inst_w;   	 // instruction input from west
output [1:0] inst_e;   	 // instruction output to east
input  [psum_bw-1:0] in_n;    // north input (activation)
input  clk;
input  reset;

// add latches
reg [1:0] inst_q;    // instruction register
reg [bw-1:0] a_q;    // activation
reg [bw-1:0] b_q;    // weight
reg [psum_bw-1:0] c_q;    // partial sum
reg load_ready_q;    // indicator --> ready to load new weight

// MAC result
wire [psum_bw-1:0] mac_out;

mac #(.bw(bw), .psum_bw(psum_bw)) mac_instance (
    	.a(a_q),
    	.b(b_q),
    	.c(c_q),
    .out(mac_out)
);

always @(posedge clk) begin

    // determine if load required
    if (reset) begin
   	 inst_q <= 2'b00;
   	 load_ready_q <= 1'b1;
   	 a_q <= 0;
   	 b_q <= 0;
   	 c_q <= 0;
    end else begin
   	 
   	 // accept new instruction
   	 inst_q[1] <= inst_w[1];

   	 // latch new activation
   	 if (inst_w[0] || inst_w[1]) begin
   		 a_q <= in_w;
   	 end

   	 // latch new weight
   	 if (inst_w[0] && load_ready_q) begin
   		 b_q <= in_w;
   		 load_ready_q <= 1'b0;
   	 end

   	 // propagate instruction to the next PE
   	 if (!load_ready_q) begin
   		 inst_q[0] <= inst_w[0];
   	 end
    end
end

// connect outputs for next PE
assign out_e = a_q;
assign out_s = c_q;
assign inst_e = inst_q;


endmodule

