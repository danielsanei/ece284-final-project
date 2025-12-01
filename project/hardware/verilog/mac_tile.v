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


// assign all reg to something to prevent flipflop -> latch
always @(posedge clk) begin

    // determine if load required
    if (reset) begin
   	 inst_q <= 2'b00;
   	 load_ready_q <= 1'b1;
   	 a_q <= 0;
   	 b_q <= 0;
   	 c_q <= 0;
    end else begin
   	 // inst[0] kernel loading
	 // inst[1] execution
	 // 00 - idle, a_q and b_q don't change
	 // 01 - weight loading, data always passes in_w->out_e, 
	 //		 if ld_rdy_q tile eat weight, (i need a weight)
	 //			in_w->b_q
	 //		 	send 2'b00 instruc to neighbor PE in inst_e (00 from reset)
	 //		 else ld_rdy_q == 0, (i don't need a weight)
	 //			send inst_w[0] to inst_e through inst_q	(inst_e[0] = 1)
	 //			neighbor can take weight if needs
	 // 10 - execute, inst_w[1] == 1
	 // 	 activation passed to neighbor too (in_w -> out_e through a_q)
	 //		 activation passed to a_q for mac calc (a_q <= in_w)
	 //		 weight (b_q) doesn't change
	 // 11 - conflict/illegal
	 //		 weight AND activation?, (activation/weight)^2 + psum


   	 // accept new instruction
   	 inst_q[1] <= inst_w[1];

   	 // latch new activation
   	 if (inst_w[0] || inst_w[1]) begin
   		 a_q <= in_w;
   	 end

   	 // latch new weight
	 // mac_tile is empty needs new weight
   	 if (inst_w[0] && load_ready_q) begin
   		 b_q <= in_w;
   		 load_ready_q <= 1'b0;
   	 end

   	 // propagate instruction to the next PE
   	 if (!load_ready_q) begin
   		 inst_q[0] <= inst_w[0];
   	 end

	 // cycle in psum from north
	 c_q <= in_n;
    end
end

// connect outputs for next PE
assign out_e = a_q;
assign inst_e = inst_q;
assign out_s = mac_out;

endmodule
