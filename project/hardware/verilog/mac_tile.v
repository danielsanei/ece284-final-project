// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission
module mac_tile (clk, out_s, in_w, out_e, in_n, inst_w, inst_e, reset, mode);

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
input mode;					// 0 = 4-bit mode, 1 = 2-bit mode

// add latches
reg [1:0] inst_q;    // instruction register
reg [bw-1:0] a_q;    // activation
reg [bw-1:0] b_q;    // weight 0 (either 4-bit or 2-bit)
reg [bw-1:0] b_q_1;	 // weight 1 (2-bit mode's secondary weight)
reg [psum_bw-1:0] c_q;    // partial sum
reg load_ready_q;    // indicator --> ready to load new weight
reg load_ready_q_1;    // indicator (second weight) --> ready to load new weight

// MAC result
wire [psum_bw-1:0] mac_out;

mac #(.bw(bw), .psum_bw(psum_bw)) mac_instance (
    	.a(a_q),
    	.b0(b_q),
		.b1(b_q_1),
    	.c(c_q),
		.mode(mode),
    .out(mac_out)
);


// assign all reg to something to prevent flipflop -> latch
always @(posedge clk) begin

    // determine if load required
    if (reset) begin
   	 inst_q <= 2'b00;
   	 load_ready_q <= 1'b1;
	 load_ready_q_1 <= 1'b1;
   	 a_q <= 0;
	 b_q <= 0;
   	 b_q_1 <= 0;
   	 c_q <= 0;
    end else begin
		// load weights (if inst[0] = load)
			// 4-bit mode: load 1 weight per PE
			// 2-bit mode: load 2 weights per PE (need 2 cycles)
		
		// weight loading mode
		if (inst_w[0] && !inst_w[1]) begin

			// ready to load weight(s)
			if (load_ready_q) begin

				// load first weight
				b_q <= in_w;
				load_ready_q <= 1'b0;

				// check if sceond weight needed for 2-bit mode
				if (mode) begin
					load_ready_q_1 <= 1'b1;		// need second weight
					inst_q[0] <= 1'b0;			// no propagate load yet
				end else begin
					load_ready_q_1 <= 1'b0;		// don't need second weight
					inst_q[0] <= inst_w[0];		// propagate (4-bit mode)
				end

			// load second weight
			end else if (load_ready_q_1 && mode) begin
				b_q_1 <= in_w;
				load_ready_q_1 <= 1'b0;
				inst_q[0] <= inst_w[0];			// propagate 2-bit load

			// weights not ready, but propagate
			end else begin
				inst_q[0] <= inst_q[0];
			end

		// execution mode
		end else if (inst_w[1] && !inst_w[0]) begin
			a_q <= in_w;				// laath activation
			inst_q[1] <= inst_w[1];		// propagate execution

		// idle mode
		end else begin
			inst_q <= inst_w;
		end

		// send PSUM from north
		c_q <= in_n;
	end
end

// connect outputs for next PE
assign out_e = a_q;
assign inst_e = inst_q;
assign out_s = mac_out;

endmodule
