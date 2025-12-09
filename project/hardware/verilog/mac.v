// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission 
module mac (out, a, b0, b1, c, mode);

parameter bw = 4;
parameter psum_bw = 16;

output signed [psum_bw-1:0] out;
input signed  [bw-1:0] a;  // activation
input signed  [bw-1:0] b0;  // weight (primary 4-bit or 2-bit weight)
input signed  [bw-1:0] b1;  // weight (secondary 2-bit weight)
input signed  [psum_bw-1:0] c;
input mode;     // 0 = 4-bit mode, 1 = 2-bit mode

// 4-bit mode
wire signed [2*bw:0] product_4bit;
wire signed [bw:0]   a_pad_4bit;

// 2-bit mode (dual MAC)
wire signed [1:0] a_high;                   // upper 2 bits (activation)
wire signed [1:0] a_low;                    // lower 2 bits (activation)
wire signed [2:0] a_high_pad;               // zero-extended for unsigned
wire signed [2:0] a_low_pad;                // ''
wire signed [6:0] product_high;             // 2-bit * 4-bit = 7 bits (signed)
wire signed [6:0] product_low;              // ''
wire signed [psum_bw-1:0] dual_product;

// 4-bit mode computation
assign a_pad_4bit = {1'b0, a}; // force to be unsigned number
assign product_4bit = a_pad_4bit * b0;

// 2-bit mode computation
assign a_high = a[3:2];
assign a_low = a[1:0];
assign a_high_pad = {1'b0, a_high};
assign a_low_pad = {1'b0, a_low};
assign product_high = a_high_pad * b0;
assign product_low = a_low_pad * b1;
assign dual_product = {{9{product_high[6]}}, product_high} + {{9{product_low[6]}}, product_low};

// MUX between modes
wire signed [psum_bw-1:0] mac_result;
assign mac_result = mode ? dual_product : product_4bit[psum_bw-1:0];

// add partial sum
assign out = mac_result + c;

// old code
// assign psum = product + c;
// assign out = psum;
// wire signed [psum_bw-1:0] psum;

endmodule



