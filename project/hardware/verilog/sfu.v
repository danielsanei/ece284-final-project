// Special Function Unit (SFU)
// Performs accumulation and ReLU operations

module sfu #(
    parameter psum_bw = 16,
    parameter col = 8
) (
    input clk,
    input reset,
    input acc,                                  // accumulation enable signal
    input signed [psum_bw*col-1:0] psum_in,     // PSUM inputs from OFIFO
    output reg [psum_bw*col-1:0] sfp_out        // final output after acc + ReLU
);
    
    // hardware components (2's complement) for each lane (output channel x8)
    reg signed [psum_bw-1:0] accumulator [0:col-1];

    wire signed [psum_bw-1:0] next_sum [0:col-1];

    wire signed [psum_bw-1:0] psum_lanes [0:col-1];

    // create, wire 8 lanes for each PSUM
    genvar g;
    generate
        for (g = 0; g < col; g = g + 1) begin : GEN_LANES
            // assign to unpack input vector per lane
            assign psum_lanes[g] = psum_in[psum_bw*(g+1)-1 : psum_bw*g];

            // accumulate
            assign next_sum[g] = accumulator[g] + psum_lanes[g];
        end
    endgenerate


    // take sum + ReLU of partial sums
    integer i;
    always @(posedge clk) begin
        // clear accumulators, outputs
        if (reset) begin
            for (i = 0; i < col; i = i + 1) begin
                accumulator[i] <= {psum_bw{1'b0}};
                sfp_out[psum_bw*i +: psum_bw] <= {psum_bw{1'b0}};
            end
        end
        // accumulate (add current psum to running total)
	    else begin
            if (acc) begin
                for (i = 0; i < col; i = i + 1) begin
                    accumulator[i] <= next_sum[i];
                end
            end
            else begin
                for (i=0; i < col; i = i +1) begin
                    // ReLU if neg
                    if (next_sum[i][psum_bw-1] == 1'b1) begin
                        sfp_out[psum_bw*i +: psum_bw] <= {psum_bw{1'b0}};
                    end
                    else begin
                        sfp_out[psum_bw*i +: psum_bw] <= next_sum[i];
                    end

                    // clear acc for next
                    accumulator[i] <= {psum_bw{1'b0}};
                end
            end
    	end
    end
endmodule
