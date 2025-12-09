// Corelet
// Performs computation: L0 FIFO --> MAC Array --> OFIFO --> SFU

module corelet #(
    parameter bw = 4,
    parameter psum_bw = 16,
    parameter col = 8,
    parameter row = 8
) (
    input clk,
    input reset,
    input [35:0] inst,                  // bundled instructions from testbench
    input [bw*row-1:0] D_xmem,          // write data from testbench into xmem
    input [psum_bw*col-1:0] D_pmem,     // read PSUMs from PMEM to SFU
    output [psum_bw*col-1:0] sfp_out,   // accumulate + ReLU result
    output ofifo_valid
);

    // extract individual instructions
    wire mode = inst[35];       // 4-bit or 2-bit mode
    wire pmem_rd = inst[34];
    wire bypass = inst[34];
    wire acc = inst[33];        // SFU accumulator (1 = continue acc, 0 = ReLU + clear acc)
    // inst[32:20] : pmem controls (core.v)
    // inst[19:7] : xmem controls (core.v)
    wire ofifo_rd = inst[6];    // read-enable (1 = output next PSUM to SFU)
    wire ififo_wr = inst[5];
    wire ififo_rd = inst[4];
    wire l0_rd = inst[3];       // load data from L0 --> MAC array (west inputs)
    wire l0_wr = inst[2];       // write data from memory (D_xmem) to L0
    wire execute = inst[1];     // compute convolution (MAC + PSUM)
    wire load = inst[0];        // load weights

    // for mac_array (inst_w[1] = execute, inst_w[0] = load)
    wire [1:0] inst_w = {execute, load};
    
    // --------------
    // connect blocks
    // --------------

    // L0 --> MAC Array
    wire [bw*row-1:0] l0_out;
    wire l0_o_full;
    wire l0_o_ready;

    // MAC Array --> OFIFO
    wire [psum_bw*col-1:0] mac_out;
    wire [col-1:0] mac_valid;

    // OFIFO --> SFU
    wire [psum_bw*col-1:0] ofifo_out;
    wire ofifo_full;
    wire ofifo_ready;

    // -------------------------------------------------------------------------
    // L0 FIFO
    // -------------------------------------------------------------------------
    //  - create L0 block: buffers vectors from xmem, outputs them to MAC array
    // -------------------------------------------------------------------------
    l0 #(
        .row (row),
        .bw (bw)
    ) l0_inst (
        .clk (clk),
        .reset (reset),
        .in (D_xmem),
        .rd(l0_rd),
        .wr (l0_wr),
        .out (l0_out),
        .o_full (l0_o_full),
        .o_ready (l0_o_ready)
    );

    // --------------------------------------------------------------------------
    // MAC Array
    // --------------------------------------------------------------------------
    //  - creates 8x8 array of MAC tiles (PEs), performs convolution computation
    // --------------------------------------------------------------------------
    mac_array #(
        .bw (bw),
        .psum_bw (psum_bw),
        .col (col),
        .row (row)
    ) mac_array_inst (
        .clk (clk),
        .reset (reset),
        .mode(mode),
        .out_s (mac_out),
        .in_w (l0_out),
        .in_n ({psum_bw*col{1'b0}}),
        .inst_w (inst_w),
        .valid (mac_valid)
    );

    // -------------------------------------------------------------------------
    // OFIFO
    // -------------------------------------------------------------------------
    //  - creates output FIFO, writes when MAC says outputs are valid
    //  - buffers psum vectors (fed into SFU at controlled pace)
    // -------------------------------------------------------------------------
    ofifo #(
        .col (col),
        .bw (psum_bw)
    ) ofifo_inst (
        .clk (clk),
        .reset (reset),
        .wr (mac_valid), 
        .rd (ofifo_rd),
        .in (mac_out),
        .out (ofifo_out),
        .o_full (ofifo_full),
        .o_ready (ofifo_ready),
        .o_valid (ofifo_valid)
    );

    // --------------------------------------------------------------------------------
    // SFU
    // --------------------------------------------------------------------------------
    //  - special function unit: receives PSUMs from OFIFO, performs accumulate + ReLU
    // --------------------------------------------------------------------------------
    wire [psum_bw*col-1:0] sfu_psum_in;
    assign sfu_psum_in = bypass ? ofifo_out : D_pmem;
    sfu #(
        .psum_bw (psum_bw),
        .col (col)
    ) sfu_inst (
        .clk (clk),
        .reset (reset),
        .bypass (bypass),
        .acc (acc),
        .psum_in (sfu_psum_in),
        .sfp_out (sfp_out)
    );

endmodule
