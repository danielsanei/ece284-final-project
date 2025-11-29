// Core
// Top-level design: SRAMs + corelet.v (L0, MAC Array, OFIFO, SFU)
// Input Memory --> Corelet --> Output Memory

module core #(
    parameter bw = 4,
    parameter row = 8,
    parameter col = 8,
    parameter psum_bw = 16,
    parameter addr_width = 11           // number of bits in SRAM (both xmem and pmem)
) (
    input clk,
    input reset,
    input [33:0] inst,                  // bundled instructions from testbench
    input [bw*row-1:0] D_xmem,          // write data from testbench into xmem
    output ofifo_valid,
    output [psum_bw*col-1:0] sfp_out    // accumulate + ReLU result
);

    // extract individual instructions
    wire [addr_width-1:0] A_xmem;       // input (activation/weight) SRAM address
    wire CEN_xmem;                      // chip enable (0: read/write, 1: idle) per cycle
    wire WEN_xmem;                      // write enable (0: write D_xmem into A_xmem, 1: idle) per cycle
    wire [addr_width-1:0] A_pmem;       // output (PSUM/output) SRAM address
    wire CEN_pmem;                      // chip enable (0: read/write, idle) per cycle
    wire WEN_pmem;                      // write enable (0: write to pmem, 1: read or idle) per cycle
    assign A_xmem = inst[17:7];
    assign CEN_xmem = inst[19];
    assign WEN_xmem = inst[18];
    assign A_pmem = inst[30:20];
    assign CEN_pmem = inst[32];
    assign WEN_pmem = inst[31];

    // --------------------------------------------------------------------------
    // Input Memory (xmem)
    // --------------------------------------------------------------------------
    //  - store activations/weights
    // --------------------------------------------------------------------------
    wire [bw*row-1:0] xmem_q;   // next activation/weight out of input memory
    sram_32b_w2048 sram_inst(
        .CLK (clk),
        .WEN (WEN_xmem),        // write enable
        .CEN (CEN_xmem),        // chip enable
        .D (D_xmem),            // write data from testbench
        .A (A_xmem),            // value in xmem to read/write
        .Q (xmem_q)             // read data to corelet.v
    );

    // --------------------------------------------------------------------------
    // Corelet
    // --------------------------------------------------------------------------
    //  - performs computation: L0 FIFO --> MAC Array --> OFIFO --> SFU
    // --------------------------------------------------------------------------
    corelet #(
        .bw (bw),
        .psum_bw (psum_bw),
        .col (col),
        .row (row)
    ) corelet_inst(
        .clk (clk),
        .reset (reset),
        .inst (inst),                   // bundled instructions from testbench
        .D_xmem (xmem_q),               // write data from testbench into xmem
        .sfp_out (sfp_out),             // accumulate + ReLU result
        .ofifo_valid (ofifo_valid)
    );

    // --------------------------------------------------------------------------
    // Output Memory (pmem)
    // --------------------------------------------------------------------------
    //  - store SFU results (PSUMs)
    // --------------------------------------------------------------------------
    localparam PMEM_DEPTH = (1 << addr_width);      // # of entries in pmem

    // output SRAM model
    reg [psum_bw*col-1:0] pmem [0:PMEM_DEPTH-1];    // storage array (each entry holds 8-channel output vector)
    reg [psum_bw*col-1:0] pmem_q;                   // read data out of pmem

    // synchronous read/write to pmem
    always @(posedge clk) begin
        // if pmem enabled this cycle
        if (!CEN_pmem) begin
            // if write enabled this cycle
            if (!WEN_pmem) begin
                // store current SFU results
                pmem[A_pmem] <= sfp_out;
            end
            // otherwise, read enabled this cycle
            else begin
                pmem_q <= pmem[A_pmem];
            end
        end
    end

endmodule