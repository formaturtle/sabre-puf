// =====================================================================
// sr_capuf.v - updated to use apuf_sync (synchronous-sampling arbiter)
//
// Changes vs. the Fix-2 version you had before:
//
//   * apuf  ->  apuf_sync.   The new module has a start/valid
//     handshake (see SYNC_ARBITER_GUIDE.md section 3). `start` is a
//     1-cycle pulse; `valid` goes high 8 cycles later when the
//     response is stable and synchronous to sys_clk.
//
//   * Top-level rst (active-high, from the PS) is converted to an
//     active-low rst_n internally for the apuf_sync contract.
//
//   * State machine restructured: IDLE -> WAIT_A (pulsed 3x for the
//     three independent Layer-A votes) -> WAIT_B -> (response+done)
//     -> IDLE. This is Fix 3 applied (three independent races
//     instead of three samples of the same racing signal).
//
//   * Fix 2 (done handshake) is preserved: `done` is cleared when a
//     new trigger arrives, asserted for 1+ cycles after response is
//     captured, and held until the next trigger deasserts it.
//
//   * offsets port on apuf_sync is connected but tied to zero. The
//     apuf_sync reference implementation does not wire offsets into
//     the physical delay chain -- that is Option 2 (LUT-chain
//     programmable delays), which is a separate redesign. Once
//     Option 2 lands, replace the 256'd0 with a real per-instance
//     offset vector derived from bram_seed.
//
//   * offset_rom still computes offset_rom[m] = bram_seed[4:0] ^ m[4:0]
//     so the storage exists; it is not yet consumed.
//
// NOT applied here:
//   * Fix 5 (widen seed slicing beyond 5 bits). Only meaningful once
//     offsets actually drive hardware delay.
// =====================================================================

`timescale 1ns / 1ps

module sr_capuf (
    input         clk,
    input         rst,            // active-high (from PS)
    input  [31:0] challenge,
    input         trigger,
    output reg    response,
    output reg    done,
    input  [31:0] bram_seed       // static seed from BRAM
);
    // -----------------------------------------------------------------
    // Internal active-low reset for the apuf_sync contract
    // -----------------------------------------------------------------
    wire rst_n = ~rst;

    // -----------------------------------------------------------------
    // State encoding
    // -----------------------------------------------------------------
    localparam [2:0] S_IDLE   = 3'd0;
    localparam [2:0] S_WAIT_A = 3'd1;
    localparam [2:0] S_WAIT_B = 3'd2;

    reg [2:0] state;
    reg [1:0] vote_cnt;        // 0..2, indexes s_votes[]
    reg       start_a_reg;     // 1-cycle pulse into Layer-A
    reg       start_b_reg;     // 1-cycle pulse into Layer-B
    reg [31:0] s_maj;
    reg [31:0] s_votes [0:2];
    integer   m;

    // -----------------------------------------------------------------
    // Challenge plumbing
    // -----------------------------------------------------------------
    wire [31:0] c_h;
    wire [31:0] c_prime;

    sparsity_h h_inst (
        .c   (challenge),
        .c_h (c_h)
    );

    assign c_prime = c_h ^ s_maj;

    // -----------------------------------------------------------------
    // Layer A: 32 synchronously-sampled APUFs
    // -----------------------------------------------------------------
    wire [31:0] s_raw;
    wire [31:0] valid_a_vec;
    wire        valid_a = &valid_a_vec;   // all 32 valids high in same cycle

    // BRAM-seeded per-instance offset (held for future Option 2 use;
    // not currently wired into apuf_sync's delay chain).
    reg [4:0] offset_rom [0:31];

    genvar j;
    generate
        for (j = 0; j < 32; j = j + 1) begin : layer_a
            apuf_sync #(.STAGES(32)) inst (
                .clk       (clk),
                .rst_n     (rst_n),
                .start     (start_a_reg),
                .challenge (c_h),
                .offsets   (256'd0),        // placeholder (see header)
                .valid     (valid_a_vec[j]),
                .response  (s_raw[j])
            );
        end
    endgenerate

    // -----------------------------------------------------------------
    // Layer B: 3 synchronously-sampled APUFs, XOR combined
    // -----------------------------------------------------------------
    wire [2:0] y_raw_vec;
    wire [2:0] valid_b_vec;
    wire       valid_b = &valid_b_vec;
    wire       y_raw   = y_raw_vec[0] ^ y_raw_vec[1] ^ y_raw_vec[2];

    genvar k;
    generate
        for (k = 0; k < 3; k = k + 1) begin : layer_b
            apuf_sync #(.STAGES(32)) inst (
                .clk       (clk),
                .rst_n     (rst_n),
                .start     (start_b_reg),
                .challenge (c_prime),
                .offsets   (256'd0),        // placeholder (see header)
                .valid     (valid_b_vec[k]),
                .response  (y_raw_vec[k])
            );
        end
    endgenerate

    // -----------------------------------------------------------------
    // State machine
    //
    // S_IDLE  : wait for trigger. On rising trigger, clear done,
    //           reset vote_cnt, pulse start_a_reg for 1 cycle, and
    //           transition to S_WAIT_A.
    //
    // S_WAIT_A: wait for valid_a (all 32 Layer-A APUFs). When it
    //           pulses, latch s_raw into s_votes[vote_cnt]. If this
    //           was the third vote, compute the bitwise majority
    //           and pulse start_b_reg for the Layer-B race. Else
    //           increment vote_cnt and pulse start_a_reg again.
    //
    // S_WAIT_B: wait for valid_b. When it pulses, capture y_raw
    //           into response, assert done, and return to IDLE.
    //
    // `done` is held high through S_IDLE until the next trigger
    // arrives (Fix 2 handshake: host sees a clean rising edge).
    // -----------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            state       <= S_IDLE;
            vote_cnt    <= 2'd0;
            start_a_reg <= 1'b0;
            start_b_reg <= 1'b0;
            done        <= 1'b0;
            response    <= 1'b0;
            s_maj       <= 32'd0;
            s_votes[0]  <= 32'd0;
            s_votes[1]  <= 32'd0;
            s_votes[2]  <= 32'd0;
            for (m = 0; m < 32; m = m + 1) begin
                offset_rom[m] <= bram_seed[4:0] ^ m[4:0];
            end
        end else begin
            // Default: drop the start pulses each cycle; the case
            // statement re-asserts them for exactly one cycle when
            // launching a new race.
            start_a_reg <= 1'b0;
            start_b_reg <= 1'b0;

            case (state)

                S_IDLE: begin
                    if (trigger) begin
                        done        <= 1'b0;
                        vote_cnt    <= 2'd0;
                        start_a_reg <= 1'b1;     // 1-cycle pulse
                        state       <= S_WAIT_A;
                    end
                end

                S_WAIT_A: begin
                    if (valid_a) begin
                        s_votes[vote_cnt] <= s_raw;
                        if (vote_cnt == 2'd2) begin
                            // Third vote: compute majority using
                            // s_raw directly (its nonblocking store
                            // into s_votes[2] hasn't taken effect yet).
                            for (m = 0; m < 32; m = m + 1) begin
                                s_maj[m] <= ((s_votes[0][m] +
                                              s_votes[1][m] +
                                              s_raw[m]) >= 2);
                            end
                            start_b_reg <= 1'b1;     // launch Layer-B
                            state       <= S_WAIT_B;
                        end else begin
                            vote_cnt    <= vote_cnt + 2'd1;
                            start_a_reg <= 1'b1;     // next Layer-A race
                            // remain in S_WAIT_A
                        end
                    end
                end

                S_WAIT_B: begin
                    if (valid_b) begin
                        response <= y_raw;
                        done     <= 1'b1;
                        state    <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
