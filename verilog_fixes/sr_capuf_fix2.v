// =====================================================================
// sr_capuf.v — with Fix 2 applied (done handshake)
//
// Change summary vs. the uploaded version:
//   * In the state==0 branch of the case statement, when trigger is
//     sampled high we now also assert:
//         done     <= 1'b0;   // clear stale done from prior query
//         vote_cnt <= 2'd0;   // defensive reset of vote counter
//     These three lines are the entire Fix 2 delta.
//
// NOT applied here:
//   * Fix 3 (three independent APUF races for Layer-A votes).
//   * Fix 5 (widen the seed-to-offsets slicing beyond bram_seed[4:0]).
//
// Simulation: pass the file to tb_sr_capuf.v and expect
//   PASS: handshake
//   FAIL: three distinct votes    (still failing until Fix 3)
//   FAIL: seed affects output     (still failing until Fix 5, because
//                                  only 5 bits of the seed feed the
//                                  offsets and they all collapse to
//                                  'seed[4:0] XOR m[4:0]')
// =====================================================================

module sr_capuf (
    input clk,
    input rst,
    input [31:0] challenge,
    input trigger,
    output reg response,
    output reg done,
    input [31:0] bram_seed     // Static seed from BRAM
);
    reg [1:0] state;
    reg [31:0] s_maj, s_votes[2:0];
    reg [1:0] vote_cnt;
    wire [31:0] c_h, c_prime;
    wire [31:0] s_raw;
    wire [2:0] y_raw_vec;
    wire y_raw;

    // Pre-computed offsets from BRAM seed (stable after reset)
    reg [4:0] offset [0:31];

    // Sparsity H
    sparsity_h h_inst (
        .c(challenge),
        .c_h(c_h)
    );

    // Layer A: 32 APUFs with BRAM-seeded offsets
    genvar j;
    generate
        for (j = 0; j < 32; j = j + 1) begin : layer_a
            apuf inst (
                .clk(clk),
                .rst(rst),
                .start(state == 1),
                .challenge(c_h),
                .offset(offset[j]),
                .response(s_raw[j])
            );
        end
    endgenerate

    // Remapping
    assign c_prime = c_h ^ s_maj;

    // Layer B: 3-XOR APUFs (SABRE-style)
    genvar k;
    generate
        for (k = 0; k < 3; k = k + 1) begin : layer_b
            apuf inst (
                .clk(clk),
                .rst(rst),
                .start(state == 2),
                .challenge(c_prime),
                .offset(5'd0),
                .response(y_raw_vec[k])
            );
        end
    endgenerate
    assign y_raw = y_raw_vec[0] ^ y_raw_vec[1] ^ y_raw_vec[2];

    // State Machine
    integer m;   // DECLARED AT MODULE LEVEL (fixes the unnamed block error)

    always @(posedge clk) begin
        if (rst) begin
            state    <= 2'd0;
            vote_cnt <= 2'd0;
            done     <= 1'b0;
            response <= 1'b0;
            // Sample BRAM seed once at reset and compute offsets
            for (m = 0; m < 32; m = m + 1) begin
                offset[m] <= bram_seed[4:0] ^ m[4:0];
            end
        end else case (state)
            // -------------------------------------------------------------
            // Fix 2: when a new trigger arrives, drop any stale `done`
            // from the previous measurement and reset the vote counter.
            // This gives the host a clean rising-edge handshake.
            // -------------------------------------------------------------
            2'd0: if (trigger) begin
                done     <= 1'b0;
                vote_cnt <= 2'd0;
                state    <= 2'd1;
            end

            2'd1: begin
                s_votes[vote_cnt] <= s_raw;
                if (vote_cnt == 2'd2) begin
                    for (m = 0; m < 32; m = m + 1)
                        s_maj[m] <= (s_votes[0][m] + s_votes[1][m] + s_votes[2][m]) >= 2;
                    state <= 2'd2;
                end else begin
                    vote_cnt <= vote_cnt + 2'd1;
                end
            end

            2'd2: begin
                response <= y_raw;
                done     <= 1'b1;
                state    <= 2'd0;
            end

            default: state <= 2'd0;
        endcase
    end
endmodule
