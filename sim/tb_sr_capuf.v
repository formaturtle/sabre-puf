// =====================================================================
// tb_sr_capuf.v
//
// Behavioral testbench for the (post-fix) SR-CAPUF top module.
// This file is a NEW test harness, not a modification of the DUT.
//
// It exercises three behaviors that your three fixes must produce:
//
//   CHECK_1  Handshake: `done` must return to 0 when a new trigger
//            arrives, and must rise to 1 when a measurement completes.
//            Multiple back-to-back queries must each produce a clean
//            0 -> 1 transition on `done`.
//
//   CHECK_2  Vote independence: on each of the three Layer-A votes,
//            the `start` signal to the Layer-A APUFs must transition
//            0 -> 1 with enough quiet time in between that the
//            delay-race arbiter can be re-armed. We detect this via
//            the number of rising edges on state/start activity across
//            a single query (>= 3 required).
//
//   CHECK_3  Seed-dependent output: the module's response must depend
//            on `bram_seed`. We drive two different seeds and ensure
//            that at least one of the 1024 test challenges produces
//            a different response across seeds. (If every challenge
//            produced the same response for both seeds, the seed
//            has no effect, so Fix 1 + Fix 5 together failed.)
//
// The testbench is intentionally forgiving about micro-timing so you
// can simulate either the original Verilog or the post-fix Verilog
// and see which checks pass.
//
// Target simulator: Xilinx XSim, Vivado 2022.2. Pure Verilog-2001,
// no SystemVerilog required.
// =====================================================================

`timescale 1ns / 1ps

module tb_sr_capuf;

    // ------------------------------------------------------------------
    // Clock and reset
    // ------------------------------------------------------------------
    reg clk;
    reg rst;

    initial clk = 1'b0;
    always  #5 clk = ~clk;            // 100 MHz

    // ------------------------------------------------------------------
    // DUT interface
    // ------------------------------------------------------------------
    reg  [31:0] challenge;
    reg         trigger;
    wire        response;
    wire        done;
    reg  [31:0] bram_seed;

    // The DUT port order matches your sr_capuf.v upload.
    sr_capuf dut (
        .clk       (clk),
        .rst       (rst),
        .challenge (challenge),
        .trigger   (trigger),
        .response  (response),
        .done      (done),
        .bram_seed (bram_seed)
    );

    // ------------------------------------------------------------------
    // Simulation-side bookkeeping
    // ------------------------------------------------------------------
    integer ok_handshake;
    integer ok_vote_independence;
    integer ok_seed_effect;

    integer i;
    integer done_rising_edges;
    reg     done_prev;

    integer start_a_rising_edges;
    reg     start_a_prev;
    // Probe into the DUT: layer_a APUFs share `state == 1`-derived start.
    // We observe generate block's internal start by peeking at state.
    // This is a simulation-only construct and does not synthesize.
    wire [1:0] dut_state = dut.state;
    wire       layer_a_start = (dut_state == 2'd1);

    // Capture per-query responses for two seed values.
    reg        resp_seedA [0:1023];
    reg        resp_seedB [0:1023];
    integer    differ_count;

    // ------------------------------------------------------------------
    // Helper task: one query, with a timeout.
    // ------------------------------------------------------------------
    task do_query;
        input [31:0] ch;
        output       r;
        integer      timeout;
        begin
            @(posedge clk);
            challenge <= ch;
            trigger   <= 1'b1;
            @(posedge clk);
            trigger   <= 1'b0;

            timeout = 0;
            // Wait for done rising edge, up to 200 cycles.
            // For the unfixed Verilog, done never de-asserts between
            // queries; we handle that separately below.
            while (done !== 1'b1 && timeout < 200) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
            // Sample response.
            r = response;
            // Hold a few cycles of idle before next query, so that an
            // unfixed DUT has time to latch state back to 0.
            repeat (8) @(posedge clk);
        end
    endtask

    // ------------------------------------------------------------------
    // Main stimulus
    // ------------------------------------------------------------------
    reg q_resp;

    initial begin
        $display("-----------------------------------------------------");
        $display("tb_sr_capuf: starting behavioral simulation");
        $display("-----------------------------------------------------");

        // Initial conditions
        challenge = 32'h0000_0000;
        trigger   = 1'b0;
        bram_seed = 32'hA5A5_5A5A;
        rst       = 1'b1;
        done_prev = 1'b0;
        start_a_prev = 1'b0;
        done_rising_edges = 0;
        start_a_rising_edges = 0;

        repeat (20) @(posedge clk);
        rst = 1'b0;
        repeat (20) @(posedge clk);

        // -----------------------------------------------------------
        // CHECK_1: handshake over N queries. Count rising edges on done.
        // Pass criterion: at least N-1 rising edges observed across N
        // queries (we allow one to be eaten by the first measurement
        // that starts with done=0 already).
        // -----------------------------------------------------------
        done_rising_edges = 0;
        for (i = 0; i < 8; i = i + 1) begin
            do_query($random, q_resp);
        end
        // ok_handshake is populated by the monitor below.

        // -----------------------------------------------------------
        // CHECK_2: vote independence. We fire one query and count how
        // many times `layer_a_start` goes 1 -> 0 -> 1 during state 1
        // activity. A correct fix produces >= 3 rising edges of
        // layer_a_start within a single query window. The broken
        // original produces exactly 1.
        // -----------------------------------------------------------
        start_a_rising_edges = 0;
        // Gate the counter: we only count during state != 0 and shortly
        // after. Set a flag here.
        @(posedge clk);
        challenge <= 32'hDEAD_BEEF;
        trigger   <= 1'b1;
        @(posedge clk);
        trigger   <= 1'b0;
        // Watch for a generous window.
        repeat (200) @(posedge clk);
        ok_vote_independence = (start_a_rising_edges >= 3) ? 1 : 0;

        // -----------------------------------------------------------
        // CHECK_3: seed-dependent output. Drive two different seeds,
        // force a re-read by pulsing reset, and run the same 1024
        // challenges. Track how many produce different responses.
        // -----------------------------------------------------------
        // Seed A
        bram_seed = 32'hA5A5_5A5A;
        rst = 1'b1; repeat (10) @(posedge clk);
        rst = 1'b0; repeat (10) @(posedge clk);

        for (i = 0; i < 1024; i = i + 1) begin
            do_query(i[31:0], q_resp);
            resp_seedA[i] = q_resp;
        end

        // Seed B
        bram_seed = 32'h5A5A_A5A5;
        rst = 1'b1; repeat (10) @(posedge clk);
        rst = 1'b0; repeat (10) @(posedge clk);

        for (i = 0; i < 1024; i = i + 1) begin
            do_query(i[31:0], q_resp);
            resp_seedB[i] = q_resp;
        end

        differ_count = 0;
        for (i = 0; i < 1024; i = i + 1) begin
            if (resp_seedA[i] !== resp_seedB[i]) differ_count = differ_count + 1;
        end
        // A well-seeded design should flip ~50% of the responses. Any
        // value above ~5% is enough to prove the seed affects the path.
        ok_seed_effect = (differ_count > 50) ? 1 : 0;

        // -----------------------------------------------------------
        // Report
        // -----------------------------------------------------------
        $display("-----------------------------------------------------");
        if (ok_handshake)         $display("PASS: handshake            (%0d rising edges on done)", done_rising_edges);
        else                      $display("FAIL: handshake            (%0d rising edges on done; expected >=7 over 8 queries)", done_rising_edges);

        if (ok_vote_independence) $display("PASS: three distinct votes (%0d rising edges on layer_a start)", start_a_rising_edges);
        else                      $display("FAIL: three distinct votes (%0d rising edges on layer_a start; expected >=3)", start_a_rising_edges);

        if (ok_seed_effect)       $display("PASS: seed affects output  (%0d of 1024 responses differ across seeds)", differ_count);
        else                      $display("FAIL: seed affects output  (%0d of 1024 responses differ across seeds; expected >50)", differ_count);
        $display("-----------------------------------------------------");

        $finish;
    end

    // ------------------------------------------------------------------
    // Edge monitors (live throughout simulation)
    // ------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst) begin
            if (done === 1'b1 && done_prev === 1'b0) begin
                done_rising_edges <= done_rising_edges + 1;
            end
            done_prev <= done;

            if (layer_a_start === 1'b1 && start_a_prev === 1'b0) begin
                start_a_rising_edges <= start_a_rising_edges + 1;
            end
            start_a_prev <= layer_a_start;
        end
    end

    // handshake pass/fail computed from the accumulated counter after CHECK_1 stimulus:
    initial begin
        @(rst == 0);
        // wait long enough for CHECK_1 to finish (8 queries * ~250 cyc)
        #25000;
        ok_handshake = (done_rising_edges >= 7) ? 1 : 0;
    end

    // Safety net: if the sim hangs for any reason, end at 5 ms.
    initial begin
        #5000000;
        $display("TIMEOUT: simulation exceeded safety limit");
        $finish;
    end

endmodule
