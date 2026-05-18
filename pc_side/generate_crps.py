"""
generate_crps.py

Simulates the Hybrid SR-CAPUF and produces 1,000,000 challenge-response
pairs (CRPs) for ML-attack evaluation, with real-time progress reporting.

Architecture simulated (matches sr_capuf.v + apuf_sync.v):

    c (32 bits) ---> [ H matrix, sparsity ] ---> c_h (32 bits)
                                                    |
                         ,----------------------- c_h
                         v
           [ Layer A: 32 APUFs, 3-vote majority ]
                         |
                         v  s_maj (32 bits)
                 c_prime = c_h XOR s_maj
                         |
                         v
              [ Layer B: 3 APUFs, XOR-combined ]
                         |
                         v
                    y (1 bit response)

Outputs (all written to ./crp_output/):
    CRPs_1M.xlsx     -- spreadsheet, columns c0..c31, y
    CRPs_1M.csv      -- same data, fast-loading for ML attacks
    ground_truth.npz -- H matrix and weight vectors (attacker's oracle)
    summary.txt      -- quality metrics (uniformity, wall time, seed)

Run:
    python generate_crps.py

Dependencies:
    pip install numpy pandas xlsxwriter tqdm
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xlsxwriter

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[warn] tqdm not installed; progress bars will fall back to plain prints.")
    print("       Install with: pip install tqdm")

# ---------------------------------------------------------------------------
# Configuration  (edit these if you want a different dataset)
# ---------------------------------------------------------------------------
N_CRPS         = 1_000_000        # number of challenge-response pairs
N_STAGES       = 32               # APUF stages (matches sr_capuf.v)
N_LAYER_A      = 32               # Layer-A APUFs (matches sr_capuf.v)
N_LAYER_B      = 3                # Layer-B APUFs (3-XOR, matches sr_capuf.v)
H_ONES_PER_ROW = 5                # sparsity of H matrix
NOISE_STD      = 0.05             # Gaussian noise on delay (reliability sim)
SEED           = 20260417         # change to regenerate a fresh dataset
CHUNK_SIZE     = 25_000           # CRPs per chunk (smaller = more updates)
OUTPUT_DIR     = Path("./crp_output")

assert N_CRPS + 1 <= 1_048_576, "N_CRPS exceeds Excel's single-sheet limit"


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    """Timestamped log line, flushed immediately for live tail."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def banner(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}", flush=True)


def progress_iter(iterable, total: int, desc: str):
    """tqdm progress bar if available, else periodic print fallback."""
    if HAS_TQDM:
        return tqdm(iterable, total=total, desc=desc, unit="chunk",
                    dynamic_ncols=True, mininterval=0.2)
    # Fallback: print every 5 chunks
    def _gen():
        t0 = time.time()
        for i, item in enumerate(iterable, 1):
            yield item
            if i % 5 == 0 or i == total:
                pct = 100.0 * i / total
                rate = i / max(time.time() - t0, 1e-6)
                print(f"  {desc}: {i}/{total} ({pct:5.1f}%)  {rate:.1f} chunk/s",
                      flush=True)
    return _gen()


# ---------------------------------------------------------------------------
# PUF building blocks
# ---------------------------------------------------------------------------
def build_h_matrix(n: int, ones_per_row: int, rng: np.random.Generator) -> np.ndarray:
    """Sparse n x n binary matrix. Each row has exactly `ones_per_row` ones."""
    H = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        idx = rng.choice(n, size=ones_per_row, replace=False)
        H[i, idx] = 1
    return H


def build_weights(n_apufs: int, n_stages: int, rng: np.random.Generator) -> np.ndarray:
    """Per-APUF delay weight vectors, shape (n_apufs, n_stages+1)."""
    return rng.normal(0.0, 1.0, size=(n_apufs, n_stages + 1)).astype(np.float32)


def challenges_to_phi(c: np.ndarray) -> np.ndarray:
    """Batch convert bit challenges to parity feature vectors.
    phi_i = prod_{j=i}^{k-1}(1 - 2*c_j),  phi_k = 1
    """
    b = (1 - 2 * c.astype(np.int32)).astype(np.float32)
    n = c.shape[1]
    phi = np.ones((c.shape[0], n + 1), dtype=np.float32)
    phi[:, :n] = np.cumprod(b[:, ::-1], axis=1)[:, ::-1]
    return phi


def apuf_batch(phi: np.ndarray,
               weights: np.ndarray,
               noise_std: float,
               rng: np.random.Generator) -> np.ndarray:
    """Evaluate a batch of APUFs on a batch of challenges.
    returns (n_crps, n_apufs) uint8 of 0/1 responses.
    """
    delta = phi @ weights.T
    if noise_std > 0.0:
        delta = delta + rng.normal(0.0, noise_std, size=delta.shape).astype(np.float32)
    return (delta > 0.0).astype(np.uint8)


def majority_vote_3(phi: np.ndarray,
                    weights: np.ndarray,
                    noise_std: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Three independent evaluations with fresh noise, majority-voted per bit."""
    r = np.zeros(phi.shape[:1] + weights.shape[:1], dtype=np.int32)
    for _ in range(3):
        r += apuf_batch(phi, weights, noise_std, rng)
    return (r >= 2).astype(np.uint8)


def evaluate_chunk(C_chunk: np.ndarray,
                   H: np.ndarray,
                   W_A: np.ndarray,
                   W_B: np.ndarray,
                   noise_std: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Run the full SR-CAPUF pipeline on a chunk of challenges.
    Returns (chunk_size,) uint8 of response bits.
    """
    C_H     = (C_chunk @ H.T) % 2
    phi_h   = challenges_to_phi(C_H)
    S_MAJ   = majority_vote_3(phi_h, W_A, noise_std, rng)
    C_PRIME = C_H ^ S_MAJ
    phi_p   = challenges_to_phi(C_PRIME)
    Y_VEC   = apuf_batch(phi_p, W_B, noise_std, rng)
    return (Y_VEC[:, 0] ^ Y_VEC[:, 1] ^ Y_VEC[:, 2]).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main generation pipeline (verbose, chunked, real-time progress)
# ---------------------------------------------------------------------------
def main() -> None:
    overall_t0 = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    rng = np.random.default_rng(SEED)

    banner("Hybrid SR-CAPUF CRP generator")
    log(f"Configuration:")
    log(f"  N_CRPS         = {N_CRPS:,}")
    log(f"  N_STAGES       = {N_STAGES}")
    log(f"  N_LAYER_A      = {N_LAYER_A}  (3-vote majority)")
    log(f"  N_LAYER_B      = {N_LAYER_B}  (XOR-combined)")
    log(f"  H sparsity     = {H_ONES_PER_ROW} ones/row")
    log(f"  NOISE_STD      = {NOISE_STD}")
    log(f"  SEED           = {SEED}")
    log(f"  CHUNK_SIZE     = {CHUNK_SIZE:,}")
    log(f"  OUTPUT_DIR     = {OUTPUT_DIR.resolve()}")

    # -- Build the simulated PUF ------------------------------------------
    banner("Step 1/4 - Building simulated PUF")
    log(f"Generating {N_STAGES}x{N_STAGES} sparse H matrix")
    H = build_h_matrix(N_STAGES, H_ONES_PER_ROW, rng)
    log(f"  H matrix density: {H.sum() / H.size:.4f} "
        f"(expected {H_ONES_PER_ROW / N_STAGES:.4f})")

    log(f"Sampling Layer-A weight vectors ({N_LAYER_A} x {N_STAGES + 1})")
    W_A = build_weights(N_LAYER_A, N_STAGES, rng)
    log(f"  W_A range: [{W_A.min():+.3f}, {W_A.max():+.3f}], "
        f"mean={W_A.mean():+.4f}, std={W_A.std():.4f}")

    log(f"Sampling Layer-B weight vectors ({N_LAYER_B} x {N_STAGES + 1})")
    W_B = build_weights(N_LAYER_B, N_STAGES, rng)
    log(f"  W_B range: [{W_B.min():+.3f}, {W_B.max():+.3f}], "
        f"mean={W_B.mean():+.4f}, std={W_B.std():.4f}")

    # -- Generate CRPs in chunks ------------------------------------------
    banner(f"Step 2/4 - Generating {N_CRPS:,} CRPs in chunks of {CHUNK_SIZE:,}")
    n_chunks = (N_CRPS + CHUNK_SIZE - 1) // CHUNK_SIZE
    log(f"Total chunks: {n_chunks}")

    C_all = np.empty((N_CRPS, N_STAGES), dtype=np.uint8)
    Y_all = np.empty((N_CRPS,),          dtype=np.uint8)

    gen_t0 = time.time()
    chunk_indices = range(n_chunks)
    bar = progress_iter(chunk_indices, n_chunks, "  generating")
    for k in bar:
        start = k * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, N_CRPS)
        actual = end - start

        C_chunk = rng.integers(0, 2, size=(actual, N_STAGES), dtype=np.uint8)
        Y_chunk = evaluate_chunk(C_chunk, H, W_A, W_B, NOISE_STD, rng)

        C_all[start:end] = C_chunk
        Y_all[start:end] = Y_chunk

        if HAS_TQDM:
            elapsed = time.time() - gen_t0
            rate = end / max(elapsed, 1e-6)
            bar.set_postfix_str(
                f"{end:,}/{N_CRPS:,} ({100*end/N_CRPS:5.1f}%) | "
                f"{rate:,.0f} CRPs/s"
            )

    gen_elapsed = time.time() - gen_t0
    log(f"Generation complete: {N_CRPS:,} CRPs in {gen_elapsed:.1f}s "
        f"({N_CRPS/gen_elapsed:,.0f} CRPs/s)")

    uniformity = float(Y_all.mean())
    log(f"Response uniformity: {uniformity:.4f}  (ideal 0.5000)")
    if not (0.45 <= uniformity <= 0.55):
        log(f"  [warn] uniformity outside [0.45, 0.55]; consider rerunning "
            f"with a different SEED")

    # -- Write CSV --------------------------------------------------------
    banner("Step 3/4 - Writing CSV (fast format for ML loading)")
    columns = [f"c{i}" for i in range(N_STAGES)] + ["y"]
    csv_path = OUTPUT_DIR / "CRPs_1M.csv"
    log(f"Writing {csv_path}")

    csv_t0 = time.time()
    with open(csv_path, "w", newline="") as f:
        f.write(",".join(columns) + "\n")

        bar = progress_iter(range(n_chunks), n_chunks, "  writing CSV")
        for k in bar:
            start = k * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, N_CRPS)
            block = np.column_stack([C_all[start:end], Y_all[start:end]])
            np.savetxt(f, block, fmt="%d", delimiter=",")

            if HAS_TQDM:
                bar.set_postfix_str(
                    f"{end:,}/{N_CRPS:,} ({100*end/N_CRPS:5.1f}%)"
                )

    csv_elapsed = time.time() - csv_t0
    csv_mb = csv_path.stat().st_size / 1_048_576
    log(f"CSV done: {csv_mb:.1f} MB in {csv_elapsed:.1f}s")

    # -- Write XLSX -------------------------------------------------------
    banner("Step 4/4 - Writing XLSX spreadsheet (this is the slowest step)")
    xlsx_path = OUTPUT_DIR / "CRPs_1M.xlsx"
    log(f"Writing {xlsx_path}")
    log("  Note: xlsxwriter has to encode every cell; expect 1-3 minutes.")

    xlsx_t0 = time.time()
    workbook = xlsxwriter.Workbook(
        str(xlsx_path),
        {"constant_memory": True},     # streaming mode -> low RAM
    )
    sheet = workbook.add_worksheet("CRPs")

    # Header row
    for col_idx, name in enumerate(columns):
        sheet.write(0, col_idx, name)

    # Data rows, written chunk-by-chunk with progress
    bar = progress_iter(range(n_chunks), n_chunks, "  writing XLSX")
    rows_written = 0
    for k in bar:
        start = k * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, N_CRPS)
        block = np.column_stack([C_all[start:end], Y_all[start:end]])

        # write_row is much faster than per-cell .write()
        for i, row_vals in enumerate(block):
            sheet.write_row(start + 1 + i, 0, row_vals.tolist())
        rows_written = end

        if HAS_TQDM:
            elapsed = time.time() - xlsx_t0
            rate = rows_written / max(elapsed, 1e-6)
            bar.set_postfix_str(
                f"{rows_written:,}/{N_CRPS:,} "
                f"({100*rows_written/N_CRPS:5.1f}%) | "
                f"{rate:,.0f} rows/s"
            )

    log("Closing XLSX (flushing to disk)...")
    workbook.close()
    xlsx_elapsed = time.time() - xlsx_t0
    xlsx_mb = xlsx_path.stat().st_size / 1_048_576
    log(f"XLSX done: {xlsx_mb:.1f} MB in {xlsx_elapsed:.1f}s")

    # -- Save ground-truth NPZ --------------------------------------------
    banner("Saving ground truth (attacker oracle) and summary")
    truth_path = OUTPUT_DIR / "ground_truth.npz"
    log(f"Writing {truth_path}")
    np.savez_compressed(
        truth_path,
        H=H, W_A=W_A, W_B=W_B,
        seed=SEED, noise_std=NOISE_STD,
    )
    truth_kb = truth_path.stat().st_size / 1024
    log(f"Ground truth: {truth_kb:.1f} KB")

    # -- Summary ----------------------------------------------------------
    summary_path = OUTPUT_DIR / "summary.txt"
    total_elapsed = time.time() - overall_t0
    summary = (
        f"Hybrid SR-CAPUF CRP dataset\n"
        f"---------------------------\n"
        f"CRPs generated      : {N_CRPS:,}\n"
        f"Challenge bits      : {N_STAGES}\n"
        f"Layer-A APUFs       : {N_LAYER_A} (3-vote majority)\n"
        f"Layer-B APUFs       : {N_LAYER_B} (XOR-combined)\n"
        f"H matrix sparsity   : {H_ONES_PER_ROW} ones/row\n"
        f"Simulation noise std: {NOISE_STD}\n"
        f"Seed                : {SEED}\n"
        f"Response uniformity : {uniformity:.4f}  (ideal 0.5000)\n"
        f"\n"
        f"Wall time breakdown:\n"
        f"  Generation        : {gen_elapsed:7.1f} s "
        f"({N_CRPS/gen_elapsed:>10,.0f} CRPs/s)\n"
        f"  CSV write         : {csv_elapsed:7.1f} s "
        f"({csv_mb:>8.1f} MB output)\n"
        f"  XLSX write        : {xlsx_elapsed:7.1f} s "
        f"({xlsx_mb:>8.1f} MB output)\n"
        f"  Total             : {total_elapsed:7.1f} s\n"
    )
    summary_path.write_text(summary)
    log(f"Summary: {summary_path}")

    banner("All done")
    print(summary, flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[interrupted by user]", flush=True)
        sys.exit(130)
