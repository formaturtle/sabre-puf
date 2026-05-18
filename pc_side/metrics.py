"""
metrics.py
==========

PC-side quality metrics for an SR-CAPUF CRP dataset produced by
`pynq_side/collect_crps.py`.

Metrics implemented
-------------------
1. Uniformity (response bias): fraction of 1s among majority-voted
   responses. Ideal 0.5.
2. Per-bit uniformity heat (when responses are bundled into words): we
   don't have word-level responses here (single-bit response per
   challenge), so this degenerates into the global uniformity above.
3. Intra-device reliability / Bit Error Rate (BER) from the per-read
   `raw` matrix in each shard. Reports mean BER and a histogram.
4. Avalanche / Strict Avalanche Criterion approximation: for each CRP in
   a sampled subset, we compute the predicted number of response flips
   when one challenge bit is toggled, by resampling additional CRPs with
   that one bit flipped. This requires a second collection step on
   hardware — we instead compute the *observed* avalanche via neighbor
   challenges in the dataset if any exist (usually not; so we report
   that the metric is not computed and instruct how to get it).
5. Min-entropy (NIST SP 800-90B Most-Common-Value estimator).
6. Approximate uniqueness for a single board by replicated-instance
   emulation (requires a separate dataset collected with different
   BRAM seeds); we expose the math and let the caller supply two seeded
   datasets.

Usage::

    python3 metrics.py --dataset /path/to/crps_run1 --report report.txt

"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------
def iter_shards(dataset_dir: str) -> Iterable[Tuple[str, Dict[str, np.ndarray]]]:
    """Yield (path, dict) for each shard_*.npz in the directory."""
    paths = sorted(glob.glob(os.path.join(dataset_dir, "shard_*.npz")))
    if not paths:
        raise FileNotFoundError(f"No shard_*.npz under {dataset_dir}")
    for p in paths:
        with np.load(p) as z:
            yield p, {k: z[k] for k in z.files}


def load_concatenated(dataset_dir: str) -> Dict[str, np.ndarray]:
    """Concatenate all shards into single arrays. May be memory-heavy
    for very large datasets; for 1M CRPs this is ~20 MB so it's fine.
    """
    parts: Dict[str, List[np.ndarray]] = {"c_bin": [], "r": [], "r_rel": [], "raw": []}
    for _, sh in iter_shards(dataset_dir):
        for k in parts:
            parts[k].append(sh[k])
    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}


# ---------------------------------------------------------------------
# Metric 1: uniformity
# ---------------------------------------------------------------------
def uniformity(r: np.ndarray) -> float:
    """Global fraction of '1' responses. Ideal 0.5."""
    return float(r.mean())


def uniformity_chi2(r: np.ndarray) -> Tuple[float, int]:
    """Chi-square statistic of observed ones vs expected N/2.

    Returns (chi2, degrees_of_freedom=1).
    """
    N = r.size
    ones = int(r.sum())
    zeros = N - ones
    exp = N / 2.0
    chi2 = ((ones - exp) ** 2) / exp + ((zeros - exp) ** 2) / exp
    return float(chi2), 1


# ---------------------------------------------------------------------
# Metric 3: reliability / BER from raw reads
# ---------------------------------------------------------------------
def reliability_stats(raw: np.ndarray, r_vote: np.ndarray) -> Dict[str, float]:
    """Compute reliability / BER from raw reads.

    For each CRP we have `repeats` individual reads. The "golden" value
    is the per-CRP majority vote. BER per CRP is (number of reads that
    differ from the majority) / repeats. The dataset-level BER is the
    mean.
    """
    if raw.ndim != 2:
        raise ValueError("raw must be shape (N, repeats)")
    N, repeats = raw.shape
    # Expand vote to match raw
    flips = (raw != r_vote[:, None]).sum(axis=1)  # shape (N,)
    ber_per_crp = flips / repeats
    return {
        "repeats": int(repeats),
        "mean_ber": float(ber_per_crp.mean()),
        "std_ber": float(ber_per_crp.std()),
        "p99_ber": float(np.percentile(ber_per_crp, 99)),
        "frac_perfectly_stable": float((flips == 0).mean()),
        "frac_unstable_above_30pct": float((ber_per_crp > 0.30).mean()),
    }


# ---------------------------------------------------------------------
# Metric 5: min-entropy (MCV estimator, SP 800-90B)
# ---------------------------------------------------------------------
def min_entropy_mcv(r: np.ndarray, alpha: float = 0.99) -> float:
    """Most Common Value min-entropy estimator (SP 800-90B section 6.3.1).

    H_min = -log2(p_upper)
    p_upper = min(1, p_hat + sqrt(log(2/(1-alpha)) / (2*N)))

    For a single-bit stream, p_hat = max(ones, zeros)/N.
    alpha=0.99 is the standard confidence level.
    """
    N = r.size
    if N == 0:
        return 0.0
    ones = float(r.sum())
    p_hat = max(ones, N - ones) / N
    # Two-sided confidence bound, alpha specified as in SP 800-90B (using
    # confidence level 1-alpha == 0.01 here).
    bound = np.sqrt(np.log(2.0 / (1.0 - alpha)) / (2.0 * N))
    p_upper = min(1.0, p_hat + bound)
    return float(-np.log2(p_upper))


# ---------------------------------------------------------------------
# Metric 6: approximated uniqueness across two seeded datasets
# ---------------------------------------------------------------------
def uniqueness_two_seeds(dataset_dir_a: str, dataset_dir_b: str) -> Dict[str, float]:
    """Compute inter-seed Hamming distance between two datasets collected
    with different BRAM seeds on the SAME board (or different instances).

    Both datasets MUST have been collected with the same PRNG seed so the
    challenge sequences align. We check this and refuse if not.

    Returns a dict with mean/stdev HD and the ideal value (0.5 * N_bits).
    """
    a = load_concatenated(dataset_dir_a)
    b = load_concatenated(dataset_dir_b)

    meta_a = json.load(open(os.path.join(dataset_dir_a, "meta.json")))
    meta_b = json.load(open(os.path.join(dataset_dir_b, "meta.json")))
    if meta_a["prng_seed"] != meta_b["prng_seed"]:
        raise ValueError(
            "prng_seed differs; challenges are not aligned so inter-seed "
            "HD is meaningless. Re-collect with the same --seed value."
        )
    if meta_a["bram_seed_fingerprint"] == meta_b["bram_seed_fingerprint"]:
        raise ValueError(
            "Both datasets report the same BRAM seed fingerprint. Provision "
            "a different seed in one of the runs."
        )

    N = min(a["r"].size, b["r"].size)
    if not np.array_equal(a["c_bin"][:N], b["c_bin"][:N]):
        raise ValueError("Challenge sequences do not match element-wise; "
                         "datasets cannot be directly compared.")
    diff = (a["r"][:N] ^ b["r"][:N]).astype(np.int32)
    mean_hd = float(diff.mean())  # average per-bit flip rate across N CRPs
    return {
        "n_aligned_crps": int(N),
        "mean_hd_per_bit": mean_hd,
        "ideal": 0.5,
        "stdev_hd_per_bit": float(diff.std()),
    }


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------
def format_report(stats: Dict) -> str:
    lines = []
    lines.append("=" * 64)
    lines.append("SR-CAPUF quality metrics")
    lines.append("=" * 64)
    lines.append(f"Dataset : {stats['dataset']}")
    lines.append(f"N CRPs  : {stats['n_crps']}")
    lines.append(f"Repeats : {stats['repeats']}")
    lines.append(f"Seed fp : {stats['bram_seed_fingerprint']}")
    lines.append("")
    lines.append("[ Uniformity ]")
    lines.append(f"  Fraction of 1s     : {stats['uniformity']:.6f}   (ideal 0.500000)")
    lines.append(f"  Chi-square (df=1)  : {stats['chi2']:.2f}")
    lines.append("")
    lines.append("[ Reliability (intra-device BER from temporal reads) ]")
    lines.append(f"  Mean BER           : {stats['ber_mean']:.6f}")
    lines.append(f"  Stdev BER          : {stats['ber_std']:.6f}")
    lines.append(f"  99th pct BER       : {stats['ber_p99']:.6f}")
    lines.append(f"  Frac. stable (0%)  : {stats['ber_stable_frac']:.4f}")
    lines.append(f"  Frac. >30% BER     : {stats['ber_unstable_frac']:.4f}")
    lines.append("")
    lines.append("[ Min-entropy (NIST SP 800-90B MCV, alpha=0.99) ]")
    lines.append(f"  H_min (per bit)    : {stats['min_entropy_mcv']:.6f}")
    lines.append("")
    if stats.get("uniqueness") is not None:
        u = stats["uniqueness"]
        lines.append("[ Uniqueness across two seeded datasets ]")
        lines.append(f"  Aligned CRPs       : {u['n_aligned_crps']}")
        lines.append(f"  Mean inter-seed HD : {u['mean_hd_per_bit']:.6f}  (ideal 0.500000)")
        lines.append(f"  Stdev             : {u['stdev_hd_per_bit']:.6f}")
    else:
        lines.append("[ Uniqueness ]")
        lines.append("  Not computed. Provide --dataset-b to a second CRP set")
        lines.append("  collected with a different --bram-seed and the SAME --seed.")
    lines.append("=" * 64)
    return "\n".join(lines)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def run(dataset: str, dataset_b: str = None) -> Dict:
    data = load_concatenated(dataset)
    try:
        meta = json.load(open(os.path.join(dataset, "meta.json")))
    except FileNotFoundError:
        meta = {}

    rel = reliability_stats(data["raw"], data["r"])
    chi2, _ = uniformity_chi2(data["r"])
    stats = {
        "dataset": dataset,
        "n_crps": int(data["r"].size),
        "repeats": rel["repeats"],
        "bram_seed_fingerprint": meta.get("bram_seed_fingerprint", "?"),
        "uniformity": uniformity(data["r"]),
        "chi2": chi2,
        "ber_mean": rel["mean_ber"],
        "ber_std": rel["std_ber"],
        "ber_p99": rel["p99_ber"],
        "ber_stable_frac": rel["frac_perfectly_stable"],
        "ber_unstable_frac": rel["frac_unstable_above_30pct"],
        "min_entropy_mcv": min_entropy_mcv(data["r"]),
        "uniqueness": None,
    }
    if dataset_b is not None:
        stats["uniqueness"] = uniqueness_two_seeds(dataset, dataset_b)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="Path to CRP directory A")
    ap.add_argument("--dataset-b", default=None,
                    help="Optional path to a second CRP directory with a "
                         "different BRAM seed, for uniqueness approximation")
    ap.add_argument("--report", default=None,
                    help="Optional path to write text report")
    args = ap.parse_args()

    stats = run(args.dataset, args.dataset_b)
    text = format_report(stats)
    print(text)
    if args.report:
        with open(args.report, "w") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
