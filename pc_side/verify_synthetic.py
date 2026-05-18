"""
verify_synthetic.py
===================

Self-test that exercises `make_phi.py` and `metrics.py` on synthetic
CRP data before you touch hardware. This catches math bugs in the
feature builder and the metric estimators without needing the FPGA.

Run it from the pc_side directory::

    python3 verify_synthetic.py

All assertions must pass. The script prints a one-line summary per test
and exits non-zero if any assertion fails.
"""
from __future__ import annotations

import os
import sys
import tempfile
import json

import numpy as np

# Add this directory to the import path so the script is runnable from
# any CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_phi import (
    challenge_bits_from_uint32,
    phi_from_bits,
    phi_from_uint32,
    apply_H_matrix,
)

# metrics.py imports relatively, avoid __main__ side effects
import metrics


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
def test_bits_roundtrip():
    """Unpack -> repack returns the original uint32."""
    rng = np.random.default_rng(0)
    c = rng.integers(0, 1 << 32, size=10_000, dtype=np.uint32)
    bits = challenge_bits_from_uint32(c)
    assert bits.shape == (10_000, 32)
    assert bits.dtype == np.uint8
    repack = np.zeros_like(c)
    for i in range(32):
        repack |= bits[:, i].astype(np.uint32) << np.uint32(i)
    assert np.array_equal(repack, c), "bit roundtrip failed"
    print("OK  test_bits_roundtrip")


def test_phi_properties():
    """Phi entries are +/-1, bias is +1, shape is (N, k+1)."""
    rng = np.random.default_rng(1)
    c = rng.integers(0, 1 << 32, size=5_000, dtype=np.uint32)
    phi = phi_from_uint32(c)
    assert phi.shape == (5_000, 33)
    assert phi.dtype == np.int8
    assert set(np.unique(phi).tolist()) <= {-1, 1}
    assert np.all(phi[:, -1] == 1), "bias column should be +1"
    print("OK  test_phi_properties")


def test_phi_matches_linear_model():
    """Synthetic APUF with known weight vector is learnable from Phi
    via the closed-form sign rule y = sign(w . Phi).
    """
    rng = np.random.default_rng(2)
    N = 20_000
    k = 32
    c = rng.integers(0, 1 << 32, size=N, dtype=np.uint32)
    phi = phi_from_uint32(c, k=k)

    # Random "delay weight" vector
    w = rng.normal(size=k + 1).astype(np.float32)
    w[-1] = 0.0  # zero bias for simplicity

    y = (phi.astype(np.float32) @ w > 0).astype(np.int8)

    # Predicting with the true w must be perfect
    y_hat = (phi.astype(np.float32) @ w > 0).astype(np.int8)
    assert (y_hat == y).mean() == 1.0

    # A random w' should score near 50%
    w2 = rng.normal(size=k + 1).astype(np.float32)
    y2 = (phi.astype(np.float32) @ w2 > 0).astype(np.int8)
    acc = (y2 == y).mean()
    assert 0.35 < acc < 0.65, f"random w should be near chance; got {acc:.3f}"
    print(f"OK  test_phi_matches_linear_model (random-w acc={acc:.3f})")


def test_apply_H_roundtrip():
    """If H is the identity, c_h == c bits."""
    rng = np.random.default_rng(3)
    c = rng.integers(0, 1 << 32, size=1_000, dtype=np.uint32)
    H = np.eye(32, dtype=np.uint8)
    c_h = apply_H_matrix(c, H)
    assert np.array_equal(c_h, challenge_bits_from_uint32(c))
    print("OK  test_apply_H_roundtrip (identity)")


def test_apply_H_involution():
    """If H is invertible mod 2, applying it twice with its inverse
    returns the original (we pick a self-inverse H here for simplicity).
    """
    # A permutation matrix is self-inverse only if it is an involution.
    # We construct such a permutation: swap pairs (0,1),(2,3),...
    k = 32
    H = np.zeros((k, k), dtype=np.uint8)
    for i in range(0, k, 2):
        H[i, i + 1] = 1
        H[i + 1, i] = 1
    rng = np.random.default_rng(4)
    c = rng.integers(0, 1 << 32, size=1_000, dtype=np.uint32)
    c_h = apply_H_matrix(c, H)
    c_hh = apply_H_matrix(c_h, H)
    assert np.array_equal(c_hh, challenge_bits_from_uint32(c)), \
        "H . H . c should equal c for involutory H"
    print("OK  test_apply_H_involution")


def test_metrics_on_synthetic_dataset():
    """Build a fake shard and run metrics.run() end-to-end."""
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(5)
        N = 50_000
        repeats = 11
        # Simulate: unbiased responses with ~3% raw read error.
        r_true = rng.integers(0, 2, size=N, dtype=np.uint8)
        flip_prob = 0.03
        raw = np.zeros((N, repeats), dtype=np.uint8)
        for i in range(repeats):
            flips = rng.random(N) < flip_prob
            raw[:, i] = r_true ^ flips.astype(np.uint8)
        # Majority vote recomputed to mirror what the driver does.
        ones = raw.sum(axis=1)
        r_vote = (ones > (repeats // 2)).astype(np.uint8)
        # reliability count
        r_rel = np.where(r_vote == 1, ones, repeats - ones).astype(np.uint8)
        c_bin = rng.integers(0, 1 << 32, size=N, dtype=np.uint32)

        shard_path = os.path.join(tmp, "shard_0000.npz")
        np.savez_compressed(shard_path,
                            c_bin=c_bin, r=r_vote, r_rel=r_rel, raw=raw)
        meta = {
            "schema_version": 1,
            "bram_seed_fingerprint": "synthetic_test",
            "prng_seed": 0,
        }
        with open(os.path.join(tmp, "meta.json"), "w") as f:
            json.dump(meta, f)

        stats = metrics.run(tmp)
        # Expectations
        assert 0.47 < stats["uniformity"] < 0.53
        # Observed BER should be near flip_prob (3%)
        assert abs(stats["ber_mean"] - flip_prob) < 0.01, \
            f"ber_mean={stats['ber_mean']}, expected ~{flip_prob}"
        # Min-entropy near 1 bit for a near-uniform source
        assert 0.9 < stats["min_entropy_mcv"] <= 1.0
        print(f"OK  test_metrics_on_synthetic_dataset "
              f"(U={stats['uniformity']:.3f} BER={stats['ber_mean']:.3f} "
              f"Hmin={stats['min_entropy_mcv']:.3f})")


def test_uniqueness_requires_aligned_challenges():
    """uniqueness_two_seeds must refuse misaligned datasets."""
    with tempfile.TemporaryDirectory() as tmp_a, \
         tempfile.TemporaryDirectory() as tmp_b:
        rng_a = np.random.default_rng(10)
        rng_b = np.random.default_rng(20)
        for tmp, rng, fp in [(tmp_a, rng_a, "aaa"), (tmp_b, rng_b, "bbb")]:
            N = 1000
            repeats = 11
            c = rng.integers(0, 1 << 32, size=N, dtype=np.uint32)
            r = rng.integers(0, 2, size=N, dtype=np.uint8)
            raw = np.tile(r[:, None], (1, repeats))
            np.savez_compressed(os.path.join(tmp, "shard_0000.npz"),
                                c_bin=c, r=r, r_rel=np.full(N, repeats, np.uint8),
                                raw=raw)
            with open(os.path.join(tmp, "meta.json"), "w") as f:
                json.dump({"bram_seed_fingerprint": fp, "prng_seed": 1}, f)
        try:
            metrics.uniqueness_two_seeds(tmp_a, tmp_b)
        except ValueError as e:
            assert "align" in str(e) or "match" in str(e) or "prng_seed" in str(e)
            print("OK  test_uniqueness_requires_aligned_challenges")
            return
        raise AssertionError("uniqueness_two_seeds did not raise on misaligned challenges")


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------
def main() -> int:
    tests = [
        test_bits_roundtrip,
        test_phi_properties,
        test_phi_matches_linear_model,
        test_apply_H_roundtrip,
        test_apply_H_involution,
        test_metrics_on_synthetic_dataset,
        test_uniqueness_requires_aligned_challenges,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            failures += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failures += 1
    print("-" * 40)
    if failures == 0:
        print(f"ALL {len(tests)} TESTS PASSED")
        return 0
    else:
        print(f"{failures} / {len(tests)} TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
