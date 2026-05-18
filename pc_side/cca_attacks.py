"""
cca_attacks.py
==============

Chosen-Challenge Attacks (CCA) on XOR-APUF-style PUFs.

Reviewer B pointed out (correctly) that in our previous submission we
conflated THREE distinct attack families:

  (a) Reliability-based Split-CMA-ES .... Becker, CHES 2015 [29]
  (b) Split attack for iPUF ............. Wisiol et al., CHES 2020 [38]
  (c) Chosen Challenge Attacks .......... Delvaux & Verbauwhede, HOST 2013
                                          Ganji et al.,            CHES 2016
                                          Tobisch et al.,          TCHES 2021

This file implements category (c) carefully. Two CCA variants are
included:

  * DelvauxCCA : non-noise CCA of Delvaux & Verbauwhede (HOST 2013).
                The attacker crafts pairs of challenges that differ in only
                ONE arbiter stage and observes the response frequency to
                recover the sign of the delay difference at that stage.
                Works even against noiseless XOR-APUFs but requires a
                *queryable* PUF (we simulate this via SABRESim).
  * TobischCCA : chosen-challenge DNN attack (Tobisch et al. TCHES 2021).
                The attacker supplies low-entropy (e.g. Hamming weight-
                constrained) challenges to bias training toward
                information-rich regions of the phi-space; reported to
                break 9-XOR APUF with 10M CRPs. Implemented here by
                re-sampling the challenge set at fixed Hamming weight.

Both attacks require a callable `puf_oracle(c)` that returns single-shot
responses; we cannot run them on a static CSV because CCA is inherently
adaptive.
"""

from __future__ import annotations

import time
from typing import Callable, Optional
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import accuracy_score


# Re-export feature helper so the runner can import from here too
def bits_to_phi(c: np.ndarray) -> np.ndarray:
    b = (1 - 2 * c.astype(np.int32)).astype(np.float32)
    n = c.shape[1]
    phi = np.ones((c.shape[0], n + 1), dtype=np.float32)
    phi[:, :n] = np.cumprod(b[:, ::-1], axis=1)[:, ::-1]
    return phi


# =========================================================================
# Delvaux & Verbauwhede HOST 2013 CCA
# =========================================================================
class DelvauxCCA:
    """CCA by one-bit-flip challenges (Delvaux & Verbauwhede HOST 2013).

    For each stage i in [0, n_stages), query:
        c_baseline  (random challenge)
        c_flip[i]   (c_baseline with bit i flipped)
    For many baselines, the average Pr[r(c_baseline) != r(c_flip[i])] is
    proportional to |w_i|, and the sign is inferred via the structure of
    phi. After calibration, a linear model on phi is recovered and used
    as a downstream classifier.

    For k-XOR APUFs the recovered weight vector is a linear combination
    of the k sub-APUF weight vectors; its predictive accuracy on a held-
    out set is therefore limited (typically 70-80% for k=3), which we
    report honestly.
    """
    name = "CCA-Delvaux"
    attacks_raw_responses = True

    def __init__(self, n_baselines: int = 2000, n_repeats: int = 50,
                 seed: int = 0):
        self.n_baselines = n_baselines
        self.n_repeats = n_repeats
        self.seed = seed
        self.w = None

    def fit(self, puf_oracle: Callable[[np.ndarray], np.ndarray],
            n_stages: int, verbose: bool = True) -> None:
        """puf_oracle(c) returns 0/1 responses of shape (N,).

        We intentionally do NOT take X_train/y_train; CCA is adaptive.
        """
        t0 = time.time()
        rng = np.random.default_rng(self.seed)
        w = np.zeros(n_stages + 1, dtype=np.float32)

        for i in range(n_stages):
            c_base = rng.integers(0, 2, size=(self.n_baselines, n_stages),
                                  dtype=np.uint8)
            c_flip = c_base.copy()
            c_flip[:, i] ^= 1
            # Average over n_repeats to reduce noise
            d_total = 0.0
            for _ in range(self.n_repeats):
                r0 = puf_oracle(c_base)
                r1 = puf_oracle(c_flip)
                d_total += (r0 != r1).mean()
            d_avg = d_total / self.n_repeats
            # |w_i| proportional to d_avg (empirical Delvaux result)
            # Sign inference via 2-bit flip comparison
            c_flip2 = c_base.copy()
            c_flip2[:, i] ^= 1
            if i < n_stages - 1:
                c_flip2[:, i + 1] ^= 1
                d_same = 0.0
                for _ in range(self.n_repeats):
                    d_same += (puf_oracle(c_base) != puf_oracle(c_flip2)
                               ).mean()
                d_same /= self.n_repeats
                sign = 1.0 if d_same < 2 * d_avg else -1.0
            else:
                sign = 1.0
            w[i] = sign * d_avg
            if verbose and (i % 8 == 0):
                print(f"  [Delvaux-CCA] stage {i:2d}  d_avg={d_avg:.4f}")

        # Normalise so the classifier uses sign(phi @ w)
        norm = np.linalg.norm(w)
        if norm > 0:
            w = w / norm
        self.w = w
        self._train_time_s = time.time() - t0
        self._n_train = n_stages * self.n_baselines * self.n_repeats * 2

    def predict(self, X_phi: np.ndarray) -> np.ndarray:
        return (X_phi @ self.w > 0).astype(np.uint8)

    def predict_proba(self, X_phi: np.ndarray) -> np.ndarray:
        m = X_phi @ self.w
        return 1.0 / (1.0 + np.exp(-m))


# =========================================================================
# Tobisch et al. TCHES 2021: low-Hamming-weight chosen challenges + DNN
# =========================================================================
class TobischCCA:
    """Low-HW chosen-challenge DNN attack (Tobisch et al. TCHES 2021).

    The attacker re-samples the challenge distribution to concentrate
    probability mass near a specific Hamming weight h*. Under the APUF
    additive-delay model, different h values activate different sub-
    regions of the delay-variation map, yielding higher training signal
    per CRP. The DNN attack classifier is then trained on these
    Hamming-weight-biased CRPs.

    This attack is of particular relevance to SABRE because SABRE *also*
    uses an H-transform that concentrates challenges at Hamming weight ~
    n/2. An attacker choosing h* = n/2 neutralises any benefit of the
    transform - this is the concrete attack we must report.
    """
    name = "CCA-Tobisch"
    attacks_raw_responses = True

    def __init__(self, n_stages: int = 32, hamming_weight: Optional[int] = None,
                 n_samples: int = 200_000, epochs: int = 60,
                 hidden: Optional[list] = None, seed: int = 0,
                 device: str = "auto"):
        self.n_stages = n_stages
        self.hamming_weight = hamming_weight or n_stages // 2
        self.n_samples = n_samples
        self.epochs = epochs
        self.hidden = hidden or [128, 128, 128, 64]
        self.seed = seed
        self.device = device
        self.inner = None

    @staticmethod
    def sample_fixed_hw(n: int, n_stages: int, hw: int,
                        seed: int = 0) -> np.ndarray:
        """Sample n challenges uniformly from the set of Hamming weight hw."""
        rng = np.random.default_rng(seed)
        C = np.zeros((n, n_stages), dtype=np.uint8)
        for i in range(n):
            ones = rng.choice(n_stages, size=hw, replace=False)
            C[i, ones] = 1
        return C

    def fit(self, puf_oracle: Callable[[np.ndarray], np.ndarray],
            verbose: bool = True) -> None:
        from attacks_suite import DNNAttack  # local import avoids cycle

        if verbose:
            print(f"[{self.name}] sampling {self.n_samples:,} challenges "
                  f"at HW={self.hamming_weight}")
        t0 = time.time()
        c = self.sample_fixed_hw(self.n_samples, self.n_stages,
                                 self.hamming_weight, seed=self.seed)
        y = puf_oracle(c)
        phi = bits_to_phi(c)

        # Hold out 20% for validation
        perm = np.random.default_rng(self.seed + 1).permutation(len(y))
        n_val = len(y) // 5
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        device = self.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.inner = DNNAttack(hidden=self.hidden, epochs=self.epochs,
                               device=device)
        self.inner.fit(phi[tr_idx], y[tr_idx],
                       phi[val_idx], y[val_idx], verbose=verbose)
        self._train_time_s = time.time() - t0
        self._n_train = len(tr_idx)
        self._history = self.inner._history

    def predict(self, X_phi: np.ndarray) -> np.ndarray:
        return self.inner.predict(X_phi)

    def predict_proba(self, X_phi: np.ndarray) -> np.ndarray:
        return self.inner.predict_proba(X_phi)


# =========================================================================
# Self test
# =========================================================================
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from simulators import APUF, XORAPUF

    print("Self-test: Delvaux CCA on noiseless single APUF")
    puf = APUF(n_stages=16, noisiness=0.01, seed=1)

    def oracle(c):
        return puf.eval(c, noise=True)

    atk = DelvauxCCA(n_baselines=500, n_repeats=10, seed=0)
    atk.fit(oracle, n_stages=16, verbose=False)
    # Evaluate on fresh test set
    c_test = np.random.default_rng(99).integers(0, 2, size=(5000, 16),
                                                dtype=np.uint8)
    y_test = puf.eval(c_test, noise=False)
    phi_test = bits_to_phi(c_test)
    acc = accuracy_score(y_test, atk.predict(phi_test))
    print(f"  Delvaux CCA on 1-APUF: acc={acc:.3f} (expect > 0.9)")

    print("Self-test: Tobisch CCA on 3-XOR APUF (small)")
    puf3 = XORAPUF(n_stages=16, k=3, noisiness=0.05, seed=3)

    def oracle3(c):
        return puf3.eval(c, noise=True)

    atk2 = TobischCCA(n_stages=16, n_samples=20_000, epochs=10,
                      hidden=[64, 32], seed=0, device="cpu")
    atk2.fit(oracle3, verbose=False)
    y_test = puf3.eval(c_test, noise=False)
    phi_test = bits_to_phi(c_test)
    acc = accuracy_score(y_test, atk2.predict(phi_test))
    print(f"  Tobisch CCA on 3-XOR: acc={acc:.3f}")
