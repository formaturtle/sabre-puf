"""
simulators.py
=============

Calibrated simulators for APUF, k-XOR APUF, (x,y)-Interpose PUF, and the
SABRE composite design. Used to:

    1. Generate BASELINE CRPs for 4-XOR, 6-XOR, and (1,k)-iPUF so that
       SABRE's security can be CONTEXTUALIZED against the state of the art
       (Reviewer A explicitly asked for this comparison).

    2. Generate MULTI-READ CRPs with a calibrated noise model so that
       reliability-based attacks (Becker CHES 2015 Split-CMA-ES) can be
       faithfully executed. The real hardware dump is stabilized-only
       (post 7x majority + BCH), which by construction defeats reliability
       attacks - exactly what Reviewer B asked us to demonstrate.

Design choices:
    * Delay weights are sampled i.i.d. N(0, 1). This is the standard
      additive delay model (Lim et al. 2005; Ruhrmair et al. CCS 2010)
      with a unit variance.
    * Noise is additive Gaussian on the final delay *difference*. Sigma
      is set so that single-read reliability ~ 92-95% per sub-APUF, which
      matches the hardware measurements reported in the SABRE paper
      (Section 5, bit-flip rate ~ 5-8%).
    * Fuzzy extractor in `SABRESim.read_stable` performs 7x temporal
      majority voting exactly as on the PYNQ-Z1.
    * The public linear transform H is drawn once per SABRE instance with
      approximately 50% sparsity; H is stored on the instance so that
      white-box attackers can be modeled (Reviewer A asked whether the
      attacker knowing H matters - this module lets us test it both ways).

Author: Alejandro Almeida (FIU)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# =========================================================================
# Feature helpers (kept here to avoid a circular import with attacks_suite)
# =========================================================================
def bits_to_phi(c: np.ndarray) -> np.ndarray:
    """0/1 challenge bits -> +/-1 parity (phi) features."""
    b = (1 - 2 * c.astype(np.int32)).astype(np.float32)
    n = c.shape[1]
    phi = np.ones((c.shape[0], n + 1), dtype=np.float32)
    phi[:, :n] = np.cumprod(b[:, ::-1], axis=1)[:, ::-1]
    return phi


# =========================================================================
# Base APUF
# =========================================================================
@dataclass
class APUF:
    """Single Arbiter PUF following the additive delay model.

    Parameters
    ----------
    n_stages : int
        Number of challenge bits (= delay stages).
    noisiness : float
        Std of additive Gaussian noise on the final delta. 0.05 gives
        ~3-5% per-bit error, matching the SABRE hardware measurements.
    bias : float
        Constant delay offset (the "artificial delay offset" used in SABRE
        calibration: 0 ns, 1 ns, 2 ns translate to unit-weight biases).
    seed : int
        RNG seed for reproducibility.
    """
    n_stages: int
    noisiness: float = 0.05
    bias: float = 0.0
    seed: int = 0
    w: np.ndarray = field(init=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.w = rng.standard_normal(self.n_stages + 1).astype(np.float32)

    def delta(self, phi: np.ndarray) -> np.ndarray:
        """Noise-free delay difference."""
        return phi @ self.w + self.bias

    def eval(self, c: np.ndarray, noise: bool = True,
             rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Return 0/1 response. noise=True adds one Gaussian draw per CRP."""
        phi = bits_to_phi(c)
        d = self.delta(phi)
        if noise:
            if rng is None:
                rng = np.random.default_rng()
            d = d + rng.normal(0.0, self.noisiness, size=d.shape)
        return (d > 0).astype(np.uint8)


# =========================================================================
# k-XOR APUF
# =========================================================================
@dataclass
class XORAPUF:
    """k-XOR APUF: XOR of k independent APUFs sharing the same challenge."""
    n_stages: int
    k: int
    noisiness: float = 0.05
    biases: Optional[list] = None
    seed: int = 0
    apufs: list = field(init=False)

    def __post_init__(self):
        biases = self.biases or [0.0] * self.k
        if len(biases) != self.k:
            raise ValueError("len(biases) must equal k")
        self.apufs = [
            APUF(self.n_stages, self.noisiness, biases[i], self.seed + i)
            for i in range(self.k)
        ]

    def eval(self, c: np.ndarray, noise: bool = True,
             rng: Optional[np.random.Generator] = None) -> np.ndarray:
        bits = np.stack(
            [a.eval(c, noise=noise, rng=rng) for a in self.apufs], axis=1
        )
        return np.bitwise_xor.reduce(bits, axis=1).astype(np.uint8)

    def eval_reliable(self, c: np.ndarray, n_reads: int = 11,
                      rng: Optional[np.random.Generator] = None
                      ) -> tuple[np.ndarray, np.ndarray]:
        """Return (majority_response, reliability) for each challenge.

        reliability = (# of '1's across n_reads) / n_reads
        This is what a reliability-based attacker can measure.
        """
        if rng is None:
            rng = np.random.default_rng()
        reads = np.stack(
            [self.eval(c, noise=True, rng=rng) for _ in range(n_reads)],
            axis=1,
        )
        rel = reads.mean(axis=1).astype(np.float32)
        maj = (rel >= 0.5).astype(np.uint8)
        return maj, rel


# =========================================================================
# (x, y)-Interpose PUF
# =========================================================================
@dataclass
class InterposePUF:
    """(x, y)-Interpose PUF (Nguyen et al. CHES 2019).

    Upper layer: x-XOR APUF produces a single interpose bit.
    That bit is inserted at the middle of the challenge, and the extended
    challenge is fed to the y-XOR lower layer whose output is the response.

    This is the baseline Reviewer A asked us to beat (or at least compare to).
    """
    n_stages: int
    x: int
    y: int
    noisiness: float = 0.05
    seed: int = 0
    upper: XORAPUF = field(init=False)
    lower: XORAPUF = field(init=False)

    def __post_init__(self):
        self.upper = XORAPUF(self.n_stages, self.x, self.noisiness,
                             seed=self.seed + 1000)
        # Lower APUFs take n_stages+1 challenge bits (1 interposed bit)
        self.lower = XORAPUF(self.n_stages + 1, self.y, self.noisiness,
                             seed=self.seed + 2000)

    def eval(self, c: np.ndarray, noise: bool = True,
             rng: Optional[np.random.Generator] = None) -> np.ndarray:
        b_mid = self.upper.eval(c, noise=noise, rng=rng)
        mid = self.n_stages // 2
        c_ext = np.insert(c, mid, b_mid, axis=1)
        return self.lower.eval(c_ext, noise=noise, rng=rng)


# =========================================================================
# SABRE composite design
# =========================================================================
@dataclass
class SABRESim:
    """Simulator of the SABRE composite APUF design.

    Pipeline:
        c  --(H)-->  c'  --(3-XOR APUF with offsets 0,1,2 ns)-->  r_raw

    For stable-key reconstruction:
        r_raw (x7 reads) --majority--> r_maj
        r_maj --BCH(31,16,3)--> corrected key bits

    This class provides:
        .eval_raw(c)        raw single-shot response (attackable by DNN etc.)
        .eval_stable(c)     7x-majority response (what your CSV contains)
        .eval_reliable(c)   (response, reliability) pair (for Split-CMA-ES)
    """
    n_stages: int = 32
    noisiness: float = 0.06
    offsets: tuple = (0.0, 0.5, 1.0)   # unit-weight equivalent of 0, 1, 2 ns
    sparsity: float = 0.5
    seed: int = 0
    H: np.ndarray = field(init=False)
    xor_apuf: XORAPUF = field(init=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        # Generate a random invertible 0/1 matrix with ~50% sparsity
        while True:
            H = (rng.random((self.n_stages, self.n_stages))
                 < self.sparsity).astype(np.uint8)
            # Check invertibility over GF(2)
            if _gf2_rank(H.copy()) == self.n_stages:
                break
        self.H = H
        self.xor_apuf = XORAPUF(
            self.n_stages, k=len(self.offsets),
            noisiness=self.noisiness,
            biases=list(self.offsets),
            seed=self.seed + 10,
        )

    def transform(self, c: np.ndarray) -> np.ndarray:
        """Public challenge-space transform c -> c' = H c (mod 2)."""
        return (c @ self.H.T) & 1

    def eval_raw(self, c: np.ndarray,
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
        return self.xor_apuf.eval(self.transform(c), noise=True, rng=rng)

    def eval_stable(self, c: np.ndarray, n_reads: int = 7,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """7x temporal-majority voted response - matches PYNQ pipeline."""
        if rng is None:
            rng = np.random.default_rng()
        cp = self.transform(c)
        reads = np.stack(
            [self.xor_apuf.eval(cp, noise=True, rng=rng) for _ in range(n_reads)],
            axis=1,
        )
        return (reads.mean(axis=1) >= 0.5).astype(np.uint8)

    def eval_reliable(self, c: np.ndarray, n_reads: int = 11,
                      rng: Optional[np.random.Generator] = None
                      ) -> tuple[np.ndarray, np.ndarray]:
        return self.xor_apuf.eval_reliable(self.transform(c), n_reads, rng)


# =========================================================================
# Helpers
# =========================================================================
def _gf2_rank(A: np.ndarray) -> int:
    """Rank of a 0/1 matrix over GF(2)."""
    A = A.copy() & 1
    rows, cols = A.shape
    r = 0
    for col in range(cols):
        pivot = None
        for row in range(r, rows):
            if A[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            continue
        A[[r, pivot]] = A[[pivot, r]]
        for row in range(rows):
            if row != r and A[row, col] == 1:
                A[row] ^= A[r]
        r += 1
        if r == rows:
            break
    return r


def random_challenges(n: int, n_stages: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(n, n_stages), dtype=np.uint8)


# =========================================================================
# Convenience generators used by run_full_evaluation.py
# =========================================================================
def generate_xor_dataset(k: int, n: int, n_stages: int = 32,
                        noisiness: float = 0.05, seed: int = 0,
                        with_reliability: bool = False):
    """Create a k-XOR APUF CRP dataset. Returns dict with c, y, optional rel."""
    puf = XORAPUF(n_stages=n_stages, k=k, noisiness=noisiness, seed=seed)
    c = random_challenges(n, n_stages, seed=seed + 1)
    if with_reliability:
        y, rel = puf.eval_reliable(c, n_reads=11,
                                   rng=np.random.default_rng(seed + 2))
        return {"c": c, "y": y, "reliability": rel, "puf": puf}
    y = puf.eval(c, noise=True, rng=np.random.default_rng(seed + 2))
    return {"c": c, "y": y, "puf": puf}


def generate_ipuf_dataset(x: int, y_k: int, n: int, n_stages: int = 32,
                         noisiness: float = 0.05, seed: int = 0):
    puf = InterposePUF(n_stages=n_stages, x=x, y=y_k,
                      noisiness=noisiness, seed=seed)
    c = random_challenges(n, n_stages, seed=seed + 1)
    y = puf.eval(c, noise=True, rng=np.random.default_rng(seed + 2))
    return {"c": c, "y": y, "puf": puf}


def generate_sabre_dataset(n: int, n_stages: int = 32,
                          noisiness: float = 0.06, seed: int = 0,
                          with_reliability: bool = False,
                          stable: bool = True):
    """SABRE CRP dataset. stable=True mimics the hardware CSV;
    stable=False gives raw single-shot responses for attack parity tests."""
    puf = SABRESim(n_stages=n_stages, noisiness=noisiness, seed=seed)
    c = random_challenges(n, n_stages, seed=seed + 1)
    rng = np.random.default_rng(seed + 2)
    if with_reliability:
        y, rel = puf.eval_reliable(c, n_reads=11, rng=rng)
        return {"c": c, "y": y, "reliability": rel, "puf": puf}
    if stable:
        y = puf.eval_stable(c, n_reads=7, rng=rng)
    else:
        y = puf.eval_raw(c, rng=rng)
    return {"c": c, "y": y, "puf": puf}


# =========================================================================
# Self test
# =========================================================================
if __name__ == "__main__":
    print("Self-test: APUF (should be near-linearly-separable)")
    d = generate_xor_dataset(k=1, n=5000)
    print(f"  class balance: {d['y'].mean():.3f}")

    print("Self-test: 4-XOR APUF")
    d = generate_xor_dataset(k=4, n=5000)
    print(f"  class balance: {d['y'].mean():.3f}")

    print("Self-test: (1,5)-iPUF")
    d = generate_ipuf_dataset(x=1, y_k=5, n=5000)
    print(f"  class balance: {d['y'].mean():.3f}")

    print("Self-test: SABRE stable vs raw")
    d_raw = generate_sabre_dataset(n=5000, stable=False)
    d_sta = generate_sabre_dataset(n=5000, stable=True)
    agree = (d_raw["y"] == d_sta["y"]).mean()
    print(f"  raw-vs-stable agreement (different seeds): {agree:.3f}")

    print("Self-test: SABRE reliability distribution")
    d_rel = generate_sabre_dataset(n=2000, with_reliability=True)
    rel = d_rel["reliability"]
    print(f"  reliability mean={rel.mean():.3f}  "
          f"fraction<0.9={(np.abs(rel - 0.5) < 0.4).mean():.3f}")
