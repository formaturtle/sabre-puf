"""
make_phi.py
===========

Build the parity (Phi) feature matrix for a 32-stage Arbiter PUF from
packed 32-bit challenges.

Background
----------
The classical additive linear delay model of a k-stage Arbiter PUF is:

    Delta_delay = w . Phi(c) + bias

where Phi(c) in {-1, +1}^(k+1) is the parity vector

    Phi_i(c) = prod_{j=i}^{k-1} (1 - 2 * c_j)     for i in [0, k-1]
    Phi_k(c) = 1                                   (bias term)

Equivalently, if we define b_j = 1 - 2*c_j in {+1, -1}, then

    Phi_i = b_i * b_{i+1} * ... * b_{k-1}

This is the feature representation that LR, SVM, CMA-ES, and MLP
modeling attacks consume.

This module exposes two helpers:

    phi_from_uint32(c, k=32) -> int8 (N, k+1)
    phi_from_bits(c_bits)    -> int8 (N, k+1)

Both are vectorized with NumPy so they run at ~1e6 CRP/s on a laptop.

We also provide:

    challenge_bits_from_uint32(c, k=32) -> uint8 (N, k)

to unpack a uint32 into LSB-first bit arrays, which is the orientation
used by your Verilog (challenge[i] selects stage i).
"""
from __future__ import annotations

import numpy as np


def challenge_bits_from_uint32(c: np.ndarray, k: int = 32) -> np.ndarray:
    """Unpack uint32 challenges to uint8 bit arrays.

    Orientation matches your Verilog: bit 0 is LSB and is consumed by
    stage 0 of the arbiter chain, i.e. `challenge[0]` in Verilog.

    Parameters
    ----------
    c : uint32 array shape (N,)
    k : int, number of stages. Must be <= 32.

    Returns
    -------
    bits : uint8 shape (N, k)
        bits[:, 0] = LSB, bits[:, k-1] = MSB bit of the k-bit slice.
    """
    c = np.asarray(c, dtype=np.uint32)
    if k > 32:
        raise ValueError("k must be <= 32 for uint32 input")
    # Bit i = (c >> i) & 1
    shifts = np.arange(k, dtype=np.uint32)
    bits = ((c[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
    return bits


def phi_from_bits(c_bits: np.ndarray) -> np.ndarray:
    """Build the parity feature matrix from 0/1 bit arrays.

    Parameters
    ----------
    c_bits : uint8 or bool shape (N, k)

    Returns
    -------
    phi : int8 shape (N, k+1)
        Entries are +1 or -1. Last column is the bias (+1).
    """
    c_bits = np.asarray(c_bits)
    if c_bits.ndim != 2:
        raise ValueError("c_bits must be 2-D (N, k)")
    N, k = c_bits.shape

    # b_j = 1 - 2*c_j in {+1, -1}
    b = (1 - 2 * c_bits.astype(np.int16)).astype(np.int8)  # (N, k)

    # Phi_i = prod_{j=i}^{k-1} b_j  -> reverse cumulative product
    # Use int16 for the cumulative product to avoid int8 overflow (product
    # of +-1 values is always +-1 so technically safe, but safer in int16),
    # then cast back to int8.
    b_rev = b[:, ::-1].astype(np.int16)
    cp_rev = np.cumprod(b_rev, axis=1, dtype=np.int16)
    phi_k = cp_rev[:, ::-1]  # shape (N, k)

    bias = np.ones((N, 1), dtype=np.int16)
    phi = np.concatenate([phi_k, bias], axis=1).astype(np.int8)
    return phi


def phi_from_uint32(c: np.ndarray, k: int = 32) -> np.ndarray:
    """Convenience: uint32 -> Phi in one call."""
    return phi_from_bits(challenge_bits_from_uint32(c, k=k))


def apply_H_matrix(c: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply the public 32x32 binary sparsity matrix H to raw challenges.

    Parameters
    ----------
    c : uint32 array shape (N,) OR bit array shape (N, k)
    H : uint8/int shape (k, k), binary entries

    Returns
    -------
    c_h_bits : uint8 shape (N, k)
        The post-H challenge bits, same orientation as input bits.

    Notes
    -----
    c_h = H . c  mod 2. This mirrors what `sparsity_h.v` does in
    hardware. You should use `c_h` (not the raw `c`) when computing Phi
    for attacks that assume knowledge of H.
    """
    if c.ndim == 1:
        c_bits = challenge_bits_from_uint32(c, k=H.shape[0])
    else:
        c_bits = np.asarray(c, dtype=np.uint8)

    H = np.asarray(H, dtype=np.uint8) & 1
    # Matrix multiply mod 2: use int, mod 2 at end.
    prod = c_bits.astype(np.int32) @ H.T.astype(np.int32)
    return (prod & 1).astype(np.uint8)


# ---------------------------------------------------------------------
# Smoke test when run as a script
# ---------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    c = rng.integers(0, 1 << 32, size=5, dtype=np.uint32)
    bits = challenge_bits_from_uint32(c)
    phi = phi_from_bits(bits)
    print("c (hex)  :", [f"0x{int(x):08X}" for x in c])
    print("bits[:,:8]:")
    print(bits[:, :8])
    print("phi[:, -5:]:")
    print(phi[:, -5:])
    print("phi shape:", phi.shape, "dtype:", phi.dtype)
    assert phi.shape == (5, 33)
    assert set(np.unique(phi).tolist()) <= {-1, 1}
    print("OK")
