#!/usr/bin/env python3
"""
collect_crps.py
===============

Runs on the PYNQ Z1 board. Collects Challenge-Response Pairs (CRPs) from
the Hybrid SR-CAPUF overlay and saves them to sharded NumPy archives for
later analysis on a PC.

Usage::

    # Smoke test: 1000 CRPs, 11 reads per challenge
    python3 collect_crps.py --n 1000 --repeats 11 --out /home/xilinx/crps_smoke

    # Full run: 1,000,000 CRPs in ten shards of 100,000
    python3 collect_crps.py --n 1000000 --repeats 11 --shard 100000 \
        --out /home/xilinx/crps_run1

Each shard produces:
    shard_0000.npz  with keys:
        c_bin : uint32 shape (S,)        raw 32-bit challenges as sent
        r     : uint8  shape (S,)        majority-voted response
        r_rel : uint8  shape (S,)        reliability count in [0, repeats]
        raw   : uint8  shape (S, repeats) individual reads

And one top-level `meta.json` with timing, seed fingerprint, bitfile hash,
and the PRNG seed so the challenge set is reproducible on the PC.

Design choices
--------------
- Challenges are drawn from a seeded numpy RNG so the set is fully
  reproducible. We save only the RNG seed and shard index in metadata;
  the PC can regenerate the challenge set if needed.
- A small held-out "test split" of ~10% of the total is reserved for
  attack evaluation. The training/test split is determined by position
  within the reproducible challenge sequence, not by a second RNG draw.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

# Ensure the driver module is importable from the same folder.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sr_capuf_driver import SRCAPUFDriver  # noqa: E402


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bitfile", default="/home/xilinx/overlays/sr_capuf.bit",
                    help="Path to the .bit overlay (needs matching .hwh)")
    ap.add_argument("--n", type=int, default=1_000_000,
                    help="Total number of CRPs to collect")
    ap.add_argument("--repeats", type=int, default=11,
                    help="Number of reads per challenge (temporal majority vote)")
    ap.add_argument("--shard", type=int, default=100_000,
                    help="CRPs per output shard")
    ap.add_argument("--seed", type=int, default=0xC0FFEE,
                    help="Challenge-sequence PRNG seed (keep for reproducibility)")
    ap.add_argument("--out", required=True,
                    help="Output directory (will be created)")
    ap.add_argument("--no-provision-seed", action="store_true",
                    help="Do not overwrite the BRAM seed (useful if you want "
                         "to re-run with an already-provisioned seed)")
    ap.add_argument("--bram-seed", type=lambda x: int(x, 0), default=None,
                    help="Specific 32-bit BRAM seed to write. Otherwise random.")
    args = ap.parse_args()

    if args.n <= 0 or args.shard <= 0 or args.repeats <= 0:
        print("ERROR: --n, --shard, --repeats must all be positive", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)

    # ---------- Load overlay ----------
    print(f"[collect_crps] loading overlay: {args.bitfile}")
    drv = SRCAPUFDriver(bitfile=args.bitfile)

    # ---------- Seed handling ----------
    seed_written = None
    if not args.no_provision_seed:
        seed_written = drv.provision_seed(seed=args.bram_seed)
        if seed_written is not None:
            print(f"[collect_crps] provisioned BRAM seed = 0x{seed_written:08X}")
        else:
            print("[collect_crps] BRAM seed channel unavailable (pre-Fix bitstream)")
    seed_fp = drv.seed_fingerprint()
    print(f"[collect_crps] seed fingerprint = {seed_fp}")

    # ---------- Self-test ----------
    print("[collect_crps] running quick self-test...")
    st = drv.self_test()
    for k, v in st.items():
        print(f"    {k}: {v}")
    if not st.get("response_variety", False):
        print("[collect_crps] WARNING: response did not vary across 64 random "
              "challenges; hardware may be stuck or done handshake broken.",
              file=sys.stderr)

    # ---------- Challenge generator ----------
    rng = np.random.default_rng(args.seed)
    # We draw in bulk per shard for efficiency.

    # ---------- Metadata scaffold ----------
    meta = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "bitfile": os.path.abspath(args.bitfile),
        "bitfile_sha256": _sha256_file(args.bitfile),
        "bram_seed_fingerprint": seed_fp,
        "bram_seed_value": (None if seed_written is None else f"0x{seed_written:08X}"),
        "n_total": args.n,
        "n_per_shard": args.shard,
        "repeats": args.repeats,
        "prng_seed": args.seed,
        "shards": [],
        "test_split_fraction": 0.10,
        "host": {
            "platform": sys.platform,
            "python": sys.version.split()[0],
        },
    }

    # ---------- Collection ----------
    n_done = 0
    shard_idx = 0
    t_start = time.perf_counter()
    while n_done < args.n:
        cur = min(args.shard, args.n - n_done)
        print(f"[collect_crps] shard {shard_idx:04d}: "
              f"collecting {cur} CRPs (total so far {n_done}/{args.n})")

        challenges = rng.integers(0, 1 << 32, size=cur, dtype=np.uint32)
        r_vote, r_rel, raw = drv.query_batch(
            challenges, repeats=args.repeats, progress_every=max(1, cur // 10))

        shard_path = os.path.join(args.out, f"shard_{shard_idx:04d}.npz")
        np.savez_compressed(
            shard_path,
            c_bin=challenges,
            r=r_vote,
            r_rel=r_rel,
            raw=raw,
        )
        meta["shards"].append({
            "index": shard_idx,
            "path": os.path.basename(shard_path),
            "n": int(cur),
            "sha256": _sha256_file(shard_path),
        })
        n_done += cur
        shard_idx += 1

    meta["wall_time_seconds"] = time.perf_counter() - t_start

    # Write metadata last so a crashed run leaves no false meta.json.
    meta_path = os.path.join(args.out, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[collect_crps] DONE. Wrote {shard_idx} shards + meta.json to {args.out}")
    print(f"[collect_crps] Wall time: {meta['wall_time_seconds']/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
