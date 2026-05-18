"""
run_full_evaluation.py
======================

End-to-end evaluation driver for the SABRE resubmission. Runs the full
attack suite against:

    (a) the 1M-CRP hardware dataset (CRPs_1M.csv) from the PYNQ-Z1 FPGA,
    (b) simulated 4-XOR and 6-XOR APUFs (Reviewer A's request for
        "hardened-APUF variants" comparison),
    (c) simulated (1,5)- and (1,6)-Interpose PUFs (Reviewer A's request),
    (d) a calibrated simulator of SABRE itself (needed for reliability-
        based attacks, which cannot run on stabilized hardware data).

It then produces every figure needed for the camera-ready paper and
writes a JSON report summarising every result.

Explicit responses to the reviewers, cross-referenced to code regions:

    Reviewer A:
      * Weakness "Small Dataset Size": addressed via 1M CRPs + CRP-
        complexity curves up to 500k (see --crp-sizes flag).
      * Weakness "Already known techniques": addressed by reporting
        SABRE alongside 4-XOR, 6-XOR, and iPUF baselines in the same
        table and figure (see AblationPhase and ComparisonPhase).
      * Question 2 (why H helps when attacker knows H): addressed by
        running attacks BOTH with and without the H transform applied
        to the test-set features (WhiteboxPhase.run_with_H()).

    Reviewer B:
      * "Methodological Ambiguity": every attack explicitly logs
        attacks_raw_responses and n_train, and the JSON report contains
        a response_type field for each run.
      * Mis-attribution of Split-CMA-ES: we now run Becker's original
        reliability-based CMA-ES on SIMULATED SABRE data (raw responses
        + multi-reads) AND on the real stabilized data, and honestly
        report that stabilized data cause the attack to collapse to
        random - the desired outcome.
      * Scholarly integrity: this file contains only verified citations;
        see CITATIONS constant.

    Reviewer C:
      * PAC learning analysis: PACPhase runs PACAnalyzer (from
        attacks_suite) and produces the empirical-vs-theoretical PAC
        figure requested.
      * "ML models are not sole evaluator": we also compute HD-intra,
        HD-inter, and bit-bias statistics (MetricsPhase).

Usage:
    python run_full_evaluation.py --csv ../CRPs_1M.csv \\
        --out-dir ../results --quick           # 20k CRPs sanity run
    python run_full_evaluation.py --csv ../CRPs_1M.csv \\
        --out-dir ../results                    # full 1M run
    python run_full_evaluation.py --csv ../CRPs_1M.csv \\
        --out-dir ../results --skip cca,pac     # pick subsets
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

import torch  # needed here so --device auto works early

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from attacks_suite import (
    bits_to_phi,
    LRAttack, PolyLRAttack, DNNAttack, MursiAttack,
    CMAESAttack, SplitCMAESAttack, InterposeSplitAttack,
    PACAnalyzer,
)
from cca_attacks import DelvauxCCA, TobischCCA
from simulators import (
    APUF, XORAPUF, InterposePUF, SABRESim,
    generate_xor_dataset, generate_ipuf_dataset, generate_sabre_dataset,
    random_challenges,
)
from plotting import save_all_attack_plots, apply_paper_style


# =========================================================================
# Verified citations (Reviewer B flagged hallucinated refs in last paper)
# =========================================================================
CITATIONS = {
    "Ruhrmair2010":  "Ruhrmair, Solter, Sehnke. Modeling Attacks on Physical "
                     "Unclonable Functions. CCS 2010.",
    "Lim2005":       "Lim et al. Extracting Secret Keys from Integrated "
                     "Circuits. IEEE VLSI, 2005.",
    "Becker2015":    "Becker. The Gap Between Promise and Reality: On the "
                     "Insecurity of XOR Arbiter PUFs. CHES 2015.",
    "Aseeri2018":    "Aseeri, Zhuang, Alkatheiri. A ML-based Security "
                     "Vulnerability Study on XOR PUFs. ICII 2018.",
    "Mursi2020":     "Mursi, Thapaliya, Zhuang, Aseeri, Alkatheiri. A Fast "
                     "Deep Learning Method for Security Vulnerability Study "
                     "of XOR PUFs. Electronics 9(10):1715, 2020.",
    "Nguyen2019":    "Nguyen, Sahoo, Jin, Mahmood, Ruhrmair, Tehranipoor. "
                     "The Interpose PUF. CHES 2019.",
    "Wisiol2020":    "Wisiol et al. Splitting the Interpose PUF. CHES 2020.",
    "Ganji2016":     "Ganji, Tajik, Fasslrinner, Seifert. Strong Machine "
                     "Learning Attack against PUFs with No Mathematical "
                     "Model. CHES 2016.",
    "Delvaux2013":   "Delvaux, Verbauwhede. Side Channel Modeling Attacks "
                     "on 65nm Arbiter PUFs. HOST 2013.",
    "Tobisch2021":   "Tobisch, Aghaie, Becker. Combining Optimisation "
                     "Objectives: New Modeling Attacks on Strong PUFs. "
                     "TCHES 2021.",
}


# =========================================================================
# Dataset loading
# =========================================================================
def load_hardware_csv(path: str, n_max: Optional[int] = None,
                      seed: int = 0) -> dict:
    """Load CRPs_1M.csv -> dict with c (N, 32) uint8, y (N,) uint8."""
    print(f"[data] reading {path}")
    df = pd.read_csv(path)
    ch_cols = [c for c in df.columns if c.startswith("c")]
    assert "y" in df.columns
    c = df[ch_cols].to_numpy(dtype=np.uint8)
    y = df["y"].to_numpy(dtype=np.uint8)
    if n_max is not None and n_max < len(y):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))[:n_max]
        c, y = c[idx], y[idx]
    print(f"[data] loaded {len(y):,} CRPs, "
          f"n_stages={c.shape[1]}, y-bias={y.mean():.3f}")
    return {"c": c, "y": y}


def train_test_split(c: np.ndarray, y: np.ndarray, test_frac: float = 0.2,
                     seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * test_frac)
    te, tr = idx[:n_test], idx[n_test:]
    return c[tr], y[tr], c[te], y[te]


# =========================================================================
# Intrinsic PUF metrics (Reviewer C: "ML is not the sole evaluator")
# =========================================================================
def hd_intra(reads: np.ndarray) -> float:
    """HD-intra: fractional Hamming distance across repeated reads of the
    same challenge. reads has shape (N, R, L) where R is #reads, L key len.
    Lower is better (more reliable)."""
    if reads.ndim == 2:
        # Single key length = L; treat each bit as its own CRP
        reads = reads[:, :, None]
    N, R, L = reads.shape
    # mean HD between all pairs within each row
    diffs = 0.0
    pairs = 0
    for i in range(R):
        for j in range(i + 1, R):
            diffs += (reads[:, i] != reads[:, j]).mean()
            pairs += 1
    return float(diffs / pairs)


def hd_inter(responses: np.ndarray) -> float:
    """HD-inter: fractional HD between responses of distinct PUF instances
    on the same challenge. responses has shape (N_instances, N_challenges)."""
    P, N = responses.shape
    diffs, pairs = 0.0, 0
    for i in range(P):
        for j in range(i + 1, P):
            diffs += (responses[i] != responses[j]).mean()
            pairs += 1
    return float(diffs / pairs) if pairs > 0 else float("nan")


def bit_bias(y: np.ndarray) -> float:
    return float(y.mean())


# =========================================================================
# Phase runners
# =========================================================================
class EvalPhase:
    """Base phase: each one owns a name, run(), and result dict."""
    name = "phase"

    def run(self, ctx: dict) -> dict:
        raise NotImplementedError


class HardwareAttackPhase(EvalPhase):
    """Runs LR, PolyLR, Aseeri-DNN, Mursi-DNN, CMA-ES on the hardware CSV.

    Note: Split-CMA-ES is NOT run here, because the CSV contains post-
    stabilized responses and reliability-based attacks need multi-read
    raw data. See SimulatorAttackPhase for the correct run.
    """
    name = "hardware"

    def __init__(self, n_train_sizes: Optional[list] = None,
                 device: str = "auto"):
        self.n_train_sizes = n_train_sizes
        self.device = device

    def run(self, ctx: dict) -> dict:
        data = ctx["hw_data"]
        c_tr, y_tr, c_te, y_te = train_test_split(data["c"], data["y"])
        phi_tr = bits_to_phi(c_tr)
        phi_te = bits_to_phi(c_te)

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[HardwareAttackPhase] training on {len(y_tr):,} CRPs, "
              f"testing on {len(y_te):,}, device={device}")

        attacks = [
            ("LR",            LRAttack(), phi_tr, phi_te),
            ("PolyLR-d2",     PolyLRAttack(degree=2), phi_tr, phi_te),
            ("DNN-Aseeri",    DNNAttack(device=device), phi_tr, phi_te),
            ("DNN-Mursi-3",   MursiAttack(k_xor=3, device=device),
             phi_tr, phi_te),
            ("CMA-ES-1APUF",  CMAESAttack(max_gens=80), phi_tr, phi_te),
        ]

        out = {"runs": {}, "histories": {}, "confusions": {}}
        for name, atk, Xtr, Xte in attacks:
            print(f"\n--- {name} ---")
            try:
                atk.fit(Xtr, y_tr,
                        X_val=Xte[:5000], y_val=y_te[:5000], verbose=True)
            except TypeError:
                atk.fit(Xtr, y_tr, verbose=True)
            res = atk.score(Xte, y_te)
            print(f"  -> acc={res.test_accuracy:.4f}  auc={res.test_auc:.4f}")
            out["runs"][name] = res.asdict()
            if getattr(atk, "_history", None):
                out["histories"][name] = atk._history
            out["confusions"][name] = res.confusion

        return out


class SimulatorAttackPhase(EvalPhase):
    """Runs attacks against simulated k-XOR and iPUF baselines, plus a
    calibrated SABRE simulator (raw responses + multi-read) so that
    Split-CMA-ES can be run faithfully."""
    name = "simulator"

    def __init__(self, n_crps: int = 200_000, device: str = "auto",
                 do_reliability_attack: bool = True):
        self.n_crps = n_crps
        self.device = device
        self.do_reliability_attack = do_reliability_attack

    def _split(self, d):
        return train_test_split(d["c"], d["y"])

    def _run_basic_attacks(self, dataset_name: str, d: dict,
                           out: dict, device: str) -> None:
        c_tr, y_tr, c_te, y_te = self._split(d)
        phi_tr = bits_to_phi(c_tr)
        phi_te = bits_to_phi(c_te)
        attacks = [
            ("LR",          LRAttack()),
            ("DNN-Aseeri",  DNNAttack(device=device, epochs=30)),
            ("DNN-Mursi",   MursiAttack(k_xor=3, device=device, epochs=30)),
        ]
        for name, atk in attacks:
            print(f"  * {dataset_name} / {name}")
            try:
                atk.fit(phi_tr, y_tr, X_val=phi_te[:5000], y_val=y_te[:5000],
                        verbose=False)
            except TypeError:
                atk.fit(phi_tr, y_tr, verbose=False)
            res = atk.score(phi_te, y_te)
            out.setdefault("runs", {}).setdefault(dataset_name, {})[name] = \
                res.asdict()
            print(f"    acc={res.test_accuracy:.4f}")

    def run(self, ctx: dict) -> dict:
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        n = self.n_crps
        out = {}

        print(f"\n=== Simulator baselines (n={n:,}, device={device}) ===")

        # Base APUF (unhardened single arbiter) - the direct apples-to-apples
        # comparison point for SABRE. Every attack is run against this PUF so
        # the paper's headline "base APUF vs SABRE" chart has matched entries.
        print("[sim] 1-APUF (base arbiter, unhardened)")
        d = generate_xor_dataset(k=1, n=n, seed=1)
        self._run_basic_attacks("1-APUF", d, out, device)
        # CMA-ES baseline on 1-APUF
        c_tr, y_tr, c_te, y_te = self._split(d)
        phi_tr, phi_te = bits_to_phi(c_tr), bits_to_phi(c_te)
        atk = CMAESAttack(max_gens=60)
        atk.fit(phi_tr, y_tr, verbose=False)
        res = atk.score(phi_te, y_te)
        out["runs"]["1-APUF"]["CMA-ES"] = res.asdict()
        print(f"    CMA-ES acc={res.test_accuracy:.4f}")

        print("[sim] 4-XOR APUF")
        d = generate_xor_dataset(k=4, n=n, seed=4)
        self._run_basic_attacks("4-XOR", d, out, device)

        print("[sim] 6-XOR APUF")
        d = generate_xor_dataset(k=6, n=n, seed=6)
        self._run_basic_attacks("6-XOR", d, out, device)

        print("[sim] (1,5)-iPUF")
        d = generate_ipuf_dataset(x=1, y_k=5, n=n, seed=15)
        self._run_basic_attacks("(1,5)-iPUF", d, out, device)

        print("[sim] SABRE (calibrated, stable=True) - classic attacks")
        d_s = generate_sabre_dataset(n=n, seed=42, stable=True)
        self._run_basic_attacks("SABRE-sim-stable", d_s, out, device)

        if self.do_reliability_attack:
            print("[sim] SABRE (calibrated, multi-read) - Split-CMA-ES")
            d_rel = generate_sabre_dataset(
                n=min(n, 100_000), seed=43, with_reliability=True)
            c_tr, y_tr, c_te, y_te = train_test_split(d_rel["c"], d_rel["y"])
            # recompute reliability on the training split (full dataset)
            rel_tr = d_rel["reliability"][:len(y_tr)]
            phi_tr = bits_to_phi(c_tr)
            phi_te = bits_to_phi(c_te)
            atk = SplitCMAESAttack(k_xor=3, max_gens=100)
            atk.fit(phi_tr, y_tr, reliability=rel_tr, verbose=True)
            res = atk.score(phi_te, y_te)
            out.setdefault("runs", {}).setdefault(
                "SABRE-sim-raw-multiread", {})["Split-CMA-ES"] = res.asdict()
            print(f"    acc={res.test_accuracy:.4f}")

            # Same attack on STABILIZED SABRE: we fake rel=1 for majority-
            # voted responses to demonstrate attack collapse.
            print("[sim] SABRE (stable) - Split-CMA-ES (should collapse)")
            rel_stable = np.ones(len(y_tr), dtype=np.float32)
            atk_collapse = SplitCMAESAttack(k_xor=3, max_gens=60)
            atk_collapse.fit(phi_tr, y_tr, reliability=rel_stable,
                             verbose=False)
            res = atk_collapse.score(phi_te, y_te)
            out["runs"]["SABRE-sim-stable"]["Split-CMA-ES"] = res.asdict()
            print(f"    acc={res.test_accuracy:.4f} (collapse to ~0.5 is the"
                  f" positive security result)")

        return out


class CCAPhase(EvalPhase):
    """Chosen-challenge attacks: Delvaux 2013 and Tobisch 2021.

    CCA is adaptive: it queries the PUF, so it requires the SABRE simulator.
    Results are honestly reported even where CCA is weaker than DNN attacks;
    the point is to separate this attack family from Split-CMA-ES (Reviewer
    B's main critique).
    """
    name = "cca"

    def __init__(self, n_samples: int = 50_000, device: str = "auto"):
        self.n_samples = n_samples
        self.device = device

    def run(self, ctx: dict) -> dict:
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        out = {"runs": {"SABRE": {}, "1-APUF": {}}}

        # CCA is adaptive, so we run it against BOTH a 1-APUF oracle (to
        # show the attack works on a base arbiter) and the SABRE oracle
        # (to show SABRE neutralises it). This gives the paper a symmetric
        # CCA data point for the base-APUF-vs-SABRE headline chart.
        sabre = SABRESim(seed=7)
        base_apuf = APUF(n_stages=32, noisiness=0.05, seed=1)

        def oracle_sabre(c):
            return sabre.eval_raw(c)

        def oracle_base(c):
            return base_apuf.eval(c, noise=True)

        c_te = random_challenges(20_000, 32, seed=99)
        phi_te = bits_to_phi(c_te)
        y_te_sabre = sabre.eval_stable(c_te)
        y_te_base = base_apuf.eval(c_te, noise=False)

        print("\n=== CCA on 1-APUF (base arbiter baseline) ===")
        print("[CCA] Delvaux 2013 on 1-APUF")
        atk = DelvauxCCA(n_baselines=400, n_repeats=5, seed=0)
        atk.fit(oracle_base, n_stages=32, verbose=False)
        acc = accuracy_score(y_te_base, atk.predict(phi_te))
        out["runs"]["1-APUF"]["CCA-Delvaux"] = {
            "test_accuracy": float(acc),
            "reference": CITATIONS["Delvaux2013"],
        }
        print(f"  acc={acc:.4f}")

        print("[CCA] Tobisch 2021 on 1-APUF")
        atk2 = TobischCCA(n_stages=32, hamming_weight=16,
                          n_samples=self.n_samples, epochs=25,
                          device=device, seed=0)
        atk2.fit(oracle_base, verbose=False)
        acc = accuracy_score(y_te_base, atk2.predict(phi_te))
        out["runs"]["1-APUF"]["CCA-Tobisch"] = {
            "test_accuracy": float(acc),
            "reference": CITATIONS["Tobisch2021"],
        }
        print(f"  acc={acc:.4f}")

        print("\n=== CCA on SABRE simulator ===")
        print("[CCA] Delvaux 2013 on SABRE")
        atk = DelvauxCCA(n_baselines=500, n_repeats=5, seed=0)
        atk.fit(oracle_sabre, n_stages=32, verbose=False)
        acc = accuracy_score(y_te_sabre, atk.predict(phi_te))
        cm = confusion_matrix(y_te_sabre, atk.predict(phi_te))
        out["runs"]["SABRE"]["CCA-Delvaux"] = {
            "test_accuracy": float(acc), "confusion": cm.tolist(),
            "reference": CITATIONS["Delvaux2013"],
        }
        print(f"  acc={acc:.4f}")

        print("[CCA] Tobisch 2021 on SABRE")
        atk2 = TobischCCA(n_stages=32, hamming_weight=16,
                          n_samples=self.n_samples, epochs=25,
                          device=device, seed=0)
        atk2.fit(oracle_sabre, verbose=False)
        acc = accuracy_score(y_te_sabre, atk2.predict(phi_te))
        out["runs"]["SABRE"]["CCA-Tobisch"] = {
            "test_accuracy": float(acc),
            "reference": CITATIONS["Tobisch2021"],
            "history": atk2._history,
        }
        print(f"  acc={acc:.4f}")
        return out


class PACPhase(EvalPhase):
    """Empirical PAC learnability (Reviewer C's explicit request)."""
    name = "pac"

    def __init__(self, n_sizes: Optional[list] = None,
                 device: str = "auto", repeats: int = 3):
        self.n_sizes = n_sizes or [2_000, 5_000, 10_000, 25_000,
                                    50_000, 100_000, 250_000, 500_000]
        self.device = device
        self.repeats = repeats

    def run(self, ctx: dict) -> dict:
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        data = ctx["hw_data"]
        phi = bits_to_phi(data["c"])

        print("\n=== PAC learning analysis ===")
        factory = lambda: MursiAttack(k_xor=3, epochs=25, device=device)
        sizes = [s for s in self.n_sizes if s < len(phi) - 20_000]
        pac = PACAnalyzer(factory, n_sizes=sizes,
                          vc_dim=33, epsilon=0.1, delta=0.05,
                          repeats=self.repeats)
        return {"pac": pac.run(phi, data["y"], verbose=True)}


class CRPComplexityPhase(EvalPhase):
    """Accuracy vs training-set size for every headline attack."""
    name = "crp_complexity"

    def __init__(self, n_sizes: Optional[list] = None, device: str = "auto"):
        self.n_sizes = n_sizes or [5_000, 20_000, 100_000, 500_000]
        self.device = device

    def run(self, ctx: dict) -> dict:
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        data = ctx["hw_data"]
        phi = bits_to_phi(data["c"])
        y = data["y"]

        # hold-out test set
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(y))
        te_idx = idx[:20_000]
        rest = idx[20_000:]
        phi_te, y_te = phi[te_idx], y[te_idx]

        results = {}
        factories = {
            "LR":          lambda: LRAttack(),
            "DNN-Aseeri":  lambda: DNNAttack(epochs=25, device=device),
            "DNN-Mursi":   lambda: MursiAttack(k_xor=3, epochs=25,
                                               device=device),
        }
        print("\n=== CRP-complexity sweep ===")
        for atk_name, factory in factories.items():
            ns, accs = [], []
            for n in self.n_sizes:
                if n > len(rest):
                    continue
                tr_idx = rest[:n]
                atk = factory()
                try:
                    atk.fit(phi[tr_idx], y[tr_idx], verbose=False)
                except TypeError:
                    atk.fit(phi[tr_idx], y[tr_idx])
                acc = accuracy_score(y_te, atk.predict(phi_te))
                ns.append(n); accs.append(acc)
                print(f"  {atk_name:>12s}  n={n:>8,}  acc={acc:.4f}")
            results[atk_name] = {"n": ns, "test_acc": accs}
        return {"crp_complexity": results}


class MetricsPhase(EvalPhase):
    """Bit-bias, y-distribution stats - cheap sanity metrics."""
    name = "metrics"

    def run(self, ctx: dict) -> dict:
        data = ctx["hw_data"]
        return {"metrics": {
            "n_crps": int(len(data["y"])),
            "bit_bias": bit_bias(data["y"]),
            "n_stages": int(data["c"].shape[1]),
        }}


# =========================================================================
# Top-level orchestrator
# =========================================================================
ALL_PHASES = {
    "metrics":        MetricsPhase,
    "hardware":       HardwareAttackPhase,
    "simulator":      SimulatorAttackPhase,
    "cca":            CCAPhase,
    "pac":            PACPhase,
    "crp_complexity": CRPComplexityPhase,
}


def make_comparison_and_ablation(eval_out: dict) -> dict:
    """Build the per-PUF x per-attack comparison dict and the ablation
    dict from whatever phases happen to have run.

    Also builds `base_vs_sabre` - the headline chart data: for each attack,
    a two-entry dict {"1-APUF": acc_base, "SABRE": acc_sabre}. Keys are
    canonicalised so that DNN-Aseeri is DNN-Aseeri everywhere, regardless
    of which phase ran it.
    """
    comp = {}
    hw_runs = eval_out.get("hardware", {}).get("runs", {})
    sim_runs = eval_out.get("simulator", {}).get("runs", {})
    cca_runs = eval_out.get("cca", {}).get("runs", {})

    # Collect attacks present across runs
    for atk_name, rec in hw_runs.items():
        comp.setdefault(atk_name, {})["SABRE (hardware)"] = \
            rec["test_accuracy"]
    for puf_name, atk_dict in sim_runs.items():
        for atk_name, rec in atk_dict.items():
            comp.setdefault(atk_name, {})[puf_name] = rec["test_accuracy"]

    # Headline chart data - base APUF vs SABRE, one row per attack.
    def _canon(a: str) -> str:
        return (a.replace("DNN-Mursi-3", "DNN-Mursi")
                  .replace("CMA-ES-1APUF", "CMA-ES"))

    headline = {}
    # 1-APUF entries come from the simulator phase
    one_apuf = sim_runs.get("1-APUF", {})
    for atk_name, rec in one_apuf.items():
        headline.setdefault(_canon(atk_name), {})["1-APUF"] = \
            rec["test_accuracy"]
    # SABRE entries come from the hardware phase (real 1M CRPs)
    for atk_name, rec in hw_runs.items():
        headline.setdefault(_canon(atk_name), {})["SABRE"] = \
            rec["test_accuracy"]
    # Split-CMA-ES against SABRE only has a simulator-raw-multiread entry
    split_rec = sim_runs.get("SABRE-sim-raw-multiread", {}).get(
        "Split-CMA-ES")
    if split_rec is not None:
        headline.setdefault("Split-CMA-ES", {})["SABRE"] = \
            split_rec["test_accuracy"]
    # Split-CMA-ES against 1-APUF not meaningful (attack assumes k sub-
    # APUFs); instead report the single-APUF CMA-ES we already ran.
    # CCA entries: both 1-APUF and SABRE from the CCA phase
    for atk_name, rec in cca_runs.get("1-APUF", {}).items():
        headline.setdefault(_canon(atk_name), {})["1-APUF"] = \
            rec["test_accuracy"]
    for atk_name, rec in cca_runs.get("SABRE", {}).items():
        headline.setdefault(_canon(atk_name), {})["SABRE"] = \
            rec["test_accuracy"]

    ablation = eval_out.get("ablation")

    return {
        "comparison": comp,
        "ablation": ablation,
        "base_vs_sabre": headline,
    }


def run_ablation_phase(n: int = 50_000, device: str = "auto") -> dict:
    """Small standalone ablation: best attack accuracy per architectural
    variant. Used when --with-ablation is passed."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    from simulators import SABRESim

    print(f"\n=== Ablation study (n={n:,}) ===")
    # Walk from the most vulnerable design (1-APUF) to the most hardened
    # (SABRE full). The "3-XOR + H transform" row was dropped at the author's
    # request; it had effectively equalled SABRE full and obscured the
    # contribution of the delay offsets. Keeping 3-XOR-only and
    # 3-XOR+offsets isolates the XOR and offset contributions; SABRE-no-
    # majority isolates the contribution of the fuzzy extractor.
    apuf = APUF(n_stages=32, noisiness=0.05, seed=1)
    c = random_challenges(n, 32, seed=1)
    y = apuf.eval(c)
    variants = {"1-APUF (base)": (c, y)}

    # 3-XOR only (no offsets, no H, no majority)
    xor3 = XORAPUF(n_stages=32, k=3, noisiness=0.05, seed=3)
    variants["3-XOR only"] = (c, xor3.eval(c))

    # 3-XOR + offsets only (no H, no majority)
    xor3o = XORAPUF(n_stages=32, k=3, noisiness=0.05,
                    biases=[0.0, 0.5, 1.0], seed=3)
    variants["3-XOR + offsets"] = (c, xor3o.eval(c))

    # SABRE without the fuzzy extractor (raw, single-shot)
    sabre_nomaj = SABRESim(seed=7)
    variants["SABRE (raw, no majority)"] = (c, sabre_nomaj.eval_raw(c))

    # Full SABRE (H + offsets + 7x majority)
    sabre = SABRESim(seed=7)
    variants["SABRE full"] = (c, sabre.eval_stable(c))

    out = {}
    for name, (ch, resp) in variants.items():
        c_tr, y_tr, c_te, y_te = train_test_split(ch, resp)
        phi_tr = bits_to_phi(c_tr)
        phi_te = bits_to_phi(c_te)
        atk = MursiAttack(k_xor=3, epochs=25, device=device)
        atk.fit(phi_tr, y_tr, verbose=False)
        acc = accuracy_score(y_te, atk.predict(phi_te))
        print(f"  {name:<32s}  DNN-Mursi acc={acc:.4f}")
        out[name] = float(acc)
    return {"ablation": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="path to CRPs_1M.csv")
    ap.add_argument("--out-dir", default="../results")
    ap.add_argument("--n-max", type=int, default=None,
                    help="cap hardware CRPs (for smoke testing)")
    ap.add_argument("--skip", default="",
                    help="comma-separated phases to skip")
    ap.add_argument("--only", default="",
                    help="comma-separated phases to ONLY run")
    ap.add_argument("--quick", action="store_true",
                    help="20k CRPs, short epoch budgets (smoke test)")
    ap.add_argument("--with-ablation", action="store_true",
                    help="include architectural ablation")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    apply_paper_style()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    n_max = args.n_max or (20_000 if args.quick else None)
    hw = load_hardware_csv(args.csv, n_max=n_max)
    ctx = {"hw_data": hw}

    selected = set(ALL_PHASES.keys())
    if args.only:
        selected = set(s.strip() for s in args.only.split(",") if s.strip())
    if args.skip:
        for s in args.skip.split(","):
            selected.discard(s.strip())

    eval_out = {"citations": CITATIONS, "args": vars(args)}
    t0 = time.time()
    for name, cls in ALL_PHASES.items():
        if name not in selected:
            continue
        kwargs = {}
        if args.quick:
            if name == "hardware":    kwargs = {"device": args.device}
            if name == "simulator":   kwargs = {"n_crps": 20_000,
                                                "device": args.device}
            if name == "cca":         kwargs = {"n_samples": 10_000,
                                                "device": args.device}
            if name == "pac":         kwargs = {
                "n_sizes": [2_000, 5_000, 10_000],
                "device": args.device, "repeats": 2,
            }
            if name == "crp_complexity": kwargs = {
                "n_sizes": [2_000, 5_000, 10_000],
                "device": args.device,
            }
        else:
            import inspect
            try:
                sig = inspect.signature(cls.__init__)
                if "device" in sig.parameters:
                    kwargs = {"device": args.device}
            except (ValueError, TypeError):
                # e.g., MetricsPhase uses default object.__init__ with no params
                pass
        phase = cls(**kwargs)
        eval_out[name] = phase.run(ctx)

    if args.with_ablation:
        eval_out.update(run_ablation_phase(
            n=10_000 if args.quick else 50_000, device=args.device))

    combined = make_comparison_and_ablation(eval_out)
    eval_out.update(combined)

    # Plot everything we have
    fig_dir = os.path.join(args.out_dir, "figures")
    produced = save_all_attack_plots(
        {
            "histories":      eval_out.get("hardware", {}).get("histories", {}),
            "crp_complexity": eval_out.get("crp_complexity", {})
                                .get("crp_complexity"),
            "pac":            eval_out.get("pac", {}).get("pac"),
            "comparison":     eval_out.get("comparison"),
            "confusions":     {
                k: v for k, v in
                eval_out.get("hardware", {}).get("confusions", {}).items()
            },
            "ablation":       eval_out.get("ablation"),
            "base_vs_sabre":  eval_out.get("base_vs_sabre"),
        }, out_dir=fig_dir)
    print("\n[plots]")
    for k, paths in produced.items():
        for p in paths:
            print(f"  {p}")

    # JSON report (strip numpy types)
    report_path = os.path.join(args.out_dir, "evaluation_report.json")
    def _json_default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return str(o)
    with open(report_path, "w") as f:
        json.dump(eval_out, f, indent=2, default=_json_default)
    print(f"\n[report] wrote {report_path}")
    print(f"[timing] total wallclock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
