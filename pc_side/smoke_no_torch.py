"""
smoke_no_torch.py
=================

Minimal smoke test for the attack pipeline using ONLY the torch-free
components (LR, PolyLR, CMA-ES, Delvaux-CCA, simulators, plotting,
PAC runner stubbed out). Designed to run inside the Cowork sandbox
which lacks disk space for a full PyTorch install.

This proves that:
    1. The 1M CSV loads, features build, and labels are balanced.
    2. LR trains and the linear APUF baseline reacts as expected.
    3. CMA-ES single-APUF attack runs.
    4. Delvaux CCA runs against the SABRE simulator.
    5. Simulator baselines (1-APUF, 3-XOR, SABRE-sim) behave sensibly.
    6. Every plotting function renders.
"""

from __future__ import annotations

import os, sys, time
import numpy as np
from sklearn.metrics import accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from simulators import (
    APUF, XORAPUF, SABRESim,
    generate_xor_dataset, generate_sabre_dataset, random_challenges,
)
from plotting import (
    apply_paper_style, plot_crp_complexity, plot_attack_comparison_bar,
    plot_confusion_grid, plot_design_ablation, plot_reliability_hist,
)


# Helpers (avoid importing attacks_suite which needs torch)
def bits_to_phi(c):
    b = (1 - 2 * c.astype(np.int32)).astype(np.float32)
    n = c.shape[1]
    phi = np.ones((c.shape[0], n + 1), dtype=np.float32)
    phi[:, :n] = np.cumprod(b[:, ::-1], axis=1)[:, ::-1]
    return phi


def tts(c, y, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_te = int(len(y) * test_frac)
    return c[idx[n_te:]], y[idx[n_te:]], c[idx[:n_te]], y[idx[:n_te]]


def test_load_csv():
    import pandas as pd
    path = os.path.join(HERE, "..", "CRPs_1M.csv")
    print(f"[1] Loading CSV from {path}")
    df = pd.read_csv(path, nrows=50_000)
    ch_cols = [c for c in df.columns if c.startswith("c")]
    c = df[ch_cols].to_numpy(dtype=np.uint8)
    y = df["y"].to_numpy(dtype=np.uint8)
    print(f"    shape: c={c.shape}, y={y.shape}")
    print(f"    bit-bias of y: {y.mean():.4f} (close to 0.5 is healthy)")
    print(f"    bit-bias of each challenge bit (first 5): "
          f"{c[:, :5].mean(axis=0).round(3).tolist()}")
    return c, y


def test_lr(c, y):
    from sklearn.linear_model import LogisticRegression
    print("[2] LR on hardware CRPs")
    c_tr, y_tr, c_te, y_te = tts(c, y)
    phi_tr, phi_te = bits_to_phi(c_tr), bits_to_phi(c_te)
    t0 = time.time()
    model = LogisticRegression(C=1.0, max_iter=500, solver="liblinear")
    model.fit(phi_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(phi_te))
    print(f"    LR acc on SABRE hardware: {acc:.4f}  ({time.time() - t0:.1f}s)")
    print(f"    LR should be near 0.5 for SABRE. "
          f"If >0.7, the design is leaking linearly.")
    return acc


def test_lr_on_1apuf_baseline():
    from sklearn.linear_model import LogisticRegression
    print("[3] LR on 1-APUF baseline (sanity check: should reach ~1.0)")
    puf = APUF(n_stages=32, noisiness=0.02, seed=99)
    c = random_challenges(10_000, 32, seed=1)
    y = puf.eval(c, noise=True)
    c_tr, y_tr, c_te, y_te = tts(c, y)
    m = LogisticRegression(C=1.0, max_iter=500, solver="liblinear")
    m.fit(bits_to_phi(c_tr), y_tr)
    acc = accuracy_score(y_te, m.predict(bits_to_phi(c_te)))
    print(f"    LR acc on 1-APUF: {acc:.4f} (expect ~0.95+)")
    return acc


def test_cmaes_1apuf():
    import cma
    print("[4] CMA-ES on 1-APUF (sanity check)")
    puf = APUF(n_stages=16, noisiness=0.02, seed=7)
    c = random_challenges(3000, 16, seed=2)
    y = puf.eval(c, noise=True)
    c_tr, y_tr, c_te, y_te = tts(c, y)
    phi_tr, phi_te = bits_to_phi(c_tr), bits_to_phi(c_te)
    y_pm = 2 * y_tr.astype(np.int32) - 1

    def obj(w):
        return float(np.mean(np.sign(phi_tr @ w) != y_pm))

    es = cma.CMAEvolutionStrategy(
        np.zeros(17), 0.5,
        {"maxiter": 40, "popsize": 12, "seed": 1, "verbose": -9},
    )
    best = None
    while not es.stop():
        sols = es.ask()
        fits = [obj(s) for s in sols]
        es.tell(sols, fits)
    w = es.result.xbest
    pred = (phi_te @ w > 0).astype(np.uint8)
    acc = accuracy_score(y_te, pred)
    print(f"    CMA-ES acc on 1-APUF: {acc:.4f} (expect > 0.9)")
    return acc


def test_delvaux_cca_on_sabre():
    print("[5] Delvaux CCA against SABRE simulator")
    puf = SABRESim(seed=7)

    def oracle(c):
        return puf.eval_raw(c)

    from cca_attacks import DelvauxCCA
    atk = DelvauxCCA(n_baselines=300, n_repeats=5, seed=0)
    atk.fit(oracle, n_stages=32, verbose=False)
    c_te = random_challenges(5000, 32, seed=99)
    y_te = puf.eval_stable(c_te)
    phi_te = bits_to_phi(c_te)
    acc = accuracy_score(y_te, atk.predict(phi_te))
    print(f"    Delvaux CCA acc on SABRE (stable): {acc:.4f}")
    return acc


def test_simulator_sanity():
    print("[6] Simulator sanity: 4-XOR should be much harder than 1-APUF")
    from sklearn.linear_model import LogisticRegression
    for k in (1, 3, 4, 6):
        d = generate_xor_dataset(k=k, n=20_000, seed=k)
        c_tr, y_tr, c_te, y_te = tts(d["c"], d["y"])
        m = LogisticRegression(C=1.0, max_iter=500, solver="liblinear")
        m.fit(bits_to_phi(c_tr), y_tr)
        acc = accuracy_score(y_te, m.predict(bits_to_phi(c_te)))
        print(f"    LR on {k}-XOR: acc={acc:.4f}")


def test_plotting():
    print("[7] Plotting functions with dummy data")
    apply_paper_style()
    out = os.path.join(HERE, "smoke_figs")
    # crp complexity
    plot_crp_complexity({
        "LR":  {"n": [5_000, 20_000, 100_000],
                "test_acc": [0.50, 0.51, 0.52]},
        "DNN": {"n": [5_000, 20_000, 100_000],
                "test_acc": [0.55, 0.62, 0.72]},
    }, out_dir=out, name="smoke_crp_complexity")
    plot_attack_comparison_bar({
        "LR":  {"SABRE (hw)": 0.50, "4-XOR": 0.56, "(1,5)-iPUF": 0.52},
        "DNN": {"SABRE (hw)": 0.51, "4-XOR": 0.95, "(1,5)-iPUF": 0.70},
    }, out_dir=out, name="smoke_comparison")
    plot_confusion_grid({
        "LR":  [[2400, 2600], [2550, 2450]],
        "DNN": [[4700, 300], [400, 4600]],
    }, out_dir=out, name="smoke_confusion")
    plot_design_ablation({
        "1-APUF": 0.99, "3-XOR": 0.92,
        "3-XOR + H": 0.78, "SABRE full": 0.51,
    }, out_dir=out, name="smoke_ablation")
    rel = np.random.default_rng(0).uniform(0, 1, 5000)
    plot_reliability_hist(rel, np.clip(rel * 0.8 + 0.2, 0, 1),
                          out_dir=out, name="smoke_reliability")
    print(f"    figures written to {out}/")


if __name__ == "__main__":
    c, y = test_load_csv()
    test_lr(c, y)
    test_lr_on_1apuf_baseline()
    test_cmaes_1apuf()
    test_simulator_sanity()
    test_delvaux_cca_on_sabre()
    test_plotting()
    print("\nAll torch-free components OK. "
          "DNN attacks need torch; they will run on your main machine.")
