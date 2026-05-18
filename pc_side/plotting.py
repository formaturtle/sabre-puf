r"""
plotting.py
===========

Paper-quality figures for the SABRE resubmission. All figures are
publication-ready: vector PDF + high-DPI PNG, consistent color palette,
tight layouts, and captions embedded so the LaTeX source can \input them
verbatim.

Design rules (derived from IEEE S&P / CCS / CHES camera-ready guidelines):
    * Colour-blind-safe Okabe-Ito palette (8 colours).
    * Sans-serif fonts for figure labels; math in serif via usetex=False
      mode (works without a TeX install).
    * 300 DPI PNG for Overleaf previews, vector PDF for the final build.
    * Error bars where repeats > 1.
    * Grid minor-ticks for log-scale axes.

Usage:
    from plotting import (
        save_all_attack_plots,
        plot_learning_curves,
        plot_crp_complexity,
        plot_pac_learning,
        plot_reliability_hist,
        plot_attack_comparison_bar,
        plot_confusion_grid,
        plot_design_ablation,
    )
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")       # headless; we're generating files
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# =========================================================================
# Global style
# =========================================================================
OKABE_ITO = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#000000",  # black
]


def apply_paper_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (6.4, 4.0),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "font.weight": "bold",
        "axes.labelsize": 15,
        "axes.labelweight": "bold",
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "legend.frameon": False,
        "legend.fontsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "lines.linewidth": 2.2,
        "lines.markersize": 6,
        "pdf.fonttype": 42,   # editable text in PDF
        "ps.fonttype": 42,
    })


def _save(fig: plt.Figure, out_dir: str, name: str) -> list[str]:
    """Save fig as both PNG and PDF into out_dir/name.{png,pdf}."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


# =========================================================================
# Figure 1: Learning curves for one attack (loss / accuracy vs epoch)
# =========================================================================
def plot_learning_curves(history: dict, title: str, out_dir: str,
                         name: str = "learning_curves") -> list[str]:
    apply_paper_style()
    # Defensive: some attacks (e.g. CMA-ES) expose different history keys.
    # Only DNN-style histories have train_loss/train_acc; bail out otherwise.
    if "train_loss" not in history or len(history.get("train_loss", [])) == 0:
        return []
    has_val = "val_loss" in history and len(history.get("val_loss", [])) > 0
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(9.5, 3.6))

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax_loss.plot(epochs, history["train_loss"],
                 color=OKABE_ITO[0], label="train")
    if has_val:
        ax_loss.plot(epochs, history["val_loss"],
                     color=OKABE_ITO[1], label="validation")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("BCE loss")
    ax_loss.set_title(f"{title} — loss")
    ax_loss.legend()

    ax_acc.plot(epochs, history["train_acc"],
                color=OKABE_ITO[0], label="train")
    if has_val:
        ax_acc.plot(epochs, history["val_acc"],
                    color=OKABE_ITO[1], label="validation")
    ax_acc.axhline(0.5, color="0.6", linestyle=":",
                   label="random guess")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0.45, 1.02)
    ax_acc.set_title(f"{title} — accuracy")
    ax_acc.legend()

    fig.tight_layout()
    return _save(fig, out_dir, name)


# =========================================================================
# Figure 2: CRP-complexity curve (accuracy vs training-set size)
# =========================================================================
def plot_crp_complexity(results: dict, out_dir: str,
                        name: str = "crp_complexity",
                        title: str = "CRP complexity of modeling attacks"
                        ) -> list[str]:
    """
    results : { attack_name : { "n": [...], "test_acc": [...] } }
    Each entry may also carry "test_acc_std" for error bars.
    """
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, (atk, data) in enumerate(results.items()):
        ns = np.asarray(data["n"])
        accs = np.asarray(data["test_acc"])
        order = np.argsort(ns)
        ns, accs = ns[order], accs[order]
        stds = data.get("test_acc_std")
        if stds is not None:
            stds = np.asarray(stds)[order]
            ax.errorbar(ns, accs, yerr=stds, marker="o",
                        color=OKABE_ITO[i % len(OKABE_ITO)],
                        label=atk, capsize=3)
        else:
            ax.plot(ns, accs, marker="o",
                    color=OKABE_ITO[i % len(OKABE_ITO)], label=atk)
    ax.axhline(0.5, color="0.6", linestyle=":", label="random")
    ax.set_xscale("log")
    ax.set_xlabel("Training CRPs")
    ax.set_ylabel("Test accuracy")
    ax.set_title(title)
    ax.set_ylim(0.45, 1.02)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, out_dir, name)


# =========================================================================
# Figure 3: Empirical PAC learning curve + theoretical bound
# =========================================================================
def plot_pac_learning(pac_result: dict, out_dir: str,
                      name: str = "pac_learning") -> list[str]:
    """
    pac_result : output of PACAnalyzer.run()
    Plots (1 - test_acc) vs n with the theoretical PAC bound m* marked.
    """
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ns = np.asarray(pac_result["n"])
    acc = np.asarray(pac_result["test_acc"])
    err = 1.0 - acc
    # average across repeats
    uniq = np.unique(ns)
    mean_err = np.array([err[ns == u].mean() for u in uniq])
    std_err = np.array([err[ns == u].std() for u in uniq])
    ax.errorbar(uniq, mean_err, yerr=std_err, marker="o",
                color=OKABE_ITO[0], label="empirical error",
                capsize=3)
    ax.axhline(pac_result["epsilon"], color=OKABE_ITO[1],
               linestyle="--",
               label=fr"PAC target $\epsilon$={pac_result['epsilon']}")
    ax.axvline(pac_result["pac_m_star"], color=OKABE_ITO[2],
               linestyle="--",
               label=fr"PAC bound $m^\ast$="
                     f"{int(pac_result['pac_m_star']):,}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training CRPs $n$")
    ax.set_ylabel("Test error $(1 - \\mathrm{acc})$")
    ax.set_title(
        f"PAC learning (VC dim = {pac_result['vc_dim']}, "
        f"$\\delta$={pac_result['delta']})"
    )
    ax.legend()
    fig.tight_layout()
    return _save(fig, out_dir, name)


# =========================================================================
# Figure 4: Reliability histogram (raw vs stabilized)
# =========================================================================
def plot_reliability_hist(
        reliability_raw: Optional[np.ndarray],
        reliability_stable: Optional[np.ndarray],
        out_dir: str,
        name: str = "reliability_hist") -> list[str]:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    if reliability_raw is not None:
        ax.hist(reliability_raw, bins=41, range=(0, 1), alpha=0.55,
                color=OKABE_ITO[1],
                label=f"raw single-shot (mean={reliability_raw.mean():.3f})")
    if reliability_stable is not None:
        ax.hist(reliability_stable, bins=41, range=(0, 1), alpha=0.55,
                color=OKABE_ITO[0],
                label=f"7x majority-voted (mean={reliability_stable.mean():.3f})")
    ax.set_xlabel("Per-CRP reliability "
                  r"$\Pr[r_i = 1]$ over 11 reads")
    ax.set_ylabel("Number of CRPs")
    ax.set_title("Response reliability distribution")
    ax.legend()
    fig.tight_layout()
    return _save(fig, out_dir, name)


# =========================================================================
# Figure 5: Attack comparison bar chart (SABRE vs baselines)
# =========================================================================
def plot_attack_comparison_bar(
        summary: dict, out_dir: str,
        name: str = "attack_comparison") -> list[str]:
    """
    summary : { attack_name : { puf_name : accuracy } }
    E.g.  {"DNN-Mursi": {"SABRE": 0.51, "4-XOR": 0.97, "6-XOR": 0.86,
                         "(1,5)-iPUF": 0.72}}
    """
    apply_paper_style()
    attacks = list(summary.keys())
    pufs = list(next(iter(summary.values())).keys())
    n_atk = len(attacks)
    n_puf = len(pufs)
    x = np.arange(n_atk)
    width = 0.8 / n_puf

    fig, ax = plt.subplots(figsize=(max(7, 1.4 * n_atk), 4.2))
    for j, p in enumerate(pufs):
        vals = [summary[a].get(p, np.nan) for a in attacks]
        ax.bar(x + j * width - 0.4 + width / 2, vals, width,
               color=OKABE_ITO[j % len(OKABE_ITO)], label=p)
    ax.axhline(0.5, color="0.5", linestyle=":", label="random")
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=25, ha="right")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.45, 1.02)
    ax.set_title("Modeling-attack accuracy: SABRE vs. hardened-APUF baselines")
    ax.legend(ncol=min(n_puf, 4), loc="upper left",
              bbox_to_anchor=(0.0, 1.02))
    fig.tight_layout()
    return _save(fig, out_dir, name)


# =========================================================================
# Figure 6: Confusion-matrix grid (one per attack)
# =========================================================================
def plot_confusion_grid(confusions: dict, out_dir: str,
                        name: str = "confusion_grid") -> list[str]:
    apply_paper_style()
    n = len(confusions)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.8 * rows))
    axes = np.atleast_2d(axes)
    for i, (atk, cm) in enumerate(confusions.items()):
        r, c = i // cols, i % cols
        ax = axes[r, c]
        cm = np.asarray(cm, dtype=np.float32)
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
        for ii in range(2):
            for jj in range(2):
                ax.text(jj, ii, f"{cm_norm[ii, jj]:.2f}",
                        ha="center", va="center", fontsize=13, fontweight="bold",
                        color="white" if cm_norm[ii, jj] > 0.5 else "black")
        ax.set_title(atk, fontsize=14, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred 0", "pred 1"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["true 0", "true 1"])
        ax.grid(False)
    # Hide unused subplots
    for i in range(n, rows * cols):
        r, c = i // cols, i % cols
        axes[r, c].axis("off")
    fig.suptitle("Per-attack confusion matrices (row-normalised)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir, name)


# =========================================================================
# Figure 7: Ablation study (which SABRE components contribute security)
# =========================================================================
def plot_design_ablation(ablation: dict, out_dir: str,
                         name: str = "design_ablation") -> list[str]:
    """
    ablation : ordered dict of { variant_name : accuracy_under_best_attack }
    E.g. {"1-APUF": 0.99, "3-XOR only": 0.92, "3-XOR + H": 0.78,
          "3-XOR + H + offsets (SABRE)": 0.51}
    """
    apply_paper_style()
    variants = list(ablation.keys())
    accs = [ablation[v] for v in variants]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    colors = [OKABE_ITO[i % len(OKABE_ITO)] for i in range(len(variants))]
    bars = ax.barh(variants, accs, color=colors)
    for bar, a in zip(bars, accs):
        ax.text(a + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{a:.2%}", va="center", fontsize=13, fontweight="bold")
    ax.axvline(0.5, color="0.5", linestyle=":", label="random")
    ax.set_xlim(0.45, 1.02)
    ax.set_xlabel("Best-attack accuracy")
    ax.set_title("SABRE architectural ablation")
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, out_dir, name)


# =========================================================================
# One-call convenience: save every figure from an eval run
# =========================================================================
def plot_base_vs_sabre(summary: dict, out_dir: str,
                       name: str = "base_vs_sabre",
                       title: str = "Modeling-attack accuracy: "
                                    "base APUF vs. SABRE"
                       ) -> list[str]:
    """Headline chart: for each attack, accuracy against the base (1-APUF)
    and against SABRE. Horizontal grouped bars for readability.

    summary : { attack_name : { "1-APUF": acc_base, "SABRE": acc_sabre } }
    """
    apply_paper_style()
    if not summary:
        return []
    # Order attacks by base-APUF accuracy descending, so the "strongest"
    # attack (against the base) appears on top.
    attacks = sorted(
        summary.keys(),
        key=lambda a: -float(summary[a].get("1-APUF", 0.0) or 0.0),
    )
    n = len(attacks)
    y = np.arange(n)
    height = 0.38

    base_vals = [summary[a].get("1-APUF", np.nan) for a in attacks]
    sabre_vals = [summary[a].get("SABRE", np.nan) for a in attacks]

    fig, ax = plt.subplots(figsize=(8.6, max(3.5, 0.55 * n + 1.2)))
    bars_base = ax.barh(y + height / 2, base_vals, height,
                        color=OKABE_ITO[1], label="Base APUF")
    bars_sabre = ax.barh(y - height / 2, sabre_vals, height,
                         color=OKABE_ITO[0], label="SABRE")

    for bars, vals in ((bars_base, base_vals), (bars_sabre, sabre_vals)):
        for bar, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2%}", va="center", fontsize=13, fontweight="bold")

    ax.axvline(0.5, color="0.4", linestyle=":", label="random (0.50)")
    ax.set_yticks(y)
    ax.set_yticklabels(attacks)
    ax.set_xlim(0.45, 1.05)
    ax.set_xlabel("Test accuracy (higher = attack succeeded)")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, out_dir, name)


def plot_cmaes_fitness(history: dict, title: str, out_dir: str,
                       name: str = "cmaes_fitness") -> list[str]:
    """Fitness-vs-generation curve for CMA-ES-style histories."""
    apply_paper_style()
    if "best_err" not in history or len(history.get("best_err", [])) == 0:
        return []
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    gens = np.asarray(history.get("gen", np.arange(len(history["best_err"]))))
    errs = np.asarray(history["best_err"])
    # If history tracks multiple sub-APUFs (Split-CMA-ES), colour per apuf
    apufs = history.get("apuf")
    if apufs is not None and len(apufs) == len(errs):
        apufs = np.asarray(apufs)
        for i in np.unique(apufs):
            mask = apufs == i
            ax.plot(np.arange(mask.sum()), errs[mask],
                    color=OKABE_ITO[int(i) % len(OKABE_ITO)],
                    label=f"sub-APUF {int(i) + 1}")
        ax.legend()
    else:
        ax.plot(gens, errs, color=OKABE_ITO[0])
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best objective (lower = better)")
    ax.set_title(f"{title} — CMA-ES convergence")
    fig.tight_layout()
    return _save(fig, out_dir, name)


def save_all_attack_plots(eval_out: dict, out_dir: str = "figs") -> dict:
    """Produce every figure from an eval dict. Returns map name -> paths."""
    produced = {}
    if "histories" in eval_out:
        for atk, hist in (eval_out["histories"] or {}).items():
            if not isinstance(hist, dict) or not hist:
                continue
            safe = atk.replace(" ", "_").replace("/", "_")
            # Dispatch by history shape. DNN attacks have "train_loss";
            # CMA-ES attacks have "best_err". Skip anything unrecognised
            # so a surprise history key never crashes the whole run.
            if "train_loss" in hist:
                paths = plot_learning_curves(
                    hist, atk, out_dir, name=f"lc_{safe}")
                if paths:
                    produced[f"lc_{safe}"] = paths
            elif "best_err" in hist:
                paths = plot_cmaes_fitness(
                    hist, atk, out_dir, name=f"cma_{safe}")
                if paths:
                    produced[f"cma_{safe}"] = paths
    if "crp_complexity" in eval_out:
        produced["crp_complexity"] = plot_crp_complexity(
            eval_out["crp_complexity"], out_dir)
    if "pac" in eval_out:
        produced["pac_learning"] = plot_pac_learning(
            eval_out["pac"], out_dir)
    if "reliability" in eval_out:
        r = eval_out["reliability"]
        produced["reliability_hist"] = plot_reliability_hist(
            r.get("raw"), r.get("stable"), out_dir)
    if "comparison" in eval_out:
        produced["attack_comparison"] = plot_attack_comparison_bar(
            eval_out["comparison"], out_dir)
    if "confusions" in eval_out:
        produced["confusion_grid"] = plot_confusion_grid(
            eval_out["confusions"], out_dir)
    if "ablation" in eval_out:
        produced["design_ablation"] = plot_design_ablation(
            eval_out["ablation"], out_dir)
    if "base_vs_sabre" in eval_out:
        produced["base_vs_sabre"] = plot_base_vs_sabre(
            eval_out["base_vs_sabre"], out_dir)
    return produced


# =========================================================================
# Self test
# =========================================================================
if __name__ == "__main__":
    # Make a dummy eval_out and check every figure renders
    dummy = {
        "histories": {
            "DNN-Aseeri": {
                "train_loss": [0.7, 0.5, 0.3, 0.2],
                "val_loss":   [0.72, 0.55, 0.35, 0.33],
                "train_acc":  [0.55, 0.65, 0.80, 0.95],
                "val_acc":    [0.54, 0.60, 0.72, 0.80],
            },
        },
        "crp_complexity": {
            "LR": {"n": [1000, 5000, 20000, 100000],
                   "test_acc": [0.52, 0.55, 0.58, 0.61]},
            "DNN-Mursi": {"n": [1000, 5000, 20000, 100000],
                          "test_acc": [0.53, 0.62, 0.78, 0.88]},
        },
        "pac": {
            "n": [1000, 1000, 5000, 5000, 20000, 20000, 100000, 100000],
            "test_acc": [0.55, 0.56, 0.63, 0.65, 0.75, 0.77, 0.85, 0.86],
            "pac_m_star": 250000, "epsilon": 0.1, "delta": 0.05,
            "vc_dim": 33,
        },
        "comparison": {
            "LR":           {"SABRE": 0.50, "4-XOR": 0.57, "(1,5)-iPUF": 0.53},
            "DNN-Aseeri":   {"SABRE": 0.51, "4-XOR": 0.95, "(1,5)-iPUF": 0.68},
            "DNN-Mursi":    {"SABRE": 0.52, "4-XOR": 0.97, "(1,5)-iPUF": 0.72},
            "Split-CMA-ES": {"SABRE": 0.51, "4-XOR": 0.90, "(1,5)-iPUF": 0.55},
        },
        "confusions": {
            "LR":         [[2400, 2600], [2550, 2450]],
            "DNN-Mursi":  [[4700, 300], [400, 4600]],
        },
        "ablation": {
            "1-APUF (baseline)":     0.99,
            "3-XOR only":            0.92,
            "3-XOR + H transform":   0.78,
            "3-XOR + offsets only":  0.85,
            "SABRE full":            0.51,
        },
    }
    paths = save_all_attack_plots(dummy, out_dir="figs_smoketest")
    for name, p in paths.items():
        print(f"  wrote {name}: {p}")
