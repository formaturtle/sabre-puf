"""Regenerate all paper figures from a saved evaluation_report.json."""
import json
import sys
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from plotting import save_all_attack_plots

REPORT = os.path.join(HERE, "..", "results", "evaluation_report.json")
FIG_DIR = os.path.join(HERE, "..", "results", "figures")

with open(REPORT) as f:
    ev = json.load(f)

payload = {
    "histories":      ev.get("hardware", {}).get("histories", {}),
    "crp_complexity": ev.get("crp_complexity", {}).get("crp_complexity"),
    "pac":            ev.get("pac", {}).get("pac"),
    "comparison":     ev.get("comparison"),
    "confusions":     ev.get("hardware", {}).get("confusions", {}),
    "ablation":       ev.get("ablation"),
    "base_vs_sabre":  ev.get("base_vs_sabre"),
}

produced = save_all_attack_plots(payload, out_dir=FIG_DIR)
print("\nRegenerated figures:")
for k, paths in produced.items():
    for p in paths:
        print(f"  {p}")
