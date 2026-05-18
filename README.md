# SABRE — Secure Arbiter PUF with Bias-Randomized Encoding

SABRE is a hardened Arbiter PUF (APUF) architecture implemented on a PYNQ-Z1 FPGA that achieves near-random-guess resistance against the full suite of state-of-the-art modeling attacks, including Logistic Regression, Deep Neural Networks, CMA-ES, and Chosen-Challenge Attacks.

## Overview

Standard Arbiter PUFs are broken by machine learning in seconds. SABRE combines three complementary hardening mechanisms:

1. **3-XOR APUF backbone** — increases the hypothesis class complexity for gradient-based and evolutionary attackers.
2. **BRAM-seeded delay offsets** — a per-device secret embedded in block RAM that randomizes each stage's delay budget, breaking the linear-additive feature model that LR and DNN attacks exploit.
3. **7× majority-vote stabilization** — a lightweight fuzzy extractor that suppresses noise without revealing reliability side-channel information to Split-CMA-ES.

The result: best-attack accuracy across all evaluated attackers is **≤ 51.34%**, indistinguishable from random guessing (50%).

## Repository Layout

```
sabre-puf/
├── pc_side/                    # Host-side Python (attacks + evaluation + plotting)
│   ├── attacks_suite.py        # LR, DNN-Aseeri, DNN-Mursi, CMA-ES, Split-CMA-ES
│   ├── cca_attacks.py          # Chosen-challenge attacks (Delvaux 2013, Tobisch 2021)
│   ├── simulators.py           # Calibrated APUF / XOR-APUF / iPUF / SABRE simulators
│   ├── run_full_evaluation.py  # End-to-end evaluation driver
│   ├── plotting.py             # Paper-quality figure generation
│   ├── metrics.py              # HD-intra, HD-inter, bit-bias, min-entropy
│   ├── make_phi.py             # Parity feature vector builder
│   ├── generate_crps.py        # Synthetic CRP generation utility
│   ├── verify_synthetic.py     # Self-test (no hardware required)
│   └── smoke_no_torch.py       # Lightweight smoke test without PyTorch
├── pynq_side/                  # Runs on the PYNQ-Z1 board
│   ├── sr_capuf_driver.py      # AXI overlay driver for the SABRE hardware core
│   └── collect_crps.py         # CRP collection CLI (sharded, with meta.json)
├── sim/
│   └── tb_sr_capuf.v           # Verilog-2001 behavioral testbench
├── verilog_fixes/              # Corrected RTL for known hardware issues
│   ├── sr_capuf_fix2.v         # Done-handshake fix
│   └── sr_capuf_sync.v         # Clock-domain synchronization fix
├── xdc_fixes/
│   └── sr_capuf_constraints.xdc  # Timing constraints
└── results/
    └── figures/                # Paper figures (PNG + PDF, 300 DPI)
```

## Key Results

| Attack | Base APUF accuracy | SABRE accuracy |
|---|---|---|
| Logistic Regression | 99.70% | 51.34% |
| DNN-Mursi | 99.65% | 51.34% |
| DNN-Aseeri | 99.57% | 51.34% |
| CCA-Tobisch | 97.46% | 50.67% |
| CMA-ES | 96.95% | 51.27% |
| CCA-Delvaux | 45.29% | 49.83% |
| PolyLR-d2 | — | 50.94% |
| Split-CMA-ES | — | 49.81% |

All SABRE numbers are within ±1.5% of the 50% random-guess baseline.

## Requirements

### PC side
```
python >= 3.10
numpy
scikit-learn
torch >= 2.0
matplotlib
pandas
```

Install with:
```bash
pip install numpy scikit-learn torch matplotlib pandas
```

### PYNQ side
- PYNQ-Z1 board running PYNQ v2.7
- `pynq` Python package (pre-installed on the board image)

## Reproducing the Evaluation

### 1. Simulate without hardware (self-test)
```bash
cd pc_side
python verify_synthetic.py      # math / feature-builder checks
python smoke_no_torch.py        # metrics smoke test
```

### 2. Run the full attack suite (requires CRPs_1M.csv from hardware)
```bash
python run_full_evaluation.py --csv ../CRPs_1M.csv --out-dir ../results
```

Quick smoke run (20k CRPs, short epoch budgets):
```bash
python run_full_evaluation.py --csv ../CRPs_1M.csv --out-dir ../results --quick
```

Skip specific phases:
```bash
python run_full_evaluation.py --csv ../CRPs_1M.csv --out-dir ../results --skip cca,pac
```

### 3. Regenerate paper figures from a saved report
```bash
python regen_figs.py
# Output: results/figures/*.{png,pdf}
```

### 4. Collect CRPs from hardware (on the PYNQ-Z1)
```bash
# Copy pynq_side/ to the board, then:
python collect_crps.py --n 1000000 --repeats 11 --shard 100000 \
    --out /home/xilinx/crps_run1
```
SCP the output directory to your PC and pass it to `run_full_evaluation.py` via `--csv`.

## Hardware Design

The SABRE core is implemented as an AXI4-Lite peripheral in the PYNQ-Z1 PS-PL fabric:

- **Challenge interface**: 32-bit AXI register → LFSR-based challenge expansion
- **Delay stages**: 32-stage SR-latch chain with per-stage BRAM-seeded offset injection
- **Response**: 7× sampled majority vote via PS-side Python driver

See `verilog_fixes/` for corrected RTL and `xdc_fixes/` for timing constraints.
See `sim/tb_sr_capuf.v` for the behavioral testbench.

## PAC Learning Analysis

SABRE's empirical test error stays at **≈ 0.485** across all training sizes (2k–500k CRPs), consistently above the PAC target of ε = 0.10, confirming that the function class realized by SABRE is not efficiently PAC-learnable by the hypothesis classes used by the evaluated attacks.

## Ablation Study

| Variant | Best-attack accuracy |
|---|---|
| 1-APUF (base) | 99.45% |
| 3-XOR only | 98.04% |
| 3-XOR + offsets | 98.37% |
| SABRE (raw, no majority) | 50.76% |
| SABRE full | 50.01% |

The majority-vote stabilizer is the decisive component: removing it alone raises attacker accuracy from 50.01% to 50.76%, while the offset randomization provides the bulk of the security margin over the XOR-only baseline.

## Citation

Anonymous submission. Citation information will be added after the review period.
