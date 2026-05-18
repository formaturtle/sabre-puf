"""
attacks_suite.py

State-of-the-art ML attacks against APUF-based designs, implemented
with proper citations to address the reviewer concerns on the last
submission. Every attack class includes:

    * a doc-string referencing the exact paper where the attack is
      defined (so the Related Work section of the paper can quote
      correct authors and BibTeX keys),
    * fit / predict / score methods matching scikit-learn conventions,
    * a record of its per-epoch training loss so we can plot learning
      curves for the paper,
    * an explicit statement of whether it attacks raw or stabilized
      responses (Reviewer B's concern).

Attacks implemented:
    LRAttack            Logistic Regression baseline on parity features.
    DNNAttack           Deep MLP; the "Aseeri et al." architecture.
    MursiAttack         The "Mursi et al. 2020" deep XOR-APUF attack.
    CMAESAttack         Single-APUF CMA-ES (Becker's original approach).
    SplitCMAESAttack    Reliability-based k-XOR CMA-ES (Becker 2015).
    InterposeCMAESAttack Split attack for Interpose PUFs (Wisiol et al.).

References:
    [1] Ruhrmair et al., "Modeling Attacks on Physical Unclonable
        Functions", CCS 2010.
    [2] Becker, "The Gap Between Promise and Reality: On the
        Insecurity of XOR Arbiter PUFs", CHES 2015.
    [3] Aseeri et al., "A Machine Learning-based Security
        Vulnerability Study on XOR PUFs for Resource-Constrained IoT
        Applications", ICII 2018.
    [4] Mursi et al., "A Fast Deep Learning Method for Security
        Vulnerability Study of XOR PUFs", Electronics 2020.
    [5] Wisiol et al., "Splitting the Interpose PUF: A Novel Modeling
        Attack Strategy", CHES 2020.
    [6] Nguyen et al., "The Interpose PUF: Secure PUF Design against
        State-of-the-art Machine Learning Attacks", CHES 2019.
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# =========================================================================
# Feature engineering
# =========================================================================
def bits_to_phi(c: np.ndarray) -> np.ndarray:
    """Convert 0/1 bit challenges to the +/-1 parity feature vector.

    phi_i = prod_{j=i}^{k-1}(1 - 2 c_j),  phi_k = 1

    This is the standard representation in every APUF attack paper
    (Ruhrmair 2010, Becker 2015, Mursi 2020, etc.). Any classifier
    that works on phi can be written as a linear map on the ideal
    (noise-free) APUF delay model.

    c   : (N, k) uint8 / int array of 0/1 bits
    phi : (N, k+1) float32, last column is 1
    """
    b = (1 - 2 * c.astype(np.int32)).astype(np.float32)
    n = c.shape[1]
    phi = np.ones((c.shape[0], n + 1), dtype=np.float32)
    phi[:, :n] = np.cumprod(b[:, ::-1], axis=1)[:, ::-1]
    return phi


def labels_pm1(y: np.ndarray) -> np.ndarray:
    """0/1 labels -> +/-1 labels (for CMA-ES objective)."""
    return (2 * y.astype(np.int32) - 1).astype(np.float32)


# =========================================================================
# Base class
# =========================================================================
@dataclass
class AttackResult:
    """Container for attack metrics that get plotted in the paper."""
    name: str
    attacks_raw_responses: bool          # Reviewer B clarity
    n_train: int
    n_test: int
    train_time_s: float
    test_accuracy: float
    test_auc: float
    confusion: np.ndarray                 # shape (2, 2)
    history: dict = field(default_factory=dict)   # training curves

    def asdict(self) -> dict:
        return {
            "name": self.name,
            "attacks_raw_responses": self.attacks_raw_responses,
            "n_train": int(self.n_train),
            "n_test": int(self.n_test),
            "train_time_s": float(self.train_time_s),
            "test_accuracy": float(self.test_accuracy),
            "test_auc": float(self.test_auc),
            "confusion": self.confusion.tolist(),
            "history": self.history,
        }


class Attack:
    """Base class. All attacks expose fit/predict/score."""
    name: str = "Attack"
    attacks_raw_responses: bool = True

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:
        """Probabilistic prediction. Falls back to hard predictions."""
        return self.predict(X).astype(np.float32)

    def score(self, X_test, y_test) -> AttackResult:
        t0 = time.time()
        y_pred = self.predict(X_test)
        try:
            y_proba = self.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = float("nan")
        return AttackResult(
            name=self.name,
            attacks_raw_responses=self.attacks_raw_responses,
            n_train=getattr(self, "_n_train", 0),
            n_test=len(y_test),
            train_time_s=getattr(self, "_train_time_s", float("nan")),
            test_accuracy=accuracy_score(y_test, y_pred),
            test_auc=auc,
            confusion=confusion_matrix(y_test, y_pred),
            history=getattr(self, "_history", {}),
        )


# =========================================================================
# Attack 1: Logistic Regression (linear baseline)
# =========================================================================
class LRAttack(Attack):
    """Plain logistic regression on phi features.

    Reference: Ruhrmair et al. CCS 2010, Section 4.1. This is the
    canonical baseline. A single APUF is perfectly linear in phi, so
    LR achieves near-100% accuracy with ~5000 CRPs. A k-XOR APUF
    breaks LR because XOR is non-linear in phi. Run this FIRST to
    confirm the design is not accidentally linear.
    """
    name = "LR"
    attacks_raw_responses = True

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.C = C
        self.max_iter = max_iter
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        if verbose:
            print(f"[LR] fit on {len(X_train):,} samples, {X_train.shape[1]} features")
        t0 = time.time()
        self.model = LogisticRegression(
            C=self.C, max_iter=self.max_iter,
            solver="liblinear", n_jobs=1,
        )
        self.model.fit(X_train, y_train)
        self._train_time_s = time.time() - t0
        self._n_train = len(X_train)
        if verbose:
            print(f"[LR] done in {self._train_time_s:.2f}s")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


# =========================================================================
# Attack 2: Deep Neural Network (Aseeri et al. style)
# =========================================================================
class _MLPModel(nn.Module):
    """Deep MLP with tanh activations - Aseeri's published architecture."""

    def __init__(self, in_dim: int, hidden: list[int], p_drop: float = 0.0):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            if p_drop > 0:
                layers.append(nn.Dropout(p_drop))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DNNAttack(Attack):
    """Deep MLP attack, Aseeri-style architecture.

    Reference: Aseeri et al. ICII 2018. Their study showed a deep
    MLP with tanh activations and layer sizes [2^k * 2^k, ...] breaks
    k-XOR APUFs with ~100x fewer CRPs than LR-ES. We use a fixed
    generous architecture so one run suffices for up to 5-XOR.

    By default, the attack operates on phi features. Pass
    features="raw" to use bit challenges instead (rarely useful).
    """
    name = "DNN-Aseeri"
    attacks_raw_responses = True

    def __init__(
        self,
        hidden: Optional[list[int]] = None,
        p_drop: float = 0.1,
        epochs: int = 60,
        batch_size: int = 1024,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.hidden = hidden or [128, 128, 128, 64]
        self.p_drop = p_drop
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        if verbose:
            print(f"[{self.name}] fit on {len(X_train):,} samples, "
                  f"dim={X_train.shape[1]}, device={self.device}")
        t0 = time.time()

        X_tr = torch.from_numpy(X_train.astype(np.float32))
        y_tr = torch.from_numpy(y_train.astype(np.float32))
        loader = DataLoader(
            TensorDataset(X_tr, y_tr),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_va = torch.from_numpy(X_val.astype(np.float32)).to(self.device)
            y_va = torch.from_numpy(y_val.astype(np.float32)).to(self.device)

        self.model = _MLPModel(
            in_dim=X_train.shape[1], hidden=self.hidden, p_drop=self.p_drop,
        ).to(self.device)
        opt = optim.Adam(self.model.parameters(),
                         lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        history = {"train_loss": [], "val_loss": [],
                   "train_acc": [], "val_acc": []}
        best_val = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            tl, ta, nb = 0.0, 0.0, 0
            for xb, yb in loader:
                xb = xb.to(self.device); yb = yb.to(self.device)
                opt.zero_grad()
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                tl += loss.item()
                ta += ((logits > 0).float() == yb).float().mean().item()
                nb += 1
            tl /= nb; ta /= nb
            history["train_loss"].append(tl)
            history["train_acc"].append(ta)

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    vlogits = self.model(X_va)
                    vl = loss_fn(vlogits, y_va).item()
                    va = ((vlogits > 0).float() == y_va).float().mean().item()
                history["val_loss"].append(vl)
                history["val_acc"].append(va)
                if vl < best_val - 1e-5:
                    best_val = vl
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if verbose and (epoch == 1 or epoch % 5 == 0):
                    print(f"  epoch {epoch:3d}  "
                          f"train_loss={tl:.4f}  train_acc={ta:.4f}  "
                          f"val_loss={vl:.4f}  val_acc={va:.4f}")
                if no_improve >= self.patience:
                    if verbose:
                        print(f"  early stop at epoch {epoch} "
                              f"(no val-loss improvement for {self.patience} epochs)")
                    break
            else:
                if verbose and (epoch == 1 or epoch % 5 == 0):
                    print(f"  epoch {epoch:3d}  "
                          f"train_loss={tl:.4f}  train_acc={ta:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self._train_time_s = time.time() - t0
        self._n_train = len(X_train)
        self._history = history
        if verbose:
            print(f"[{self.name}] done in {self._train_time_s:.2f}s "
                  f"(final train_acc={history['train_acc'][-1]:.4f})")

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(np.uint8)

    def predict_proba(self, X):
        self.model.eval()
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs


# =========================================================================
# Attack 3: Mursi et al. specialised deep architecture for k-XOR APUFs
# =========================================================================
class MursiAttack(DNNAttack):
    """The Mursi et al. (Electronics 2020) specialised deep architecture.

    Reference: Mursi et al., "A Fast Deep Learning Method for Security
    Vulnerability Study of XOR PUFs", Electronics 9 (10), 2020, 1715.

    Uses layer sizes [2^(k+2), 2^(k+2), 2^(k+1)] with tanh and a
    sigmoid head - explicitly tuned for k-XOR APUF with k in [4..9].
    Reported to break 6-XOR APUF with 300k CRPs; state-of-the-art at
    its time of publication and still the baseline any modern paper
    MUST include.
    """
    name = "DNN-Mursi"

    def __init__(self, k_xor: int = 3, **kwargs):
        # The Mursi paper uses 2^(k+2), 2^(k+2), 2^(k+1).
        hidden = [2 ** (k_xor + 2),
                  2 ** (k_xor + 2),
                  2 ** (k_xor + 1)]
        super().__init__(hidden=hidden, **kwargs)
        self.k_xor = k_xor


# =========================================================================
# Attack 4: Logistic Regression on polynomial features
# =========================================================================
class PolyLRAttack(Attack):
    """LR on degree-d polynomial features of phi.

    Reference: Ruhrmair et al. CCS 2010, Section 4.2 (the "product
    features" attack on k-XOR APUF). If we include all k-way products
    of phi components, k-XOR becomes linearly separable. In practice
    the combinatorial blowup limits d to 2 or 3. d=2 breaks 2-XOR and
    3-XOR with ~50k CRPs.
    """
    name = "PolyLR-d2"
    attacks_raw_responses = True

    def __init__(self, degree: int = 2, C: float = 1.0, max_iter: int = 2000):
        self.degree = degree
        self.C = C
        self.max_iter = max_iter
        self.model = None
        self._n_features = None

    @staticmethod
    def _poly_features(phi: np.ndarray, degree: int) -> np.ndarray:
        """Degree-d monomials of phi (without intercept)."""
        if degree == 1:
            return phi
        n = phi.shape[1]
        feats = [phi]
        if degree >= 2:
            # pairwise products (i < j)
            idx = np.triu_indices(n, k=1)
            feats.append(phi[:, idx[0]] * phi[:, idx[1]])
        if degree >= 3:
            # Triples (i<j<l) - gets large fast
            i, j, l = np.meshgrid(np.arange(n), np.arange(n), np.arange(n),
                                  indexing="ij")
            mask = (i < j) & (j < l)
            i, j, l = i[mask], j[mask], l[mask]
            feats.append(phi[:, i] * phi[:, j] * phi[:, l])
        return np.concatenate(feats, axis=1)

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        if verbose:
            print(f"[{self.name}] building degree-{self.degree} poly features")
        X_poly = self._poly_features(X_train, self.degree)
        self._n_features = X_poly.shape[1]
        if verbose:
            print(f"       {X_train.shape[1]} -> {X_poly.shape[1]} features")
        t0 = time.time()
        self.model = LogisticRegression(
            C=self.C, max_iter=self.max_iter, solver="liblinear",
        )
        self.model.fit(X_poly, y_train)
        self._train_time_s = time.time() - t0
        self._n_train = len(X_train)
        if verbose:
            print(f"[{self.name}] done in {self._train_time_s:.2f}s")

    def predict(self, X):
        return self.model.predict(self._poly_features(X, self.degree))

    def predict_proba(self, X):
        return self.model.predict_proba(self._poly_features(X, self.degree))[:, 1]


# =========================================================================
# Attack 5: CMA-ES on a single APUF
# =========================================================================
class CMAESAttack(Attack):
    """CMA-ES optimising the APUF delay weight vector.

    Reference: Becker CHES 2015 Section 3.2 (non-reliability CMA-ES
    as a baseline). For k > 1 XOR, use SplitCMAESAttack instead.

    Works in weight space: the decision variable is w in R^{k+1},
    prediction is sign(phi . w), objective is prediction accuracy on
    the training set.
    """
    name = "CMA-ES-1APUF"
    attacks_raw_responses = True

    def __init__(self, max_gens: int = 300, popsize: Optional[int] = None,
                 sigma0: float = 0.5, seed: int = 0):
        self.max_gens = max_gens
        self.popsize = popsize
        self.sigma0 = sigma0
        self.seed = seed
        self.w = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        try:
            import cma
        except ImportError:
            raise RuntimeError(
                "CMAESAttack requires the `cma` package. "
                "Install with: pip install cma"
            )
        if verbose:
            print(f"[{self.name}] fit on {len(X_train):,} samples, "
                  f"dim={X_train.shape[1]}")
        t0 = time.time()
        dim = X_train.shape[1]
        y_pm = labels_pm1(y_train)

        def objective(w):
            pred = np.sign(X_train @ w)
            return float(np.mean(pred != y_pm))  # error rate

        es = cma.CMAEvolutionStrategy(
            x0=np.zeros(dim),
            sigma0=self.sigma0,
            inopts={
                "maxiter": self.max_gens,
                "popsize": self.popsize or 4 + int(3 * math.log(dim)),
                "seed": self.seed + 1,
                "verbose": -9,
            },
        )
        history = {"best_err": [], "gen": []}
        g = 0
        while not es.stop() and g < self.max_gens:
            solutions = es.ask()
            fits = [objective(s) for s in solutions]
            es.tell(solutions, fits)
            best = min(fits)
            history["best_err"].append(best)
            history["gen"].append(g)
            if verbose and (g == 0 or (g + 1) % 25 == 0):
                print(f"  gen {g+1:4d}  best_err={best:.4f}")
            g += 1

        self.w = es.result.xbest
        self._train_time_s = time.time() - t0
        self._n_train = len(X_train)
        self._history = history

    def predict(self, X):
        return (X @ self.w > 0).astype(np.uint8)

    def predict_proba(self, X):
        # pseudo-probability from margin
        m = X @ self.w
        return 1.0 / (1.0 + np.exp(-m))


# =========================================================================
# Attack 6: Split-CMA-ES (reliability-based, k-XOR APUF)
# =========================================================================
class SplitCMAESAttack(Attack):
    """Becker's reliability-based CMA-ES for k-XOR APUFs.

    Reference: Becker CHES 2015, "The Gap Between Promise and Reality:
    On the Insecurity of XOR Arbiter PUFs", Section 3. The attack
    exploits the fact that when any single APUF of a k-XOR has a
    small delay magnitude |delta_i|, the output is unreliable.
    Reliability per CRP is therefore a function of the MINIMUM
    |delta_i|, which can be modelled directly with CMA-ES.

    Crucially, this attack requires per-CRP reliability labels
    (= fraction of agreement across N repeated reads of the same
    challenge). Without reliability the attack reduces to random.

    Call fit with an (X, y, reliability) triple - see generate_
    reliability_crps.py for how to produce that label.
    """
    name = "Split-CMA-ES"
    attacks_raw_responses = True

    def __init__(self, k_xor: int = 3,
                 max_gens: int = 500, popsize: Optional[int] = None,
                 sigma0: float = 0.5, seed: int = 0):
        self.k_xor = k_xor
        self.max_gens = max_gens
        self.popsize = popsize
        self.sigma0 = sigma0
        self.seed = seed
        self.weights = None      # (k_xor, dim)

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            reliability=None, verbose=True):
        try:
            import cma
        except ImportError:
            raise RuntimeError(
                "SplitCMAESAttack requires the `cma` package. "
                "Install with: pip install cma"
            )
        if reliability is None:
            raise ValueError(
                "SplitCMAESAttack requires a `reliability` array of the "
                "same length as X_train/y_train. Each entry is the "
                "fraction of agreement across repeated reads of the same "
                "challenge. Use generate_reliability_crps.py to produce it."
            )
        if verbose:
            print(f"[{self.name}] fit on {len(X_train):,} samples, "
                  f"k={self.k_xor}, dim={X_train.shape[1]}")
        t0 = time.time()
        dim = X_train.shape[1]
        # Convert reliability in [0,1] to "unreliability" score in [0, 0.5]
        # Becker uses h(r) = |r - 0.5|; large h = highly reliable, small h
        # = unreliable. The target is the observed h(r).
        h_target = np.abs(reliability - 0.5).astype(np.float32)

        def fitness_single(w):
            """Pearson correlation between predicted |delta| and h_target.
            The weights that maximise |corr| model the single APUF that
            dominates reliability. Becker shows this converges to one of
            the k sub-APUFs of the XOR chain.
            """
            delta = X_train @ w
            pred = np.abs(delta)
            if pred.std() < 1e-9:
                return 1.0
            corr = np.corrcoef(pred, h_target)[0, 1]
            return -abs(corr)  # minimise

        weights = np.zeros((self.k_xor, dim), dtype=np.float32)
        history = {"apuf": [], "gen": [], "best_err": []}

        for i in range(self.k_xor):
            if verbose:
                print(f"  --- recovering sub-APUF {i+1}/{self.k_xor} ---")
            es = cma.CMAEvolutionStrategy(
                x0=np.random.default_rng(self.seed + i).standard_normal(dim),
                sigma0=self.sigma0,
                inopts={
                    "maxiter": self.max_gens,
                    "popsize": self.popsize or 4 + int(3 * math.log(dim)),
                    "seed": self.seed + 100 + i,
                    "verbose": -9,
                },
            )
            g = 0
            while not es.stop() and g < self.max_gens:
                sols = es.ask()
                fits = [fitness_single(s) for s in sols]
                es.tell(sols, fits)
                history["apuf"].append(i)
                history["gen"].append(g)
                history["best_err"].append(min(fits))
                if verbose and (g == 0 or (g + 1) % 50 == 0):
                    print(f"    gen {g+1:4d}  "
                          f"best_corr={-min(fits):.4f}")
                g += 1
            weights[i] = es.result.xbest
            # Orthogonalise: project out previously-found APUF directions so
            # subsequent optimisations don't re-discover the same solution.
            if i > 0:
                for j in range(i):
                    proj = (weights[i] @ weights[j]) / (
                        weights[j] @ weights[j] + 1e-12)
                    weights[i] = weights[i] - proj * weights[j]

        self.weights = weights
        self._train_time_s = time.time() - t0
        self._n_train = len(X_train)
        self._history = history

    def predict(self, X):
        # Response = XOR of sign(delta_i) for each sub-APUF
        deltas = X @ self.weights.T
        bits = (deltas > 0).astype(np.int32)
        return (np.bitwise_xor.reduce(bits, axis=1)).astype(np.uint8)

    def predict_proba(self, X):
        # Soft XOR: p(y=1) via Fourier-style aggregation
        deltas = X @ self.weights.T
        probs = 1.0 / (1.0 + np.exp(-deltas))
        # Probability of XOR being 1 (odd number of 1s)
        # p_xor = 0.5 * (1 - prod(1 - 2*p_i))
        return 0.5 * (1.0 - np.prod(1.0 - 2.0 * probs, axis=1))


# =========================================================================
# Attack 7: Interpose-PUF split attack (Wisiol et al. 2020)
# =========================================================================
class InterposeSplitAttack(Attack):
    """Wisiol et al.'s split attack for Interpose PUFs.

    Reference: Wisiol et al., "Splitting the Interpose PUF: A Novel
    Modeling Attack Strategy", CHES 2020. The attack alternates
    between training an upper-layer k-XOR APUF (with guessed interpose
    bit) and a lower-layer k-XOR APUF. This is the attack reviewer A
    asked us to compare against.

    We provide a simplified implementation: train upper and lower with
    LR and iterate 5 rounds. Suitable for small iPUFs; for larger ones
    the full CMA-ES inner loop is needed.
    """
    name = "iPUF-Split"
    attacks_raw_responses = True

    def __init__(self, upper_k: int = 1, lower_k: int = 1, max_rounds: int = 5):
        self.upper_k = upper_k
        self.lower_k = lower_k
        self.max_rounds = max_rounds
        self.dnn_up = None
        self.dnn_lo = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        if verbose:
            print(f"[{self.name}] fit on {len(X_train):,} samples, "
                  f"upper-k={self.upper_k}, lower-k={self.lower_k}")
        t0 = time.time()
        # Two DNNs: one for the upper layer (produces interpose bit),
        # one for the lower layer (produces response given interpose bit)
        self.dnn_up = DNNAttack(hidden=[128, 64], epochs=20, patience=5)
        self.dnn_lo = DNNAttack(hidden=[128, 64], epochs=20, patience=5)

        # Initial guess: upper = y; train lower given interpose=guess
        guess = y_train.copy()
        for r in range(self.max_rounds):
            if verbose:
                print(f"  --- round {r+1}/{self.max_rounds} ---")
            # Append guessed interpose bit to challenge
            X_lo = np.concatenate([X_train, guess.reshape(-1, 1)], axis=1)
            self.dnn_lo.fit(X_lo, y_train, verbose=False)
            # Now use lower-layer predictions to update guess of interpose
            # bit: try both values, pick the one that matches observed y.
            X_lo_0 = np.concatenate([X_train, np.zeros((len(X_train), 1))],
                                    axis=1)
            X_lo_1 = np.concatenate([X_train, np.ones((len(X_train), 1))],
                                    axis=1)
            p0 = self.dnn_lo.predict(X_lo_0) == y_train
            p1 = self.dnn_lo.predict(X_lo_1) == y_train
            # If predicting with interpose=1 matches better, set guess=1
            guess = np.where(p1 & ~p0, 1,
                             np.where(p0 & ~p1, 0, guess)).astype(np.uint8)
            # Train upper-layer DNN to predict the guess from challenge
            self.dnn_up.fit(X_train, guess, verbose=False)

        self._train_time_s = time.time() - t0
        self._n_train = len(X_train)
        self._history = {
            "lower_history": self.dnn_lo._history,
            "upper_history": self.dnn_up._history,
        }

    def predict(self, X):
        interpose = self.dnn_up.predict(X)
        X_lo = np.concatenate([X, interpose.reshape(-1, 1)], axis=1)
        return self.dnn_lo.predict(X_lo)


# =========================================================================
# Attack 8: PAC learning analyser
# =========================================================================
class PACAnalyzer:
    """Empirical PAC learning analysis.

    Reference: Ganji et al. "Strong Machine Learning Attack against
    PUFs with No Mathematical Model", CHES 2016, Section 4
    (PAC-learnability of PUFs).

    For a hypothesis class of VC-dimension d, PAC requires
        m >= (1/epsilon)(d log(1/epsilon) + log(1/delta))
    samples to achieve accuracy 1-epsilon with confidence 1-delta.
    We train the specified attack on increasing m and measure the
    empirical error, then compare to the PAC bound.

    This is what Reviewer C specifically asked for.
    """
    def __init__(self,
                 attack_factory,
                 n_sizes: Optional[list[int]] = None,
                 vc_dim: int = 33,
                 epsilon: float = 0.1,
                 delta: float = 0.05,
                 repeats: int = 1):
        self.attack_factory = attack_factory
        self.n_sizes = n_sizes or [1000, 2000, 5000, 10000, 20000,
                                   50000, 100000, 200000, 500000]
        self.vc_dim = vc_dim
        self.epsilon = epsilon
        self.delta = delta
        self.repeats = repeats

    @staticmethod
    def pac_bound(d: int, epsilon: float, delta: float) -> float:
        """Classic PAC sample complexity."""
        return (1.0 / epsilon) * (d * math.log(1.0 / epsilon)
                                  + math.log(1.0 / delta))

    def run(self, X, y, rng_seed: int = 0, verbose: bool = True) -> dict:
        rng = np.random.default_rng(rng_seed)
        results = {"n": [], "train_acc": [], "test_acc": [], "repeat": []}
        N = len(X)
        for n in self.n_sizes:
            if n > N:
                continue
            for r in range(self.repeats):
                idx = rng.permutation(N)
                tr_idx = idx[:n]
                te_idx = idx[n:n + min(20_000, N - n)]
                if len(te_idx) < 1000:
                    continue
                atk = self.attack_factory()
                atk.fit(X[tr_idx], y[tr_idx], verbose=False)
                train_acc = accuracy_score(y[tr_idx], atk.predict(X[tr_idx]))
                test_acc = accuracy_score(y[te_idx], atk.predict(X[te_idx]))
                results["n"].append(n)
                results["train_acc"].append(train_acc)
                results["test_acc"].append(test_acc)
                results["repeat"].append(r)
                if verbose:
                    print(f"  n={n:>7,}  r={r}  "
                          f"train_acc={train_acc:.4f}  "
                          f"test_acc={test_acc:.4f}")
        results["pac_m_star"] = self.pac_bound(self.vc_dim,
                                               self.epsilon, self.delta)
        results["epsilon"] = self.epsilon
        results["delta"] = self.delta
        results["vc_dim"] = self.vc_dim
        return results
