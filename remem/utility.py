"""Memory utility models.

`HeuristicUtility`  – transparent hand-weighted score. It is the *baseline* utility; its
                      weights are design choices, not learned, and documented below.
`LearnedUtility`    – L2-regularised logistic regression (numpy, Newton/IRLS) predicting
                      P(memory contributes to a successful outcome | features).

Both expose the same interface:

    score(features) -> (rank_score, probability, components)

`rank_score` orders candidates; `probability` is a [0,1] estimate used by cost-aware
selection; `components` maps feature name -> additive contribution (for explanations).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from remem.features import ALL_FEATURES, FEATURE_INDEX, names_for_groups


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ── transparent baseline ────────────────────────────────────────────────────────

#: weights of the transparent score. Units are cosine-similarity points. Rationale:
#: similarity dominates; recency/confidence are small tie-breakers; outcome terms are large
#: enough that a memory with a consistently bad record (class_rate ~0, waste ~1) loses
#: ~0.3 relative to an unseen memory but cannot fully override a clearly better semantic match.
HEURISTIC_WEIGHTS: dict[str, float] = {
    "sim": 1.00,
    "recency": 0.10,
    "confidence": 0.05,
    "rank_inv": 0.0,        # set from UtilityConfig.rank_weight
    "log_ctx": 0.03,        # retrieval frequency: small positive (strawman)
    "retr_recency": 0.03,
    "class_rate": 0.80,     # centred at the prior mean below
    "waste_rate": -0.30,
    "blame_rate": -0.50,
    "post_std": 0.0,        # exploration handled via explore_c
    "stale": -0.50,
    "log_contra": -0.10,
}
HEURISTIC_CENTERS: dict[str, float] = {"class_rate": 0.25}
#: heuristic score -> pseudo-probability for cost-aware selection (fixed, uncalibrated)
HEURISTIC_PROB_SLOPE = 6.0
HEURISTIC_PROB_CENTER = 0.55


class HeuristicUtility:
    kind = "heuristic"

    def __init__(self, feature_groups: tuple[str, ...], explore_c: float = 0.0, rank_weight: float = 0.0) -> None:
        self.names = names_for_groups(feature_groups)
        self.explore_c = explore_c
        self.weights = dict(HEURISTIC_WEIGHTS)
        self.weights["rank_inv"] = rank_weight

    def score(self, feats: list[float]):
        comps: dict[str, float] = {}
        total = 0.0
        for n in self.names:
            w = self.weights.get(n, 0.0)
            if w == 0.0:
                continue
            v = feats[FEATURE_INDEX[n]] - HEURISTIC_CENTERS.get(n, 0.0)
            comps[n] = w * v
            total += comps[n]
        bonus = self.explore_c * feats[FEATURE_INDEX["post_std"]] if "outcome" in _groups_of(self.names) else 0.0
        if bonus:
            comps["explore"] = bonus
        total += bonus
        prob = float(_sigmoid(HEURISTIC_PROB_SLOPE * (total - HEURISTIC_PROB_CENTER)))
        return total, prob, comps


def _groups_of(names: tuple[str, ...]) -> set[str]:
    from remem.features import GROUPS
    return {g for g, ns in GROUPS.items() if any(n in names for n in ns)}


# ── learned model ───────────────────────────────────────────────────────────────

@dataclass
class FitReport:
    n: int
    n_pos: int
    iters: int
    train_logloss: float
    converged: bool


class LearnedUtility:
    kind = "learned"

    def __init__(
        self,
        names: tuple[str, ...],
        weights: np.ndarray,
        bias: float,
        mean: np.ndarray,
        std: np.ndarray,
        explore_c: float = 0.0,
        meta: dict | None = None,
    ) -> None:
        self.names = tuple(names)
        self.w = np.asarray(weights, dtype=float)
        self.b = float(bias)
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)
        self.explore_c = explore_c
        self.meta = meta or {}
        self._idx = [FEATURE_INDEX[n] for n in self.names]

    # inference
    def score(self, feats: list[float]):
        x = np.asarray([feats[i] for i in self._idx], dtype=float)
        z = (x - self.mean) / self.std
        contrib = self.w * z
        logit = float(contrib.sum() + self.b)
        p = float(_sigmoid(logit))
        comps = {n: float(c) for n, c in zip(self.names, contrib)}
        rank = p
        if self.explore_c and "post_std" in FEATURE_INDEX:
            bonus = self.explore_c * feats[FEATURE_INDEX["post_std"]]
            rank += bonus
            comps["explore"] = bonus
        return rank, p, comps

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """X has columns in ALL_FEATURES order."""
        sub = X[:, self._idx]
        return _sigmoid(((sub - self.mean) / self.std) @ self.w + self.b)

    # training
    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        feature_groups: tuple[str, ...],
        *,
        l2: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-8,
        explore_c: float = 0.0,
        meta: dict | None = None,
    ) -> tuple["LearnedUtility", FitReport]:
        """Fit by Newton's method. X columns follow ALL_FEATURES. Deterministic (no RNG)."""
        names = names_for_groups(feature_groups)
        idx = [FEATURE_INDEX[n] for n in names]
        Xs = np.asarray(X, dtype=float)[:, idx]
        y = np.asarray(y, dtype=float)
        mean = Xs.mean(axis=0)
        std = Xs.std(axis=0)
        std[std < 1e-9] = 1.0
        Z = (Xs - mean) / std
        n, d = Z.shape
        A = np.hstack([Z, np.ones((n, 1))])
        beta = np.zeros(d + 1)
        reg = np.full(d + 1, l2)
        reg[-1] = 0.0  # do not penalise the bias
        converged = False
        it = 0
        for it in range(1, max_iter + 1):
            p = _sigmoid(A @ beta)
            grad = A.T @ (p - y) + reg * beta
            W = p * (1 - p) + 1e-9
            H = (A * W[:, None]).T @ A + np.diag(reg) + 1e-9 * np.eye(d + 1)
            step = np.linalg.solve(H, grad)
            beta -= step
            if np.max(np.abs(step)) < tol:
                converged = True
                break
        p = np.clip(_sigmoid(A @ beta), 1e-12, 1 - 1e-12)
        logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        model = cls(names, beta[:-1], beta[-1], mean, std, explore_c=explore_c, meta=meta)
        return model, FitReport(n=n, n_pos=int(y.sum()), iters=it, train_logloss=logloss, converged=converged)

    # persistence
    def to_json(self) -> dict:
        return {
            "kind": "logistic_regression",
            "feature_names": list(self.names),
            "weights": self.w.tolist(),
            "bias": self.b,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "meta": self.meta,
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_json(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, explore_c: float = 0.0) -> "LearnedUtility":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = [n for n in d["feature_names"] if n not in FEATURE_INDEX]
        if unknown:
            raise ValueError(f"model uses unknown features {unknown}; known: {ALL_FEATURES}")
        return cls(tuple(d["feature_names"]), np.array(d["weights"]), d["bias"],
                   np.array(d["mean"]), np.array(d["std"]), explore_c=explore_c, meta=d.get("meta", {}))


def build_utility(cfg) -> HeuristicUtility | LearnedUtility:
    """Factory from `UtilityConfig`."""
    if cfg.kind == "heuristic":
        return HeuristicUtility(cfg.feature_groups, cfg.explore_c, cfg.rank_weight)
    if cfg.kind == "learned":
        if not cfg.model_path:
            raise ValueError("utility.kind=learned requires utility.model_path")
        return LearnedUtility.load(cfg.model_path, explore_c=cfg.explore_c)
    raise ValueError(f"unknown utility kind {cfg.kind!r}")
