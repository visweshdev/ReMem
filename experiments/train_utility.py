"""Collect interaction traces on TRAIN seeds and fit the learned memory-utility models.

    python experiments/train_utility.py [--workers 8] [--out results/models]

Outputs
  results/models/utility_static.json   features: static + history           (no outcome information)
  results/models/utility_full.json     features: static + history + outcome + contradiction
  results/models/train_report.json     data sizes, leave-one-seed-out CV (log-loss, AUC), weights
  results/traces/train_traces.npz      the raw (features, label) rows

Labels are *observed* labels (attributed credit >= 0.5 on a successful task), not ground truth.
Test seeds are never touched here.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from remem.config import load_systems  # noqa: E402
from remem.eval.agent import AgentConfig  # noqa: E402
from remem.eval.metrics import auc  # noqa: E402
from remem.eval.pools import ensure_pools, load_pools, make_embedder  # noqa: E402
from remem.eval.protocol import TRAIN_FAMILIES, TRAIN_SEEDS, train_reader_mode  # noqa: E402
from remem.eval.runner import apply_world_selection, run_stream  # noqa: E402
from remem.eval.world import make_world  # noqa: E402
from remem.features import ALL_FEATURES  # noqa: E402
from remem.utility import LearnedUtility  # noqa: E402

log = logging.getLogger("train_utility")

FEATURE_SETS = {
    "utility_static": ("static", "history"),
    "utility_full": ("static", "history", "outcome", "contradiction"),
}
L2_GRID = (0.1, 1.0, 10.0, 100.0)


def collect(cache_dir: Path, workers: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    jobs = [(f, s, "vector", 20, None) for f in TRAIN_FAMILIES for s in TRAIN_SEEDS]
    ensure_pools(jobs, cache_dir, workers)
    systems = load_systems(ROOT / "configs" / "systems.yaml")
    emb = make_embedder()
    X, y, seeds, fams = [], [], [], []
    for fam in TRAIN_FAMILIES:
        for seed in TRAIN_SEEDS:
            world = make_world(fam, seed)
            pools = load_pools(cache_dir, fam, seed, "vector", 20)
            cfg = apply_world_selection(systems["collect"], world)
            run = run_stream(world, pools, cfg, emb, AgentConfig(mode=train_reader_mode(seed)), seed=seed, collect=True)
            for row in run.trace_rows:
                X.append(row.features)
                y.append(row.label)
                seeds.append(seed)
                fams.append(fam)
            log.info("collected %s seed=%d rows=%d success=%.3f", fam, seed, len(run.trace_rows),
                     np.mean([r.success for r in run.records]))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), np.asarray(seeds), fams


def cv_scores(X, y, seeds, groups, l2) -> tuple[float, float]:
    """Leave-one-seed-out CV: mean log-loss and mean AUC."""
    losses, aucs = [], []
    for held in sorted(set(seeds.tolist())):
        tr, te = seeds != held, seeds == held
        model, _ = LearnedUtility.fit(X[tr], y[tr], groups, l2=l2)
        p = np.clip(model.predict_proba(X[te]), 1e-12, 1 - 1e-12)
        losses.append(float(-np.mean(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p))))
        aucs.append(auc(p, y[te]))
    return float(np.mean(losses)), float(np.nanmean(aucs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "results" / "models"))
    ap.add_argument("--cache", default=str(ROOT / "results" / "cache" / "pools"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    X, y, seeds, fams = collect(Path(args.cache), args.workers)
    log.info("rows=%d positives=%.3f", len(y), y.mean())

    trace_dir = ROOT / "results" / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(trace_dir / "train_traces.npz", X=X, y=y, seeds=seeds, families=np.asarray(fams),
                        feature_names=np.asarray(ALL_FEATURES))

    report = {"n_rows": int(len(y)), "positive_rate": float(y.mean()), "seeds": list(TRAIN_SEEDS),
              "families": list(TRAIN_FAMILIES), "models": {}}
    for name, groups in FEATURE_SETS.items():
        grid = {}
        for l2 in L2_GRID:
            ll, a = cv_scores(X, y, seeds, groups, l2)
            grid[str(l2)] = {"cv_logloss": ll, "cv_auc": a}
            log.info("%s l2=%g cv_logloss=%.4f cv_auc=%.4f", name, l2, ll, a)
        best = min(L2_GRID, key=lambda l: grid[str(l)]["cv_logloss"])
        model, fit = LearnedUtility.fit(X, y, groups, l2=best, meta={
            "groups": list(groups), "l2": best, "n_rows": int(len(y)), "train_seeds": list(TRAIN_SEEDS),
            "families": list(TRAIN_FAMILIES), "labels": "attributed credit >= 0.5 on successful tasks"})
        model.save(out / f"{name}.json")
        report["models"][name] = {
            "groups": list(groups), "chosen_l2": best, "cv": grid, "train_logloss": fit.train_logloss,
            "converged": fit.converged, "weights": dict(zip(model.names, map(float, model.w))), "bias": model.b,
        }
    (out / "train_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    log.info("done in %.1fs -> %s", time.time() - t0, out)


if __name__ == "__main__":
    main()
