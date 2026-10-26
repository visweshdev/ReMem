"""Attempt to make the learned utility beat the heuristic: on-policy data aggregation (DAgger-style).

    python experiments/train_utility.py            # round 0: traces from the heuristic policy (already done)
    python experiments/train_utility_dagger.py     # rounds 1..R, then a DEV-seed comparison

Why: the round-0 model is trained on what the *heuristic* chose to show, then deployed as its own ranker, so it is
scored on memories it rarely saw labels for (distribution shift). Each round here collects new traces on TRAIN seeds
with the current learned model as the ranker (plus the same forced exploration), adds them to all earlier traces,
and refits. Every round is compared with the heuristic on DEV seeds only; test seeds are not touched. A model is
promoted to configs/systems.yaml by hand only if it wins on DEV, and the report says either way.

Outputs results/models/dagger/utility_full_r{r}.json and results/models/dagger/report.json.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from remem.config import load_systems  # noqa: E402
from remem.eval.agent import AgentConfig  # noqa: E402
from remem.eval.pools import ensure_pools, load_pools, make_embedder  # noqa: E402
from remem.eval.protocol import DEV_SEEDS, HEADLINE_FAMILIES, TRAIN_FAMILIES, TRAIN_SEEDS, train_reader_mode  # noqa: E402
from remem.eval.runner import apply_world_selection, run_stream  # noqa: E402
from remem.eval.world import make_world  # noqa: E402
from remem.utility import LearnedUtility  # noqa: E402

GROUPS = ("static", "history", "outcome", "contradiction")
CACHE = ROOT / "results" / "cache" / "pools"


def _collect_one(job):
    fam, seed, model_path = job
    systems = load_systems(ROOT / "configs" / "systems.yaml")
    base = systems["collect"]
    cfg = replace(base, ranker="learned",
                  utility=replace(base.utility, kind="learned", feature_groups=GROUPS, model_path=model_path))
    world = make_world(fam, seed)
    run = run_stream(world, load_pools(CACHE, fam, seed, "vector", 20), apply_world_selection(cfg, world),
                     make_embedder(), AgentConfig(mode=train_reader_mode(seed)), seed=seed, collect=True)
    return [r.features for r in run.trace_rows], [r.label for r in run.trace_rows]


def _dev_one(job):
    fam, seed, variant, model_path = job
    systems = load_systems(ROOT / "configs" / "systems.yaml")
    if variant == "heuristic":
        cfg = systems["loo_heuristic_utility"]
    else:
        cfg = systems["C_remem"]
        cfg = replace(cfg, utility=replace(cfg.utility, model_path=model_path))
    world = make_world(fam, seed)
    run = run_stream(world, load_pools(CACHE, fam, seed, "vector", 20), apply_world_selection(cfg, world),
                     make_embedder(), AgentConfig(mode="first"), seed=seed)
    return variant, fam, seed, float(np.mean([r.success for r in run.records]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--l2", type=float, default=None, help="default: the value chosen by CV in round 0")
    a = ap.parse_args()
    out = ROOT / "results" / "models" / "dagger"
    out.mkdir(parents=True, exist_ok=True)

    tr = np.load(ROOT / "results" / "traces" / "train_traces.npz")
    X, y = tr["X"], tr["y"]
    l2 = a.l2 or json.loads((ROOT / "results" / "models" / "train_report.json").read_text())["models"]["utility_full"]["chosen_l2"]
    model_path = str(ROOT / "results" / "models" / "utility_full.json")
    models = {"r0": model_path}
    jobs_pools = [(f, s, "vector", 20, None) for f in TRAIN_FAMILIES for s in TRAIN_SEEDS] + \
                 [(f, s, "vector", 20, None) for f in HEADLINE_FAMILIES for s in DEV_SEEDS]
    ensure_pools(jobs_pools, CACHE, a.workers)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for r in range(1, a.rounds + 1):
            newX, newy = [], []
            for fx, fy in ex.map(_collect_one, [(f, s, model_path) for f in TRAIN_FAMILIES for s in TRAIN_SEEDS]):
                newX += fx
                newy += fy
            X = np.vstack([X, np.asarray(newX, dtype=float)])
            y = np.concatenate([y, np.asarray(newy, dtype=int)])
            model, fit = LearnedUtility.fit(X, y, GROUPS, l2=l2, meta={"groups": list(GROUPS), "l2": l2, "round": r,
                                                                         "n_rows": int(len(y)), "train_seeds": list(TRAIN_SEEDS)})
            model_path = str(out / f"utility_full_r{r}.json")
            model.save(Path(model_path))
            models[f"r{r}"] = model_path
            print(f"round {r}: +{len(newy)} on-policy rows, total {len(y)}, pos rate {y.mean():.3f} ({time.time() - t0:.0f}s)",
                  flush=True)

        variants = {"heuristic": None, **models}
        jobs = [(f, s, v, p) for v, p in variants.items() for f in HEADLINE_FAMILIES for s in DEV_SEEDS]
        res: dict[str, list[float]] = {}
        per_fam: dict[str, dict[str, list[float]]] = {}
        for v, f, s, succ in ex.map(_dev_one, jobs):
            res.setdefault(v, []).append(succ)
            per_fam.setdefault(v, {}).setdefault(f, []).append(succ)
    summary = {v: float(np.mean(x)) for v, x in res.items()}
    for v, m in summary.items():
        print(f"DEV {v:<10s} success={100 * m:.2f}%")
    best = max(summary, key=summary.get)
    report = {"dev_seeds": list(DEV_SEEDS), "families": list(HEADLINE_FAMILIES), "reader": "first", "l2": l2,
              "dev_success": summary, "dev_per_family": {v: {f: float(np.mean(x)) for f, x in d.items()} for v, d in per_fam.items()},
              "best_on_dev": best, "models": models}
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"best on DEV: {best}; wrote {out / 'report.json'}")


if __name__ == "__main__":
    main()
