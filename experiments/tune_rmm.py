"""Choose the RMM-style baseline's learning rate and temperature on DEV seeds (never test seeds).

    python experiments/tune_rmm.py

Prints mean task success over the core families for each (lr, tau) and the best pair. The chosen values are then
written by hand into configs/systems.yaml (`rmm_style`), so the tuning is visible and reproducible.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from remem.config import load_systems  # noqa: E402
from remem.eval.agent import AgentConfig  # noqa: E402
from remem.eval.pools import load_pools, make_embedder  # noqa: E402
from remem.eval.protocol import DEV_SEEDS, HEADLINE_FAMILIES  # noqa: E402
from remem.eval.runner import apply_world_selection, run_stream  # noqa: E402
from remem.eval.world import make_world  # noqa: E402

LRS = (1e-3, 1e-2, 1e-1, 1.0)
TAUS = (0.02, 0.05, 0.2)


def main() -> None:
    cache = ROOT / "results" / "cache" / "pools"
    base = load_systems(ROOT / "configs" / "systems.yaml")["rmm_style"]
    emb = make_embedder()
    results = {}
    for lr, tau in itertools.product(LRS, TAUS):
        cfg0 = replace(base, utility=replace(base.utility, rmm_lr=lr, rmm_tau=tau))
        succ = []
        for fam in HEADLINE_FAMILIES:
            for seed in DEV_SEEDS:
                w = make_world(fam, seed)
                run = run_stream(w, load_pools(cache, fam, seed, "vector", 20), apply_world_selection(cfg0, w), emb,
                                 AgentConfig(mode="first"), seed=seed)
                succ.append(np.mean([r.success for r in run.records]))
        results[(lr, tau)] = float(np.mean(succ))
        print(f"lr={lr:<6g} tau={tau:<5g} dev success={100 * results[(lr, tau)]:.1f}%", flush=True)
    best = max(results, key=results.get)
    print(f"best: lr={best[0]:g} tau={best[1]:g} ({100 * results[best]:.1f}%)")


if __name__ == "__main__":
    main()
