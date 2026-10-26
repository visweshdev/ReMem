"""Re-run the headline comparison with a real (free, local) embedding model instead of the offline hash embedder.

    pip install sentence-transformers
    python experiments/run_real_embeddings.py                    # all-MiniLM-L6-v2, 7 core families x 15 test seeds
    python experiments/run_real_embeddings.py --seeds 0 --families distractor   # quick check

Same worlds, systems and simulated reader ("first") as the main benchmark; only the embedder (and therefore every
candidate pool) changes. Writes results/real_embeddings.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SYSTEMS = ("A_vector", "B_static", "C_remem")
_EMB = None


def _init(model: str, threads: int) -> None:
    global _EMB
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    torch.set_num_threads(threads)
    from remem.embedders import SentenceTransformerEmbedder
    _EMB = SentenceTransformerEmbedder(model)


def run_one(job: tuple[str, int]) -> list[dict]:
    from remem.backends import VectorBackend
    from remem.config import load_systems
    from remem.eval.agent import AgentConfig
    from remem.eval.metrics import summarize_run
    from remem.eval.runner import apply_world_selection, build_pools, run_stream
    from remem.eval.world import make_world

    family, seed = job
    world = make_world(family, seed)
    _EMB.warm([f.text for f in world.facts] + [t.query for t in world.tasks])
    systems = load_systems(ROOT / "configs" / "systems.yaml")
    pools_by_size: dict[int, object] = {}
    rows = []
    for name in SYSTEMS:
        cfg = apply_world_selection(systems[name], world)
        if cfg.utility.model_path:
            cfg = replace(cfg, utility=replace(cfg.utility, model_path=str(ROOT / cfg.utility.model_path)))
        if cfg.pool_size not in pools_by_size:
            pools_by_size[cfg.pool_size] = build_pools(world, VectorBackend(_EMB), cfg.pool_size)
        run = run_stream(world, pools_by_size[cfg.pool_size], cfg, _EMB, AgentConfig(mode="first"), seed=seed)
        m = summarize_run(run)
        rows.append({"family": family, "seed": seed, "system": name, "task_success": m["task_success"],
                     "pool_recall": m["pool_recall"], "tokens_per_task": m["tokens_per_task"]})
    return rows


def _ci(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    mu = sum(xs) / n
    if n < 2:
        return mu, float("nan")
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))
    # two-sided 95% Student-t critical values (df 1..30); normal approximation beyond
    t = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
         2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]
    return mu, float((t[n - 2] if n - 1 <= 30 else 1.96) * sd / math.sqrt(n))


def main() -> None:
    from remem.eval.protocol import HEADLINE_FAMILIES, TEST_SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--families", default=",".join(HEADLINE_FAMILIES))
    ap.add_argument("--seeds", default=",".join(map(str, TEST_SEEDS)))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2, help="torch threads per worker")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "real_embeddings.json")
    a = ap.parse_args()

    jobs = [(f, int(s)) for f in a.families.split(",") for s in a.seeds.split(",")]
    rows: list[dict] = []
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.model, a.threads)) as ex:
        for i, r in enumerate(ex.map(run_one, jobs), 1):
            rows.extend(r)
            print(f"[{i}/{len(jobs)}] {r[0]['family']} seed {r[0]['seed']}: " +
                  "  ".join(f"{x['system']}={100 * x['task_success']:.1f}%" for x in r) +
                  f"  ({time.time() - t0:.0f}s)", flush=True)

    summary = {}
    for s in SYSTEMS:
        per_seed = {}
        for r in rows:
            if r["system"] == s:
                per_seed.setdefault(r["seed"], []).append(r)
        succ = [sum(x["task_success"] for x in v) / len(v) for v in per_seed.values()]
        pool = [sum(x["pool_recall"] for x in v) / len(v) for v in per_seed.values()]
        mu, hw = _ci(succ)
        summary[s] = {"task_success": mu, "ci95": hw, "pool_recall": sum(pool) / len(pool)}
    deltas = {}
    for s in ("A_vector", "B_static"):
        d = []
        for seed in sorted({r["seed"] for r in rows}):
            c = [r["task_success"] for r in rows if r["system"] == "C_remem" and r["seed"] == seed]
            b = [r["task_success"] for r in rows if r["system"] == s and r["seed"] == seed]
            d.append(sum(c) / len(c) - sum(b) / len(b))
        mu, hw = _ci(d)
        deltas[f"C_remem - {s}"] = {"mean": mu, "ci95": hw}
    out = {"model": a.model, "families": a.families.split(","), "seeds": a.seeds.split(","), "reader": "first",
           "summary": summary, "paired_deltas": deltas, "rows": rows}
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"summary": summary, "paired_deltas": deltas}, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
