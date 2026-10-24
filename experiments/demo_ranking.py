"""Find and print concrete cases where ReMem outranks a more-similar memory with a more useful one.

    python experiments/demo_ranking.py [--family distractor] [--seed 0] [--out results/demo_ranking.md]

For one test world it replays ReMem (D) and, at every task, looks for a *served* memory B and a
*not-served* memory A in the same candidate pool such that sim(A) > sim(B), B is the gold memory
for the task and A is a look-alike that never helped. It prints both memories with similarity,
outcome history and the utility components that decided the order. Also writes the memory
lifecycle of the demoted look-alike and the promoted gold memory.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from remem.config import load_systems  # noqa: E402
from remem.eval.agent import AgentConfig  # noqa: E402
from remem.eval.pools import load_pools, make_embedder  # noqa: E402
from remem.eval.runner import apply_world_selection, run_stream  # noqa: E402
from remem.eval.world import make_world  # noqa: E402
from remem.viz.ledger_cli import lifecycle  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="distractor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "demo_ranking.md"))
    ap.add_argument("--cache", default=str(ROOT / "results" / "cache" / "pools"))
    a = ap.parse_args()

    world = make_world(a.family, a.seed)
    systems = load_systems(ROOT / "configs" / "systems.yaml")
    cfg = apply_world_selection(systems["C_remem"], world)
    cfg = replace(cfg, utility=replace(cfg.utility, model_path=str(ROOT / cfg.utility.model_path)))
    pools = load_pools(a.cache, a.family, a.seed, cfg.backend, cfg.pool_size)
    kinds = {f.fact_id: f.kind for f in world.facts}
    found: list[dict] = []

    def hook(task, ctx, res, system):
        gold = set(world.gold_ids(task))
        served = {s.cand.fact_id: s for s in ctx.selected}
        for g in gold:
            if g not in served:
                continue
            for s in ctx.scored:
                fid = s.cand.fact_id
                if fid in served or kinds.get(fid) not in ("trap", "hub"):
                    continue
                if s.cand.sim > served[g].cand.sim + 0.02:
                    rec_a, rec_b = system.ledger.get(fid), system.ledger.get(g)
                    found.append({
                        "task": task.task_id, "query": task.query, "step": system.ledger.step,
                        "A": (fid, s.cand.text, s.cand.sim, s.prob, dict(s.components), rec_a.n_success_ctx, rec_a.n_credited,
                              rec_a.state.value, lifecycle(rec_a, system.ledger)),
                        "B": (g, served[g].cand.text, served[g].cand.sim, served[g].prob, dict(served[g].components),
                              rec_b.n_success_ctx, rec_b.n_credited, rec_b.state.value, lifecycle(rec_b, system.ledger)),
                    })
                    return

    run_stream(world, pools, cfg, make_embedder(), AgentConfig(mode="first"), seed=a.seed, on_task=hook)

    def fmt(tag, t):
        fid, text, sim, p, comps, expo, cred, state, life = t
        top = ", ".join(f"{k}={v:+.2f}" for k, v in sorted(comps.items(), key=lambda kv: -abs(kv[1]))[:4])
        return (f"- **{tag}** `{fid}` — \"{text}\"\n  - similarity {sim:.3f}, P(contributes) {p:.2f}, state {state}\n"
                f"  - shown in {expo} successful tasks, credited {cred:.1f}\n  - utility components: {top}\n  - lifecycle: {life}")

    md = [f"# Ranking demonstration ({a.family}, seed {a.seed})\n",
          f"{len(found)} tasks had a more-similar look-alike *not served* while a less-similar gold memory *was* served.\n",
          "Memory **A** has higher semantic similarity but a record of never helping; memory **B** is less similar "
          "but has contributed to successful tasks before.\n"]
    if found:
        # pick the latest-in-stream examples (history has accumulated) and the first one for contrast
        picks = [found[0]] + found[-2:]
        for f in picks:
            md.append(f"\n## Task {f['task']} (ledger step {f['step']}): {f['query']}\n")
            md.append(fmt("A (more similar, NOT served)", f["A"]))
            md.append(fmt("B (less similar, served)", f["B"]))
    else:
        md.append("\nNo such case in this world.\n")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
