"""Run ReMem vs similarity-only retrieval with Claude as the agent (real model, real cost).

    pip install anthropic
    export ANTHROPIC_API_KEY=...
    python experiments/run_claude_agent.py --model claude-opus-5-5 --n-tasks 60 --estimate   # print expected cost only
    python experiments/run_claude_agent.py --model claude-opus-5-5 --n-tasks 60

Each system answers the same tasks of one synthetic world (same memories, same candidate pools); only what goes into
Claude's context differs. Results go to results/claude_agent_<model>_<family>.json with per-task records.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from remem.backends import VectorBackend  # noqa: E402
from remem.claude_agent import PRICES, ClaudeAgent  # noqa: E402
from remem.config import load_systems  # noqa: E402
from remem.eval.metrics import summarize_run  # noqa: E402
from remem.eval.pools import make_embedder  # noqa: E402
from remem.eval.runner import apply_world_selection, build_pools, run_stream  # noqa: E402
from remem.eval.world import T0, make_world  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="e.g. claude-opus-5-5, claude-sonnet-5-5, claude-haiku-5-5")
    ap.add_argument("--family", default="contradiction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=60)
    ap.add_argument("--systems", default="A_vector,C_remem")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--estimate", action="store_true", help="print a cost estimate and exit without calling the API")
    a = ap.parse_args()

    systems = a.systems.split(",")
    n_calls = a.n_tasks * len(systems)
    if a.model in PRICES:
        pin, pout = PRICES[a.model]
        # ~450 input tokens per call (system + 5 memories + question); output incl. thinking ~150-600 at low effort
        lo, hi = n_calls * (450 * pin + 150 * pout) / 1e6, n_calls * (450 * pin + 600 * pout) / 1e6
        print(f"{n_calls} calls to {a.model}: estimated ${lo:.2f}-${hi:.2f}")
    if a.estimate:
        return

    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic")
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        sys.exit("Set ANTHROPIC_API_KEY before running this script.")
    client = anthropic.Anthropic()

    emb = make_embedder()
    world = make_world(a.family, a.seed, {"n_tasks": a.n_tasks})
    pools = build_pools(world, VectorBackend(emb), 20)
    cfgs = load_systems(ROOT / "configs" / "systems.yaml")
    out = {"args": vars(a), "results": {}}
    total_cost = 0.0
    for name in systems:
        cfg = apply_world_selection(cfgs[name], world)
        if cfg.utility.model_path:
            cfg = replace(cfg, utility=replace(cfg.utility, model_path=str(ROOT / cfg.utility.model_path)))
        holder = {}

        def factory(w, seed):
            holder["agent"] = ClaudeAgent(w, client, a.model, t0=T0, effort=a.effort)
            return holder["agent"]

        t = time.time()
        try:
            run = run_stream(world, pools, cfg, emb, seed=a.seed, agent_factory=factory)
        except anthropic.AuthenticationError:
            sys.exit("Invalid ANTHROPIC_API_KEY.")
        except anthropic.NotFoundError:
            sys.exit(f"Model {a.model!r} not found for this key.")
        ag = holder["agent"]
        m = summarize_run(run)
        cost = ag.cost_usd()
        total_cost += cost or 0.0
        out["results"][name] = {
            "task_success": m["task_success"], "gold_recall": m["gold_recall"], "tokens_per_task": m["tokens_per_task"],
            "calls": ag.calls, "refusals": ag.refusals, "parse_failures": ag.parse_failures,
            "input_tokens": ag.input_tokens, "output_tokens": ag.output_tokens, "cost_usd": cost,
            "seconds": round(time.time() - t, 1),
            "per_task": [{"task_id": r.task_id, "kind": r.kind, "success": r.success, "gold_in_context": r.gold_in_context,
                          "n_gold": r.n_gold, "stale_in_context": r.n_stale_selected} for r in run.records],
        }
        print(f"{name:10s} success={100 * m['task_success']:.1f}%  gold_recall={100 * m['gold_recall']:.1f}%  "
              f"context_tokens/task={m['tokens_per_task']:.1f}  refusals={ag.refusals}  "
              f"cost=${(cost or 0):.3f}", flush=True)
    path = ROOT / "results" / f"claude_agent_{a.model}_{a.family}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"total cost ${total_cost:.3f}; wrote {path}")


if __name__ == "__main__":
    main()
