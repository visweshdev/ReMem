"""Run ReMem vs baselines with a REAL OpenAI-compatible LLM as the agent (and optionally real embeddings).

    export OPENAI_API_KEY=...            # or any key your server accepts
    export OPENAI_BASE_URL=http://127.0.0.1:8001/v1    # optional, for a local server
    python experiments/run_llm_agent.py --model <chat-model> --family distractor --seed 0 --n-tasks 60

NOT validated in the repository's development environment (no endpoint was available): this script and
`remem/llm.py` are tested with a fake client only. Treat the first run as a smoke test, and read the per-task
records before trusting any aggregate. Costs real tokens: roughly (systems x tasks) model calls.

Candidate pools use the offline hash embedder by default (`--embed offline`) so pools are cacheable and
cheap; `--embed openai --embed-model M --embed-dim D` re-embeds with a real model (pools are then built fresh).
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
from remem.config import load_systems  # noqa: E402
from remem.eval.metrics import summarize_run  # noqa: E402
from remem.eval.pools import make_embedder  # noqa: E402
from remem.eval.runner import apply_world_selection, build_pools, run_stream  # noqa: E402
from remem.eval.world import T0, make_world  # noqa: E402
from remem.llm import LLMAgent, OpenAIEmbedder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", default="distractor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default=None, help="comma-separated seeds, e.g. 0,1,2 (overrides --seed)")
    ap.add_argument("--n-tasks", type=int, default=60)
    ap.add_argument("--systems", default="A_vector,C_remem")
    ap.add_argument("--embed", choices=["offline", "openai"], default="offline")
    ap.add_argument("--embed-model")
    ap.add_argument("--embed-dim", type=int, default=1024)
    ap.add_argument("--ollama", action="store_true", help="use a local Ollama server (free; no API key)")
    ap.add_argument("--output", default="schema", choices=["schema", "object", "text"],
                    help="schema: constrained JSON with the exact answer keys (default) | object: any JSON | text: free text")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("This script needs the optional 'openai' package: pip install openai  (or: pip install .[llm])")
    if a.ollama:   # Ollama's OpenAI-compatible endpoint; the key is required by the SDK but ignored by the server
        client = OpenAI(base_url=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/v1"), api_key="ollama")
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("Set OPENAI_API_KEY (and OPENAI_BASE_URL for a local/compatible server), or use --ollama.")
        client = OpenAI()   # reads OPENAI_API_KEY / OPENAI_BASE_URL from the environment
    emb = make_embedder() if a.embed == "offline" else OpenAIEmbedder(client, a.embed_model, a.embed_dim)
    systems = load_systems(ROOT / "configs" / "systems.yaml")
    seeds = [int(x) for x in str(a.seeds).split(",")] if a.seeds else [a.seed]
    out = {"args": vars(a), "runs": []}
    out_path = Path(a.out) if a.out else ROOT / "results" / "llm_agent" / f"{a.model.replace(':', '_')}_{a.family}.json"
    for seed in seeds:
        world = make_world(a.family, seed, {"n_tasks": a.n_tasks})
        pools = build_pools(world, VectorBackend(emb), 20)
        for name in a.systems.split(","):
            cfg = apply_world_selection(systems[name], world)
            if cfg.utility.model_path:
                cfg = replace(cfg, utility=replace(cfg.utility, model_path=str(ROOT / cfg.utility.model_path)))
            holder = {}

            def factory(w, s):
                holder["a"] = LLMAgent(w, client, a.model, t0=T0, json_mode={"schema": "schema", "object": True, "text": False}[a.output])
                return holder["a"]

            t = time.time()
            run = run_stream(world, pools, cfg, emb, seed=seed, agent_factory=factory)
            m = summarize_run(run)
            row = {"seed": seed, "system": name, "task_success": m["task_success"], "gold_recall": m["gold_recall"],
                   "stale_task_rate": m["stale_task_rate"], "tokens_per_task": m["tokens_per_task"],
                   "llm_calls": holder["a"].n_calls, "parse_failures": holder["a"].parse_failures,
                   "seconds": round(time.time() - t, 1),
                   "per_task": [{"task_id": r.task_id, "kind": r.kind, "success": r.success,
                                 "gold_in_context": r.gold_in_context, "n_gold": r.n_gold,
                                 "stale_in_context": r.n_stale_selected} for r in run.records]}
            out["runs"].append(row)
            print(f"seed {seed} {name:10s} success={100 * row['task_success']:.1f}% gold_recall={100 * row['gold_recall']:.1f}% "
                  f"stale_tasks={100 * row['stale_task_rate']:.1f}% parse_failures={row['parse_failures']} "
                  f"({row['seconds']}s)", flush=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")   # save after every run
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
