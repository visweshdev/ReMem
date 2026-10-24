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
    ap.add_argument("--n-tasks", type=int, default=60)
    ap.add_argument("--systems", default="A_vector,C_remem")
    ap.add_argument("--embed", choices=["offline", "openai"], default="offline")
    ap.add_argument("--embed-model")
    ap.add_argument("--embed-dim", type=int, default=1024)
    ap.add_argument("--out", default=str(ROOT / "results" / "llm_agent.json"))
    a = ap.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("This script needs the optional 'openai' package: pip install openai  (or: pip install .[llm])")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (and OPENAI_BASE_URL for a local/compatible server) before running this script.")
    client = OpenAI()   # reads OPENAI_API_KEY / OPENAI_BASE_URL from the environment
    emb = make_embedder() if a.embed == "offline" else OpenAIEmbedder(client, a.embed_model, a.embed_dim)
    world = make_world(a.family, a.seed, {"n_tasks": a.n_tasks})
    pools = build_pools(world, VectorBackend(emb), 20)
    systems = load_systems(ROOT / "configs" / "systems.yaml")
    out = {}
    for name in a.systems.split(","):
        cfg = apply_world_selection(systems[name], world)
        if cfg.utility.model_path:
            cfg = replace(cfg, utility=replace(cfg.utility, model_path=str(ROOT / cfg.utility.model_path)))
        agent_holder = {}

        def factory(w, seed):
            agent_holder["a"] = LLMAgent(w, client, a.model, t0=T0)
            return agent_holder["a"]

        run = run_stream(world, pools, cfg, emb, seed=a.seed, agent_factory=factory)
        m = summarize_run(run)
        out[name] = {"task_success": m["task_success"], "tokens_per_task": m["tokens_per_task"],
                     "llm_calls": agent_holder["a"].n_calls, "n_tasks": m["n_tasks"]}
        print(name, out[name])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "results": out}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
