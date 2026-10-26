"""Record every prompt, reply and outcome of a local-LLM run, to diagnose why systems differ.

    python experiments/trace_llm_prompts.py --family contradiction --seed 0 --n-tasks 80

Writes results/llm_agent/traces_<family>_<seed>.jsonl (one line per system x task).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from remem.backends import VectorBackend  # noqa: E402
from remem.config import load_systems  # noqa: E402
from remem.eval.pools import make_embedder  # noqa: E402
from remem.eval.runner import apply_world_selection, build_pools, run_stream  # noqa: E402
from remem.eval.world import T0, make_world  # noqa: E402
from remem.llm import LLMAgent, build_prompt  # noqa: E402


class Tracing(LLMAgent):
    def __init__(self, *a, sink, system, **kw):
        super().__init__(*a, **kw)
        self.sink, self.system = sink, system

    def act(self, task, items):
        orig = self.client.chat.completions.create
        box = {}

        def spy(**kw):
            box["resp"] = orig(**kw)
            return box["resp"]

        self.client.chat.completions.create = spy
        try:
            res = super().act(task, items)
        finally:
            self.client.chat.completions.create = orig
        gold = set(self.world.gold_ids(task))
        self.sink.write(json.dumps({
            "system": self.system, "task_id": task.task_id, "success": res.success, "truth": self.world.answer(task),
            "reply": box["resp"].choices[0].message.content, "gold_pos": [i for i, (fid, _, _) in enumerate(items) if fid in gold],
            "prompt": build_prompt(task, items, self.t0)}) + "\n")
        self.sink.flush()
        return res


def main() -> None:
    from openai import OpenAI
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="contradiction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=80)
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--systems", default="A_vector,C_remem")
    ap.add_argument("--output", default="schema", choices=["schema", "object"])
    a = ap.parse_args()
    client = OpenAI(base_url=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/v1"), api_key="ollama")
    emb = make_embedder()
    world = make_world(a.family, a.seed, {"n_tasks": a.n_tasks})
    pools = build_pools(world, VectorBackend(emb), 20)
    cfgs = load_systems(ROOT / "configs" / "systems.yaml")
    out = ROOT / "results" / "llm_agent" / f"traces_{a.family}_{a.seed}_{a.output}.jsonl"
    with open(out, "w", encoding="utf-8") as sink:
        for name in a.systems.split(","):
            cfg = apply_world_selection(cfgs[name], world)
            if cfg.utility.model_path:
                from dataclasses import replace
                cfg = replace(cfg, utility=replace(cfg.utility, model_path=str(ROOT / cfg.utility.model_path)))
            run_stream(world, pools, cfg, emb, seed=a.seed, agent_factory=lambda w, s, name=name: Tracing(
                w, client, a.model, t0=T0, json_mode=a.output if a.output != 'object' else True, sink=sink, system=name))
            print("done", name, flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
