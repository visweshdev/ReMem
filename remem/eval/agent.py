"""Deterministic simulated agent ("reader") used as the memory consumer in the benchmark.

This is NOT an LLM. It is a controlled consumer whose behaviour is fully specified here, so
that differences between memory systems come from what they put in the context and nothing
else. Its assumptions, which the README repeats as limitations:

  * It reads each context item with the template-independent parser `parse_text`.
  * To answer slot (subject, attribute) it needs a context item that asserts a value for it.
      mode "first"  : takes the first matching item in context order (position bias).
      mode "latest" : takes the matching item with the newest timestamp (<= as_of if given).
  * Distraction: with probability  base + per_irrelevant * (#non-matching items)  (capped) it
    ignores the context and guesses. Wasted context therefore has a cost.
  * A missing or distracted slot is a guess among the attribute's value pool, so it can be right
    by chance (1/|pool|): "success without the memory" is possible.
  * It self-reports citations imperfectly: it cites the item it used with probability cite_p, and
    each other item with probability false_cite_p.
  * All randomness comes from a per-task seed, and random draws that decide success are made in
    a fixed order, so a counterfactual replay with one item removed changes nothing but the context.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from remem.eval.world import Task, World, parse_text
from remem.types import AgentTrace


@dataclass
class AgentConfig:
    mode: str = "first"              # first | latest
    base_distract: float = 0.02
    per_irrelevant: float = 0.05
    max_distract: float = 0.5
    cite_p: float = 0.9
    false_cite_p: float = 0.04


@dataclass
class AgentResult:
    trace: AgentTrace
    success: bool
    used_ids: list[str] = field(default_factory=list)     # hidden: items the agent actually relied on
    distracted: bool = False
    guessed_slots: list[str] = field(default_factory=list)
    n_irrelevant: int = 0


Item = tuple[str, str, float]    # (fact_id, text served, fact timestamp)


class SimAgent:
    def __init__(self, world: World, cfg: AgentConfig, seed: int) -> None:
        self.world = world
        self.cfg = cfg
        self.seed = seed

    def act(self, task: Task, items: list[Item]) -> AgentResult:
        cfg, world = self.cfg, self.world
        rng = random.Random(f"{self.seed}:{task.task_id}")
        u_distract = rng.random()
        guess_draws = {f"{s}.{a}": rng.random() for s, a in task.slots}

        parsed = [(fid, text, ts, parse_text(text, world.vocab)) for fid, text, ts in items]
        slot_keys = set(task.slots)
        n_irr = sum(1 for _, _, _, p in parsed if p is None or (p[0], p[1]) not in slot_keys)
        p_dist = min(cfg.max_distract, cfg.base_distract + cfg.per_irrelevant * n_irr)
        distracted = u_distract < p_dist

        answer: dict[str, str] = {}
        used: list[str] = []
        guessed: list[str] = []
        ref = task.as_of
        for s, a in task.slots:
            key = f"{s}.{a}"
            matches = [(fid, ts, p[2]) for fid, _, ts, p in parsed if p is not None and (p[0], p[1]) == (s, a)]
            if cfg.mode == "latest" and matches:
                cands = [m for m in matches if ref is None or m[1] <= ref]
                matches = [max(cands, key=lambda m: m[1])] if cands else []
            if matches and not distracted:
                fid, _, val = matches[0]
                answer[key] = val
                used.append(fid)
            else:
                pool = world.vocab.attrs[a].pool
                answer[key] = pool[int(guess_draws[key] * len(pool)) % len(pool)]
                guessed.append(key)

        truth = world.answer(task)
        success = all(answer[k] == truth[k] for k in truth)
        cited: list[str] = []
        for fid, _, _, _ in parsed:
            if fid in used:
                if rng.random() < cfg.cite_p:
                    cited.append(fid)
            elif rng.random() < cfg.false_cite_p:
                cited.append(fid)
        trace = AgentTrace(answer=dict(answer), cited_ids=cited, tool_args=dict(answer))
        return AgentResult(trace, success, used, distracted, guessed, n_irr)
