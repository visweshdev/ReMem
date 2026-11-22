"""Replay a support-routing assistant over several sessions with a local acceptance checker."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from statistics import mean, median
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from remem.api import Hit, ReMem, default_config
from remem.config import to_dict

ARMS = ("plain_search", "feedback_off", "signals", "signals_loo")
INTENTS = ("invoice refund", "shipping address change", "login recovery", "subscription cancellation")
WARMUP = ("Route this customer support request for {scope}.", "Which queue handles {scope} tickets?",
          "Find the support routing procedure for {scope}.", "Choose the right support queue for {scope}.")
HOLDOUT = ("Where should I forward a case about {scope}?", "Assign the {scope} help request to an internal team.",
           "A customer needs help with {scope}. Choose an assignment.", "Send the {scope} case to the responsible queue.")
SESSION_SIZE, WARMUP_SESSIONS, TEST_SESSIONS = 8, 8, 6
TARGET = re.compile(r"queue ([a-z0-9]+)")


def configs():
    remem = default_config(top_k=3, budget_tokens=100)
    plain = replace(remem, ranker="sim", outcome_feedback=False, retention=replace(remem.retention, mode="none"),
                    selection=replace(remem.selection, mode="rank_pack", min_sim=0, guard_top=0))
    return dict(zip(ARMS, (plain, replace(remem, outcome_feedback=False), remem,
                           replace(remem, attribution=replace(remem.attribution, method="signals_loo")))))


def choose_queue(hits: list[Hit]) -> str | None:
    for hit in hits:
        match = TARGET.search(hit.text)
        if match:
            return match[1]
    return None


def run_episode(root: Path, arm: str, seed: int, *, config=None) -> dict:
    rng = random.Random(seed)
    config = configs()[arm] if config is None else config
    teams = ("willow", "pine", "elm", "birch", "cedar", "ash", "maple", "oak", "alder", "almon", "linden", "yew")
    names = rng.sample(teams, len(teams))
    queues = dict(zip(INTENTS, names))
    documents = []
    for i, scope in enumerate(INTENTS):
        queue = queues[scope]
        documents.extend([
            (f"policy-{i}", f"Team handbook for {scope}. Escalate via queue {queue}.", 100),
            (f"example-{i}", f"Route this customer support request for {scope}. Example support routing uses queue {names[i+4]}.", 200),
            (f"ticket-{i}", f"A customer needs help with {scope}. Past example from queue {names[i+8]}.", 300),
            (f"glossary-{i}", f"Training glossary for {scope}: the sample team queue {queue} is used in formatting examples.", 150),
        ])
    rng.shuffle(documents)
    init = time.perf_counter()
    memory = ReMem(root, config=config, autosave=False)
    for ident, text, timestamp in documents:
        memory.add(text, role="note", fact_id=ident, when=timestamp)
    memory.save()
    indexing_ms = (time.perf_counter() - init) * 1000
    size_before = sum(file.stat().st_size for file in root.iterdir() if file.is_file())

    rows = []
    replay_calls = 0
    valid_credit = wrong_credit = 0.0
    for session in range(WARMUP_SESSIONS + TEST_SESSIONS):
        memory = ReMem(root, autosave=False)
        tests = session >= WARMUP_SESSIONS
        memory.add(f"Support hand-off {session + 1}. Review pending tickets during the next shift.",
                   role="note", fact_id=f"handoff-{session}", when=900 + session)
        body = HOLDOUT if tests else WARMUP
        tasks = list(INTENTS) * (SESSION_SIZE // len(INTENTS))
        rng.shuffle(tasks)
        for j, scope in enumerate(tasks):
            index = session * SESSION_SIZE + j
            query = body[(index + session) % len(body)].format(scope=scope)
            t0 = time.perf_counter()
            result = memory.search(query, now=1000 + index)
            latency_ms = (time.perf_counter() - t0) * 1000
            chosen = choose_queue(result.items)
            success = chosen == queues[scope]
            if not tests and result.search_id:
                # Deterministic citation noise simulates a reader picking an unrelated shown item.
                cited = result.items[-1:] if index % 3 == 0 else result.items[:1]

                def replay(hits: list[Hit], expected=queues[scope]) -> bool:
                    nonlocal replay_calls
                    replay_calls += 1
                    return choose_queue(hits) == expected

                credit = memory.feedback(result.search_id, success, used_ids=[hit.id for hit in cited], answer=chosen,
                                         replay=replay if arm == "signals_loo" else None)
                ident = f"policy-{INTENTS.index(scope)}"
                valid_credit += credit.get(ident, 0)
                wrong_credit += sum(value for key, value in credit.items() if key != ident)
            rows.append({"session": session, "query": query, "phase": "test" if tests else "warmup", "correct": success,
                         "selected": [hit.id for hit in result.items], "tokens": result.tokens, "retrieval_ms": latency_ms})
        memory.save()

    latency = sorted(row["retrieval_ms"] for row in rows)
    return {"seed": seed, "arm": arm, "warmup_accuracy": mean(row["correct"] for row in rows if row["phase"] == "warmup"),
            "test_accuracy": mean(row["correct"] for row in rows if row["phase"] == "test"),
            "indexing_ms": indexing_ms, "retrieval_p50_ms": median(latency), "retrieval_p95_ms": latency[int(.95 * len(latency))],
            "mean_context_tokens": mean(row["tokens"] for row in rows), "memories": memory.stats()["memories"],
            "initial_storage_bytes": size_before, "final_storage_bytes": sum(file.stat().st_size for file in root.iterdir() if file.is_file()),
            "replay_calls": replay_calls, "policy_credit": valid_credit, "other_credit": wrong_credit, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    seeds = range(5) if args.split == "dev" else range(100, 120)
    with TemporaryDirectory(prefix="remem-routing-") as temp:
        base = Path(temp)
        episodes = [run_episode(base / f"{arm}-{seed}", arm, seed) for seed in seeds for arm in ARMS]
    metrics = [k for k in episodes[0] if k not in ("seed", "arm", "rows")]
    summary = {arm: {key: mean(ep[key] for ep in episodes if ep["arm"] == arm) for key in metrics} for arm in ARMS}
    by_arm = {arm: np.array([ep["test_accuracy"] for ep in episodes if ep["arm"] == arm]) for arm in ARMS}
    sample = np.random.default_rng(0).integers(len(seeds), size=(10_000, len(seeds)))
    paired = {}
    for arm, baseline in (("signals", "feedback_off"), ("signals_loo", "feedback_off"), ("signals_loo", "signals"),
                          ("signals", "plain_search"), ("signals_loo", "plain_search")):
        differences = by_arm[arm] - by_arm[baseline]
        paired[f"{arm}_minus_{baseline}"] = {"delta": differences.mean(), "ci95": np.quantile(differences[sample].mean(axis=1),
                                                                                           [.025, .975]).tolist()}
    output = {"split": args.split, "seeds": list(seeds), "warmup_tasks_per_seed": SESSION_SIZE * WARMUP_SESSIONS,
              "test_tasks_per_seed": SESSION_SIZE * TEST_SESSIONS, "config": {k: to_dict(v) for k, v in configs().items()},
              "summary": summary, "paired_test_accuracy": paired,
              "episodes": [{k: v for k, v in ep.items() if k != "rows"} for ep in episodes]}
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
