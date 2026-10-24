"""Automatic failure analysis: classify every task of a run into failure/anomaly categories and keep examples.

Categories (a task can carry several tags):

  gold_not_in_pool          the backend never offered the correct memory
  gold_pooled_not_selected  the correct memory was offered but ranking/selection left it out
  utility_demoted_gold      ... and it had a top-k similarity rank, i.e. the utility layer pushed it out
  useful_filtered           the correct memory was removed by retention/validity (archived or invalidated)
  irrelevant_retrieved      the task failed and the agent was distracted by irrelevant context
  outdated_preferred        the task failed while a stale/other-valued memory for the slot was in context
  success_without_memory    the task succeeded although some gold memory was missing (agent guessed right)
  fail_despite_gold         the task failed although every gold memory was in context
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from remem.eval.runner import RunResult, TaskRecord
from remem.eval.world import World

CATEGORIES = (
    "gold_not_in_pool", "gold_pooled_not_selected", "utility_demoted_gold", "useful_filtered",
    "irrelevant_retrieved", "outdated_preferred", "success_without_memory", "fail_despite_gold",
)


def classify(rec: TaskRecord, top_k: int) -> list[str]:
    tags: list[str] = []
    missing_pool = rec.n_gold - rec.gold_in_pool
    if missing_pool > 0:
        tags.append("gold_not_in_pool")
    unselected_pooled = rec.gold_in_pool - rec.gold_in_context - rec.gold_filtered
    if unselected_pooled > 0:
        tags.append("gold_pooled_not_selected")
        for g_rank, in_ctx in zip(rec.gold_ranks_pool, rec.gold_ranks_ctx):
            if in_ctx < 0 and 0 <= g_rank < top_k:
                tags.append("utility_demoted_gold")
                break
    if rec.gold_filtered > 0:
        tags.append("useful_filtered")
    if not rec.success and rec.distracted and rec.n_irrelevant > 0:
        tags.append("irrelevant_retrieved")
    if not rec.success and rec.n_stale_selected > 0:
        tags.append("outdated_preferred")
    if rec.success and rec.gold_in_context < rec.n_gold:
        tags.append("success_without_memory")
    if not rec.success and rec.gold_in_context == rec.n_gold:
        tags.append("fail_despite_gold")
    return tags


def analyse(run: RunResult, world: World, top_k: int, max_examples: int = 3) -> dict[str, Any]:
    text = {f.fact_id: f for f in world.facts}
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {c: [] for c in CATEGORIES}
    for rec in run.records:
        tags = classify(rec, top_k)
        for t in set(tags):
            counts[t] += 1
            if len(examples[t]) < max_examples:
                examples[t].append({
                    "family": run.family, "seed": run.seed, "system": run.system, "task_id": rec.task_id,
                    "kind": rec.kind, "query": rec.query, "success": rec.success, "tags": sorted(set(tags)),
                    "gold": [
                        {"id": g, "text": text[g].text, "pool_rank": pr, "ctx_pos": cp, "state": st}
                        for g, pr, cp, st in zip(rec.gold_ids, rec.gold_ranks_pool, rec.gold_ranks_ctx, rec.gold_states)
                    ],
                    "selected": [
                        {"id": i, "kind": k, "state": st, "text": text[i].text}
                        for i, k, st in zip(rec.selected_ids, rec.selected_kinds, rec.selected_states)
                    ],
                    "agent_distracted": rec.distracted, "agent_guessed": rec.guessed_slots,
                })
    return {"n_tasks": len(run.records), "counts": dict(counts), "examples": examples}
