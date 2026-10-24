"""Metrics computed from `RunResult`s (hidden ground truth enters only here)."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from remem.eval.runner import RunResult, TaskRecord

# two-sided 95% t critical values, df = 1..30 (n seeds - 1)
_T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
        2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def mean_ci(values: Sequence[float]) -> tuple[float, float]:
    """(mean, half-width of the 95% t-interval) over independent seeds."""
    v = np.asarray([x for x in values if x is not None and not math.isnan(x)], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    if len(v) == 1:
        return float(v[0]), float("nan")
    t = _T95[min(len(v) - 2, len(_T95) - 1)]
    return float(v.mean()), float(t * v.std(ddof=1) / math.sqrt(len(v)))


def auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """ROC AUC via the Mann-Whitney U statistic (ties get half credit)."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def brier_and_ece(p: Sequence[float], y: Sequence[int], bins: int = 10) -> tuple[float, float]:
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    y = np.asarray(y, dtype=float)
    if len(p) == 0:
        return float("nan"), float("nan")
    brier = float(np.mean((p - y) ** 2))
    ece = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return brier, float(ece)


def _frac(num: float, den: float) -> float:
    return num / den if den else float("nan")


def summarize_records(recs: Sequence[TaskRecord]) -> dict[str, float]:
    n = len(recs)
    if n == 0:
        return {}
    gold_total = sum(r.n_gold for r in recs)
    sel_total = sum(r.n_selected for r in recs)
    lat = np.asarray([r.latency_ms for r in recs])
    out = {
        "n_tasks": n,
        "task_success": sum(r.success for r in recs) / n,
        "gold_recall": _frac(sum(r.gold_in_context for r in recs), gold_total),
        "pool_recall": _frac(sum(r.gold_in_pool for r in recs), gold_total),
        "all_gold_in_context": sum(r.gold_in_context == r.n_gold for r in recs) / n,
        "context_precision": _frac(sum(r.gold_in_context for r in recs), sel_total),
        "irrelevant_per_task": sum(r.n_irrelevant for r in recs) / n,
        "useful_per_task": sum(r.gold_in_context for r in recs) / n,
        "tokens_per_task": sum(r.tokens for r in recs) / n,
        "memories_per_task": sel_total / n,
        "latency_ms_mean": float(lat.mean()),
        "latency_ms_p95": float(np.percentile(lat, 95)),
        "stale_task_rate": sum(r.n_stale_selected > 0 for r in recs) / n,
        "trap_per_task": sum(r.n_trap_selected for r in recs) / n,
        "hub_per_task": sum(r.n_hub_selected for r in recs) / n,
        "success_without_gold": sum(r.success and r.gold_in_context < r.n_gold for r in recs) / n,
        "fail_with_gold": sum((not r.success) and r.gold_in_context == r.n_gold for r in recs) / n,
        "gold_filtered_rate": _frac(sum(r.gold_filtered for r in recs), gold_total),
        "tokens_per_success": _frac(sum(r.tokens for r in recs), sum(r.success for r in recs)),
    }
    for kind in ("current", "historical", "critical_dormant", "critical_shown"):
        sub = [r for r in recs if r.kind == kind]
        if sub:
            out[f"success_{kind}"] = sum(r.success for r in sub) / len(sub)
    if any(r.budget is not None for r in recs):
        out["budget_overrun_rate"] = sum(r.budget is not None and r.tokens > r.budget for r in recs) / n
    return out


def summarize_run(run: RunResult) -> dict[str, float]:
    recs = run.records
    out = summarize_records(recs)
    half = recs[len(recs) // 2:]
    out["task_success_second_half"] = sum(r.success for r in half) / max(1, len(half))
    out["task_success_first_half"] = sum(r.success for r in recs[: len(recs) // 2]) / max(1, len(recs) // 2)
    out["extra_agent_calls_per_task"] = sum(r.extra_agent_calls for r in recs) / max(1, len(recs))

    # utility prediction quality over considered candidates
    ps = [t[0] for r in recs for t in r.scored_probs]
    ss = [t[1] for r in recs for t in r.scored_probs]
    ys = [t[2] for r in recs for t in r.scored_probs]
    out["utility_auc"] = auc(ps, ys)
    out["sim_auc"] = auc(ss, ys)
    brier, ece = brier_and_ece(ps, ys)
    out["utility_brier"] = brier
    out["utility_ece"] = ece

    # per-task top-1: does the highest-scored considered candidate belong to gold?
    # (computed from scored_probs ordering is not stored per id; use AUC above as the primary measure)

    # attribution accuracy (hidden labels): credited iff credit >= 0.5; truth: fact is gold for the task
    tp = fp = fn = 0
    for r in recs:
        if not r.attributed or not r.success:
            continue
        gold = set(r.gold_ids)
        for fid, credit in r.attributed.items():
            credited = credit >= 0.5
            if credited and fid in gold:
                tp += 1
            elif credited:
                fp += 1
        fn += sum(1 for g in gold if g in r.attributed and r.attributed[g] < 0.5)
    out["attr_precision"] = _frac(tp, tp + fp)
    out["attr_recall"] = _frac(tp, tp + fn)

    # memory growth
    last = run.snapshots[-1]
    out["mem_total"] = last["n_memories"]
    out["mem_served_tokens"] = last["served_tokens"]
    out["mem_total_tokens"] = last["total_tokens"]
    for st, c in last["states"].items():
        out[f"state_{st}"] = c
    out["false_invalidations"] = run.false_invalidations
    out["missed_invalidations"] = run.missed_invalidations
    return out


def aggregate(per_seed: Iterable[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Per-metric mean and 95% CI half-width across seeds."""
    rows = list(per_seed)
    keys = sorted({k for r in rows for k in r})
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        m, ci = mean_ci([r.get(k, float("nan")) for r in rows])
        out[k] = {"mean": m, "ci95": ci, "n": len(rows)}
    return out


def paired_delta(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Mean and 95% CI of (a - b) over paired seeds."""
    return mean_ci([x - y for x, y in zip(a, b)])
