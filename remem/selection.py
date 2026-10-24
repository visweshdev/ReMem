"""Context selection: rank-order packing vs cost-aware selection under a token budget."""

from __future__ import annotations

from remem.config import SelectionConfig
from remem.types import ScoredCandidate


def rank_pack(ranked: list[ScoredCandidate], cfg: SelectionConfig) -> list[ScoredCandidate]:
    """Take candidates in rank order up to top_k; skip any that does not fit the remaining budget.

    This is what a fixed-k retriever does once a token budget is imposed from outside.
    """
    out: list[ScoredCandidate] = []
    used = 0
    for sc in ranked:
        if len(out) >= cfg.top_k:
            break
        if cfg.budget_tokens is not None and used + sc.tokens > cfg.budget_tokens:
            continue
        out.append(sc)
        used += sc.tokens
    return out


def cost_aware(ranked: list[ScoredCandidate], cfg: SelectionConfig) -> list[ScoredCandidate]:
    """Maximise expected utility minus token cost, optionally under a hard token budget.

    value_i = p_i - lambda * tokens_i / 100, where p_i is the model's P(contributes).
    * candidates with p_i < p_floor are dropped (not worth any context) but min_items are kept;
    * with a budget: exact 0/1 knapsack over (count <= top_k, tokens <= budget) maximising sum value_i;
    * without a budget: top by value, up to top_k.
    Expected-utility additivity is an approximation (a task needing two memories only succeeds
    with both); documented in the README.
    """
    vals = {sc.cand.fact_id: sc.prob - cfg.cost_lambda * sc.tokens / 100.0 for sc in ranked}
    pool = [sc for sc in ranked if sc.prob >= cfg.p_floor and vals[sc.cand.fact_id] > 0]
    if len(pool) < cfg.min_items:
        have = {s.cand.fact_id for s in pool}
        for sc in ranked:
            if sc.cand.fact_id not in have:
                pool.append(sc)
                have.add(sc.cand.fact_id)
            if len(pool) >= cfg.min_items:
                break
    pool = pool[: max(cfg.top_k * 4, cfg.top_k)]
    if cfg.budget_tokens is None:
        chosen = pool[: cfg.top_k]
    else:
        chosen = _knapsack(pool, vals, cfg.top_k, cfg.budget_tokens)
    order = {sc.cand.fact_id: i for i, sc in enumerate(ranked)}
    return sorted(chosen, key=lambda s: order[s.cand.fact_id])


def _knapsack(pool: list[ScoredCandidate], vals: dict[str, float], k: int, budget: int) -> list[ScoredCandidate]:
    n = len(pool)
    if n == 0:
        return []
    if budget > 1500:  # DP table would be large; fall back to greedy by density
        out, used = [], 0
        for sc in sorted(pool, key=lambda s: -vals[s.cand.fact_id] / max(1, s.tokens)):
            if len(out) < k and used + sc.tokens <= budget:
                out.append(sc)
                used += sc.tokens
        return out
    NEG = float("-inf")
    # dp[c][t] = best value using exactly c items and exactly t tokens
    dp = [[NEG] * (budget + 1) for _ in range(k + 1)]
    choice: list[list[list[int]]] = [[[] for _ in range(budget + 1)] for _ in range(k + 1)]
    dp[0][0] = 0.0
    for idx, sc in enumerate(pool):
        w, v = sc.tokens, vals[sc.cand.fact_id]
        if w > budget:
            continue
        for c in range(k - 1, -1, -1):
            row, nxt = dp[c], dp[c + 1]
            for t in range(budget - w, -1, -1):
                if row[t] == NEG:
                    continue
                cand_v = row[t] + v
                if cand_v > nxt[t + w]:
                    nxt[t + w] = cand_v
                    choice[c + 1][t + w] = choice[c][t] + [idx]
    best, best_idx = NEG, []
    for c in range(1, k + 1):
        for t in range(budget + 1):
            if dp[c][t] > best:
                best, best_idx = dp[c][t], choice[c][t]
    return [pool[i] for i in best_idx]


def select(ranked: list[ScoredCandidate], cfg: SelectionConfig) -> list[ScoredCandidate]:
    if cfg.mode == "rank_pack":
        return rank_pack(ranked, cfg)
    if cfg.mode == "cost_aware":
        return cost_aware(ranked, cfg)
    raise ValueError(f"unknown selection mode {cfg.mode!r}")
