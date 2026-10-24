import itertools

from remem.config import SelectionConfig
from remem.selection import cost_aware, rank_pack, select
from tests.conftest import mk_scored


def _items():
    return [
        mk_scored("big", 0.90, tokens=44),
        mk_scored("s1", 0.30, tokens=11),
        mk_scored("s2", 0.28, tokens=11),
        mk_scored("s3", 0.27, tokens=11),
        mk_scored("s4", 0.26, tokens=11),
    ]


def test_rank_pack_respects_k_and_budget():
    ranked = _items()
    out = rank_pack(ranked, SelectionConfig(mode="rank_pack", top_k=3, budget_tokens=None))
    assert [s.cand.fact_id for s in out] == ["big", "s1", "s2"]
    out = rank_pack(ranked, SelectionConfig(mode="rank_pack", top_k=5, budget_tokens=50))
    assert sum(s.tokens for s in out) <= 50 and out[0].cand.fact_id == "big"


def test_cost_aware_knapsack_is_optimal_vs_bruteforce():
    ranked = _items()
    cfg = SelectionConfig(mode="cost_aware", top_k=4, budget_tokens=50, p_floor=0.0)
    got = {s.cand.fact_id for s in cost_aware(ranked, cfg)}
    best, best_v = None, -1
    for r in range(1, 5):
        for combo in itertools.combinations(ranked, r):
            if sum(c.tokens for c in combo) <= 50:
                v = sum(c.prob for c in combo)
                if v > best_v:
                    best, best_v = {c.cand.fact_id for c in combo}, v
    assert got == best


def test_cost_aware_drops_low_probability_candidates_but_keeps_min_items():
    ranked = _items()
    out = cost_aware(ranked, SelectionConfig(mode="cost_aware", top_k=5, p_floor=0.5))
    assert [s.cand.fact_id for s in out] == ["big"]
    low = [mk_scored("x", 0.05), mk_scored("y", 0.04)]
    assert len(cost_aware(low, SelectionConfig(mode="cost_aware", top_k=5, p_floor=0.5, min_items=1))) == 1


def test_cost_aware_never_exceeds_budget():
    out = select(_items(), SelectionConfig(mode="cost_aware", top_k=5, budget_tokens=30, p_floor=0.0))
    assert sum(s.tokens for s in out) <= 30
