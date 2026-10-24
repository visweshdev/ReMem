"""Feature construction for the memory utility models.

Features are grouped so ablations can switch whole information sources on and off:

  static         what the retrieval backend already knows at query time (similarity, rank, age, size)
  history        retrieval frequency / recency   (the "frequency is not usefulness" strawman)
  outcome        what happened when this memory was shown before (Beta posteriors, waste, blame)
  contradiction  staleness / contradiction flags
"""

from __future__ import annotations

import math

from remem.config import UtilityConfig
from remem.ledger import MemoryLedger
from remem.types import Candidate, MemoryRecord

STATIC = ("sim", "sim_gap", "rank_inv", "recency", "confidence", "log_tokens")
HISTORY = ("log_ctx", "retr_recency")
OUTCOME = ("contrib_rate", "class_rate", "waste_rate", "blame_rate", "post_std")
CONTRADICTION = ("stale", "log_contra")

GROUPS: dict[str, tuple[str, ...]] = {
    "static": STATIC,
    "history": HISTORY,
    "outcome": OUTCOME,
    "contradiction": CONTRADICTION,
}
ALL_FEATURES: tuple[str, ...] = STATIC + HISTORY + OUTCOME + CONTRADICTION
FEATURE_INDEX = {n: i for i, n in enumerate(ALL_FEATURES)}


def names_for_groups(groups: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    for g in ("static", "history", "outcome", "contradiction"):
        if g in groups:
            out.extend(GROUPS[g])
    return tuple(out)


class FeatureBuilder:
    def __init__(self, cfg: UtilityConfig, ledger: MemoryLedger) -> None:
        self.cfg = cfg
        self.ledger = ledger

    def build(
        self,
        cand: Candidate,
        rec: MemoryRecord,
        *,
        pool_max_sim: float,
        now: float,
        task_class: str,
        as_of: float | None,
    ) -> list[float]:
        c = self.cfg
        led = self.ledger
        age_days = max(0.0, (now - cand.time) / 86400.0)
        recency = math.exp(-age_days / max(1e-6, c.recency_tau_days))
        retr_recency = 0.0
        if rec.last_retrieved_step is not None:
            retr_recency = math.exp(-(led.step - rec.last_retrieved_step) / 50.0)
        exposures = rec.n_success_ctx
        stale = 0.0
        if rec.superseded_by is not None:
            ref = now if as_of is None else as_of
            if rec.valid_to is not None and ref >= rec.valid_to:
                stale = 1.0
        vec = {
            "sim": cand.sim,
            "sim_gap": cand.sim - pool_max_sim,
            "rank_inv": 1.0 / (1.0 + cand.base_rank),
            "recency": recency,
            "confidence": rec.confidence,
            "log_tokens": math.log1p(rec.tokens) / 5.0,
            "log_ctx": math.log1p(rec.n_retrieved) / 3.0,
            "retr_recency": retr_recency,
            "contrib_rate": led.contrib_rate(rec, c.prior_a, c.prior_b),
            "class_rate": led.class_rate(rec, task_class, c.prior_a, c.prior_b, c.class_shrink),
            "waste_rate": rec.n_wasted / (exposures + 2.0),
            "blame_rate": rec.n_blamed / (rec.n_retrieved + 2.0),
            "post_std": led.contrib_std(rec, c.prior_a, c.prior_b),
            "stale": stale,
            "log_contra": math.log1p(rec.contradictions),
        }
        return [vec[n] for n in ALL_FEATURES]
