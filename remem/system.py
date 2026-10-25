"""ReMemSystem: the outcome-aware layer that sits on top of any candidate backend.

    ingest(fact)                -> register in ledger, run contradiction detection
    retrieve(query, pool, ...)  -> validity filter -> utility ranking -> selection -> Context
    observe(ctx, trace, outcome)-> attribute credit -> update ledger -> retention transitions

Baselines are the same class with features switched off (`ranker: sim`, `outcome_feedback:
false`, ...), which guarantees identical plumbing, candidate pools and token accounting across
all compared systems.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable

from remem.attribution import Attribution, Attributor
from remem.config import SystemConfig
from remem.contradiction import ContradictionDetector
from remem.features import ALL_FEATURES, FeatureBuilder
from remem.ledger import MemoryLedger
from remem.offline import ConceptHashEmbedder, count_tokens
from remem.retention import RetentionPolicy
from remem.selection import select
from remem.types import (
    AgentTrace, Candidate, Context, MemState, Outcome, ScoredCandidate,
)
from remem.utility import HeuristicUtility, LearnedUtility, build_utility


@dataclass
class IngestedFact:
    fact_id: str
    text: str
    time: float
    entities: tuple[str, ...] = ()
    attribute_keys: tuple[str, ...] = ()
    confidence: float = 1.0
    semantic_role: str = "state"


@dataclass
class TraceRow:
    """One (memory, task) training example, as observed by the system (no ground truth)."""

    task_id: int
    fact_id: str
    features: list[float]
    credit: float
    label: int
    system: str


class ReMemSystem:
    def __init__(self, cfg: SystemConfig, embedder: ConceptHashEmbedder, *, seed: int = 0, collect: bool = False) -> None:
        self.cfg = cfg
        self.embedder = embedder
        self.ledger = MemoryLedger()
        self.rng = random.Random(seed)
        self.ts_rng = random.Random(f"thompson:{seed}")   # separate stream: exploration never perturbs other draws
        self.collect = collect
        self.trace_rows: list[TraceRow] = []
        self.utility = self._build_ranker()
        self.features = FeatureBuilder(cfg.utility, self.ledger)
        self.attributor = Attributor(cfg.attribution)
        self.detector = ContradictionDetector(cfg.contradiction, self.ledger, self._text_sim)
        self.retention = RetentionPolicy(cfg.retention, self.ledger, cfg.utility.prior_a, cfg.utility.prior_b)
        self._emb_cache: dict[str, list[float]] = {}
        self.rmm = None
        self._rmm_pending: dict[int, tuple] = {}
        if cfg.ranker == "rmm":
            from remem.baselines.rmm import RMMReranker
            self.rmm = RMMReranker(embedder.dim, lr=cfg.utility.rmm_lr, tau=cfg.utility.rmm_tau, seed=seed)

    # ── construction helpers ───────────────────────────────────────────────────
    def _build_ranker(self):
        c = self.cfg
        if c.ranker in ("sim", "rmm"):
            return None
        if c.ranker == "static":
            return HeuristicUtility(("static",), 0.0, 0.0)
        # Thompson exploration is applied in `retrieve`; the utility object then carries no UCB bonus
        explore_c = 0.0 if c.utility.explore_mode == "thompson" else c.utility.explore_c
        if c.ranker == "heuristic":
            return build_utility(_with(c.utility, kind="heuristic", explore_c=explore_c))
        if c.ranker == "learned":
            return build_utility(_with(c.utility, kind="learned", explore_c=explore_c))
        raise ValueError(f"unknown ranker {c.ranker!r}")

    def _text_sim(self, a: str, b: str) -> float:
        ea, eb = self._emb(a), self._emb(b)
        return sum(x * y for x, y in zip(ea, eb))

    def _emb(self, text: str) -> list[float]:
        v = self._emb_cache.get(text)
        if v is None:
            v = self.embedder.embed_one(text)
            self._emb_cache[text] = v
        return v

    # ── ingestion ──────────────────────────────────────────────────────────────
    def ingest(self, fact: IngestedFact) -> list:
        rec = self.ledger.register(
            fact_id=fact.fact_id, text=fact.text, time=fact.time,
            entities=fact.entities, attribute_keys=fact.attribute_keys, confidence=fact.confidence,
            semantic_role=fact.semantic_role,
        )
        return self.detector.on_ingest(rec)

    # ── retrieval ──────────────────────────────────────────────────────────────
    def retrieve(
        self,
        task_id: int,
        query: str,
        pool: list[Candidate],
        *,
        task_class: str,
        now: float,
        as_of: float | None = None,
        pool_time_ms: float = 0.0,
    ) -> Context:
        t0 = time.perf_counter()
        cfg = self.cfg
        step = self.ledger.tick()
        self.retention.frequency_prune(step)
        self.detector.decay(now)
        ref = now if as_of is None else as_of

        considered: list[tuple[Candidate, object, MemState]] = []
        for cand in pool:
            rec = self.ledger.get(cand.fact_id)
            if rec is None:  # candidate the system never saw ingested; register defensively
                rec = self.ledger.register(cand)
            rec.n_pooled += 1
            if cfg.contradiction.enabled and cfg.contradiction.as_of_validity:
                if not ContradictionDetector.valid_at(rec, ref):
                    continue
            if cfg.selection.min_sim > 0 and cand.sim < cfg.selection.min_sim:
                continue                                    # not about this query at all
            eff = self.retention.effective_state(rec, task_class)
            if eff == MemState.ARCHIVED:
                continue
            if cfg.retention.mode == "frequency_prune" and rec.state == MemState.ARCHIVED:
                continue
            considered.append((cand, rec, eff))

        max_sim = max((c.sim for c, _, _ in considered), default=0.0)
        scored: list[ScoredCandidate] = []
        for cand, rec, eff in considered:
            text, toks = self.retention.served_text(rec, eff)
            if self.utility is None:
                feats = []
                rank = prob = cand.sim
                comps = {"sim": cand.sim}
            else:
                cand_f = Candidate(**{**cand.__dict__, "confidence": rec.confidence})
                feats = self.features.build(cand_f, rec, pool_max_sim=max_sim, now=now,
                                            task_class=task_class, as_of=as_of)
                rank, prob, comps = self.utility.score(feats)
                if cfg.utility.explore_mode == "thompson" and cfg.utility.explore_c and cfg.outcome_feedback:
                    bonus = self._thompson_bonus(rec)
                    rank += bonus
                    prob = min(1.0, max(0.0, prob + bonus))
                    comps["thompson"] = bonus
            if eff in (MemState.LOW_VALUE, MemState.COMPRESSED):
                rank *= cfg.retention.low_value_penalty if rank > 0 else 1.0 / cfg.retention.low_value_penalty
                prob *= cfg.retention.low_value_penalty
                comps["state_penalty"] = -(1 - cfg.retention.low_value_penalty)
            rec.last_utility = prob
            scored.append(ScoredCandidate(
                cand=cand, utility=rank, prob=prob,
                components=comps, state=eff.value, text=text, tokens=toks, features=feats,
                reason=_reason(comps, cand, rec),
            ))
        # rank by the model's ordering score; ties by backend rank then id (deterministic)
        scored.sort(key=lambda s: (-s.utility, s.cand.base_rank, s.cand.fact_id))
        if self.rmm is not None and scored:
            import numpy as np
            q = np.asarray(self._emb(query))
            M = np.asarray([self._emb(sc.cand.text) for sc in scored])
            order = self.rmm.sample_order(self.rmm.scores(q, M))
            pos = {i: r for r, i in enumerate(order)}
            for i, sc in enumerate(scored):
                sc.utility = -float(pos[i])                  # sampled order becomes the ranking
            self._rmm_pending[task_id] = (q, M, [sc.cand.fact_id for sc in scored])
            scored.sort(key=lambda s: -s.utility)
        selected = select(scored, cfg.selection)
        selected = self._apply_guard(scored, selected, task_class)

        if self.collect and cfg.collect_exploration > 0:
            chosen = {s.cand.fact_id for s in selected}
            extras = [s for s in scored if s.cand.fact_id not in chosen]
            self.rng.shuffle(extras)
            selected = selected + extras[: cfg.collect_exploration]

        for sc in selected:
            rec = self.ledger.get(sc.cand.fact_id)
            rec.n_retrieved += 1
            rec.last_retrieved_step = step
            rec.last_reason = sc.reason
            self.ledger.log(rec, "RETRIEVED", f"task={task_id} class={task_class} {sc.reason}", sc.utility)

        used = sum(s.tokens for s in selected)
        latency_ms = (time.perf_counter() - t0) * 1000.0 + pool_time_ms
        return Context(
            task_id=task_id, query=query, task_class=task_class, as_of=as_of,
            selected=selected, scored=scored, budget_tokens=cfg.selection.budget_tokens,
            tokens_used=used, latency_ms=latency_ms, pool_size=len(pool),
        )

    # ── exploration and safeguards ─────────────────────────────────────────────
    def _thompson_bonus(self, rec) -> float:
        """c * (theta - posterior mean), theta ~ Beta over the memory's contribution rate.

        Zero-mean noise whose scale shrinks as evidence accumulates, so a memory with a thin record is
        occasionally promoted into context (and so gets feedback) while a well-measured one is not.
        """
        u = self.cfg.utility
        credit = min(rec.n_credited, float(rec.n_success_ctx))
        a, b = u.prior_a + credit, u.prior_b + max(0.0, rec.n_success_ctx - credit)
        return u.explore_c * (self.ts_rng.betavariate(a, b) - a / (a + b))

    def _apply_guard(self, scored: list[ScoredCandidate], selected: list[ScoredCandidate], task_class: str) -> list[ScoredCandidate]:
        """Never silently drop the top-m most similar candidates unless the ledger has evidence they are useless.

        Addresses the measured failure mode in which the utility layer pushes a correct, highly similar memory
        out of context. Evidence of uselessness = shown >= guard_min_evidence times for this task class with a
        class contribution rate below guard_rate. A guarded candidate replaces the lowest-probability unguarded
        selected item (or is added if there is room), respecting the token budget.
        """
        sel = self.cfg.selection
        if sel.guard_top <= 0 or not scored:
            return selected
        u = self.cfg.utility
        best_sim = max(s.cand.sim for s in scored)
        if best_sim <= 0:
            return selected
        guarded = [s for s in sorted(scored, key=lambda s: (-s.cand.sim, s.cand.base_rank, s.cand.fact_id))[: sel.guard_top]
                   if s.cand.sim >= sel.guard_min_rel_sim * best_sim]      # never guard junk that merely ranks top-m
        guarded_ids = {g.cand.fact_id for g in guarded}
        out = list(selected)
        have = {s.cand.fact_id for s in out}

        def fits(lst):
            return sel.budget_tokens is None or sum(x.tokens for x in lst) <= sel.budget_tokens

        for g in guarded:
            fid = g.cand.fact_id
            if fid in have:
                continue
            rec = self.ledger.get(fid)
            cs = rec.classes.get(task_class)
            if cs is not None and cs.n_ctx >= sel.guard_min_evidence and                     self.ledger.class_rate(rec, task_class, u.prior_a, u.prior_b, u.class_shrink) < sel.guard_rate:
                continue                                    # evidence says it is useless for this task class
            if len(out) < sel.top_k and fits(out + [g]):
                out.append(g)
            else:
                victims = [x for x in out if x.cand.fact_id not in guarded_ids]
                if not victims:
                    continue
                v = min(victims, key=lambda x: x.prob)
                trial = [x for x in out if x is not v] + [g]
                if not fits(trial):
                    continue
                out = trial
            have.add(fid)
            self.ledger.log(rec, "GUARDED", f"served by similarity guard (class={task_class})")
        order = {s.cand.fact_id: i for i, s in enumerate(scored)}
        return sorted(out, key=lambda s: order[s.cand.fact_id])

    # ── feedback ───────────────────────────────────────────────────────────────
    def observe(
        self,
        ctx: Context,
        trace: AgentTrace,
        outcome: Outcome,
        replay: Callable[[list[ScoredCandidate]], bool] | None = None,
    ) -> dict[str, Attribution]:
        cfg = self.cfg
        if self.rmm is not None and ctx.task_id in self._rmm_pending:
            q, M, ids = self._rmm_pending.pop(ctx.task_id)
            idx = {fid: i for i, fid in enumerate(ids)}
            sel = [idx[s.cand.fact_id] for s in ctx.selected if s.cand.fact_id in idx]
            cited = set(trace.cited_ids)
            self.rmm.update(q, M, sel, [1.0 if ids[i] in cited else -1.0 for i in sel])   # citation reward only
        if not cfg.outcome_feedback or not ctx.selected:
            return {}
        led = self.ledger
        success = True if cfg.attribution.ignore_outcome else outcome.success   # ablation: usage-only feedback
        atts = self.attributor.attribute(ctx.selected, trace, success, replay)
        thr = cfg.attribution.credit_threshold
        for sc in ctx.selected:
            rec = led.get(sc.cand.fact_id)
            att = atts[sc.cand.fact_id]
            if success:
                rec.n_success_ctx += 1
                rec.n_credited += att.credit
                cs = led.class_stats(rec, ctx.task_class)
                cs.n_ctx += 1
                cs.credit += att.credit
                if att.credit >= thr:
                    rec.last_credit_step = led.step
                    led.log(rec, "USED", f"task={ctx.task_id} cited={att.cited} used={att.used} necessary={att.necessary}")
                    led.log(rec, "CREDITED", f"credit={att.credit:.2f} -> contribution rate {led.contrib_rate(rec, cfg.utility.prior_a, cfg.utility.prior_b):.3f}", sc.utility)
                    self.retention.on_credited(rec)
                    self.detector.on_credited(rec)
                else:
                    rec.n_wasted += 1
                    led.log(rec, "NO_CONTRIBUTION", f"task={ctx.task_id} credit={att.credit:.2f}", sc.utility)
                if self.collect and sc.features:
                    self.trace_rows.append(TraceRow(ctx.task_id, rec.fact_id, sc.features, att.credit,
                                                    int(att.credit >= thr), cfg.name))
            else:
                rec.n_fail_ctx += 1
                if att.blame > 0:
                    rec.n_blamed += att.blame
                    led.log(rec, "BLAMED", f"task={ctx.task_id} failed action used this memory")
                    self.detector.on_failure_blame(rec)
            self.retention.update(rec)
        return atts


def _with(ucfg, **kw):
    from dataclasses import replace
    return replace(ucfg, **kw)


def _reason(comps: dict[str, float], cand: Candidate, rec) -> str:
    top = sorted(comps.items(), key=lambda kv: -abs(kv[1]))[:3]
    parts = ", ".join(f"{k}={v:+.2f}" for k, v in top)
    return f"sim={cand.sim:.2f} rank#{cand.base_rank} [{parts}]"
