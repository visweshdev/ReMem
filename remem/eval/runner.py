"""Stream runner: replays a world through one memory system and records everything measurable.

Information barrier: the memory system receives (query, task_class, as_of, now, candidate pool)
and, after acting, the agent trace and a boolean outcome. Ground truth (gold ids, true answer,
fact kinds) is used only here, to fill `TaskRecord`s for the metrics.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from remem.backends.base import Backend, StoredFact
from remem.config import SystemConfig
from remem.eval.agent import AgentConfig, SimAgent
from remem.eval.world import Task, World
from remem.offline import ConceptHashEmbedder
from remem.system import IngestedFact, ReMemSystem
from remem.types import Candidate, MemState, Outcome

Pools = dict[int, tuple[list[Candidate], float]]


# ── candidate pools (backend replay) ───────────────────────────────────────────

def stored_from_world(f, session: str) -> StoredFact:
    return StoredFact(
        fact_id=f.fact_id, text=f.text, time=f.time, entities=f.entities, attribute_keys=f.attribute_keys,
        origin="user", session_id=session, semantic_role=f.role,
    )


def build_pools(world: World, backend: Backend, k: int) -> Pools:
    """Replay the world through a backend once; cache the candidate pool of every task.

    Every system compared on the same backend consumes the identical pool, so differences in
    results come only from ranking, selection and lifecycle policy. The measured retrieval time
    is stored with the pool and added to each system's own re-ranking time.
    """
    pools: Pools = {}
    for ev in world.events:
        if ev.kind == "ingest":
            backend.ingest([stored_from_world(f, ev.session) for f in ev.facts])
        else:
            assert ev.task is not None
            cands, ms = backend.candidates(ev.task.query, k=k)
            pools[ev.task.task_id] = (cands, ms)
    return pools


# ── records ────────────────────────────────────────────────────────────────────

@dataclass
class TaskRecord:
    task_id: int
    time: float
    kind: str
    task_class: str
    query: str
    success: bool
    gold_ids: list[str]
    selected_ids: list[str]
    selected_kinds: list[str]
    selected_states: list[str]
    tokens: int
    budget: int | None
    latency_ms: float
    n_selected: int
    gold_in_pool: int
    gold_in_context: int
    n_gold: int
    gold_ranks_pool: list[int]           # base rank of each gold fact in the pool (-1 if absent)
    gold_ranks_ctx: list[int]            # position in the served context (-1 if absent)
    gold_states: list[str]               # ledger state of each gold fact at retrieval time
    gold_filtered: int                   # gold facts removed from consideration by validity / archive filters
    n_irrelevant: int                    # selected items that are not gold
    n_stale_selected: int                # selected items that assert an outdated/other value for a required slot
    n_trap_selected: int
    n_hub_selected: int
    used_ids: list[str]
    distracted: bool
    guessed_slots: list[str]
    attributed: dict[str, float]         # fact_id -> credit assigned (empty if no feedback)
    scored_probs: list[tuple[float, float, int]]   # (probability, sim, is_gold) per considered candidate
    extra_agent_calls: int = 0


@dataclass
class RunResult:
    system: str
    family: str
    seed: int
    records: list[TaskRecord]
    snapshots: list[dict[str, Any]]
    ledger: Any
    trace_rows: list
    conflicts: list
    cfg: SystemConfig
    n_facts: int
    false_invalidations: int
    missed_invalidations: int


def run_stream(
    world: World,
    pools: Pools,
    cfg: SystemConfig,
    embedder: ConceptHashEmbedder,
    agent_cfg: AgentConfig | None = None,
    *,
    seed: int = 0,
    collect: bool = False,
    feedback_noise: float = 0.0,
    snapshot_every: int = 25,
    on_task=None,
    agent_factory=None,
) -> RunResult:
    agent_cfg = agent_cfg or AgentConfig()
    system = ReMemSystem(cfg, embedder, seed=seed, collect=collect)
    # any object with act(task, items) -> AgentResult works (SimAgent by default, LLMAgent for real models)
    agent = agent_factory(world, seed) if agent_factory is not None else SimAgent(world, agent_cfg, seed=seed)
    noise_rng = random.Random(f"noise:{seed}:{world.family}")
    kind_of = {f.fact_id: f for f in world.facts}
    records: list[TaskRecord] = []
    snapshots: list[dict[str, Any]] = []

    for ev in world.events:
        if ev.kind == "ingest":
            for f in ev.facts:
                system.ingest(IngestedFact(f.fact_id, f.text, f.time, f.entities, f.attribute_keys, 1.0, f.role))
            continue
        task = ev.task
        assert task is not None
        pool, pool_ms = pools[task.task_id]
        ctx = system.retrieve(task.task_id, task.query, pool, task_class=task.task_class, now=task.time,
                              as_of=task.as_of, pool_time_ms=pool_ms)
        items = [(s.cand.fact_id, s.text, s.cand.time) for s in ctx.selected]
        res = agent.act(task, items)
        success_obs = res.success
        if feedback_noise and noise_rng.random() < feedback_noise:
            success_obs = not success_obs

        def replay(subset, _task=task):
            return agent.act(_task, [(s.cand.fact_id, s.text, s.cand.time) for s in subset]).success

        calls_before = system.attributor.extra_agent_calls
        atts = system.observe(ctx, res.trace, Outcome(success_obs), replay if cfg.attribution.method == "signals_loo" else None)
        extra_calls = system.attributor.extra_agent_calls - calls_before

        records.append(_record(world, task, ctx, res, atts, pool, kind_of, system, extra_calls))
        if on_task is not None:
            on_task(task, ctx, res, system)   # read-only hook for demos / debugging
        if len(records) % snapshot_every == 0:
            snapshots.append(_snapshot(system, len(records)))
    snapshots.append(_snapshot(system, len(records)))

    false_inv, missed_inv = _invalidation_audit(world, system)
    return RunResult(cfg.name, world.family, seed, records, snapshots, system.ledger, system.trace_rows,
                     system.detector.conflicts, cfg, len(world.facts), false_inv, missed_inv)


def _snapshot(system: ReMemSystem, n_tasks: int) -> dict[str, Any]:
    led = system.ledger
    return {
        "n_tasks": n_tasks, "n_memories": len(led.records), "states": led.state_counts(),
        "served_tokens": led.served_tokens(), "total_tokens": led.total_tokens(),
    }


def _invalidation_audit(world: World, system: ReMemSystem) -> tuple[int, int]:
    """Ground-truth audit of supersession: false = invalidated but still the current value at the end;
    missed = an outdated exclusive-attribute fact that is still not invalidated."""
    if not system.cfg.contradiction.enabled:
        return 0, 0
    final_ids = {}
    for key, hist in world.history.items():
        final_ids[key] = hist[-1][2]
    outdated = {fid for key, hist in world.history.items() for _, _, fid in hist[:-1]}
    current = set(final_ids.values())
    # facts of non-exclusive attribute (depends_on) are always current
    depends = {f.fact_id for f in world.facts if f.kind == "depends"}
    false_inv = missed = 0
    for rec in system.ledger.records.values():
        inv = rec.state == MemState.INVALIDATED
        if inv and (rec.fact_id in current or rec.fact_id in depends):
            false_inv += 1
        if (not inv) and rec.fact_id in outdated:
            missed += 1
    return false_inv, missed


def _record(world, task: Task, ctx, res, atts, pool, kind_of, system, extra_calls) -> TaskRecord:
    gold = world.gold_ids(task)
    gold_set = set(gold)
    sel_ids = [s.cand.fact_id for s in ctx.selected]
    pool_rank = {c.fact_id: c.base_rank for c in pool}
    scored_ids = {s.cand.fact_id for s in ctx.scored}
    slot_keys = set(task.slots)

    stale = 0
    for fid in sel_ids:
        f = kind_of[fid]
        if fid not in gold_set and (f.subject, f.attr) in slot_keys and f.value is not None:
            stale += 1
    gold_states = []
    for g in gold:
        rec = system.ledger.get(g)
        gold_states.append(rec.state.value if rec else "MISSING")
    return TaskRecord(
        task_id=task.task_id, time=task.time, kind=task.kind, task_class=task.task_class, query=task.query,
        success=res.success, gold_ids=gold, selected_ids=sel_ids,
        selected_kinds=[kind_of[i].kind for i in sel_ids], selected_states=[s.state for s in ctx.selected],
        tokens=ctx.tokens_used, budget=ctx.budget_tokens, latency_ms=ctx.latency_ms, n_selected=len(sel_ids),
        gold_in_pool=sum(1 for g in gold if g in pool_rank), gold_in_context=sum(1 for g in gold if g in sel_ids),
        n_gold=len(gold), gold_ranks_pool=[pool_rank.get(g, -1) for g in gold],
        gold_ranks_ctx=[sel_ids.index(g) if g in sel_ids else -1 for g in gold],
        gold_states=gold_states,
        gold_filtered=sum(1 for g in gold if g in pool_rank and g not in scored_ids),
        n_irrelevant=sum(1 for i in sel_ids if i not in gold_set),
        n_stale_selected=stale,
        n_trap_selected=sum(1 for i in sel_ids if kind_of[i].kind == "trap"),
        n_hub_selected=sum(1 for i in sel_ids if kind_of[i].kind == "hub"),
        used_ids=res.used_ids, distracted=res.distracted, guessed_slots=res.guessed_slots,
        attributed={k: v.credit for k, v in atts.items()},
        scored_probs=[(s.prob, s.cand.sim, int(s.cand.fact_id in gold_set)) for s in ctx.scored],
        extra_agent_calls=extra_calls,
    )


def apply_world_selection(cfg: SystemConfig, world: World) -> SystemConfig:
    """Families that impose a token budget / k override the system's selection limits."""
    from dataclasses import replace
    p = world.params
    sel = replace(cfg.selection, budget_tokens=p.get("budget_tokens"), top_k=p.get("top_k", cfg.selection.top_k))
    return replace(cfg, selection=sel)
