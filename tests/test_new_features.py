"""Tests for ReMem's additions: similarity guard, Thompson exploration, schema learning, ambiguous-only LOO."""

from dataclasses import replace

from remem.attribution import Attributor
from remem.config import AttributionConfig, ContradictionConfig, SelectionConfig, SystemConfig, UtilityConfig
from remem.contradiction import ContradictionDetector
from remem.ledger import MemoryLedger
from remem.system import IngestedFact, ReMemSystem
from remem.types import AgentTrace, MemState
from tests.conftest import mk_cand, mk_scored

DAY = 86400.0


# ── similarity guard ───────────────────────────────────────────────────────────

def _guard_system(embedder, a_class_evidence):
    cfg = SystemConfig(
        name="g", ranker="heuristic", outcome_feedback=True,
        utility=UtilityConfig(kind="heuristic", feature_groups=("static", "outcome")),
        selection=SelectionConfig(mode="rank_pack", top_k=1, guard_top=1, guard_min_evidence=3, guard_rate=0.2),
    )
    sysm = ReMemSystem(cfg, embedder)
    for fid in ("A", "B"):
        sysm.ingest(IngestedFact(fid, f"text {fid}", 1.7e9))
    a, b = sysm.ledger.get("A"), sysm.ledger.get("B")
    a.n_success_ctx = a_class_evidence
    a.classes["q"] = sysm.ledger.class_stats(a, "q")
    a.classes["q"].n_ctx = a_class_evidence                       # shown that often, never credited
    b.n_success_ctx, b.n_credited = 10, 10.0
    b.classes["q"] = sysm.ledger.class_stats(b, "q")
    b.classes["q"].n_ctx, b.classes["q"].credit = 10, 10.0
    pool = [mk_cand("A", "text A", sim=0.9, rank=0), mk_cand("B", "text B", sim=0.6, rank=1)]
    return sysm, pool


def test_guard_serves_top_similarity_memory_when_evidence_is_thin(embedder):
    sysm, pool = _guard_system(embedder, a_class_evidence=2)
    ctx = sysm.retrieve(1, "q", pool, task_class="q", now=1.8e9)
    assert [s.cand.fact_id for s in ctx.selected] == ["A"]
    assert any(e.kind == "GUARDED" for e in sysm.ledger.get("A").events)
    # without the guard the utility layer would have served B
    sysm.cfg = replace(sysm.cfg, selection=replace(sysm.cfg.selection, guard_top=0))
    assert [s.cand.fact_id for s in sysm.retrieve(2, "q", pool, task_class="q", now=1.8e9).selected] == ["B"]


def test_guard_yields_to_evidence_of_uselessness(embedder):
    sysm, pool = _guard_system(embedder, a_class_evidence=5)
    ctx = sysm.retrieve(1, "q", pool, task_class="q", now=1.8e9)
    assert [s.cand.fact_id for s in ctx.selected] == ["B"]


def test_guard_respects_token_budget(embedder):
    sysm, pool = _guard_system(embedder, a_class_evidence=2)
    sysm.cfg = replace(sysm.cfg, selection=replace(sysm.cfg.selection, budget_tokens=3))
    sysm.ledger.get("A").original_tokens = 50                      # the guarded memory cannot fit the budget
    ctx = sysm.retrieve(1, "q", pool, task_class="q", now=1.8e9)
    assert ctx.tokens_used <= 3 and "A" not in [s.cand.fact_id for s in ctx.selected]


# ── Thompson exploration ───────────────────────────────────────────────────────

def test_thompson_bonus_is_deterministic_zero_mean_and_shrinks_with_evidence(embedder):
    cfg = SystemConfig(name="t", ranker="heuristic", outcome_feedback=True,
                       utility=UtilityConfig(kind="heuristic", explore_c=1.0, explore_mode="thompson"))
    s1, s2 = ReMemSystem(cfg, embedder, seed=3), ReMemSystem(cfg, embedder, seed=3)
    for s in (s1, s2):
        s.ingest(IngestedFact("A", "x", 1.0))
    r1, r2 = s1.ledger.get("A"), s2.ledger.get("A")
    assert [s1._thompson_bonus(r1) for _ in range(5)] == [s2._thompson_bonus(r2) for _ in range(5)]
    thin = [s1._thompson_bonus(r1) for _ in range(4000)]
    r1.n_success_ctx, r1.n_credited = 200, 100.0
    thick = [s1._thompson_bonus(r1) for _ in range(4000)]
    mean = lambda v: sum(v) / len(v)                                      # noqa: E731
    std = lambda v: (sum((x - mean(v)) ** 2 for x in v) / len(v)) ** 0.5  # noqa: E731
    assert abs(mean(thin)) < 0.02
    assert std(thick) < 0.4 * std(thin)


# ── schema learning ────────────────────────────────────────────────────────────

def _schema_detector(embedder, **kw):
    led = MemoryLedger()
    cfg = ContradictionConfig(enabled=True, exclusive_attributes=(), learn_schema=True, **kw)
    sim = lambda a, b: sum(x * y for x, y in zip(embedder.embed_one(a), embedder.embed_one(b)))  # noqa: E731
    return led, ContradictionDetector(cfg, led, sim)


def _pair(led, det, ent, attr, v_old, v_new):
    ids = []
    for t, v in ((1 * DAY, v_old), (20 * DAY, v_new)):
        fid = f"{ent}_{attr}_{v}"
        rec = led.register(fact_id=fid, text=f"{ent} favourite {attr} is {v}.", time=t, entities=(ent,), attribute_keys=(attr,))
        det.on_ingest(rec)
        ids.append(fid)
    return ids


def test_schema_is_learned_from_outcome_confirmations_and_applied_retroactively(embedder):
    led, det = _schema_detector(embedder)
    pairs = [_pair(led, det, e, "city", "paris", "berlin") for e in ("ann", "bob", "cy")]
    assert all(led.get(o).state == MemState.ACTIVE for o, _ in pairs)       # not known single-valued yet
    assert not det.is_exclusive("city")
    for old, _ in pairs[:2]:                                                  # two failed actions blamed the old value
        det.on_failure_blame(led.get(old))
    assert det.is_exclusive("city") and det.schema_log[-1][1] == "city"
    assert led.get(pairs[2][0]).state == MemState.INVALIDATED                 # pending pair resolved retroactively
    ent, attr = "dee", "city"
    _, new = _pair(led, det, ent, attr, "rome", "oslo")                       # future updates are caught at ingest
    assert led.get("dee_city_rome").state == MemState.INVALIDATED


def test_counter_evidence_prevents_learning_multivalued_attribute(embedder):
    led, det = _schema_detector(embedder)
    pairs = [_pair(led, det, e, "hobby", "chess", "golf") for e in ("ann", "bob", "cy", "dee")]
    for old, _ in pairs:                                                      # old value still credited after the new arrived
        det.on_credited(led.get(old))
    det.on_failure_blame(led.get(pairs[0][0]))                                # one stray blame
    assert not det.is_exclusive("hobby")
    assert all(led.get(o).state == MemState.ACTIVE for o, _ in pairs[1:])


def test_schema_learning_is_off_by_default(embedder):
    led = MemoryLedger()
    sim = lambda a, b: 0.9                                                    # noqa: E731
    det = ContradictionDetector(ContradictionConfig(enabled=True), led, sim)
    old, _ = _pair(led, det, "ann", "city", "paris", "berlin")
    det.on_failure_blame(led.get(old))
    assert det.schema_evidence == {} and not det.learned_exclusive


# ── cheaper leave-one-out ──────────────────────────────────────────────────────

def test_ambiguous_only_loo_replays_just_the_uncertain_memories():
    ctx = [mk_scored("both", 0.9, text="payments eu-west-1"),                # cited + used  -> 1.0, no replay
           mk_scored("used", 0.8, text="billing eu-west-1"),                 # used only     -> 0.5, replay
           mk_scored("none", 0.7, text="weekly notes")]                      # neither       -> 0.0, no replay
    tr = AgentTrace({"k": "eu-west-1"}, ["both"], {"k": "eu-west-1"})
    calls = []

    def replay(subset):
        calls.append([s.cand.fact_id for s in subset])
        return any(s.cand.fact_id == "both" for s in subset)                  # task still succeeds without 'used'

    cheap = Attributor(AttributionConfig(method="signals_loo", loo_policy="ambiguous"))
    atts = cheap.attribute(ctx, tr, True, replay)
    assert cheap.extra_agent_calls == 1 and len(calls) == 1
    assert atts["both"].credit == 1.0 and atts["none"].credit == 0.0
    assert atts["used"].necessary is False and atts["used"].credit < 0.5       # spurious value-overlap discounted
    full = Attributor(AttributionConfig(method="signals_loo", loo_policy="all"))
    full.attribute(ctx, tr, True, replay)
    assert full.extra_agent_calls == 3


def test_guard_never_serves_unrelated_memories_in_a_tiny_store(embedder):
    from remem import ReMem
    m = ReMem(None, embedder=embedder)
    m.add("The payments service is hosted in region eu-west-1.", entity="payments", attribute="region")
    m.add("Team lunch moved to Thursday.")
    m.add("The billing service is hosted in region us-east-1.", entity="billing", attribute="region")
    texts = [h.text for h in m.search("Which region does payments deploy to?").items]
    assert texts[0].startswith("The payments") and not any("lunch" in t for t in texts)
