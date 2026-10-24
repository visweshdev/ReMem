"""End-to-end behaviour of the feedback loop on a small world with the flat-vector backend (fast)."""

from dataclasses import replace

import pytest

from remem.backends import VectorBackend
from remem.config import load_systems
from remem.eval.agent import AgentConfig
from remem.eval.runner import apply_world_selection, build_pools, run_stream
from remem.eval.world import make_world
from remem.types import MemState
from tests.conftest import ROOT


@pytest.fixture(scope="module")
def setup(embedder):
    world = make_world("useless_frequent", 7, {"n_tasks": 150, "filler": 20})
    pools = build_pools(world, VectorBackend(embedder), 20)
    systems = load_systems(ROOT / "configs" / "systems.yaml")
    return world, pools, systems


def _run(setup, embedder, name, **sys_over):
    world, pools, systems = setup
    cfg = apply_world_selection(systems[name], world)
    for k, v in sys_over.items():
        cfg = replace(cfg, **{k: v})
    return run_stream(world, pools, cfg, embedder, AgentConfig(mode="first"), seed=7)


def _heuristic_adaptive(systems):
    base = systems["collect"]
    return replace(base, name="heur", collect_exploration=0)


def test_runs_are_deterministic(setup, embedder):
    a = _run(setup, embedder, "A_vector", backend="vector")
    b = _run(setup, embedder, "A_vector", backend="vector")
    assert [r.selected_ids for r in a.records] == [r.selected_ids for r in b.records]
    assert [r.success for r in a.records] == [r.success for r in b.records]


def test_feedback_loop_reduces_useless_exposure_and_logs_lifecycle(setup, embedder):
    world, pools, systems = setup
    base = _run(setup, embedder, "A_vector", backend="vector")
    cfg = _heuristic_adaptive(systems)
    adaptive = run_stream(world, pools, apply_world_selection(cfg, world), embedder, AgentConfig(mode="first"), seed=7)
    junk = lambda run: sum(r.n_trap_selected + r.n_hub_selected for r in run.records)   # noqa: E731
    assert junk(adaptive) < 0.6 * junk(base)
    # some trap/hub memory was demoted, and its history shows why
    demoted = [m for m in adaptive.ledger.records.values() if m.state in (MemState.LOW_VALUE, MemState.COMPRESSED, MemState.ARCHIVED)]
    assert demoted
    kinds = {e.kind for e in demoted[0].events}
    assert {"CREATED", "RETRIEVED", "NO_CONTRIBUTION", "STATE_CHANGE"} <= kinds


def test_gold_memories_are_credited(setup, embedder):
    world, pools, systems = setup
    run = run_stream(world, pools, apply_world_selection(_heuristic_adaptive(systems), world), embedder,
                     AgentConfig(mode="first"), seed=7)
    gold_ids = {g for t in world.tasks for g in world.gold_ids(t)}
    credited = [run.ledger.get(g).n_credited for g in gold_ids if run.ledger.get(g).n_retrieved >= 3]
    assert credited and sum(c > 0 for c in credited) / len(credited) > 0.7


def test_ledger_persistence_roundtrip(setup, embedder, tmp_path):
    run = _run(setup, embedder, "A_vector", backend="vector")
    run.ledger.save(tmp_path / "l.json")
    from remem.ledger import MemoryLedger
    led = MemoryLedger.load(tmp_path / "l.json")
    assert led.state_counts() == run.ledger.state_counts() and led.step == run.ledger.step
    some = next(iter(run.ledger.records))
    assert led.get(some).events[0].kind == "CREATED"


def test_contradiction_handling_removes_stale_memory_from_context(embedder):
    world = make_world("contradiction", 5, {"n_tasks": 120, "filler": 20})
    pools = build_pools(world, VectorBackend(embedder), 20)
    systems = load_systems(ROOT / "configs" / "systems.yaml")
    off = apply_world_selection(replace(systems["A_vector"], backend="vector"), world)
    on = replace(off, contradiction=replace(systems["C_remem"].contradiction))
    r_off = run_stream(world, pools, off, embedder, AgentConfig(mode="first"), seed=5)
    r_on = run_stream(world, pools, on, embedder, AgentConfig(mode="first"), seed=5)
    stale = lambda run: sum(r.n_stale_selected > 0 for r in run.records)   # noqa: E731
    assert stale(r_on) < 0.3 * stale(r_off)
    assert r_on.false_invalidations == 0 and r_on.missed_invalidations == 0
    assert sum(r.success for r in r_on.records) > sum(r.success for r in r_off.records)


def test_historical_queries_still_find_invalidated_memories(embedder):
    world = make_world("state_change", 5, {"n_tasks": 150, "filler": 20})
    pools = build_pools(world, VectorBackend(embedder), 20)
    systems = load_systems(ROOT / "configs" / "systems.yaml")
    cfg = apply_world_selection(replace(systems["A_vector"], backend="vector"), world)
    on = replace(cfg, contradiction=replace(systems["C_remem"].contradiction))
    run = run_stream(world, pools, on, embedder, AgentConfig(mode="latest"), seed=5)
    hist = [r for r in run.records if r.kind == "historical"]
    assert hist, "world must contain historical queries"
    # the gold (now invalidated) memory is offered for as-of queries, not filtered out
    assert sum(r.gold_filtered for r in hist) == 0
    assert sum(r.gold_in_context for r in hist) / sum(r.n_gold for r in hist) > 0.6


def test_ledger_cli_renders_lifecycle_and_record(setup, embedder, tmp_path, capsys):
    from remem.viz.ledger_cli import lifecycle, main
    world, pools, systems = setup
    run = run_stream(world, pools, apply_world_selection(_heuristic_adaptive(systems), world), embedder,
                     AgentConfig(mode="first"), seed=7)
    run.ledger.save(tmp_path / "l.json")
    gold = world.gold_ids(world.tasks[-1])[0]
    text = lifecycle(run.ledger.get(gold), run.ledger)
    assert text.startswith("CREATED") and "utility" in text
    assert main([str(tmp_path / "l.json"), "--id", gold]) == 0
    out = capsys.readouterr().out
    assert "Lifecycle" in out and "Why retrieved" in out and gold in out
    assert main([str(tmp_path / "l.json"), "--list", "--state", "LOW_VALUE"]) == 0
