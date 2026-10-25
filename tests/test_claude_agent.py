"""Claude agent adapter, tested against a fake client (no API calls)."""

import json
from types import SimpleNamespace

from remem.claude_agent import ClaudeAgent, answer_schema
from remem.eval.world import T0, make_world


class FakeAnthropic:
    def __init__(self, reply_fn, stop_reason="end_turn"):
        self.reply_fn, self.stop_reason, self.calls = reply_fn, stop_reason, []
        self.messages = SimpleNamespace(create=self._create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(stop_reason=self.stop_reason,
                               content=[SimpleNamespace(type="thinking", thinking=""),
                                        SimpleNamespace(type="text", text=self.reply_fn(kw))],
                               usage=SimpleNamespace(input_tokens=400, output_tokens=120))


def _task_with_gold():
    w = make_world("contradiction", 0)
    t = w.tasks[5]
    gid = w.gold_ids(t)[0]
    g = next(f for f in w.facts if f.fact_id == gid)
    return w, t, g


def test_correct_answer_from_served_memory_succeeds_and_cites():
    w, t, g = _task_with_gold()
    key = f"{t.slots[0][0]}.{t.slots[0][1]}"
    client = FakeAnthropic(lambda kw: json.dumps({"answers": {key: w.answer(t)[key]}, "cited": [g.fact_id, "fact_bogus"]}))
    agent = ClaudeAgent(w, client, "claude-opus-5-5", t0=T0)
    res = agent.act(t, [(g.fact_id, g.text, g.time)])
    assert res.success and res.trace.cited_ids == [g.fact_id]
    kw = client.calls[0]
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["output_config"]["format"]["schema"] == answer_schema(t) and kw["output_config"]["effort"] == "low"
    assert f"[{g.fact_id}]" in kw["messages"][0]["content"] and "temperature" not in kw
    assert agent.input_tokens == 400 and abs(agent.cost_usd() - (400 * 4 + 120 * 20) / 1e6) < 1e-12


def test_refusal_and_garbage_are_failures_and_models_without_fallback_use_plain_create():
    w, t, g = _task_with_gold()
    refused = ClaudeAgent(w, FakeAnthropic(lambda kw: "", stop_reason="refusal"), "claude-opus-5-5", t0=T0)
    assert not refused.act(t, []).success and refused.refusals == 1
    client = FakeAnthropic(lambda kw: "not json")
    plain = ClaudeAgent(w, client, "claude-haiku-5-5", t0=T0)
    assert not plain.act(t, []).success and plain.parse_failures == 1
    assert "fallbacks" not in client.calls[0]


def test_schema_requires_every_slot():
    w = make_world("distractor", 0)
    t = next(t for t in w.tasks if len(t.slots) == 2)
    s = answer_schema(t)["properties"]["answers"]
    assert set(s["required"]) == {f"{a}.{b}" for a, b in t.slots} and s["additionalProperties"] is False


def test_runs_inside_the_harness(embedder):
    from remem.backends import VectorBackend
    from remem.config import load_systems
    from remem.eval.runner import apply_world_selection, build_pools, run_stream
    from tests.conftest import ROOT

    w = make_world("distractor", 3, {"n_tasks": 10, "filler": 10})
    pools = build_pools(w, VectorBackend(embedder), 20)
    cfg = apply_world_selection(load_systems(ROOT / "configs" / "systems.yaml")["A_vector"], w)

    def oracle(kw):   # answers correctly whenever the gold memory text was served
        task = next(t for t in w.tasks if t.query in kw["messages"][0]["content"])
        served = kw["messages"][0]["content"]
        gold = [next(f for f in w.facts if f.fact_id == g) for g in w.gold_ids(task)]
        ans = {f"{s}.{a}": (w.answer(task)[f"{s}.{a}"] if gf.text in served else "unknown")
               for (s, a), gf in zip(task.slots, gold)}
        return json.dumps({"answers": ans, "cited": [gf.fact_id for gf in gold if gf.text in served]})

    run = run_stream(w, pools, cfg, embedder, seed=3,
                     agent_factory=lambda wd, s: ClaudeAgent(wd, FakeAnthropic(oracle), "claude-sonnet-5-5", t0=T0))
    assert [r.success for r in run.records] == [r.gold_in_context == r.n_gold for r in run.records]
