"""LLM adapters, tested against fake clients ONLY (no live endpoint was available when written)."""

import json
from types import SimpleNamespace

from remem.eval.world import T0, make_world
from remem.llm import LLMAgent, OpenAIEmbedder, build_prompt, parse_reply


class FakeChat:
    def __init__(self, reply):
        self.reply, self.calls = reply, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        text = self.reply(kw) if callable(self.reply) else self.reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _task_and_gold(world):
    t = world.tasks[0]
    gid = world.gold_ids(t)[0]
    fact = next(f for f in world.facts if f.fact_id == gid)
    return t, fact


def test_agent_scores_correct_answer_filters_invented_citations_and_shows_ids():
    w = make_world("distractor", 0)
    t, g = _task_and_gold(w)
    key = f"{t.slots[0][0]}.{t.slots[0][1]}"
    reply = json.dumps({"answers": {key: w.answer(t)[key]}, "cited": [g.fact_id, "fact_99999"]})
    client = FakeChat("Sure! " + reply + " done")                      # tolerant extraction
    agent = LLMAgent(w, client, "m", t0=T0)
    res = agent.act(t, [(g.fact_id, g.text, g.time)])
    assert res.success and res.trace.cited_ids == [g.fact_id] and res.used_ids == [g.fact_id]
    prompt = client.calls[0]["messages"][1]["content"]
    assert f"[{g.fact_id}]" in prompt and t.query in prompt and "Memories:" in prompt
    assert client.calls[0]["temperature"] == 0.0 and agent.n_calls == 1


def test_wrong_or_malformed_replies_are_failures_not_crashes():
    w = make_world("distractor", 0)
    t, g = _task_and_gold(w)
    key = f"{t.slots[0][0]}.{t.slots[0][1]}"
    wrong = LLMAgent(w, FakeChat(json.dumps({"answers": {key: "nonsense"}, "cited": []})), "m", t0=T0).act(t, [])
    garbage = LLMAgent(w, FakeChat("I cannot help"), "m", t0=T0).act(t, [])
    assert not wrong.success and not garbage.success and garbage.trace.answer == {}
    assert parse_reply('{"answers": "x", "cited": []}') == ({}, [])


def test_prompt_includes_as_of_for_historical_queries():
    w = make_world("state_change", 1)
    t = next(t for t in w.tasks if t.as_of is not None)
    assert "as of day" in build_prompt(t, [], T0).lower()


def test_agent_works_inside_the_runner_with_replay(embedder):
    from remem.backends import VectorBackend
    from remem.config import load_systems
    from remem.eval.runner import apply_world_selection, build_pools, run_stream
    from tests.conftest import ROOT

    w = make_world("distractor", 2, {"n_tasks": 12, "filler": 10})
    pools = build_pools(w, VectorBackend(embedder), 20)
    cfg = apply_world_selection(load_systems(ROOT / "configs" / "systems.yaml")["A_vector"], w)
    cfg = cfg.__class__(**{**cfg.__dict__, "backend": "vector"})

    client = FakeChat(json.dumps({"answers": {}, "cited": []}))      # plumbing only: every answer is missing
    run = run_stream(w, pools, cfg, embedder, seed=2, agent_factory=lambda wd, s: LLMAgent(wd, client, "m", t0=T0))
    assert len(run.records) == len(w.tasks) and len(client.calls) == len(w.tasks)
    assert not any(r.success for r in run.records)


def test_openai_embedder_batches_and_normalises():
    calls = []

    def create(model, input):
        calls.append(len(input))
        return SimpleNamespace(data=[SimpleNamespace(embedding=[3.0, 4.0]) for _ in input])

    emb = OpenAIEmbedder(SimpleNamespace(embeddings=SimpleNamespace(create=create)), "e", 2)
    out = emb.embed_texts(["a"] * 5, batch_size=2)
    assert calls == [2, 2, 1] and out[0] == [0.6, 0.8] and emb.embed_one("x") == [0.6, 0.8]


def test_schema_mode_constrains_the_answer_keys():
    import json as _json
    from types import SimpleNamespace

    from remem.eval.world import T0, make_world
    from remem.llm import LLMAgent
    w = make_world("distractor", 0)
    t = w.tasks[0]
    calls = []

    def create(**kw):
        calls.append(kw)
        msg = SimpleNamespace(content=_json.dumps({"answers": w.answer(t), "cited": []}))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert LLMAgent(w, client, "m", t0=T0, json_mode="schema").act(t, []).success
    rf = calls[0]["response_format"]
    assert rf["type"] == "json_schema" and set(rf["json_schema"]["schema"]["properties"]["answers"]["required"]) == set(w.answer(t))
