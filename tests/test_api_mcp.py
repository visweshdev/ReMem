"""High-level API, persistence, MCP tool functions and vector-store backends."""

import importlib.util
import time

import pytest

from remem import ReMem

DAY = 86400.0


def _billing(m):
    old = m.add("The billing service exposes API version v1.", entity="billing", attribute="api_version",
                when=time.time() - 30 * DAY)
    note = m.add("Notes: someone asked which API version billing exposes; nothing decided.",
                 entity="billing", attribute="api_version", role="note")
    new = m.add("The billing service migrated and now exposes API version v2.", entity="billing", attribute="api_version")
    return old, note, new


def test_api_supersedes_learns_and_persists(tmp_path, embedder):
    m = ReMem(tmp_path, embedder=embedder)
    old, note, new = _billing(m)
    assert m.inspect(old)["state"] == "INVALIDATED" and m.inspect(old)["superseded_by"] == new
    for _ in range(7):
        r = m.search("Which API version does billing expose?", task="api")
        m.feedback(r.search_id, True, used_ids=[h.id for h in r.items if "v2" in h.text])
    assert [h.id for h in r.items][0] == new
    assert m.inspect(note)["state"] == "LOW_VALUE"
    credited = m.inspect(new)["credited"]
    m2 = ReMem(tmp_path, embedder=embedder)                       # reload from disk
    assert m2.stats()["memories"] == 3
    assert m2.inspect(old)["state"] == "INVALIDATED" and m2.inspect(note)["state"] == "LOW_VALUE"
    assert m2.inspect(new)["credited"] == credited
    assert m2.search("Which API version does billing expose?", task="api").items[0].id == new


def test_api_validates_input_and_feedback_ids(embedder):
    m = ReMem(None, embedder=embedder)
    with pytest.raises(ValueError):
        m.add("   ")
    assert m.search("anything").items == []
    m.add("The payments service is hosted in region eu-west-1.")
    r = m.search("payments region")
    m.feedback(r.search_id, True)
    with pytest.raises(KeyError):
        m.feedback(r.search_id, True)                             # each search gets one feedback
    with pytest.raises(KeyError):
        m.inspect("nope")


def test_historical_query_returns_superseded_fact(embedder):
    m = ReMem(None, embedder=embedder)
    t0 = time.time() - 60 * DAY
    old = m.add("The billing service exposes API version v1.", entity="billing", attribute="api_version", when=t0)
    m.add("The billing service exposes API version v2.", entity="billing", attribute="api_version", when=t0 + 30 * DAY)
    past = m.search("Which API version does billing expose?", as_of=t0 + 10 * DAY)
    assert [h.id for h in past.items] == [old]


def test_mcp_tool_functions(tmp_path, monkeypatch):
    import remem.mcp_server as srv
    monkeypatch.setenv("REMEM_PATH", str(tmp_path / "store"))
    monkeypatch.setattr(srv, "_mem", None)
    srv.memory_remember("The billing service exposes API version v1.", "billing", "api_version")
    new = srv.memory_remember("The billing service migrated and now exposes API version v2.", "billing", "api_version")["id"]
    rec = srv.memory_recall("Which API version does billing expose?", "api")
    assert rec["items"][0]["id"] == new
    assert srv.memory_feedback(rec["search_id"], True, [new])["credit"][new] >= 0.5
    assert srv.memory_inspect(new)["credited"] >= 0.5
    assert srv.memory_stats()["states"]["INVALIDATED"] == 1
    assert {m["id"] for m in srv.memory_list(state="ACTIVE")} == {new}


@pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="mcp not installed")
def test_mcp_server_registers_tools():
    import asyncio
    from remem.mcp_server import build_server
    names = {t.name for t in asyncio.run(build_server().list_tools())}
    assert names == {"remember", "recall", "feedback", "inspect", "list_memories", "stats"}


@pytest.mark.parametrize("backend", ["FaissBackend", "ChromaBackend"])
def test_vector_store_backends_match_the_builtin_ranking(backend, embedder):
    module = {"FaissBackend": "faiss", "ChromaBackend": "chromadb"}[backend]
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"{module} not installed")
    import remem.backends as b
    from remem.backends import StoredFact, VectorBackend
    facts = [StoredFact(f"fact_{i}", t, float(i)) for i, t in enumerate([
        "The payments service is hosted in region eu-west-1.", "Team lunch moved to Thursday.",
        "The billing service is hosted in region us-east-1.", "Search runs on API version v3."])]
    ref, other = VectorBackend(embedder), getattr(b, backend)(embedder)
    ref.ingest(facts)
    other.ingest(facts)
    q = "Which region does payments deploy to?"
    assert [c.fact_id for c in other.candidates(q, k=3)[0]] == [c.fact_id for c in ref.candidates(q, k=3)[0]]
