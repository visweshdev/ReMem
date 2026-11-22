"""Safe fact replacement: same entity+attribute alone does not retire a fact unless the update is declared or confirmed.

The 25:50 case is from the real LongMemEval run: a small extractor filed "personal best 25:50" and a later, vaguer
"hopes to beat their personal best" under the same key, and key-match replacement threw the answer away.
"""

import json
from types import SimpleNamespace

import pytest

from remem import ReMem
from remem.extract import UPDATE_PROMPT, ExtractedFact, LLMUpdateChecker

DAY = 86400.0
BEST = "The user's personal best time in the charity 5K run is 25:50."
VAGUE = "The user is training for a charity 5K run and hopes to beat their personal best time."


class FakeChat:
    def __init__(self, reply):
        self.reply, self.calls = reply, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])


def _add_pair(m, update="check"):
    old = m.add(BEST, entity="user", attribute="personal_best_time", when=1 * DAY)
    new = m.add(VAGUE, entity="user", attribute="personal_best_time", when=5 * DAY, update=update)
    return old, new


def test_default_keeps_both_facts_without_a_checker(embedder):
    m = ReMem(None, embedder=embedder)
    old, new = _add_pair(m)
    assert m.inspect(old)["state"] == "ACTIVE" and m.inspect(new)["state"] == "ACTIVE"
    assert any("25:50" in h.text for h in m.search("What is my 5K personal best?", now=6 * DAY).items)


def test_checker_saying_no_keeps_the_25_50_fact(embedder):
    client = FakeChat("No")
    m = ReMem(None, embedder=embedder, update_checker=LLMUpdateChecker(client, "m"))
    old, _ = _add_pair(m)
    assert m.inspect(old)["state"] == "ACTIVE"
    prompt = client.calls[0]["messages"][0]["content"]
    assert BEST in prompt and VAGUE in prompt and client.calls[0]["temperature"] == 0


def test_checker_saying_yes_retires_the_older_fact(embedder):
    m = ReMem(None, embedder=embedder, update_checker=LLMUpdateChecker(FakeChat("Yes."), "m"))
    old = m.add("The user's personal best 5K time is 27:12.", entity="user", attribute="personal_best_time", when=1 * DAY)
    new = m.add(BEST, entity="user", attribute="personal_best_time", when=5 * DAY)
    assert m.inspect(old)["state"] == "INVALIDATED" and m.inspect(old)["superseded_by"] == new


def test_checker_is_not_called_when_values_match_or_keys_differ(embedder):
    calls = []
    m = ReMem(None, embedder=embedder, update_checker=lambda o, n: calls.append((o, n)) or True)
    m.add(BEST, entity="user", attribute="personal_best_time", when=1 * DAY)
    m.add(BEST, entity="user", attribute="personal_best_time", when=2 * DAY)           # same value restated
    m.add("The user lives in Lisbon.", entity="user", attribute="city", when=3 * DAY)  # different key
    assert calls == []


def test_declared_update_and_keep(embedder):
    m = ReMem(None, embedder=embedder)
    old, _ = _add_pair(m, update="replace")
    assert m.inspect(old)["state"] == "INVALIDATED"
    m2 = ReMem(None, embedder=embedder, update_checker=lambda o, n: True)
    old2, _ = _add_pair(m2, update="keep")
    assert m2.inspect(old2)["state"] == "ACTIVE"
    with pytest.raises(ValueError):
        m.add("x y z", update="sometimes")


@pytest.mark.parametrize("policy", ["replace", "check"])
def test_updates_can_reverse_a_fact_without_changing_its_value_tokens(embedder, policy):
    m = ReMem(embedder=embedder, update_checker=lambda old, new: "not" in new)
    key = {"entity": "server", "attribute": "status"}
    old = m.add("The server is available.", when=DAY, **key)
    new = m.add("The server is not available.", when=2 * DAY, update=policy, **key)
    assert m.inspect(old)["superseded_by"] == new
    assert [hit.id for hit in m.search("Is the server available?", as_of=DAY).items] == [old]
    assert [hit.id for hit in m.search("Is the server available?", now=3 * DAY).items] == [new]


def test_update_policy_and_source_survive_reload(tmp_path, embedder):
    m = ReMem(tmp_path, embedder=embedder)
    old, _ = _add_pair(m)
    m.add("Billing uses API v2.", entity="billing", attribute="api_version", source="user: billing is on v2 now")
    m2 = ReMem(tmp_path, embedder=embedder)                     # replay facts.jsonl: still no replacement
    assert m2.inspect(old)["state"] == "ACTIVE"
    facts = [json.loads(line) for line in (tmp_path / "facts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert facts[1]["update_policy"] == "check" and facts[2]["source"] == "user: billing is on v2 now"
    hit = next(h for h in m2.search("billing API version").items if "v2" in h.text)
    assert hit.source == "user: billing is on v2 now"


def test_mcp_keeps_both_unless_the_model_says_it_is_an_update(tmp_path, monkeypatch):
    import remem.mcp_server as srv
    monkeypatch.setenv("REMEM_PATH", str(tmp_path / "store"))
    monkeypatch.setattr(srv, "_mem", None)
    monkeypatch.setattr(srv, "_unreported", {})
    monkeypatch.setenv("REMEM_AUTO_FEEDBACK", "0")
    a = srv.memory_remember(BEST, "user", "personal_best_time")["id"]
    srv.memory_remember(VAGUE, "user", "personal_best_time")
    assert srv.memory_inspect(a)["state"] == "ACTIVE"
    srv.memory_remember("The user's personal best 5K time is now 24:59.", "user", "personal_best_time", is_update=True)
    assert srv.memory_inspect(a)["state"] == "INVALIDATED"


def test_update_prompt_mentions_both_facts():
    assert "{old}" in UPDATE_PROMPT and "{new}" in UPDATE_PROMPT


def test_missing_source_evidence_blocks_hallucinated_update(embedder):
    client = FakeChat("Yes")
    memory = ReMem(embedder=embedder, update_checker=LLMUpdateChecker(client, "m"))
    key = {"entity": "runner", "attribute": "record"}
    old = memory.add(BEST, when=DAY, source="I finished in 25:50.", **key)
    memory.add("The new record is 24:00.", when=2 * DAY, source="I hope to beat my record.", **key)
    assert memory.inspect(old)["state"] == "ACTIVE"
    assert client.calls == []


def test_sources_are_passed_for_out_of_order_insertion_and_strict_reply(embedder):
    client = FakeChat("Yes, but the race is still planned.")
    memory = ReMem(embedder=embedder, update_checker=LLMUpdateChecker(client, "m"))
    key = {"entity": "runner", "attribute": "record"}
    memory.add(BEST, when=2 * DAY, source="My new time is 25:50.", **key)
    old = memory.add("The previous time was 27:12.", when=DAY, source="I finished in 27:12.", **key)
    assert memory.inspect(old)["state"] == "ACTIVE"
    prompt = client.calls[0]["messages"][0]["content"]
    assert "Older source: I finished in 27:12." in prompt
    assert "Newer source: My new time is 25:50." in prompt


def test_source_checks_require_strictly_older_fact(embedder):
    client = FakeChat("Yes")
    memory = ReMem(embedder=embedder, update_checker=LLMUpdateChecker(client, "m"))
    key = {"entity": "runner", "attribute": "record", "when": DAY}
    ids = [memory.add(text, **key) for text in (BEST, VAGUE)]
    assert all(memory.inspect(ident)["state"] == "ACTIVE" for ident in ids)
    assert client.calls == []


def test_facts_from_one_conversation_share_a_recorded_timestamp(embedder):
    client = FakeChat("Yes")
    memory = ReMem(embedder=embedder, update_checker=LLMUpdateChecker(client, "m"))
    records = [ExtractedFact(text, "runner", "record", "state") for text in (BEST, VAGUE)]
    extractor = SimpleNamespace(extract=lambda _: records)

    ids = memory.add_conversation("My new record is 25:50.", extractor=extractor)

    assert len({memory.inspect(ident)["valid_from"] for ident in ids}) == 1
    assert all(memory.inspect(ident)["state"] == "ACTIVE" for ident in ids)
    assert client.calls == []
