"""Safe fact replacement: same entity+attribute alone does not retire a fact unless the update is declared or confirmed.

The 25:50 case is from the real LongMemEval run: a small extractor filed "personal best 25:50" and a later, vaguer
"hopes to beat their personal best" under the same key, and key-match replacement threw the answer away.
"""

import json
from types import SimpleNamespace

import pytest

from remem import ReMem
from remem.extract import UPDATE_PROMPT, LLMUpdateChecker

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
