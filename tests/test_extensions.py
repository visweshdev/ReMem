"""Fact extraction, tokenizer switch, LangChain retriever and the HTML dashboard."""

import importlib.util
import json
from types import SimpleNamespace

import pytest

from remem import ReMem
from remem.extract import LLMFactExtractor, SentenceSplitter, parse_facts


class FakeChat:
    def __init__(self, reply):
        self.reply, self.calls = reply, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])


def test_llm_extractor_parses_and_normalises_and_feeds_supersession(embedder):
    reply = "Here you go:\n" + json.dumps({"facts": [
        {"text": "The billing service exposes API version v1.", "entity": "Billing", "attribute": "API Version", "role": "state"},
        {"text": "The billing service now exposes API version v2.", "entity": "billing", "attribute": "api_version", "role": "STATE"},
        {"text": "", "entity": "x"},                                  # skipped
        {"text": "Team lunch is on Thursday.", "role": "weird"},      # unknown role -> note
    ]})
    client = FakeChat(reply)
    facts = LLMFactExtractor(client, "m").extract("User: we moved billing to v2.")
    assert [f.attribute for f in facts] == ["api_version", "api_version", None]
    assert facts[0].entity == "billing" and facts[1].role == "state" and facts[2].role == "note"
    assert "User: we moved billing to v2." in client.calls[0]["messages"][0]["content"]
    m = ReMem(None, embedder=embedder)
    ids = m.add_conversation("ignored", extractor=SimpleNamespace(extract=lambda t: facts))
    assert len(ids) == 3 and m.inspect(ids[0])["state"] == "INVALIDATED"


def test_parse_facts_never_raises_on_garbage():
    assert parse_facts("no json here") == [] and parse_facts('{"facts": "nope"}') == [] and parse_facts("{bad json}") == []


def test_sentence_splitter_fallback(embedder):
    facts = SentenceSplitter().extract("User: Hi! The payments service runs in eu-west-1. What about billing?\nAssistant: Billing runs in us-east-1 now.")
    assert [f.text for f in facts] == ["The payments service runs in eu-west-1.", "Billing runs in us-east-1 now."]
    assert ReMem(None, embedder=embedder).add_conversation("The payments service runs in eu-west-1.")


@pytest.mark.skipif(importlib.util.find_spec("tiktoken") is None, reason="tiktoken not installed")
def test_tokenizer_switch():
    from remem.offline import count_tokens, set_tokenizer
    text = "The payments service is hosted in eu-west-1."
    proxy = count_tokens(text)
    try:
        set_tokenizer("tiktoken")
        real = count_tokens(text)
    finally:
        set_tokenizer(None)
    assert real > 0 and count_tokens(text) == proxy
    with pytest.raises(ValueError):
        set_tokenizer("nonsense")


@pytest.mark.skipif(importlib.util.find_spec("langchain_core") is None, reason="langchain-core not installed")
def test_langchain_retriever_round_trip(embedder):
    from remem.integrations.langchain import ReMemRetriever
    r = ReMemRetriever(memory=ReMem(None, embedder=embedder), k=3, task="api")
    r.add_texts(["The billing service exposes API version v2.", "Team lunch moved to Thursday."])
    docs = r.invoke("Which API version does billing expose?")
    assert docs and "v2" in docs[0].page_content and docs[0].metadata["search_id"]
    credit = r.feedback(True, used_ids=[docs[0].metadata["id"]])
    assert credit[docs[0].metadata["id"]] >= 0.5
    with pytest.raises(ValueError):
        r.feedback(True)                                              # the search was already reported


def test_dashboard_html_is_self_contained_and_escapes(embedder, tmp_path):
    from remem.viz.dashboard import build_html, main
    m = ReMem(tmp_path, embedder=embedder)
    m.add("Payload </script><b>not markup</b> stays text.")
    m.add("The billing service exposes API version v2.")
    page = build_html(m.system.ledger, "test")
    assert "</script><b>" not in page and r"<\/script>" in page        # cannot break out of the script tag
    assert "http" not in page.split("<script>")[0].lower().replace("http-equiv", "")   # no external assets
    out = tmp_path / "d.html"
    assert main([str(tmp_path), "-o", str(out)]) == 0 and out.stat().st_size > 1000
