"""Turn raw conversation text into ReMem facts (text + entity + attribute + role).

    from remem.extract import LLMFactExtractor
    ex = LLMFactExtractor(OpenAI(), model="gpt-4o-mini")
    for f in ex.extract("User: we moved billing to API v2 last week. Also lunch is Thursday."):
        mem.add(f.text, entity=f.entity, attribute=f.attribute, role=f.role)
    # or simply: mem.add_conversation(text, extractor=ex)

`SentenceSplitter` is a dependency-free fallback that stores each sentence as a note (no entity/attribute,
so no automatic supersession). `LLMFactExtractor` works with any OpenAI-compatible chat client.

STATUS: LLMFactExtractor is tested against a fake client only; its extraction quality with a real model
has not been measured in this repository.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

PROMPT = """Extract durable facts worth remembering from the conversation below.

Rules:
- One self-contained third-person sentence per fact, with names resolved (no "he", "it", "we").
- Skip greetings, chit-chat, questions and anything not worth remembering later.
- entity: the main thing the fact is about, lowercase (e.g. "billing", "alice"), or null.
- attribute: a short snake_case key for WHAT about the entity it states (e.g. "api_version", "city",
  "owner"), or null. Facts that state the same attribute of the same entity must use the same key.
- role: "state" if it states a current property that a later fact could replace; "event" for things that
  happened; "preference" for likes/dislikes; "note" for anything else.

Return JSON only: {"facts": [{"text": "...", "entity": "...", "attribute": "...", "role": "state"}]}

Conversation:
"""

VALID_ROLES = {"state", "event", "preference", "note", "constraint"}


@dataclass
class ExtractedFact:
    text: str
    entity: str | None = None
    attribute: str | None = None
    role: str = "note"


class SentenceSplitter:
    """Fallback extractor: every non-trivial sentence becomes a note."""

    def __init__(self, min_words: int = 4) -> None:
        self.min_words = min_words

    def extract(self, text: str) -> list[ExtractedFact]:
        parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
        out = []
        for p in parts:
            p = re.sub(r"^\s*(user|assistant|system)\s*:\s*", "", p.strip(), flags=re.I)
            if len(p.split()) >= self.min_words and not p.endswith("?"):
                out.append(ExtractedFact(p, role="note"))
        return out


class LLMFactExtractor:
    def __init__(self, client: Any, model: str, *, temperature: float = 0.0, max_tokens: int = 800) -> None:
        self.client, self.model, self.temperature, self.max_tokens = client, model, temperature, max_tokens

    def extract(self, text: str) -> list[ExtractedFact]:
        resp = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": PROMPT + text}],
        )
        return parse_facts(resp.choices[0].message.content)


def parse_facts(reply: str) -> list[ExtractedFact]:
    """Tolerant parser for the extractor's JSON; malformed items are skipped, never raised."""
    m = re.search(r"\{.*\}", reply or "", re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for item in data.get("facts", []) if isinstance(data, dict) else []:
        if not isinstance(item, dict) or not str(item.get("text", "")).strip():
            continue
        role = str(item.get("role") or "note").lower()
        ent = item.get("entity")
        att = item.get("attribute")
        out.append(ExtractedFact(
            text=str(item["text"]).strip(),
            entity=str(ent).strip().lower() if ent else None,
            attribute=re.sub(r"\W+", "_", str(att).strip().lower()).strip("_") if att else None,
            role=role if role in VALID_ROLES else "note",
        ))
    return out
