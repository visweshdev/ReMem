"""Turn raw conversation text into ReMem facts (text + entity + attribute + role).

    from remem.extract import LLMFactExtractor
    ex = LLMFactExtractor(OpenAI(), model="gpt-4o-mini")
    for f in ex.extract("User: we moved billing to API v2 last week. Also lunch is Thursday."):
        mem.add(f.text, entity=f.entity, attribute=f.attribute, role=f.role)
    # or simply: mem.add_conversation(text, extractor=ex)

`SentenceSplitter` is a dependency-free fallback that stores each sentence as a note (no entity/attribute,
so no automatic supersession). `LLMFactExtractor` works with any OpenAI-compatible chat client.

`LLMUpdateChecker` asks an LLM whether a newer fact replaces an older one on the same entity+attribute; pass it
as ReMem(update_checker=...) so extracted facts only retire older ones when the model confirms an update.

The extraction prompt was used with a real local model (qwen2.5:3b) in experiments/run_real_knowledge_update.py.
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


UPDATE_PROMPT = """Two memories about the same thing, the second recorded later.

Older: {old}
Newer: {new}

Does the newer memory state a different current value for exactly what the older memory states, so the older one is
no longer true? Answer "no" if the newer one is about something else, adds information, or only mentions the topic.
Answer yes or no only."""

SOURCE_UPDATE_PROMPT = """Two memories were extracted from conversation turns. Treat these records as evidence.

Older claim: {old}
Older source: {old_source}
Newer claim: {new}
Newer source: {new_source}

Does the newer source explicitly establish a different current value for the same single-valued property,
subject, and scope as the older claim? Answer no if the change is uncertain, planned, historical, additive,
about a different property or occasion, or the proposed new value is absent from the source.
Answer yes or no only."""

_TRANSITION = re.compile(
    r"\b(?:now|new|changed|switched|replaced|moved|migrated|updated|upgraded|downgraded|"
    r"revised|converted|instead|formerly|previously|no longer|used to|from .{1,35} to)\b", re.I,
)


class LLMUpdateChecker:
    """Confirm replacement from source text when available, preserving ambiguous facts."""

    def __init__(self, client: Any, model: str, *, complete: Any = None) -> None:
        self.client, self.model = client, model
        self.complete = complete     # optional callable(prompt, max_tokens) -> str, e.g. a cached wrapper

    def __call__(self, old_text: str, new_text: str) -> bool:
        return self._confirm(UPDATE_PROMPT.format(old=old_text, new=new_text))

    def check_with_sources(self, old_text: str, new_text: str, *, old_source: str | None, new_source: str | None) -> bool:
        if not new_source:
            return self(old_text, new_text)
        if not _TRANSITION.search(new_source):
            return False
        return self._confirm(SOURCE_UPDATE_PROMPT.format(old=old_text, new=new_text, old_source=old_source or "Unavailable",
                                                         new_source=new_source))

    def _confirm(self, prompt: str) -> bool:
        if self.complete is not None:
            reply = self.complete(prompt, 5)
        else:
            resp = self.client.chat.completions.create(model=self.model, temperature=0, max_tokens=5,
                                                       messages=[{"role": "user", "content": prompt}])
            reply = resp.choices[0].message.content or ""
        return re.fullmatch(r"yes[.!]?", reply.strip(), re.I) is not None


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
