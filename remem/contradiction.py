"""Contradiction / staleness handling: supersession, validity intervals, confidence decay.

Inputs are what a typical LLM fact-extraction step emits per fact: `entities`,
`attribute_keys` and a `semantic_role`, plus the fact text and its timestamp.
A *key* is (entity, attribute_key). Two facts conflict when they share a key, are about the
differ in *value-bearing* tokens (rare in the corpus or digit-bearing; update cues and template
wording ignored). The text-similarity guard `min_sim` applies only to the cue path. Whether a conflict means "the old fact is no longer true" depends on the attribute:

  * attribute listed as exclusive (single-valued: region, owner, city ...) -> supersede
  * otherwise only when the newer text carries an explicit update cue ("migrated", "no longer" ...)
  * independent of both: a failed action that relied on memory M while a newer fact on the same
    key exists supersedes M (`outcome_contradiction`) — the outcome-aware part.

A superseded memory is not deleted. Its `valid_to` is set and state becomes INVALIDATED; it is
excluded from default retrieval but still returned for historical (`as_of`) queries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from remem.config import ContradictionConfig
from remem.ledger import MemoryLedger
from remem.offline import content_tokens
from remem.types import MemoryRecord, MemState

Key = tuple[str, str]


@dataclass
class Conflict:
    old_id: str
    new_id: str
    kind: str          # exclusive | cue | outcome
    sim: float


class ContradictionDetector:
    def __init__(
        self,
        cfg: ContradictionConfig,
        ledger: MemoryLedger,
        text_sim: Callable[[str, str], float],
    ) -> None:
        self.cfg = cfg
        self.ledger = ledger
        self.text_sim = text_sim
        self.by_key: dict[Key, list[str]] = {}
        self.volatile_attrs: set[str] = set()
        self.conflicts: list[Conflict] = []
        # schema learning: which attributes are single-valued, inferred from outcome evidence
        self.learned_exclusive: set[str] = set()
        self.schema_evidence: dict[str, list[float]] = {}          # attr -> [confirmations e, counter-evidence m]
        self.pending: dict[str, list[tuple[str, str]]] = {}        # attr -> (old_id, new_id) pairs with differing values, unresolved
        self.schema_log: list[tuple[int, str, float, float]] = []  # (ledger step, attr, e, m) when an attribute was learned
        self._cue_tokens = {t for cue in cfg.update_cues for t in content_tokens(cue)} | set(cfg.update_cues)

    # ── ingest-time detection ──────────────────────────────────────────────────
    def keys_of(self, rec: MemoryRecord) -> list[Key]:
        return [(e.lower(), a.lower()) for e in rec.entities for a in rec.attribute_keys]

    def on_ingest(self, rec: MemoryRecord) -> list[Conflict]:
        found: list[Conflict] = []
        if not self.cfg.enabled:
            return found
        if rec.semantic_role not in self.cfg.roles:
            return found
        for key in self.keys_of(rec):
            peers = self.by_key.setdefault(key, [])
            for pid in list(peers):
                other = self.ledger.get(pid)
                if other is None or other.fact_id == rec.fact_id:
                    continue
                conflict = self._compare(other, rec, key)
                if conflict is not None:
                    self._apply(conflict)
                    found.append(conflict)
            if rec.fact_id not in peers:
                peers.append(rec.fact_id)
        return found

    def _value_tokens(self, text: str) -> set[str]:
        """Tokens that can carry a *value*: rare in the corpus or containing a digit, minus update cues.

        Template wording ("service", "hosted", "deployment") is frequent across facts and is ignored,
        so a same-value paraphrase is not mistaken for a conflict. With a tiny corpus there are no
        reliable frequencies, so every token is considered.
        """
        toks = set(content_tokens(text)) - self._cue_tokens
        if len(self.ledger.records) < self.cfg.min_corpus_for_df:
            return toks
        return {t for t in toks if any(c.isdigit() for c in t) or self.ledger.df_fraction(t) <= self.cfg.max_value_df}

    def _compare(self, a: MemoryRecord, b: MemoryRecord, key: Key) -> Conflict | None:
        """Decide if a and b conflict; the one with the later timestamp wins."""
        old, new = (a, b) if a.time <= b.time else (b, a)
        if old.state == MemState.INVALIDATED and old.valid_to is not None and old.valid_to <= new.time:
            return None
        ov, nv = self._value_tokens(old.original_text), self._value_tokens(new.original_text)
        if not (nv - ov) or not (ov - nv):
            return None               # same value restated (or nothing value-like differs): reconfirmation
        exclusive = self.is_exclusive(key[1]) or (
            old.semantic_role in self.cfg.exclusive_roles and new.semantic_role in self.cfg.exclusive_roles)
        if exclusive:                 # declared single-valued: same key + different value is enough
            return Conflict(old.fact_id, new.fact_id, "exclusive", self.text_sim(old.original_text, new.original_text))
        has_cue = bool(set(content_tokens(new.original_text)) & self._cue_tokens)
        if has_cue:                   # explicit update wording needs the facts to be about the same thing
            sim = self.text_sim(old.original_text, new.original_text)
            if sim >= self.cfg.min_sim:
                return Conflict(old.fact_id, new.fact_id, "cue", sim)
        if self.cfg.learn_schema:     # unresolved: same key, different value, attribute not (yet) known single-valued
            self.pending.setdefault(key[1], []).append((old.fact_id, new.fact_id))
        return None

    # ── schema learning ────────────────────────────────────────────────────────
    def is_exclusive(self, attr: str) -> bool:
        a = attr.lower()
        return a in {x.lower() for x in self.cfg.exclusive_attributes} or a in self.learned_exclusive

    def _evidence(self, attr: str, e: float = 0.0, m: float = 0.0) -> None:
        """Update the single-valued-attribute posterior; once confident, learn it and supersede the pending pairs."""
        if not self.cfg.learn_schema:
            return
        a = attr.lower()
        ev = self.schema_evidence.setdefault(a, [0.0, 0.0])
        ev[0] += e
        ev[1] += m
        if a in self.learned_exclusive or a in {x.lower() for x in self.cfg.exclusive_attributes}:
            return
        if ev[0] >= self.cfg.schema_min_evidence and (ev[0] + 1) / (ev[0] + ev[1] + 2) >= self.cfg.schema_threshold:
            self.learned_exclusive.add(a)
            self.schema_log.append((self.ledger.step, a, ev[0], ev[1]))
            for old_id, new_id in self.pending.pop(a, []):
                old = self.ledger.get(old_id)
                if old is not None and old.state != MemState.INVALIDATED:
                    self._apply(Conflict(old_id, new_id, "learned",
                                         self.text_sim(old.original_text, self.ledger.get(new_id).original_text)))

    def on_credited(self, rec: MemoryRecord) -> None:
        """A successful task credited `rec` although a newer different-valued fact exists: it was still true,
        so the attribute is probably multi-valued."""
        if not self.cfg.learn_schema:
            return
        for attr in {a.lower() for a in rec.attribute_keys}:
            keep = []
            for old_id, new_id in self.pending.get(attr, []):
                if old_id == rec.fact_id:
                    self._evidence(attr, m=1.0)
                else:
                    keep.append((old_id, new_id))
            if attr in self.pending:
                self.pending[attr] = keep

    def _apply(self, c: Conflict) -> None:
        led = self.ledger
        old, new = led.get(c.old_id), led.get(c.new_id)
        assert old is not None and new is not None
        old.valid_to = new.time
        old.superseded_by = new.fact_id
        old.contradictions += 1
        if old.fact_id not in new.supersedes:
            new.supersedes.append(old.fact_id)
        led.set_state(old, MemState.INVALIDATED, f"superseded by {new.fact_id} ({c.kind}, sim={c.sim:.2f})")
        led.log(new, "CONTRADICTED", f"supersedes {old.fact_id} ({c.kind})")
        for a in old.attribute_keys:
            self.volatile_attrs.add(a.lower())
        self.conflicts.append(c)
        if c.kind == "cue":           # explicit replacement wording is evidence the attribute is single-valued
            for a in old.attribute_keys:
                self._evidence(a, e=1.0)

    # ── outcome-aware path ─────────────────────────────────────────────────────
    def on_failure_blame(self, rec: MemoryRecord) -> Conflict | None:
        """A failed action relied on `rec`; if a newer fact exists on the same key, supersede it."""
        if not (self.cfg.enabled and self.cfg.outcome_invalidation) or rec.state == MemState.INVALIDATED:
            return None
        if rec.semantic_role not in self.cfg.roles:   # a question/note cannot be "contradicted"
            return None
        best: MemoryRecord | None = None
        for key in self.keys_of(rec):
            for pid in self.by_key.get(key, []):
                other = self.ledger.get(pid)
                if other is None or other.fact_id == rec.fact_id or other.time <= rec.time:
                    continue
                if other.state == MemState.INVALIDATED:
                    continue
                if best is None or other.time > best.time:
                    best = other
        if best is None:
            return None
        c = Conflict(rec.fact_id, best.fact_id, "outcome", self.text_sim(rec.original_text, best.original_text))
        self._apply(c)
        for key in self.keys_of(rec):
            self._evidence(key[1], e=1.0)   # a failed action that used the stale value confirms single-valuedness
            self.pending[key[1]] = [p for p in self.pending.get(key[1], []) if p[0] != rec.fact_id]
        return c

    # ── confidence decay ───────────────────────────────────────────────────────
    def decay(self, now: float) -> None:
        """Half-life decay of confidence for memories on *volatile* attributes (ones observed to change).

        Stable attributes never decay, so old-but-true facts are not punished for age alone.
        """
        hl = self.cfg.conf_half_life_days
        if not (self.cfg.enabled and hl > 0):
            return
        for rec in self.ledger.records.values():
            if not any(a.lower() in self.volatile_attrs for a in rec.attribute_keys):
                continue
            age = max(0.0, (now - rec.time) / 86400.0)
            rec.confidence = max(self.cfg.conf_floor, math.pow(0.5, age / hl))

    # ── validity ───────────────────────────────────────────────────────────────
    @staticmethod
    def valid_at(rec: MemoryRecord, ref: float) -> bool:
        if rec.valid_from > ref:
            return False
        return rec.valid_to is None or ref < rec.valid_to
