"""MemoryLedger: sidecar store of per-memory state, outcome statistics and lifecycle events.

Retrieval backends store facts but carry no usage/outcome/validity fields, so ReMem
keeps this ledger alongside them, keyed by `fact_id`. Nothing here mutates backend state;
"invalidate" or "archive" only changes what the adaptive layer will *offer* to the agent.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

from remem.offline import content_tokens, count_tokens
from remem.types import Candidate, ClassStats, LifecycleEvent, MemoryRecord, MemState

MAX_EVENTS = 200


class MemoryLedger:
    def __init__(self) -> None:
        self.records: dict[str, MemoryRecord] = {}
        self.step: int = 0
        self._df: Counter[str] = Counter()

    # ── lifecycle ──────────────────────────────────────────────────────────────
    def tick(self) -> int:
        self.step += 1
        return self.step

    def register(self, cand: Candidate | None = None, **kw) -> MemoryRecord:
        """Create a record for a new memory (idempotent)."""
        fid = cand.fact_id if cand is not None else kw["fact_id"]
        if fid in self.records:
            return self.records[fid]
        src = cand if cand is not None else None
        text = src.text if src else kw["text"]
        rec = MemoryRecord(
            fact_id=fid,
            text=text,
            original_text=text,
            tokens=src.tokens if src else count_tokens(text),
            original_tokens=src.tokens if src else count_tokens(text),
            created_step=self.step,
            time=src.time if src else kw["time"],
            confidence=src.confidence if src else kw.get("confidence", 1.0),
            entities=tuple(src.entities) if src else tuple(kw.get("entities", ())),
            attribute_keys=tuple(src.attribute_keys) if src else tuple(kw.get("attribute_keys", ())),
            semantic_role=kw.get("semantic_role", "state"),
            valid_from=src.time if src else kw["time"],
        )
        rec.last_updated_step = self.step
        self.records[fid] = rec
        self._df.update(set(content_tokens(text)))
        self.log(rec, "CREATED", f"t={rec.time:.0f} tokens={rec.tokens}")
        return rec

    def get(self, fact_id: str) -> MemoryRecord | None:
        return self.records.get(fact_id)

    def log(self, rec: MemoryRecord, kind: str, detail: str = "", utility: float | None = None) -> None:
        rec.events.append(LifecycleEvent(self.step, kind, detail, utility))
        if len(rec.events) > MAX_EVENTS:
            del rec.events[1 : len(rec.events) - MAX_EVENTS + 1]  # keep CREATED, drop oldest after it
        rec.last_updated_step = self.step

    def set_state(self, rec: MemoryRecord, new: MemState, why: str) -> bool:
        if rec.state == new:
            return False
        old = rec.state
        rec.state = new
        self.log(rec, "STATE_CHANGE", f"{old.value} -> {new.value}: {why}", rec.last_utility)
        return True

    # ── statistics (posterior over "contributes when shown") ───────────────────
    def exposures(self, rec: MemoryRecord) -> int:
        """Times shown inside a *successful* task. Failed tasks give no evidence of uselessness."""
        return rec.n_success_ctx

    def contrib_rate(self, rec: MemoryRecord, a: float = 1.0, b: float = 3.0) -> float:
        return (a + rec.n_credited) / (a + b + rec.n_success_ctx)

    def contrib_std(self, rec: MemoryRecord, a: float = 1.0, b: float = 3.0) -> float:
        p = self.contrib_rate(rec, a, b)
        return math.sqrt(p * (1.0 - p) / (a + b + rec.n_success_ctx + 1.0))

    def class_rate(self, rec: MemoryRecord, task_class: str, a: float, b: float, shrink: float) -> float:
        g = self.contrib_rate(rec, a, b)
        cs = rec.classes.get(task_class)
        if cs is None:
            return g
        return (cs.credit + shrink * g) / (cs.n_ctx + shrink)

    def class_stats(self, rec: MemoryRecord, task_class: str) -> ClassStats:
        return rec.classes.setdefault(task_class, ClassStats())

    # ── corpus statistics used by compression ──────────────────────────────────
    def df_fraction(self, token: str) -> float:
        n = max(1, len(self.records))
        return self._df.get(token, 0) / n

    # ── views ──────────────────────────────────────────────────────────────────
    def state_counts(self) -> dict[str, int]:
        c = Counter(r.state.value for r in self.records.values())
        return {s.value: c.get(s.value, 0) for s in MemState}

    def served_tokens(self, states: Iterable[MemState] = (MemState.ACTIVE, MemState.LOW_VALUE, MemState.COMPRESSED)) -> int:
        allowed = set(states)
        return sum(r.tokens for r in self.records.values() if r.state in allowed)

    def total_tokens(self) -> int:
        return sum(r.original_tokens for r in self.records.values())

    # ── persistence ────────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        def rec_dict(r: MemoryRecord) -> dict:
            return {
                "fact_id": r.fact_id, "text": r.text, "original_text": r.original_text,
                "tokens": r.tokens, "original_tokens": r.original_tokens,
                "created_step": r.created_step, "time": r.time, "confidence": r.confidence,
                "entities": list(r.entities), "attribute_keys": list(r.attribute_keys), "semantic_role": r.semantic_role,
                "state": r.state.value, "valid_from": r.valid_from, "valid_to": r.valid_to,
                "superseded_by": r.superseded_by, "supersedes": r.supersedes,
                "contradictions": r.contradictions, "n_pooled": r.n_pooled,
                "n_retrieved": r.n_retrieved, "n_credited": r.n_credited, "n_wasted": r.n_wasted,
                "n_blamed": r.n_blamed, "n_success_ctx": r.n_success_ctx, "n_fail_ctx": r.n_fail_ctx,
                "last_retrieved_step": r.last_retrieved_step, "last_credit_step": r.last_credit_step,
                "last_updated_step": r.last_updated_step,
                "classes": {k: v.to_dict() for k, v in r.classes.items()},
                "events": [e.to_dict() for e in r.events],
                "last_reason": r.last_reason, "last_utility": r.last_utility,
            }
        return {"step": self.step, "records": [rec_dict(r) for r in self.records.values()]}

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MemoryLedger":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        led = cls()
        led.step = int(raw["step"])
        for d in raw["records"]:
            rec = MemoryRecord(
                fact_id=d["fact_id"], text=d["text"], original_text=d["original_text"],
                tokens=d["tokens"], original_tokens=d["original_tokens"],
                created_step=d["created_step"], time=d["time"], confidence=d["confidence"],
                entities=tuple(d["entities"]), attribute_keys=tuple(d["attribute_keys"]), semantic_role=d.get("semantic_role", "state"),
                state=MemState(d["state"]), valid_from=d["valid_from"], valid_to=d["valid_to"],
                superseded_by=d["superseded_by"], supersedes=list(d["supersedes"]),
                contradictions=d["contradictions"], n_pooled=d["n_pooled"],
                n_retrieved=d["n_retrieved"], n_credited=d["n_credited"], n_wasted=d["n_wasted"],
                n_blamed=d["n_blamed"], n_success_ctx=d["n_success_ctx"], n_fail_ctx=d["n_fail_ctx"],
                last_retrieved_step=d["last_retrieved_step"], last_credit_step=d["last_credit_step"],
                last_updated_step=d["last_updated_step"],
                classes={k: ClassStats(**v) for k, v in d["classes"].items()},
                events=[LifecycleEvent(**e) for e in d["events"]],
                last_reason=d["last_reason"], last_utility=d["last_utility"],
            )
            led.records[rec.fact_id] = rec
            led._df.update(set(content_tokens(rec.original_text)))
        return led
