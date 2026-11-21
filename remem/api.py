"""High-level API: a persistent, outcome-aware memory in a few lines.

    from remem import ReMem

    mem = ReMem("~/.remem/project")                        # directory; created and reloaded automatically
    mem.add("The billing service exposes API version v2.", entity="billing", attribute="api_version")
    hits = mem.search("Which API version does billing expose?")
    for h in hits.items:
        print(h.id, h.text, h.score)
    mem.feedback(hits.search_id, success=True, used_ids=[hits.items[0].id])

State on disk (plain JSON, human-readable):
    <path>/facts.jsonl     every stored fact (append-only; never deleted)
    <path>/ledger.json     per-memory state, statistics, validity and lifecycle events
    <path>/config.json     the SystemConfig the store was created with

Searches awaiting feedback are kept in memory only: give feedback in the same process that searched.

Status: fact replacement (`add(update=...)`) was evaluated on real data (LongMemEval, README). Learning from feedback
(`feedback`) is EXPERIMENTAL and untested on real data.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from remem.backends import StoredFact, VectorBackend
from remem.config import (
    ContradictionConfig,
    RetentionConfig,
    SelectionConfig,
    SystemConfig,
    UtilityConfig,
    from_dict,
    to_dict,
)
from remem.ledger import MemoryLedger
from remem.offline import ConceptHashEmbedder
from remem.system import IngestedFact, ReMemSystem
from remem.types import AgentTrace, Context, MemState, Outcome


def default_config(top_k: int = 5, budget_tokens: int | None = None) -> SystemConfig:
    """Sensible library defaults: transparent heuristic utility (no trained model needed), feedback on,
    supersession on, adaptive retention, cost-aware selection with the similarity guard."""
    return SystemConfig(
        name="remem",
        backend="vector",
        ranker="heuristic",
        outcome_feedback=True,
        pool_size=30,
        utility=UtilityConfig(kind="heuristic", explore_c=0.5),
        # two "state" facts with the same (entity, attribute) and different values: the newer replaces the older.
        # Store multi-valued things (likes, dependencies) with role="preference" or "note" to keep them all.
        contradiction=ContradictionConfig(enabled=True, conf_half_life_days=120.0, exclusive_roles=("state",),
                                          learn_schema=True),
        retention=RetentionConfig(mode="adaptive"),
        selection=SelectionConfig(mode="cost_aware", top_k=top_k, budget_tokens=budget_tokens, p_floor=0.12, guard_top=3,
                                  min_sim=0.1),
    )


@dataclass
class Hit:
    id: str
    text: str
    score: float          # ReMem's estimate that this memory will help (0..1)
    similarity: float     # raw cosine similarity to the query
    state: str
    why: str              # top utility components behind the score
    source: str | None = None   # the original text this fact was extracted from (add_conversation / add(source=...))
    time: float | None = None   # when the fact was recorded

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    search_id: str
    query: str
    items: list[Hit] = field(default_factory=list)
    tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"search_id": self.search_id, "query": self.query, "tokens": self.tokens,
                "items": [h.to_dict() for h in self.items]}


_ADDITIVE = re.compile(r"\b(also|another|additionally|as well|too|in addition|one of|among)\b", re.I)


def _warn_if_list_like(text: str, new: StoredFact, existing) -> None:
    """Two "state" facts with the same entity+attribute replace each other. If the new one reads like an
    additional value ("also uses Redis") rather than an update, the older value would be lost: warn."""
    if not _ADDITIVE.search(text):
        return
    if any(f.semantic_role == "state" and f.entities == new.entities and f.attribute_keys == new.attribute_keys
           for f in existing):
        warnings.warn(f"{text!r} looks like an additional value for {new.entities[0]}.{new.attribute_keys[0]}, but "
                      "role='state' will supersede the existing fact. Use role='preference' or 'note' for "
                      "multi-valued attributes.", UserWarning, stacklevel=3)


class ReMem:
    """Persistent outcome-aware memory over a built-in vector index.

    Thread-safe for use from one process (e.g. an MCP server). Not safe for several processes writing the
    same directory at once.
    """

    def __init__(self, path: str | Path | None = None, *, embedder: Any = None, backend: Any = None,
                 config: SystemConfig | None = None, autosave: bool = True, update_checker: Any = None) -> None:
        """`update_checker(old_text, new_text) -> bool` confirms that a new fact replaces an older one on the same
        entity+attribute (e.g. remem.extract.LLMUpdateChecker). Facts added with update="check" (the default) only
        retire older facts when it says yes; without a checker they never retire anything."""
        self.path = Path(path).expanduser() if path is not None else None
        self.autosave = autosave and self.path is not None
        self._lock = threading.RLock()
        self.embedder = embedder or ConceptHashEmbedder(dim=256)
        if self.path is not None and (self.path / "config.json").exists() and config is None:
            config = from_dict(SystemConfig, json.loads((self.path / "config.json").read_text(encoding="utf-8")))
        self.config = config or default_config()
        # any object with ingest(facts) / candidates(query, k): VectorBackend, FaissBackend, ChromaBackend, ...
        self.backend = backend if backend is not None else VectorBackend(self.embedder)
        self.system = ReMemSystem(self.config, self.embedder)
        self.update_checker = update_checker
        if update_checker is not None:
            self.system.detector.confirm = lambda old, new: bool(update_checker(old.original_text, new.original_text))
        self._facts: dict[str, StoredFact] = {}
        self._pending: dict[str, Context] = {}
        self._task_counter = 0
        if self.path is not None:
            self.path.mkdir(parents=True, exist_ok=True)
            self._load()

    # ── writing ────────────────────────────────────────────────────────────────
    def add(self, text: str, *, entity: str | None = None, attribute: str | None = None, role: str = "state",
            when: float | None = None, fact_id: str | None = None, update: str = "check",
            source: str | None = None) -> str:
        """Store a fact. `entity`/`attribute` (e.g. "billing", "api_version") let a newer fact retire an older one
        with the same key and a different value. Whether it does is set by `update`:
            "check"   (default) only if the update checker confirms it is an update; without a checker, keep both
            "replace" yes: you know this fact updates the older one (e.g. "the server moved to west-2")
            "keep"    never
        Same key alone is not trusted by default: on real data, LLM-extracted keys are broad, and replacing on key
        match retired correct facts (README, Results). `role` other than "state" never supersedes anything.
        `source`: the original text the fact came from, returned with search hits. Returns the fact id."""
        if update not in ("check", "replace", "keep"):
            raise ValueError(f"update must be 'check', 'replace' or 'keep', not {update!r}")
        text = (text or "").strip()
        if not text:
            raise ValueError("cannot remember an empty text")
        with self._lock:
            fid = fact_id or f"fact_{uuid.uuid4().hex[:12]}"
            if fid in self._facts:
                raise ValueError(f"fact id {fid!r} already exists")
            fact = StoredFact(fid, text, float(when if when is not None else time.time()),
                              (entity.strip().lower(),) if entity else (),
                              (attribute.strip().lower(),) if attribute else (), semantic_role=role,
                              update_policy=update if role == "state" else "keep", source=source)
            if role == "state" and entity and attribute and update == "replace":
                _warn_if_list_like(text, fact, self._facts.values())
            self._ingest(fact)
            if self.path is not None:
                with open(self.path / "facts.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(fact)) + "\n")
            self._maybe_save()
            return fid

    def add_conversation(self, text: str, *, extractor: Any = None, when: float | None = None) -> list[str]:
        """Extract facts from raw conversation text and store them. `extractor` is any object with
        `extract(text) -> list[ExtractedFact]` (e.g. remem.extract.LLMFactExtractor); defaults to the
        dependency-free SentenceSplitter, which stores each sentence as a note."""
        from remem.extract import SentenceSplitter
        ex = extractor or SentenceSplitter()
        return [self.add(f.text, entity=f.entity, attribute=f.attribute, role=f.role, when=when, update="check",
                         source=text)
                for f in ex.extract(text)]

    def _ingest(self, fact: StoredFact) -> None:
        self._facts[fact.fact_id] = fact
        self.backend.ingest([fact])
        self.system.ingest(IngestedFact(fact.fact_id, fact.text, fact.time, fact.entities, fact.attribute_keys,
                                        fact.confidence, fact.semantic_role, fact.update_policy))

    # ── reading ────────────────────────────────────────────────────────────────
    def search(self, query: str, *, k: int | None = None, task: str = "default", as_of: float | None = None,
               budget_tokens: int | None = None, now: float | None = None) -> SearchResult:
        """Retrieve the memories most likely to help with `query`. `task` groups similar requests so that
        usefulness is learned per kind of task; `as_of` (unix time) asks about the past. `now` (unix time,
        default: the clock) is the moment recency is measured from, e.g. when replaying old conversations; memories
        recorded after `now` are not yet valid and are not returned."""
        with self._lock:
            if not self._facts:
                return SearchResult(search_id="", query=query)
            cfg = self.config
            if k is not None or budget_tokens is not None:
                sel = cfg.selection
                from dataclasses import replace
                sel = replace(sel, top_k=k or sel.top_k, budget_tokens=budget_tokens if budget_tokens is not None else sel.budget_tokens)
                self.system.cfg = replace(cfg, selection=sel)
            try:
                pool, ms = self.backend.candidates(query, k=cfg.pool_size)
                self._task_counter += 1
                ctx = self.system.retrieve(self._task_counter, query, pool, task_class=task, now=time.time() if now is None else float(now),
                                           as_of=as_of, pool_time_ms=ms)
            finally:
                self.system.cfg = cfg
            sid = uuid.uuid4().hex[:12]
            self._pending[sid] = ctx
            if len(self._pending) > 1000:                 # bound memory if feedback is never given
                self._pending.pop(next(iter(self._pending)))
            items = []
            for s in ctx.selected:
                f = self._facts.get(s.cand.fact_id)
                items.append(Hit(s.cand.fact_id, s.text, round(s.prob, 4), round(s.cand.sim, 4), s.state, s.reason,
                                 f.source if f else None, f.time if f else None))
            self._maybe_save()
            return SearchResult(sid, query, items, ctx.tokens_used)

    # ── feedback ───────────────────────────────────────────────────────────────
    def feedback(self, search_id: str, success: bool, *, used_ids: list[str] | None = None,
                 answer: str | None = None) -> dict[str, float]:
        """Report how the task that used `search_id` went. `used_ids`: the memories you actually relied on
        (the strongest signal). `answer`: optional final answer text; memories containing it count as used.
        Returns the credit each served memory received.

        EXPERIMENTAL: learning from feedback has not been tested on real data (no public benchmark repeats
        questions with outcomes). It is exercised only by unit tests and the development simulator in dev/."""
        with self._lock:
            ctx = self._pending.pop(search_id, None)
            if ctx is None:
                raise KeyError(f"unknown or already-used search_id {search_id!r}")
            served = {s.cand.fact_id for s in ctx.selected}
            cited = [i for i in (used_ids or []) if i in served]
            answers = {"answer": answer} if answer else {}
            atts = self.system.observe(ctx, AgentTrace(answer=answers, cited_ids=cited, tool_args={}),
                                       Outcome(success=bool(success)))
            self._maybe_save()
            return {fid: round(a.credit, 3) for fid, a in atts.items()}

    # ── inspection ─────────────────────────────────────────────────────────────
    def inspect(self, fact_id: str) -> dict[str, Any]:
        with self._lock:
            rec = self.system.ledger.get(fact_id)
            if rec is None:
                raise KeyError(f"unknown memory {fact_id!r}")
            from remem.viz.ledger_cli import lifecycle
            led = self.system.ledger
            return {
                "id": rec.fact_id, "text": rec.original_text, "state": rec.state.value,
                "contribution_rate": round(led.contrib_rate(rec), 4), "shown": rec.n_retrieved,
                "credited": round(rec.n_credited, 3), "wasted": rec.n_wasted, "blamed": rec.n_blamed,
                "superseded_by": rec.superseded_by, "supersedes": rec.supersedes,
                "valid_from": rec.valid_from, "valid_to": rec.valid_to,
                "last_reason": rec.last_reason, "lifecycle": lifecycle(rec, led),
                "recent_events": [e.to_dict() for e in rec.events[-10:]],
            }

    def stats(self) -> dict[str, Any]:
        with self._lock:
            led = self.system.ledger
            return {"memories": len(led.records), "states": led.state_counts(),
                    "served_tokens": led.served_tokens(), "pending_feedback": len(self._pending),
                    "path": str(self.path) if self.path else None}

    def list(self, state: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            recs = [r for r in self.system.ledger.records.values() if state is None or r.state.value == state.upper()]
            recs.sort(key=lambda r: -r.last_updated_step)
            return [{"id": r.fact_id, "state": r.state.value, "text": r.original_text, "shown": r.n_retrieved,
                     "credited": round(r.n_credited, 3)} for r in recs[:limit]]

    # ── persistence ────────────────────────────────────────────────────────────
    def save(self) -> None:
        if self.path is None:
            return
        with self._lock:
            self.system.ledger.save(self.path / "ledger.json.tmp")
            (self.path / "ledger.json.tmp").replace(self.path / "ledger.json")
            (self.path / "config.json").write_text(json.dumps(to_dict(self.config), indent=1), encoding="utf-8")

    def _maybe_save(self) -> None:
        if self.autosave:
            self.save()

    def _load(self) -> None:
        facts_path = self.path / "facts.jsonl"
        if facts_path.exists():
            for line in facts_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    d["entities"], d["attribute_keys"] = tuple(d["entities"]), tuple(d["attribute_keys"])
                    self._ingest(StoredFact(**d))
        ledger_path = self.path / "ledger.json"
        if ledger_path.exists():
            # replaying facts rebuilt keys and supersessions; the saved ledger restores what was learned from use
            saved = MemoryLedger.load(ledger_path)
            live = self.system.ledger
            for fid, rec in saved.records.items():
                if fid in live.records:
                    live.records[fid] = rec
            live.step = max(live.step, saved.step)
            self._task_counter = live.step
        self.save()


__all__ = ["ReMem", "SearchResult", "Hit", "default_config", "MemState"]
