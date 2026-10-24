"""Candidate-generation backends.

A backend owns *storage and indexing* and returns an over-fetched candidate pool for a query.
ReMem never writes into a backend; it only re-ranks / filters what the backend returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from remem.offline import ConceptHashEmbedder, count_tokens
from remem.types import Candidate


@dataclass
class StoredFact:
    fact_id: str
    text: str
    time: float
    entities: tuple[str, ...] = ()
    attribute_keys: tuple[str, ...] = ()
    confidence: float = 1.0
    origin: str = "user"
    session_id: str = "s0"
    semantic_role: str = "state"

    @property
    def tokens(self) -> int:
        return count_tokens(self.text)


class Backend(Protocol):
    name: str

    def ingest(self, facts: list[StoredFact]) -> None: ...

    def candidates(self, query: str, *, k: int) -> tuple[list[Candidate], float]:
        """Return (candidates ordered by the backend's own ranking, retrieval latency in ms)."""
        ...


@dataclass
class EmbeddingStore:
    """Shared fact-embedding cache so every backend scores `sim` with the same function."""

    embedder: ConceptHashEmbedder
    vecs: dict[str, list[float]] = field(default_factory=dict)

    def add(self, fact: StoredFact) -> list[float]:
        v = self.vecs.get(fact.fact_id)
        if v is None:
            v = self.embedder.embed_one(fact.text)
            self.vecs[fact.fact_id] = v
        return v
