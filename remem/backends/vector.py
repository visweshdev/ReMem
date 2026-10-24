"""Baseline A: naive flat vector retrieval (exact cosine over every stored fact)."""

from __future__ import annotations

import time

import numpy as np

from remem.backends.base import EmbeddingStore, StoredFact
from remem.offline import ConceptHashEmbedder
from remem.types import Candidate


class VectorBackend:
    name = "vector"

    def __init__(self, embedder: ConceptHashEmbedder) -> None:
        self.embedder = embedder
        self.store = EmbeddingStore(embedder)
        self.facts: list[StoredFact] = []
        self._mat: np.ndarray | None = None

    def ingest(self, facts: list[StoredFact]) -> None:
        for f in facts:
            self.store.add(f)
            self.facts.append(f)
        self._mat = None

    def candidates(self, query: str, *, k: int) -> tuple[list[Candidate], float]:
        t0 = time.perf_counter()
        if self._mat is None:
            self._mat = np.asarray([self.store.vecs[f.fact_id] for f in self.facts], dtype=np.float32)
        q = np.asarray(self.embedder.embed_one(query), dtype=np.float32)
        sims = self._mat @ q
        order = sorted(range(len(self.facts)), key=lambda i: (-float(sims[i]), self.facts[i].fact_id))[:k]
        out = [
            Candidate(
                fact_id=self.facts[i].fact_id, text=self.facts[i].text, tokens=self.facts[i].tokens,
                time=self.facts[i].time, sim=float(sims[i]), base_rank=r,
                confidence=self.facts[i].confidence, entities=self.facts[i].entities,
                attribute_keys=self.facts[i].attribute_keys, source=self.name,
            )
            for r, i in enumerate(order)
        ]
        return out, (time.perf_counter() - t0) * 1000.0
