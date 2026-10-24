"""Vector-store backends: FAISS (exact inner product) and Chroma (HNSW, cosine).

Both implement the `Backend` protocol (`ingest(facts)`, `candidates(query, k)`), so they drop into
`ReMem(..., backend=...)`, `ReMemSystem` pools and the benchmark. Similarity in the returned candidates is always
recomputed with the shared embedder, so every backend reports comparable `sim` values.

    pip install faiss-cpu      # FaissBackend
    pip install chromadb       # ChromaBackend
"""

from __future__ import annotations

import time
import uuid

import numpy as np

from remem.backends.base import EmbeddingStore, StoredFact
from remem.types import Candidate


def _candidate(f: StoredFact, sim: float, rank: int, source: str) -> Candidate:
    return Candidate(fact_id=f.fact_id, text=f.text, tokens=f.tokens, time=f.time, sim=float(sim), base_rank=rank,
                     confidence=f.confidence, entities=f.entities, attribute_keys=f.attribute_keys, source=source)


class FaissBackend:
    name = "faiss"

    def __init__(self, embedder) -> None:
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install faiss-cpu") from exc
        self._faiss = faiss
        self.embedder = embedder
        self.store = EmbeddingStore(embedder)
        self.index = faiss.IndexFlatIP(embedder.dim)
        self.facts: list[StoredFact] = []

    def ingest(self, facts: list[StoredFact]) -> None:
        if not facts:
            return
        vecs = np.asarray([self.store.add(f) for f in facts], dtype=np.float32)
        self.index.add(vecs)
        self.facts.extend(facts)

    def candidates(self, query: str, *, k: int) -> tuple[list[Candidate], float]:
        t0 = time.perf_counter()
        if not self.facts:
            return [], 0.0
        q = np.asarray([self.embedder.embed_one(query)], dtype=np.float32)
        sims, idx = self.index.search(q, min(k, len(self.facts)))
        out = [_candidate(self.facts[i], s, r, self.name) for r, (s, i) in enumerate(zip(sims[0], idx[0])) if i >= 0]
        return out, (time.perf_counter() - t0) * 1000.0


class ChromaBackend:
    """Chroma collection with cosine HNSW. In-memory by default; pass `path` for a persistent Chroma store
    (then do not also re-ingest the same facts on every start)."""

    name = "chroma"

    def __init__(self, embedder, *, path: str | None = None, collection: str | None = None) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install chromadb") from exc
        self.embedder = embedder
        self.store = EmbeddingStore(embedder)
        client = chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        self.col = client.get_or_create_collection(collection or f"remem_{uuid.uuid4().hex[:8]}",
                                                   metadata={"hnsw:space": "cosine"})
        self.facts: dict[str, StoredFact] = {}

    def ingest(self, facts: list[StoredFact]) -> None:
        new = [f for f in facts if f.fact_id not in self.facts]
        if not new:
            return
        self.col.add(ids=[f.fact_id for f in new], embeddings=[self.store.add(f) for f in new],
                     documents=[f.text for f in new])
        self.facts.update({f.fact_id: f for f in new})

    def candidates(self, query: str, *, k: int) -> tuple[list[Candidate], float]:
        t0 = time.perf_counter()
        if not self.facts:
            return [], 0.0
        q = self.embedder.embed_one(query)
        res = self.col.query(query_embeddings=[q], n_results=min(k, len(self.facts)))
        out = []
        for r, fid in enumerate(res["ids"][0]):
            f = self.facts[fid]
            v = self.store.vecs[fid]
            out.append(_candidate(f, sum(a * b for a, b in zip(q, v)), r, self.name))
        return out, (time.perf_counter() - t0) * 1000.0
