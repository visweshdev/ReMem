"""Embedders. All share the interface used across ReMem: `dim`, `embed_one(text)`, `embed_texts(texts)`,
returning L2-normalised lists of floats.

    ConceptHashEmbedder        remem.offline   offline, deterministic, weak (tests and the synthetic benchmark)
    SentenceTransformerEmbedder (here)        local neural model; needs `pip install sentence-transformers`
    OpenAIEmbedder             remem.llm       any OpenAI-compatible embeddings endpoint
"""

from __future__ import annotations


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "all-MiniLM-L6-v2", device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ImportError("pip install sentence-transformers  (or: pip install remem[embeddings])") from exc
        self.model_name = model
        self._model = SentenceTransformer(model, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self._cache: dict[str, list[float]] = {}

    def warm(self, texts: list[str], *, batch_size: int = 64) -> None:
        """Batch-embed texts not yet cached (much faster than one call per text)."""
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if not todo:
            return
        vecs = self._model.encode(todo, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        for t, v in zip(todo, vecs):
            self._cache[t] = [float(x) for x in v]
        if len(self._cache) > 500_000:          # bound memory on very long runs
            for k in list(self._cache)[: len(self._cache) // 2]:
                del self._cache[k]

    def embed_one(self, text: str) -> list[float]:
        v = self._cache.get(text)
        if v is None:
            self.warm([text])
            v = self._cache[text]
        return v

    def embed_texts(self, texts: list[str], *, batch_size: int = 64) -> list[list[float]]:
        self.warm(list(texts), batch_size=batch_size)
        return [self._cache[t] if t in self._cache else self.embed_one(t) for t in texts]
