"""Embedders. All share the interface used across ReMem: `dim`, `embed_one(text)`, `embed_texts(texts)`,
returning L2-normalised lists of floats.

    ConceptHashEmbedder        remem.offline   offline, deterministic, weak (tests and the synthetic benchmark)
    SentenceTransformerEmbedder (here)        local neural model; needs `pip install sentence-transformers`
    OpenAIEmbedder             remem.llm       any OpenAI-compatible embeddings endpoint
"""

from __future__ import annotations

from functools import lru_cache


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "all-MiniLM-L6-v2", device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ImportError("pip install sentence-transformers  (or: pip install remem[embeddings])") from exc
        self.model_name = model
        self._model = SentenceTransformer(model, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self._cached = lru_cache(maxsize=50_000)(self._embed_uncached)

    def _embed_uncached(self, text: str) -> tuple[float, ...]:
        v = self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        return tuple(float(x) for x in v)

    def embed_one(self, text: str) -> list[float]:
        return list(self._cached(text))

    def embed_texts(self, texts: list[str], *, batch_size: int = 64) -> list[list[float]]:
        unique = list(dict.fromkeys(texts))
        vecs = self._model.encode(unique, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        table = {t: [float(x) for x in v] for t, v in zip(unique, vecs)}
        return [table[t] for t in texts]
