"""OpenAI-compatible embeddings for ReMem.

    OpenAIEmbedder   same interface as ConceptHashEmbedder (`dim`, `embed_one`, `embed_texts`)

The LLM agent that plays the simulator's reader lives with the simulator: dev/remem_sim/llm_agent.py.
"""

from __future__ import annotations

from typing import Any


class OpenAIEmbedder:
    def __init__(self, client: Any, model: str, dim: int) -> None:
        self.client, self.model, self.dim = client, model, int(dim)

    def embed_texts(self, texts: list[str], *, batch_size: int = 128) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            resp = self.client.embeddings.create(model=self.model, input=texts[i : i + batch_size])
            out.extend(_normalise(list(d.embedding)) for d in resp.data)
        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


def _normalise(v: list[float]) -> list[float]:
    n = sum(x * x for x in v) ** 0.5
    return v if n < 1e-9 else [x / n for x in v]
