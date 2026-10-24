"""Deterministic, network-free text utilities for the offline benchmark.

`ConceptHashEmbedder` is a signed feature-hashing embedder with an optional synonym lexicon. It lets the
benchmark and the unit tests run without a model server and gives bit-reproducible similarities.
It is NOT a substitute for a trained embedding model: retrieval quality under it says nothing about
quality under real embeddings (see README "Limitations"). `remem.llm.OpenAIEmbedder` is a drop-in
replacement with the same interface.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """a an the is are was were be been being of to in on at for by with and or as it its this that
    these those from into than then so if but not no does do did has have had will would can could
    should what which who whom whose when where how why""".split()
)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


_TOKEN_COUNTER = None   # optional real tokenizer, see set_tokenizer()


def set_tokenizer(name: str | None) -> None:
    """Choose how tokens are counted for budgets and cost accounting.

    None / "proxy"        words + punctuation (default; deterministic, no dependency)
    "tiktoken"            tiktoken cl100k_base (pip install tiktoken)
    "tiktoken:<encoding>" a specific tiktoken encoding, e.g. "tiktoken:o200k_base"

    Also read from the REMEM_TOKENIZER environment variable at import time. Set it before storing memories:
    token counts are computed when a memory is stored.
    """
    global _TOKEN_COUNTER
    if not name or name == "proxy":
        _TOKEN_COUNTER = None
        return
    if name.startswith("tiktoken"):
        import tiktoken
        enc = tiktoken.get_encoding(name.split(":", 1)[1] if ":" in name else "cl100k_base")
        _TOKEN_COUNTER = lambda text: len(enc.encode(text))  # noqa: E731
        return
    raise ValueError(f"unknown tokenizer {name!r}; use 'proxy' or 'tiktoken[:encoding]'")


def count_tokens(text: str) -> int:
    """Token count used for budgets and cost accounting everywhere in remem.

    Default is a cheap deterministic proxy (words + punctuation): relative costs are right, absolute numbers
    are not. Call set_tokenizer("tiktoken") for real BPE counts.
    """
    if _TOKEN_COUNTER is not None:
        return _TOKEN_COUNTER(text)
    return len(re.findall(r"\w+|[^\w\s]", text))


if os.environ.get("REMEM_TOKENIZER"):
    set_tokenizer(os.environ["REMEM_TOKENIZER"])


def _stable_hash(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


class ConceptHashEmbedder:
    """Signed feature-hashing embedder with an optional synonym lexicon.

    `lexicon` maps a surface word to a concept id, so that e.g. "hosted", "deploys"
    and "runs" can share a dimension. Unigrams get weight 1, adjacent-pair bigrams
    0.5. Output is L2-normalised. Entirely deterministic across processes.
    """

    def __init__(self, dim: int = 256, lexicon: dict[str, str] | None = None) -> None:
        self.dim = int(dim)
        self.lexicon = {k.lower(): v.lower() for k, v in (lexicon or {}).items()}

    def embed_one(self, text: str) -> list[float]:
        toks = [self.lexicon.get(t, t) for t in content_tokens(text)]
        vec = [0.0] * self.dim
        counts = Counter(toks)
        for tok, c in counts.items():
            self._add(vec, tok, 1.0 + math.log(c))
        for a, b in zip(toks, toks[1:]):
            self._add(vec, f"{a}|{b}", 0.5)
        norm = math.sqrt(sum(x * x for x in vec))
        if norm < 1e-9:
            vec[0] = 1.0
            return vec
        return [x / norm for x in vec]

    def _add(self, vec: list[float], key: str, w: float) -> None:
        h = _stable_hash(key)
        sign = 1.0 if (h >> 63) & 1 else -1.0
        vec[h % self.dim] += sign * w

    def embed_texts(self, texts: list[str], *, batch_size: int = 256) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]

