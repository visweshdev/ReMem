"""Candidate-pool construction with on-disk caching and process-level parallelism.

Pools depend only on (family, seed, backend, pool size, world overrides) and the code of the
backend, so they are cached under results/cache/pools. Delete the directory to rebuild; the
pools are deterministic, so a rebuild reproduces the same files.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from remem.backends import VectorBackend
from remem.eval.runner import Pools, build_pools
from remem.eval.world import LEXICON, World, make_world
from remem.offline import ConceptHashEmbedder

EMBED_DIM = 256


def make_embedder() -> ConceptHashEmbedder:
    return ConceptHashEmbedder(EMBED_DIM, LEXICON)


def _key(family: str, seed: int, backend: str, k: int, overrides: dict[str, Any] | None) -> str:
    from dataclasses import asdict
    from remem.eval.world import FAMILIES
    fam_params = asdict(FAMILIES[family]) if family in FAMILIES else {}
    blob = json.dumps([family, seed, backend, k, overrides or {}, EMBED_DIM, fam_params], sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def pool_path(cache_dir: Path, family: str, seed: int, backend: str, k: int, overrides: dict[str, Any] | None) -> Path:
    return Path(cache_dir) / f"{family}_{seed}_{backend}_{k}_{_key(family, seed, backend, k, overrides)}.pkl"


def _build(args) -> str:
    family, seed, backend, k, overrides, cache_dir = args
    path = pool_path(cache_dir, family, seed, backend, k, overrides)
    if path.exists():
        return str(path)
    world = make_world(family, seed, overrides)
    emb = make_embedder()
    if backend != "vector":
        raise ValueError(f"unknown backend {backend!r}; only 'vector' is built in (implement remem.backends.base.Backend for others)")
    be = VectorBackend(emb)
    pools = build_pools(world, be, k)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(pools, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return str(path)


def ensure_pools(jobs: list[tuple[str, int, str, int, dict[str, Any] | None]], cache_dir: str | Path,
                 workers: int = 4) -> None:
    """Build every missing pool file for (family, seed, backend, k, overrides) jobs."""
    todo = [(f, s, b, k, o, str(cache_dir)) for f, s, b, k, o in jobs
            if not pool_path(Path(cache_dir), f, s, b, k, o).exists()]
    if not todo:
        return
    if workers <= 1 or len(todo) == 1:
        for j in todo:
            _build(j)
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for _ in ex.map(_build, todo):
            pass


def load_pools(cache_dir: str | Path, family: str, seed: int, backend: str, k: int,
               overrides: dict[str, Any] | None = None) -> Pools:
    path = pool_path(Path(cache_dir), family, seed, backend, k, overrides)
    if not path.exists():
        _build((family, seed, backend, k, overrides, str(cache_dir)))
    with open(path, "rb") as fh:
        return pickle.load(fh)
