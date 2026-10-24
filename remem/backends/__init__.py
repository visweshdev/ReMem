from remem.backends.base import Backend, StoredFact
from remem.backends.vector import VectorBackend


def __getattr__(name):  # optional backends import their libraries lazily
    if name in ("FaissBackend", "ChromaBackend"):
        from remem.backends import stores
        return getattr(stores, name)
    raise AttributeError(name)


__all__ = ["Backend", "StoredFact", "VectorBackend", "FaissBackend", "ChromaBackend"]
