"""ReMem as an MCP server: long-term memory that learns which memories actually help.

Run (stdio transport, for Claude Code / Claude Desktop / any MCP client):

    pip install "remem[mcp]"            # or: pip install mcp
    remem-mcp                           # or: python -m remem.mcp_server

Environment:
    REMEM_PATH       memory directory (default ~/.remem/default)
    REMEM_EMBEDDER   "hash" (default, offline, weak) | "sentence-transformers" (local model, much better)
    REMEM_ST_MODEL   sentence-transformers model name (default all-MiniLM-L6-v2)

Claude Code:   claude mcp add remem -- remem-mcp
Claude Desktop (claude_desktop_config.json):
    {"mcpServers": {"remem": {"command": "remem-mcp", "env": {"REMEM_PATH": "C:/Users/you/.remem/work"}}}}

The feedback loop only works if the agent calls `memory_feedback` after using recalled memories. The tool
descriptions below say so explicitly; agents that never report outcomes get plain similarity search.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from remem.api import ReMem

_mem: ReMem | None = None


def get_memory() -> ReMem:
    global _mem
    if _mem is None:
        path = Path(os.environ.get("REMEM_PATH", "~/.remem/default")).expanduser()
        _mem = ReMem(path, embedder=_make_embedder())
    return _mem


def _make_embedder():
    kind = os.environ.get("REMEM_EMBEDDER", "hash").lower()
    if kind in ("st", "sentence-transformers", "sentence_transformers"):
        from remem.embedders import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(os.environ.get("REMEM_ST_MODEL", "all-MiniLM-L6-v2"))
    from remem.offline import ConceptHashEmbedder
    return ConceptHashEmbedder(dim=256)


# ── tool implementations (plain functions so they can be unit-tested without an MCP client) ──

def memory_remember(text: str, entity: str | None = None, attribute: str | None = None, kind: str = "fact") -> dict[str, Any]:
    role = "state" if kind == "fact" else "event"
    fid = get_memory().add(text, entity=entity, attribute=attribute, role=role)
    return {"id": fid, "stored": text}


def memory_recall(query: str, task: str = "default", k: int = 5) -> dict[str, Any]:
    return get_memory().search(query, k=k, task=task).to_dict()


def memory_feedback(search_id: str, success: bool, used_ids: list[str] | None = None, answer: str | None = None) -> dict[str, Any]:
    credits = get_memory().feedback(search_id, success, used_ids=used_ids, answer=answer)
    return {"recorded": True, "credit": credits}


def memory_inspect(memory_id: str) -> dict[str, Any]:
    return get_memory().inspect(memory_id)


def memory_list(state: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    return get_memory().list(state=state, limit=limit)


def memory_stats() -> dict[str, Any]:
    return get_memory().stats()


def build_server():
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("remem")

    @mcp.tool()
    def remember(text: str, entity: str | None = None, attribute: str | None = None, kind: str = "fact") -> dict:
        """Store a memory for future sessions. Write one self-contained fact per call
        (e.g. "The billing service exposes API version v2."). Optional `entity` + `attribute`
        (e.g. "billing", "api_version") let ReMem retire an older fact automatically when a newer one with a
        different value arrives. Use kind="note" for observations that are not statements of current fact."""
        return memory_remember(text, entity, attribute, kind)

    @mcp.tool()
    def recall(query: str, task: str = "default", k: int = 5) -> dict:
        """Retrieve memories likely to help with `query`. Returns a `search_id` and ranked items.
        `task` is a short label for the kind of work (e.g. "deploy", "debug") so usefulness is learned per task.
        IMPORTANT: after you finish the work that used these memories, call `feedback` with this search_id."""
        return memory_recall(query, task, k)

    @mcp.tool()
    def feedback(search_id: str, success: bool, used_ids: list[str] | None = None, answer: str | None = None) -> dict:
        """Report the outcome of the task that used a `recall` result. Call this once per recall, after acting:
        success=true if the task succeeded, false if it failed or the memories were wrong/outdated.
        `used_ids`: ids of the memories you actually relied on (most important). This is how ReMem learns
        which memories help; without it ranking stays similarity-only."""
        return memory_feedback(search_id, success, used_ids, answer)

    @mcp.tool()
    def inspect(memory_id: str) -> dict:
        """Show one memory's state, usefulness statistics, supersession links and lifecycle history."""
        return memory_inspect(memory_id)

    @mcp.tool()
    def list_memories(state: str | None = None, limit: int = 20) -> list:
        """List recently updated memories, optionally filtered by state
        (ACTIVE, LOW_VALUE, COMPRESSED, ARCHIVED, INVALIDATED)."""
        return memory_list(state, limit)

    @mcp.tool()
    def stats() -> dict:
        """Counts of memories by state and where the store lives on disk."""
        return memory_stats()

    return mcp


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
