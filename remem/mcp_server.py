"""ReMem as an MCP server: long-term memory that learns which memories actually help.

Run (stdio transport, for Claude Code / Claude Desktop / any MCP client):

    pip install "remem[mcp]"            # or: pip install mcp
    remem-mcp                           # or: python -m remem.mcp_server

Environment:
    REMEM_PATH       memory directory (default ~/.remem/default)
    REMEM_EMBEDDER   "hash" (default, offline, weak) | "sentence-transformers" (local model, much better)
    REMEM_ST_MODEL   sentence-transformers model name (default all-MiniLM-L6-v2)
    REMEM_AUTO_FEEDBACK  "1" (default) | "0": infer outcomes for recalls the agent never reported (see below)

Claude Code:   claude mcp add remem -- remem-mcp
Claude Desktop (claude_desktop_config.json):
    {"mcpServers": {"remem": {"command": "remem-mcp", "env": {"REMEM_PATH": "C:/Users/you/.remem/work"}}}}

The feedback loop works best if the agent calls `feedback` after using recalled memories. Agents often forget, so
by default the server also infers outcomes for unreported recalls (implicit feedback):
  * correction: `remember` stores a fact whose entity+attribute matches a served "state" memory but with different
    text -> that recall is recorded as a failure, blaming the served memory (the user corrected it);
  * no complaint: a later `recall` with the same `task` arrives and the earlier one was never reported or corrected
    -> the earlier recall is recorded as a success, with no memory explicitly cited (weak, presence-only credit).
Inferred outcomes are noisier than explicit ones; set REMEM_AUTO_FEEDBACK=0 to disable them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from remem.api import ReMem

_mem: ReMem | None = None
_unreported: dict[str, dict[str, Any]] = {}      # search_id -> {"task", "ids"} for recalls without feedback yet


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

def _auto_enabled() -> bool:
    return os.environ.get("REMEM_AUTO_FEEDBACK", "1").strip().lower() not in ("0", "false", "no", "off")


def _infer(search_id: str, success: bool, used: list[str], reason: str) -> dict[str, Any]:
    _unreported.pop(search_id, None)
    try:
        get_memory().feedback(search_id, success, used_ids=used)
    except KeyError:                                   # already reported or evicted
        return {}
    return {"search_id": search_id, "success": success, ("used" if success else "blamed"): used, "reason": reason}


def memory_remember(text: str, entity: str | None = None, attribute: str | None = None, kind: str = "fact") -> dict[str, Any]:
    role = "state" if kind == "fact" else "event"
    mem = get_memory()
    inferred = []
    if _auto_enabled() and role == "state" and entity and attribute:
        key = ((entity.strip().lower(),), (attribute.strip().lower(),))
        for sid, info in list(_unreported.items()):
            wrong = [i for i in info["ids"] if (f := mem._facts.get(i)) is not None and f.semantic_role == "state"
                     and (f.entities, f.attribute_keys) == key and f.text.strip() != text.strip()]
            if wrong:
                inferred.append(_infer(sid, False, wrong, "corrected by a newer fact"))
    fid = mem.add(text, entity=entity, attribute=attribute, role=role)
    out: dict[str, Any] = {"id": fid, "stored": text}
    inferred = [x for x in inferred if x]
    if inferred:
        out["implicit_feedback"] = inferred
    return out


def memory_recall(query: str, task: str = "default", k: int = 5) -> dict[str, Any]:
    inferred = []
    if _auto_enabled():
        for sid, info in list(_unreported.items()):
            if info["task"] == task:
                inferred.append(_infer(sid, True, [], "no correction before the next recall for this task"))
    res = get_memory().search(query, k=k, task=task).to_dict()
    if res["search_id"]:
        _unreported[res["search_id"]] = {"task": task, "ids": [h["id"] for h in res["items"]]}
        while len(_unreported) > 1000:
            _unreported.pop(next(iter(_unreported)))
    inferred = [x for x in inferred if x]
    if inferred:
        res["implicit_feedback"] = inferred
    return res


def memory_feedback(search_id: str, success: bool, used_ids: list[str] | None = None, answer: str | None = None) -> dict[str, Any]:
    _unreported.pop(search_id, None)
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
        IMPORTANT: after you finish the work that used these memories, call `feedback` with this search_id.
        (If you do not, the server infers an outcome: success if no correction arrives before the next recall
        for the same task, failure if a newer fact corrects a served one.)"""
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
