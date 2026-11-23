# ReMem as an MCP memory server

Run this from a clone of ReMem. The embedding model runs locally after its initial download. The MCP integration is tested with
both major versions of the Python SDK used by this release. [Integration checks](../.github/workflows/tests.yml)

```bash
pip install -e ".[mcp,embeddings]"
```

## Claude Code

```bash
claude mcp add remem --env REMEM_EMBEDDER=sentence-transformers -- remem-mcp
```

## Claude Desktop

Add this to `claude_desktop_config.json` (Settings → Developer → Edit Config), then restart Claude Desktop:

```json
{
  "mcpServers": {
    "remem": {
      "command": "remem-mcp",
      "env": {
        "REMEM_PATH": "C:/Users/you/.remem/personal",
        "REMEM_EMBEDDER": "sentence-transformers"
      }
    }
  }
}
```

If Claude Desktop cannot locate `remem-mcp`, set `command` to the absolute path of the Python interpreter where you installed ReMem
and set `args` to `["-m", "remem.mcp_server"]`. For a Windows virtual environment this interpreter is `.venv/Scripts/python.exe`.

## What the model gets

| Tool | What it does |
|---|---|
| `remember` | store a fact with optional `entity` and `attribute`; set `is_update=true` to declare a replacement |
| `recall` | ranked memories plus a `search_id` |
| `feedback` | report whether the task that used a recall succeeded, and which memories helped |
| `inspect`, `list_memories`, `stats` | look inside the store |

For example, if the server moves from `east-1` to `west-2`, call `remember` with `entity="server"`, `attribute="region"`,
and `is_update=true`. Leave `is_update` at its default `false` when the new fact adds information.

With `REMEM_AUTO_FEEDBACK=1` (the default), the server infers outcomes for unreported recalls. A correction through `remember`
with `is_update=true` blames a served state memory with the same entity and attribute when its text differs. A later `recall` for
the same task marks the earlier unreported recall as a weak success. Set `REMEM_AUTO_FEEDBACK=0` to disable inference.

The benchmark tested explicit feedback from reference-graded answers. The MCP server's inferred feedback remains unevaluated.
[Feedback experiment](../docs/FEEDBACK_ABLATION.md)

A prompt that works well: *"Use the remem tools. Recall before answering questions about my projects, remember new
facts I tell you, and call feedback after each answer."*
