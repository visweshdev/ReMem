# ReMem as an MCP memory server

Install it once (free, everything runs locally):

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

If `remem-mcp` is not on the PATH Claude Desktop sees, use `"command": "python", "args": ["-m", "remem.mcp_server"]`.

## What the model gets

| Tool | What it does |
|---|---|
| `remember` | store a fact with optional `entity` and `attribute`; set `is_update=true` to declare a replacement |
| `recall` | ranked memories plus a `search_id` |
| `feedback` | report whether the task that used a recall succeeded, and which memories helped |
| `inspect`, `list_memories`, `stats` | look inside the store |

For example, if the server moves from `east-1` to `west-2`, call `remember` with `entity="server"`, `attribute="region"`,
and `is_update=true`. Leave `is_update` at its default `false` when the new fact adds information.

If the model forgets to call `feedback`, the server infers an outcome (`REMEM_AUTO_FEEDBACK=1`, the default).
A correction through `remember` with `is_update=true` counts as a failure for the memory it replaced. A later `recall` for
the same task with no correction in between counts as a weak success. Set `REMEM_AUTO_FEEDBACK=0` to turn that off.

A prompt that works well: *"Use the remem tools. Recall before answering questions about my projects, remember new
facts I tell you, and call feedback after each answer."*
