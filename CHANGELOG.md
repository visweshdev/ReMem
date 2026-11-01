# Changelog

All notable changes are listed here. Versions follow [semantic versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-11-01

### Added
- `remem` command-line tool: `add`, `search`, `list`, `inspect`, `stats`, `dashboard`, `demo`, and an interactive
  `shell` where you mark search results good or bad and watch the ranking change.
- Lint checks (ruff) in CI, tests on Windows, and a separate benchmark smoke job.
- `CONTRIBUTING.md`, issue and pull-request templates, `SECURITY.md`, `py.typed`.
- Install video now also shows the `remem shell`.

### Changed
- matplotlib is now optional (`pip install "remem[plots]"`); the core install needs only numpy and pyyaml.
  The benchmark still runs without it and skips the plots.

## [0.1.0] - 2026-10-27

First public release.

### Added
- Outcome-aware memory layer: per-memory ledger, utility model (heuristic and logistic regression), credit
  assignment, supersession with validity intervals, adaptive retention, token-budgeted selection, similarity guard.
- Python API (`ReMem`: `add`, `search`, `feedback`, `inspect`, `add_conversation`) with on-disk persistence.
- MCP server (`remem-mcp`) with inferred feedback, LangChain retriever, HTML dashboard, FAISS and Chroma backends,
  sentence-transformers embeddings, LLM fact extraction.
- Controlled synthetic benchmark (15 test seeds), RMM-style baseline, real-embedding run, free local-LLM run
  (Ollama), LongMemEval retrieval run, and on-policy retraining experiment, with negative results reported.

[Unreleased]: https://github.com/visweshdev/ReMem/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/visweshdev/ReMem/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/visweshdev/ReMem/releases/tag/v0.1.0
