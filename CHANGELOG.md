# Changelog

All notable changes are listed here. Versions follow [semantic versioning](https://semver.org/).

## [Unreleased]

### Added
- Real-data evaluation: `experiments/run_real_knowledge_update.py` runs LongMemEval's 72 knowledge-update questions end to end
  with a local LLM (fact extraction, answering) and a larger local judge. Results in `results/real_knowledge_update/`.
  ReMem 0.2 scored 45.8% vs 63.9% for plain search over raw chat because supersession on LLM-extracted keys retired correct facts.
- "Newest fact per key" baselines (rule only, no learning) in the simulator configs.

### Changed
- `ReMem.add` keeps same-key facts unless a checker confirms the update or the caller sets `update="replace"`. The CLI uses
  `--replace` and the MCP `remember` tool uses `is_update=true` for explicit replacements.
- The README reports only real-benchmark results (LongMemEval with a real LLM, LongMemEval retrieval). The synthetic simulator
  stays in the repository for development; its output is no longer presented as evidence.
- The simulator no longer hands ReMem the benchmark's own single-valued attributes; ReMem must learn them (`schema_given` keeps the
  old setting as a comparison). The learned scoring model was retrained accordingly.
- `docs/make_figures.py` reads its numbers from result files; the synthetic charts were removed from the README.

### Fixed
- A checker exception leaves the fact out of live and persisted storage so the caller can retry with the same ID.
- Reopening a saved store uses the recorded update decisions without contacting the checker.
- Replacements now handle changes such as negation, where the old and new facts can share the same value-bearing tokens.

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
