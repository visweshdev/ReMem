# Changelog

All notable changes are listed here. Versions follow [semantic versioning](https://semver.org/).

## [Unreleased]

- Published the 72-question LongMemEval supersession rerun, with paired intervals, scored traces, and the original
  0.2 result. Added a conflict-resolution transfer check, temporal and multi-session retrieval results, and a development
  comparison of two extraction and answer models. [Evaluation report](docs/SUPERSESSION_EVALUATION.md)
- Opened a public request for independent installation and benchmark reproductions. [Reproduction guide](docs/REPRODUCING.md)

## [0.3.0] - 2026-11-22

### Added
- GitHub Actions jobs install and test the optional integrations. Release builds verify the wheel and source archive in clean
  Windows and Linux jobs before publishing checksummed assets to GitHub Releases.
- The Python feedback API accepts an isolated replay callback for leave-one-out credit assignment. The support-routing example
  exercises persisted sessions and reports held-out task success, retrieval latency, context tokens, and storage growth.
- The optional MCP server supports both major versions of the Python SDK and tests both in CI.
- Real-data evaluation: `experiments/run_real_knowledge_update.py` runs LongMemEval's 72 knowledge-update questions end to end
  with a local LLM (fact extraction, answering) and a larger local judge. Results in `results/real_knowledge_update/`.
  ReMem 0.2 scored 45.8% vs 63.9% for plain search over raw chat because supersession on LLM-extracted keys retired correct facts.
- The [LoCoMo feedback ablation](docs/FEEDBACK_ABLATION.md) compares plain search and ReMem with feedback off and on
  across 80 held-out questions under fixed model and token budgets. The two ReMem arms tied on answer accuracy after
  excluding one apparent speaker attribution error. Per-question results and warm-up traces are in `results/feedback_ablation/`.
- "Newest fact per key" baselines (rule only, no learning) in the simulator configs.

### Changed
- `LLMUpdateChecker` uses the original source turns when available and requires explicit evidence of a completed update.
  Ambiguous or same-timestamp claims stay active.
- `ReMem.add` keeps same-key facts unless a checker confirms the update or the caller sets `update="replace"`. The CLI uses
  `--replace` and the MCP `remember` tool uses `is_update=true` for explicit replacements.
- The README reports public benchmark results from LongMemEval and LoCoMo. The in-repo simulator remains available for development.
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

[Unreleased]: https://github.com/visweshdev/ReMem/compare/v0.3.0...master
[0.3.0]: https://github.com/visweshdev/ReMem/releases/tag/v0.3.0
[0.2.0]: https://github.com/visweshdev/ReMem/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/visweshdev/ReMem/releases/tag/v0.1.0
