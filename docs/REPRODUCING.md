# Reproduce a ReMem release and its evaluations

Use Python 3.12 and a fresh environment. The release workflow builds a wheel and source distribution, checks metadata and hashes, and installs each artifact in separate Windows and Linux jobs. A version tag matching `pyproject.toml` publishes the tested artifacts on GitHub Releases. Build from the `v0.3.0` tag to reproduce the release. Use `master` for the evaluation scripts and reports published after that tag. [Release workflow](../.github/workflows/release.yml)

Download the wheel from the release and install it with `python -m pip install remem-0.3.0-py3-none-any.whl`. [ReMem v0.3.0](https://github.com/visweshdev/ReMem/releases/tag/v0.3.0)

```bash
python -m venv .venv
source .venv/bin/activate  # PowerShell: .venv\Scripts\Activate.ps1
python -m pip install build twine
python -m build
python -m twine check --strict dist/*
```

To reproduce the clean install, install the built wheel into a second environment, change to a directory outside the checkout, and run `remem demo`. Repeat with the source archive. Verify that `python -m pip check` succeeds. Pull requests run optional installation checks for MCP, LangChain, FAISS, Chroma, and two dependency combinations. [Integration workflow](../.github/workflows/tests.yml)

## Evaluation review

1. Record the commit, Python version, OS, model tags and exact model digests, and SHA-256 of each dataset. For a new result, retain the generated responses and judge decisions.
2. Compare the source-checked LongMemEval run against its baselines and review the per-question decisions. The same report includes commands for temporal and multi-session retrieval, the MemoryAgentBench transfer check, and the alternate-model pilot. [Regression protocol](SUPERSESSION_EVALUATION.md) · [Original artifacts](../results/real_knowledge_update/v0.2_all72/summary.json)
3. Follow the LoCoMo protocol and check the speaker attribution issue accounting for its single answer flip. [Feedback ablation](FEEDBACK_ABLATION.md)
4. Run the support-routing test with 20 seeds. Compare the outcome and credit metrics, and record performance on the reviewer's machine. [Support-routing test](WORKFLOW_ASSISTANT.md)
5. Report any version-specific installation or persistence discrepancy with the command, observed result, and environment. Replay tests should use a local evaluator with no live side effects.

Independent reviewers can share their results and environment details in the public tracking issue. [Independent reproduction of ReMem 0.3.0](https://github.com/visweshdev/ReMem/issues/1)
