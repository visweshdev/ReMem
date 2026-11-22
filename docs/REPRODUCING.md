# Reproduce a ReMem release and its evaluations

Use Python 3.12 and a fresh environment. The release workflow builds a wheel and source distribution, checks metadata and hashes, and installs each artifact in separate Windows and Linux jobs. A version tag matching `pyproject.toml` publishes the tested artifacts on GitHub Releases. [Release workflow](../.github/workflows/release.yml)

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
2. Reproduce the original LongMemEval baseline and check its per-question verdicts. The original 72-question table remains available. [Original artifacts](../results/real_knowledge_update/v0.2_all72/summary.json)
3. Run the [LoCoMo ablation](FEEDBACK_ABLATION.md) and confirm the speaker attribution issue accounting for its single answer flip.
4. Run the [support-routing test](WORKFLOW_ASSISTANT.md) with 20 seeds. Compare the outcome and credit metrics, and record performance on the reviewer's machine separately.
5. Report any version-specific installation or persistence discrepancy with the command, observed result, and environment. Replay tests should use a local evaluator with no live side effects.
