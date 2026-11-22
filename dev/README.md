# dev/: development simulator (not evidence)

Everything in this folder is a **synthetic simulator** used while building ReMem: author-generated worlds
(`remem_sim/eval/world.py`) and a *simulated* reader instead of an LLM (`remem_sim/eval/agent.py`). It checks that each
mechanism does what it was written to do. Its numbers depend on assumptions chosen by the author, so they are
**not evidence about real-world use** and are not reported in the main README.

Public benchmark results and reproduction commands are in the [main README](../README.md#10-results). The
[LoCoMo feedback report](../docs/FEEDBACK_ABLATION.md) compares three retrieval settings with a local answer model
on held-out questions. The experiments and artifacts for LoCoMo and LongMemEval live at the repository root.

| Path | What it is |
|---|---|
| `remem_sim/eval/` | world generator, simulated reader, runner, metrics, failure analysis, report |
| `remem_sim/baselines/rmm.py` | RMM-style reranker (reimplementation of arXiv 2503.08026 §6, not the authors' code) |
| `remem_sim/claude_agent.py`, `remem_sim/llm_agent.py` | real models playing the simulator's reader |
| `configs/systems.yaml` | every simulated system; ReMem gets no hand-written list of which attributes change (`schema_given` is the labelled exception) |
| `experiments/` | `run_benchmark.py` (main driver), `train_utility*.py`, `tune_rmm.py`, `run_llm_agent.py`, `run_claude_agent.py`, `run_real_embeddings.py`, `trace_llm_prompts.py`, `demo_ranking.py` |
| `results/` | simulator output (`RESULTS.md` and JSON); `results/llm_agent/` and `real_embeddings.json` are from ReMem 0.2 and were not re-run |
| `tests/` | the simulator's own tests |

```bash
pip install -e ".[dev]"
python -m pytest dev/tests -q                       # simulator tests
python dev/experiments/run_benchmark.py --quick     # smoke run (~20 s)
python dev/experiments/run_benchmark.py             # full run, 15 seeds (~10 min, resumable)
```

What the simulator showed, for the record (ReMem 0.3 configs, 15 seeds, simulated reader): more than half of ReMem's
gain over plain similarity came from the simple "keep the newest fact per key" rule, which needs no learning. On real
data that rule *hurt* (see the main README), which is why the library no longer replaces facts on key match alone.
