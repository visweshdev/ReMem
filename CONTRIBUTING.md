# Contributing to ReMem

Thanks for your interest. Bug reports, ideas, and pull requests are all welcome.

## Set up

```bash
git clone https://github.com/visweshdev/ReMem.git
cd ReMem
pip install -e ".[dev]"        # the library plus pytest, ruff and matplotlib
python -m pytest tests -q      # should finish in under a minute
ruff check .                   # style checks; CI runs the same command
```

## Making a change

1. Open an issue first for anything larger than a small fix, so we can agree on the approach.
2. Keep the change focused, and add or update a test in `tests/` for any behaviour you change.
3. Run `python -m pytest tests -q` and `ruff check .` before opening the pull request.
4. If you change an experiment or a reported number, regenerate the result with the matching script in `experiments/` and
   update the corresponding table and report. For LoCoMo, use the documented split and refresh the reference review if the
   generated answers change. [LoCoMo reproduction steps](docs/FEEDBACK_ABLATION.md#reproduce)
5. For distribution or evaluation changes, use a clean environment and record the package version and input fingerprints.
   [Release and evaluation reproduction](docs/REPRODUCING.md)

## Reporting results honestly

Use public benchmarks and real models for reported evaluations. Describe how the benchmark conversations were produced,
record the model and budget settings, and report gains and regressions with representative failures.

## Where things live

- `remem/`: the library (`api.py` is the main entry point; `cli.py` is the `remem` command)
- `tests/`: unit and integration tests
- `experiments/`: public benchmark runners for LongMemEval, LoCoMo, and MemoryAgentBench
- `results/`: published benchmark artifacts, including per-question results where available
- `dev/`: the development simulator (synthetic worlds, simulated reader). Useful for checking a mechanism; its numbers
  are never reported as evidence. Tests: `python -m pytest dev/tests -q`
- `docs/`: design notes, related work, evaluation reports, reproduction instructions, and figure and video generators
