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
4. If you change an experiment or a reported number, regenerate the result with the script in `experiments/` and
   update the README table. Never edit a number by hand.

## Reporting results honestly

ReMem reports where it does **not** help as clearly as where it does. Changes to evaluation code should keep
test seeds (0-14) untouched for tuning, and new results should be reported whatever they show.

## Where things live

- `remem/`: the library (`api.py` is the main entry point; `cli.py` is the `remem` command)
- `tests/`: unit and integration tests
- `experiments/`: scripts that produce every number in the README
- `docs/`: design notes, related work, figure and video generators
