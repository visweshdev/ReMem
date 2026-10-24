"""Developer CLI for inspecting a ReMem ledger (a debugging/research tool, not a UI).

    python -m remem.viz.ledger_cli results/runs/<run>/ledger.json --summary
    python -m remem.viz.ledger_cli <ledger.json> --list --sort credited --top 15
    python -m remem.viz.ledger_cli <ledger.json> --list --state ARCHIVED
    python -m remem.viz.ledger_cli <ledger.json> --id fact_00012         # full record + lifecycle
    python -m remem.viz.ledger_cli <ledger.json> --contradictions

Ledgers are written by experiments/run_benchmark.py under results/runs/ (or by
`MemoryLedger.save`). The lifecycle line is a run-length encoding of the memory's event log,
e.g.  CREATED -> RETRIEVED -> NO_CONTRIBUTION ... [RETRIEVED x12, NO_CONTRIBUTION x9] ; STATE_CHANGE: ACTIVE -> LOW_VALUE ; utility 0.58 -> 0.31 (DOWN)
"""

from __future__ import annotations

import argparse
import sys
from itertools import groupby
from pathlib import Path

from remem.ledger import MemoryLedger
from remem.types import MemoryRecord

SORTS = {
    "utility": lambda r: -(r.last_utility if r.last_utility is not None else -1.0),
    "credited": lambda r: -r.n_credited,
    "retrieved": lambda r: -r.n_retrieved,
    "wasted": lambda r: -r.n_wasted,
    "updated": lambda r: -r.last_updated_step,
    "id": lambda r: r.fact_id,
}


def _rle(kinds: list[str]) -> str:
    return " -> ".join((k if n == 1 else f"{k} x{n}") for k, n in ((k, len(list(g))) for k, g in groupby(kinds)))


def lifecycle(rec: MemoryRecord, led: MemoryLedger, head: int = 7) -> str:
    """Compact lifecycle: first events, per-kind totals, state changes and utility trend.

    e.g.  CREATED -> RETRIEVED -> USED -> CREDITED -> RETRIEVED -> NO_CONTRIBUTION ... [RETRIEVED x12, USED x3,
          NO_CONTRIBUTION x9] ; STATE_CHANGE: ACTIVE->LOW_VALUE ; utility 0.58 -> 0.31 (DOWN)
    """
    ev = rec.events
    parts = [_rle([e.kind for e in ev[:head]])]
    if len(ev) > head:
        totals: dict[str, int] = {}
        for e in ev[head:]:
            totals[e.kind] = totals.get(e.kind, 0) + 1
        parts.append("... [" + ", ".join(f"{k} x{n}" for k, n in totals.items()) + "]")
    changes = [e.detail.split(":")[0] for e in ev if e.kind == "STATE_CHANGE"]
    if changes:
        parts.append("; STATE_CHANGE: " + ", ".join(changes))
    utils = [e.utility for e in ev if e.utility is not None]
    if len(utils) >= 2:
        d = utils[-1] - utils[0]
        parts.append(f"; utility {utils[0]:.2f} -> {utils[-1]:.2f} ({'UP' if d > 0.02 else 'DOWN' if d < -0.02 else 'FLAT'})")
    return " ".join(parts)


def row(rec: MemoryRecord) -> str:
    u = "  -  " if rec.last_utility is None else f"{rec.last_utility:5.2f}"
    return (f"{rec.fact_id:<11} {rec.state.value:<11} {u}  ret={rec.n_retrieved:<4} cred={rec.n_credited:5.1f} "
            f"waste={rec.n_wasted:<3} blame={rec.n_blamed:3.0f} contra={rec.contradictions} upd@{rec.last_updated_step:<4} "
            f"{rec.original_text[:58]}")


def show(rec: MemoryRecord, led: MemoryLedger, out=None) -> None:
    out = sys.stdout if out is None else out      # resolve at call time, not import time
    w = lambda s="": print(s, file=out)  # noqa: E731
    w(f"Memory            {rec.fact_id}")
    w(f"Content           {rec.original_text}")
    if rec.text != rec.original_text:
        w(f"Served as         {rec.text}   ({rec.tokens} of {rec.original_tokens} tokens)")
    w(f"State             {rec.state.value}")
    w(f"Utility (last)    {'-' if rec.last_utility is None else f'{rec.last_utility:.3f}'}   "
      f"contribution rate {led.contrib_rate(rec):.3f} (Beta posterior), exposures {led.exposures(rec)}")
    w(f"Retrieval history pooled {rec.n_pooled}x, selected {rec.n_retrieved}x, last at step {rec.last_retrieved_step}")
    w(f"Outcome history   credited {rec.n_credited:.1f}, wasted {rec.n_wasted}x, blamed {rec.n_blamed:.0f}, "
      f"in successful tasks {rec.n_success_ctx}, in failed tasks {rec.n_fail_ctx}")
    if rec.classes:
        w("Per task class    " + ", ".join(f"{k}: shown {v.n_ctx}, credit {v.credit:.1f}" for k, v in sorted(rec.classes.items())))
    w(f"Validity          valid_from={rec.valid_from:.0f} valid_to={rec.valid_to if rec.valid_to is None else f'{rec.valid_to:.0f}'}")
    w(f"Contradictions    count={rec.contradictions} superseded_by={rec.superseded_by} supersedes={rec.supersedes}")
    w(f"Last updated      step {rec.last_updated_step}")
    w(f"Why retrieved     {rec.last_reason or '-'}")
    w()
    w("Lifecycle         " + lifecycle(rec, led))
    w()
    w("Recent events:")
    for e in rec.events[-12:]:
        u = "" if e.utility is None else f" u={e.utility:.2f}"
        w(f"  step {e.step:>5}  {e.kind:<16} {e.detail[:100]}{u}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="remem.viz.ledger_cli")
    ap.add_argument("ledger", type=Path)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--state")
    ap.add_argument("--sort", choices=sorted(SORTS), default="utility")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--id")
    ap.add_argument("--contradictions", action="store_true")
    a = ap.parse_args(argv)
    led = MemoryLedger.load(a.ledger)

    if a.id:
        rec = led.get(a.id)
        if rec is None:
            print(f"unknown memory id {a.id!r}", file=sys.stderr)
            return 2
        show(rec, led)
        return 0
    if a.contradictions:
        for rec in sorted(led.records.values(), key=lambda r: r.fact_id):
            if rec.superseded_by or rec.supersedes:
                print(f"{rec.fact_id} [{rec.state.value}] supersedes={rec.supersedes} superseded_by={rec.superseded_by}\n    {rec.original_text}")
        return 0
    if a.list:
        recs = [r for r in led.records.values() if not a.state or r.state.value == a.state.upper()]
        recs.sort(key=SORTS[a.sort])
        print(f"{'id':<11} {'state':<11} util   stats")
        for r in recs[: a.top]:
            print(row(r))
        print(f"({len(recs)} memories match)")
        return 0
    # default / --summary
    print(f"{len(led.records)} memories, ledger step {led.step}")
    for st, c in led.state_counts().items():
        print(f"  {st:<12} {c}")
    print(f"  served tokens {led.served_tokens()} / total stored tokens {led.total_tokens()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
