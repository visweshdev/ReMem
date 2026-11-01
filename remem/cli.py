"""`remem` command-line tool: try ReMem without writing Python.

    remem add "Our server is in east-1." --entity server --attribute region
    remem search "Where is our server?"
    remem list                     remem inspect <id>          remem stats
    remem shell                    # interactive: search, then mark results good/bad and watch the ranking change
    remem demo                     # the example from the README
    remem dashboard -o memory.html

The memory lives in a folder: `--store PATH` (default: $REMEM_PATH, else ~/.remem/cli). Feedback needs the search
that produced it, and pending searches only live inside one process, so outcomes are given in `remem shell`
(or from Python with `mem.feedback`).
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

DEFAULT_STORE = "~/.remem/cli"


def _memory(store: str):
    from remem.api import ReMem
    return ReMem(Path(store).expanduser())


def _print_hits(res, out) -> None:
    if not res.items:
        print("  (no memories match)", file=out)
        return
    for i, h in enumerate(res.items, 1):
        print(f"  {i}. [{h.id}] {h.text}", file=out)
        print(f"     score {h.score:.2f}  similarity {h.similarity:.2f}  {h.state}", file=out)


def _print_list(rows, out) -> None:
    if not rows:
        print("  (empty)", file=out)
    for r in rows:
        print(f"  {r['state']:<12} shown {r['shown']:>3}  helped {r['credited']:>5.1f}  [{r['id']}] {r['text']}", file=out)


def _print_inspect(info, out) -> None:
    for key in ("id", "text", "state", "shown", "credited", "wasted", "blamed", "contribution_rate", "superseded_by",
                "supersedes", "last_reason"):
        if info.get(key) not in (None, [], ""):
            print(f"  {key:<18} {info[key]}", file=out)


def cmd_shell(mem, inp=sys.stdin, out=sys.stdout) -> None:
    """A small REPL. Commands: add, search, good [n ...], bad, list, inspect, stats, help, quit."""
    help_text = ("commands:\n"
                 "  add <text> [--entity E --attribute A]   store a fact (same entity+attribute replaces the old one)\n"
                 "  search <question>                       retrieve memories\n"
                 "  good [n ...]                            the last search helped; n = which results you used\n"
                 "  bad                                     the last search did not help\n"
                 "  list | inspect <id> | stats | help | quit")
    print("ReMem shell. Type 'help' for commands.", file=out)
    last = None
    while True:
        print("remem> ", end="", file=out, flush=True)
        line = inp.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        cmd = cmd.lower()
        try:
            if cmd in ("quit", "exit", "q"):
                break
            elif cmd == "help":
                print(help_text, file=out)
            elif cmd == "add":
                p = argparse.ArgumentParser(prog="add", add_help=False, exit_on_error=False)
                p.add_argument("text", nargs="+")
                p.add_argument("--entity")
                p.add_argument("--attribute")
                p.add_argument("--role", default="state")
                a = p.parse_args(shlex.split(rest))
                fid = mem.add(" ".join(a.text), entity=a.entity, attribute=a.attribute, role=a.role)
                print(f"  stored [{fid}]", file=out)
            elif cmd == "search":
                last = mem.search(rest, task="shell")
                _print_hits(last, out)
                if last.items:
                    print("  tell me how it went: 'good 1' (result 1 helped) or 'bad'", file=out)
            elif cmd in ("good", "bad"):
                if last is None or not last.search_id:
                    print("  search first", file=out)
                    continue
                used = []
                for tok in rest.split():
                    n = int(tok)
                    if not 1 <= n <= len(last.items):
                        raise ValueError(f"no result number {n}")
                    used.append(last.items[n - 1].id)
                credit = mem.feedback(last.search_id, success=(cmd == "good"), used_ids=used)
                last = None
                print("  recorded: " + ", ".join(f"[{k}] {v:+.2f}" for k, v in credit.items()), file=out)
            elif cmd == "list":
                _print_list(mem.list(limit=30), out)
            elif cmd == "inspect":
                _print_inspect(mem.inspect(rest.strip()), out)
            elif cmd == "stats":
                print(f"  {mem.stats()['states']}", file=out)
            else:
                print(f"  unknown command {cmd!r}; type 'help'", file=out)
        except (ValueError, KeyError, argparse.ArgumentError) as e:
            print(f"  error: {e}", file=out)
        except SystemExit:                     # argparse usage errors inside the shell must not end it
            print("  usage: add <text> [--entity E --attribute A]", file=out)
    print("bye", file=out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="remem", description="ReMem: memory for AI agents that learns what actually helps.")
    ap.add_argument("--store", default=os.environ.get("REMEM_PATH", DEFAULT_STORE),
                    help=f"memory folder (default: $REMEM_PATH or {DEFAULT_STORE})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add", help="store a fact")
    p.add_argument("text")
    p.add_argument("--entity")
    p.add_argument("--attribute")
    p.add_argument("--role", default="state", help='"state" (default; replaces same entity+attribute), "note", ...')
    p = sub.add_parser("search", help="retrieve memories for a question")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p = sub.add_parser("list", help="list memories")
    p.add_argument("--state", help="ACTIVE, LOW_VALUE, COMPRESSED, ARCHIVED or INVALIDATED")
    p.add_argument("--limit", type=int, default=50)
    p = sub.add_parser("inspect", help="show one memory's history")
    p.add_argument("id")
    sub.add_parser("stats", help="counts by state")
    sub.add_parser("shell", help="interactive mode with feedback")
    sub.add_parser("demo", help="run the README demo (in memory, nothing saved)")
    p = sub.add_parser("dashboard", help="write a self-contained HTML dashboard")
    p.add_argument("-o", "--out", default="memory.html")
    a = ap.parse_args(argv)

    if a.cmd == "demo":
        import runpy
        demo = Path(__file__).resolve().parents[1] / "examples" / "basic_api.py"
        if demo.exists():
            runpy.run_path(str(demo), run_name="__main__")
        else:
            _builtin_demo()
        return 0
    if a.cmd == "dashboard":
        from remem.viz.dashboard import main as dash_main
        return dash_main([str(Path(a.store).expanduser()), "-o", a.out]) or 0

    mem = _memory(a.store)
    try:
        if a.cmd == "add":
            print(mem.add(a.text, entity=a.entity, attribute=a.attribute, role=a.role))
        elif a.cmd == "search":
            _print_hits(mem.search(a.query, k=a.k, task="cli"), sys.stdout)
        elif a.cmd == "list":
            _print_list(mem.list(state=a.state, limit=a.limit), sys.stdout)
        elif a.cmd == "inspect":
            _print_inspect(mem.inspect(a.id), sys.stdout)
        elif a.cmd == "stats":
            s = mem.stats()
            print(f"{s['memories']} memories in {s['path']}")
            print(s["states"])
        elif a.cmd == "shell":
            cmd_shell(mem)
    except (KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def _builtin_demo() -> None:
    """Fallback when the examples folder is not present (installed from a wheel)."""
    from remem.api import ReMem
    mem = ReMem()
    old = mem.add("The deploy pipeline targets cluster east-1.", entity="deploy", attribute="cluster")
    note = mem.add("Note: someone asked which cluster the deploy pipeline targets; nobody answered.", role="note")
    new = mem.add("The deploy pipeline moved to cluster west-2.", entity="deploy", attribute="cluster")
    for _ in range(7):
        res = mem.search("Which cluster does the deploy pipeline target?", k=2, task="deploy")
        mem.feedback(res.search_id, success=True, used_ids=[new])
    for fid in (old, note, new):
        info = mem.inspect(fid)
        print(f"{info['state']:<12} {info['text']}")


if __name__ == "__main__":
    raise SystemExit(main())
