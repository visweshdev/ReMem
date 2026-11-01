"""The `remem` command-line tool."""

import io

from remem import cli


def test_add_search_list_inspect_stats(tmp_path, capsys):
    store = str(tmp_path / "s")
    assert cli.main(["--store", store, "add", "Our server is in east-1.", "--entity", "server", "--attribute", "region"]) == 0
    old = capsys.readouterr().out.strip()
    assert cli.main(["--store", store, "add", "Our server moved to west-2.", "--entity", "server", "--attribute", "region"]) == 0
    capsys.readouterr()
    cli.main(["--store", store, "search", "Where is our server?"])
    out = capsys.readouterr().out
    assert "west-2" in out and "east-1" not in out
    cli.main(["--store", store, "list"])
    assert "INVALIDATED" in capsys.readouterr().out
    cli.main(["--store", store, "inspect", old])
    assert "superseded_by" in capsys.readouterr().out
    cli.main(["--store", store, "stats"])
    assert "2 memories" in capsys.readouterr().out
    assert cli.main(["--store", store, "inspect", "nope"]) == 1


def test_shell_feedback_changes_the_ranking(tmp_path):
    from remem.api import ReMem
    mem = ReMem(tmp_path / "s")
    script = ["add The deploy pipeline targets cluster west-2. --entity deploy --attribute cluster",
              "add Note: someone asked which cluster the deploy pipeline targets; nobody answered. --role note"]
    for _ in range(6):
        script += ["search Which cluster does the deploy pipeline target?", "good 1" if _ else "bad"]
    script += ["good 9", "add", "frobnicate", "quit"]
    out = io.StringIO()
    cli.cmd_shell(mem, io.StringIO("\n".join(script) + "\n"), out)
    text = out.getvalue()
    assert "recorded:" in text and "search first" in text and "unknown command" in text
    assert "usage: add" in text or "arguments are required" in text   # differs by Python version
    states = {r["text"][:4]: r["state"] for r in mem.list()}
    assert states["Note"] == "LOW_VALUE" and states["The "] == "ACTIVE"


def test_demo_runs(capsys):
    assert cli.main(["demo"]) == 0
    assert "INVALIDATED" in capsys.readouterr().out
