"""The README quickstart must keep working."""

import runpy
from pathlib import Path


def test_quickstart_runs_and_shows_supersession_and_demotion(capsys):
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples" / "quickstart.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "fact_1: state=INVALIDATED" in out
    assert "fact_2: state=LOW_VALUE" in out
    assert "fact_3: state=ACTIVE" in out
