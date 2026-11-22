"""Checked updates stay consistent across failures and restarts."""

import json

import pytest

from remem import ReMem

KEY = {"entity": "billing", "attribute": "api_version"}
V1 = "The billing service uses API version v1."
V2 = "The billing service uses API version v2."
V3 = "The billing service uses API version v3."


def unavailable_checker(_older, _newer):
    raise ConnectionError("checker unavailable")


def test_checker_failure_leaves_both_live_and_saved_state_unchanged(tmp_path, embedder):
    available = False

    def checker(older, newer):
        if not available and older == V2:
            unavailable_checker(older, newer)
        return True

    memory = ReMem(tmp_path, embedder=embedder, update_checker=checker)
    memory.add(V1, fact_id="v1", when=1, update="keep", **KEY)
    memory.add(V2, fact_id="v2", when=2, update="keep", **KEY)
    files = [tmp_path / "facts.jsonl", tmp_path / "ledger.json"]
    before = [path.read_bytes() for path in files]

    with pytest.raises(ConnectionError, match="checker unavailable"):
        memory.add(V3, fact_id="v3", when=3, **KEY)

    assert [path.read_bytes() for path in files] == before
    assert memory.stats()["memories"] == 2
    assert all(memory.inspect(fid)["state"] == "ACTIVE" for fid in ("v1", "v2"))
    candidates, _ = memory.backend.candidates("billing API", k=10)
    assert {candidate.fact_id for candidate in candidates} == {"v1", "v2"}

    available = True
    assert memory.add(V3, fact_id="v3", when=3, **KEY) == "v3"
    assert all(memory.inspect(fid)["superseded_by"] == "v3" for fid in ("v1", "v2"))


@pytest.mark.parametrize("saved_snapshot", ["current", "stale", "missing", "legacy"])
def test_reopen_uses_saved_decisions_without_calling_the_checker(tmp_path, embedder, saved_snapshot):
    memory = ReMem(tmp_path, embedder=embedder, update_checker=lambda _old, _new: True,
                   autosave=saved_snapshot != "stale")
    old = memory.add(V1, when=1, **KEY)
    memory.save()
    new = memory.add(V2, when=2, **KEY)
    if saved_snapshot == "missing":
        (tmp_path / "ledger.json").unlink()
    elif saved_snapshot == "legacy":
        path = tmp_path / "facts.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            row.pop("confirmed_updates")
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    restored = ReMem(tmp_path, embedder=embedder, update_checker=unavailable_checker)
    assert restored.inspect(old)["superseded_by"] == new
    assert restored.inspect(new)["state"] == "ACTIVE"
    with pytest.raises(ConnectionError, match="checker unavailable"):
        restored.add(V3, when=3, **KEY)
    assert restored.stats()["memories"] == 2


def test_replay_restores_a_confirmed_replacement_of_a_backfilled_fact(tmp_path, embedder):
    memory = ReMem(tmp_path, embedder=embedder, update_checker=lambda _old, _new: True, autosave=False)
    newest = memory.add(V2, when=2, **KEY)
    memory.save()
    backfilled = memory.add(V1, when=1, update="keep", **KEY)

    restored = ReMem(tmp_path, embedder=embedder)
    assert restored.inspect(backfilled)["superseded_by"] == newest
    assert restored.inspect(backfilled)["valid_to"] == 2
