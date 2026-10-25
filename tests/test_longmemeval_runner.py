"""LongMemEval runner on a tiny file in the dataset's documented format (no download needed)."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_longmemeval import evaluate, parse_date  # noqa: E402

TINY = [
    {
        "question_id": "q1", "question_type": "knowledge-update",
        "question": "Which city did I say I moved to?", "answer": "Berlin", "question_date": "2023/06/01 (Thu) 10:00",
        "haystack_session_ids": ["s_a", "s_b", "s_c"],
        "haystack_dates": ["2023/05/01 (Mon) 09:00", "2023/05/10 (Wed) 12:30", "2023/05/20 (Sat) 02:21"],
        "haystack_sessions": [
            [{"role": "user", "content": "Can you recommend a pasta recipe for tonight?"},
             {"role": "assistant", "content": "Try a simple aglio e olio with garlic and chili."}],
            [{"role": "user", "content": "I just moved to Berlin last week for my new job.", "has_answer": True},
             {"role": "assistant", "content": "Congratulations on the move to Berlin!"}],
            [{"role": "user", "content": "What is a good stretching routine?"},
             {"role": "assistant", "content": "Hamstring and hip flexor stretches, 30 seconds each."}],
        ],
        "answer_session_ids": ["s_b"],
    },
    {"question_id": "q2_abs", "question_type": "single-session-user", "question": "x?", "answer": "",
     "question_date": "2023/06/01", "haystack_session_ids": [], "haystack_dates": [], "haystack_sessions": [],
     "answer_session_ids": []},
]


def test_parse_date_formats():
    assert parse_date("2023/05/20 (Sat) 02:21") < parse_date("2023/05/21")


def test_evaluate_finds_the_evidence_and_skips_abstention(embedder):
    res = evaluate(TINY, embedder, ks=(1, 2), log=lambda *_: None)
    assert res["n_questions"] == 1
    for system in ("similarity", "static", "remem"):
        assert res["summary"][system]["session_recall@2"] == 1.0
    assert res["summary"]["similarity"]["turn_recall@2"] == 1.0
    assert "knowledge-update" in res["per_type"]


def test_cli_end_to_end_with_hash_embedder(tmp_path):
    data = tmp_path / "tiny.json"
    data.write_text(json.dumps(TINY), encoding="utf-8")
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ROOT / "experiments" / "run_longmemeval.py"), "--data", str(data),
                        "--embedder", "hash", "--out", str(out)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    saved = json.loads((out / "retrieval.json").read_text(encoding="utf-8"))
    assert saved["n_questions"] == 1 and set(saved["summary"]) == {"similarity", "static", "remem"}
