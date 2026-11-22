"""experiments/run_real_knowledge_update.py with a fake LLM: the pipeline runs end to end without Ollama."""

import json
import re
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_real_knowledge_update import SYSTEMS, paired_ci, run_question, split_questions  # noqa: E402


def _inst():
    return {
        "question_id": "q1", "question": "What is my personal best time in the charity 5K run?", "answer": "25:50",
        "question_date": "2023/06/25 (Sun) 13:22",
        "haystack_dates": ["2023/05/23 (Tue) 13:01", "2023/05/30 (Tue) 13:53", "2023/06/01 (Thu) 10:00"],
        "haystack_sessions": [
            [{"role": "user", "content": "I set a personal best in a charity 5K run with a time of 27:12.", "has_answer": True}],
            [{"role": "user", "content": "My new personal best in the charity 5K run is 25:50.", "has_answer": True}],
            [{"role": "user", "content": "I'm training again and hope to beat my personal best in the 5K."}],
        ],
    }


class FakeLLM:
    """Extraction: one fact per turn, all under the same key. Update check: yes only for a real new time.
    Answers: echo the newest time seen in the context."""

    def __init__(self):
        self.prompts = []

    def __call__(self, prompt, max_tokens=400):
        self.prompts.append(prompt)
        if prompt.startswith("Extract durable facts"):
            turn = prompt.rsplit("user: ", 1)[1]
            return json.dumps({"facts": [{"text": turn, "entity": "user", "attribute": "personal_best_time",
                                          "role": "state"}]})
        if prompt.startswith("Two memories"):
            newer = re.split(r"Newer(?: claim)?:", prompt, maxsplit=1)[1]
            return "yes" if re.search(r"\d+:\d+", newer.split("\n")[0]) else "no"
        times = [t for t in ("25:50", "27:12") if t in prompt]
        return times[0] if times else "I don't know"


def test_split_is_deterministic_and_disjoint():
    qs = [{"question_id": f"id{i}"} for i in range(10)]
    dev, test = split_questions(qs, "dev"), split_questions(qs, "test")
    assert len(dev) == len(test) == 5 and not {q["question_id"] for q in dev} & {q["question_id"] for q in test}
    assert split_questions(list(reversed(qs)), "dev") == dev and split_questions(qs, "all") == qs


def test_pipeline_runs_and_safe_replacement_keeps_the_answer(embedder):
    llm = FakeLLM()
    out = run_question(_inst(), embedder, llm, threading.Lock())
    assert set(out["systems"]) == set(SYSTEMS) and out["n_facts"] == 3
    # the extractor never sees the question
    assert not any("What is my personal best" in p for p in llm.prompts if p.startswith("Extract durable facts"))
    # ReMem 0.2's rule lets the vague later fact replace 25:50; the safe default keeps it
    assert "25:50" not in out["systems"]["facts_newest_key"]["context"]
    assert "25:50" in out["systems"]["facts_remem"]["context"]
    assert out["systems"]["facts_remem"]["newest_evidence_in_context"]
    assert "(said:" in out["systems"]["facts_remem"]["context"]
    assert "(said:" not in out["systems"]["facts_remem_no_source"]["context"]


def test_paired_ci():
    m, lo, hi = paired_ci([1, 1, 1, 0], [0, 0, 1, 0], n_boot=500)
    assert m == 0.5 and lo <= m <= hi
