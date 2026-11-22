"""Protocol checks for the LoCoMo feedback ablation without model downloads."""

from dataclasses import replace

from experiments.run_feedback_ablation import frozen_search, grade, make_split, parse_response, system_configs
from remem.api import ReMem


def test_only_feedback_flag_differs_between_remem_arms():
    configs = system_configs()
    assert replace(configs["feedback_on"], outcome_feedback=False) == configs["feedback_off"]
    assert configs["plain_search"].selection.budget_tokens == configs["feedback_on"].selection.budget_tokens


def test_split_keeps_duplicate_questions_out_of_holdout_comparison():
    questions = [{"question": f"Question {category} {i}?", "answer": "Berlin", "category": category, "evidence": ["D1:1"]}
                 for category in (1, 4) for i in range(14)]
    questions.append({**questions[0], "question": questions[0]["question"].upper()})
    sample = {"sample_id": "fixture", "qa": questions}
    warmup, held_out = make_split(sample, {"D1:1"})
    assert (len(warmup), len(held_out)) == (12, 8)
    assert {q.text.casefold() for q in warmup}.isdisjoint(q.text.casefold() for q in held_out)
    assert (warmup, held_out) == make_split(sample, {"D1:1"})


def test_heldout_search_does_not_write_to_the_warmup_ledger(embedder):
    memory = ReMem(embedder=embedder, config=system_configs()["feedback_on"])
    memory.add("I moved to Berlin", fact_id="D1:1", when=100)
    warmup = memory.search("Where did I move?", now=200)
    memory.feedback(warmup.search_id, True, used_ids=["D1:1"], answer="Berlin")
    state = memory.inspect("D1:1")
    pending = memory.stats()["pending_feedback"]

    result = frozen_search(memory, "What city did I move to?", 200)

    assert [hit.id for hit in result.items] == ["D1:1"]
    assert memory.inspect("D1:1") == state
    assert memory.stats()["pending_feedback"] == pending


def test_feedback_uses_only_citations_seen_in_the_context():
    answer, cited = parse_response("Answer: Berlin.\nEvidence: [D1:9], d2:2", ["D2:2", "D3:3"])
    assert (answer, cited) == ("Berlin.", ["D2:2"])
    assert parse_response("Evidence: [D2:2]\nAnswer: Berlin.", ["D2:2"]) == ("Berlin.", ["D2:2"])
    assert grade(answer, "Berlin") == (True, 1.0)
    assert grade("Unknown", "Berlin") == (False, 0.0)
