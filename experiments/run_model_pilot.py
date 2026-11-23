"""Cross two extractor and answer models on fixed LongMemEval development excerpts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_real_knowledge_update import CachedLLM
from remem.extract import PROMPT, parse_facts

ANSWER = """Answer the question using the dated facts below. If a current fact changes, choose the latest value.
Return just the shortest answer. Use Unknown if needed.

{facts}

Question: {question}
Answer:"""
NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
REFERENCE_TAGS = {"qwen2.5:3b": "qwen2.5-3b-4k", "llama3.2:3b": "llama3.2:3b"}


def normalize(text: str) -> tuple[str, ...]:
    return tuple(word for word in re.findall(r"\w+", text.casefold()) if word not in {"a", "an", "the"})


def normalize_numbers(words: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(NUMBER_WORDS.index(word)) if word in NUMBER_WORDS else word for word in words)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "results/real_knowledge_update/dev/per_question.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--questions", type=int, default=12, help="number of short-reference development questions")
    parser.add_argument("--url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--qwen-tag", default=REFERENCE_TAGS["qwen2.5:3b"])
    parser.add_argument("--llama-tag", default=REFERENCE_TAGS["llama3.2:3b"])
    parser.add_argument("--cache", type=Path, default=ROOT / "results/cache/reader_pilot.jsonl",
                        help="use a new path to measure tokens for every model call")
    args = parser.parse_args()

    from openai import OpenAI

    data = args.data.read_bytes()
    cases = [json.loads(line) for line in data.splitlines()]
    cases = [row for row in cases if 0 < len(normalize(str(row["answer"]))) <= 3]
    if not 1 <= args.questions <= len(cases):
        parser.error(f"--questions must be between 1 and {len(cases)} for this development file")
    cases = cases[:args.questions]
    tags = {"qwen2.5:3b": args.qwen_tag, "llama3.2:3b": args.llama_tag}
    client = OpenAI(base_url=args.url, api_key="ollama", timeout=180)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    models = {}
    for name, tag in tags.items():
        cache_key = name if tag == REFERENCE_TAGS[name] else f"{name} served as {tag}"
        models[name] = CachedLLM(client, cache_key, args.cache, tag, seed=0)
    output = []
    for row in cases:
        context = row["systems"]["raw_similarity"]["context"]
        chunks = re.split(r"(?m)^- \[(\d{4}-\d{2}-\d{2})\] ", context)[1:]
        sources = list(zip(chunks[::2], chunks[1::2]))
        if not sources:
            raise ValueError(f"{row['question_id']} has no dated excerpts")
        output.append({"id": row["question_id"], "question": row["question"], "reference": str(row["answer"]),
                       "sources": sources, "extractions": {}, "answers": {}})

    for name, complete in models.items():
        for row in output:
            facts = []
            for date, text in row["sources"]:
                extracted = parse_facts(complete(PROMPT + "user: " + text.strip(), 160))[:2]
                facts.extend(f"- [{date}] {fact.text}" for fact in extracted)
            row["extractions"][name] = facts
            print(f"{name}: extracted {row['id']} ({len(facts)} facts)", flush=True)

    for name, complete in models.items():
        for row in output:
            for extractor, facts in row["extractions"].items():
                prompt = ANSWER.format(facts="\n".join(facts) or "- (no facts)", question=row["question"])
                answer = complete(prompt, 64)
                predicted, reference = normalize(answer), normalize(row["reference"])
                row["answers"][f"{extractor} -> {name}"] = {
                    "answer": answer, "correct": predicted == reference,
                    "number_normalized_correct": normalize_numbers(predicted) == normalize_numbers(reference),
                }

    summary = {arm: sum(row["answers"][arm]["correct"] for row in output) for arm in output[0]["answers"]}
    number_summary = {arm: sum(row["answers"][arm]["number_normalized_correct"] for row in output) for arm in summary}
    results = {"data_sha256": hashlib.sha256(data).hexdigest(), "tags": tags, "temperature": 0, "seed": 0,
               "n_questions": len(output), "selection": "first development references of one to three normalized words",
               "max_tokens": {"extraction": 160, "answer": 64}, "prompts": {"extraction": PROMPT, "answer": ANSWER},
               "judge": "exact match ignoring case and articles, with a sensitivity check normalizing zero through ten",
               "counts": summary, "number_normalized_counts": number_summary,
               "usage": {name: dict(model.usage) for name, model in models.items()}, "rows": output}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": summary, "number_normalized_counts": number_summary}, indent=2))


if __name__ == "__main__":
    main()
