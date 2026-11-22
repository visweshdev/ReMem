"""Compare cosine retrieval and ReMem with and without outcome feedback on LoCoMo."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from copy import copy, deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from remem.api import ReMem, default_config
from remem.config import to_dict
from remem.embedders import SentenceTransformerEmbedder
from remem.offline import set_tokenizer

SYSTEMS = ("plain_search", "feedback_off", "feedback_on")
TOP_K, POOL_SIZE = 5, 30
MEMORY_BUDGET, MODEL_CONTEXT, MAX_OUTPUT = 400, 1024, 48
SPLIT_SEED, GENERATION_SEED = 42, 0
TASK = "conversation_qa"
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
METRICS = ("accuracy", "answer_f1", "evidence_hit", "evidence_recall", "evidence_mrr")


@dataclass(frozen=True)
class Question:
    index: int
    text: str
    answer: str
    evidence: frozenset[str]
    category: int


def normalize(text: str) -> list[str]:
    words = re.findall(r"\w+", text.casefold())
    return [word for word in words if word not in {"a", "an", "the"}]


def grade(answer: str, reference: str) -> tuple[bool, float]:
    predicted, expected = Counter(normalize(answer)), Counter(normalize(reference))
    overlap = sum((predicted & expected).values())
    return bool(expected) and predicted == expected, 2 * overlap / max(1, predicted.total() + expected.total())


def make_split(sample: dict, turn_ids: set[str]) -> tuple[list[Question], list[Question]]:
    """Hash questions independently of their position, answer, and retrieval results."""
    eligible = []
    seen = set()
    for i, item in enumerate(sample["qa"]):
        evidence = frozenset(item.get("evidence", []))
        key = " ".join(normalize(item["question"]))
        answer = str(item.get("answer", ""))
        if item["category"] in (1, 4) and evidence and evidence <= turn_ids and normalize(answer) and key not in seen:
            eligible.append(Question(i, item["question"], answer, evidence, item["category"]))
            seen.add(key)

    def order(q: Question) -> bytes:
        return hashlib.sha256(f"{SPLIT_SEED}/{sample['sample_id']}/{q.text}".encode()).digest()

    warmup, held_out = [], []
    for category, warmup_count in ((1, 3), (4, 9)):
        group = sorted((q for q in eligible if q.category == category), key=order)
        if len(group) < 4 + warmup_count:
            raise ValueError(f"{sample['sample_id']} lacks category {category} questions for the fixed split")
        held_out.extend(group[:4])
        warmup.extend(group[4:4 + warmup_count])
    return sorted(warmup, key=order), sorted(held_out, key=order)


def conversation_turns(conversation: dict) -> dict[str, tuple[float, str]]:
    turns = {}
    for key, session in conversation.items():
        if not re.fullmatch(r"session_\d+", key):
            continue
        date = datetime.strptime(conversation[f"{key}_date_time"], "%I:%M %p on %d %B, %Y").replace(tzinfo=timezone.utc)
        for turn in session:
            text = " ".join(turn["text"].split())
            if text:
                turns[turn["dia_id"]] = (date.timestamp(), f"{turn['speaker']}: {text}")
    return turns


def system_configs():
    on = replace(default_config(top_k=TOP_K, budget_tokens=MEMORY_BUDGET), pool_size=POOL_SIZE)
    off = replace(on, outcome_feedback=False)
    plain = replace(off, ranker="sim", contradiction=replace(off.contradiction, enabled=False),
                    retention=replace(off.retention, mode="none"),
                    selection=replace(off.selection, mode="rank_pack", guard_top=0, min_sim=0.0))
    return dict(zip(SYSTEMS, (plain, off, on)))


def frozen_search(memory: ReMem, query: str, now: float):
    # search() writes exposure history. Copy the in-memory system for each test query,
    # sharing the read-only corpus, index, and embedder across the resulting one-shot searches.
    snapshot = copy(memory)
    snapshot.system = deepcopy(memory.system, {id(memory.embedder): memory.embedder})
    snapshot._pending = {}
    return snapshot.search(query, task=TASK, now=now)


def answer_prompt(question, hits) -> str:
    context = "\n".join(f"[{hit.id}] {hit.text}" for hit in hits)
    return ("Answer the question using only the conversation excerpts. Give a short answer, with no explanation. "
            "If the excerpts do not give the answer, write Unknown. After the answer list only the excerpt IDs "
            "you relied on. Use exactly two lines: 'Answer: ...' and 'Evidence: ...'.\n\n"
            f"Excerpts:\n{context}\n\nQuestion: {question}")


def parse_response(response: str, served_ids: list[str]) -> tuple[str, list[str]]:
    parts = re.split(r"\b(Answer|Evidence)\s*:", response, flags=re.I)
    fields = {label.casefold(): value.strip() for label, value in zip(parts[1::2], parts[2::2])}
    answer = fields.get("answer", parts[0].strip())
    cited = set(re.findall(r"\bD\d+:\d+\b", fields.get("evidence", "").upper()))
    return answer, [ident for ident in served_ids if ident.upper() in cited]


class OllamaAnswerer:
    def __init__(self, url: str, model: str, threads: int, cache: Path) -> None:
        self.url, self.model, self.cache = url.rstrip("/"), model, cache
        self.options = {"temperature": 0, "seed": GENERATION_SEED, "num_ctx": MODEL_CONTEXT,
                        "num_predict": MAX_OUTPUT, "num_thread": threads}
        listed = self.request("tags")["models"]
        self.digest = next((item["digest"] for item in listed if item["name"] == model), None)
        if self.digest is None:
            raise ValueError(f"{model} is missing from the local Ollama server")
        self.version = self.request("version")["version"]
        self.cached = {}
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            with cache.open(encoding="utf-8") as handle:
                for line in handle:
                    key, value = json.loads(line)
                    self.cached[key] = value

    def request(self, endpoint: str, data: dict | None = None) -> dict:
        body = None if data is None else json.dumps(data).encode()
        request = Request(f"{self.url}/api/{endpoint}", data=body, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=120) as response:
            return json.load(response)

    def __call__(self, question: str, hits) -> dict:
        prompt = answer_prompt(question, hits)
        key = hashlib.sha256(json.dumps([self.digest, self.version, self.options, prompt]).encode()).hexdigest()
        if key not in self.cached:
            response = self.request("generate", {"model": self.model, "prompt": prompt,
                                                 "stream": False, "options": self.options})
            output = {"response": response["response"], "prompt_tokens": response["prompt_eval_count"],
                      "output_tokens": response["eval_count"], "stop_reason": response.get("done_reason")}
            if output["prompt_tokens"] > MODEL_CONTEXT - MAX_OUTPUT or output["output_tokens"] > MAX_OUTPUT:
                raise ValueError("Ollama reported token usage beyond the shared budget")
            with self.cache.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps([key, output]) + "\n")
            self.cached[key] = output
        return self.cached[key]


def run_conversation(sample, embedder, generate, log=print):
    turns = conversation_turns(sample["conversation"])
    warmup, held_out = make_split(sample, set(turns))
    now = max(date for date, _ in turns.values()) + 1
    embedder.warm([text for _, text in turns.values()] + [q.text for q in warmup])
    memories = {name: ReMem(embedder=embedder, config=cfg) for name, cfg in system_configs().items()}
    for memory in memories.values():
        for ident, (date, text) in turns.items():
            memory.add(text, role="note", fact_id=ident, when=date)

    training = []
    on = memories["feedback_on"]
    for question in warmup:
        searches = {name: memory.search(question.text, task=TASK, now=now) for name, memory in memories.items()}
        served = searches["feedback_on"]
        generated = generate(question.text, served.items)
        ids = [hit.id for hit in served.items]
        answer, citations = parse_response(generated["response"], ids)
        correct, _ = grade(answer, question.answer)
        credit = on.feedback(served.search_id, success=correct, used_ids=citations, answer=answer)
        training.append({"index": question.index, "correct": correct, "citations": citations, "credit": credit})

    embedder.warm([q.text for q in held_out])
    rows = []
    for question in held_out:
        pool, _ = on.backend.candidates(question.text, k=POOL_SIZE)
        gold = question.evidence
        prior = {ident: on.system.ledger.get(ident).n_credited for ident in gold}
        row = {"conversation": sample["sample_id"], "index": question.index, "category": question.category,
               "evidence_ids": sorted(gold), "pool_evidence_ids": sorted(gold & {c.fact_id for c in pool}),
               "prior_evidence_credit": prior, "systems": {}}
        for name, memory in memories.items():
            served = frozen_search(memory, question.text, now)
            ids = [hit.id for hit in served.items]
            generated = generate(question.text, served.items)
            answer, citations = parse_response(generated["response"], ids)
            correct, f1 = grade(answer, question.answer)
            matches = [i for i, ident in enumerate(ids, 1) if ident in gold]
            row["systems"][name] = {
                "accuracy": correct, "answer_f1": f1, "evidence_hit": bool(matches),
                "evidence_recall": len(matches) / len(gold), "evidence_mrr": 1 / matches[0] if matches else 0,
                "answer": answer, "citations": citations, "retrieved_ids": ids,
                "retrieval": [{"id": hit.id, "similarity": hit.similarity, "score": hit.score, "why": hit.why,
                               "state": hit.state} for hit in served.items],
                "memory_tokens": served.tokens, "prompt_tokens": generated["prompt_tokens"],
                "output_tokens": generated["output_tokens"], "stop_reason": generated["stop_reason"],
            }
        rows.append(row)
    log(f"{sample['sample_id']}: {len(training)} warm-up, {len(rows)} held out", flush=True)
    return training, rows


def summarize(rows):
    summary = {name: {key: mean(row["systems"][name][key] for row in rows)
                      for key in (*METRICS, "memory_tokens", "prompt_tokens", "output_tokens")} for name in SYSTEMS}
    return summary


def paired_deltas(rows):
    groups = sorted({row["conversation"] for row in rows})
    rng = np.random.default_rng(0)
    draws = rng.integers(len(groups), size=(10_000, len(groups)))
    report = {}
    for name, baseline in (("feedback_off", "plain_search"), ("feedback_on", "plain_search"),
                           ("feedback_on", "feedback_off")):
        metrics = {}
        for key in METRICS:
            deltas = [float(row["systems"][name][key]) - float(row["systems"][baseline][key]) for row in rows]
            per_group = np.array([mean(d for row, d in zip(rows, deltas) if row["conversation"] == group) for group in groups])
            metrics[key] = {"delta": mean(deltas), "ci95": np.quantile(per_group[draws].mean(axis=1), [0.025, 0.975]).tolist(),
                            "wins": sum(d > 0 for d in deltas), "losses": sum(d < 0 for d in deltas)}
        report[f"{name}_minus_{baseline}"] = metrics
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "feedback_ablation")
    parser.add_argument("--cache", type=Path, default=ROOT / "results" / "cache" / "feedback_answers.jsonl")
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--embedder", default=EMBEDDER, help="Sentence Transformers model ID or snapshot path")
    args = parser.parse_args()

    raw = args.data.read_bytes()
    samples = json.loads(raw)
    if not samples:
        parser.error("dataset contains no conversations")
    set_tokenizer("proxy")
    answerer = OllamaAnswerer(args.url, args.model, args.threads, args.cache)
    embedder = SentenceTransformerEmbedder(args.embedder)
    args.out.mkdir(parents=True, exist_ok=True)
    training, rows = {}, []
    questions_path = args.out / "per_question.jsonl"
    progress_path = questions_path.with_suffix(".jsonl.tmp")
    with progress_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            trace, results = run_conversation(sample, embedder, answerer)
            training[sample["sample_id"]] = trace
            rows.extend(results)
            for row in results:
                handle.write(json.dumps(row) + "\n")
            handle.flush()

    partitions = {"credited_evidence": [row for row in rows if any(row["prior_evidence_credit"].values())],
                  "no_credited_evidence": [row for row in rows if not any(row["prior_evidence_credit"].values())]}
    configs = {name: to_dict(cfg) for name, cfg in system_configs().items()}
    summary = {"dataset": args.data.name, "dataset_sha256": hashlib.sha256(raw).hexdigest(),
               "conversations": len(training), "held_out_questions": len(rows),
               "warmup_questions": sum(len(trace) for trace in training.values()),
               "warmup_correct": sum(turn["correct"] for trace in training.values() for turn in trace),
               "model": answerer.model, "model_digest": answerer.digest, "ollama_version": answerer.version,
               "model_options": answerer.options, "embedder": args.embedder, "split_seed": SPLIT_SEED,
               "memory_budget_proxy_tokens": MEMORY_BUDGET, "system_configs": configs,
               "pool_evidence_hit": mean(bool(row["pool_evidence_ids"]) for row in rows),
               "summary": summarize(rows), "paired": paired_deltas(rows),
               "by_prior_evidence_credit": {key: {"n": len(group), "summary": summarize(group)}
                                            for key, group in partitions.items() if group}}
    progress_path.replace(questions_path)
    (args.out / "warmup.json").write_text(json.dumps(training, indent=2) + "\n", encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["summary"], indent=2))


if __name__ == "__main__":
    main()
