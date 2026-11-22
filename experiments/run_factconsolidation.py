"""Run a retrieval-only transfer check on MemoryAgentBench's 6k single-hop conflicts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_real_knowledge_update import CachedLLM
from remem.api import ReMem, default_config
from remem.embedders import SentenceTransformerEmbedder
from remem.extract import LLMUpdateChecker

SOURCE = "factconsolidation_sh_6k"
ARMS = {"plain_search": "keep", "newest_per_key": "replace", "source_checked": "check"}
PREDICATE = re.compile(
    r" (?:was born in the city of|died in the city of|worked in the city of|"
    r"is located in the continent of|was created in the country of|is a citizen of|"
    r"is associated with the sport of|plays the position of|is married to|is employed by|is famous for|is) ",
    re.I,
)


def extract_slot(text: str) -> tuple[str, str]:
    match = PREDICATE.search(text)
    if match is None:
        return "", ""
    return text[:match.end()].strip().casefold(), text[match.end():].rstrip(".")


def evaluate(sample: dict, embedder, complete) -> dict:
    records = []
    for line in sample["context"].splitlines():
        match = re.fullmatch(r"(\d+)\. (.+)", line.strip())
        if match:
            text = match[2]
            key, value = extract_slot(text)
            records.append((match[1], text, key, value))
    embedder.warm([text for _, text, _, _ in records] + sample["questions"])
    cfg = default_config(top_k=5)
    cfg = replace(cfg, ranker="sim", outcome_feedback=False, retention=replace(cfg.retention, mode="none"),
                  contradiction=replace(cfg.contradiction, conf_half_life_days=0),
                  selection=replace(cfg.selection, mode="rank_pack", guard_top=0, min_sim=0))
    checker = LLMUpdateChecker(None, "", complete=complete)
    arms = {}
    for arm, policy in ARMS.items():
        memory = ReMem(embedder=embedder, config=cfg, update_checker=checker if policy == "check" else None)
        for ident, text, key, _ in records:
            memory.add(text, entity=key or None, attribute="slot" if key else None, role="state" if key else "note",
                       fact_id=ident, when=float(ident), update=policy, source=text)
        arms[arm] = memory
    values = {ident: value.casefold() for ident, _, _, value in records}
    rows = []
    for ident, question, refs in zip(sample["metadata"]["qa_pair_ids"], sample["questions"], sample["answers"]):
        gold = {ref.casefold().strip().rstrip(".") for ref in refs}
        systems = {}
        for arm, memory in arms.items():
            retrieved = [hit.id for hit in memory.search(question, now=len(records)).items]
            found = [values[r] for r in retrieved]
            systems[arm] = {"top1_value_match": bool(found and found[0] in gold), "top5_value_hit": bool(gold.intersection(found)),
                            "retrieved_ids": retrieved}
        rows.append({"id": ident, "systems": systems})
    return {"source": SOURCE, "records": len(records), "parsed_slots": sum(bool(key) for _, _, key, _ in records),
            "questions": len(rows), "summary": {arm: {
                "top1_value_match": mean(r["systems"][arm]["top1_value_match"] for r in rows),
                "top5_value_hit": mean(r["systems"][arm]["top5_value_hit"] for r in rows),
                "retirements": len(memory.system.detector.conflicts),
            } for arm, memory in arms.items()}, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--serve-model", default="qwen2.5-3b-4k")
    parser.add_argument("--url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    import pyarrow.parquet as parquet
    from openai import OpenAI

    table = parquet.read_table(args.data).to_pylist()
    sample, = (r for r in table if r["metadata"]["source"] == SOURCE)
    cache = ROOT / "results" / "cache" / "factconsolidation.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    complete = CachedLLM(OpenAI(base_url=args.url, api_key="ollama"), args.model, cache, args.serve_model, seed=0)
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    results = evaluate(sample, embedder, complete)
    results.update(data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(), model=args.model,
                   serve_model=args.serve_model, embedder="all-MiniLM-L6-v2", seed=0, temperature=0, usage=dict(complete.usage))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
