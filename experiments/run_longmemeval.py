"""LongMemEval (ICLR 2025) retrieval evaluation, plus optional QA with a real LLM.

Data (download yourself; ~hundreds of MB):
    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
    (format: https://github.com/xiaowu0162/LongMemEval#-dataset-format)

Retrieval only (no LLM needed; uses the dataset's evidence labels):
    python experiments/run_longmemeval.py --data data/longmemeval_s_cleaned.json --embedder st --limit 100

Retrieval + QA hypotheses for the official judge (costs LLM calls):
    python experiments/run_longmemeval.py --data ... --qa --model <chat-model>
    # then, in the LongMemEval repo:  python3 src/evaluation/evaluate_qa.py gpt-4o results/longmemeval/hyp_<system>.jsonl data/longmemeval_oracle.json

Each question has its own haystack and is asked once, so ReMem's per-memory feedback can never accumulate
here; contradiction handling also needs entity/attribute keys, which raw turns do not have. Expect ReMem to behave
like the static-score baseline on this benchmark. That is the honest test of what transfers without feedback.

Metrics (abstention questions, whose ids end in `_abs`, are excluded because they have no evidence):
    turn_recall@k     at least one evidence turn (has_answer=true) is among the k retrieved memories
    session_recall@k  at least one retrieved memory comes from an evidence session (answer_session_ids)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from remem.api import ReMem, default_config  # noqa: E402
from remem.backends import VectorBackend  # noqa: E402

SYSTEMS = ("similarity", "static", "remem")


def parse_date(s: str) -> float:
    """LongMemEval dates look like '2023/05/20 (Sat) 02:21'."""
    s = re.sub(r"\s*\([A-Za-z]{3}\)\s*", " ", str(s)).strip()
    for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    raise ValueError(f"unrecognised LongMemEval date {s!r}")


def make_memory(system: str, embedder, k: int) -> ReMem:
    from dataclasses import replace
    cfg = default_config(top_k=k)
    if system == "similarity":
        cfg = replace(cfg, ranker="sim", outcome_feedback=False,
                      selection=replace(cfg.selection, mode="rank_pack", guard_top=0, min_sim=0.0))
    elif system == "static":
        cfg = replace(cfg, ranker="static", outcome_feedback=False,
                      selection=replace(cfg.selection, mode="rank_pack", guard_top=0, min_sim=0.0))
    cfg = replace(cfg, pool_size=max(cfg.pool_size, 4 * k))
    return ReMem(None, embedder=embedder, backend=VectorBackend(embedder), config=cfg, autosave=False)


def evaluate(instances: list[dict], embedder, ks=(5, 10), systems=SYSTEMS, qa_agent=None, log=print) -> dict:
    kmax = max(ks)
    out = {s: {f"turn_recall@{k}": 0 for k in ks} | {f"session_recall@{k}": 0 for k in ks} for s in systems}
    by_type: dict[str, dict[str, list[int]]] = {}
    hyps = {s: [] for s in systems}
    n = 0
    for inst in instances:
        if str(inst["question_id"]).endswith("_abs"):
            continue
        n += 1
        answer_sessions = set(inst.get("answer_session_ids", []))
        q_time = parse_date(inst["question_date"]) if inst.get("question_date") else time.time()
        if hasattr(embedder, "warm"):              # batch-embed this question's whole history once, shared by all systems
            embedder.warm([f"{turn['role']}: {turn['content']}".strip()[:4000] for session in inst["haystack_sessions"]
                           for turn in session] + [inst["question"]])
        for system in systems:
            mem = make_memory(system, embedder, kmax)
            evidence, session_of = set(), {}
            for si, (sid, date, session) in enumerate(zip(inst["haystack_session_ids"], inst["haystack_dates"], inst["haystack_sessions"])):
                t = parse_date(date)
                for j, turn in enumerate(session):
                    text = f"{turn['role']}: {turn['content']}".strip()
                    if len(text) < 8:
                        continue
                    fid = mem.add(text[:4000], when=t, role="note", fact_id=f"{si}:{sid}::{j}")   # session ids can repeat within a haystack
                    session_of[fid] = sid
                    if turn.get("has_answer"):
                        evidence.add(fid)
            res = mem.search(inst["question"], k=kmax, now=q_time)     # recency relative to the question date
            ids = [h.id for h in res.items]
            for k in ks:
                turn_hit = int(any(i in evidence for i in ids[:k]))
                sess_hit = int(any(session_of[i] in answer_sessions for i in ids[:k]))
                out[system][f"turn_recall@{k}"] += turn_hit
                out[system][f"session_recall@{k}"] += sess_hit
                by_type.setdefault(inst["question_type"], {}).setdefault(f"{system}|turn_recall@{k}", []).append(turn_hit)
            if qa_agent is not None:
                hyps[system].append({"question_id": inst["question_id"],
                                     "hypothesis": qa_agent(inst["question"], [h.text for h in res.items[:kmax]], q_time)})
        if n % 10 == 0:
            log(f"{n} questions: " + ", ".join(f"{s} turn@{ks[0]}={out[s][f'turn_recall@{ks[0]}'] / n:.3f}" for s in systems))
    summary = {s: {m: v / max(1, n) for m, v in d.items()} for s, d in out.items()}
    per_type = {t: {m: sum(v) / len(v) for m, v in d.items()} for t, d in by_type.items()}
    return {"n_questions": n, "summary": summary, "per_type": per_type, "hypotheses": hyps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    ap.add_argument("--embedder", choices=["hash", "st"], default="st")
    ap.add_argument("--st-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--qa", action="store_true", help="also write QA hypotheses with an LLM (needs OPENAI_API_KEY)")
    ap.add_argument("--model", help="chat model for --qa")
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "longmemeval")
    a = ap.parse_args()

    instances = json.loads(a.data.read_text(encoding="utf-8"))[: a.limit]
    if a.embedder == "st":
        from remem.embedders import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder(a.st_model)
    else:
        from remem.offline import ConceptHashEmbedder
        embedder = ConceptHashEmbedder(256)

    qa_agent = None
    if a.qa:
        import os
        if not (a.model and os.environ.get("OPENAI_API_KEY")):
            sys.exit("--qa needs --model and OPENAI_API_KEY (and OPENAI_BASE_URL for a compatible server)")
        from openai import OpenAI
        client = OpenAI()

        def qa_agent(question, memories, q_time):
            date = datetime.fromtimestamp(q_time, tz=timezone.utc).strftime("%Y-%m-%d")
            ctx = "\n".join(f"- {m}" for m in memories)
            r = client.chat.completions.create(model=a.model, temperature=0, max_tokens=200, messages=[
                {"role": "user", "content": f"Today is {date}. Answer using only these chat-history excerpts.\n{ctx}\n\n"
                                            f"Question: {question}\nAnswer concisely."}])
            return r.choices[0].message.content.strip()

    res = evaluate(instances, embedder, systems=tuple(a.systems.split(",")), qa_agent=qa_agent)
    a.out.mkdir(parents=True, exist_ok=True)
    meta = {"data": a.data.name, "limit": a.limit, "embedder": a.embedder if a.embedder == "hash" else a.st_model}
    (a.out / "retrieval.json").write_text(json.dumps({**meta, **{k: v for k, v in res.items() if k != "hypotheses"}},
                                                     indent=1), encoding="utf-8")
    if a.qa:
        for s, rows in res["hypotheses"].items():
            with open(a.out / f"hyp_{s}.jsonl", "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
    print(json.dumps(res["summary"], indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
