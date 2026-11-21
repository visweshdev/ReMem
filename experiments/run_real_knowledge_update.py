"""Real-data test of outdated-fact handling: LongMemEval knowledge-update questions, a real LLM, no simulator.

Every one of LongMemEval's 72 non-abstention "knowledge-update" questions has a user who states a value in one session
and a newer value in a later one (e.g. a 5K personal best of 27:12, later 25:50). The question asks for the
current value. This is the real-world version of what ReMem's supersession is meant to fix, so it is tested here
end to end with nothing synthetic: real chat history, a real (small, local) LLM extracting facts, answering and
judging.

Pipeline per question
  1. Candidate pool: the 15 user turns most similar to the question (MiniLM), dated no later than the question.
     All systems see the same pool. (Extraction is only run on the pool to keep CPU cost manageable; a deployed
     system would extract every turn at write time. The extractor never sees the question.)
  2. Fact extraction (systems 2-4): each pool turn, in chronological order, goes through remem.extract's prompt,
     plus the list of entity.attribute keys already used in this user's memory, so the model can reuse a key
     for the same thing. Keys are whatever the LLM writes: no hand-written schema.
  3. Systems, each serving k=5 memories:
       raw_similarity          the 5 most similar raw user turns (plain RAG over chat history)
       facts_similarity        extracted facts, similarity only
       facts_newest_key        extracted facts, similarity + a newer fact on the same (entity, attribute) always
                               replaces the older one (ReMem 0.2's rule)
       facts_remem_no_source   extracted facts, ReMem defaults: a same-key fact is retired only if the LLM confirms
                               the newer one is an update (remem.extract.LLMUpdateChecker)
       facts_remem             the same, and every fact is shown with the sentence it was extracted from
     ReMem never receives feedback here: every question is asked once, so outcome learning cannot be tested on this
     data. This is a test of fact replacement on real text only.
  4. Answer: the same LLM, the same prompt for every system, memories shown with their dates.
  5. Judge: a larger local model (default qwen3-coder:30b, run after the 3B phase because both do not fit in RAM)
     with LongMemEval's knowledge-update judging rule (an answer that mentions the old value but gives the updated
     one counts as correct). The 3B model was tried as judge first and rejected: it marked "25:50" wrong against
     "25 minutes and 50 seconds (or 25:50)". Hypotheses are also written for the official GPT-4o judge.
  Judge-free metrics from provenance: whether the context holds the newest evidence turn, and whether it holds
  only the outdated evidence ("stale only").

  Split: questions are ordered by sha1(question_id); the first half is `dev` (for design choices), the second
  half `test` (run once, reported).

    ollama pull qwen2.5:3b && ollama pull qwen3-coder:30b
    python experiments/run_real_knowledge_update.py --data data/longmemeval_s_cleaned.json --split dev
    python experiments/run_real_knowledge_update.py --data data/longmemeval_s_cleaned.json --split test
On a CPU-only machine with little RAM, Ollama's default 32k context is slow; serve 4k-context copies of the same weights
(Modelfile: `FROM qwen2.5:3b` + `PARAMETER num_ctx 4096`) and pass --serve-model / --serve-judge with their tags.
Resumable: every LLM call is cached in <out>/llm_cache.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_longmemeval import parse_date  # noqa: E402
from remem.api import ReMem, default_config  # noqa: E402
from remem.backends import VectorBackend  # noqa: E402
from remem.extract import PROMPT, LLMUpdateChecker, parse_facts  # noqa: E402

SYSTEMS = ("raw_similarity", "facts_similarity", "facts_newest_key", "facts_remem_no_source", "facts_remem")
POOL, K = 15, 5
SOURCE_CHARS = 300     # how much of the original sentence is shown next to a fact


def split_questions(questions: list[dict], split: str) -> list[dict]:
    """Deterministic half split by sha1(question_id): 'dev' (design choices), 'test' (reported), 'all'."""
    if split == "all":
        return questions
    ordered = sorted(questions, key=lambda x: hashlib.sha1(x["question_id"].encode()).hexdigest())
    half = len(ordered) // 2
    return ordered[:half] if split == "dev" else ordered[half:]

ANSWER_PROMPT = """Today is {date}. Below are memories about the user from earlier conversations, each with the date it was recorded.
{ctx}

Question: {question}
Answer concisely using only these memories. If memories disagree, the most recent one is the current truth."""

# adapted from LongMemEval's evaluate_qa.py (knowledge-update rule)
JUDGE_PROMPT = """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only."""


class CachedLLM:
    def __init__(self, client, model: str, cache: Path, serve_as: str | None = None) -> None:
        # `model` names the weights (and keys the cache); `serve_as` is the Ollama tag actually called, e.g. a copy of
        # the same weights with a smaller context window (same outputs for prompts that fit, much less RAM)
        self.client, self.model, self.path, self.serve_as = client, model, cache, serve_as or model
        self.lock = threading.Lock()
        self.cache: dict[str, str] = {}
        if cache.exists():
            for line in cache.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.cache[r["k"]] = r["v"]

    def __call__(self, prompt: str, max_tokens: int = 400) -> str:
        k = hashlib.sha1(f"{self.model}\n{max_tokens}\n{prompt}".encode()).hexdigest()
        if k in self.cache:
            return self.cache[k]
        r = self.client.chat.completions.create(model=self.serve_as, temperature=0, max_tokens=max_tokens,
                                                messages=[{"role": "user", "content": prompt}])
        v = (r.choices[0].message.content or "").strip()
        with self.lock:
            self.cache[k] = v
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"k": k, "v": v}) + "\n")
        return v


def day(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def make_memory(system: str, embedder, llm=None) -> ReMem:
    cfg = default_config(top_k=K)
    if system == "facts_similarity":
        cfg = replace(cfg, ranker="sim", outcome_feedback=False, contradiction=replace(cfg.contradiction, enabled=False),
                      retention=replace(cfg.retention, mode="none"),
                      selection=replace(cfg.selection, mode="rank_pack", guard_top=0, min_sim=0.0))
    elif system == "facts_newest_key":
        cfg = replace(cfg, ranker="sim", outcome_feedback=False,
                      contradiction=replace(cfg.contradiction, learn_schema=False, outcome_invalidation=False,
                                            conf_half_life_days=0.0),
                      retention=replace(cfg.retention, mode="none"),
                      selection=replace(cfg.selection, mode="rank_pack", guard_top=0, min_sim=0.0))
    checker = LLMUpdateChecker(None, "", complete=llm) if system.startswith("facts_remem") and llm is not None else None
    return ReMem(None, embedder=embedder, backend=VectorBackend(embedder), config=cfg, autosave=False,
                 update_checker=checker)


def run_question(inst: dict, embedder, llm: CachedLLM, emb_lock: threading.Lock) -> dict:
    q_time = parse_date(inst["question_date"])
    turns = []   # (text, time, is_evidence)
    for date, session in zip(inst["haystack_dates"], inst["haystack_sessions"]):
        t = parse_date(date)
        if t > q_time:
            continue
        for j, turn in enumerate(session):
            if turn["role"] == "user" and len(turn["content"]) >= 8:
                turns.append((turn["content"][:1500], t + j, bool(turn.get("has_answer"))))
    with emb_lock:
        E = np.asarray(embedder.embed_texts([x[0] for x in turns]))
        qv = np.asarray(embedder.embed_one(inst["question"]))
    order = list(np.argsort(-(E @ qv))[:POOL])
    pool = [turns[i] for i in order]                                  # similarity order
    ev_times = sorted(t for _, t, e in turns if e)
    newest_ev = ev_times[-1] if ev_times else None
    old_ev = set(ev_times[:-1])

    # extraction at write time: chronological, question never shown
    facts = []   # (text, entity, attribute, role, time, source_text)
    keys: list[str] = []
    for text, t, _ in sorted(pool, key=lambda x: x[1]):
        hint = ("\nKeys already used in this user's memory (reuse one if the fact is about the same thing): "
                + ", ".join(keys) + "\n") if keys else ""
        reply = llm(PROMPT.replace("Conversation:\n", hint + "Conversation:\n") + "user: " + text, 400)
        for f in parse_facts(reply):
            facts.append((f.text, f.entity, f.attribute, f.role, t, text))
            if f.entity and f.attribute and f"{f.entity}.{f.attribute}" not in keys:
                keys.append(f"{f.entity}.{f.attribute}")

    with emb_lock:   # embed every fact once up front, so add() below only reads the embedder's cache
        if hasattr(embedder, "warm"):
            embedder.warm([f[0] for f in facts])

    out = {"question_id": inst["question_id"], "question": inst["question"], "answer": inst["answer"],
           "n_facts": len(facts), "n_keys": len(keys), "systems": {}}
    for system in SYSTEMS:
        if system == "raw_similarity":
            served = [(text, t) for text, t, _ in pool[:K]]
            sources = [t for _, t in served]
        else:
            update = "replace" if system == "facts_newest_key" else "check" if system.startswith("facts_remem") else "keep"
            mem = make_memory(system, embedder, llm)
            for i, (text, ent, att, role, t, src_text) in enumerate(facts):
                mem.add(text, entity=ent, attribute=att, role=role, when=t, fact_id=f"f{i}", update=update,
                        source=src_text)
            with emb_lock:
                res = mem.search(inst["question"], k=K, now=q_time + 1)
            if system == "facts_remem":
                served = [(f"{h.text} (said: \"{(h.source or '')[:SOURCE_CHARS].strip()}\")", h.time) for h in res.items]
            else:
                served = [(h.text, h.time) for h in res.items]
            sources = [h.time for h in res.items]
        ctx = "\n".join(f"- [{day(t)}] {text}" for text, t in sorted(served, key=lambda x: x[1])) or "- (no memories)"
        hyp = llm(ANSWER_PROMPT.format(date=day(q_time), ctx=ctx, question=inst["question"]), 150)
        has_new = newest_ev in sources
        out["systems"][system] = {
            "hypothesis": hyp,
            "newest_evidence_in_context": has_new,
            "stale_only": (not has_new) and any(s in old_ev for s in sources),
            "n_served": len(served), "context": ctx,
        }
    return out


def paired_ci(a: list[int], b: list[int], n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = random.Random(seed)
    boots = sorted(float(np.mean([d[rng.randrange(len(d))] for _ in d])) for _ in range(n_boot))
    return float(d.mean()), boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--model", default="qwen2.5:3b", help="extracts facts and answers")
    ap.add_argument("--judge-model", default="qwen3-coder:30b")
    ap.add_argument("--serve-model", help="Ollama tag to call for --model (e.g. a 4k-context copy: see README)")
    ap.add_argument("--serve-judge", help="Ollama tag to call for --judge-model")
    ap.add_argument("--url", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--split", choices=["dev", "test", "all"], default="test")
    ap.add_argument("--out", type=Path, default=None, help="default: results/real_knowledge_update/<split>")
    ap.add_argument("--cache", type=Path, default=ROOT / "results" / "real_knowledge_update" / "llm_cache.jsonl")
    a = ap.parse_args()

    from openai import OpenAI

    from remem.embedders import SentenceTransformerEmbedder
    data = json.loads(a.data.read_text(encoding="utf-8"))
    ku = [x for x in data if x["question_type"] == "knowledge-update" and not x["question_id"].endswith("_abs")]
    ku = split_questions(ku, a.split)[: a.limit]
    a.out = a.out or ROOT / "results" / "real_knowledge_update" / a.split
    a.out.mkdir(parents=True, exist_ok=True)
    a.cache.parent.mkdir(parents=True, exist_ok=True)
    llm = CachedLLM(OpenAI(base_url=a.url, api_key="ollama"), a.model, a.cache, a.serve_model)
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    lock = threading.Lock()

    rows = []
    with ThreadPoolExecutor(a.threads) as ex:
        for i, r in enumerate(ex.map(lambda inst: run_question(inst, embedder, llm, lock), ku), 1):
            rows.append(r)
            print(f"{i}/{len(ku)} answered", flush=True)

    # phase 2: judge (one model loaded at a time; sequential keeps the big model from thrashing)
    judge = CachedLLM(llm.client, a.judge_model, a.cache, a.serve_judge)
    for i, r in enumerate(rows, 1):
        for s in SYSTEMS:
            v = judge(JUDGE_PROMPT.format(question=r["question"], answer=r["answer"],
                                          response=r["systems"][s]["hypothesis"]), 5)
            r["systems"][s]["judge_reply"] = v
            r["systems"][s]["correct"] = v.lower().lstrip().startswith("yes")
        if i % 10 == 0:
            print(f"{i}/{len(rows)} judged", flush=True)

    n = len(rows)
    summary = {}
    for s in SYSTEMS:
        col = [x["systems"][s] for x in rows]
        summary[s] = {"accuracy": sum(c["correct"] for c in col) / n,
                      "newest_evidence_in_context": sum(c["newest_evidence_in_context"] for c in col) / n,
                      "stale_only_context": sum(c["stale_only"] for c in col) / n,
                      "mean_served": float(np.mean([c["n_served"] for c in col]))}
    paired = {}
    for s in SYSTEMS[1:]:
        for base in SYSTEMS:
            if SYSTEMS.index(base) >= SYSTEMS.index(s):
                continue
            m, lo, hi = paired_ci([int(x["systems"][s]["correct"]) for x in rows],
                                  [int(x["systems"][base]["correct"]) for x in rows])
            paired[f"{s} - {base}"] = {"delta": m, "ci95": [lo, hi]}
    meta = {"data": a.data.name, "split": a.split, "model": a.model, "judge_model": a.judge_model, "n_questions": n,
            "pool": POOL, "k": K, "source_chars": SOURCE_CHARS,
            "question_ids": [x["question_id"] for x in rows],
            "mean_facts_per_question": float(np.mean([x["n_facts"] for x in rows])),
            "mean_keys_per_question": float(np.mean([x["n_keys"] for x in rows]))}
    (a.out / "summary.json").write_text(json.dumps({**meta, "summary": summary, "paired_accuracy": paired}, indent=1),
                                        encoding="utf-8")
    with open(a.out / "per_question.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    for s in SYSTEMS:
        with open(a.out / f"hyp_{s}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({"question_id": r["question_id"], "hypothesis": r["systems"][s]["hypothesis"]}) + "\n")
    print(json.dumps({**meta, "summary": summary, "paired_accuracy": paired}, indent=1))


if __name__ == "__main__":
    main()
