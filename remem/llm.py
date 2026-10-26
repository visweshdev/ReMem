"""Adapters for running ReMem with a real OpenAI-compatible model instead of the simulated agent.

    OpenAIEmbedder   same interface as ConceptHashEmbedder (`dim`, `embed_one`, `embed_texts`)
    LLMAgent         same interface as SimAgent (`act(task, items) -> AgentResult`)

STATUS: unit-tested against a fake client only (tests/test_llm_adapters.py). These adapters have **not**
been run against a live endpoint in this repository's development environment, so no reported number
uses them. `experiments/run_llm_agent.py` is the entry point for doing that on your own machine.

The agent sees only the served memories (ids, timestamps, text) and the question. It is asked for JSON
`{"answers": {"<subject>.<attribute>": "<value>"}, "cited": ["<memory id>", ...]}`. Success is scored by the
evaluation harness against the world's hidden truth, exactly as for the simulated agent, and the memory layer
only ever receives the trace and the boolean outcome.
"""

from __future__ import annotations

import json
import re
from typing import Any

from remem.eval.agent import AgentResult
from remem.eval.world import Task, World
from remem.types import AgentTrace

SYSTEM_PROMPT = (
    "You answer questions about a software project using ONLY the memory snippets provided. "
    "Each memory has an id and a date. If memories conflict, prefer the one that is valid for the question. "
    "If the memories do not contain the answer, give your best guess. "
    'Reply with JSON only: {"answers": {"<subject>.<attribute>": "<value>"}, "cited": ["<memory id>", ...]}. '
    "List in `cited` exactly the memory ids you relied on."
)


class OpenAIEmbedder:
    def __init__(self, client: Any, model: str, dim: int) -> None:
        self.client, self.model, self.dim = client, model, int(dim)

    def embed_texts(self, texts: list[str], *, batch_size: int = 128) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            resp = self.client.embeddings.create(model=self.model, input=texts[i : i + batch_size])
            out.extend(_normalise(list(d.embedding)) for d in resp.data)
        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


def _normalise(v: list[float]) -> list[float]:
    n = sum(x * x for x in v) ** 0.5
    return v if n < 1e-9 else [x / n for x in v]


def build_prompt(task: Task, items: list[tuple[str, str, float]], t0: float) -> str:
    lines = []
    for fid, text, ts in items:
        day = int((ts - t0) // 86400)
        lines.append(f"[{fid}] (day {day}) {text}")
    slots = ", ".join(f"{s}.{a}" for s, a in task.slots)
    ctx = "\n".join(lines) if lines else "(no memories)"
    asof = f"\nAnswer as of day {int((task.as_of - t0) // 86400)}." if task.as_of is not None else ""
    return f"Memories:\n{ctx}\n\nQuestion: {task.query}{asof}\nAnswer slots needed: {slots}"


def parse_reply(text: str) -> tuple[dict[str, str], list[str]]:
    """Tolerant JSON extraction; returns ({}, []) when nothing usable is found (scored as a failure)."""
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return {}, []
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}, []
    answers = d.get("answers", {})
    cited = d.get("cited", [])
    if not isinstance(answers, dict) or not isinstance(cited, list):
        return {}, []
    return {str(k): str(v).strip() for k, v in answers.items()}, [str(c) for c in cited]


class LLMAgent:
    def __init__(self, world: World, client: Any, model: str, *, t0: float, temperature: float = 0.0, max_tokens: int = 200,
                 json_mode: bool = False) -> None:
        self.world, self.client, self.model, self.t0 = world, client, model, t0
        self.temperature, self.max_tokens = temperature, max_tokens
        self.n_calls = 0
        self.json_mode = json_mode           # response_format=json_object (supported by OpenAI and Ollama)
        self.parse_failures = 0

    def act(self, task: Task, items: list[tuple[str, str, float]]) -> AgentResult:
        extra = {"response_format": {"type": "json_object"}} if self.json_mode else {}
        resp = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens, **extra,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": build_prompt(task, items, self.t0)}],
        )
        self.n_calls += 1
        answers, cited = parse_reply(resp.choices[0].message.content)
        if not answers:
            self.parse_failures += 1
        truth = self.world.answer(task)
        success = all(answers.get(k, "").lower() == v.lower() for k, v in truth.items())
        served = {fid for fid, _, _ in items}
        cited = [c for c in cited if c in served]            # ignore ids the agent made up
        trace = AgentTrace(answer=answers, cited_ids=cited, tool_args=dict(answers))
        return AgentResult(trace, success, used_ids=list(cited), distracted=False,
                           guessed_slots=[k for k in truth if k not in answers], n_irrelevant=0)
