"""LLM agent for the development simulator: a real OpenAI-compatible model plays the simulator's reader.

Same interface as remem_sim.eval.agent.SimAgent (`act(task, items) -> AgentResult`). Simulator only: not evidence.
"""

from __future__ import annotations

import json
import re
from typing import Any

from remem.types import AgentTrace
from remem_sim.eval.agent import AgentResult
from remem_sim.eval.world import Task, World

SYSTEM_PROMPT = (
    "You answer questions about a software project using ONLY the memory snippets provided. "
    "Each memory has an id and a date. If memories conflict, prefer the one that is valid for the question. "
    "If the memories do not contain the answer, give your best guess. "
    'Reply with JSON only: {"answers": {"<subject>.<attribute>": "<value>"}, "cited": ["<memory id>", ...]}. '
    "List in `cited` exactly the memory ids you relied on."
)


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
                 json_mode: bool | str = False) -> None:
        self.world, self.client, self.model, self.t0 = world, client, model, t0
        self.temperature, self.max_tokens = temperature, max_tokens
        self.n_calls = 0
        # False: free text | True / "object": response_format=json_object | "schema": JSON schema with the exact
        # answer slots (constrained decoding; supported by OpenAI and Ollama), so a reply can't use a wrong key
        self.json_mode = json_mode
        self.parse_failures = 0

    def act(self, task: Task, items: list[tuple[str, str, float]]) -> AgentResult:
        if self.json_mode == "schema":
            from remem_sim.claude_agent import answer_schema
            extra = {"response_format": {"type": "json_schema", "json_schema": {
                "name": "answer", "strict": True, "schema": answer_schema(task)}}}
        elif self.json_mode:
            extra = {"response_format": {"type": "json_object"}}
        else:
            extra = {}
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
