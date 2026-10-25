"""Claude (Anthropic API) as the agent in ReMem's evaluation, via the official `anthropic` SDK.

    pip install anthropic
    export ANTHROPIC_API_KEY=...
    python experiments/run_claude_agent.py --model claude-opus-5-5

Same interface as SimAgent / LLMAgent: `act(task, items) -> AgentResult`. Claude sees only the served memories (id, day,
text) and the question, and must return JSON (enforced with structured outputs, `output_config.format`) containing one answer
per required slot and the ids of the memories it relied on. Success is scored against the world's hidden truth by the
harness; the memory layer only receives the trace and the boolean outcome.

Refusals (`stop_reason == "refusal"`) are scored as failures. For models that support it, the server-side refusal fallback
is enabled (`fallbacks: "default"`), so a declined request is retried on a fallback model inside the same call.
"""

from __future__ import annotations

import json
from typing import Any

from remem.eval.agent import AgentResult
from remem.eval.world import Task, World
from remem.types import AgentTrace

SYSTEM = (
    "You answer questions about a software project using ONLY the memory snippets provided. Each memory has an id and the "
    "day it was recorded. If memories conflict, prefer the one valid for the question (the most recent, unless the question "
    "asks about a specific day). If no memory contains the answer, give your best guess. In `cited`, list exactly the ids "
    "of the memories you relied on, and none other."
)

#: models that accept the server-side refusal fallback ("default" form)
FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}

#: USD per million tokens (input, output), for the cost estimate printed by the runner
PRICES = {
    "claude-fable-5-1": (10.0, 50.0), "claude-opus-5-5": (4.0, 20.0), "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0), "claude-sonnet-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50),
}


def build_prompt(task: Task, items: list[tuple[str, str, float]], t0: float) -> str:
    lines = [f"[{fid}] (day {int((ts - t0) // 86400)}) {text}" for fid, text, ts in items]
    asof = f"\nAnswer as of day {int((task.as_of - t0) // 86400)}." if task.as_of is not None else ""
    return ("Memories:\n" + ("\n".join(lines) if lines else "(none)") +
            f"\n\nQuestion: {task.query}{asof}")


def answer_schema(task: Task) -> dict[str, Any]:
    slots = [f"{s}.{a}" for s, a in task.slots]
    return {
        "type": "object",
        "properties": {
            "answers": {"type": "object", "properties": {k: {"type": "string"} for k in slots},
                        "required": slots, "additionalProperties": False},
            "cited": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["answers", "cited"],
        "additionalProperties": False,
    }


class ClaudeAgent:
    def __init__(self, world: World, client: Any, model: str, *, t0: float, effort: str = "low",
                 max_tokens: int = 16000) -> None:
        self.world, self.client, self.model, self.t0 = world, client, model, t0
        self.effort, self.max_tokens = effort, max_tokens
        self.calls = self.refusals = self.parse_failures = 0
        self.input_tokens = self.output_tokens = 0

    def _create(self, task: Task, items):
        kwargs = dict(
            model=self.model, max_tokens=self.max_tokens, system=SYSTEM,
            messages=[{"role": "user", "content": build_prompt(task, items, self.t0)}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": answer_schema(task)}},
        )
        if self.model in FALLBACK_MODELS:
            return self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        return self.client.messages.create(**kwargs)

    def act(self, task: Task, items: list[tuple[str, str, float]]) -> AgentResult:
        resp = self._create(task, items)
        self.calls += 1
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        answers, cited = {}, []
        if resp.stop_reason == "refusal":
            self.refusals += 1
        else:
            text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
            try:
                data = json.loads(text)
                answers = {str(k): str(v).strip() for k, v in data.get("answers", {}).items()}
                cited = [str(c) for c in data.get("cited", [])]
            except (json.JSONDecodeError, AttributeError):
                self.parse_failures += 1
        truth = self.world.answer(task)
        success = all(answers.get(k, "").lower() == v.lower() for k, v in truth.items())
        served = {fid for fid, _, _ in items}
        cited = [c for c in cited if c in served]                 # ignore ids the model invented
        trace = AgentTrace(answer=answers, cited_ids=cited, tool_args=dict(answers))
        return AgentResult(trace, success, used_ids=list(cited), distracted=False,
                           guessed_slots=[k for k in truth if k not in answers], n_irrelevant=0)

    def cost_usd(self) -> float | None:
        if self.model not in PRICES:
            return None
        pin, pout = PRICES[self.model]
        return self.input_tokens / 1e6 * pin + self.output_tokens / 1e6 * pout
