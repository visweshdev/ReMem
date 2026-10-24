"""Core data types for the outcome-aware memory layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemState(str, Enum):
    ACTIVE = "ACTIVE"
    LOW_VALUE = "LOW_VALUE"
    COMPRESSED = "COMPRESSED"
    ARCHIVED = "ARCHIVED"
    INVALIDATED = "INVALIDATED"


@dataclass
class Candidate:
    """One memory offered by a backend for one query (before ReMem ranking)."""

    fact_id: str
    text: str
    tokens: int
    time: float                       # fact timestamp (unix seconds)
    sim: float                        # cosine(query, memory) under the shared embedder
    base_rank: int                    # rank in the backend's own ordering (0 = best)
    confidence: float = 1.0
    entities: tuple[str, ...] = ()
    attribute_keys: tuple[str, ...] = ()
    source: str = ""                  # backend name
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredCandidate:
    cand: Candidate
    utility: float                    # ordering score (model probability, or heuristic score)
    prob: float                       # estimated P(contributes); used by cost-aware selection
    components: dict[str, float]      # per-feature contribution, for explanations
    state: str                        # MemState value at retrieval time
    text: str                         # text actually served (compressed if COMPRESSED)
    tokens: int                       # tokens of the served text
    features: list[float] = field(default_factory=list)
    reason: str = ""


@dataclass
class Context:
    """What the memory layer hands to the agent for one task."""

    task_id: int
    query: str
    task_class: str
    as_of: float | None
    selected: list[ScoredCandidate]
    scored: list[ScoredCandidate]     # all considered candidates (for diagnostics)
    budget_tokens: int | None
    tokens_used: int
    latency_ms: float
    pool_size: int


@dataclass
class AgentTrace:
    """What the agent reports back after acting. Contains no ground truth."""

    answer: dict[str, str]            # slot -> value the agent committed to
    cited_ids: list[str]              # memory ids the agent says it relied on (noisy)
    tool_args: dict[str, str]         # arguments of the tool call it made
    n_llm_calls: int = 1


@dataclass
class Outcome:
    """Environment feedback. `success` is the only thing the memory layer may learn from."""

    success: bool
    latency_ms: float = 0.0


@dataclass
class LifecycleEvent:
    step: int
    kind: str                         # CREATED RETRIEVED USED CREDITED NO_CONTRIBUTION BLAMED
                                      # STATE_CHANGE SUPERSEDED REVIVED CONTRADICTED
    detail: str = ""
    utility: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "kind": self.kind, "detail": self.detail, "utility": self.utility}


@dataclass
class ClassStats:
    n_ctx: int = 0
    credit: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {"n_ctx": self.n_ctx, "credit": self.credit}


@dataclass
class MemoryRecord:
    """Ledger entry for one stored fact. Mutable sidecar keyed by fact_id; the backend's own record is never modified."""

    fact_id: str
    text: str                         # currently served text (may be compressed)
    original_text: str
    tokens: int
    original_tokens: int
    created_step: int
    time: float
    confidence: float = 1.0
    entities: tuple[str, ...] = ()
    attribute_keys: tuple[str, ...] = ()
    semantic_role: str = "state"
    state: MemState = MemState.ACTIVE
    valid_from: float = 0.0
    valid_to: float | None = None
    superseded_by: str | None = None
    supersedes: list[str] = field(default_factory=list)
    contradictions: int = 0           # times flagged as contradicted
    # retrieval / outcome statistics
    n_pooled: int = 0                 # times it was a candidate
    n_retrieved: int = 0              # times it was selected into context
    n_credited: float = 0.0           # sum of per-task credit in [0,1] over successful tasks
    n_wasted: int = 0                 # in context but received credit < threshold
    n_blamed: float = 0.0             # sum of blame over failed tasks
    n_success_ctx: int = 0
    n_fail_ctx: int = 0
    last_retrieved_step: int | None = None
    last_credit_step: int | None = None
    last_updated_step: int = 0
    classes: dict[str, ClassStats] = field(default_factory=dict)
    events: list[LifecycleEvent] = field(default_factory=list)
    last_reason: str = ""
    last_utility: float | None = None
