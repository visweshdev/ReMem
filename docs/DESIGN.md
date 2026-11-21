# ReMem design note

## 1. Idea

A retrieval-only memory ranks by similarity. ReMem adds a **feedback loop**: after the agent acts, it observes whether the
memories it was shown contributed to a successful outcome, keeps a per-memory record of that, and uses the record to rank,
retain, supersede and budget memories.

```
        ┌──────────────────────────── remem ─────────────────────────────┐
query ─►│ Backend (flat vector, or any Backend) → over-fetched pool       │
        │        ▼                                                        │
        │ validity + retention filter  ◄── Ledger (sidecar, by fact_id)   │
        │        ▼                          state, stats, validity, log   │
        │ FeatureBuilder → utility model (heuristic | logistic regression)│
        │        ▼                                                        │
        │ Selector (rank-pack | cost-aware knapsack) + similarity guard   │
        └────────┬────────────────────────────────────────────────────────┘
                 ▼ context
               agent ──► trace (answer, cited ids, tool args) + outcome
                 ▼
        Attributor → credit/blame per memory ─► Ledger update ─► RetentionPolicy
        ContradictionDetector (at ingest; and when a failed action used a stale memory) ◄── schema learner
```

The seam is the candidate list: a backend returns candidates; ReMem re-ranks and filters them. It never writes into a backend.

## 2. Modules

| File | Responsibility |
|---|---|
| `remem/types.py` | `MemState`, `Candidate`, `Context`, `AgentTrace`, `Outcome`, `MemoryRecord` |
| `remem/config.py` | dataclass configs (JSON/YAML) |
| `remem/ledger.py` | sidecar store, statistics, lifecycle events, JSON persistence |
| `remem/features.py` | feature groups: static, history, outcome, contradiction |
| `remem/utility.py` | transparent heuristic; L2 logistic regression (numpy Newton) |
| `remem/attribution.py` | presence / signals / signals + leave-one-out (all or ambiguous-only) |
| `remem/contradiction.py` | supersession, validity intervals, volatile-attribute decay, outcome-triggered invalidation, schema learning |
| `remem/retention.py` | ACTIVE → LOW_VALUE → COMPRESSED → ARCHIVED, INVALIDATED, revival; frequency-prune control |
| `remem/selection.py` | rank-pack and exact 0/1 knapsack |
| `remem/system.py` | `ReMemSystem.ingest / retrieve / observe`; guard, exploration |
| `remem/backends/` | `Backend` protocol and the flat-vector backend |
| `remem/llm.py` | OpenAI-compatible embeddings |
| `remem/viz/ledger_cli.py` | developer inspector |

## 3. Utility target and labels

Target: `P(memory contributes to a successful outcome | features at retrieval time)`. Labels are *observed* (attributed credit ≥ 0.5 on
a successful task). The default ranker is a transparent heuristic; the logistic-regression model must be fitted on your own traces.

## 4. Credit assignment and its limits

Signals: cited by the agent (self-reported), value-overlap with the committed answer, outcome, optional counterfactual replay. Failure
produces blame for memories whose value was used. Limits: citations are noisy; value-use is lexical (a decoy mentioning the answer
counts); replay needs a deterministic, replayable agent and under-credits redundant copies; none of it proves causality.

## 5. Retention rules

Transitions depend on the Beta-posterior contribution rate and on *exposures* (times shown inside a successful task), never on retrieval
count alone. Demotions are class-conditional. Nothing is deleted. Credited use revives. Accumulated credit protects from ARCHIVED.

## 6. Evaluation

Only public data and real models are used for reported results: LongMemEval knowledge-update questions answered by a real LLM, and
LongMemEval retrieval (README §9–§10). Learning from feedback has no real-data test yet.

## 7. Deviations found while building

1. The contradiction detector must only consider state-asserting roles; otherwise a note ("Earlier question …") could supersede a fact.
2. A token-level diff gave false conflicts on paraphrases and missed true ones: it now compares value-bearing tokens only, and for declared
   single-valued attributes the key match suffices (the similarity guard applies only to the cue path).
3. Leave-one-out's discount for "not necessary" memories equalled the contribution threshold; it is now 0.4 (< 0.5).
4. Retention had to be class-conditional so a memory useless for one task class is not hidden from another.
5. On real LongMemEval data, supersession on LLM-extracted keys retired correct facts and lowered answer accuracy (README §10).
