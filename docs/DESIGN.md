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
| `remem/config.py` | dataclass configs + YAML (`configs/systems.yaml`) |
| `remem/ledger.py` | sidecar store, statistics, lifecycle events, JSON persistence |
| `remem/features.py` | feature groups: static, history, outcome, contradiction |
| `remem/utility.py` | transparent heuristic; L2 logistic regression (numpy Newton) |
| `remem/attribution.py` | presence / signals / signals + leave-one-out (all or ambiguous-only) |
| `remem/contradiction.py` | supersession, validity intervals, volatile-attribute decay, outcome-triggered invalidation, schema learning |
| `remem/retention.py` | ACTIVE → LOW_VALUE → COMPRESSED → ARCHIVED, INVALIDATED, revival; frequency-prune control |
| `remem/selection.py` | rank-pack and exact 0/1 knapsack |
| `remem/system.py` | `ReMemSystem.ingest / retrieve / observe`; guard, exploration |
| `remem/backends/` | `Backend` protocol and the flat-vector backend |
| `remem/eval/` | seeded world generator, simulated agent, runner, metrics, failure analysis, report |
| `remem/llm.py` | OpenAI-compatible agent/embedder adapters (unverified against a live endpoint) |
| `remem/viz/ledger_cli.py` | developer inspector |

## 3. Utility target and labels

Target: `P(memory contributes to a successful outcome | features at retrieval time)`. Labels are *observed* (attributed credit ≥ 0.5 on
a successful task); hidden ground truth is used only to score the system. The learned model is trained on traces from training worlds
disjoint from the test worlds.

## 4. Credit assignment and its limits

Signals: cited by the agent (self-reported), value-overlap with the committed answer, outcome, optional counterfactual replay. Failure
produces blame for memories whose value was used. Limits: citations are noisy; value-use is lexical (a decoy mentioning the answer
counts); replay needs a deterministic, replayable agent and under-credits redundant copies; none of it proves causality.

## 5. Retention rules

Transitions depend on the Beta-posterior contribution rate and on *exposures* (times shown inside a successful task), never on retrieval
count alone. Demotions are class-conditional. Nothing is deleted. Credited use revives. Accumulated credit protects from ARCHIVED.

## 6. Evaluation design (guarding against a rigged benchmark)

* All systems consume the same candidate pool per query (cached), so only ranking / selection / lifecycle differ.
* Seven task families, a no-recurrence control family where per-memory history cannot help, two reader modes, feedback-noise,
  distraction-strength and extraction-noise sensitivity, a frequency-prune retention control.
* Seeds: train 100–103, dev 200–201 (sanity checks, `guard_top`), test 0–4 (all reported numbers).
* Failure analysis is generated automatically.

## 7. Deviations found while building

1. The contradiction detector must only consider state-asserting roles; otherwise a note ("Earlier question …") could supersede a fact.
2. A token-level diff gave false conflicts on paraphrases and missed true ones: it now compares value-bearing tokens only, and for declared
   single-valued attributes the key match suffices (the similarity guard applies only to the cue path).
3. Leave-one-out's discount for "not necessary" memories equalled the contribution threshold; it is now 0.4 (< 0.5).
4. Retention had to be class-conditional so a memory useless for one task class is not hidden from another.
5. A learned model on retrieval-history features is worse than no model (it learns "frequent ⇒ useful"); kept in the ablation as a finding.
6. The learned model did not beat the heuristic, and usage-only feedback matched outcome-gated feedback: both reported as neutral/negative.
7. The similarity guard reduces its target failure but not task success; Thompson sampling is worse than the UCB bonus.
8. Pool cache keys must include the family parameters, or a changed family silently reuses stale pools.
9. LoCoMo/LongMemEval and related-work verification were not possible (no endpoints/data, no web access).
