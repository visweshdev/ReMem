# Related work: what is verified and how

**How this was checked.** On 2026-10-23, in a browser session, I read the arXiv abstract of each paper and, for Mem0 and Zep, searched the
full HTML text for specific terms (update/delete, invalidation, hierarchy, feedback, utility, token). **A search term that is not found
does not prove a feature is absent**; I did not read the papers end to end, and I did not read any code or product documentation.
Treat "not found" as "not seen in this check".

| System | Hierarchical organisation | Temporal handling | Learns utility from use / outcome feedback | Token-budget-aware retrieval |
|---|---|---|---|---|
| **Mem0** ([arXiv 2504.19413](https://arxiv.org/abs/2504.19413)) | Not described as hierarchical (extracted-fact store; a graph variant adds entities/relations) | Entities carry creation timestamps. An LLM chooses ADD / UPDATE / DELETE / NOOP per candidate fact; DELETE removes memories contradicted by new information. In the graph variant an LLM resolver marks conflicting relationships *invalid* instead of deleting them, to enable temporal reasoning | Not found | Reports large token-cost and latency savings versus full-context prompting; no retrieval-time budget-selection mechanism seen |
| **Zep / Graphiti** ([arXiv 2501.13956](https://arxiv.org/abs/2501.13956)) | Yes: three tiers of subgraphs (episode, semantic entity, community) | Yes: bi-temporal model; temporal extraction and edge invalidation of superseded facts | Not found | Reports ~90% latency reduction versus baselines; retrieves a fixed number (20) of edges/nodes in its evaluation |
| **RMM**, Reflective Memory Management ([arXiv 2503.08026](https://arxiv.org/abs/2503.08026)) | Summarises across utterance, turn and session granularities | Not assessed | **Yes, per its abstract: "Retrospective Reflection … iteratively refines the retrieval in an online reinforcement learning manner based on LLMs' cited evidence."** | Not assessed |
| **ReMem (this repository)** | Not provided (backend-agnostic) | Validity intervals, supersession, `as_of` queries | Yes: logistic-regression and heuristic utility; credit from citations / value use / optional counterfactual replay | Yes: expected-utility knapsack under a token budget |

## Benchmarks (not memory systems)

LongMemEval and LoCoMo are long-conversation memory benchmarks (Zep's paper evaluates on a LongMemEval variant with ~115k-token
conversations; Mem0's evaluates on LoCoMo). **Neither was run here, so no LongMemEval/LoCoMo number in this project is ours.**
ReMem uses its own seeded, controlled tasks (`remem/eval/world.py`) because it needs a downstream task-success signal.

## What this means for ReMem's claims

* **Citation-based feedback on retrieval is not new.** RMM's abstract describes online-RL refinement of retrieval from the LLM's cited
  evidence, which is the same family of idea as ReMem's citation signal. ReMem should not be presented as introducing it. I have not read
  RMM's method in detail, so I cannot say how its credit assignment, scope or evaluation differ from ReMem's.
* **Supersession without deletion is not new either.** Zep (edge invalidation) and Mem0's graph variant (marking relationships invalid)
  both keep superseded facts for temporal reasoning. ReMem's version is rule-based with learned attribute cardinality and outcome-triggered
  invalidation; those specifics were not found in the papers, but that is a weak statement given the depth of this check.
* What ReMem offers is the **combination** (outcome-derived utility, an explicit retention state machine, supersession, token-budgeted
  selection) in one backend-agnostic layer, plus a controlled evaluation with ablations, a control family and failure analysis. Whether it
  beats these systems with real LLM agents is **untested**.
