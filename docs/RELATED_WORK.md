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

## RMM in more detail (method sections read 2026-10-25)

From §6.1–6.3 and §7.1 of [arXiv 2503.08026](https://arxiv.org/html/2503.08026):

* **What learns:** a reranker on top of a frozen retriever. Query and memory embeddings pass through residual linear layers
  (q' = q + W_q q, m' = m + W_m m); relevance is q'·m'; the top-M of the retriever's top-K are drawn with Gumbel-perturbed sampling.
* **Reward:** per retrieved memory, **+1 if the generator cited it in its response, −1 otherwise.** The reward does not use whether
  the answer or task was correct.
* **Update:** REINFORCE, Δφ = η (R − b) ∇ log P(selected | q, candidates); η = 1e-3, b a hyper-parameter.
* **Setup:** Gemini-1.5-Flash/Pro generator, Contriever / Stella / GTE retrievers, top-K 20 → top-M 5, evaluated on MSC and LongMemEval.
* **Not found in the method:** deletion/invalidation of outdated memories, conflict handling, or a token budget. Evolving facts are
  handled by its other component, Prospective Reflection (multi-granularity summarisation of utterances, turns and sessions).

### How ReMem differs, and what our comparison shows

| | RMM (paper) | ReMem |
|---|---|---|
| What is learned | one global reranker (embedding adaptation) shared by all memories | per-memory statistics (Beta posteriors, per task class) fed to a small utility model |
| Feedback signal | citation only (+1 / −1) | citation + value use, gated by task success; blame on failure; optional counterfactual replay |
| Generalises to never-seen memories | yes, in principle (it adapts the embedding space) | no: a new memory starts at the prior (the `distractor_norecur` control shows no gain) |
| Outdated facts | via summarisation (Prospective Reflection) | explicit supersession, validity intervals, `as_of` queries |
| Retention / token budget | not described | explicit state machine; knapsack under a budget |

**RMM-style baseline in this repository.** `remem/baselines/rmm.py` reimplements §6 (residual linear adaptation, Gumbel sampling,
citation reward, REINFORCE). It is *our* reimplementation, not the authors' code, and approximates log P(selected) per item. Its gradient
is checked against numerical differentiation (`tests/test_rmm.py`). Learning rate and temperature were chosen on dev seeds
(`experiments/tune_rmm.py`). Prospective Reflection is not reimplemented. Results are in the README; on this benchmark more RMM-style
learning made results *worse* on dev seeds (the best setting was the smallest learning rate), which may reflect our weak hash embeddings,
the small amount of data per world, or our approximation, not RMM as published.

## What this means for ReMem's claims

* **Citation-based feedback on retrieval is not new.** RMM's abstract describes online-RL refinement of retrieval from the LLM's cited
  evidence, which is the same family of idea as ReMem's citation signal. ReMem should not be presented as introducing it. See the section above for how they differ.
* **Supersession without deletion is not new either.** Zep (edge invalidation) and Mem0's graph variant (marking relationships invalid)
  both keep superseded facts for temporal reasoning. ReMem's version is rule-based with learned attribute cardinality and outcome-triggered
  invalidation; those specifics were not found in the papers, but that is a weak statement given the depth of this check.
* What ReMem offers is the **combination** (outcome-derived utility, an explicit retention state machine, supersession, token-budgeted
  selection) in one backend-agnostic layer, plus a controlled evaluation with ablations, a control family and failure analysis. Whether it
  beats these systems with real LLM agents is **untested**.
