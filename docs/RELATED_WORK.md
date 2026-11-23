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

LongMemEval and LoCoMo are long-conversation memory benchmarks. Zep's paper uses a LongMemEval variant with roughly 115k-token
conversations, and Mem0's paper uses LoCoMo. ReMem reports knowledge-update QA and retrieval on LongMemEval. Its LoCoMo experiment
compares plain search and both feedback settings on 80 held-out questions, under a shared model and token budget. The results apply to
that split and protocol. [LongMemEval results](../README.md#10-results) · [LoCoMo protocol and results](FEEDBACK_ABLATION.md)

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
| Generalises to never-seen memories | yes, in principle (it adapts the embedding space) | no: a new memory starts at the prior |
| Outdated facts | via summarisation (Prospective Reflection) | explicit supersession, validity intervals, `as_of` queries |
| Retention / token budget | not described | explicit state machine; knapsack under a budget |

## What this means for ReMem's claims

* **Citation-based feedback on retrieval is not new.** RMM's abstract describes online-RL refinement of retrieval from the LLM's cited
  evidence, which is the same family of idea as ReMem's citation signal. ReMem should not be presented as introducing it. See the section above for how they differ.
* **Supersession without deletion is not new either.** Zep (edge invalidation) and Mem0's graph variant (marking relationships invalid)
  both keep superseded facts for temporal reasoning. In the current ReMem API, the caller can declare a replacement or configure an update
  checker. The lower-level detector includes schema learning and outcome-triggered invalidation. [Replacement policy](SUPERSESSION_EVALUATION.md#replacement-policy)
* ReMem combines outcome-derived utility, an explicit retention state machine, supersession, and token-budgeted selection in one
  backend-agnostic layer. Plain search led the original LongMemEval run and LoCoMo evidence retrieval. In the LongMemEval rerun,
  source-checked ReMem scored 69.4% and plain cosine scored 62.5%, with a paired confidence interval spanning zero.
  [LongMemEval rerun](SUPERSESSION_EVALUATION.md#longmemeval-results) · [LoCoMo ablation](FEEDBACK_ABLATION.md)
