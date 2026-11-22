# Does outcome feedback improve ReMem?

The experiment compares plain cosine search, ReMem with feedback disabled, and ReMem with feedback enabled on the same 80 held-out questions over all 10 LoCoMo conversations. LoCoMo supplies multiple annotated questions per conversation, so feedback from one question can change retrieval for another. The conversations were generated for the benchmark. [LoCoMo dataset and paper](https://github.com/snap-research/locomo)

## Protocol

- **Split.** For each conversation, select questions in LoCoMo categories 1 and 4 with a nonempty answer and resolvable evidence IDs. Deduplicate normalized question text within the conversation. A seeded hash of the conversation ID and question text assigns eight held-out questions (four per category) and 12 warm-up questions (three from category 1, nine from category 4). The split holds out questions within known conversations. Questions can ask about related facts, and the warm-up and test sets can share evidence turns.
- **Controlled retrieval.** All three arms index the same text turns with `all-MiniLM-L6-v2` and see the same 30 cosine candidates. Plain search packs candidates by cosine rank. Both ReMem arms use the default heuristic utility, adaptive retention, and cost-aware selection, with `outcome_feedback` as the only ReMem configuration difference. All arms have a five-turn cap and the same 400-token ReMem budget, measured as words plus punctuation. Raw turns are stored as notes without structured keys. Session timestamps support recency scoring. The answer prompt contains turn IDs, speakers, and dialogue text.
- **Warm-up.** All arms search the same 12 warm-up questions in the same order. Feedback-on generates an answer and cites retrieved turn IDs. Normalized exact match on the warm-up reference supplies a binary success signal. `ReMem.feedback()` receives that outcome, the model's answer, and its valid citations. Warm-up references and evidence IDs are never sent to the model or used as citations. Feedback-off still records retrieval history, as the API normally does.
- **Held-out scoring.** Every held-out search starts from the post-warm-up state for its arm. No held-out search or outcome changes that state. A fixed Qwen2.5 3B model generates all answers with the same prompt, temperature zero, a 1,024-token model context, and at most 48 generated tokens. The runner checks the model-reported prompt and completion counts. Identical prompts share their completed response.
- **Metrics.** Accuracy is normalized exact match of word multisets, ignoring case, punctuation, and articles. Token F1 gives partial credit for overlap. Both are lexical measures. Evidence hit is the fraction of questions with any labelled turn in the served context. Evidence recall is the mean fraction of labelled turns served. MRR uses the rank of the first labelled turn and gives zero to a miss. These retrieval metrics use the final context, with up to five turns under the shared budget. Paired 95% bootstrap intervals resample whole conversations (10,000 draws) to preserve within-conversation dependence.
- **Reference-equivalence review.** Codex reviewed every distinct held-out answer against its question and reference with arm labels hidden. The review accepts paraphrases and full-sentence responses that convey the required facts without a material contradiction. An incomplete list or ambiguous answer earns zero. This single-reviewer check is supplementary and leaves the measured feedback signal and exact-match scores intact.

## Results

| System | Exact-match accuracy | Reference-equivalent accuracy | Answer token F1 | Evidence hit | Evidence recall | Evidence MRR | Mean prompt tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Plain search | 7.50% (6/80) | **15.00% (12/80)** | **14.45%** | **37.50%** | **28.99%** | **0.192** | 267.2 |
| ReMem, feedback off | 6.25% (5/80) | 12.50% (10/80) | 12.02% | 31.25% | 24.63% | 0.153 | 264.8 |
| ReMem, feedback on | 7.50% (6/80) | 13.75% (11/80) | 12.14% | 32.50% | 25.88% | 0.155 | 266.6 |

Feedback-on gained **1.25 percentage points** over feedback-off on both answer measures (paired 95% interval **0 to +3.75 pp**). Both gains came from the same question, `conv-43/164`, which appears to have a speaker attribution error. Excluding that question leaves the ReMem arms tied on exact match (5/79) and reference-equivalent accuracy (10/79). Against plain search, feedback-on tied on exact match and trailed by 1.25 pp on the reference review (95% interval **−5.0 to +2.5 pp**).

The retrieval effect was small. Feedback changed the selected turns on 12 of 80 questions and changed the ordered list on 17. It increased the evidence hit rate by 1.25 pp over feedback-off (95% interval **0 to +3.75 pp**). Plain search served a labelled turn on 30 questions, compared with 26 for feedback-on. The shared 30-candidate pool contained any labelled turn on 55 of 80 questions, so the reranker could never surface the annotated evidence on the remaining 25.

Only 11 of 120 warm-up answers passed exact match and awarded positive credit. Just two held-out questions had a labelled turn with prior credit. The run measures transfer under sparse, reference-graded feedback over generated conversations. The 112-answer review is reference-based and inherits errors in the published answers. The sample covers two question categories and ten conversations.

## Representative cases

- **Prior credit improved evidence rank and the answer lost detail.** For the dance-studio question (`conv-30/25`), `D13:7` earned 1.0 credit on warm-up question 66 and moved from fifth to first. Feedback-on also brought in `D18:10`, which describes plans for a web platform. The answer mentioned those plans and dropped the studio’s classes and workshops. Both ReMem arms missed the full reference, and plain search missed both labelled turns.
- **Retrieval improved against a suspect label.** In `conv-43/164`, the feedback ledger recorded a successful warm-up exposure where an unrelated turn, `D27:17`, earned zero credit. The feedback-on context replaced it with the labelled turn and produced the expected answer, Spanish. `D27:6` is spoken by John, and Tim says he studied French in `D27:7`. This item accounts for the entire measured answer-accuracy gain over feedback-off.
- **The candidate pool missed the evidence.** In `conv-26/127`, the turn identifying a stained-glass window (`D14:17`) was outside the top 30. Both ReMem arms answered “Unknown,” and plain search returned a description that never named the object.
- **The model missed information present in the context.** In `conv-26/109`, all three prompts contained the same five turns. Plain search placed the turn about making pots (`D8:2`) third and answered correctly in a full sentence. Both ReMem arms placed it fourth and answered “Unknown.”

See the [summary and paired intervals](../results/feedback_ablation/summary.json), [per-question retrieval and answers](../results/feedback_ablation/per_question.jsonl), [warm-up credit trace](../results/feedback_ablation/warmup.json), and [reference review](../results/feedback_ablation/reference_review.json). Conversation IDs and zero-based QA indices identify items in the [LoCoMo release](https://github.com/snap-research/locomo).

## Reproduce

Install the embedding dependencies, download `locomo10.json` to `data/`, and make the local answer model available to Ollama. [Dataset file](https://github.com/snap-research/locomo/blob/main/data/locomo10.json) · [LoCoMo license](https://github.com/snap-research/locomo/blob/main/LICENSE.txt)

```bash
python -m pip install -e ".[embeddings]"
ollama pull qwen2.5:3b
python experiments/run_feedback_ablation.py --data data/locomo10.json
```

Use `--url` for an Ollama instance on a different local port. `--embedder` accepts a Sentence Transformers snapshot path. The reference run used MiniLM revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The runner records the dataset SHA-256, Ollama version, exact model digest, generation options, and effective system configurations. Cached model calls are reusable under the same inputs and options.
