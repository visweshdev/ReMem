# Supersession regression and transfer tests

Broad keys from an extractor can group together a measured result, a plan, and a related detail. In the LongMemEval development set, a later statement about hoping to beat a 5K record displaced the actual **25:50** result under newest-per-key replacement. Another case grouped the location of a planned birthday trip with a different island. The original 72-question comparison measured 63.9% for plain search and 45.8% for ReMem 0.2. [Original run and paired intervals](../results/real_knowledge_update/v0.2_all72/summary.json)

## Replacement policy

The built-in checker requires strictly ordered observations. When an original source turn is available, `LLMUpdateChecker` looks for a transition cue and asks the model to verify an explicit current change in the same property and scope. Ambiguous verdicts, plans, and additions preserve the previous fact. Fact-only checks remain available when source text is absent. A caller that knows an update has occurred can declare it with `update="replace"`.

ReMem saves confirmed replacement IDs and reuses them on reopen. Historical queries can retrieve a superseded fact through its validity interval. This policy trades recall of implicit updates for protection against false replacement.

## LongMemEval protocol

The runner preserves the historical SHA-1 question-ID split of 36 development and 36 comparison questions. The replacement policy was chosen using the development cases. The historical ReMem 0.2 result used all 72 questions, so the second half is a retrospective comparison for this change. FactConsolidation provides a separate transfer dataset.

The raw cosine, extracted-fact cosine, newest-per-key, and source-checked arms share a 15-turn candidate pool and a five-memory cap. The development run also includes the text-only checker with and without sources displayed to the answer model. The runner records evidence coverage, retirements of labelled evidence, prompt size, paired accuracy intervals, prompts, model tags, token consumption for live calls, and cache use. Source labels identify the newest annotated evidence by timestamp. Historical questions may require older evidence.

```bash
python -m pip install -e ".[embeddings,llm]"
ollama pull qwen2.5:3b
ollama pull qwen3-coder:30b
mkdir -p data/repro
printf 'FROM qwen2.5:3b\nPARAMETER num_ctx 4096\n' > data/repro/Modelfile.qwen25
printf 'FROM qwen3-coder:30b\nPARAMETER num_ctx 4096\n' > data/repro/Modelfile.qwen3
ollama create qwen2.5-3b-4k -f data/repro/Modelfile.qwen25
ollama create qwen3-coder-30b-4k -f data/repro/Modelfile.qwen3
python experiments/run_real_knowledge_update.py --data data/longmemeval_s_cleaned.json \
  --serve-model qwen2.5-3b-4k --serve-judge qwen3-coder-30b-4k \
  --split dev --out data/repro/longmemeval/dev
python experiments/run_real_knowledge_update.py --data data/longmemeval_s_cleaned.json \
  --serve-model qwen2.5-3b-4k --serve-judge qwen3-coder-30b-4k --split test \
  --systems raw_similarity facts_similarity facts_newest_key facts_remem --out data/repro/longmemeval/test
```

Place the cleaned `longmemeval_s_cleaned.json` in `data/` before running the commands. Use `--generate-only` to save hypotheses, and `--judge-from path/to/generated.jsonl` with the same split and system list to judge them later. `--extract-model` and `--model` allow distinct extractors and answer models. `--seed` records an explicit generator seed for a fresh run. The historical cache uses temperature zero with the runtime's default seed. [Cleaned dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/blob/main/longmemeval_s_cleaned.json) · [Dataset format](https://github.com/xiaowu0162/LongMemEval)

## LongMemEval results

| System | Development (36) | Retrospective comparison (36) | All 72 | Newest labelled evidence in context, all 72 |
| --- | ---: | ---: | ---: | ---: |
| Plain cosine on raw turns | 52.8% (19/36) | 72.2% (26/36) | 62.5% (45/72) | 75.0% |
| Cosine on extracted facts | 52.8% (19/36) | 75.0% (27/36) | 63.9% (46/72) | 81.9% |
| Newest per key | 44.4% (16/36) | 47.2% (17/36) | 45.8% (33/72) | 55.6% |
| Source-checked ReMem | **55.6% (20/36)** | **83.3% (30/36)** | **69.4% (50/72)** | **84.7%** |

Over all 72 questions, source-checked ReMem gained **23.6 percentage points** over newest-per-key (paired 95% interval **+11.1 to +36.1 pp**). The comparisons with plain cosine (**+6.9 pp**, 95% interval **−5.6 to +18.1**) and extracted-fact cosine (**+5.6 pp**, 95% interval **−4.2 to +15.3**) remain uncertain. The 36 retrospective comparison questions show the same pattern. [Development results](../results/supersession_regression/dev/summary.json) · [Comparison results](../results/supersession_regression/test/summary.json) · [Combined results](../results/supersession_regression/all72.json)

The forced rule retired 944 extracted facts, including 74 from the newest labelled evidence turns. The guarded arm retired six, including two from those turns. These counts track provenance and can include several extracted facts from one turn. They measure which evidence became unavailable, and the question may also require earlier evidence.

The strong difference is between end-to-end configurations. The guarded arm uses ReMem ranking and includes source excerpts, with a mean prompt size of 471 proxy tokens versus 390 for raw cosine, 167 for extracted-fact cosine, and 166 for newest-per-key. On development questions, the guarded checker and the previous text-only checker with the same source presentation made identical scored answer decisions. That matched comparison shows no observed answer gain from the checker change alone. The retrospective split compares the four main systems. It has no matched text-only-checker arm.

The run used Qwen2.5 3B for extraction, checking, and answers, with Qwen3-Coder 30B judging both splits. It reused historical responses where the prompt and model matched. Fresh development generations ran on CPU. Most fresh comparison generations ran on the integrated GPU, with some cached CPU responses from an earlier attempt. The original ReMem 0.2 table still records its own raw-search score of 63.9%. [Model versions](../results/supersession_regression/model_versions.json) · [Original result](../results/real_knowledge_update/v0.2_all72/summary.json)

### Representative cases

- **Unrelated replacement.** On `6a1eabeb`, newest-per-key retired a record containing the 25:50 time when an interval-training note used the same key. Source-checked ReMem kept the evidence and answered 25:50.
- **Wrong time frame despite retrieval.** On `10e09553`, ReMem retrieved both dated fishing trips and answered with nine bass from the later trip. The question asked about the earlier trip, when the user caught seven.
- **Extraction and ranking gap.** On `c4ea545c`, ReMem's final context omitted both gym schedules and included unrelated fitness and meditation notes. Plain cosine retrieved enough of the raw statements to answer the frequency comparison.

The scored traces include the contexts, hypotheses, judge verdicts, and old/new facts for each recorded retirement. [Development traces](../results/supersession_regression/dev/per_question.jsonl) · [Comparison traces](../results/supersession_regression/test/per_question.jsonl)

## External transfer check

MemoryAgentBench's 6k single-hop FactConsolidation slice contains 100 questions over one 455-record corpus. Its instructions define the later record as authoritative, even when the sentence contains no transition cue. The diagnostic uses the same MiniLM cosine ranking and five-turn cap for each arm. A fixed parser extracts scope and value from the dataset's sentence templates. We score the value in the top result and the occurrence of a matching value in the top five. An unrelated turn can contain the same value, so these metrics measure retrieval rather than model-generated QA.

| System | Top-1 value match | Top-5 value hit | Retired facts |
| --- | ---: | ---: | ---: |
| Plain search | 29% | 81% | 0 |
| Newest per key | **80%** | 81% | 133 |
| Source-checked replacement | 29% | 81% | 0 |

The conservative checker preserved the implicit conflicts. Newest-per-key was useful when the corpus itself guaranteed that later records replaced earlier ones and the slot parser could identify the shared key. This is one generated corpus. [Run and per-question retrievals](../results/held_out_factconsolidation/summary.json) · [Dataset revision](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/tree/7ea066982b140a19337e17e60d45d4076e042faf/data)

```bash
python -m pip install -e ".[embeddings,llm]" pyarrow
python experiments/run_factconsolidation.py --data data/mab_conflicts.parquet --out data/repro/factconsolidation.json
```

Use the dataset's `Conflict_Resolution-00000-of-00001.parquet` as `data/mab_conflicts.parquet`. The reference run uses the `qwen2.5-3b-4k` Ollama tag. Use `--serve-model qwen2.5:3b` for the stock tag if the local serving context accommodates the prompts.

## Temporal and multi-session retrieval

The breadth check uses the ten non-abstention questions with the smallest SHA-256 question-ID hashes in each category. The table reports whether at least one labelled turn reached the top five or ten. The comparison uses raw dialogue turns and a fresh store for each question, with MiniLM embeddings. It measures evidence retrieval. [Selected questions and results](../results/benchmark_breadth/retrieval.json)

| Category | Cosine @5 | ReMem @5 | Cosine @10 | ReMem @10 |
| --- | ---: | ---: | ---: | ---: |
| Temporal reasoning | 40% | 40% | 60% | 60% |
| Multi-session | 80% | 80% | 100% | 90% |

```bash
python experiments/run_longmemeval.py --data data/longmemeval_s_cleaned.json --question-type temporal-reasoning multi-session --per-type-limit 10 --out data/repro/benchmark_breadth
```

## Alternate model pilot

The model pilot uses the first 12 development questions with references of one to three normalized words. Qwen2.5 3B and Llama 3.2 3B each extract facts from the same five dated excerpts. Each answer model receives both sets of facts under the same prompt and 64-token output cap. The seed is zero and extraction has a 160-token cap per excerpt.

| Extractor | Answer model | Strict exact match | With digits and number words equated |
| --- | --- | ---: | ---: |
| Qwen2.5 3B | Qwen2.5 3B | 4/12 | 6/12 |
| Qwen2.5 3B | Llama 3.2 3B | 4/12 | 5/12 |
| Llama 3.2 3B | Qwen2.5 3B | 3/12 | 3/12 |
| Llama 3.2 3B | Llama 3.2 3B | 2/12 | 2/12 |

Every combination answered `Unknown` for the question about a previous goal. On `8fb83627`, both readers answered `5` from the Qwen extraction and `4` from the Llama extraction. The reference is `Five`, so the number-normalized review credits the answers based on the Qwen extraction. The short-reference development subset supports error analysis. The result records 129 fresh model calls, their input and output token counts, and 39 cache hits. [Twelve-question traces](../results/model_pilot/development_12.json) · [Initial two-question run](../results/model_pilot/development.json) · [Model digests](../results/supersession_regression/model_versions.json)

```bash
ollama pull llama3.2:3b
python experiments/run_model_pilot.py --questions 12 --out data/repro/model_pilot.json
```

Use the `qwen2.5-3b-4k` tag set up above. For a fresh token measurement, pass `--cache data/repro/model_pilot_cache.jsonl` with a new path.
