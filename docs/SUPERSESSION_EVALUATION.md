# Supersession regression and transfer tests

Broad keys from an extractor can group together a measured result, a plan, and a related detail. In the LongMemEval development set, a later statement about hoping to beat a 5K record displaced the actual **25:50** result under newest-per-key replacement. Another case grouped the location of a planned birthday trip with a different island. The original 72-question comparison measured 63.9% for plain search and 45.8% for ReMem 0.2. [Original run and paired intervals](../results/real_knowledge_update/v0.2_all72/summary.json)

## Replacement policy

The built-in checker requires strictly ordered observations. When an original source turn is available, `LLMUpdateChecker` looks for a transition cue and asks the model to verify an explicit current change in the same property and scope. Ambiguous verdicts, plans, and additions preserve the previous fact. Fact-only checks remain available when source text is absent. A caller that knows an update has occurred can declare it with `update="replace"`.

ReMem saves confirmed replacement IDs and reuses them on reopen. Historical queries can retrieve a superseded fact through its validity interval. This policy trades recall of implicit updates for protection against false replacement.

## LongMemEval protocol

The runner preserves the historical SHA-1 question-ID split of 36 development and 36 comparison questions. The replacement policy was chosen using the development cases. The historical ReMem 0.2 result used all 72 questions, so the second half is a retrospective comparison for this change. FactConsolidation provides a separate transfer dataset.

The raw cosine, extracted-fact cosine, newest-per-key, text-only check, and source-checked arms share a 15-turn candidate pool and a five-memory cap. The latter two have the same source presentation in the answer prompt. The runner also retains a text-only presentation for the legacy checker. It records evidence coverage, retirements of labelled evidence, prompt size, paired accuracy intervals, prompts, model tags, token consumption for live calls, and cache use. Source labels identify the newest annotated evidence by timestamp. Historical questions may require older evidence.

```bash
python experiments/run_real_knowledge_update.py --data data/longmemeval_s_cleaned.json --split dev --out results/supersession_regression/dev
python experiments/run_real_knowledge_update.py --data data/longmemeval_s_cleaned.json --split test --out results/supersession_regression/test
```

The experiment separates generation and judging when needed. Use `--generate-only` to save hypotheses, and `--judge-from path/to/generated.jsonl` with the same split and system list to judge them later. `--extract-model` and `--model` allow distinct extractors and answer models. `--seed` records an explicit generator seed for a fresh run. The historical cache uses temperature zero with the runtime's default seed. [LongMemEval dataset and format](https://github.com/xiaowu0162/LongMemEval)

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
python experiments/run_factconsolidation.py --data data/mab_conflicts.parquet --out results/held_out_factconsolidation/summary.json
```

Use the dataset's `Conflict_Resolution-00000-of-00001.parquet` as `data/mab_conflicts.parquet`. The reference run uses the `qwen2.5-3b-4k` Ollama tag. Use `--serve-model qwen2.5:3b` for the stock tag if the local serving context accommodates the prompts.
