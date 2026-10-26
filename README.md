<p align="center">
  <img src="docs/img/banner.svg" alt="ReMem: memory for AI agents that learns what actually helps" width="100%">
</p>

<p align="center">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-2f7d5b">
  <img alt="85 tests" src="https://img.shields.io/badge/tests-85%20passing-2f7d5b">
  <img alt="MCP server" src="https://img.shields.io/badge/MCP-server-6b4fbb">
  <img alt="LangChain retriever" src="https://img.shields.io/badge/LangChain-retriever-1c3c3c">
  <img alt="Runs locally" src="https://img.shields.io/badge/runs-100%25%20local-555">
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#results-at-a-glance">Results</a> ·
  <a href="#use-it-with-claude-mcp">MCP</a> ·
  <a href="#11-limitations-read-these">Limitations</a> ·
  <a href="docs/DESIGN.md">Design</a>
</p>

Most agent memories retrieve whatever *sounds* most similar to the question. That breaks in two everyday ways: a look-alike
note that never helps keeps winning, and a fact that was replaced last week is served right next to its replacement.

**ReMem is a memory layer that learns from outcomes.** After each task you tell it whether the task worked and which
memories helped. It uses that to rank what helps higher, retire what was replaced, demote what never helps, and spend fewer
tokens. Nothing is ever deleted, and every decision can be inspected.

<p align="center">
  <img src="docs/img/demo.gif" alt="Terminal demo: an outdated fact is retired and a useless look-alike note is demoted after feedback" width="100%">
</p>
<p align="center"><sub>Real output of <a href="examples/basic_api.py"><code>examples/basic_api.py</code></a>: the outdated fact is retired, and the note that never helps is demoted to <code>LOW_VALUE</code>.</sub></p>

> [!IMPORTANT]
> **Research prototype, reported honestly.** In a controlled synthetic benchmark ReMem beats similarity search by **+10 to +14
> points** (also with real embeddings). With a real (small, local) LLM as the agent the gain is **within noise**, and on the public
> LongMemEval benchmark, where no question repeats, it behaves like plain search. Details in [Results](#results-at-a-glance) and
> [Limitations](#11-limitations-read-these).

## Quickstart

```bash
git clone https://github.com/<your-username>/ReMem.git
cd ReMem
pip install -e .                 # core needs only numpy, pyyaml, matplotlib
python examples/basic_api.py     # the demo above
```

```python
from remem import ReMem

mem = ReMem("my_memory")                                          # a folder; saved and reloaded automatically
mem.add("Our server is in east-1.", entity="server", attribute="region")
mem.add("Our server moved to west-2.", entity="server", attribute="region")   # retires the east-1 fact

hits = mem.search("Where is our server?")
print(hits.items[0].text)                                         # Our server moved to west-2.

# ... your agent answers using hits.items ...
mem.feedback(hits.search_id, success=True, used_ids=[hits.items[0].id])      # this is what it learns from
```

Three calls: **add** facts, **search** before answering, **feedback** after. `mem.inspect(id)` shows any memory's state, credit
and history, and `mem.search(..., as_of=timestamp)` asks about the past, when retired facts come back.

## How it works

<p align="center">
  <img src="docs/img/how_it_works.svg" alt="The loop: add facts, search, your agent acts, report the outcome, the memory ledger learns, and the next search is better informed" width="100%">
</p>

Every memory moves through an explicit lifecycle. Each transition is logged with its reason, and nothing is deleted:

```mermaid
stateDiagram-v2
    direction LR
    [*] --> ACTIVE: add
    ACTIVE --> LOW_VALUE: shown often, never helps
    LOW_VALUE --> ACTIVE: helps again
    LOW_VALUE --> COMPRESSED
    COMPRESSED --> ARCHIVED
    ARCHIVED --> ACTIVE: revived by credited use
    ACTIVE --> INVALIDATED: newer fact replaces it
    INVALIDATED --> [*]: still returned for as_of queries
```

| | Plain similarity memory | ReMem |
|---|---|---|
| **Ranking** | similarity only | similarity × usefulness learned from outcomes (transparent heuristic or logistic regression) |
| **Replaced facts** | served next to the new one | retired with a validity interval; still queryable "as of" a past date |
| **Useless look-alikes** | keep winning forever | demoted per kind of task once evidence says they never help |
| **Rarely needed but critical** | pruned if retention is frequency-based | protected: retention reacts to *uselessness when shown*, never to rarity |
| **Context size** | fixed top-k | expected-utility knapsack under a token budget (−9.5 tokens per task) |
| **Explainability** | a score | per-memory ledger: shown / credited / blamed counts, state history, reasons |

## Results at a glance

<p align="center">
  <img src="docs/img/results.png" alt="Task success: similarity only 62.6%, RMM-style 63.1%, static score 65.9%, ReMem 76.4%" width="80%">
</p>

Same memories, same candidate pools, same simulated agent; 7 task families × 15 test seeds that were never used for tuning. The
gain is concentrated where facts change, and it vanishes on the control family where nothing recurs, which is the expected outcome:

<p align="center">
  <img src="docs/img/families.png" alt="Per-family task success, similarity only vs ReMem; largest gains on state change and contradiction, none on the control" width="80%">
</p>

And every other setting tested, including the ones where ReMem does **not** help:

<p align="center">
  <img src="docs/img/evidence.png" alt="ReMem minus similarity-only: +13.8 synthetic, +10.2 with real embeddings, +5.6 with a timestamp-aware reader, +0.4 and -2.9 with a real 3B LLM (within noise), +1.3 on LongMemEval" width="90%">
</p>

The full numbers, ablations, failure analysis and the story behind the real-LLM result are in [§10 Results](#10-results).

## Use it with Claude (MCP)

ReMem ships an MCP server, so Claude Code, Claude Desktop or any MCP client can use it as long-term memory:

```bash
pip install -e ".[mcp,embeddings]"
claude mcp add remem --env REMEM_EMBEDDER=sentence-transformers -- remem-mcp     # Claude Code
```

The model gets `remember`, `recall`, `feedback`, `inspect`, `list_memories` and `stats`. If it forgets to call `feedback`, the server
infers outcomes: a correction counts as a failure, and moving on without one counts as a weak success. Claude Desktop setup:
[`examples/claude_desktop_mcp.md`](examples/claude_desktop_mcp.md).

## More ways to use it

<details>
<summary><b>Real embeddings and vector stores</b> (sentence-transformers, FAISS, Chroma)</summary>

```python
from remem import ReMem
from remem.embedders import SentenceTransformerEmbedder      # pip install -e ".[embeddings]"
from remem.backends import FaissBackend, ChromaBackend        # pip install -e ".[faiss]" / ".[chroma]"

emb = SentenceTransformerEmbedder("all-MiniLM-L6-v2")        # free, runs locally
mem = ReMem("~/.remem/app", embedder=emb, backend=FaissBackend(emb))
```

The built-in hash embedder (`remem.offline.ConceptHashEmbedder`) exists for tests and the synthetic benchmark; use a real model in practice.
`remem.llm.OpenAIEmbedder` wraps any OpenAI-compatible embeddings endpoint. Any object with `ingest(facts)` and `candidates(query, k)`
works as a backend.
</details>

<details>
<summary><b>Facts from raw conversations</b> (LLM extraction)</summary>

```python
from openai import OpenAI
from remem.extract import LLMFactExtractor
mem.add_conversation(transcript, extractor=LLMFactExtractor(OpenAI(), model="gpt-4o-mini"))
```

Without an extractor, `add_conversation` stores each sentence as a note (no automatic supersession). The LLM extractor is tested against a
fake client only; its extraction quality with a real model has not been measured here.

Facts with the same `entity` + `attribute` and role `"state"` (the default) replace each other. Store multi-valued things (likes,
dependencies) with `role="preference"` or `"note"`; if a new `"state"` fact reads like an *additional* value ("also uses Memcached"),
`add` emits a `UserWarning`.
</details>

<details>
<summary><b>LangChain retriever</b></summary>

```python
from remem.integrations.langchain import ReMemRetriever      # pip install -e ".[langchain]"
retriever = ReMemRetriever(memory=mem, k=5, task="support")
docs = retriever.invoke("Which plan is the customer on?")
retriever.feedback(success=True, used_ids=[docs[0].metadata["id"]])
```

Runnable: [`examples/langchain_retriever.py`](examples/langchain_retriever.py).
</details>

<details>
<summary><b>MCP server details</b> (environment variables, inferred feedback)</summary>

Claude Desktop, `claude_desktop_config.json`:

```json
{"mcpServers": {"remem": {"command": "remem-mcp", "env": {"REMEM_PATH": "C:/Users/you/.remem/work", "REMEM_EMBEDDER": "sentence-transformers"}}}}
```

If `remember` corrects a served `"state"` memory (same entity and attribute, different text), that recall counts as a failure blamed on the
corrected memory. If a later `recall` for the same `task` arrives with no correction in between, the earlier recall counts as a weak success,
with no memory explicitly cited. Inferred outcomes are noisier than explicit ones and have not been benchmarked separately.

Environment: `REMEM_PATH` (store directory, default `~/.remem/default`), `REMEM_EMBEDDER` (`hash` default, or `sentence-transformers`),
`REMEM_ST_MODEL`, `REMEM_AUTO_FEEDBACK` (`0` turns inference off).
</details>

<details>
<summary><b>Dashboard, inspector, real token counts</b></summary>

```bash
python -m remem.viz.dashboard ~/.remem/myproject -o memory.html          # self-contained HTML; open in a browser
python -m remem.viz.ledger_cli ~/.remem/myproject/ledger.json --list --state LOW_VALUE
```

Token budgets use a word+punctuation proxy by default; `remem.offline.set_tokenizer("tiktoken")` (or `REMEM_TOKENIZER=tiktoken`) switches
to real BPE counts.
</details>

<details>
<summary><b>Reproduce the experiments</b> (all free and local)</summary>

```bash
python -m pytest tests -q                                   # 85 tests, under a minute
python experiments/run_benchmark.py --quick                 # smoke run (~20 s)
python experiments/run_benchmark.py                         # full protocol, 15 seeds (~20 min, resumable)
python experiments/run_real_embeddings.py                   # headline comparison with MiniLM embeddings
# real LLM, free: install Ollama (https://ollama.com), then  ollama pull qwen2.5:3b
python experiments/run_llm_agent.py --ollama --model qwen2.5:3b --family contradiction --seeds 0,1,2
OPENAI_API_KEY=... python experiments/run_llm_agent.py --model <chat-model>        # any OpenAI-compatible API instead
python experiments/run_longmemeval.py --data data/longmemeval_s_cleaned.json        # LongMemEval retrieval, no LLM needed
python docs/make_figures.py                                 # regenerate the README images
```

LongMemEval data: `longmemeval_s_cleaned.json` from https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned (put it in `data/`).
</details>

## What's new compared with prior work

Citation-based feedback on retrieval is prior art (RMM, arXiv 2503.08026), and Zep and Mem0's graph variant already keep superseded facts
as invalid instead of deleting them. ReMem's contribution is the **combination** (outcome-gated utility, supersession, outcome-aware
retention and token-budgeted selection in one inspectable ledger) plus a **controlled evaluation that reports its negative results**. An
RMM-style reranker is reimplemented as a baseline. See [docs/RELATED_WORK.md](docs/RELATED_WORK.md).

---

## Technical documentation

1. [Problem](#1-problem) · 2. [Architecture](#2-architecture) · 3. [Utility model](#3-memory-utility-model) · 4. [Outcome feedback and credit assignment](#4-outcome-feedback-and-credit-assignment) · 5. [Adaptive retention](#5-adaptive-retention) · 6. [Contradiction and staleness](#6-contradiction-and-staleness) · 7. [Cost-aware retrieval](#7-cost-aware-retrieval) · 8. [Additional mechanisms](#8-additional-mechanisms) · 9. [Evaluation methodology](#9-evaluation-methodology) · 10. [Results](#10-results) · 11. [Limitations](#11-limitations-read-these) · 12. [Failure analysis](#12-failure-analysis) · 13. [Future work](#13-future-work) · [Related work](docs/RELATED_WORK.md)

## 1. Problem

Long-horizon agents accumulate thousands of memories. Which to show, keep, shrink or retire depends on whether they
*help the agent succeed*, which similarity cannot tell you: a memory can be close to every query and never useful; another can be
rarely needed and indispensable; a third can be correct once and false after a migration.

## 2. Architecture

```mermaid
flowchart LR
  Q[query, task class, as-of] --> B[Backend: flat vector]
  B -->|over-fetched pool| V[validity + retention filter]
  L[(Memory ledger, keyed by fact id)] --> F[feature builder]
  V --> F --> U[utility model]
  U --> S[selector + similarity guard]
  S --> C[context] --> A[agent]
  A --> T[trace: answer, citations, tool args] --> AT[credit assignment]
  A --> O[outcome] --> AT
  AT --> L
  L --> R[retention state machine]
  I[ingest] --> CD[contradiction detector + schema learner] --> L
  O -. failed action used a stale memory .-> CD
```

| Module (`remem/`) | Role |
|---|---|
| `ledger.py`, `types.py` | per-memory state, statistics, validity, lifecycle events; JSON persistence |
| `features.py`, `utility.py` | feature groups; transparent heuristic and L2 logistic regression |
| `attribution.py` | credit assignment: presence / signals / signals + leave-one-out (all or ambiguous-only) |
| `contradiction.py` | supersession, validity intervals, volatile-attribute confidence decay, outcome-triggered invalidation, schema learning |
| `retention.py` | the state machine and revival |
| `selection.py`, `system.py` | rank packing vs knapsack; `ReMemSystem.ingest / retrieve / observe`; guard and exploration |
| `backends/` | `Backend` protocol and the flat-vector backend |
| `offline.py` | deterministic hash embedder with synonym lexicon (offline benchmark/tests) |
| `eval/` | seeded world generator, simulated agent, runner, metrics, failure analysis, report |
| `llm.py` | OpenAI-compatible agent and embedder adapters (unverified, see §8) |
| `viz/ledger_cli.py` | developer inspector |
| `api.py` | `ReMem`: the high-level, persistent API (`add`, `search`, `feedback`, `inspect`, `add_conversation`) |
| `mcp_server.py` | MCP server (`remem-mcp`) exposing remember / recall / feedback / inspect |
| `embedders.py`, `backends/stores.py` | sentence-transformers embeddings; FAISS and Chroma backends |
| `extract.py`, `integrations/langchain.py` | LLM fact extraction; LangChain retriever |
| `baselines/rmm.py` | RMM-style reranker used as a comparison baseline (reimplementation) |
| `viz/dashboard.py` | self-contained HTML dashboard |

## 3. Memory utility model

Target: `P(memory contributes to a successful outcome | features at retrieval time)`. Four feature groups let ablations switch
information sources off: **static** (similarity, gap to best, backend rank, recency, confidence, token cost), **history**
(retrieval count/recency, the "frequency" strawman), **outcome** (Beta-posterior contribution rate overall and per task class,
waste rate, blame rate, posterior std), **contradiction** (stale flag, contradiction count).

* `HeuristicUtility`: documented weighted sum (weights are design choices, listed in `utility.py`).
* `LearnedUtility`: L2 logistic regression (numpy Newton/IRLS, deterministic) trained on traces from **disjoint training worlds**; labels
  are *observed* (attributed credit ≥ 0.5 on a successful task), never ground truth; regularisation by leave-one-seed-out CV
  (`results/models/train_report.json`).
* Exploration: `+ c·posterior_std` (UCB-style) by default; Thompson sampling is available (§8).

## 4. Outcome feedback and credit assignment

`retrieve → agent acts → success/failure → credit assignment → ledger update → retention`. Only **shown** memories receive feedback
(bandit feedback), so retention reacts to evidence of uselessness *when shown*, never to low retrieval frequency.

Credit uses observable signals only: whether the agent **cited** the memory (self-reported, noisy) and whether the memory
**contains a value the agent committed to**. Credit is per memory (not split equally); failures assign *blame* to memories whose
value was used. Optional **leave-one-out replay** upgrades memories whose removal flips success to failure. Stated limitations:
citations are self-reported; value use is lexical (a decoy mentioning the answer counts); LOO needs a replayable agent and
under-credits redundant copies; none of it establishes causality.

## 5. Adaptive retention

States `ACTIVE, LOW_VALUE, COMPRESSED, ARCHIVED, INVALIDATED`; every transition is logged with its reason; nothing is deleted
(`original_text` is kept). Transitions depend on the posterior contribution rate and the number of *exposures*; demotions apply
per task class ("useless for deploys" ≠ "useless for approvals"); credited use revives; memories with accumulated credit cannot be
archived. A frequency-prune **control** exists only to show what this design avoids.

## 6. Contradiction and staleness

At ingest, two state-asserting facts conflict when they share an `(entity, attribute)` key and differ in *value-bearing* tokens.
Single-valued attributes supersede on key match; otherwise explicit update wording ("migrated", "no longer") is required. The old
fact gets `valid_to` and state `INVALIDATED`; it is excluded from default retrieval but returned for historical `as_of`
queries. A failed action that used memory M while a newer same-key fact exists supersedes M (outcome-triggered). Confidence decays
only for attributes observed to change, so old-but-true facts are not punished for age.

## 7. Cost-aware retrieval

Candidates with predicted `p < p_floor` are not worth context; under a hard token budget an exact 0/1 knapsack maximises Σp subject
to count and token limits. Tokens are a word+punctuation proxy, not a real tokenizer. Additivity of expected utility is an
approximation (a task needing two memories only succeeds with both).

## 8. Additional mechanisms

Each was tested and measured; results (including those that did not help) are in §10.

1. **Similarity guard** (`selection.guard_top`, default 3): the top-m most similar candidates are always served unless the ledger has
   evidence they are useless for this task class (shown ≥ 3 times, class contribution rate < 0.2). Targets a measured failure: the
   utility layer pushing a correct, highly similar memory out of context. `guard_top=3` was chosen on dev seeds.
2. **Thompson sampling** as an alternative exploration rule (`utility.explore_mode: thompson`). **Worse than the default UCB bonus**
   at every strength tried, so it is not the default.
3. **Learned single-valued-attribute schema** (`contradiction.learn_schema`): an attribute becomes single-valued after ≥ 2 outcome
   confirmations (failed actions that used a stale value, or explicit replacement wording) with posterior ≥ 0.6; counter-evidence
   (an older value still credited after a newer one arrived) blocks it. Pending conflicts are superseded retroactively.
4. **Cheaper leave-one-out** (`attribution.loo_policy: ambiguous`): replay only memories where exactly one of cited/used holds.
5. **LLM adapters** (`remem/llm.py`, `experiments/run_llm_agent.py`): `LLMAgent` (OpenAI-compatible chat, JSON answers + citations) and
   `OpenAIEmbedder`. **Not verified against a live endpoint**; covered by fake-client unit tests only. No reported number uses them.

## 9. Evaluation methodology

* **Shared candidate pool.** Every system consumes the identical over-fetched pool per query (replayed from a cache); differences come
  only from ranking, selection and lifecycle. Reported latency = pool time + the system's own re-ranking time.
* **Systems.** A flat vector, similarity only · B flat vector + static score (similarity + recency + confidence) · C ReMem ·
  cumulative ablation · leave-one-out · attribution variants · retention control.
* **Tasks** (`remem/eval/world.py`, seeded; ground truth hidden from the memory layer): `distractor`, `long_horizon`, `contradiction`,
  `rare_critical` (dormant and shown-elsewhere critical memories), `useless_frequent`, `budget_tight`/`budget_medium`, `state_change`
  (with historical `as_of` queries), and a **control**, `distractor_norecur`, where no (entity, attribute) pair recurs so per-memory
  history cannot help.
* **Agent.** A deterministic *simulated* reader with position-biased (`first`) and timestamp-aware (`latest`) modes, distraction
  proportional to irrelevant context, chance-level guessing and noisy self-citation. Everything is reported for both modes.
* **Seeds.** Train 100–103 (fit utility model), dev 200–201 (sanity checks, choosing `guard_top`, tuning the RMM-style baseline), **test 0–14 (all reported numbers)**.
  Means with 95% t-intervals over seeds; paired differences over seeds. The whole pipeline is deterministic.

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q                      # 85 tests, under a minute
python experiments/run_benchmark.py --quick    # smoke run
python experiments/run_benchmark.py            # full protocol, 15 seeds; trains the model if missing (~20 min, 12 workers; resumable)
python experiments/demo_ranking.py             # "more similar vs more useful" demonstration
python -m remem.viz.ledger_cli results/runs/distractor_seed0_C_remem/ledger.json --list --state LOW_VALUE
```

## 10. Results

**15 test seeds**, mean ± 95% t-interval, simulated agent, offline hash embedder. Full generated tables:
[`results/RESULTS.md`](results/RESULTS.md); aggregates: `results/{baseline,static,remem,ablation,new_features,sensitivity}.json`; plots: `results/plots/`.

### Headline (mean over 7 core families)

| System | Task success % (reader `first`) | (`latest`) | Gold recall % | Context precision % | Tokens/task |
|---|---:|---:|---:|---:|---:|
| A flat vector, similarity only | 62.6 ± 2.4 | 70.7 ± 2.1 | 81.5 | 17.3 | 47.3 |
| B flat vector + static score | 65.9 ± 2.4 | 70.8 ± 2.1 | 81.7 | 17.3 | 47.4 |
| RMM-style reranker (our reimplementation, see below) | 63.1 ± 1.9 | 70.7 ± 1.9 | 80.7 | 17.1 | 44.5 |
| **C ReMem** | **76.4 ± 1.8** | **76.4 ± 1.8** | 87.7 | 21.5 | **37.8** |

Paired over seeds: C − A = **+13.8 ± 1.8 pp** (`first`), **+5.6 ± 1.2 pp** (`latest`); C − B = +10.5 ± 1.8 / +5.5 ± 1.3 pp; tokens −9.5 ± 0.9 per
task. The ReMem layer adds a few milliseconds per query (Python; see `results/RESULTS.md` for latency).

**Read the `first`/`latest` gap before the headline.** A timestamp-aware reader already copes with most stale facts, so the gain
shrinks from 13.8 to 5.6 pp. Most of the `first`-reader gain is contradiction handling.

### Per family (task success %, reader `first`)

| System | distractor | long_horizon | contradiction | rare_critical | useless_frequent | budget_tight | state_change | *distractor_norecur (control)* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A similarity only | 73.5 | 73.5 | 52.6 | 74.9 | 76.0 | 41.0 | 46.6 | 68.8 |
| B + static score | 73.5 | 73.5 | 63.3 | 74.9 | 75.8 | 41.0 | 59.3 | 68.7 |
| **C ReMem** | 79.3 | 79.9 | 82.4 | 81.6 | 79.9 | 49.5 | 82.0 | **68.7** |

With 15 seeds the per-family intervals are ±2 to ±8 pp, and ReMem is ahead of similarity-only on every core family. On the control,
where per-memory history cannot help, **ReMem gives no gain (68.7 vs 68.8)**. `budget_tight` stays low for every system because long gold
facts rank poorly on similarity and nearly fill the 45-token budget.

### Cumulative ablation (reader `first`, core families)

| System | Task success % | Tokens/task | Gold recall % |
|---|---:|---:|---:|
| flat vector baseline | 62.6 ± 2.4 | 47.3 | 81.5 |
| + temporal relevance | 65.9 ± 2.4 | 47.4 | 81.7 |
| + utility score (heuristic, retrieval-history features) | 64.0 ± 2.4 | 47.1 | 80.6 |
| + learned utility model (static + retrieval-history features) | **60.5 ± 2.3** | 46.9 | 75.3 |
| + outcome feedback | 66.3 ± 2.2 | 43.4 | 84.4 |
| + adaptive retention | 66.7 ± 2.0 | 42.2 | 85.0 |
| + contradiction handling | 73.9 ± 1.8 | 42.3 | 87.7 |
| + cost-aware selection (= full without similarity guard) | 75.8 ± 1.7 | 37.6 | 86.9 |

* **Retrieval frequency is not usefulness: measured.** Retrieval-history features, heuristic or learned, *lower* success (64.0, 60.5): in
  training traces frequently retrieved memories are mostly useful, so the model learns "frequent ⇒ useful", and look-alike traps are frequent too.
* Outcome-derived signals repair this (+5.8 pp, −3.5 tokens). **Contradiction handling is the largest single step (+7.2 pp)**, concentrated in
  the two families with updates. Leave-one-out from the full system agrees: removing contradiction handling −8.1 pp, outcome feedback −6.5,
  cost-aware selection −2.1, adaptive retention −1.4; removing the similarity guard changes success by −0.6 (within noise).
* **The learned model does not beat the transparent heuristic.** Full system with the heuristic utility: 76.8 ± 1.6 (learned 76.4 ± 1.8); AUC
  0.958 vs 0.954. The learned model's advantage is calibration (Brier 0.067 vs 0.092, ECE 0.150 vs 0.218), which matters for the cost-aware
  threshold, not ranking. AUC over considered candidates is 0.954 vs 0.876 for similarity alone.

### Credit assignment (same system, different attribution; reader `first`)

| Attribution | Task success % | Credit precision | Credit recall | Extra agent calls/task |
|---|---:|---:|---:|---:|
| presence (equal credit to everything in context) | 69.8 ± 2.2 | 0.19 | 1.00 | 0 |
| usage signals only, **outcome ignored** | 78.4 ± 1.5 | 0.52 | 1.00 | 0 |
| **signals (default)** | 76.4 ± 1.8 | 0.49 | 1.00 | 0 |
| signals + leave-one-out, all memories | 82.4 ± 1.7 | 1.00 | 0.82 | 2.34 |
| **signals + leave-one-out, ambiguous only** | **82.4 ± 1.4** | 0.99 | 0.96 | **0.41** |

Equal credit is worse than signal-based credit (69.8 vs 76.4). Ambiguous-only leave-one-out matches full leave-one-out for 82% fewer extra
agent calls. **Honest caveat on the name:** crediting purely from citation/value use while *ignoring* success/failure scores slightly *higher* than
the default (78.4 vs 76.4, overlapping intervals). In this benchmark the benefit comes from *usage* signals; the success bit gates credit and drives
blame and outcome-triggered invalidation, but we could not show it adds accuracy. Flipping 10% or 20% of outcome labels leaves ReMem at 76.5 / 76.9,
consistent with that.

### Comparison with an RMM-style reranker

RMM ([arXiv 2503.08026](https://arxiv.org/abs/2503.08026)) learns a reranker from the LLM's citations with REINFORCE. `remem/baselines/rmm.py`
reimplements its §6 (residual linear embedding adaptation, Gumbel sampling, +1/−1 citation reward); it is **not the authors' code**, its gradient
is verified numerically, and its learning rate and temperature were tuned on dev seeds. Here it scores **63.1%, within noise of plain similarity
(62.6%)**, and on dev seeds more learning made it worse. This says the RMM-style mechanism did not help *in this synthetic setting with hash
embeddings and ~300 tasks per world*; it says nothing about RMM as published (different data, real embeddings, Gemini, plus its Prospective
Reflection component, which is not reimplemented). Details: [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md).

### Retention (`rare_critical`, reader `first`)

| System | Overall % | Dormant-critical % | Shown-elsewhere critical % | memories ARCHIVED |
|---|---:|---:|---:|---:|
| A similarity only (no retention) | 74.9 | 81.7 ± 8.2 | 74.2 ± 8.5 | 0 |
| C ReMem (adaptive retention) | 81.6 | 81.7 ± 8.2 | 73.3 ± 8.2 | 0.3 |
| **frequency-prune control** | 77.4 | **8.3 ± 5.0** | **33.3 ± 8.1** | **118.2** |

Frequency pruning archives about 60% of memory and destroys the rarely needed ones; the exposure/outcome-based policy leaves them untouched.
Adaptive retention itself engages rarely in these worlds (on average 6.0 LOW_VALUE, 0.6 COMPRESSED, 0.4 ARCHIVED of 207) and saves about 2% of
eligible tokens; **no accuracy benefit is claimed**.

### Contradiction and state change (reader `first`)

ReMem removes stale-memory-in-context events (`contradiction` 38.6% → 0.0% of tasks; `state_change` 53.6% → 0.0%), with **0 false invalidations**
(multi-valued `depends_on` facts are never superseded), and **0 missed** given the single-valued-attribute schema. Historical (`as_of`) queries:
78.6% vs 48.1% for similarity-only. With 20% of facts losing their attribute key (imperfect extraction) ReMem still reaches 70.4% vs 48.7% for
similarity-only (static score: 55.9%) on the two update families.

**Learned schema (none given), 15 seeds.** Stale-in-context task rate / missed invalidations per world / success:

| Setting | no schema | learned schema | schema given |
|---|---:|---:|---:|
| `state_change`, explicit update wording | 7.8% / 2.6 / 78.2 | 3.2% / 0.5 / 80.2 | 0.0% / 0 / 82.0 |
| `state_change`, **no** update wording | 11.2% / 5.1 / 77.3 | 4.5% / 1.2 / 79.8 | 0.0% / 0 / 82.3 |
| `contradiction`, explicit wording | 2.7% / 0.7 / 81.1 | 1.7% / 0.3 / 81.6 | 0.0% / 0 / 82.4 |
| `contradiction`, no wording | 5.9% / 1.7 / 79.2 | 4.5% / 0.7 / 79.6 | 0.0% / 0 / 80.2 |

Learning recovers about half of the gap to a hand-written schema where the gap is largest (`state_change` without wording: +2.5 pp, 76% of the missed
invalidations removed), and is within noise on `contradiction`. It never produced a false invalidation.

### Guard and exploration (core families, reader `first`)

| System | Task success % | Tasks where utility pushed out a top-5-similar gold % | Tokens/task |
|---|---:|---:|---:|
| A similarity only | 62.6 | 0.4 | 47.3 |
| full without guard | 75.8 ± 1.7 | **2.0** | 37.6 |
| **C ReMem (guard_top=3)** | 76.4 ± 1.8 | **0.7** | 37.8 |
| full with Thompson instead of UCB | 74.7 ± 1.8 | 0.9 | 40.8 |

The guard does what it was built for (that failure drops by ~65%); its effect on task success (+0.6 pp) is within noise. Thompson sampling is worse
than UCB (−1.7 pp, +3 tokens/task).

### Robustness (core families, `first`)

| Perturbation | A | B | C |
|---|---:|---:|---:|
| none | 62.6 | 65.9 | 76.4 |
| agent not distracted by irrelevant context (0 per item) | 73.8 | 77.5 | 88.1 |
| agent twice as distractible (0.10 per item) | 51.6 | 54.5 | 64.6 |
| 20% of outcome labels flipped | 62.6 | 65.9 | 76.9 |

The absolute level moves a lot with the assumed cost of wasted context; ReMem's lead over similarity-only does not (+14.3 / +13.0 pp).

### With real embeddings (all-MiniLM-L6-v2)

`experiments/run_real_embeddings.py`: the same 7 core families x 15 test seeds and reader `first`, but every candidate pool comes from a real,
free, local embedding model instead of the hash embedder. The learned utility model was *not* retrained for these embeddings.

| System | Task success (95% CI) | Gold memory in candidate pool |
|---|---:|---:|
| A similarity only | 69.1 ± 2.0 | 99.4% |
| B static score | 70.3 ± 1.9 | 99.4% |
| **C ReMem** | **79.2 ± 1.8** | 99.4% |

Paired difference C - A: **+10.2 ± 1.4 pp** (hash embedder: +13.8 ± 1.8). Better embeddings raise the baseline and close the "gold never retrieved"
gap (3% -> 0.6%); most of ReMem's gain remains, concentrated where facts change (state_change 53.2 -> 83.9, contradiction 58.2 -> 82.0) and smallest
under a tight token budget (60.6 -> 61.9). Raw rows: `results/real_embeddings.json`.

### With a real LLM (free, local qwen2.5:3b)

`experiments/run_llm_agent.py --ollama`: the agent is a real 3-billion-parameter model running locally through Ollama (CPU only, temperature 0);
it sees the served memories and must answer in JSON, and success is checked against the world's hidden truth. 80 tasks x 3 seeds per family,
hash embedder, A = similarity only, C = ReMem. ReMem learns from the LLM's real outcomes and citations.

| Family | Output format | A similarity | C ReMem | C - A (95% CI over seeds) | Tasks with a stale fact in context (A / C) |
|---|---|---:|---:|---:|---:|
| contradiction | JSON schema (exact keys) | 84.6 | 81.7 | -2.9 ± 23.3 | 29.6% / 0.0% |
| distractor | JSON schema (exact keys) | 87.1 | 87.5 | +0.4 ± 4.7 | 0 / 0 |
| contradiction | free JSON | 83.3 | 75.4 | -7.9 ± 21.1 | 29.6% / 0.0% |
| distractor | free JSON | 80.8 | 75.4 | -5.4 ± 22.0 | 0 / 0 |

**No measurable gain with a real model.** What happened, in order:

1. The first run let the model write free-form JSON. ReMem scored *lower*. Recording every prompt and reply
   (`experiments/trace_llm_prompts.py`, `results/llm_agent/traces_distractor_2_object.jsonl`) showed that 22 of ReMem's 26 failures in the
   worst world, and 10 of similarity's 15, contained the right answer under a malformed key (e.g. `"project.who_is_responsible_for_recs": "alice"`
   instead of `"recs.owner"`). ReMem's contexts hold more real facts and fewer junk notes, which seems to derail the small model's formatting
   more often.
2. The runner then constrained the output with a JSON schema of the exact answer keys (as the Claude adapter already does) and both systems were
   re-run. Both numbers are reported, not just the second.
3. With the format fixed, the systems are within noise. ReMem still removes every stale fact from the context, but this model, told to prefer the
   most recent memory, mostly copes with stale facts on its own, so removing them buys little. The simulated reader in the main benchmark is hurt
   more by stale and irrelevant context than this LLM is; the headline gains depend on that assumption.

Three seeds with one small model are too few to rule out a small effect either way; a larger model or more seeds may differ.

### Trying to make the learned model beat the heuristic (on-policy retraining)

The learned utility was fitted on what the heuristic chose to show, then deployed as its own ranker (distribution shift).
`experiments/train_utility_dagger.py` collects new traces with the learned model itself (3 rounds, DAgger-style aggregation, 170k rows) and
compares on DEV seeds only: heuristic 75.7%, round 0 76.4%, rounds 1-3 76.3-76.4%. **No gain**, so the round-0 model stays; the new on-policy
data adds rows but little new information because the two policies already serve nearly the same memories. Report:
`results/models/dagger/report.json`.

### LongMemEval (public benchmark, retrieval only)

`experiments/run_longmemeval.py` on `longmemeval_s_cleaned.json` (ICLR 2025; ~50 sessions of chat history per question), MiniLM embeddings, every
turn stored as one memory, no LLM. 470 questions (the 30 abstention questions have no evidence and are excluded). *turn@k*: an evidence turn
(`has_answer`) is among the k retrieved memories; *session@k*: a retrieved memory comes from an evidence session.

| System | turn@5 | turn@10 | session@5 | session@10 |
|---|---:|---:|---:|---:|
| similarity only | 74.7 | 86.0 | 95.1 | 97.0 |
| static score | 74.7 | 86.0 | 95.1 | 97.0 |
| ReMem | 74.7 | 86.0 | 95.1 | 97.0 |

**No difference, as expected:** every question has its own history and is asked once, so ReMem never receives feedback, and raw turns carry no
entity/attribute keys for supersession. In this run recency was measured from the wall clock, so every turn looked equally old and recency had
no effect (`results/longmemeval_clocknow/retrieval.json`).

A second full run measures time from each question's date (`search(..., now=question_date)`, `results/longmemeval/retrieval.json`):

| System | turn@5 | turn@10 | session@5 | session@10 |
|---|---:|---:|---:|---:|
| similarity only | 70.4 | 82.3 | 91.1 | 93.2 |
| static score | 71.7 | 82.8 | 91.1 | 93.4 |
| ReMem | 71.7 | 82.8 | 91.1 | 93.4 |

All numbers are lower here because of a dataset quirk: 69 of the 470 questions have haystack sessions timestamped *after* the question (41 of
them evidence sessions), and every system, as designed, does not serve memories recorded after the moment it is asked about. Since all three
systems lose the same turns, the comparison inside this table is fair: recency from the question date helps a little (+1.3 pp turn@5, largest on
knowledge-update questions, 80.6 -> 83.3), and ReMem adds nothing beyond the static score. Only retrieval is measured; answer accuracy would need
an LLM and the official judge.

### Ranking demonstration and lifecycle

`python experiments/demo_ranking.py` writes `results/demo_ranking.md`: tasks where a look-alike with higher similarity is **not** served while a
lower-similarity gold memory is (104 such tasks in the first test world). `python -m remem.viz.dashboard <store>` renders every memory's state,
credit, supersession links and lifecycle as a self-contained HTML page; `python -m remem.viz.ledger_cli` is the terminal equivalent. Per-feature
"utility components" come from a regression with correlated features, so individual component signs can be large and opposite; read them as a
decomposition of the score, not as independent effects.

## 11. Limitations (read these)

* **The headline gain is not confirmed with a real LLM.** With a free local 3B model as the agent, ReMem and similarity search are within noise
  (3 seeds, 2 families, one small model). The headline numbers use a deterministic simulated reader whose distraction strength (0.05 per
  irrelevant item), sensitivity to stale facts and chance-guessing are assumptions; the real model was less hurt by stale facts than the simulated
  reader. No larger or hosted model has been tested.
* **No gain on LongMemEval.** Each question there has its own history and is asked once, so per-memory feedback never accumulates, and raw turns
  carry no entity/attribute keys for supersession (without LLM extraction). ReMem ranks like the static baseline there, as in the
  `distractor_norecur` control. Only retrieval recall was measured, not answer accuracy.
* **Offline hash embedder in most tables.** Hash embeddings with a hand-written synonym lexicon replace a trained embedding model; about 3% of gold
  facts are never retrieved (no stemming). The headline comparison was repeated with MiniLM (+10.2 pp instead of +13.8); the ablations, retention,
  credit-assignment and LLM tables were not.
* **Only flat vector stores are built in** (numpy, FAISS, Chroma). An earlier version also compared against a hierarchical-memory backend; that
  backend and every comparison against it were removed, so no claim here concerns any hierarchical memory system.
* **Synthetic, author-designed tasks.** They exercise the mechanisms and some (traps, repeated pairs) favour per-memory history by construction; the
  control family and the negative results are included for that reason. Gains require *recurring* queries.
* **Extraction is assumed in the benchmark.** The detector uses `entities`, `attribute_keys` and `semantic_role` fields as an LLM extractor would emit;
  there they come from the generator (noise sensitivity above). `LLMFactExtractor` exists but its quality with a real model is unmeasured.
* **Library supersession rule.** Through `ReMem.add`, two `"state"` facts with the same entity and attribute replace each other; a multi-valued
  attribute stored as `"state"` would lose values. Store those as `"preference"` or `"note"`.
* **Feedback only on shown memories** (bandit feedback); exploration is a heuristic; memories never shown never get labels. In the MCP server,
  inferred feedback (when the agent never calls `feedback`) is a guess and has not been evaluated.
* **Credit assignment is not causal.** Default credit precision is only 0.49; leave-one-out fixes it at a cost of extra agent calls (0.41–2.34/task).
* **The learned model does not beat the heuristic**, and three rounds of on-policy retraining did not change that (DEV seeds).
* **Several mechanisms did not help or are within noise:** retention (accuracy), similarity guard (success), Thompson sampling (worse), outcome gating
  (usage-only matches it), and the RMM-style reranker (within noise of similarity; our reimplementation, not the authors' code).
* **Statistics.** 15 seeds, t-intervals over seeds; worlds come from one author-written generator, so seeds are not independent real-world samples.
  Tokens are a word+punctuation proxy by default (`set_tokenizer("tiktoken")` for real counts).
* **Process disclosure.** Hyper-parameters were set a priori, except those chosen on dev seeds: `guard_top=3`, rejecting Thompson sampling, and the
  RMM-style baseline's learning rate and temperature. Dev seeds were also used for fixing defects found by unit tests (detector false positives, an LOO
  discount equal to the threshold, trap invalidation by the outcome path) and for raising critical tasks per world from 6+6 to 8+8. The test seeds were run
  several times as the code changed; the reported numbers are from the final run only.

## 12. Failure analysis

[`results/failure_analysis.md`](results/failure_analysis.md) (+ `failures.jsonl`): per-task tags, all 15 test seeds, reader `first`.

| Category | A similarity only | C ReMem |
|---|---:|---:|
| `gold_not_in_pool` (backend never offered it) | 3.1% | 3.1% |
| `gold_pooled_not_selected` | 19.8% | 11.3% |
| `utility_demoted_gold` (top-5 by similarity, pushed out) | 0.6% | 0.7% |
| `useful_filtered` (archived/invalidated although needed) | 0 | 0 |
| `irrelevant_retrieved` (failed + distracted) | 18.5% | 16.1% |
| `outdated_preferred` | 8.6% | **0.0%** |
| `success_without_memory` (lucky guess) | 4.4% | 2.7% |
| `fail_despite_gold` (all gold served, still failed) | 20.4% | 13.2% |

What the system still gets wrong: (1) the utility layer still pushes out a highly similar correct memory in 0.7% of tasks (2.0% without the guard);
(2) irrelevant context still causes 16% of failures; (3) 3.1% of gold facts are never in the pool, a limitation of the offline embedder; (4) about one task
in eight fails with every needed memory present, because the simulated agent can be distracted.

## 13. Future work

Repeat the real-LLM test with larger models and more seeds, and LongMemEval with LLM fact extraction and the QA judge; measure how well an LLM's
self-citations track real dependence; evaluate `LLMFactExtractor` quality; let per-memory utility generalise to unseen memories (the RMM idea of
adapting the embedding space, combined with ReMem's outcome gating); off-policy correction for the learned model; more backends (Qdrant, pgvector).

## Repository map

```
remem/                 the library: api.py (ReMem), mcp_server.py, system.py, ledger, utility, attribution, contradiction, retention, selection
remem/backends/        numpy vector, FAISS, Chroma            remem/embedders.py   sentence-transformers
remem/extract.py       fact extraction                         remem/integrations/  LangChain retriever
remem/baselines/       RMM-style reranker (reimplementation)  remem/viz/           dashboard (HTML), ledger_cli
remem/eval/            synthetic worlds, simulated agent, runner, metrics, failure analysis, report
experiments/           run_benchmark.py, train_utility.py, train_utility_dagger.py, tune_rmm.py, demo_ranking.py,
                       run_llm_agent.py (--ollama: free local LLM), trace_llm_prompts.py, run_longmemeval.py, run_real_embeddings.py
examples/              basic_api.py, quickstart.py, langchain_retriever.py, claude_desktop_mcp.md
configs/systems.yaml   every compared system                   tests/ (85 tests)
results/               RESULTS.md, failure_analysis.md, *.json, plots/, runs/<…>/ledger.json, models/, records/
docs/                  DESIGN.md, RELATED_WORK.md, make_figures.py (README GIF + chart)   .github/workflows/  CI
```

License: MIT (see `LICENSE`). © 2026 Viswesh Kesarla.

## Acknowledgement

Motivated by gaps observed in the MemForest agent-memory system (Chen et al.): https://github.com/Concyclics/MemForest
