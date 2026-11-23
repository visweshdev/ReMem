# Support routing with outcome feedback

The assistant reads retrieved ticket-routing memories, proposes a queue, and reports whether a local dispatcher accepted the route. A support team can carry the ReMem store across shifts. The example opens the store again for every session and adds a hand-off note. It uses a deterministic reader and a simulated dispatcher with known routing rules, so failures and counterfactual replays are reproducible.

```bash
python examples/workflow_assistant.py --split dev
python examples/workflow_assistant.py --split test --out data/repro/workflow_assistant.json
```

## Held-out results

Five development seeds precede the 20 fixed test seeds. Each test seed contains 64 warm-up tasks and 48 held-out tasks from a separate set of question templates. Feedback runs during warm-up. Held-out searches receive no outcomes and continue to record exposure. Every arm uses the same documents, reader, task order, three-hit cap, and 100-proxy-token budget. One third of the warm-up tasks deliberately report the last shown memory as the citation.

| Retrieval | Correct routes | Mean context tokens | Mean indexing time | Mean per-seed retrieval p95 | Mean final store |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain cosine | 73.85% | 44.0 | 43.8 ms | 2.21 ms | 98,786 B |
| ReMem, feedback off | 69.48% | 44.0 | 42.4 ms | 3.75 ms | 111,478 B |
| ReMem, citation and answer signals | **95.63%** | 42.0 | 37.7 ms | 3.81 ms | 157,188 B |
| ReMem, leave-one-out replay | 95.31% | **38.4** | 41.9 ms | 4.20 ms | 146,024 B |

Outcome signals improved held-out routing by **21.77 percentage points** over plain cosine (paired seed-bootstrap 95% interval **+19.48 to +24.06 pp**) and **26.15 pp** over feedback-off (**+23.75 to +28.54 pp**). This benefit applies to repeated routing tasks with stable rules and explicit acceptance feedback. The scenario uses intentionally similar distractors and a small corpus. [Held-out results and configurations](../results/workflow_assistant/held_out.json)

Replay reduced mean credit to non-policy records from 30.48 to 8.73 per episode. It used about 145 extra deterministic reader calls per episode. The paired accuracy difference from signal-only feedback was −0.31 pp (95% interval −2.92 to +2.19 pp). Credit can stay ambiguous when a training memory repeats the correct queue, and removing one of two redundant memories can leave the route unchanged.

## Costs and failure modes

- Similar phrasing makes example tickets outrank the authoritative team handbook. The dispatcher rejects the wrong queue and gives the failure a concrete outcome. Repeated successful tasks provide a signal for the handbook.
- Training glossaries sometimes repeat the correct queue. The lexical credit heuristic counts that overlap even when the reader used a different memory. Removing each candidate before replay helps diagnose dependence, with extra cost proportional to the number of candidates replayed.
- Each store grows from 16 to 30 memories. The initial stores averaged about 24 KB. The table shows JSON store size and local wall times from one Windows run. The plain arm uses the same ReMem API and ledger with cosine ranking, which makes this an algorithm comparison inside one implementation. Retrieval latency excludes feedback, replay, and session reopen time. Indexing time includes the initial write and save of 16 short records. These are small-corpus desktop measurements.

With `attribution.method="signals_loo"`, `ReMem.feedback(..., replay=...)` passes the reduced set of retrieved `Hit` objects to a callback. For a live assistant, replay against a sandbox or a side-effect-free evaluator so repeating an action cannot submit another ticket.
