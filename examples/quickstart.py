"""Minimal use of ReMem as a library: store facts, retrieve, report the outcome, repeat.

    python examples/quickstart.py

Shows the three things ReMem adds on top of a plain vector store:
  1. a superseded fact ("API v1") stops being served once a newer one ("migrated to v2") arrives;
  2. a look-alike memory that never helps is demoted after a few uses;
  3. every memory keeps an inspectable history.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from a source checkout without installing

from remem.backends import StoredFact, VectorBackend
from remem.config import ContradictionConfig, RetentionConfig, SelectionConfig, SystemConfig, UtilityConfig
from remem.offline import ConceptHashEmbedder, tokenize
from remem.system import IngestedFact, ReMemSystem
from remem.types import AgentTrace, Outcome

DAY = 86400.0
embedder = ConceptHashEmbedder(dim=256)                 # swap for remem.llm.OpenAIEmbedder in real use
backend = VectorBackend(embedder)
memory = ReMemSystem(
    SystemConfig(
        ranker="heuristic",                             # transparent utility; no trained model needed
        outcome_feedback=True,
        utility=UtilityConfig(kind="heuristic"),
        contradiction=ContradictionConfig(enabled=True, exclusive_attributes=("api_version",)),
        retention=RetentionConfig(mode="adaptive"),
        selection=SelectionConfig(top_k=2),
    ),
    embedder,
)


def remember(fact_id, text, day, entity="", attribute="", role="state"):
    fact = StoredFact(fact_id, text, day * DAY, (entity,) if entity else (), (attribute,) if attribute else (),
                      semantic_role=role)
    backend.ingest([fact])
    memory.ingest(IngestedFact(fact.fact_id, fact.text, fact.time, fact.entities, fact.attribute_keys, 1.0, role))


def ask(task_id, question, day, correct_value):
    """Retrieve, let a toy 'agent' answer from the served memories, then tell ReMem how it went."""
    pool, _ = backend.candidates(question, k=10)
    ctx = memory.retrieve(task_id, question, pool, task_class="api", now=day * DAY)
    served = [s.cand for s in ctx.selected]
    answer = next((v for c in served for v in ("v1", "v2") if v in tokenize(c.text)), "unknown")
    used = [c.fact_id for c in served if answer in tokenize(c.text)]
    memory.observe(ctx, AgentTrace({"api": answer}, cited_ids=used, tool_args={"api": answer}),
                   Outcome(success=(answer == correct_value)))
    return [c.text for c in served], answer


remember("fact_1", "The billing service exposes API version v1.", day=1, entity="billing", attribute="api_version")
remember("fact_2", "Notes: someone asked which API version billing exposes; no answer recorded.", day=2,
         entity="billing", attribute="api_version", role="event")     # a look-alike that never helps

print("before the migration:", ask(1, "Which API version does billing expose?", day=3, correct_value="v1"))
remember("fact_3", "The billing service migrated and now exposes API version v2.", day=10,
         entity="billing", attribute="api_version")
for t in range(2, 9):
    served, answer = ask(t, "Which API version does billing expose?", day=10 + t, correct_value="v2")
print("after the migration: ", (served, answer))

for fid in ("fact_1", "fact_2", "fact_3"):
    rec = memory.ledger.get(fid)
    print(f"{fid}: state={rec.state.value:<11} retrieved={rec.n_retrieved} credited={rec.n_credited:.1f} | {rec.original_text}")
