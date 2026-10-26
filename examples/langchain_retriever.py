"""ReMem as a LangChain retriever (no LLM needed to run this file).

    pip install ".[langchain]"
    python examples/langchain_retriever.py

Use `retriever` anywhere LangChain expects a retriever (e.g. in an LCEL chain). After the chain has run, call
`retriever.feedback(...)` so ReMem learns which documents actually helped.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from remem import ReMem
from remem.integrations.langchain import ReMemRetriever

mem = ReMem()
mem.add("Customer Acme is on the Enterprise plan.", entity="acme", attribute="plan")
mem.add("Reminder: ask Acme about their plan renewal sometime.", role="note")
retriever = ReMemRetriever(memory=mem, k=2, task="support")

for round_ in range(1, 6):
    docs = retriever.invoke("Which plan is Acme on?")
    # ... your chain would answer from `docs` here; pretend it answered correctly from the plan fact ...
    used = [d.metadata["id"] for d in docs if "Enterprise" in d.page_content]
    retriever.feedback(success=True, used_ids=used)
    print(f"round {round_}: " + " | ".join(f"{d.page_content[:40]} ({d.metadata['score']:.2f})" for d in docs))
