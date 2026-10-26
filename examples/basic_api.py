"""The high-level API in 30 lines: add -> search -> feedback -> better results next time.

    python examples/basic_api.py

A plain vector store would keep ranking the look-alike note first forever (it is the most similar text).
ReMem learns from feedback that it never helps, and retires the outdated fact when the newer one arrives.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from a source checkout without installing

from remem import ReMem

mem = ReMem()                                                  # in-memory; ReMem("~/.remem/demo") persists to disk
DAY = 86400.0
old = mem.add("The deploy pipeline targets Kubernetes cluster east-1.", entity="deploy", attribute="cluster", when=1 * DAY)
note = mem.add("Note: someone asked which cluster the deploy pipeline targets; nobody answered.", role="note", when=2 * DAY)
QUESTION = "Which cluster does the deploy pipeline target?"


def show(label):
    res = mem.search(QUESTION, k=2, task="deploy")
    print(f"{label:<31}" + " | ".join(f"{h.text[:46]:<46} score={h.score:.2f}" for h in res.items))
    return res


res = show("1. first search:")
new = mem.add("The deploy pipeline moved to cluster west-2.", entity="deploy", attribute="cluster", when=10 * DAY)
res = show("2. after the update:")         # the east-1 fact is superseded and no longer served
for _ in range(6):                          # the agent keeps succeeding using the west-2 fact, never the note
    mem.feedback(res.search_id, success=True, used_ids=[new])
    res = mem.search(QUESTION, k=2, task="deploy")
mem.feedback(res.search_id, success=True, used_ids=[new])
show("3. after 7 rounds of feedback:")

for fid in (old, note, new):
    info = mem.inspect(fid)
    print(f"   {info['state']:<12} used {info['credited']:.1f}/{info['shown']}x  {info['text']}")
