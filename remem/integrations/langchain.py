"""LangChain integration: ReMem as a retriever whose ranking improves from feedback.

    from remem import ReMem
    from remem.integrations.langchain import ReMemRetriever

    retriever = ReMemRetriever(memory=ReMem("~/.remem/app"), k=5, task="support")
    docs = retriever.invoke("Which plan is the customer on?")     # standard LangChain retriever call
    ... run your chain ...
    retriever.feedback(success=True, used_ids=[docs[0].metadata["id"]])   # close the loop

Each Document carries `metadata = {id, score, similarity, state, search_id}`. `feedback()` reports on the most
recent search unless a `search_id` is given.

    pip install langchain-core
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr

from remem.api import ReMem


class ReMemRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    memory: ReMem
    k: int = 5
    task: str = "default"
    _last_search_id: str | None = PrivateAttr(default=None)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        res = self.memory.search(query, k=self.k, task=self.task)
        self._last_search_id = res.search_id or None
        return [Document(page_content=h.text, metadata={"id": h.id, "score": h.score, "similarity": h.similarity,
                                                         "state": h.state, "search_id": res.search_id})
                for h in res.items]

    def feedback(self, success: bool, *, used_ids: list[str] | None = None, answer: str | None = None,
                 search_id: str | None = None) -> dict[str, float]:
        sid = search_id or self._last_search_id
        if not sid:
            raise ValueError("no search to give feedback on; call the retriever first")
        if sid == self._last_search_id:
            self._last_search_id = None
        return self.memory.feedback(sid, success, used_ids=used_ids, answer=answer)

    def add_texts(self, texts: list[str], **kwargs: Any) -> list[str]:
        """Convenience: store plain texts (keyword arguments are passed to ReMem.add)."""
        return [self.memory.add(t, **kwargs) for t in texts]
