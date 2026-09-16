"""Adapters to make project services compatible with Ragas interfaces."""

from langchain_core.embeddings import Embeddings


class BGEM3Embeddings(Embeddings):
    """LangChain Embeddings wrapper around the project's BGE-M3 EmbeddingService.

    Extracts only the dense vector for Ragas compatibility.
    """

    def __init__(self, embed_service):
        self._svc = embed_service

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = self._svc.encode(texts, max_length=512)
        return [r["dense"] for r in results]

    def embed_query(self, text: str) -> list[float]:
        return self._svc.encode_query(text)["dense"]
