from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


def hits_to_documents(hits: list[dict]) -> list[Document]:
    """Map raw Milvus hybrid-search hits to LangChain documents."""
    return [
        Document(
            page_content=h["text"],
            metadata={
                "parent_text": h["parent_text"],
                "doc_title": h["doc_title"],
                "parent_doc_id": h["parent_doc_id"],
                "score": h["score"],
            },
        )
        for h in hits
    ]


class HybridRetriever(BaseRetriever):
    """LangChain retriever bridging Milvus dense+sparse hybrid search.

    Kept free of FastAPI/agent imports so it can be reused by the query API,
    the agent tool layer, and the evaluation harness.
    """

    store: object
    embed_service: object
    settings: object
    fusion_top_k: int | None = None

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        q_embed = self.embed_service.encode_query(query)
        hits = self.store.hybrid_search(
            query_dense=q_embed["dense"],
            query_sparse=q_embed["sparse"],
            dense_top_k=self.settings.hybrid_dense_top_k,
            sparse_top_k=self.settings.hybrid_sparse_top_k,
            rrf_k=self.settings.rrf_k,
            fusion_top_k=self.fusion_top_k or self.settings.hybrid_fusion_top_k,
        )
        return hits_to_documents(hits)
