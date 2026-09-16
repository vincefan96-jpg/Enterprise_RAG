"""Retrieval tool layer.

The agent binds these as tools and chooses which to call; the actual execution
is performed by the retrieve node so it can accumulate the underlying
documents for grading and generation.
"""

from langchain_core.documents import Document
from langchain_core.tools import tool

from app.services.retriever import HybridRetriever, hits_to_documents


def search_documents(
    store, embed_service, settings, query: str, fusion_top_k: int | None = None
) -> list[Document]:
    """Run hybrid (dense + sparse + RRF) retrieval against the knowledge base.

    ``fusion_top_k`` overrides the configured candidate count; used by the
    corrective loop to widen the net on retries.
    """
    retriever = HybridRetriever(
        store=store,
        embed_service=embed_service,
        settings=settings,
        fusion_top_k=fusion_top_k,
    )
    return retriever.invoke(query)


def limit_documents(docs: list[Document], max_docs: int) -> list[Document]:
    """Keep chunks belonging to at most ``max_docs`` distinct documents.

    Retrieval order is preserved, so the most relevant documents survive.
    Trimming the cross-document noise keeps the prompt focused when the
    knowledge base contains several product lines with similar clauses.
    """
    if max_docs <= 0:
        return docs

    titles: set[str] = set()
    kept: list[Document] = []
    for doc in docs:
        title = doc.metadata.get("doc_title", "未知")
        if title not in titles:
            if len(titles) >= max_docs:
                continue
            titles.add(title)
        kept.append(doc)
    return kept


def search_documents_batch(
    store, embed_service, settings, queries: list[str], fusion_top_k: int | None = None
) -> list[list[Document]]:
    """Hybrid search for several queries; queries are embedded in one batch.

    Used by parallel retrieval so sub-questions hit the (non-thread-safe)
    embedding model once instead of racing on it.
    """
    if not queries:
        return []

    embeddings = embed_service.encode(queries, max_length=512)
    results = []
    for embedding in embeddings:
        hits = store.hybrid_search(
            query_dense=embedding["dense"],
            query_sparse=embedding["sparse"],
            dense_top_k=settings.hybrid_dense_top_k,
            sparse_top_k=settings.hybrid_sparse_top_k,
            rrf_k=settings.rrf_k,
            fusion_top_k=fusion_top_k or settings.hybrid_fusion_top_k,
        )
        results.append(hits_to_documents(hits))
    return results


def format_docs(docs: list[Document]) -> str:
    if not docs:
        return "无结果"
    parts = []
    for d in docs[:5]:
        title = d.metadata.get("doc_title", "未知")
        text = d.metadata.get("parent_text") or d.page_content
        parts.append(f"[{title}] {text[:300]}")
    return "\n---\n".join(parts)


def make_tools(store, embed_service, settings) -> list:
    """Build the tools the retrieval agent may call (schemas only; executed by nodes)."""

    @tool
    def search_knowledge_base(query: str) -> str:
        """在企业知识库中检索与 query 相关的文档内容。query 应为简明的检索词。"""
        return format_docs(search_documents(store, embed_service, settings, query))

    @tool
    def list_documents() -> str:
        """列出知识库中所有文档的标题，用于回答"有哪些文档 / 制度"这类元问题。"""
        titles = store.list_documents()
        return "、".join(titles) if titles else "知识库为空"

    return [search_knowledge_base, list_documents]
