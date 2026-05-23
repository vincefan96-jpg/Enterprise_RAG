import json
import asyncio
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from app.models.schemas import QueryRequest, QueryResponse
from app.config import get_settings

router = APIRouter(prefix="/api", tags=["query"])


class HybridRetriever(BaseRetriever):
    store: object
    embed_service: object
    settings: object

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        q_embed = self.embed_service.encode_query(query)
        hits = self.store.hybrid_search(
            query_dense=q_embed["dense"],
            query_sparse=q_embed["sparse"],
            dense_top_k=self.settings.hybrid_dense_top_k,
            sparse_top_k=self.settings.hybrid_sparse_top_k,
            rrf_k=self.settings.rrf_k,
            fusion_top_k=self.settings.hybrid_fusion_top_k,
        )
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


def _service_state(request: Request):
    settings = get_settings()
    state = request.app.state
    return (
        settings,
        state.milvus_store,
        state.embedding_service,
        state.reranker_service,
        state.llm_service,
    )


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest):
    settings, store, embed, reranker, llm_service = _service_state(request)
    retriever = HybridRetriever(store=store, embed_service=embed, settings=settings)

    docs = await asyncio.to_thread(
        llm_service.retrieve_and_rerank, body.question, retriever, reranker
    )
    context = llm_service._format_context(docs)
    sources = llm_service._extract_sources(docs)

    chain = llm_service.build_answer_chain()
    answer = await chain.ainvoke({"context": context, "question": body.question})

    return QueryResponse(answer=answer, sources=sources)


@router.post("/query/stream")
async def query_stream(request: Request, body: QueryRequest):
    settings, store, embed, reranker, llm_service = _service_state(request)
    retriever = HybridRetriever(store=store, embed_service=embed, settings=settings)

    async def generate():
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': '正在检索相关文档...'})}\n\n"

            docs = await asyncio.to_thread(
                llm_service.retrieve_and_rerank, body.question, retriever, reranker
            )
            context = llm_service._format_context(docs)
            sources = llm_service._extract_sources(docs)

            yield f"data: {json.dumps({'type': 'status', 'message': '正在生成回答...'})}\n\n"

            chain = llm_service.build_answer_chain()
            async for chunk in chain.astream({"context": context, "question": body.question}):
                yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"

            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
