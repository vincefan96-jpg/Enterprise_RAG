import asyncio
import contextlib
import json
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from app.models.schemas import QueryRequest, QueryResponse
from app.config import get_settings

router = APIRouter(prefix="/api", tags=["query"])


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _history_to_messages(history, max_turns: int) -> list:
    if not history:
        return []
    messages = []
    for turn in history[-max_turns * 2:]:
        if turn.role == "user":
            messages.append(HumanMessage(content=turn.content))
        elif turn.role in ("assistant", "ai"):
            messages.append(AIMessage(content=turn.content))
    return messages


def _require_services(request: Request) -> None:
    state = request.app.state
    missing = [
        name
        for name, attr in (
            ("Milvus", "milvus_store"),
            ("Embedding", "embedding_service"),
            ("LLM", "llm_service"),
        )
        if getattr(state, attr, None) is None
    ]
    if missing:
        raise HTTPException(
            503, f"服务未就绪: {'、'.join(missing)}；请检查后端启动日志"
        )


def _get_checkpointer(request: Request):
    checkpointer = getattr(request.app.state, "agent_checkpointer", None)
    if checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver

        checkpointer = InMemorySaver()
        request.app.state.agent_checkpointer = checkpointer
    return checkpointer


def _build_graph(request: Request, settings):
    from app.agent.graph import build_agent_graph
    from app.agent.nodes import AgentServices

    services = AgentServices(
        settings=settings,
        store=request.app.state.milvus_store,
        embed_service=request.app.state.embedding_service,
        reranker_service=request.app.state.reranker_service,
        llm_service=request.app.state.llm_service,
    )
    return build_agent_graph(
        services,
        enable_correction=True,
        checkpointer=_get_checkpointer(request),
    )


def _get_graph(request: Request, settings):
    graph = getattr(request.app.state, "agent_graph", None)
    if graph is None:
        graph = _build_graph(request, settings)
        request.app.state.agent_graph = graph
    return graph


def _thread_config(body: QueryRequest) -> dict:
    return {"configurable": {"thread_id": body.session_id or uuid.uuid4().hex}}


def _base_input(body: QueryRequest, seed: list) -> dict:
    return {
        "question": body.question,
        "messages": [*seed, HumanMessage(content=body.question)],
        "search_query": "",
        "sub_questions": [],
        "documents": [],
        "iterations": 0,
        "grade": "",
        "grade_reason": "",
        "answer": "",
        "sources": [],
    }


async def _prepare_input(graph, config, body: QueryRequest, settings) -> dict:
    """Seed prior history only when the thread has no server-side memory yet."""
    existing = None
    try:
        snapshot = await graph.aget_state(config)
        if snapshot is not None:
            existing = (snapshot.values or {}).get("messages")
    except Exception:
        existing = None
    seed = [] if existing else _history_to_messages(body.history, settings.agent_history_turns)
    return _base_input(body, seed)


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest):
    settings = get_settings()
    _require_services(request)
    graph = _get_graph(request, settings)
    config = _thread_config(body)
    state_in = await _prepare_input(graph, config, body, settings)
    result = await graph.ainvoke(state_in, config)
    return QueryResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
    )


@router.post("/query/stream")
async def query_stream(request: Request, body: QueryRequest):
    settings = get_settings()
    _require_services(request)
    queue: asyncio.Queue = asyncio.Queue()

    async def emit(payload: dict) -> None:
        await queue.put(payload)

    from app.agent.nodes import reset_emit, set_emit

    graph = _get_graph(request, settings)
    config = _thread_config(body)

    async def run():
        try:
            state_in = await _prepare_input(graph, config, body, settings)
            await graph.ainvoke(state_in, config)
        except Exception as e:
            await queue.put({"type": "error", "message": str(e)})
        finally:
            await queue.put(None)

    async def generate():
        token = set_emit(emit)
        task = asyncio.create_task(run())
        try:
            yield _sse({"type": "status", "message": "正在处理..."})
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield _sse(item)
            yield "data: [DONE]\n\n"
        finally:
            reset_emit(token)
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    return StreamingResponse(generate(), media_type="text/event-stream")
