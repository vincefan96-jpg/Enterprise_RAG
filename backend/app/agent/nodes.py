import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.agent.prompts import (
    AGENT_RETRIEVE_PROMPT,
    DECOMPOSE_PROMPT,
    GRADE_PROMPT,
    REFINE_PROMPT,
    RETRIEVE_TASK,
    REWRITE_PROMPT,
    ROUTE_PROMPT,
)
from app.agent.tools import (
    format_docs,
    limit_documents,
    make_tools,
    search_documents,
    search_documents_batch,
)


class RouteDecision(BaseModel):
    route: Literal["direct", "knowledge", "multihop"]


class GradeDecision(BaseModel):
    sufficient: bool
    reason: str


class SubQuestions(BaseModel):
    questions: list[str]


INSUFFICIENT_NOTE = (
    "\n\n（注：现有文档信息可能不足以完整回答该问题，"
    "以上为基于已检索内容的尽力回答。）"
)


async def _noop_emit(payload: dict) -> None:
    return None


# Per-request trace sink; falls back to AgentServices.emit when unset.
_emit_sink: ContextVar = ContextVar("agent_emit_sink", default=None)


def set_emit(sink) -> object:
    return _emit_sink.set(sink)


def reset_emit(token) -> None:
    _emit_sink.reset(token)


@dataclass
class AgentServices:
    settings: object
    store: object
    embed_service: object
    reranker_service: object
    llm_service: object
    # Async sink for trace/content events. The API injects a queue writer;
    # defaults to a no-op for sync callers (e.g. evaluation).
    emit: object = _noop_emit


async def _emit(services: AgentServices, payload: dict) -> None:
    sink = _emit_sink.get() or services.emit
    await sink(payload)


def _format_history(messages: list) -> str:
    lines = []
    for m in messages:
        kind = getattr(m, "type", "")
        if kind == "human":
            lines.append(f"用户: {getattr(m, 'content', '')}")
        elif kind == "ai":
            lines.append(f"助手: {getattr(m, 'content', '')}")
    return "\n".join(lines) or "（无历史）"


def _recent_history(messages: list, max_turns: int) -> list:
    if max_turns <= 0:
        return list(messages)
    return list(messages[-max_turns * 2:])


def _dedupe_documents(existing: list, new: list) -> list:
    merged = list(existing)
    seen = {d.metadata.get("parent_doc_id") for d in merged}
    for doc in new:
        parent_id = doc.metadata.get("parent_doc_id")
        if parent_id not in seen:
            seen.add(parent_id)
            merged.append(doc)
    return merged


def _historic_messages(state: dict) -> list:
    """Everything before the current turn's question."""
    messages = state.get("messages", []) or []
    return messages[:-1] if messages else []


def make_route_node(services: AgentServices):
    structured = services.llm_service.llm.with_structured_output(
        RouteDecision, method="function_calling"
    )

    async def route_node(state: dict) -> dict:
        question = state.get("question", "")
        try:
            decision = await structured.ainvoke(
                [SystemMessage(content=ROUTE_PROMPT), HumanMessage(content=question)]
            )
            route = decision.route
        except Exception as e:
            print(f"WARNING: route classification failed ({e}); falling back to knowledge")
            route = "knowledge"
        await _emit(services, {"type": "step", "node": "route", "detail": route})
        return {"route": route}

    return route_node


def make_rewrite_node(services: AgentServices):
    llm = services.llm_service.llm

    async def rewrite_node(state: dict) -> dict:
        question = state.get("question", "")
        history = _recent_history(
            _historic_messages(state), services.settings.agent_history_turns
        )
        rewritten = question

        if history:
            prompt = REWRITE_PROMPT.format(
                history=_format_history(history), question=question
            )
            try:
                response = await llm.ainvoke([HumanMessage(content=prompt)])
                text = (getattr(response, "content", "") or "").strip()
                if text:
                    rewritten = text
            except Exception:
                rewritten = question

        await _emit(services, {"type": "step", "node": "rewrite", "detail": rewritten})
        return {"search_query": rewritten}

    return rewrite_node


def make_decompose_node(services: AgentServices):
    structured = services.llm_service.llm.with_structured_output(
        SubQuestions, method="function_calling"
    )

    async def decompose_node(state: dict) -> dict:
        question = state.get("question", "")
        try:
            result = await structured.ainvoke(
                [HumanMessage(content=DECOMPOSE_PROMPT.format(question=question))]
            )
            sub_questions = [q.strip() for q in result.questions if q and q.strip()][:4]
        except Exception as e:
            print(f"WARNING: decompose failed ({e}); using original question")
            sub_questions = []
        if not sub_questions:
            sub_questions = [question]

        await _emit(
            services, {"type": "step", "node": "decompose", "detail": sub_questions}
        )
        return {"sub_questions": sub_questions}

    return decompose_node


def make_agent_retrieve_node(services: AgentServices):
    settings = services.settings
    max_calls = max(1, getattr(settings, "agent_max_tool_calls", 3))
    tools = make_tools(services.store, services.embed_service, settings)
    llm_with_tools = services.llm_service.llm.bind_tools(tools)

    async def _search(query: str, top_k: int | None = None) -> list:
        hits = await asyncio.to_thread(
            search_documents,
            services.store,
            services.embed_service,
            settings,
            query,
            top_k,
        )
        if services.reranker_service and hits:
            hits = await asyncio.to_thread(
                services.reranker_service.compress_documents, hits, query
            )
        return hits

    async def agent_retrieve_node(state: dict) -> dict:
        question = state.get("question", "")
        search_query = state.get("search_query") or question
        sub_questions = state.get("sub_questions") or []
        docs = list(state.get("documents") or [])

        # Corrective retries widen the retrieval window: a rephrased query alone
        # cannot recover a chunk that ranks below the initial top-k.
        retry_top_k = max(
            settings.hybrid_fusion_top_k,
            getattr(settings, "agent_retry_fusion_top_k", 20),
        )
        wide_top_k = retry_top_k if state.get("iterations", 0) > 0 else None

        known = list(dict.fromkeys(d.metadata.get("doc_title", "未知") for d in docs))
        task = RETRIEVE_TASK.format(
            question=question,
            search_query=search_query,
            sub_questions="\n".join(f"- {q}" for q in sub_questions) or "（无）",
            known_titles="、".join(known) or "（无）",
        )
        messages = [SystemMessage(content=AGENT_RETRIEVE_PROMPT), HumanMessage(content=task)]

        calls_made = 0
        while calls_made < max_calls:
            ai = await llm_with_tools.ainvoke(messages)
            tool_calls = list(getattr(ai, "tool_calls", None) or [])
            if not tool_calls:
                break
            messages.append(ai)

            stop_early = False
            for call in tool_calls:
                calls_made += 1
                name = call.get("name", "")
                args = call.get("args") or {}
                call_id = call.get("id") or f"call_{calls_made}"

                if name == "list_documents":
                    titles = await asyncio.to_thread(services.store.list_documents)
                    content = "、".join(titles) if titles else "知识库为空"
                    await _emit(services, {"type": "step", "node": "tool", "detail": "list_documents()"})
                elif name == "search_knowledge_base":
                    query = (args.get("query") or search_query).strip()
                    hits = await _search(query, wide_top_k)
                    known_ids = {d.metadata.get("parent_doc_id") for d in docs}
                    docs = _dedupe_documents(docs, hits)
                    added = len({d.metadata.get("parent_doc_id") for d in docs} - known_ids)
                    content = format_docs(hits)
                    detail = f"search_knowledge_base({query})"
                    if wide_top_k:
                        detail += f" [重试 top_k={wide_top_k}]"
                    await _emit(
                        services,
                        {"type": "step", "node": "tool", "detail": detail},
                    )
                    if hits and added == 0:
                        # Search returned only already-seen chunks; more of the
                        # same won't help, so stop this retrieval round.
                        stop_early = True
                        await _emit(
                            services,
                            {"type": "step", "node": "tool", "detail": "无新增文档，提前结束检索"},
                        )
                else:
                    content = f"未知工具: {name}"

                messages.append(ToolMessage(content=content, tool_call_id=call_id))
                if stop_early or calls_made >= max_calls:
                    break

            if stop_early:
                break

        if calls_made == 0:
            # Model did not use tools; fall back to a deterministic retrieval.
            docs = _dedupe_documents(docs, await _search(search_query, wide_top_k))

        iterations = state.get("iterations", 0) + 1
        titles = list(dict.fromkeys(d.metadata.get("doc_title", "未知") for d in docs))
        await _emit(services, {"type": "step", "node": "retrieve", "detail": titles})
        return {"documents": docs, "iterations": iterations}

    return agent_retrieve_node


def make_parallel_retrieve_node(services: AgentServices):
    """Retrieve every sub-question concurrently (multihop path).

    The tool loop exists to let the model choose queries; when the question
    has already been decomposed there is nothing left to choose, so the
    sub-questions are searched in parallel instead of one after another.
    """
    settings = services.settings

    async def _rerank(hits: list, query: str) -> list:
        if services.reranker_service and hits:
            return await asyncio.to_thread(
                services.reranker_service.compress_documents, hits, query
            )
        return hits

    async def parallel_retrieve_node(state: dict) -> dict:
        question = state.get("question", "")
        sub_questions = [q for q in (state.get("sub_questions") or []) if q and q.strip()]
        if not sub_questions:
            sub_questions = [question]

        docs = list(state.get("documents") or [])
        retry_top_k = max(
            settings.hybrid_fusion_top_k,
            getattr(settings, "agent_retry_fusion_top_k", 50),
        )
        wide_top_k = retry_top_k if state.get("iterations", 0) > 0 else None

        # Sub-question embeddings are computed in one batched call, then the
        # per-query searches / reranks run as concurrent tasks.
        batch = await asyncio.to_thread(
            search_documents_batch,
            services.store,
            services.embed_service,
            settings,
            sub_questions,
            wide_top_k,
        )
        reranked = await asyncio.gather(
            *(_rerank(hits, sub_q) for hits, sub_q in zip(batch, sub_questions)),
            return_exceptions=True,
        )

        for sub_q, result in zip(sub_questions, reranked):
            if isinstance(result, Exception):
                print(f"WARNING: parallel search failed for {sub_q!r}: {result}")
                continue
            detail = f"search_knowledge_base({sub_q}) [并行]"
            if wide_top_k:
                detail += f" [重试 top_k={wide_top_k}]"
            await _emit(services, {"type": "step", "node": "tool", "detail": detail})
            docs = _dedupe_documents(docs, result)

        iterations = state.get("iterations", 0) + 1
        titles = list(dict.fromkeys(d.metadata.get("doc_title", "未知") for d in docs))
        await _emit(services, {"type": "step", "node": "retrieve", "detail": titles})
        return {"documents": docs, "iterations": iterations}

    return parallel_retrieve_node


def make_grade_node(services: AgentServices):
    llm_service = services.llm_service
    max_docs = getattr(services.settings, "context_max_docs", 5)
    structured = llm_service.llm.with_structured_output(
        GradeDecision, method="function_calling"
    )

    async def grade_node(state: dict) -> dict:
        docs = limit_documents(state.get("documents", []) or [], max_docs)
        context = llm_service._format_context(docs)
        prompt = GRADE_PROMPT.format(
            question=state.get("question", ""), context=context
        )
        try:
            decision = await structured.ainvoke([HumanMessage(content=prompt)])
            grade = "sufficient" if decision.sufficient else "insufficient"
            reason = decision.reason
        except Exception as e:
            print(f"WARNING: grade failed ({e}); assuming sufficient")
            grade = "sufficient"
            reason = ""

        await _emit(services, {"type": "step", "node": "grade", "detail": grade})
        return {"grade": grade, "grade_reason": reason}

    return grade_node


def make_refine_node(services: AgentServices):
    llm = services.llm_service.llm

    async def refine_node(state: dict) -> dict:
        previous = state.get("search_query") or state.get("question", "")
        prompt = REFINE_PROMPT.format(
            query=previous,
            reason=state.get("grade_reason", ""),
            history=_format_history(
                _recent_history(
                    _historic_messages(state),
                    services.settings.agent_history_turns,
                )
            ),
            question=state.get("question", ""),
        )
        refined = previous
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            text = (getattr(response, "content", "") or "").strip()
            if text:
                refined = text
        except Exception:
            refined = previous

        await _emit(services, {"type": "step", "node": "refine", "detail": refined})
        return {"search_query": refined, "sub_questions": []}

    return refine_node


def make_generate_node(services: AgentServices):
    llm_service = services.llm_service
    max_docs = getattr(services.settings, "context_max_docs", 5)

    async def generate_node(state: dict) -> dict:
        docs = limit_documents(state.get("documents", []) or [], max_docs)
        route = state.get("route", "knowledge")

        mapping = {}
        fallback_sources = []
        if route == "direct":
            chain = llm_service.build_direct_chain()
            inputs = {"question": state.get("question", "")}
        else:
            context, mapping = llm_service.build_numbered_context(docs)
            fallback_sources = llm_service._extract_sources(docs)
            chain = llm_service.build_answer_chain()
            inputs = {"context": context, "question": state.get("question", "")}

        answer = ""
        async for chunk in chain.astream(inputs):
            if chunk:
                answer += chunk
                await _emit(services, {"type": "content", "content": chunk})

        if route != "direct" and state.get("grade") == "insufficient":
            answer += INSUFFICIENT_NOTE
            await _emit(services, {"type": "content", "content": INSUFFICIENT_NOTE})

        sources = llm_service.extract_cited_sources(answer, mapping)
        if route != "direct" and not sources:
            # Model did not emit citation markers; fall back to retrieved docs.
            sources = fallback_sources

        await _emit(services, {"type": "sources", "sources": sources})
        return {
            "answer": answer,
            "sources": sources,
            "messages": [AIMessage(content=answer)],
        }

    return generate_node
