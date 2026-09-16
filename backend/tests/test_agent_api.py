"""Single-seam tests for the agentic RAG API (P2).

Everything is exercised through the HTTP API (the highest seam) with fake
services injected via ``app.state``. No Milvus, no models, no network.
"""

import json

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import AIMessage, HumanMessage

from app.api import query as query_module
from app.config import get_settings


class FakeAnswerChain:
    def __init__(self, text: str):
        self.text = text

    async def astream(self, inputs, **kwargs):
        for i in range(0, len(self.text), 4):
            yield self.text[i : i + 4]

    async def ainvoke(self, inputs, **kwargs):
        return self.text


class _Structured:
    def __init__(self, schema, llm, method):
        self.schema = schema
        self.llm = llm
        self.method = method

    async def ainvoke(self, messages, **kwargs):
        # Mimic DeepSeek: only function_calling structured output is accepted.
        if self.method != "function_calling":
            raise RuntimeError("This response_format type is unavailable now")
        name = getattr(self.schema, "__name__", "")
        if name == "GradeDecision":
            sufficient = self.llm.grades.pop(0) if self.llm.grades else True
            return self.schema(sufficient=sufficient, reason="stub reason")
        if name == "SubQuestions":
            return self.schema(questions=self.llm.sub_questions or ["默认子问题"])
        return self.schema(route=self.llm.route)


class FakeToolLLM:
    def __init__(self, llm):
        self.llm = llm

    async def ainvoke(self, messages, **kwargs):
        self.llm.last_retrieve_task = "\n".join(
            str(getattr(m, "content", "")) for m in messages
        )
        if self.llm.tool_responses:
            return self.llm.tool_responses.pop(0)
        return AIMessage(content="done")


class FakeLLM:
    def __init__(self, route="knowledge", rewritten="改写后的查询", grades=None,
                 sub_questions=None, tool_responses=None):
        self.route = route
        self.rewritten = rewritten
        self.grades = list(grades) if grades else []
        self.sub_questions = list(sub_questions) if sub_questions else []
        self.tool_responses = list(tool_responses) if tool_responses else []
        self.last_prompt = None
        self.last_retrieve_task = None

    def with_structured_output(self, schema, **kwargs):
        return _Structured(schema, self, kwargs.get("method"))

    def bind_tools(self, tools, **kwargs):
        return FakeToolLLM(self)

    async def ainvoke(self, messages, **kwargs):
        self.last_prompt = "\n".join(
            str(getattr(m, "content", "")) for m in messages
        )
        return AIMessage(content=self.rewritten)

    def invoke(self, messages, **kwargs):
        self.last_prompt = "\n".join(
            str(getattr(m, "content", "")) for m in messages
        )
        return AIMessage(content=self.rewritten)


class FakeLLMService:
    def __init__(self, route="knowledge", rewritten="改写后的查询", answer="这是答案",
                 grades=None, sub_questions=None, tool_responses=None):
        self.llm = FakeLLM(route, rewritten, grades, sub_questions, tool_responses)
        self.answer = answer

    def build_answer_chain(self):
        return FakeAnswerChain(self.answer)

    def build_direct_chain(self):
        return FakeAnswerChain(self.answer)

    def _format_context(self, docs):
        return "\n".join(d.metadata.get("parent_text", "") for d in docs)

    def build_numbered_context(self, docs):
        self.context_docs = list(docs)
        titles = self._extract_sources(docs)
        return "CTX", {i + 1: t for i, t in enumerate(titles)}

    @staticmethod
    def extract_cited_sources(answer, mapping):
        import re as _re

        cited = []
        for n in _re.findall(r"\[(\d+)\]", answer or ""):
            title = mapping.get(int(n))
            if title and title not in cited:
                cited.append(title)
        return cited

    def _extract_sources(self, docs):
        return list(dict.fromkeys(d.metadata.get("doc_title", "未知") for d in docs))[:5]


class FakeEmbed:
    def __init__(self):
        self.last_query = None
        self.last_batch = None

    def encode(self, texts, max_length=512):
        self.last_batch = list(texts)
        return [{"dense": [0.0] * 1024, "sparse": {1: 1.0}} for _ in texts]

    def encode_query(self, query):
        self.last_query = query
        return {"dense": [0.0] * 1024, "sparse": {1: 1.0}}


class FakeReranker:
    def __init__(self):
        self.calls = 0

    def compress_documents(self, docs, query, callbacks=None):
        self.calls += 1
        return docs


class FakeStore:
    def __init__(self):
        self.hybrid_calls = 0
        self.hybrid_top_ks = []
        self.hits = [
            {
                "id": "1",
                "text": "子块",
                "parent_text": "父块内容",
                "doc_title": "docA.txt",
                "parent_doc_id": "p1",
                "score": 0.9,
            }
        ]

    def hybrid_search(self, **kwargs):
        self.hybrid_calls += 1
        self.hybrid_top_ks.append(kwargs.get("fusion_top_k"))
        return list(self.hits)

    def list_documents(self):
        return ["docA.txt", "docB.txt"]


@pytest.fixture
def fakes():
    return {
        "store": FakeStore(),
        "embed": FakeEmbed(),
        "reranker": FakeReranker(),
        "llm": FakeLLMService(),
    }


def make_app(fakes):
    app = FastAPI()
    app.include_router(query_module.router)
    app.state.milvus_store = fakes["store"]
    app.state.embedding_service = fakes["embed"]
    app.state.reranker_service = fakes["reranker"]
    app.state.llm_service = fakes["llm"]
    return app


def parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                events.append({"type": "done"})
            else:
                events.append(json.loads(data))
    return events


def step_nodes(events):
    return [e["node"] for e in events if e["type"] == "step"]


async def post(app, path, payload):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(path, json=payload)


def tool_call(name, query=None, call_id="c1"):
    args = {"query": query} if query is not None else {}
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


# ---------------------------------------------------------------------------
# routing / basic flow
# ---------------------------------------------------------------------------


async def test_direct_route_skips_retrieval(fakes):
    fakes["llm"].llm.route = "direct"
    resp = await post(make_app(fakes), "/api/query", {"question": "你好"})

    assert resp.status_code == 200
    assert resp.json()["answer"] == "这是答案"
    assert fakes["store"].hybrid_calls == 0


async def test_query_returns_503_when_services_missing(fakes):
    app = make_app(fakes)
    app.state.llm_service = None

    sync_resp = await post(app, "/api/query", {"question": "问题"})
    stream_resp = await post(app, "/api/query/stream", {"question": "问题"})

    assert sync_resp.status_code == 503
    assert stream_resp.status_code == 503


async def test_knowledge_route_rewrites_and_retrieves(fakes):
    fakes["llm"].llm.route = "knowledge"
    fakes["llm"].llm.rewritten = "省外出差报销标准"
    payload = {
        "question": "那省外呢？",
        "history": [
            {"role": "user", "content": "差旅报销标准是多少？"},
            {"role": "assistant", "content": "市内交通..."},
        ],
    }
    resp = await post(make_app(fakes), "/api/query", payload)

    assert resp.status_code == 200
    assert fakes["store"].hybrid_calls == 1
    assert fakes["embed"].last_query == "省外出差报销标准"
    assert resp.json()["sources"] == ["docA.txt"]


async def test_no_history_uses_original_question(fakes):
    fakes["llm"].llm.route = "knowledge"
    fakes["llm"].llm.rewritten = "REWRITTEN"
    resp = await post(make_app(fakes), "/api/query", {"question": "原始问题"})

    assert resp.status_code == 200
    assert fakes["embed"].last_query == "原始问题"


async def test_invalid_route_falls_back_to_knowledge(fakes):
    fakes["llm"].llm.route = "bogus"
    resp = await post(make_app(fakes), "/api/query", {"question": "某制度"})

    assert resp.status_code == 200
    assert fakes["store"].hybrid_calls == 1


async def test_stream_emits_step_then_content_then_sources(fakes):
    fakes["llm"].llm.route = "knowledge"
    fakes["llm"].llm.rewritten = "独立查询"
    fakes["llm"].answer = "分块回答内容"
    resp = await post(make_app(fakes), "/api/query/stream", {"question": "问题"})

    events = parse_sse(resp.text)
    types = [e["type"] for e in events]
    assert types[0] == "status"
    assert types[-1] == "done"
    assert step_nodes(events) == ["route", "rewrite", "retrieve", "grade"]

    content = "".join(e["content"] for e in events if e["type"] == "content")
    assert content == "分块回答内容"
    assert events[-2]["type"] == "sources"
    assert events[-2]["sources"] == ["docA.txt"]


async def test_stream_emits_on_repeated_requests_with_cached_graph(fakes):
    app = make_app(fakes)
    first = parse_sse(
        (await post(app, "/api/query/stream", {"question": "第一问", "session_id": "s1"})).text
    )
    second = parse_sse(
        (await post(app, "/api/query/stream", {"question": "第二问", "session_id": "s1"})).text
    )

    assert step_nodes(first) == ["route", "rewrite", "retrieve", "grade"]
    assert step_nodes(second) == ["route", "rewrite", "retrieve", "grade"]
    assert second[-1]["type"] == "done"
    assert any(e["type"] == "content" for e in second)


# ---------------------------------------------------------------------------
# P1 corrective loop
# ---------------------------------------------------------------------------


async def test_p1_sufficient_grades_once(fakes):
    fakes["llm"].llm.route = "knowledge"
    fakes["llm"].llm.grades = [True]
    resp = await post(make_app(fakes), "/api/query/stream", {"question": "问题"})

    events = parse_sse(resp.text)
    assert step_nodes(events) == ["route", "rewrite", "retrieve", "grade"]
    assert fakes["store"].hybrid_calls == 1


async def test_p1_insufficient_then_sufficient_refines(fakes):
    fakes["llm"].llm.route = "knowledge"
    fakes["llm"].llm.rewritten = "纠错后的查询"
    fakes["llm"].llm.grades = [False, True]
    resp = await post(make_app(fakes), "/api/query/stream", {"question": "原始问题"})

    events = parse_sse(resp.text)
    assert step_nodes(events) == [
        "route", "rewrite", "retrieve", "grade",
        "refine", "retrieve", "grade",
    ]
    assert fakes["store"].hybrid_calls == 2
    assert fakes["embed"].last_query == "纠错后的查询"
    assert "尽力回答" not in resp.text


async def test_p1_exhausted_answers_with_note(fakes):
    fakes["llm"].llm.route = "knowledge"
    fakes["llm"].llm.grades = [False, False]
    resp = await post(make_app(fakes), "/api/query/stream", {"question": "原始问题"})

    events = parse_sse(resp.text)
    assert step_nodes(events) == [
        "route", "rewrite", "retrieve", "grade",
        "refine", "retrieve", "grade",
    ]
    assert fakes["store"].hybrid_calls == 2
    content = "".join(e["content"] for e in events if e["type"] == "content")
    assert "尽力回答" in content


async def test_corrective_retry_widens_retrieval_window(fakes):
    fakes["llm"].llm.grades = [False, False]
    fakes["llm"].llm.tool_responses = [
        tool_call("search_knowledge_base", "q1", "c1"),
        tool_call("search_knowledge_base", "q2", "c2"),
        tool_call("search_knowledge_base", "q3", "c3"),
    ]
    resp = await post(make_app(fakes), "/api/query/stream", {"question": "问题"})

    events = parse_sse(resp.text)
    top_ks = fakes["store"].hybrid_top_ks
    assert len(top_ks) == 3
    assert top_ks[0] is not None and top_ks[1] is not None
    assert top_ks[0] == top_ks[1] < top_ks[2]
    tools = [str(e.get("detail")) for e in events if e.get("node") == "tool"]
    assert any("重试 top_k=" in d for d in tools)


async def test_p1_grade_failure_degrades_to_sufficient(fakes):
    fakes["llm"].llm.route = "knowledge"
    fakes["llm"].llm.grades = ["not-a-bool"]
    resp = await post(make_app(fakes), "/api/query/stream", {"question": "问题"})

    events = parse_sse(resp.text)
    assert step_nodes(events) == ["route", "rewrite", "retrieve", "grade"]
    assert fakes["store"].hybrid_calls == 1


# ---------------------------------------------------------------------------
# P2 autonomous tool loop
# ---------------------------------------------------------------------------


async def test_tool_loop_search(fakes):
    fakes["llm"].llm.tool_responses = [tool_call("search_knowledge_base", "工具查询")]
    resp = await post(make_app(fakes), "/api/query/stream", {"question": "问题"})

    events = parse_sse(resp.text)
    tools = [e for e in events if e["type"] == "step" and e["node"] == "tool"]
    assert tools and "工具查询" in tools[0]["detail"]
    assert fakes["store"].hybrid_calls == 1
    assert fakes["embed"].last_query == "工具查询"


async def test_tool_loop_list_documents_skips_vector_search(fakes):
    fakes["llm"].llm.tool_responses = [tool_call("list_documents")]
    resp = await post(make_app(fakes), "/api/query/stream", {"question": "有哪些文档"})

    events = parse_sse(resp.text)
    tools = [e for e in events if e["type"] == "step" and e["node"] == "tool"]
    assert tools and tools[0]["detail"] == "list_documents()"
    assert fakes["store"].hybrid_calls == 0


async def test_tool_loop_falls_back_when_model_calls_no_tools(fakes):
    resp = await post(make_app(fakes), "/api/query/stream", {"question": "原始问题"})

    events = parse_sse(resp.text)
    assert not [e for e in events if e.get("node") == "tool"]
    assert fakes["store"].hybrid_calls == 1
    assert fakes["embed"].last_query == "原始问题"


async def test_tool_loop_stops_early_when_no_new_documents(fakes):
    fakes["llm"].llm.tool_responses = [
        tool_call("search_knowledge_base", "q1", "c1"),
        tool_call("search_knowledge_base", "q2", "c2"),
        tool_call("search_knowledge_base", "q3", "c3"),
    ]
    resp = await post(make_app(fakes), "/api/query/stream", {"question": "问题"})

    events = parse_sse(resp.text)
    assert fakes["store"].hybrid_calls == 2
    assert fakes["embed"].last_query == "q2"
    tools = [str(e.get("detail")) for e in events if e.get("node") == "tool"]
    assert any("无新增文档" in d for d in tools)


# ---------------------------------------------------------------------------
# P2 multihop
# ---------------------------------------------------------------------------


async def test_multihop_decomposes(fakes):
    fakes["llm"].llm.route = "multihop"
    fakes["llm"].llm.sub_questions = ["子问题A", "子问题B"]
    resp = await post(make_app(fakes), "/api/query/stream", {"question": "对比A和B"})

    events = parse_sse(resp.text)
    decomp = [e for e in events if e["type"] == "step" and e["node"] == "decompose"]
    assert decomp and decomp[0]["detail"] == ["子问题A", "子问题B"]

    tools = [str(e.get("detail")) for e in events if e.get("node") == "tool"]
    assert len(tools) == 2
    assert all("子问题" in detail and "[并行]" in detail for detail in tools)


# ---------------------------------------------------------------------------
# P2 server-side memory
# ---------------------------------------------------------------------------


async def test_memory_across_turns(fakes):
    app = make_app(fakes)
    await post(app, "/api/query", {"question": "第一问", "session_id": "s1"})
    fakes["llm"].llm.last_prompt = None

    await post(app, "/api/query", {"question": "那它呢", "session_id": "s1"})
    assert "第一问" in (fakes["llm"].llm.last_prompt or "")


async def test_memory_isolated_between_sessions(fakes):
    app = make_app(fakes)
    await post(app, "/api/query", {"question": "第一问", "session_id": "s1"})
    fakes["llm"].llm.last_prompt = None

    await post(app, "/api/query", {"question": "另一个问题", "session_id": "s2"})
    assert not fakes["llm"].llm.last_prompt


async def test_history_seeded_on_new_thread(fakes):
    payload = {
        "question": "那省外呢？",
        "session_id": "s3",
        "history": [
            {"role": "user", "content": "差旅报销标准是多少？"},
            {"role": "assistant", "content": "市内交通..."},
        ],
    }
    await post(make_app(fakes), "/api/query", payload)
    assert "差旅报销标准是多少" in (fakes["llm"].llm.last_prompt or "")


# ---------------------------------------------------------------------------
# P0 graph regression (used by evaluation)
# ---------------------------------------------------------------------------


async def test_multihop_retrieves_sub_questions_in_parallel(fakes):
    fakes["llm"].llm.route = "multihop"
    fakes["llm"].llm.sub_questions = ["子问题一", "子问题二", "子问题三"]
    fakes["store"].hits = two_doc_hits()
    resp = await post(make_app(fakes), "/api/query/stream", {"question": "多跳问题"})

    events = parse_sse(resp.text)
    assert step_nodes(events) == [
        "route", "decompose", "tool", "tool", "tool", "retrieve", "grade",
    ]
    tools = [str(e.get("detail")) for e in events if e.get("node") == "tool"]
    assert len(tools) == 3
    assert all("[并行]" in detail for detail in tools)
    assert fakes["store"].hybrid_calls == 3

    retrieve = [e for e in events if e.get("node") == "retrieve"][0]
    assert set(retrieve["detail"]) == {"docA.txt", "docB.txt"}


async def test_p0_graph_has_no_grade_steps(fakes):
    from app.agent.graph import build_agent_graph
    from app.agent.nodes import AgentServices

    events = []

    async def emit(payload):
        events.append(payload)

    fakes["llm"].llm.route = "knowledge"
    graph = build_agent_graph(
        AgentServices(
            settings=get_settings(),
            store=fakes["store"],
            embed_service=fakes["embed"],
            reranker_service=fakes["reranker"],
            llm_service=fakes["llm"],
            emit=emit,
        ),
        enable_correction=False,
    )
    await graph.ainvoke(
        {
            "question": "问题",
            "messages": [HumanMessage(content="问题")],
            "search_query": "",
            "sub_questions": [],
            "documents": [],
            "iterations": 0,
            "grade": "",
            "grade_reason": "",
            "answer": "",
            "sources": [],
        }
    )

    assert step_nodes(events) == ["route", "rewrite", "retrieve"]
    assert fakes["store"].hybrid_calls == 1


# ---------------------------------------------------------------------------
# source attribution
# ---------------------------------------------------------------------------


def two_doc_hits():
    return [
        {"id": "1", "text": "a", "parent_text": "A", "doc_title": "docA.txt",
         "parent_doc_id": "p1", "score": 0.9},
        {"id": "2", "text": "b", "parent_text": "B", "doc_title": "docB.txt",
         "parent_doc_id": "p2", "score": 0.5},
    ]


def test_recent_history_keeps_only_last_turns():
    from app.agent.nodes import _recent_history

    messages = [HumanMessage(content=str(i)) for i in range(10)]

    assert [m.content for m in _recent_history(messages, 2)] == ["6", "7", "8", "9"]
    assert len(_recent_history(messages, 0)) == 10


def test_limit_documents_keeps_first_distinct_titles():
    from langchain_core.documents import Document

    from app.agent.tools import limit_documents

    docs = [
        Document(page_content="a1", metadata={"doc_title": "A"}),
        Document(page_content="a2", metadata={"doc_title": "A"}),
        Document(page_content="b1", metadata={"doc_title": "B"}),
        Document(page_content="c1", metadata={"doc_title": "C"}),
    ]

    kept = limit_documents(docs, 2)

    assert [d.metadata["doc_title"] for d in kept] == ["A", "A", "B"]
    assert len(limit_documents(docs, 0)) == 4


async def test_generate_context_is_limited_to_max_docs(fakes):
    fakes["store"].hits = [
        {
            "id": str(i),
            "text": f"t{i}",
            "parent_text": f"p{i}",
            "doc_title": f"doc{i}.txt",
            "parent_doc_id": f"pid{i}",
            "score": 1.0 - i / 10,
        }
        for i in range(8)
    ]
    resp = await post(make_app(fakes), "/api/query", {"question": "问题"})

    assert resp.status_code == 200
    docs = fakes["llm"].context_docs
    assert docs is not None
    assert len({d.metadata["doc_title"] for d in docs}) <= 5


async def test_sources_only_include_cited_documents(fakes):
    fakes["store"].hits = two_doc_hits()
    fakes["llm"].answer = "只用了A的内容 [1]"
    resp = await post(make_app(fakes), "/api/query/stream", {"question": "A的问题"})

    events = parse_sse(resp.text)
    sources = [e for e in events if e["type"] == "sources"][0]["sources"]
    assert sources == ["docA.txt"]


async def test_sources_fall_back_when_no_citations(fakes):
    fakes["store"].hits = two_doc_hits()
    fakes["llm"].answer = "没有标注的答案"
    resp = await post(make_app(fakes), "/api/query/stream", {"question": "A的问题"})

    events = parse_sse(resp.text)
    sources = [e for e in events if e["type"] == "sources"][0]["sources"]
    assert sources == ["docA.txt", "docB.txt"]
