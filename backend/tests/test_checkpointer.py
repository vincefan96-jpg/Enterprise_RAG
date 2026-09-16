from pathlib import Path
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.agent.checkpointer import open_checkpointer, resolve_checkpointer_path

pytest.importorskip("langgraph.checkpoint.sqlite")


class _Settings:
    checkpointer_backend = "sqlite"

    def __init__(self, path):
        self.checkpointer_path = path


class _State(TypedDict):
    messages: Annotated[list, add_messages]


def _reply(state):
    return {"messages": [AIMessage(content="pong")]}


def _build_graph(checkpointer):
    builder = StateGraph(_State)
    builder.add_node("reply", _reply)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    return builder.compile(checkpointer=checkpointer)


def _config(thread_id="t1"):
    return {"configurable": {"thread_id": thread_id}}


async def test_sqlite_checkpointer_persists_across_reopen(tmp_path):
    settings = _Settings(str(tmp_path / "checkpoints.sqlite"))
    config = _config()

    async with open_checkpointer(settings) as checkpointer:
        graph = _build_graph(checkpointer)
        await graph.ainvoke(
            {"messages": [HumanMessage(content="ping")]}, config
        )

    assert Path(settings.checkpointer_path).exists()

    async with open_checkpointer(settings) as checkpointer:
        graph = _build_graph(checkpointer)
        snapshot = await graph.aget_state(config)
        assert len(snapshot.values["messages"]) == 2


async def test_threads_do_not_share_memory(tmp_path):
    settings = _Settings(str(tmp_path / "checkpoints.sqlite"))

    async with open_checkpointer(settings) as checkpointer:
        graph = _build_graph(checkpointer)
        await graph.ainvoke({"messages": [HumanMessage(content="a")]}, _config("t1"))
        await graph.ainvoke({"messages": [HumanMessage(content="b")]}, _config("t2"))

        t1 = await graph.aget_state(_config("t1"))
        t2 = await graph.aget_state(_config("t2"))
        assert len(t1.values["messages"]) == 2
        assert len(t2.values["messages"]) == 2
        assert t1.values["messages"][0].content == "a"


async def test_memory_backend_does_not_touch_disk(tmp_path):
    settings = _Settings(str(tmp_path / "unused.sqlite"))
    settings.checkpointer_backend = "memory"

    async with open_checkpointer(settings) as checkpointer:
        assert type(checkpointer).__name__ == "InMemorySaver"
        graph = _build_graph(checkpointer)
        await graph.ainvoke({"messages": [HumanMessage(content="ping")]}, _config())

    assert not Path(settings.checkpointer_path).exists()


def test_agent_checkpointer_env_alias(monkeypatch):
    from app.config import Settings

    monkeypatch.setenv("AGENT_CHECKPOINTER_BACKEND", "memory")
    monkeypatch.setenv("AGENT_CHECKPOINTER_PATH", "custom/path.sqlite")

    settings = Settings()

    assert settings.checkpointer_backend == "memory"
    assert settings.checkpointer_path == "custom/path.sqlite"


def test_resolve_checkpointer_path_is_repo_relative(tmp_path):
    resolved = resolve_checkpointer_path("backend/data/checkpoints.sqlite")

    assert resolved.is_absolute()
    assert resolved.parts[-3:] == ("backend", "data", "checkpoints.sqlite")
    assert resolve_checkpointer_path(str(tmp_path)) == tmp_path
