from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """State carried through the agent graph.

    ``messages`` is the conversation memory (persisted by the checkpointer via
    ``add_messages``); every other channel is reset per turn by the API input.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    route: str
    search_query: str
    sub_questions: list[str]
    documents: list[Document]
    grade: str
    grade_reason: str
    iterations: int
    answer: str
    sources: list[str]
