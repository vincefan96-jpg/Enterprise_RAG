from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field


class RefusalDecision(BaseModel):
    """LLM-judge output for the refusal / hallucination metric."""

    refused: bool = Field(
        description="回答是否明确表示知识库没有相关信息、无法回答或资料不足"
    )
    reason: str = Field(default="", description="判断依据")


class AnswerVerdict(BaseModel):
    """LLM-judge output for the rubric-based answer metric."""

    verdict: Literal["correct", "partial", "incorrect"] = Field(
        description="对照标准答案与评判要点后的判定"
    )
    reason: str = Field(default="", description="判断依据")


@dataclass
class TestQuery:
    question: str
    relevant_doc_titles: list[str] = field(default_factory=list)
    ground_truth_answer: str = ""
    # Optional annotations carried from richer test sets; unused by the
    # P0 metrics but consumed by later ones (evidence = chunk-level hit,
    # expect_refusal = negative/refusal scoring).
    id: str = ""
    qtype: str = ""
    difficulty: str = ""
    source: str = ""
    evidence: str = ""
    keywords: str = ""
    rubric: str = ""
    expect_refusal: bool = False
