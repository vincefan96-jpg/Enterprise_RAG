"""Rubric-based answer judging using the test set's own grading criteria.

The general_50 test set ships a 评判要点 (rubric) per question and states
that answers should be graded with 标准答案 + 评判要点. Ragas' own
AnswerCorrectness proved unreliable on short Chinese answers (it scored
verbatim-correct answers at 0.5-0.7), so answer quality is judged here
instead, with ``keyword_hit`` as a free deterministic cross-check.
"""

from .metrics import keyword_hit
from .models import AnswerVerdict

RUBRIC_JUDGE_PROMPT = """你是 RAG 评测的判卷员。请根据【标准答案】和【评判要点】判断【模型回答】是否正确。

判定标准：
- correct：核心结论与标准答案一致，且满足评判要点要求的关键信息（措辞不同不算错）
- partial：部分正确，缺少评判要点强调的关键信息，或存在次要错误
- incorrect：核心结论错误、答非所问，或与标准答案矛盾

问题：{question}
标准答案：{reference}
评判要点：{rubric}
模型回答：{answer}"""


def judge_answers(llm, items: list[dict], verbose: bool = True) -> list[dict]:
    """Judge answers against their rubric; one LLM call per item.

    Each item needs: id, question, answer, reference, rubric, keywords.
    """
    judge = llm.with_structured_output(AnswerVerdict, method="function_calling")

    rows = []
    for item in items:
        verdict = None
        reason = ""
        try:
            decision = judge.invoke(
                RUBRIC_JUDGE_PROMPT.format(
                    question=item["question"],
                    reference=item["reference"] or "（无标准答案）",
                    rubric=item["rubric"] or "（无评判要点）",
                    answer=item["answer"] or "（空回答）",
                )
            )
            verdict = decision.verdict
            reason = decision.reason or ""
        except Exception as e:
            print(f"    Rubric judge failed ({item['id']}): {e}")

        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": item["answer"],
                "reference": item["reference"],
                "verdict": verdict,
                "reason": reason,
                "keyword_hit": keyword_hit(item["answer"], item["keywords"]),
            }
        )

        if verbose:
            print(f"    [{item['id']}] {verdict}")

    return rows
