"""Ragas-based evaluation runner.

Metrics:
  Retrieval  — Hit@k / recall / MRR by document title, computed locally
               (no LLM judge). Replaces the previous whole-document
               ContextPrecision/ContextRecall, which treated every parent
               chunk of the ground-truth document as relevant and therefore
               scored documents, not the retrieved passages.
  Timing     — real per-stage wall clock derived from the LangGraph update
               stream (pipeline vs generation), replacing the placeholder
               ``elapsed / 2`` split.
  Generation — Faithfulness, AnswerRelevancy, AnswerCorrectness (Ragas,
               judged by the LLM). AnswerCorrectness is included only when
               every question has a ground-truth answer.
"""

import asyncio
import time

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, AnswerCorrectness

from langchain_core.messages import HumanMessage

from .models import RefusalDecision, TestQuery
from .adapters import BGEM3Embeddings
from .answer_judge import judge_answers
from .metrics import (
    answer_metrics,
    average_metrics,
    evidence_hit,
    refusal_metrics,
    title_hit_metrics,
)

REFUSAL_JUDGE_PROMPT = """判断下面这个 RAG 系统的回答是否属于"拒答"：即明确表示知识库中没有相关信息、无法从现有资料回答，或引导用户查阅其他资料。

注意：
- 只要给出了具体答案内容（即使可能不完整或错误），就不算拒答。
- 说明"资料未收录 / 未定义该指标 / 无法从现有资料确定"即算拒答，即使附带了解释或建议。

问题：{question}
回答：{answer}"""


class RagasEvaluationRunner:
    def __init__(self, store, embed_service, reranker, llm_service, settings,
                 agent_mode: str = "p1"):
        self.store = store
        self.embed_service = embed_service
        self.reranker = reranker
        self.llm_service = llm_service
        self.settings = settings
        self.agent_mode = agent_mode

        # Wrap LLM and embeddings for Ragas internals (judge LLM + embedding sim)
        self._ragas_llm = LangchainLLMWrapper(llm_service.llm)
        self._ragas_emb = LangchainEmbeddingsWrapper(
            BGEM3Embeddings(embed_service)
        )

    # ------------------------------------------------------------------
    # running the agent graph
    # ------------------------------------------------------------------

    def _run_graph(self, question: str):
        """Run the agent graph; return (documents, answer, node timings in ms).

        Node durations come from ``astream(stream_mode="updates")``, so the
        retrieval/generation split is measured, not estimated.
        """
        from app.agent.graph import build_agent_graph
        from app.agent.nodes import AgentServices

        graph = build_agent_graph(
            AgentServices(
                settings=self.settings,
                store=self.store,
                embed_service=self.embed_service,
                reranker_service=self.reranker,
                llm_service=self.llm_service,
            ),
            enable_correction=(self.agent_mode == "p1"),
        )
        return asyncio.run(self._stream_graph(graph, question))

    @staticmethod
    async def _stream_graph(graph, question: str):
        state = {
            "question": question,
            "messages": [HumanMessage(content=question)],
            "search_query": "",
            "sub_questions": [],
            "documents": [],
            "iterations": 0,
            "grade": "",
            "grade_reason": "",
            "answer": "",
            "sources": [],
        }

        documents: list = []
        answer = ""
        timings: dict[str, float] = {}

        last = time.perf_counter()
        async for update in graph.astream(state, stream_mode="updates"):
            now = time.perf_counter()
            for node, partial in update.items():
                timings[node] = timings.get(node, 0.0) + (now - last) * 1000
                last = now
                if not isinstance(partial, dict):
                    continue
                if "documents" in partial:
                    documents = partial["documents"] or []
                if "answer" in partial:
                    answer = partial["answer"] or ""

        return documents, answer, timings

    @staticmethod
    def _timing_summary(timings: dict[str, float]) -> dict:
        total = sum(timings.values())
        generation = timings.get("generate", 0.0)
        return {
            "total_ms": round(total, 1),
            "pipeline_ms": round(total - generation, 1),
            "generation_ms": round(generation, 1),
            "nodes": {node: round(ms, 1) for node, ms in timings.items()},
        }

    def _evaluate_question(self, tq: TestQuery) -> dict:
        """Run one question through the agent graph and collect raw signals."""
        self.embed_service.invalidate_cache()
        start = time.perf_counter()
        try:
            docs, answer, timings = self._run_graph(tq.question)
        except Exception as e:
            print(f"    Error: {e}")
            docs, answer, timings = [], "", {}
        wall_ms = (time.perf_counter() - start) * 1000

        retrieved_titles: list[str] = []
        seen: set[str] = set()
        for doc in docs:
            title = doc.metadata.get("doc_title", "未知")
            if title not in seen:
                seen.add(title)
                retrieved_titles.append(title)

        return {
            "id": tq.id,
            "qtype": tq.qtype,
            "difficulty": tq.difficulty,
            "expect_refusal": tq.expect_refusal,
            "evidence": tq.evidence,
            "rubric": tq.rubric,
            "keywords": tq.keywords,
            "question": tq.question,
            "reference": tq.ground_truth_answer or "",
            "relevant_titles": list(tq.relevant_doc_titles),
            "retrieved_titles": retrieved_titles,
            "retrieved_contexts": [
                d.metadata.get("parent_text", d.page_content) for d in docs
            ],
            "answer": answer,
            "timing": {**self._timing_summary(timings), "wall_ms": round(wall_ms, 1)},
        }

    def _collect(self, dataset: list[TestQuery]) -> list[dict]:
        records = []
        for i, tq in enumerate(dataset):
            print(f"  Evaluating [{i + 1}/{len(dataset)}] {tq.question[:50]}...")
            records.append(self._evaluate_question(tq))
        return records

    # ------------------------------------------------------------------
    # reports
    # ------------------------------------------------------------------

    @staticmethod
    def _retrieval_report(records: list[dict]) -> dict:
        rows = []
        for r in records:
            rows.append(
                {
                    "id": r["id"],
                    "qtype": r["qtype"],
                    "difficulty": r["difficulty"],
                    "expect_refusal": r["expect_refusal"],
                    "question": r["question"],
                    "relevant_titles": r["relevant_titles"],
                    "retrieved_titles": r["retrieved_titles"],
                    **title_hit_metrics(r["retrieved_titles"], r["relevant_titles"]),
                    "evidence_hit": evidence_hit(
                        r["retrieved_contexts"], r["evidence"]
                    ),
                    "timing": r["timing"],
                }
            )

        flat = [
            {k: v for k, v in row.items() if k != "timing"} for row in rows
        ]
        averages = average_metrics(flat)
        averages.update(average_metrics([r["timing"] for r in records]))
        return {"averages": averages, "per_sample": rows}

    def _generation_report(self, records: list[dict]) -> dict:
        # Refusal items are scored by the refusal judge instead: their
        # "ground truth" is a refusal, which Ragas faithfulness/relevance
        # cannot judge meaningfully against retrieved contexts.
        scored = [r for r in records if not r["expect_refusal"]]
        if len(scored) != len(records):
            print(
                f"  (refusal items excluded from Ragas generation metrics: "
                f"{len(records) - len(scored)})"
            )

        timing_averages = average_metrics([r["timing"] for r in records])
        if not scored:
            return {
                "averages": timing_averages,
                "per_sample": [],
                "timing_per_sample": [r["timing"] for r in records],
            }

        samples = [
            SingleTurnSample(
                user_input=r["question"],
                response=r["answer"],
                retrieved_contexts=r["retrieved_contexts"],
                reference=r["reference"],
            )
            for r in scored
        ]

        metrics = [Faithfulness(), AnswerRelevancy()]
        has_reference = all(
            (r["reference"] or "").strip() for r in scored
        )
        if has_reference:
            metrics.append(AnswerCorrectness())
        else:
            print(
                "  (AnswerCorrectness skipped: some questions have no "
                "ground_truth_answer)"
            )

        result = evaluate(
            dataset=EvaluationDataset(samples),
            metrics=metrics,
            llm=self._ragas_llm,
            embeddings=self._ragas_emb,
        )
        formatted = self._format_result(result)
        averages = dict(formatted.get("averages") or {})
        averages.update(timing_averages)
        formatted["averages"] = averages
        formatted["timing_per_sample"] = [r["timing"] for r in records]
        return formatted

    def _refusal_report(self, records: list[dict]) -> dict:
        judge = self.llm_service.llm.with_structured_output(
            RefusalDecision, method="function_calling"
        )

        rows = []
        for r in records:
            refused = None
            try:
                decision = judge.invoke(
                    REFUSAL_JUDGE_PROMPT.format(
                        question=r["question"], answer=r["answer"] or "（空回答）"
                    )
                )
                refused = bool(getattr(decision, "refused", False))
            except Exception as e:
                print(f"    Refusal judge failed ({r['id']}): {e}")

            expected = bool(r["expect_refusal"])
            rows.append(
                {
                    "id": r["id"],
                    "question": r["question"],
                    "expect_refusal": expected,
                    "judge_refused": refused,
                    "correct": None if refused is None else refused == expected,
                    "answer": r["answer"],
                }
            )

        judged = [x for x in rows if x["judge_refused"] is not None]
        print(
            f"  Refusal judge: {len(judged)}/{len(rows)} judged, "
            f"{sum(1 for x in judged if x['judge_refused'])} refused"
        )
        return {"averages": refusal_metrics(rows), "per_sample": rows}

    def _answer_report(self, records: list[dict]) -> dict:
        items = [
            {
                "id": r["id"],
                "question": r["question"],
                "answer": r["answer"],
                "reference": r["reference"],
                "rubric": r["rubric"],
                "keywords": r["keywords"],
            }
            for r in records
            if not r["expect_refusal"]
        ]

        rows = judge_answers(self.llm_service.llm, items)
        judged = [r for r in rows if r["verdict"]]
        print(
            f"  Rubric judge: {len(judged)}/{len(rows)} judged, "
            f"{sum(1 for r in judged if r['verdict'] == 'correct')} correct"
        )
        return {
            "averages": answer_metrics(rows),
            "per_sample": rows,
            "judge": "rubric (标准答案 + 评判要点) + keyword_hit",
        }

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def run_retrieval(self, dataset: list[TestQuery]) -> dict:
        return {"retrieval": self._retrieval_report(self._collect(dataset))}

    def run_generation(self, dataset: list[TestQuery]) -> dict:
        records = self._collect(dataset)
        return {
            "generation": self._generation_report(records),
            "refusal": self._refusal_report(records),
            "answer": self._answer_report(records),
        }

    def run_refusal(self, dataset: list[TestQuery]) -> dict:
        return {"refusal": self._refusal_report(self._collect(dataset))}

    def run_answer(self, dataset: list[TestQuery]) -> dict:
        return {"answer": self._answer_report(self._collect(dataset))}

    def run_full(self, dataset: list[TestQuery]) -> dict:
        records = self._collect(dataset)
        return {
            "retrieval": self._retrieval_report(records),
            "generation": self._generation_report(records),
            "refusal": self._refusal_report(records),
            "answer": self._answer_report(records),
        }

    # ------------------------------------------------------------------
    # result formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_result(result) -> dict:
        """Extract averages and per-sample scores from a Ragas EvaluationResult."""
        try:
            df = result.to_pandas()
        except Exception:
            df = None

        try:
            avg = dict(result._repr_dict) if hasattr(result, "_repr_dict") else {}
        except Exception:
            avg = {}

        per_sample = (
            df.to_dict(orient="records") if df is not None else []
        )

        return {
            "averages": avg,
            "per_sample": per_sample,
        }
