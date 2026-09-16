"""Re-score answers stored in a saved report with the rubric judge.

No agent graph / Milvus / embedding models are needed: the answers already
exist in the report, only the judge LLM is called.

Usage:
  python -m ragas_eval.rescore_answers \
      --report ragas_eval/reports/2026-09-15_general50_p1_full.json \
      --testset ragas_eval/testsets/general_50/testset.json \
      [--output rescored.json] [--limit N]
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.tokenizer_patch import apply as _apply_tokenizer_patch

_apply_tokenizer_patch()

from app.config import get_settings
from .answer_judge import judge_answers
from .metrics import answer_metrics


def _extract_answers(report: dict) -> list[dict]:
    """Answers from a report, preferring the section that covers all items."""
    refusal = (report.get("refusal") or {}).get("per_sample")
    if refusal:
        return [
            {
                "id": r["id"],
                "question": r["question"],
                "answer": r["answer"],
                "expect_refusal": bool(r["expect_refusal"]),
            }
            for r in refusal
        ]

    generation = (report.get("generation") or {}).get("per_sample") or []
    return [
        {
            "id": r.get("id", ""),
            "question": r["user_input"],
            "answer": r["response"],
            "expect_refusal": False,
        }
        for r in generation
    ]


def main():
    parser = argparse.ArgumentParser(description="Offline rubric re-scoring")
    parser.add_argument("--report", required=True)
    parser.add_argument("--testset", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from .generator import SyntheticTestSetGenerator

    report = json.load(open(args.report, encoding="utf-8"))
    dataset = {q.id: q for q in SyntheticTestSetGenerator.load(args.testset)}

    answers = _extract_answers(report)
    if args.limit:
        answers = answers[: args.limit]

    items = []
    for a in answers:
        if a["expect_refusal"]:
            continue
        tq = dataset.get(a["id"])
        if tq is None:
            continue
        items.append(
            {
                "id": a["id"],
                "question": a["question"],
                "answer": a["answer"],
                "reference": tq.ground_truth_answer,
                "rubric": tq.rubric,
                "keywords": tq.keywords,
            }
        )

    settings = get_settings()
    from app.services.llm_service import LLMService

    llm = LLMService(settings)
    print(f"Judging {len(items)} answers from {os.path.basename(args.report)}")
    rows = judge_answers(llm.llm, items, verbose=False)
    averages = answer_metrics(rows)

    print()
    print("=" * 60)
    print("  Rubric Re-Scoring Report")
    print("=" * 60)
    for name, score in averages.items():
        print(f"  {name:24s} {score}")

    wrong = [r for r in rows if r["verdict"] != "correct"]
    if wrong:
        print(f"\n  Non-correct verdicts ({len(wrong)}):")
        for r in wrong:
            print(f"    [{r['id']}] {r['verdict']}: {r['reason'][:100]}")

    if args.output:
        payload = {
            "mode": "answer (offline rescore)",
            "source_report": args.report,
            "testset": args.testset,
            "timestamp": datetime.now().isoformat(),
            "answer": {"averages": averages, "per_sample": rows},
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n  Saved to: {args.output}")


if __name__ == "__main__":
    main()
