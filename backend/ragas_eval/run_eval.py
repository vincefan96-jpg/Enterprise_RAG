"""Ragas-based evaluation CLI.

Usage:
  python -m ragas_eval.run_eval generate --num-questions 3 --output testset.json
  python -m ragas_eval.run_eval eval --testset testset.json --mode full
  python -m ragas_eval.run_eval eval --testset testset.json --mode retrieval
  python -m ragas_eval.run_eval eval --testset testset.json --mode generation
  python -m ragas_eval.run_eval eval --testset testset.json --agent-mode p0

Metrics:
  retrieval  — Hit@k / recall / MRR by doc_title (local, no LLM judge)
  generation — Faithfulness / AnswerRelevancy / AnswerCorrectness (Ragas)
  both       — real per-stage timings (pipeline vs generation) from the
               LangGraph update stream
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.tokenizer_patch import apply as _apply_tokenizer_patch

_apply_tokenizer_patch()

from app.config import get_settings
from .generator import SyntheticTestSetGenerator
from ragas_eval.runner import RagasEvaluationRunner


# ---------------------------------------------------------------------------
# service lifecycle
# ---------------------------------------------------------------------------


def init_services(settings, skip_models=False):
    from app.services.milvus_store import MilvusStore

    store = MilvusStore(settings)
    try:
        store.connect()
        asyncio.run(store.init_collection())
    except Exception as e:
        raise ConnectionError(
            f"\n  Cannot connect to Milvus at {settings.milvus_uri}\n"
            "  Please start Docker infrastructure first:\n"
            "    cd d:/RAG1.0 && docker compose up -d\n"
            f"  Original error: {e}"
        ) from e

    embed = None
    reranker = None
    if not skip_models:
        from app.services.embedding_service import EmbeddingService

        embed = EmbeddingService(settings)

        from app.services.reranker_service import RerankerService

        reranker = RerankerService(settings)

    from app.services.llm_service import LLMService

    llm = LLMService(settings)

    return store, embed, reranker, llm


def cleanup(embed, reranker, store):
    if embed:
        embed.cleanup()
    if reranker:
        reranker.cleanup()
    if store and store.client:
        store.client.close()


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_generate(args):
    settings = get_settings()
    if args.collection:
        settings.milvus_collection = args.collection
    store, embed, reranker, llm = init_services(settings, skip_models=True)

    try:
        doc_titles = None
        if args.doc_titles:
            doc_titles = [t.strip() for t in args.doc_titles.split(",")]

        generator = SyntheticTestSetGenerator(store, settings)
        dataset = generator.generate(
            doc_titles=doc_titles,
            num_questions=args.num_questions,
        )

        if not dataset:
            print("No questions generated. Make sure documents are uploaded.")
            return

        SyntheticTestSetGenerator.save(dataset, args.output)
        print(f"Generated {len(dataset)} questions -> {args.output}")
    finally:
        cleanup(embed, reranker, store)


def cmd_eval(args):
    settings = get_settings()
    if args.collection:
        settings.milvus_collection = args.collection
    if args.fusion_top_k:
        settings.hybrid_fusion_top_k = args.fusion_top_k
    if args.reranker_top_n:
        settings.reranker_top_n = args.reranker_top_n
    if args.agent_retry_top_k:
        settings.agent_retry_fusion_top_k = args.agent_retry_top_k
    if args.context_max_docs >= 0:
        settings.context_max_docs = args.context_max_docs
    dataset = SyntheticTestSetGenerator.load(args.testset)
    if not dataset:
        print(f"No queries found in {args.testset}")
        return
    if args.limit:
        dataset = dataset[: args.limit]

    store, embed, reranker, llm = init_services(settings)

    try:
        runner = RagasEvaluationRunner(
            store, embed, reranker, llm, settings, agent_mode=args.agent_mode
        )
        timestamp = datetime.now().isoformat().replace(":", "-")

        if args.mode == "retrieval":
            sections = runner.run_retrieval(dataset)
        elif args.mode == "generation":
            sections = runner.run_generation(dataset)
        elif args.mode == "refusal":
            sections = runner.run_refusal(dataset)
        elif args.mode == "answer":
            sections = runner.run_answer(dataset)
        else:  # full
            sections = runner.run_full(dataset)

        report = {
            "mode": args.mode,
            "num_queries": len(dataset),
            "agent_mode": args.agent_mode,
            "collection": settings.milvus_collection,
            "retrieval_metrics": "hit@k / recall / mrr (by doc_title) + evidence_hit",
            "hybrid_fusion_top_k": settings.hybrid_fusion_top_k,
            "agent_retry_fusion_top_k": settings.agent_retry_fusion_top_k,
            "rrf_k": settings.rrf_k,
            "reranker_top_n": settings.reranker_top_n,
            "agent_max_iterations": settings.agent_max_iterations,
            "timestamp": timestamp,
            **sections,
        }

        # console report
        _print_report(report)

        # save JSON
        output_path = os.path.join(
            args.output_dir, f"ragas_eval_report_{timestamp}.json"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n  Report saved to: {output_path}")

    finally:
        cleanup(embed, reranker, store)


def _print_report(report: dict):
    print()
    print("=" * 60)
    print("  RAGAS Evaluation Report")
    print("=" * 60)
    print(f"  Mode:              {report.get('mode', '?')}")
    print(f"  Agent mode:        {report.get('agent_mode', '?')}")
    print(f"  Test queries:      {report.get('num_queries', 0)}")

    for section_key, section_title in [
        ("retrieval", "Retrieval Metrics (hit@k / recall / MRR / evidence)"),
        ("answer", "Answer Quality (rubric judge, negatives excluded)"),
        ("generation", "Generation Metrics (Ragas)"),
        ("refusal", "Refusal / Hallucination Metrics (LLM judge)"),
    ]:
        section = report.get(section_key)
        if not section:
            continue
        print()
        print(f"  --- {section_title} ---")
        averages = section.get("averages", {})
        if averages:
            for name, score in averages.items():
                try:
                    print(f"  {name:30s} {float(score):.4f}")
                except (ValueError, TypeError):
                    print(f"  {name:30s} {score}")
        else:
            print("  (no aggregate scores)")

        per_sample = section.get("per_sample", [])
        if per_sample:
            print(f"  (per-sample details: {len(per_sample)} rows in JSON report)")

    print()
    print("=" * 60)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Ragas RAG Evaluation Toolkit")
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate synthetic test dataset")
    gen.add_argument("--num-questions", type=int, default=3)
    gen.add_argument("--output", default="testset.json")
    gen.add_argument(
        "--doc-titles", default=None, help="Comma-separated doc titles"
    )
    gen.add_argument(
        "--collection", default=None, help="Milvus collection override"
    )

    ev = sub.add_parser("eval", help="Run Ragas evaluation")
    ev.add_argument("--testset", required=True)
    ev.add_argument(
        "--mode",
        choices=["retrieval", "generation", "refusal", "answer", "full"],
        default="full",
    )
    ev.add_argument(
        "--agent-mode",
        choices=["p0", "p1"],
        default="p1",
        help="p1 runs the corrective loop; p0 runs the single-pass graph",
    )
    ev.add_argument("--output-dir", default=".")
    ev.add_argument(
        "--collection", default=None, help="Milvus collection override"
    )
    ev.add_argument(
        "--limit", type=int, default=0, help="Only evaluate the first N queries"
    )
    ev.add_argument(
        "--fusion-top-k",
        type=int,
        default=0,
        help="Override HYBRID_FUSION_TOP_K (retrieval pressure experiments)",
    )
    ev.add_argument(
        "--reranker-top-n",
        type=int,
        default=0,
        help="Override RERANKER_TOP_N",
    )
    ev.add_argument(
        "--agent-retry-top-k",
        type=int,
        default=0,
        help="Override AGENT_RETRY_FUSION_TOP_K (corrective-retry window)",
    )
    ev.add_argument(
        "--context-max-docs",
        type=int,
        default=-1,
        help="Override CONTEXT_MAX_DOCS (0 = unlimited)",
    )

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
