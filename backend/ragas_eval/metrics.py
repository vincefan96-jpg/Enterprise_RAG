"""Pure retrieval metrics computed locally — no Ragas/Milvus imports.

Retrieval is judged at document-title level: a retrieved chunk counts as a
hit when its ``doc_title`` is in the question's ``relevant_doc_titles``.
This is cheap and does not depend on an LLM judge.

``evidence_hit`` is the finer-grained variant: it checks whether the
verbatim supporting quote of a question is present in the retrieved
contexts, which is what the corrective loop is supposed to improve.
"""

import re

DEFAULT_KS = (1, 3, 5)

_WHITESPACE = re.compile(r"\s+")
_STRIP_CHARS = "「」\"'"


def _normalize(text: str) -> str:
    return _WHITESPACE.sub("", (text or "").translate(str.maketrans("", "", _STRIP_CHARS)))


def evidence_hit(contexts: list[str], evidence: str) -> float | None:
    """1.0 when the verbatim evidence quote appears in the retrieved contexts.

    Returns ``None`` for negative/unscored items (no evidence to look for),
    so they are skipped by the aggregator.
    """
    quote = _normalize(evidence)
    if not quote or quote.startswith("无（") or "负样本" in quote:
        return None

    joined = _normalize("\n".join(contexts or []))
    return 1.0 if quote in joined else 0.0


def title_hit_metrics(
    retrieved_titles: list[str],
    relevant_titles: list[str],
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict:
    """Hit@k / recall / MRR for an ordered list of retrieved doc titles.

    Returns ``{}`` when the question has no ground-truth document, so such
    questions are skipped by the aggregator instead of scoring 0.
    """
    relevant = {t for t in (relevant_titles or []) if t}
    if not relevant:
        return {}

    retrieved = [t for t in (retrieved_titles or [])]
    result: dict[str, float | int | None] = {
        f"hit@{k}": 1.0 if any(t in relevant for t in retrieved[:k]) else 0.0
        for k in ks
    }
    result["recall"] = len(set(retrieved) & relevant) / len(relevant)
    rank = next(
        (i + 1 for i, t in enumerate(retrieved) if t in relevant), None
    )
    result["mrr"] = 1.0 / rank if rank else 0.0
    result["first_relevant_rank"] = rank
    return result


def average_metrics(rows: list[dict]) -> dict:
    """Average numeric values across rows, ignoring None/NaN and nested dicts."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}

    for row in rows:
        for key, value in row.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if value != value:  # NaN
                continue
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1

    return {
        key: round(totals[key] / counts[key], 4) for key in totals
    }


def refusal_metrics(rows: list[dict]) -> dict:
    """Refusal / hallucination rates from judge verdicts.

    Each row needs ``expect_refusal`` (ground truth) and ``judge_refused``
    (None when the judge failed, which drops the row from the averages).
    """
    negatives = [
        r
        for r in rows
        if r["expect_refusal"] and r.get("judge_refused") is not None
    ]
    positives = [
        r
        for r in rows
        if not r["expect_refusal"] and r.get("judge_refused") is not None
    ]

    out: dict[str, float] = {}
    if negatives:
        refused = sum(1 for r in negatives if r["judge_refused"])
        out["refusal_recall"] = round(refused / len(negatives), 4)
        out["hallucination_rate"] = round(
            1 - refused / len(negatives), 4
        )
    if positives:
        out["false_refusal_rate"] = round(
            sum(1 for r in positives if r["judge_refused"]) / len(positives), 4
        )

    judged = negatives + positives
    if judged:
        out["refusal_accuracy"] = round(
            sum(
                1
                for r in judged
                if r["judge_refused"] == r["expect_refusal"]
            )
            / len(judged),
            4,
        )
    return out


_KEYWORD_SPLIT = re.compile(r"[、,，;；]")
_KEYWORD_STRIP = "「」\"'（）()。．,，;；:： "


def _normalize_phrase(text: str) -> str:
    return _WHITESPACE.sub(
        "", (text or "").translate(str.maketrans("", "", _KEYWORD_STRIP))
    ).lower()


def keyword_hit(answer: str, keywords: str) -> float | None:
    """Fraction of expected keywords present in the answer.

    Deterministic and free. Matching is normalized substring first, then a
    character-set fallback for keywords of 3+ characters (handles phrasing
    like 关键词「仅 Windows」 vs 回答「仅提供 Windows 版」).
    """
    expected = [k for k in _KEYWORD_SPLIT.split(keywords or "") if k.strip()]
    if not expected:
        return None

    normalized_answer = _normalize_phrase(answer)
    answer_chars = set(normalized_answer)

    hits = 0
    for keyword in expected:
        token = _normalize_phrase(keyword)
        if not token:
            continue
        if token in normalized_answer:
            hits += 1
        elif len(token) >= 3 and set(token) <= answer_chars:
            hits += 1

    return round(hits / len(expected), 4)


def answer_metrics(rows: list[dict]) -> dict:
    """Rubric verdict distribution plus keyword coverage."""
    judged = [r for r in rows if r.get("verdict")]
    out: dict[str, float] = {}
    if judged:
        total = len(judged)
        correct = sum(1 for r in judged if r["verdict"] == "correct")
        partial = sum(1 for r in judged if r["verdict"] == "partial")
        out["answer_accuracy"] = round(correct / total, 4)
        out["answer_partial_rate"] = round(partial / total, 4)
        out["answer_error_rate"] = round(
            (total - correct - partial) / total, 4
        )
        out["rubric_score"] = round((correct + 0.5 * partial) / total, 4)

    keyword_scores = [
        r["keyword_hit"]
        for r in rows
        if isinstance(r.get("keyword_hit"), float)
    ]
    if keyword_scores:
        out["keyword_hit"] = round(
            sum(keyword_scores) / len(keyword_scores), 4
        )
    return out
