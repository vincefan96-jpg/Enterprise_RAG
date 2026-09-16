from ragas_eval.metrics import (
    answer_metrics,
    average_metrics,
    evidence_hit,
    keyword_hit,
    refusal_metrics,
    title_hit_metrics,
)


def test_hit_at_k_and_mrr_for_first_rank_hit():
    m = title_hit_metrics(["docA.txt"], ["docA.txt"])

    assert m["hit@1"] == 1.0
    assert m["hit@3"] == 1.0
    assert m["mrr"] == 1.0
    assert m["first_relevant_rank"] == 1
    assert m["recall"] == 1.0


def test_hit_at_k_and_mrr_for_second_rank_hit():
    m = title_hit_metrics(["docB.txt", "docA.txt"], ["docA.txt"])

    assert m["hit@1"] == 0.0
    assert m["hit@3"] == 1.0
    assert m["mrr"] == 0.5
    assert m["first_relevant_rank"] == 2


def test_recall_counts_all_relevant_documents():
    m = title_hit_metrics(
        ["docA.txt", "docC.txt"], ["docA.txt", "docB.txt"]
    )

    assert m["recall"] == 0.5
    assert m["hit@1"] == 1.0
    assert m["mrr"] == 1.0


def test_miss_returns_zero_scores():
    m = title_hit_metrics(["docX.txt", "docY.txt"], ["docA.txt"])

    assert m["hit@3"] == 0.0
    assert m["mrr"] == 0.0
    assert m["first_relevant_rank"] is None
    assert m["recall"] == 0.0


def test_question_without_ground_truth_document_is_skipped():
    assert title_hit_metrics(["docA.txt"], []) == {}
    assert title_hit_metrics([], []) == {}


def test_average_metrics_ignores_none_nan_and_nested_dicts():
    rows = [
        {"hit@1": 1.0, "mrr": 1.0, "first_relevant_rank": 1, "timing": {"a": 1}},
        {"hit@1": 0.0, "mrr": 0.5, "first_relevant_rank": None, "timing": {"a": 2}},
        {"hit@1": float("nan"), "mrr": 0.0, "first_relevant_rank": 3},
        {"hit@1": 1.0, "mrr": 1.0},
    ]

    avg = average_metrics(rows)

    assert avg["hit@1"] == 0.6667  # NaN row dropped -> (1 + 0 + 1) / 3
    assert avg["mrr"] == 0.625  # (1 + 0.5 + 0 + 1) / 4
    assert avg["first_relevant_rank"] == 2.0  # (1 + 3) / 2
    assert "timing" not in avg


def test_average_metrics_on_empty_rows():
    assert average_metrics([]) == {}


def test_evidence_hit_matches_despite_whitespace_differences():
    contexts = ["v3.2 版本于 2026 年 3 月 18 日正式发布，替代 v3.0 与 v3.1。"]
    evidence = "「v3.2 版本于 2026年3月18日 正式发布」"

    assert evidence_hit(contexts, evidence) == 1.0


def test_evidence_hit_misses_when_quote_absent():
    assert evidence_hit(["完全无关的段落"], "「境外补贴按当地标准表单执行。」") == 0.0


def test_evidence_hit_skips_negative_samples_and_empty_evidence():
    assert evidence_hit(["任意"], "无（本条为负样本，正确行为是拒答）") is None
    assert evidence_hit(["任意"], "") is None
    assert evidence_hit([], None) is None


def test_refusal_metrics_mix_of_verdicts():
    rows = [
        {"expect_refusal": True, "judge_refused": True},    # correct refusal
        {"expect_refusal": True, "judge_refused": False},   # hallucination
        {"expect_refusal": False, "judge_refused": False},  # answered
        {"expect_refusal": False, "judge_refused": True},   # false refusal
        {"expect_refusal": True, "judge_refused": None},    # judge failed
    ]

    m = refusal_metrics(rows)

    assert m["refusal_recall"] == 0.5
    assert m["hallucination_rate"] == 0.5
    assert m["false_refusal_rate"] == 0.5
    assert m["refusal_accuracy"] == 0.5


def test_refusal_metrics_without_negatives():
    m = refusal_metrics([{"expect_refusal": False, "judge_refused": False}])

    assert "refusal_recall" not in m
    assert "hallucination_rate" not in m
    assert m["false_refusal_rate"] == 0.0
    assert m["refusal_accuracy"] == 1.0


def test_keyword_hit_full_and_partial():
    assert keyword_hit("答案是 2026 年 3 月 18 日，已发布。", "2026 年 3 月 18 日") == 1.0
    assert (
        keyword_hit(
            "服务期顺延 3 天。", "3 天、99.0%"
        )
        == 0.5
    )


def test_keyword_hit_character_set_fallback_for_phrasing():
    # 关键词「仅 Windows」 vs 回答「仅提供 Windows 版」
    assert keyword_hit("桌面客户端仅提供 Windows 版，要求 Win10 以上。", "仅 Windows") == 1.0


def test_keyword_hit_without_keywords_is_skipped():
    assert keyword_hit("任意回答", "") is None
    assert keyword_hit("任意回答", None) is None


def test_answer_metrics_verdict_distribution_and_keywords():
    rows = [
        {"verdict": "correct", "keyword_hit": 1.0},
        {"verdict": "correct", "keyword_hit": 0.5},
        {"verdict": "partial", "keyword_hit": 1.0},
        {"verdict": "incorrect", "keyword_hit": 0.0},
        {"verdict": None, "keyword_hit": 1.0},  # judge failed
    ]

    m = answer_metrics(rows)

    assert m["answer_accuracy"] == 0.5
    assert m["answer_partial_rate"] == 0.25
    assert m["answer_error_rate"] == 0.25
    assert m["rubric_score"] == 0.625
    assert m["keyword_hit"] == 0.7
