"""Generate distractor documents for retrieval-pressure evaluation.

Creates same-style 云枢科技 documents for *other* products/policies so the
eval corpus has enough noise for top-k retrieval to actually miss things.
Generated docs are checked against every question's expected keywords and
verbatim evidence; any document leaking an answer is regenerated or dropped.

Usage:
  python -m ragas_eval.generate_distractors [--count 36] [--out DIR]
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.tokenizer_patch import apply as _apply_tokenizer_patch

_apply_tokenizer_patch()

ENTITIES = [
    ("云枢数据中台 DataHub", "数据产品部"),
    ("云枢低代码平台 NimbusApp", "应用平台部"),
    ("云枢智能分析 NimbusInsight", "分析产品部"),
    ("云枢统一身份 NimbusID", "身份与安全部"),
    ("云枢运维中枢 NimbusOps", "平台运维部"),
    ("云枢边缘计算网关 EdgeOne", "边缘计算部"),
    ("云枢 AI 助手 Copilot", "智能应用部"),
    ("云枢云盘 DriveBox", "协同办公部"),
    ("云枢工单系统 ServiceHub", "服务管理部"),
    ("云枢监控平台 WatchTower", "可观测性部"),
    ("云枢数据交换平台 DataLink", "数据集成部"),
    ("云枢开放平台 OpenNimbus", "生态合作部"),
]

KINDS = [
    ("产品手册", "产品概述、版本历史、模块规格上限表（含数字）、客户端要求"),
    ("定价与销售政策", "版本定价表（含价格数字）、折扣规则、续费与退款、价格保护"),
    ("售后技术支持与 SLA 服务协议", "服务时段、响应时限表、可用性承诺与赔付规则"),
    ("信息安全与数据合规制度", "数据分级、访问审批、日志与录屏留存、违规处理"),
    ("差旅与费用报销管理制度", "出差住宿标准表（含金额）、补贴规则、报销流程与时限"),
    ("运维操作规范", "变更发布、故障分级与响应时限表、值班与升级机制"),
]

DOC_PROMPT = """你是「云枢科技」的内部文档撰写者。请写一份完整的《{title}》。

要求：
- 文档主题：{entity} 的{kind}，归属部门：{dept}
- 必须包含章节编号（§1、§2……），至少包含一个 Markdown 表格，并给出 6-10 个具体数字（金额、时限、版本号、容量等）
- 篇幅 400-700 字，风格与公司其他制度文档一致，直接输出正文，不要任何解释或前后缀
- 第一行是文档标题，第二行写：归属部门：{dept}　|　版本：{version}

严禁（这些事实已属于其他文档，不得出现，包括近似写法）：
{blocklist}"""


def load_blocklist(testset_path: str) -> list[str]:
    data = json.load(open(testset_path, encoding="utf-8"))
    tokens = set()
    for q in data["queries"]:
        for kw in re.split(r"[、,，;；]", q.get("keywords") or ""):
            kw = kw.strip()
            if not kw or kw in {"拒答", "无该指标"}:
                continue
            if re.search(r"\d", kw) or len(kw) >= 5:
                tokens.add(kw)
    return sorted(tokens)


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text or "").translate(
        str.maketrans("", "", "「」\"'（）()。，,；;：: 　")
    )


# Only high-signal facts are blocked: verbatim evidence quotes plus
# distinctive identifiers (dates, amounts with 4+ digits, decimals with %).
# Generic numbers ("5 个工作日", "85 折") are allowed — they make the
# distractors realistic without making questions answerable.
DATE_RE = re.compile(r"\d{4}年\d{0,2}月?\d{0,2}日?")
AMOUNT_RE = re.compile(r"\d{4,}元")
PERCENT_RE = re.compile(r"\d+\.\d+%")


def is_distinctive_fact(token: str) -> bool:
    return bool(
        DATE_RE.search(token) or AMOUNT_RE.search(token) or PERCENT_RE.search(token)
    )


def load_guards(testset_path: str):
    data = json.load(open(testset_path, encoding="utf-8"))
    keywords, evidence = [], []
    for q in data["queries"]:
        for kw in re.split(r"[、，;；]", q.get("keywords") or ""):
            kw = kw.strip()
            if kw and is_distinctive_fact(normalize(kw)):
                keywords.append(normalize(kw))
        ev = normalize(q.get("evidence") or "")
        if len(ev) >= 10 and not (q.get("evidence") or "").startswith("无"):
            evidence.append(ev)
    return keywords, evidence


def leaks(text: str, keywords: list[str], evidence: list[str]) -> list[str]:
    normalized = normalize(text)
    hits = [k for k in keywords if k and k in normalized]
    hits += [e for e in evidence if e and e in normalized]
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=36)
    parser.add_argument(
        "--out",
        default=os.path.join("ragas_eval", "testsets", "general_50", "distractors"),
    )
    parser.add_argument(
        "--testset",
        default=os.path.join("ragas_eval", "testsets", "general_50", "testset.json"),
    )
    args = parser.parse_args()

    from langchain_openai import ChatOpenAI
    from app.config import get_settings

    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0.7,
        max_tokens=2048,
        request_timeout=120,
        max_retries=2,
    )

    os.makedirs(args.out, exist_ok=True)
    blocklist_tokens = load_blocklist(args.testset)
    keywords, evidence = load_guards(args.testset)
    blocklist = "、".join(blocklist_tokens)
    print(f"blocklist tokens: {len(blocklist_tokens)} | guards: {len(keywords)} kw, {len(evidence)} evidence")

    specs = []
    for i, (entity, dept) in enumerate(ENTITIES):
        for j in range(3):
            kind, outline = KINDS[(i + j) % len(KINDS)]
            specs.append((entity, dept, kind, outline))
    specs = specs[: args.count]

    written, dropped, skipped = [], [], []
    for idx, (entity, dept, kind, outline) in enumerate(specs, start=1):
        title = f"{entity}·{kind}"
        version = f"v{idx // 10 + 1}.{idx % 10}"
        name = f"X{idx:02d}_{entity}_{kind}.txt".replace(" ", "")
        path = os.path.join(args.out, name)

        if os.path.exists(path) and not leaks(
            open(path, encoding="utf-8").read(), keywords, evidence
        ):
            skipped.append(name)
            print(f"  [{idx}] skip (exists): {name}")
            continue

        prompt = DOC_PROMPT.format(
            title=title,
            entity=entity,
            kind=kind,
            dept=dept,
            version=version,
            blocklist=blocklist,
        )

        body = ""
        for attempt in range(2):
            try:
                body = llm.invoke(prompt).content.strip()
            except Exception as e:
                print(f"  [{idx}] LLM error: {e}")
                body = ""
            hits = leaks(body, keywords, evidence) if body else []
            if body and not hits:
                break
            if hits:
                print(f"  [{idx}] leak detected {hits[:3]}, retrying")
                prompt += "\n\n上一版出现了被禁止的事实，请彻底避开并重写。"

        name = f"X{idx:02d}_{entity}_{kind}.txt".replace(" ", "")
        if not body or hits:
            dropped.append(name)
            print(f"  [{idx}] DROPPED ({title})")
            continue

        with open(path, "w", encoding="utf-8") as f:
            f.write(body + "\n")
        written.append(name)
        print(f"  [{idx}] {name} ({len(body)} chars)")

    total = len(written) + len(skipped)
    print(
        f"\nwritten: {len(written)} | skipped: {len(skipped)} | dropped: {len(dropped)} "
        f"| total present: {total} -> {args.out}"
    )
    if dropped:
        print("dropped:", dropped)


if __name__ == "__main__":
    main()
