from app.services.chunker import DocumentChunker

TEXT = (
    "第一段：云枢云平台 v3.2 于 2026 年 3 月 18 日发布。" * 12
    + "\n\n"
    + "第二段：企业版日志留存 3 年，标准版 180 天。" * 12
    + "\n\n"
    + "第三段：差旅住宿标准一线城市 500 元/晚。" * 12
    + "\n\n"
    + "第四段：P1 故障须 10 分钟内建立作战室。" * 12
)


def make_chunker():
    return DocumentChunker(parent_size=300, child_size=120, overlap=40)


def test_every_child_is_contained_in_its_parent():
    chunks = make_chunker().split(TEXT, "demo.txt")

    assert chunks
    for chunk in chunks:
        assert chunk.text.strip() in chunk.parent_text, chunk.text[:60]


def test_chunks_from_same_parent_share_parent_id():
    chunks = make_chunker().split(TEXT, "demo.txt")

    by_parent: dict[str, set[str]] = {}
    for chunk in chunks:
        by_parent.setdefault(chunk.parent_doc_id, set()).add(chunk.parent_text)

    # one parent_text per parent_doc_id, and children never span two parents
    assert all(len(texts) == 1 for texts in by_parent.values())
    assert len(by_parent) == len(make_chunker().parent_splitter.split_text(TEXT))


def test_chunk_index_is_sequential_and_ids_are_unique():
    chunks = make_chunker().split(TEXT, "demo.txt")

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len({c.id for c in chunks}) == len(chunks)
    assert len({c.parent_doc_id for c in chunks}) == len(
        {c.parent_text for c in chunks}
    )


def test_short_text_yields_single_parent_and_child():
    chunks = make_chunker().split("只有一句话。", "tiny.txt")

    assert len(chunks) == 1
    assert chunks[0].text.strip() in chunks[0].parent_text
    assert chunks[0].chunk_index == 0


def test_parent_text_covers_the_whole_document():
    chunker = make_chunker()
    chunks = chunker.split(TEXT, "demo.txt")

    parents = list(dict.fromkeys(c.parent_text for c in chunks))
    joined = "\n\n".join(parents)

    # every paragraph of the source survives in some parent
    for paragraph in [p for p in TEXT.split("\n\n") if p.strip()]:
        head = paragraph.strip()[:20]
        assert head in joined
