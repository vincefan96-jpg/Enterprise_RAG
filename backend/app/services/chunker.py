import uuid
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    id: str
    text: str
    parent_text: str
    parent_doc_id: str
    doc_title: str
    chunk_index: int
    source_type: str = ""
    file_path: str = ""


class DocumentChunker:
    def __init__(self, parent_size: int = 1500, child_size: int = 500, overlap: int = 100):
        self.parent_splitter = RecursiveCharacterTextSplitter( #智能递归分割器，按优先级尝试分隔符
            chunk_size=parent_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],#优先级
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

    def split(self, text: str, doc_title: str) -> list[Chunk]:
        parent_texts = self.parent_splitter.split_text(text)
        child_texts = self.child_splitter.split_text(text)

        parent_ids = {}
        chunks = []
        for i, child in enumerate(child_texts):
            # 关键公式：将子块索引映射到父块索引
            parent_idx = min(i * len(parent_texts) // len(child_texts), len(parent_texts) - 1)
            parent = parent_texts[parent_idx]

            # 为每个唯一的父块生成ID
            if parent not in parent_ids:
                parent_ids[parent] = str(uuid.uuid4())

            # 创建 Chunk 对象
            chunks.append(Chunk(
                id=str(uuid.uuid4()),
                text=child,
                parent_text=parent,
                parent_doc_id=parent_ids[parent],
                doc_title=doc_title,
                chunk_index=i,
            ))
        return chunks
#父子分块：
# 用户提问
#    ↓
# 使用【子块】进行向量检索（更精确）
#    ↓
# 找到最相关的子块
#    ↓
# 返回对应的【父块】作为上下文（更完整）
#    ↓
# LLM 基于父块生成答案