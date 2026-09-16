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
        """Split into parent chunks, then children *inside* each parent.

        Children are produced per parent so every child is guaranteed to be
        contained in the parent it points at (the previous proportional
        index mapping paired children with unrelated parents).
        """
        chunks: list[Chunk] = []
        child_index = 0

        for parent_text in self.parent_splitter.split_text(text):
            if not parent_text.strip():
                continue
            parent_id = str(uuid.uuid4())
            for child_text in self.child_splitter.split_text(parent_text):
                if not child_text.strip():
                    continue
                chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    text=child_text,
                    parent_text=parent_text,
                    parent_doc_id=parent_id,
                    doc_title=doc_title,
                    chunk_index=child_index,
                ))
                child_index += 1

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