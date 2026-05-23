import gc
from FlagEmbedding import FlagReranker
from langchain_core.documents import Document
from langchain_core.callbacks import Callbacks
from langchain_core.documents.compressor import BaseDocumentCompressor
from app.config import Settings
import torch


class RerankerService(BaseDocumentCompressor):
    model_config = {"arbitrary_types_allowed": True}

    top_n: int = 5

    def __init__(self, settings: Settings):
        super().__init__()
        object.__setattr__(self, "model", FlagReranker(
            settings.reranker_model_path,
            use_fp16=True,
            device=settings.reranker_device,
        ))
        self.top_n = settings.reranker_top_n

    def cleanup(self):
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def compress_documents(self, documents: list[Document], query: str,
                           callbacks: Callbacks = None) -> list[Document]:
        if not documents:
            return []
        #构造 query-document 对输出一个相关性分数
        pairs = [[query, doc.page_content] for doc in documents]
        #计算每个 document 对 query 的相关性分数
        scores = self.model.compute_score(pairs)#返回的分数是 相关性得分 （值越大越相关）
        if not isinstance(scores, list):
            scores = [scores]
        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        result = [doc for doc, _ in scored[:self.top_n]]
        torch.cuda.empty_cache()
        return result
