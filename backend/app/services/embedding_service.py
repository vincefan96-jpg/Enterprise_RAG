import gc
from cachetools import TTLCache
from FlagEmbedding import BGEM3FlagModel
from app.config import Settings
import torch


class EmbeddingService:
    def __init__(self, settings: Settings):
        self.model = BGEM3FlagModel(
            settings.bge_model_path,
            use_fp16=True,#使用FP16精度以节省显存和提高速度
            device=settings.bge_device,
        )
        self._query_cache = TTLCache(maxsize=1000, ttl=3600)#创建最多1000条、有效期1小时的查询缓存

    def cleanup(self):
        del self.model
        self._query_cache.clear() #删除模型引用并清空缓存
        gc.collect() #强制垃圾回收
        if torch.cuda.is_available(): #如果使用GPU，则清空CUDA缓存释放显存
            torch.cuda.empty_cache()

    def encode(self, texts: list[str], max_length: int = 512) -> list[dict]:
        output = self.model.encode(
            texts,
            batch_size=4, #批量处理文本(batch_size=4)
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        results = []
        for i in range(len(texts)):
            results.append({
                "dense": output["dense_vecs"][i].tolist(),
                "sparse": self._sparse_to_dict(output["lexical_weights"][i]),
            })
        if torch.cuda.is_available(): #每次编码后清理GPU缓存
            torch.cuda.empty_cache()
        return results

    #专门用于文档编码，使用更大的max_length(8192 vs 512)
    def encode_documents(self, texts: list[str]) -> list[dict]:
        """Encode documents with longer max_length for better context capture."""
        return self.encode(texts, max_length=8192)

    def encode_query(self, query: str) -> dict:
        if query in self._query_cache: #先检查缓存，避免重复计算
            return self._query_cache[query]
        result = self.encode([query], max_length=512)[0] #对单个查询进行编码
        self._query_cache[query] = result #将结果存入缓存供后续使用
        return result

    def invalidate_cache(self):
        self._query_cache.clear() #提供手动清空缓存的方法

    def _sparse_to_dict(self, weights: dict) -> dict:
        return {int(k): float(v) for k, v in weights.items()} #将稀疏权重转换为标准字典格式

    @property
    def dense_dim(self) -> int:
        return self.model.model.config.hidden_size #返回密集向量的维度大小,从模型配置中获取隐藏层大小
