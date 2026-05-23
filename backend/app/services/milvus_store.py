from pymilvus import (
    MilvusClient, FieldSchema, CollectionSchema,
    DataType, AnnSearchRequest, RRFRanker,
)
from app.config import Settings


class MilvusStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = None

    def connect(self):
        self.client = MilvusClient(uri=self.settings.milvus_uri)

    async def init_collection(self):
        self.connect()
        name = self.settings.milvus_collection

        if self.client.has_collection(name):
            self.client.load_collection(name)
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4000),
            FieldSchema(name="parent_text", dtype=DataType.VARCHAR, max_length=12000),
            FieldSchema(name="parent_doc_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="doc_title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="chunk_index", dtype=DataType.INT32),
            FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=32),
        ]
        schema = CollectionSchema(fields, description="RAG Knowledge Base")

        index_params = self.client.prepare_index_params()#创建索引
        index_params.add_index(
            field_name="dense_vector",
            index_type="IVF_FLAT",
            metric_type="IP",
            params={"nlist": 128},
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_WAND",
            metric_type="IP",
        )

        self.client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params,
        )
        self.client.load_collection(name)#创建集合后立即加载到内存，使其可进行查询

    def insert(self, chunks: list, dense_vectors: list[list[float]], sparse_vectors: list[dict]):
        data = []
        for i, c in enumerate(chunks):
            data.append({
                "text": c.text,
                "parent_text": c.parent_text,
                "parent_doc_id": c.parent_doc_id,
                "dense_vector": dense_vectors[i],
                "sparse_vector": sparse_vectors[i],
                "doc_title": c.doc_title,
                "chunk_index": c.chunk_index,
                "source_type": c.source_type,
            })
        self.client.insert(
            collection_name=self.settings.milvus_collection,
            data=data,
        )
    #混合搜索，实现了 稠密 + 稀疏 双路召回 + RRF 融合
    def hybrid_search(self, query_dense: list[float], query_sparse: dict,
                      dense_top_k: int, sparse_top_k: int,
                      rrf_k: int, fusion_top_k: int) -> list[dict]:
        search_params_dense = {"metric_type": "IP", "params": {"nprobe": 8}}
        req_dense = AnnSearchRequest(
            data=[query_dense],
            anns_field="dense_vector",
            param=search_params_dense,
            limit=dense_top_k,
        )
        search_params_sparse = {"metric_type": "IP"}
        req_sparse = AnnSearchRequest(
            data=[query_sparse],
            anns_field="sparse_vector",
            param=search_params_sparse,
            limit=sparse_top_k,
        )
        rerank = RRFRanker(k=rrf_k) #rrf_k ：控制 RRF 公式中的平滑常数，值越大，排名靠后结果的影响越小。
        results = self.client.hybrid_search(
            collection_name=self.settings.milvus_collection,
            reqs=[req_dense, req_sparse],
            ranker=rerank,
            limit=fusion_top_k,#最终返回的结果数量
            output_fields=["text", "parent_text", "doc_title", "parent_doc_id"],
        )
        hits = []#将原始结果转换为统一的字典格式，方便上层使用
        for hit in results[0]:
            entity = hit.get("entity", hit)
            hits.append({
                "id": hit.get("id", ""),
                "text": entity.get("text", ""),
                "parent_text": entity.get("parent_text", ""),
                "doc_title": entity.get("doc_title", ""),
                "parent_doc_id": entity.get("parent_doc_id", ""),
                "score": hit.get("distance", hit.get("score", 0.0)),
            })
        return hits

    def delete_by_doc_title(self, doc_title: str):
        self.client.delete(
            collection_name=self.settings.milvus_collection,
            filter=f'doc_title == "{doc_title}"',
        )

    def list_documents(self) -> list[str]:
        results = self.client.query(
            collection_name=self.settings.milvus_collection,
            filter="id >= 0",
            output_fields=["doc_title"],
            limit=10000,
        )
        seen = set()
        for r in results:
            seen.add(r.get("doc_title", ""))
        return sorted(seen)

    def close(self):
        if self.client:
            self.client.close()
