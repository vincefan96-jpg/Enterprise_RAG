from pymilvus import (
    MilvusClient, FieldSchema, CollectionSchema,
    DataType, AnnSearchRequest, RRFRanker,
)
from app.config import Settings

# Alias for the ORM connection used by list_documents pagination.
_LIST_CONN_ALIAS = "list_documents"

VARCHAR_LIMITS = {
    "text": 4000,
    "parent_text": 12000,
    "parent_doc_id": 256,
    "doc_title": 512,
    "source_type": 32,
    "file_path": 1024,
}


def filter_by_doc_title(doc_title: str) -> str:
    escaped = doc_title.replace("\\", "\\\\").replace('"', '\\"')
    return f'doc_title == "{escaped}"'


class MilvusStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = None
        self._has_file_path_field = False
        self._initialized = False

    @property
    def is_ready(self) -> bool:
        return self._initialized

    def connect(self):
        if self.client is None:
            self.client = MilvusClient(uri=self.settings.milvus_uri)

    def _init_collection_sync(self):
        self.connect()
        name = self.settings.milvus_collection

        if self.client.has_collection(name):
            desc = self.client.describe_collection(name)
            field_names = {f["name"] for f in desc["fields"]}
            self._has_file_path_field = "file_path" in field_names
            self.client.load_collection(name)
            self._initialized = True
            return

        self._has_file_path_field = True

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
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=1024),
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
        self._initialized = True

    async def init_collection(self):
        self._init_collection_sync()

    def _ensure_ready(self):
        """Lazily (re)initialize so a Milvus that was down at startup self-heals."""
        if self._initialized:
            return
        try:
            self._init_collection_sync()
        except Exception:
            self.client = None
            raise

    def validate_chunks(self, chunks: list) -> None:
        for index, c in enumerate(chunks):
            for field, limit in VARCHAR_LIMITS.items():
                value = getattr(c, field, "") or ""
                if len(value) > limit:
                    raise ValueError(
                        f"第 {index + 1} 个文本块的 {field} 长度为 {len(value)} 字符，"
                        f"超过 Milvus 字段上限 {limit}；请调小 PARENT_CHUNK_SIZE / "
                        f"CHILD_CHUNK_SIZE 或缩短文件名"
                    )

    def insert(self, chunks: list, dense_vectors: list[list[float]], sparse_vectors: list[dict]):
        self._ensure_ready()
        self.validate_chunks(chunks)
        data = []
        for i, c in enumerate(chunks):
            item = {
                "text": c.text,
                "parent_text": c.parent_text,
                "parent_doc_id": c.parent_doc_id,
                "dense_vector": dense_vectors[i],
                "sparse_vector": sparse_vectors[i],
                "doc_title": c.doc_title,
                "chunk_index": c.chunk_index,
                "source_type": c.source_type,
            }
            if self._has_file_path_field:
                item["file_path"] = c.file_path
            data.append(item)
        self.client.insert(
            collection_name=self.settings.milvus_collection,
            data=data,
        )
        # Persist immediately so inserts survive a restart even without auto-flush.
        self.client.flush(collection_name=self.settings.milvus_collection)
    #混合搜索，实现了 稠密 + 稀疏 双路召回 + RRF 融合
    def hybrid_search(self, query_dense: list[float], query_sparse: dict,
                      dense_top_k: int, sparse_top_k: int,
                      rrf_k: int, fusion_top_k: int) -> list[dict]:
        self._ensure_ready()
        search_params_dense = {"metric_type": "IP", "params": {"nprobe": self.settings.milvus_nprobe}}
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

    def document_exists(self, doc_title: str) -> bool:
        self._ensure_ready()
        results = self.client.query(
            collection_name=self.settings.milvus_collection,
            filter=filter_by_doc_title(doc_title),
            output_fields=["id"],
            limit=1,
        )
        return bool(results)

    def delete_by_doc_title(self, doc_title: str) -> tuple[str, int]:
        """Delete every chunk of a document; returns (file_path, delete_count)."""
        self._ensure_ready()
        file_path = ""
        if self._has_file_path_field:
            results = self.client.query(
                collection_name=self.settings.milvus_collection,
                filter=filter_by_doc_title(doc_title),
                output_fields=["file_path"],
                limit=1,
            )
            file_path = results[0]["file_path"] if results else ""
        result = self.client.delete(
            collection_name=self.settings.milvus_collection,
            filter=filter_by_doc_title(doc_title),
        )
        # Deletes are not durable until flushed; without this they are lost on
        # restart and the documents reappear.
        self.client.flush(collection_name=self.settings.milvus_collection)
        deleted = result.get("delete_count", 0) if isinstance(result, dict) else 0
        return file_path, deleted

    def list_documents(self) -> list[str]:
        """Distinct document titles in the collection.

        Paginated via the ORM query iterator so collections larger than the
        server-side query window (16384) are not silently truncated; falls
        back to a single page if the iterator is unavailable.
        """
        self._ensure_ready()
        seen: set[str] = set()

        try:
            from pymilvus import Collection, connections

            connections.connect(alias=_LIST_CONN_ALIAS, uri=self.settings.milvus_uri)
            try:
                collection = Collection(
                    self.settings.milvus_collection, using=_LIST_CONN_ALIAS
                )
                iterator = collection.query_iterator(
                    batch_size=1000, expr="", output_fields=["doc_title"]
                )
                try:
                    while True:
                        rows = iterator.next()
                        if not rows:
                            break
                        for r in rows:
                            seen.add(r.get("doc_title", ""))
                finally:
                    iterator.close()
            finally:
                connections.disconnect(_LIST_CONN_ALIAS)
        except Exception:
            seen = {
                r.get("doc_title", "")
                for r in self.client.query(
                    collection_name=self.settings.milvus_collection,
                    filter="id >= 0",
                    output_fields=["doc_title"],
                    limit=10000,
                )
            }

        return sorted(seen)

    def close(self):
        if self.client:
            self.client.close()
        self.client = None
        self._initialized = False
