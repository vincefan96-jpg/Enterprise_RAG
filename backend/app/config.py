from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "knowledge_base"

    bge_model_path: str = "BAAI/bge-m3"
    bge_device: str = "cuda"

    reranker_model_path: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cuda"
    reranker_top_n: int = 10

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    deepseek_temperature: float = 0.1
    deepseek_max_tokens: int = 1024
    deepseek_timeout: int = 60
    deepseek_max_retries: int = 2

    milvus_nprobe: int = 32

    hybrid_dense_top_k: int = 50
    hybrid_sparse_top_k: int = 50
    rrf_k: int = 60
    hybrid_fusion_top_k: int = 30

    parent_chunk_size: int = 1000
    child_chunk_size: int = 250
    chunk_overlap: int = 100

    upload_dir: str = "./uploads"
    max_upload_size_mb: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()
