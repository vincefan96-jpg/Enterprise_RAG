from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings
from functools import lru_cache

# Resolve repo-root paths explicitly so the app reads/writes the same
# locations regardless of the process working directory.
ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"


class Settings(BaseSettings):
    model_config = {
        "env_file": str(ENV_FILE),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

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

    upload_dir: str = "backend/uploads"
    max_upload_size_mb: int = 50
    skip_models: bool = False

    agent_history_turns: int = 6
    agent_max_iterations: int = 2
    agent_max_tool_calls: int = 3
    agent_retry_fusion_top_k: int = 50
    context_max_docs: int = 5

    # Conversation memory: "sqlite" persists across restarts, "memory" does not.
    checkpointer_backend: str = Field(
        default="sqlite",
        validation_alias=AliasChoices(
            "AGENT_CHECKPOINTER_BACKEND", "CHECKPOINTER_BACKEND"
        ),
    )
    checkpointer_path: str = Field(
        default="backend/data/checkpoints.sqlite",
        validation_alias=AliasChoices(
            "AGENT_CHECKPOINTER_PATH", "CHECKPOINTER_PATH"
        ),
    )

    @field_validator("upload_dir")
    @classmethod
    def _resolve_upload_dir(cls, value: str) -> str:
        path = Path(value)
        return str(path if path.is_absolute() else ROOT / path)


@lru_cache
def get_settings() -> Settings:
    return Settings()
