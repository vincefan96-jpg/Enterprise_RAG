from app.services.tokenizer_patch import apply as _apply_tokenizer_patch
_apply_tokenizer_patch()

import os
import gc
import atexit
import signal
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import documents, query
import torch

SKIP_MODELS = os.getenv("SKIP_MODELS", "0") == "1"


def _cleanup_models(app):
    """Release GPU memory held by embedding and reranker services."""
    for attr in ("embedding_service", "reranker_service"):
        svc = getattr(app.state, attr, None)
        if svc is not None and hasattr(svc, "cleanup"):
            try:
                svc.cleanup()
            except Exception:
                pass
        setattr(app.state, attr, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory released")


def _register_cleanup(app):
    """Ensure cleanup runs on normal exit and SIGINT/SIGTERM."""
    ran = {"done": False}

    def _do():
        if not ran["done"]:
            ran["done"] = True
            _cleanup_models(app)

    atexit.register(_do)

    def _handler(signum, frame):
        _do()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass  # signal only works in main thread on Windows


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.config import get_settings
    settings = get_settings()

    from app.services.milvus_store import MilvusStore
    store = MilvusStore(settings)
    try:
        await store.init_collection()
        app.state.milvus_ready = True
    except Exception as e:
        print(f"WARNING: Milvus init failed: {e}")
        app.state.milvus_ready = False
    app.state.milvus_store = store

    if not SKIP_MODELS:
        from app.services.embedding_service import EmbeddingService
        try:
            embed = EmbeddingService(settings)
            app.state.embedding_service = embed
            app.state.embedding_ready = True
        except Exception as e:
            print(f"WARNING: Embedding init failed: {e}")
            app.state.embedding_ready = False

        from app.services.reranker_service import RerankerService
        try:
            reranker = RerankerService(settings)
            app.state.reranker_service = reranker
            app.state.reranker_ready = True
        except Exception as e:
            print(f"WARNING: Reranker init failed: {e}")
            app.state.reranker_ready = False
    else:
        app.state.embedding_service = None
        app.state.embedding_ready = False
        app.state.reranker_service = None
        app.state.reranker_ready = False

    from app.services.llm_service import LLMService
    try:
        llm = LLMService(settings)
        app.state.llm_service = llm
        app.state.llm_ready = True
    except Exception as e:
        print(f"WARNING: LLM init failed: {e}")
        app.state.llm_ready = False

    _register_cleanup(app)

    yield

    _cleanup_models(app)
    store.close()


app = FastAPI(title="RAG Knowledge Base", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(query.router)


@app.get("/api/health")
async def health():
    state = app.state
    return {
        "api": "ok",
        "milvus": getattr(state, "milvus_ready", False),
        "embedding": getattr(state, "embedding_ready", False),
        "reranker": getattr(state, "reranker_ready", False),
        "llm": getattr(state, "llm_ready", False),
    }
