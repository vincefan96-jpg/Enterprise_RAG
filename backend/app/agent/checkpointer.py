"""Checkpointer factory: SQLite when configured, in-memory as fallback.

The SQLite saver stores LangGraph checkpoints (conversation memory) on disk,
so a session survives backend restarts. Paths are resolved against the repo
root, matching how ``config.py`` resolves ``.env``.
"""

from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from app.config import Settings

ROOT = Path(__file__).resolve().parents[3]


def resolve_checkpointer_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


@asynccontextmanager
async def open_checkpointer(settings: Settings):
    backend = (getattr(settings, "checkpointer_backend", "sqlite") or "sqlite").lower()

    if backend == "sqlite":
        stack = AsyncExitStack()
        saver = None
        path = resolve_checkpointer_path(settings.checkpointer_path)
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            path.parent.mkdir(parents=True, exist_ok=True)
            saver = await stack.enter_async_context(
                AsyncSqliteSaver.from_conn_string(str(path))
            )
        except Exception as e:  # missing package, locked file, ...
            print(f"WARNING: SQLite checkpointer unavailable ({e}); using in-memory")
            await stack.aclose()

        if saver is not None:
            print(f"Checkpointer: SQLite ({path})")
            try:
                yield saver
            finally:
                await stack.aclose()
            return

    from langgraph.checkpoint.memory import InMemorySaver

    print("Checkpointer: in-memory (conversations are lost on restart)")
    yield InMemorySaver()
