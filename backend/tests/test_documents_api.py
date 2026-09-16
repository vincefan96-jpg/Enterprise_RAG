"""Upload/delete API tests with fake store and embedding services."""

import httpx
import pytest
from fastapi import FastAPI

from app.api import documents as documents_module
from app.config import get_settings


class FakeStore:
    def __init__(self, existing=()):
        self.titles = set(existing)
        self.file_paths = {}
        self.inserted = []
        self.deleted = []
        self.reject = None

    def validate_chunks(self, chunks):
        if self.reject:
            raise ValueError(self.reject)

    def document_exists(self, title):
        return title in self.titles

    def delete_by_doc_title(self, title):
        self.deleted.append(title)
        if title not in self.titles:
            return "", 0
        self.titles.discard(title)
        return self.file_paths.pop(title, ""), 3

    def insert(self, chunks, dense_vectors, sparse_vectors):
        self.inserted.append(chunks)
        title = chunks[0].doc_title
        self.titles.add(title)
        self.file_paths[title] = chunks[0].file_path

    def list_documents(self):
        return sorted(self.titles)


class FakeEmbed:
    def encode_documents(self, texts):
        return [{"dense": [0.0], "sparse": {1: 1.0}} for _ in texts]


def make_app(store, embed):
    app = FastAPI()
    app.include_router(documents_module.router)
    app.state.milvus_store = store
    app.state.embedding_service = embed
    return app


async def request(app, method, path, **kwargs):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, **kwargs)


async def upload(app, filename="demo.txt", content=b"hello"):
    return await request(
        app,
        "POST",
        "/api/documents/upload",
        files={"file": (filename, content, "text/plain")},
    )


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "upload_dir", str(tmp_path))
    return tmp_path


@pytest.fixture
def parsed(monkeypatch):
    monkeypatch.setattr(
        documents_module.DocumentParser,
        "parse",
        lambda self, file_path: "第一段内容。\n\n第二段内容。",
    )


async def test_upload_indexes_document(upload_dir, parsed):
    store = FakeStore()
    resp = await upload(make_app(store, FakeEmbed()))

    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "demo.txt"
    assert body["chunks"] >= 1
    assert store.inserted and store.inserted[0][0].doc_title == "demo.txt"
    assert len(list(upload_dir.iterdir())) == 1


async def test_upload_replaces_same_title(upload_dir, parsed):
    store = FakeStore(existing=["demo.txt"])
    resp = await upload(make_app(store, FakeEmbed()))

    assert resp.status_code == 200
    assert "已覆盖同名文档" in resp.json()["message"]
    assert store.deleted == ["demo.txt"]
    assert store.titles == {"demo.txt"}


async def test_upload_rejects_legacy_doc(upload_dir, parsed):
    resp = await upload(make_app(FakeStore(), FakeEmbed()), filename="old.doc")

    assert resp.status_code == 400
    assert "docx" in resp.json()["detail"]


async def test_upload_rejects_empty_extracted_text(upload_dir, monkeypatch):
    monkeypatch.setattr(
        documents_module.DocumentParser, "parse", lambda self, path: "   \n"
    )
    resp = await upload(make_app(FakeStore(), FakeEmbed()))

    assert resp.status_code == 400
    assert not list(upload_dir.iterdir())


async def test_upload_returns_503_without_embedding(upload_dir, parsed):
    resp = await upload(make_app(FakeStore(), None))

    assert resp.status_code == 503
    assert not list(upload_dir.iterdir())


async def test_upload_returns_400_on_chunk_validation(upload_dir, parsed):
    store = FakeStore()
    store.reject = "文本块超长"
    resp = await upload(make_app(store, FakeEmbed()))

    assert resp.status_code == 400
    assert resp.json()["detail"] == "文本块超长"
    assert not list(upload_dir.iterdir())


async def test_delete_missing_document_is_404():
    store = FakeStore()
    resp = await request(
        make_app(store, FakeEmbed()), "DELETE", "/api/documents/nope.txt"
    )

    assert resp.status_code == 404
    assert store.deleted == ["nope.txt"]


async def test_delete_existing_document_removes_file(upload_dir, parsed):
    store = FakeStore(existing=["demo.txt"])
    app = make_app(store, FakeEmbed())
    await upload(app)

    resp = await request(app, "DELETE", "/api/documents/demo.txt")

    assert resp.status_code == 200
    assert store.titles == set()
    assert not list(upload_dir.iterdir())
