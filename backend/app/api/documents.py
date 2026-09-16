import asyncio
import os
import re
import uuid
import traceback
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from app.models.schemas import DocumentUploadResponse, DocumentListResponse, DeleteResponse
from app.services.document_parser import DocumentParser
from app.services.chunker import DocumentChunker
from app.config import get_settings

router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def _write_bytes(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)):
    settings = get_settings()

    ext = Path(file.filename or "").suffix.lower()
    if ext == ".doc":
        raise HTTPException(400, "暂不支持旧版 .doc，请另存为 .docx 后上传")
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"不支持的文件类型: {ext or '(无扩展名)'}")

    title = _CONTROL_CHARS.sub("", Path(file.filename).name).strip()
    if not title:
        raise HTTPException(400, "文件名无效")

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if file.size is not None and file.size > max_bytes:
        raise HTTPException(400, f"文件超过 {settings.max_upload_size_mb}MB 限制")
    contents = await file.read()
    if len(contents) > max_bytes:
        raise HTTPException(400, f"文件超过 {settings.max_upload_size_mb}MB 限制")

    os.makedirs(settings.upload_dir, exist_ok=True)
    file_id = uuid.uuid4().hex
    save_path = os.path.join(settings.upload_dir, f"{file_id}{ext}")
    await asyncio.to_thread(_write_bytes, save_path, contents)

    try:
        parser = DocumentParser()
        text = await asyncio.to_thread(parser.parse, save_path)
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(500, f"文档解析失败: {str(e)}")

    if not text.strip():
        os.remove(save_path)
        raise HTTPException(400, "未从文档中提取到文本内容（可能是扫描版 PDF 或空文件）")

    chunker = DocumentChunker(
        parent_size=settings.parent_chunk_size,
        child_size=settings.child_chunk_size,
        overlap=settings.chunk_overlap,
    )
    chunks = chunker.split(text, title)
    if not chunks:
        os.remove(save_path)
        raise HTTPException(400, "文档分块结果为空")
    for c in chunks:
        c.source_type = ext.lstrip(".")
        c.file_path = save_path

    store = request.app.state.milvus_store
    embed_service = request.app.state.embedding_service
    if embed_service is None:
        os.remove(save_path)
        raise HTTPException(503, "Embedding 模型未加载，无法建立索引；请检查启动日志或 SKIP_MODELS 配置")

    replaced = False
    try:
        await asyncio.to_thread(store.validate_chunks, chunks)
        texts = [c.text for c in chunks]
        embeddings = await asyncio.to_thread(embed_service.encode_documents, texts)
        dense_vecs = [e["dense"] for e in embeddings]
        sparse_vecs = [e["sparse"] for e in embeddings]

        old_path = ""
        if await asyncio.to_thread(store.document_exists, title):
            replaced = True
            old_path, _ = await asyncio.to_thread(store.delete_by_doc_title, title)
        await asyncio.to_thread(store.insert, chunks, dense_vecs, sparse_vecs)
        if (
            old_path
            and os.path.abspath(old_path) != os.path.abspath(save_path)
            and os.path.isfile(old_path)
        ):
            os.remove(old_path)
    except ValueError as e:
        os.remove(save_path)
        raise HTTPException(400, str(e))
    except Exception as e:
        os.remove(save_path)
        traceback.print_exc()
        raise HTTPException(500, f"索引失败: {str(e)}")

    action = "已覆盖同名文档，" if replaced else ""
    return DocumentUploadResponse(
        id=file_id,
        title=title,
        chunks=len(chunks),
        message=f"{action}成功上传并索引 {len(chunks)} 个文本块",
    )


@router.get("/list", response_model=DocumentListResponse)
async def list_documents(request: Request):
    store = request.app.state.milvus_store
    docs = await asyncio.to_thread(store.list_documents)
    return DocumentListResponse(documents=docs)


@router.delete("/{doc_title:path}", response_model=DeleteResponse)
async def delete_document(doc_title: str, request: Request):
    store = request.app.state.milvus_store
    file_path, deleted = await asyncio.to_thread(store.delete_by_doc_title, doc_title)
    if deleted == 0:
        raise HTTPException(404, f"文档不存在: {doc_title}")
    if file_path and os.path.isfile(file_path):
        os.remove(file_path)
    return DeleteResponse(message=f"已删除文档: {doc_title}")
