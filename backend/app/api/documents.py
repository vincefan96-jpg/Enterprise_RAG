import os
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from app.models.schemas import DocumentUploadResponse, DocumentListResponse, DeleteResponse
from app.services.document_parser import DocumentParser
from app.services.chunker import DocumentChunker
from app.config import get_settings

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)):
    settings = get_settings()

    ext = Path(file.filename).suffix.lower()
    if ext not in (".pdf", ".docx", ".doc", ".txt"):
        raise HTTPException(400, f"不支持的文件类型: {ext}")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(400, f"文件超过 {settings.max_upload_size_mb}MB 限制")

    os.makedirs(settings.upload_dir, exist_ok=True)
    file_id = uuid.uuid4().hex
    save_path = os.path.join(settings.upload_dir, f"{file_id}{ext}")
    with open(save_path, "wb") as f:
        f.write(contents)

    try:
        parser = DocumentParser()
        text = parser.parse(save_path)
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(500, f"文档解析失败: {str(e)}")

    chunker = DocumentChunker(
        parent_size=settings.parent_chunk_size,
        child_size=settings.child_chunk_size,
        overlap=settings.chunk_overlap,
    )
    chunks = chunker.split(text, file.filename)
    for c in chunks:
        c.source_type = ext.lstrip(".")

    embed_service = request.app.state.embedding_service
    texts = [c.text for c in chunks]
    embeddings = embed_service.encode_documents(texts)
    dense_vecs = [e["dense"] for e in embeddings]
    sparse_vecs = [e["sparse"] for e in embeddings]

    store = request.app.state.milvus_store
    store.insert(chunks, dense_vecs, sparse_vecs)

    return DocumentUploadResponse(
        id=file_id,
        title=file.filename,
        chunks=len(chunks),
        message=f"成功上传并索引 {len(chunks)} 个文本块",
    )


@router.get("/list", response_model=DocumentListResponse)
async def list_documents(request: Request):
    store = request.app.state.milvus_store
    docs = store.list_documents()
    return DocumentListResponse(documents=docs)


@router.delete("/{doc_title:path}", response_model=DeleteResponse)
async def delete_document(doc_title: str, request: Request):
    store = request.app.state.milvus_store
    store.delete_by_doc_title(doc_title)
    return DeleteResponse(message=f"已删除文档: {doc_title}")
