from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    id: str
    title: str
    chunks: int
    message: str


class DocumentListResponse(BaseModel):
    documents: list[str]


class DeleteResponse(BaseModel):
    message: str


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


