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
    history: list["ChatTurn"] | None = None
    session_id: str | None = None


class ChatTurn(BaseModel):
    role: str
    content: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


