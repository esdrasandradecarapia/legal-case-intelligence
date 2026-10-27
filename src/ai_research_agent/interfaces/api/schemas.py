from pydantic import BaseModel


class IngestDocumentRequest(BaseModel):
    id: str
    name: str
    text: str


class IngestDocumentResponse(BaseModel):
    status: str


class AskQuestionRequest(BaseModel):
    question: str


class SourceResponse(BaseModel):
    document_id: str
    index: int
    text: str


class AskQuestionResponse(BaseModel):
    text: str
    sources: list[SourceResponse]