from fastapi import FastAPI

from ai_research_agent.application.agent.router import AgentRouter
from ai_research_agent.application.ask_question import AskQuestion
from ai_research_agent.application.ingest_document import IngestDocument
from ai_research_agent.domain.chunking import Chunker
from ai_research_agent.domain.entities import Document
from ai_research_agent.infrastructure.config import Settings
from ai_research_agent.infrastructure.order_api_adapter import FakeOrderRepository
from ai_research_agent.infrastructure.pgvector_store import PgVectorStore
from ai_research_agent.infrastructure.providers import build_embedder, build_llm
from ai_research_agent.interfaces.api.schemas import (
    AskQuestionRequest,
    AskQuestionResponse,
    IngestDocumentRequest,
    IngestDocumentResponse,
    SourceResponse,
)

app = FastAPI(title="Ecommerce Support Agent")

settings = Settings()
embedder = build_embedder(settings)
llm = build_llm(settings)
vector_store = PgVectorStore(
    host=settings.database_host,
    port=settings.database_port,
    user=settings.database_user,
    password=settings.database_password,
    database=settings.database_name,
)
chunker = Chunker(chunk_size=200, overlap=20)
order_repository = FakeOrderRepository()

ask_question = AskQuestion(embedder=embedder, vector_store=vector_store, llm=llm)
agent_router = AgentRouter(
    tool_calling_llm=llm,
    ask_question=ask_question,
    order_repository=order_repository,
)


@app.post("/documents", response_model=IngestDocumentResponse)
def ingest_document(request: IngestDocumentRequest) -> IngestDocumentResponse:
    use_case = IngestDocument(chunker=chunker, embedder=embedder, vector_store=vector_store)

    document = Document(id=request.id, name=request.name, text=request.text)
    use_case.execute(document)

    return IngestDocumentResponse(status="ingested")


@app.post("/ask", response_model=AskQuestionResponse)
def ask_question(request: AskQuestionRequest) -> AskQuestionResponse:
    answer = agent_router.route(request.question)

    return AskQuestionResponse(
        text=answer.text,
        sources=[
            SourceResponse(document_id=s.document_id, index=s.index, text=s.text)
            for s in answer.sources
        ],
    )