import dataclasses
from dataclasses import FrozenInstanceError

import pytest

from ai_research_agent.domain.entities import Answer, Chunk, Document, RetrievedChunk


def test_document_is_immutable():
    document = Document(
        id="1",
        name="case.txt",
        text="content",
    )

    with pytest.raises(FrozenInstanceError):
        document.text = "changed"


def test_chunk_is_immutable():
    chunk = Chunk(
        document_id="1",
        index=0,
        text="content",
    )

    with pytest.raises(FrozenInstanceError):
        chunk.text = "changed"


def test_retrieved_chunk_holds_chunk_and_score():
    chunk = Chunk(document_id="document-1", index=0, text="abcd")
    retrieved = RetrievedChunk(chunk=chunk, score=0.87)

    assert retrieved.chunk == chunk
    assert retrieved.score == 0.87


def test_retrieved_chunk_is_immutable():
    chunk = Chunk(document_id="document-1", index=0, text="abcd")
    retrieved = RetrievedChunk(chunk=chunk, score=0.87)

    with pytest.raises(dataclasses.FrozenInstanceError):
        retrieved.score = 0.5


def test_answer_holds_text_and_sources():
    chunk = Chunk(document_id="document-1", index=0, text="abcd")
    answer = Answer(text="Resposta gerada.", sources=[chunk])

    assert answer.text == "Resposta gerada."
    assert answer.sources == [chunk]


def test_answer_is_immutable():
    chunk = Chunk(document_id="document-1", index=0, text="abcd")
    answer = Answer(text="Resposta gerada.", sources=[chunk])

    with pytest.raises(dataclasses.FrozenInstanceError):
        answer.text = "Outra coisa"
