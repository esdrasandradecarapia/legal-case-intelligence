import pytest

from ai_research_agent.domain.chunking import Chunker
from ai_research_agent.domain.entities import Document


def test_short_text_produces_single_chunk(): ...


def test_overlap_between_consecutive_chunks():
    document = Document(id="1", name="a.txt", text="abcdefghij")
    chunker = Chunker(chunk_size=4, overlap=1)

    chunks = chunker.split(document)

    texts = [chunk.text for chunk in chunks]
    assert texts == ["abcd", "defg", "ghij"]


def test_empty_text_produces_no_chunks():
    document = Document(id="1", name="empty.txt", text="")
    chunker = Chunker(chunk_size=4, overlap=1)

    chunks = chunker.split(document)

    assert chunks == []


def test_chunk_size_must_be_positive():
    with pytest.raises(ValueError):
        Chunker(chunk_size=0, overlap=1)


def test_chunk_indexes_are_sequential():
    document = Document(id="1", name="a.txt", text="abcdefghij")
    chunker = Chunker(chunk_size=4, overlap=1)

    chunks = chunker.split(document)

    indexes = [chunk.index for chunk in chunks]

    assert indexes == [0, 1, 2]


def test_chunks_keep_document_id():
    document = Document(
        id="document-123",
        name="a.txt",
        text="abcdefghij",
    )
    chunker = Chunker(chunk_size=4, overlap=1)

    chunks = chunker.split(document)

    assert all(chunk.document_id == "document-123" for chunk in chunks)


def test_negative_chunk_size_is_invalid():
    with pytest.raises(ValueError):
        Chunker(chunk_size=-1, overlap=0)


def test_negative_overlap_is_invalid():
    with pytest.raises(ValueError):
        Chunker(chunk_size=4, overlap=-1)


def test_overlap_cannot_equal_chunk_size():
    with pytest.raises(ValueError):
        Chunker(chunk_size=4, overlap=4)


def test_overlap_cannot_be_greater_than_chunk_size():
    with pytest.raises(ValueError):
        Chunker(chunk_size=4, overlap=5)
