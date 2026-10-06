from dataclasses import FrozenInstanceError

import pytest

from ai_research_agent.domain.entities import Chunk, Document


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
