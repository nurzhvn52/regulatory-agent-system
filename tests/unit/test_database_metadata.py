from regagent.infrastructure.database import models  # noqa: F401
from regagent.infrastructure.database.base import Base


def test_initial_metadata_contains_research_tables() -> None:
    assert set(Base.metadata.tables) == {
        "documents",
        "document_versions",
        "document_sections",
        "chunks",
        "embeddings",
        "agent_specs",
        "agent_runs",
    }


def test_embedding_column_allows_multiple_model_dimensions() -> None:
    embedding_type = Base.metadata.tables["embeddings"].c.embedding.type

    assert embedding_type.dim is None


def test_embeddings_record_reproducible_model_provenance() -> None:
    embeddings = Base.metadata.tables["embeddings"]

    assert "model_revision" in embeddings.c
    assert "normalized" in embeddings.c
    assert "embedding_config" in embeddings.c


def test_chunks_record_strategy_and_multi_section_provenance() -> None:
    chunks = Base.metadata.tables["chunks"]

    assert "source_section_ids" in chunks.c
    assert "chunking_strategy" in chunks.c
    assert "chunking_config" in chunks.c


def test_sections_preserve_detected_legal_semantics() -> None:
    sections = Base.metadata.tables["document_sections"]

    assert "kind" in sections.c
    assert "label" in sections.c
    assert "page" in sections.c
    assert "pipeline_signature" in sections.c
    assert "pipeline_signature" in Base.metadata.tables["chunks"].c
