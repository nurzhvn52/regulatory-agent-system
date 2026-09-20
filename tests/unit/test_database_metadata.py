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
