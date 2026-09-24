from datetime import date
from uuid import UUID

from sqlalchemy import and_
from sqlalchemy.dialects import postgresql

from regagent.domain.retrieval import RetrievalFilters
from regagent.infrastructure.database.retrieval_repository import SqlAlchemyRetrievalRepository


def test_date_scoped_search_excludes_unknown_effective_from() -> None:
    filters = RetrievalFilters(
        pipeline_signature="a" * 64,
        effective_on=date(2026, 1, 1),
        version_ids=(UUID(int=3),),
    )
    conditions = SqlAlchemyRetrievalRepository._scope_conditions(filters)
    sql = str(and_(*conditions).compile(dialect=postgresql.dialect()))

    assert "document_versions.effective_from IS NOT NULL" in sql
    assert "document_versions.effective_from <= " in sql
    assert "document_versions.id IN " in sql
