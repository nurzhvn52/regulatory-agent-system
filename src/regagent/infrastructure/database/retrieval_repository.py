"""PostgreSQL corpus loading, embedding persistence, and exact vector search."""

from collections.abc import Sequence
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Select, and_, cast, or_, select
from sqlalchemy.dialects.postgresql import insert

from regagent.application.retrieval.models import (
    CorpusChunk,
    EmbeddingModelSpec,
    EmbeddingVector,
)
from regagent.domain.documents import ActType, Language
from regagent.domain.retrieval import RetrievalFilters, RetrievalHit
from regagent.infrastructure.database.models import (
    ChunkRecord,
    DocumentRecord,
    DocumentSectionRecord,
    DocumentVersionRecord,
    EmbeddingRecord,
)
from regagent.infrastructure.database.session import Database


class SqlAlchemyRetrievalRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def list_chunks(self, filters: RetrievalFilters) -> list[CorpusChunk]:
        statement = (
            self._chunk_select()
            .where(*self._scope_conditions(filters))
            .order_by(
                DocumentRecord.id,
                DocumentVersionRecord.id,
                ChunkRecord.ordinal,
            )
        )
        async with self._database.session() as session:
            rows = (await session.execute(statement)).all()
            section_rows = (
                await session.scalars(
                    select(DocumentSectionRecord).where(
                        DocumentSectionRecord.version_id.in_({row[0].version_id for row in rows}),
                        DocumentSectionRecord.pipeline_signature == filters.pipeline_signature,
                    )
                )
            ).all()
        sections = {(s.version_id, str(s.id)): s.article for s in section_rows}
        chunks = []
        for row in rows:
            chunk = row[0]
            articles = tuple(
                dict.fromkeys(
                    article
                    for section_id in chunk.source_section_ids
                    if (article := sections.get((chunk.version_id, section_id))) is not None
                )
            )
            chunks.append(self._to_chunk(*row).model_copy(update={"articles": articles}))
        return chunks

    async def list_unembedded_chunks(
        self,
        filters: RetrievalFilters,
        model: EmbeddingModelSpec,
    ) -> list[CorpusChunk]:
        embedding_match = and_(
            EmbeddingRecord.chunk_id == ChunkRecord.id,
            EmbeddingRecord.model_name == model.name,
            EmbeddingRecord.model_revision == model.revision,
            EmbeddingRecord.normalized == model.normalized,
            EmbeddingRecord.embedding_config == model.config,
        )
        if model.dimensions is not None:
            embedding_match = and_(embedding_match, EmbeddingRecord.dimensions == model.dimensions)
        statement = (
            self._chunk_select()
            .outerjoin(EmbeddingRecord, embedding_match)
            .where(*self._scope_conditions(filters), EmbeddingRecord.id.is_(None))
            .order_by(DocumentRecord.id, DocumentVersionRecord.id, ChunkRecord.ordinal)
        )
        async with self._database.session() as session:
            rows = (await session.execute(statement)).all()
        return [self._to_chunk(*row) for row in rows]

    async def store_embeddings(
        self,
        model: EmbeddingModelSpec,
        embeddings: Sequence[EmbeddingVector],
    ) -> int:
        if not embeddings:
            return 0
        dimensions = len(embeddings[0].values)
        if model.dimensions is not None and dimensions != model.dimensions:
            raise ValueError("Embedding dimensions do not match the model specification")
        if any(len(item.values) != dimensions for item in embeddings):
            raise ValueError("Cannot persist embeddings with mixed dimensions")
        values = [
            {
                "chunk_id": item.chunk_id,
                "model_name": model.name,
                "model_revision": model.revision,
                "dimensions": dimensions,
                "normalized": model.normalized,
                "embedding_config": model.config,
                "embedding": item.values,
            }
            for item in embeddings
        ]
        statement = (
            insert(EmbeddingRecord)
            .values(values)
            .on_conflict_do_nothing(index_elements=["chunk_id", "model_name", "model_revision"])
            .returning(EmbeddingRecord.id)
        )
        async with self._database.session() as session, session.begin():
            inserted = (await session.scalars(statement)).all()
        return len(inserted)

    async def dense_search(
        self,
        filters: RetrievalFilters,
        model: EmbeddingModelSpec,
        query_vector: Sequence[float],
        top_k: int,
    ) -> list[RetrievalHit]:
        if not query_vector:
            raise ValueError("Query embedding cannot be empty")
        dimensions = len(query_vector)
        if model.dimensions is not None and dimensions != model.dimensions:
            raise ValueError("Query dimensions do not match the model specification")
        distance = cast(EmbeddingRecord.embedding, Vector(dimensions)).cosine_distance(
            list(query_vector)
        )
        statement = (
            self._chunk_select(distance.label("distance"))
            .join(EmbeddingRecord, EmbeddingRecord.chunk_id == ChunkRecord.id)
            .where(
                *self._scope_conditions(filters),
                EmbeddingRecord.model_name == model.name,
                EmbeddingRecord.model_revision == model.revision,
                EmbeddingRecord.dimensions == dimensions,
                EmbeddingRecord.normalized == model.normalized,
                EmbeddingRecord.embedding_config == model.config,
            )
            .order_by(
                distance,
                DocumentVersionRecord.source_url,
                DocumentVersionRecord.content_hash,
                ChunkRecord.ordinal,
            )
            .limit(top_k)
        )
        async with self._database.session() as session:
            rows = (await session.execute(statement)).all()
        return [
            self._to_chunk(chunk, section, version, document).to_hit(
                score=1.0 - float(row_distance),
                rank=rank,
            )
            for rank, (chunk, section, version, document, row_distance) in enumerate(
                rows,
                start=1,
            )
        ]

    @staticmethod
    def _chunk_select(*extra_columns: Any) -> Select[Any]:
        return (
            select(
                ChunkRecord,
                DocumentSectionRecord,
                DocumentVersionRecord,
                DocumentRecord,
                *extra_columns,
            )
            .join(
                DocumentSectionRecord,
                DocumentSectionRecord.id == ChunkRecord.section_id,
            )
            .join(
                DocumentVersionRecord,
                DocumentVersionRecord.id == ChunkRecord.version_id,
            )
            .join(DocumentRecord, DocumentRecord.id == DocumentVersionRecord.document_id)
        )

    @staticmethod
    def _scope_conditions(filters: RetrievalFilters) -> list[Any]:
        conditions: list[Any] = [
            ChunkRecord.pipeline_signature == filters.pipeline_signature,
            ChunkRecord.chunking_strategy == filters.chunking_strategy,
            DocumentSectionRecord.pipeline_signature == filters.pipeline_signature,
        ]
        if filters.languages:
            conditions.append(
                DocumentRecord.language.in_(language.value for language in filters.languages)
            )
        if filters.act_types:
            conditions.append(
                DocumentRecord.act_type.in_(act_type.value for act_type in filters.act_types)
            )
        if filters.document_ids:
            conditions.append(DocumentRecord.id.in_(filters.document_ids))
        if filters.version_ids:
            conditions.append(DocumentVersionRecord.id.in_(filters.version_ids))
        if filters.effective_on:
            conditions.extend(
                [
                    DocumentVersionRecord.effective_from.is_not(None),
                    DocumentVersionRecord.effective_from <= filters.effective_on,
                    or_(
                        DocumentVersionRecord.effective_to.is_(None),
                        DocumentVersionRecord.effective_to >= filters.effective_on,
                    ),
                ]
            )
        return conditions

    @staticmethod
    def _to_chunk(
        chunk: ChunkRecord,
        section: DocumentSectionRecord,
        version: DocumentVersionRecord,
        document: DocumentRecord,
    ) -> CorpusChunk:
        return CorpusChunk(
            chunk_id=chunk.id,
            document_id=document.id,
            version_id=version.id,
            text=chunk.text,
            title=document.title,
            language=Language(document.language),
            act_type=ActType(document.act_type),
            article=section.article,
            paragraph=section.paragraph,
            source_url=version.source_url,
            ordinal=chunk.ordinal,
            source_content_hash=version.content_hash,
            chunking_config=chunk.chunking_config,
        )
