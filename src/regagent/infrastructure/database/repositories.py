"""PostgreSQL persistence for idempotent document ingestion."""

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.sql import Select

from regagent.application.ingestion.models import (
    ChunkDraft,
    ChunkingStrategy,
    IngestionPayload,
    IngestionResult,
    StructuredSection,
)
from regagent.infrastructure.database.models import (
    ChunkRecord,
    DocumentRecord,
    DocumentSectionRecord,
    DocumentVersionRecord,
)
from regagent.infrastructure.database.session import Database


class SqlAlchemyDocumentRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def store(self, payload: IngestionPayload) -> IngestionResult:
        async with self._database.session() as session, session.begin():
            document = await session.scalar(self._document_query(payload))
            if document is None:
                document = self._new_document(payload)
                session.add(document)
                await session.flush()

            version = await session.scalar(
                select(DocumentVersionRecord).where(
                    DocumentVersionRecord.document_id == document.id,
                    DocumentVersionRecord.content_hash == payload.version.content_hash,
                )
            )
            version_created = version is None
            if version is None:
                version = self._new_version(payload, document.id)
                session.add(version)
                await session.flush()

            processing_exists = bool(
                await session.scalar(
                    select(func.count())
                    .select_from(DocumentSectionRecord)
                    .where(
                        DocumentSectionRecord.version_id == version.id,
                        DocumentSectionRecord.pipeline_signature
                        == payload.pipeline_signature,
                    )
                )
            )
            if not processing_exists:
                session.add_all(
                    self._new_section(
                        section,
                        version.id,
                        payload.pipeline_signature,
                    )
                    for section in payload.sections
                )
                await session.flush()

            existing_strategies = set(
                (
                    await session.scalars(
                        select(ChunkRecord.chunking_strategy)
                        .where(
                            ChunkRecord.version_id == version.id,
                            ChunkRecord.pipeline_signature == payload.pipeline_signature,
                        )
                        .distinct()
                    )
                ).all()
            )
            chunks_to_store = [
                chunk
                for chunk in payload.chunks
                if chunk.strategy.value not in existing_strategies
            ]
            session.add_all(
                self._new_chunk(chunk, version.id, payload.pipeline_signature)
                for chunk in chunks_to_store
            )
            await session.flush()

            section_count = await session.scalar(
                select(func.count())
                .select_from(DocumentSectionRecord)
                .where(
                    DocumentSectionRecord.version_id == version.id,
                    DocumentSectionRecord.pipeline_signature == payload.pipeline_signature,
                )
            )
            chunk_count = await session.scalar(
                select(func.count())
                .select_from(ChunkRecord)
                .where(
                    ChunkRecord.version_id == version.id,
                    ChunkRecord.pipeline_signature == payload.pipeline_signature,
                )
            )
            strategies = tuple(
                ChunkingStrategy(value)
                for value in sorted(
                    set(existing_strategies)
                    | {chunk.strategy.value for chunk in chunks_to_store}
                )
            )

            return IngestionResult(
                document_id=document.id,
                version_id=version.id,
                pipeline_signature=payload.pipeline_signature,
                section_count=section_count or 0,
                chunk_count=chunk_count or 0,
                chunking_strategies=strategies,
                duplicate=(
                    not version_created and processing_exists and not chunks_to_store
                ),
            )

    @staticmethod
    def _document_query(payload: IngestionPayload) -> Select[tuple[DocumentRecord]]:
        document = payload.document
        conditions = [
            DocumentRecord.language == document.language.value,
            DocumentRecord.issuer == document.issuer,
            DocumentRecord.jurisdiction == document.jurisdiction,
        ]
        if document.official_number:
            conditions.append(DocumentRecord.official_number == document.official_number)
        else:
            conditions.append(DocumentRecord.title == document.title)
        return select(DocumentRecord).where(*conditions).limit(1)

    @staticmethod
    def _new_document(payload: IngestionPayload) -> DocumentRecord:
        document = payload.document
        return DocumentRecord(
            id=document.id,
            title=document.title,
            act_type=document.act_type.value,
            issuer=document.issuer,
            jurisdiction=document.jurisdiction,
            language=document.language.value,
            official_number=document.official_number,
        )

    @staticmethod
    def _new_version(
        payload: IngestionPayload,
        document_id: UUID,
    ) -> DocumentVersionRecord:
        version = payload.version
        return DocumentVersionRecord(
            id=version.id,
            document_id=document_id,
            adopted_at=version.adopted_at,
            effective_from=version.effective_from,
            effective_to=version.effective_to,
            version_label=version.version_label,
            content_hash=version.content_hash,
            source_url=str(version.source.url),
            source_publisher=version.source.publisher,
            source_identifier=version.source.source_identifier,
            retrieved_at=version.source.retrieved_at,
        )

    @staticmethod
    def _new_section(
        section: StructuredSection,
        version_id: UUID,
        pipeline_signature: str,
    ) -> DocumentSectionRecord:
        return DocumentSectionRecord(
            id=section.id,
            version_id=version_id,
            pipeline_signature=pipeline_signature,
            parent_id=section.parent_id,
            ordinal=section.ordinal,
            kind=section.kind.value,
            label=section.label,
            heading=section.heading,
            article=section.article,
            paragraph=section.paragraph,
            hierarchy_path=list(section.hierarchy_path),
            text=section.text,
            page=section.page,
        )

    @staticmethod
    def _new_chunk(
        chunk: ChunkDraft,
        version_id: UUID,
        pipeline_signature: str,
    ) -> ChunkRecord:
        return ChunkRecord(
            id=chunk.id,
            version_id=version_id,
            pipeline_signature=pipeline_signature,
            section_id=chunk.source_section_ids[0],
            ordinal=chunk.ordinal,
            text=chunk.text,
            content_hash=chunk.content_hash,
            token_count=chunk.token_count,
            hierarchy_path=list(chunk.hierarchy_path),
            source_section_ids=[str(value) for value in chunk.source_section_ids],
            chunking_strategy=chunk.strategy.value,
            chunking_config=chunk.config,
        )
