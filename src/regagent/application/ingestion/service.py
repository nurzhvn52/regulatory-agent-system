"""Application service orchestrating parsing, structure analysis, and persistence."""

import hashlib

from regagent.application.ingestion.models import (
    IngestionPayload,
    IngestionRequest,
    IngestionResult,
    NormalizedSource,
)
from regagent.application.ingestion.normalizer import TextNormalizer
from regagent.application.ingestion.ports import Chunker, DocumentRepository, SourceLoader
from regagent.application.ingestion.structure import LegalStructureParser
from regagent.domain.documents import Document, DocumentVersion, SourceReference


class IngestionService:
    def __init__(
        self,
        source_loader: SourceLoader,
        repository: DocumentRepository,
        chunkers: tuple[Chunker, ...],
        normalizer: TextNormalizer | None = None,
        structure_parser: LegalStructureParser | None = None,
    ) -> None:
        if not chunkers:
            raise ValueError("At least one chunker is required")
        self._source_loader = source_loader
        self._repository = repository
        self._chunkers = chunkers
        self._normalizer = normalizer or TextNormalizer()
        self._structure_parser = structure_parser or LegalStructureParser()

    async def ingest(self, request: IngestionRequest) -> IngestionResult:
        if not request.file_path.is_file():
            raise FileNotFoundError(request.file_path)

        parsed = self._source_loader.parse(request.file_path)
        normalized = self._normalizer.normalize(parsed)
        pipeline_signature = self._pipeline_signature(normalized)
        sections = self._structure_parser.parse(normalized, pipeline_signature)
        chunks = tuple(
            chunk
            for chunker in self._chunkers
            for chunk in chunker.chunk(normalized, sections)
        )

        document = Document(
            title=request.title or normalized.title,
            act_type=request.act_type,
            issuer=request.issuer,
            jurisdiction=request.jurisdiction,
            language=request.language,
            official_number=request.official_number,
        )
        source = SourceReference(
            url=request.source_url,
            retrieved_at=request.retrieved_at,
            publisher=request.source_publisher,
            source_identifier=request.source_identifier,
        )
        version = DocumentVersion(
            document_id=document.id,
            adopted_at=request.adopted_at,
            effective_from=request.effective_from,
            effective_to=request.effective_to,
            version_label=request.version_label,
            content_hash=normalized.content_hash,
            source=source,
        )
        return await self._repository.store(
            IngestionPayload(
                document=document,
                version=version,
                pipeline_signature=pipeline_signature,
                sections=sections,
                chunks=chunks,
            )
        )

    def _pipeline_signature(self, source: NormalizedSource) -> str:
        components = (
            f"parser={source.parser_name}:{source.parser_version}",
            f"normalizer={self._normalizer.VERSION}",
            f"structure={self._structure_parser.VERSION}",
        )
        return hashlib.sha256("|".join(components).encode("utf-8")).hexdigest()
