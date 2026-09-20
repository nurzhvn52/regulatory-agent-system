from datetime import date
from pathlib import Path
from uuid import uuid4

import pytest

from regagent.application.ingestion.chunking import FixedWindowChunker, LegalStructureChunker
from regagent.application.ingestion.models import (
    ChunkingStrategy,
    IngestionPayload,
    IngestionRequest,
    IngestionResult,
)
from regagent.application.ingestion.service import IngestionService
from regagent.domain.documents import ActType, Language
from regagent.infrastructure.parsers import ParserRegistry


class InMemoryRepository:
    def __init__(self) -> None:
        self.payload: IngestionPayload | None = None

    async def store(self, payload: IngestionPayload) -> IngestionResult:
        self.payload = payload
        return IngestionResult(
            document_id=payload.document.id,
            version_id=payload.version.id,
            pipeline_signature=payload.pipeline_signature,
            section_count=len(payload.sections),
            chunk_count=len(payload.chunks),
            chunking_strategies=tuple(
                sorted({chunk.strategy for chunk in payload.chunks}, key=str)
            ),
            duplicate=False,
        )


@pytest.mark.asyncio
async def test_service_builds_both_experimental_chunk_sets() -> None:
    repository = InMemoryRepository()
    service = IngestionService(
        source_loader=ParserRegistry(),
        repository=repository,
        chunkers=(
            FixedWindowChunker(max_tokens=20, overlap_tokens=4),
            LegalStructureChunker(max_tokens=20, overlap_tokens=4),
        ),
    )
    path = Path(__file__).parents[1] / "fixtures" / "sample_regulation_ru.html"

    result = await service.ingest(
        IngestionRequest(
            file_path=path,
            language=Language.RU,
            act_type=ActType.REGULATION,
            issuer="Тестовый орган",
            source_url="https://example.gov.kz/test-act",
            source_publisher="Тестовый официальный источник",
            retrieved_at=date(2026, 9, 20),
            official_number=str(uuid4()),
        )
    )

    assert repository.payload is not None
    assert set(result.chunking_strategies) == {
        ChunkingStrategy.FIXED_WINDOW,
        ChunkingStrategy.LEGAL_STRUCTURE,
    }
    assert result.section_count > 0
    assert result.chunk_count > 0
