"""Interfaces at the ingestion boundary."""

from pathlib import Path
from typing import Protocol

from regagent.application.ingestion.models import (
    ChunkDraft,
    IngestionPayload,
    IngestionResult,
    NormalizedSource,
    ParsedSource,
    StructuredSection,
)


class SourceLoader(Protocol):
    def parse(self, path: Path) -> ParsedSource: ...


class DocumentRepository(Protocol):
    async def store(self, payload: IngestionPayload) -> IngestionResult: ...


class Chunker(Protocol):
    def chunk(
        self,
        source: NormalizedSource,
        sections: tuple[StructuredSection, ...],
    ) -> tuple[ChunkDraft, ...]: ...

