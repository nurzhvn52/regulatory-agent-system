"""Persistence boundary for retrieval and embedding indexing."""

from collections.abc import Sequence
from typing import Protocol

from regagent.application.retrieval.models import (
    CorpusChunk,
    EmbeddingModelSpec,
    EmbeddingVector,
)
from regagent.domain.retrieval import RetrievalFilters, RetrievalHit


class RetrievalRepository(Protocol):
    async def list_chunks(self, filters: RetrievalFilters) -> list[CorpusChunk]: ...

    async def list_unembedded_chunks(
        self,
        filters: RetrievalFilters,
        model: EmbeddingModelSpec,
    ) -> list[CorpusChunk]: ...

    async def store_embeddings(
        self,
        model: EmbeddingModelSpec,
        embeddings: Sequence[EmbeddingVector],
    ) -> int: ...

    async def dense_search(
        self,
        filters: RetrievalFilters,
        model: EmbeddingModelSpec,
        query_vector: Sequence[float],
        top_k: int,
    ) -> list[RetrievalHit]: ...
