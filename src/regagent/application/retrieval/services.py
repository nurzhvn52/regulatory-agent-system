"""Dense indexing, dense retrieval, and reciprocal-rank fusion."""

import asyncio
from collections import defaultdict
from collections.abc import Sequence

from regagent.application.ports import EmbeddingProvider, Retriever
from regagent.application.retrieval.models import EmbeddingVector, IndexingResult
from regagent.application.retrieval.ports import RetrievalRepository
from regagent.domain.retrieval import RetrievalFilters, RetrievalHit, RetrievalQuery


class EmbeddingIndexService:
    def __init__(
        self,
        repository: RetrievalRepository,
        provider: EmbeddingProvider,
        *,
        batch_size: int = 8,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._repository = repository
        self._provider = provider
        self._batch_size = batch_size

    async def index(self, filters: RetrievalFilters) -> IndexingResult:
        model = self._provider.model_spec
        all_chunks = await self._repository.list_chunks(filters)
        missing = await self._repository.list_unembedded_chunks(filters, model)
        indexed = 0
        dimensions = 0
        for start in range(0, len(missing), self._batch_size):
            batch = missing[start : start + self._batch_size]
            values = await self._provider.embed_documents([chunk.text for chunk in batch])
            if len(values) != len(batch):
                raise ValueError("Embedding provider returned an unexpected vector count")
            if values:
                dimensions = len(values[0])
            vectors = []
            for chunk, vector in zip(batch, values, strict=True):
                if not vector or len(vector) != dimensions:
                    raise ValueError("Embedding provider returned inconsistent dimensions")
                vectors.append(EmbeddingVector(chunk_id=chunk.chunk_id, values=vector))
            indexed += await self._repository.store_embeddings(model, vectors)
        return IndexingResult(
            model=model,
            dimensions=dimensions or self._provider.dimensions,
            discovered_chunks=len(all_chunks),
            indexed_chunks=indexed,
            skipped_chunks=len(all_chunks) - len(missing),
        )


class DenseRetriever:
    def __init__(
        self,
        repository: RetrievalRepository,
        provider: EmbeddingProvider,
    ) -> None:
        self._repository = repository
        self._provider = provider

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalHit]:
        vector = await self._provider.embed_query(query.text)
        return await self._repository.dense_search(
            query.filters,
            self._provider.model_spec,
            vector,
            query.top_k,
        )


class ReciprocalRankFusionRetriever:
    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        rank_constant: int = 60,
        candidate_multiplier: int = 4,
    ) -> None:
        if len(retrievers) < 2:
            raise ValueError("RRF requires at least two retrievers")
        if rank_constant < 1 or candidate_multiplier < 1:
            raise ValueError("RRF parameters must be positive")
        self._retrievers = tuple(retrievers)
        self._rank_constant = rank_constant
        self._candidate_multiplier = candidate_multiplier

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalHit]:
        candidate_query = query.model_copy(
            update={"top_k": min(100, query.top_k * self._candidate_multiplier)}
        )
        rankings = await asyncio.gather(
            *(retriever.retrieve(candidate_query) for retriever in self._retrievers)
        )
        return reciprocal_rank_fusion(
            rankings,
            top_k=query.top_k,
            rank_constant=self._rank_constant,
        )


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievalHit]],
    *,
    top_k: int,
    rank_constant: int = 60,
) -> list[RetrievalHit]:
    if top_k < 1 or rank_constant < 1:
        raise ValueError("top_k and rank_constant must be positive")
    scores: dict[object, float] = defaultdict(float)
    hits: dict[object, RetrievalHit] = {}
    for ranking in rankings:
        seen: set[object] = set()
        for position, hit in enumerate(ranking, start=1):
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            scores[hit.chunk_id] += 1.0 / (rank_constant + position)
            hits.setdefault(hit.chunk_id, hit)
    ordered_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], str(chunk_id)))
    return [
        hits[chunk_id].model_copy(update={"score": scores[chunk_id], "rank": rank})
        for rank, chunk_id in enumerate(ordered_ids[:top_k], start=1)
    ]
