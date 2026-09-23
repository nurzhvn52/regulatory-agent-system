"""Information-retrieval metrics and an evaluation runner."""

import math
from collections.abc import Iterable, Sequence
from statistics import fmean
from uuid import UUID

from regagent.application.ports import Retriever
from regagent.application.retrieval.models import (
    AggregateEvaluation,
    EvaluationResult,
    RetrievalEvaluationCase,
    RetrievalMetrics,
)
from regagent.domain.retrieval import RetrievalQuery


def evaluate_ranking(
    retrieved_ids: Sequence[UUID],
    relevant_ids: Iterable[UUID],
    *,
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> RetrievalMetrics:
    relevant = frozenset(relevant_ids)
    if not relevant:
        raise ValueError("At least one relevant chunk is required")
    if not k_values or any(value < 1 for value in k_values):
        raise ValueError("k_values must contain positive integers")
    recall = {
        k: len(relevant.intersection(retrieved_ids[:k])) / len(relevant) for k in k_values
    }
    reciprocal_rank = next(
        (
            1.0 / rank
            for rank, chunk_id in enumerate(retrieved_ids, start=1)
            if chunk_id in relevant
        ),
        0.0,
    )
    ndcg = {k: _ndcg_at_k(retrieved_ids, relevant, k) for k in k_values}
    return RetrievalMetrics(
        recall_at_k=recall,
        reciprocal_rank=reciprocal_rank,
        ndcg_at_k=ndcg,
    )


def _ndcg_at_k(retrieved_ids: Sequence[UUID], relevant: frozenset[UUID], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved_ids[:k], start=1)
        if chunk_id in relevant
    )
    ideal_count = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal_dcg


class RetrievalEvaluator:
    def __init__(
        self,
        retriever: Retriever,
        *,
        k_values: Sequence[int] = (1, 3, 5, 10),
    ) -> None:
        self._retriever = retriever
        self._k_values = tuple(k_values)

    async def evaluate(
        self,
        cases: Sequence[RetrievalEvaluationCase],
    ) -> AggregateEvaluation:
        results: list[EvaluationResult] = []
        max_k = max(self._k_values)
        for case in cases:
            hits = await self._retriever.retrieve(
                RetrievalQuery(text=case.query, top_k=max_k, filters=case.filters)
            )
            metrics = evaluate_ranking(
                [hit.chunk_id for hit in hits],
                case.relevant_chunk_ids,
                k_values=self._k_values,
            )
            results.append(
                EvaluationResult(case_id=case.case_id, metrics=metrics, hits=tuple(hits))
            )
        return AggregateEvaluation(
            case_count=len(results),
            mean_recall_at_k={
                k: _mean(result.metrics.recall_at_k[k] for result in results)
                for k in self._k_values
            },
            mean_reciprocal_rank=_mean(
                result.metrics.reciprocal_rank for result in results
            ),
            mean_ndcg_at_k={
                k: _mean(result.metrics.ndcg_at_k[k] for result in results)
                for k in self._k_values
            },
            cases=tuple(results),
        )


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(values)
    return fmean(materialized) if materialized else 0.0
