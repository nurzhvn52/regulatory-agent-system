from uuid import UUID

from regagent.application.retrieval.evaluation import evaluate_ranking
from regagent.application.retrieval.services import reciprocal_rank_fusion
from regagent.domain.documents import Language
from regagent.domain.retrieval import RetrievalHit


def _hit(number: int, rank: int, score: float = 1.0) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=UUID(int=number),
        document_id=UUID(int=100),
        version_id=UUID(int=200),
        text=f"chunk {number}",
        score=score,
        rank=rank,
        title="Закон",
        language=Language.RU,
        source_url="https://example.test/law",
    )


def test_rrf_rewards_chunks_found_by_both_channels() -> None:
    fused = reciprocal_rank_fusion(
        [
            [_hit(1, 1), _hit(2, 2)],
            [_hit(2, 1), _hit(3, 2)],
        ],
        top_k=3,
        rank_constant=60,
    )

    assert [hit.chunk_id for hit in fused] == [UUID(int=2), UUID(int=1), UUID(int=3)]
    assert [hit.rank for hit in fused] == [1, 2, 3]


def test_recall_mrr_and_ndcg_use_binary_relevance() -> None:
    metrics = evaluate_ranking(
        [UUID(int=3), UUID(int=2), UUID(int=1)],
        {UUID(int=1), UUID(int=2)},
        k_values=(1, 2, 3),
    )

    assert metrics.recall_at_k == {1: 0.0, 2: 0.5, 3: 1.0}
    assert metrics.reciprocal_rank == 0.5
    assert metrics.ndcg_at_k[1] == 0.0
    assert 0 < metrics.ndcg_at_k[3] < 1
