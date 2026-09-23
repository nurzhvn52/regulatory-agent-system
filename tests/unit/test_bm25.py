from uuid import UUID

from regagent.application.retrieval.bm25 import BM25Index, tokenize
from regagent.application.retrieval.models import CorpusChunk
from regagent.domain.documents import ActType, Language


def _chunk(number: int, text: str) -> CorpusChunk:
    return CorpusChunk(
        chunk_id=UUID(int=number),
        document_id=UUID(int=100),
        version_id=UUID(int=200),
        text=text,
        title="Закон",
        language=Language.RU,
        act_type=ActType.LAW,
        article=f"Статья {number}",
        source_url="https://example.test/law",
        ordinal=number,
    )


def test_tokenizer_preserves_cyrillic_kazakh_and_legal_numbers() -> None:
    assert tokenize("Статья 74-1. Құқық және міндет") == (
        "статья",
        "74-1",
        "құқық",
        "және",
        "міндет",
    )


def test_bm25_ranks_term_specific_legal_chunk_first() -> None:
    chunks = [
        _chunk(1, "Общие положения и определения закона"),
        _chunk(2, "Персональные данные подлежат защите оператором"),
        _chunk(3, "Порядок государственного контроля"),
    ]

    hits = BM25Index(chunks).search("защите персональные данные", top_k=2)

    assert [chunk.chunk_id for chunk, _ in hits] == [UUID(int=2)]
    assert hits[0][1] > 0


def test_bm25_empty_or_unknown_query_has_no_hits() -> None:
    index = BM25Index([_chunk(1, "известный термин")])

    assert index.search("", top_k=10) == []
    assert index.search("несуществующий", top_k=10) == []
