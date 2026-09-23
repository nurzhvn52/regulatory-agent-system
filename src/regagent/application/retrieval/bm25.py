"""Dependency-free Okapi BM25 baseline for multilingual legal text."""

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence

from regagent.application.retrieval.models import CorpusChunk
from regagent.application.retrieval.ports import RetrievalRepository
from regagent.domain.retrieval import RetrievalHit, RetrievalQuery

_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[-–—/][^\W_]+)*", flags=re.UNICODE)


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize Russian, Kazakh, English, and legal numbering without stemming."""

    return tuple(match.group(0).casefold() for match in _TOKEN_PATTERN.finditer(text))


class BM25Index:
    """Immutable in-memory Okapi BM25 index built for one explicit corpus snapshot."""

    def __init__(
        self,
        chunks: Sequence[CorpusChunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self._chunks = tuple(chunks)
        self._k1 = k1
        self._b = b
        self._term_frequencies = tuple(Counter(tokenize(chunk.text)) for chunk in chunks)
        self._lengths = tuple(sum(frequencies.values()) for frequencies in self._term_frequencies)
        self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        document_frequencies: dict[str, int] = defaultdict(int)
        for frequencies in self._term_frequencies:
            for term in frequencies:
                document_frequencies[term] += 1
        document_count = len(self._chunks)
        self._idf = {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }

    def search(self, query: str, top_k: int) -> list[tuple[CorpusChunk, float]]:
        query_terms = tokenize(query)
        if not query_terms or not self._chunks:
            return []
        scores: list[tuple[CorpusChunk, float]] = []
        for chunk, frequencies, length in zip(
            self._chunks,
            self._term_frequencies,
            self._lengths,
            strict=True,
        ):
            score = self._score(query_terms, frequencies, length)
            if score > 0:
                scores.append((chunk, score))
        scores.sort(
            key=lambda item: (
                -item[1],
                item[0].source_url,
                item[0].source_content_hash,
                item[0].ordinal,
            )
        )
        return scores[:top_k]

    def _score(
        self,
        query_terms: Sequence[str],
        frequencies: Counter[str],
        length: int,
    ) -> float:
        score = 0.0
        average_length = self._average_length or 1.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            denominator = frequency + self._k1 * (1 - self._b + self._b * length / average_length)
            score += self._idf.get(term, 0.0) * frequency * (self._k1 + 1) / denominator
        return score


class BM25Retriever:
    def __init__(self, repository: RetrievalRepository) -> None:
        self._repository = repository

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalHit]:
        chunks = await self._repository.list_chunks(query.filters)
        results = BM25Index(chunks).search(query.text, query.top_k)
        return [
            chunk.to_hit(score=score, rank=rank)
            for rank, (chunk, score) in enumerate(results, start=1)
        ]
