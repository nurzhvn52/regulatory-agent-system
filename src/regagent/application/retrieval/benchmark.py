"""Portable benchmark labels resolved against an explicit corpus snapshot."""

import hashlib
import json
from collections.abc import Sequence
from datetime import date
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from regagent.application.retrieval.models import CorpusChunk, RetrievalEvaluationCase
from regagent.domain.documents import Language
from regagent.domain.retrieval import RetrievalFilters


class BenchmarkSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str = Field(min_length=1)
    language: Language
    source_url: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class ArticleLabel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    article: str = Field(min_length=1)


class BenchmarkCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(min_length=1)
    pair_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    language: Language
    relevant_articles: tuple[ArticleLabel, ...] = Field(min_length=1)
    review_status: Literal["draft", "reviewed"] = "draft"
    reviewer: str | None = None
    reviewed_at: date | None = None

    @model_validator(mode="after")
    def check_review(self) -> "BenchmarkCase":
        if not self.query.strip():
            raise ValueError("Query must not be blank")
        if self.review_status == "reviewed" and (
            not self.reviewer or not self.reviewer.strip() or self.reviewed_at is None
        ):
            raise ValueError("Reviewed cases require reviewer and reviewed_at")
        return self


class BenchmarkDataset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str = Field(min_length=1)
    schema_version: Literal["1"] = "1"
    description: str
    sources: tuple[BenchmarkSource, ...] = Field(min_length=1)
    cases: tuple[BenchmarkCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def check_references(self) -> "BenchmarkDataset":
        sources = {source.source_id: source for source in self.sources}
        if len(sources) != len(self.sources):
            raise ValueError("Source IDs must be unique")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("Case IDs must be unique")
        for case in self.cases:
            for label in case.relevant_articles:
                if label.source_id not in sources:
                    raise ValueError(f"Unknown source {label.source_id} in {case.case_id}")
                if sources[label.source_id].language != case.language:
                    raise ValueError("This benchmark uses monolingual retrieval per case")
        return self


class ResolvedBenchmark(BaseModel):
    model_config = ConfigDict(frozen=True)

    cases: tuple[RetrievalEvaluationCase, ...]
    chunks: tuple[CorpusChunk, ...]
    corpus_sha256: str
    stable_chunk_keys: dict[UUID, str]
    contains_drafts: bool


def stable_chunk_key(chunk: CorpusChunk) -> str:
    return hashlib.sha256(
        json.dumps(
            [
                chunk.source_url,
                chunk.source_content_hash,
                chunk.language.value,
                chunk.ordinal,
                chunk.text,
                chunk.articles,
                chunk.chunking_config,
            ],
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def resolve_benchmark(
    dataset: BenchmarkDataset,
    chunks: Sequence[CorpusChunk],
    scope: RetrievalFilters,
    *,
    allow_draft: bool = False,
) -> ResolvedBenchmark:
    contains_drafts = any(case.review_status == "draft" for case in dataset.cases)
    if contains_drafts and not allow_draft:
        raise ValueError("Dataset has unreviewed cases; use --allow-draft for a pilot run")
    source_chunks: dict[str, list[CorpusChunk]] = {}
    for source in dataset.sources:
        matching = [
            chunk
            for chunk in chunks
            if (
                chunk.source_url == source.source_url
                and chunk.source_content_hash == source.content_hash
                and chunk.language == source.language
            )
        ]
        if not matching:
            raise ValueError(f"Missing pinned source/pipeline/chunking: {source.source_id}")
        if len({chunk.version_id for chunk in matching}) != 1:
            raise ValueError(f"Ambiguous source version: {source.source_id}")
        source_chunks[source.source_id] = matching
    selected = {c.chunk_id: c for group in source_chunks.values() for c in group}
    cases = []
    for case in dataset.cases:
        relevant: set[UUID] = set()
        for label in case.relevant_articles:
            matches = {
                c.chunk_id for c in source_chunks[label.source_id] if label.article in c.articles
            }
            if not matches:
                raise ValueError(f"Unresolved article {label.article} in {case.case_id}")
            relevant.update(matches)
        versions = tuple(
            sorted(
                {c.version_id for c in selected.values() if c.language == case.language}, key=str
            )
        )
        filters = scope.model_copy(
            update={
                "languages": (case.language,),
                "version_ids": versions,
            }
        )
        cases.append(
            RetrievalEvaluationCase(
                case_id=case.case_id,
                query=case.query,
                relevant_chunk_ids=frozenset(relevant),
                filters=filters,
            )
        )
    keys = {chunk_id: stable_chunk_key(c) for chunk_id, c in selected.items()}
    fingerprint = hashlib.sha256(
        json.dumps(
            [scope.pipeline_signature, scope.chunking_strategy, sorted(keys.values())],
        ).encode("utf-8")
    ).hexdigest()
    return ResolvedBenchmark(
        cases=tuple(cases),
        chunks=tuple(selected.values()),
        corpus_sha256=fingerprint,
        stable_chunk_keys=keys,
        contains_drafts=contains_drafts,
    )
