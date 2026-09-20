"""Experimental fixed-window and legal-structure chunking strategies."""

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from regagent.application.ingestion.models import (
    ChunkDraft,
    ChunkingStrategy,
    NormalizedSource,
    SectionKind,
    StructuredSection,
)


class WhitespaceTokenCounter:
    """Deterministic baseline counter; model tokenizers are added at indexing time."""

    @staticmethod
    def tokens(text: str) -> list[str]:
        return text.split()

    def count(self, text: str) -> int:
        return len(self.tokens(text))


@dataclass(frozen=True)
class _TokenWithSource:
    value: str
    section: StructuredSection


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ordered_unique(values: Iterable[UUID]) -> tuple[UUID, ...]:
    return tuple(dict.fromkeys(values))


class FixedWindowChunker:
    """Baseline that disregards legal boundaries and slides over the full document."""

    def __init__(self, max_tokens: int = 256, overlap_tokens: int = 32) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if overlap_tokens < 0 or overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be between 0 and max_tokens - 1")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self._counter = WhitespaceTokenCounter()

    def chunk(
        self,
        source: NormalizedSource,
        sections: tuple[StructuredSection, ...],
    ) -> tuple[ChunkDraft, ...]:
        del source
        flattened = [
            _TokenWithSource(token, section)
            for section in sections
            for token in self._counter.tokens(section.text)
        ]
        if not flattened:
            return ()

        step = self.max_tokens - self.overlap_tokens
        chunks: list[ChunkDraft] = []
        for start in range(0, len(flattened), step):
            window = flattened[start : start + self.max_tokens]
            if not window:
                break
            text = " ".join(item.value for item in window)
            chunks.append(
                ChunkDraft(
                    ordinal=len(chunks),
                    text=text,
                    content_hash=_content_hash(text),
                    token_count=len(window),
                    hierarchy_path=window[0].section.hierarchy_path,
                    source_section_ids=_ordered_unique(item.section.id for item in window),
                    strategy=ChunkingStrategy.FIXED_WINDOW,
                    config=self.config,
                )
            )
            if start + self.max_tokens >= len(flattened):
                break
        return tuple(chunks)

    @property
    def config(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "overlap_tokens": self.overlap_tokens,
            "token_counter": "whitespace_v1",
        }


class LegalStructureChunker:
    """Keep article boundaries while merging short adjacent legal blocks."""

    def __init__(self, max_tokens: int = 256, overlap_tokens: int = 32) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if overlap_tokens < 0 or overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be between 0 and max_tokens - 1")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self._counter = WhitespaceTokenCounter()

    def chunk(
        self,
        source: NormalizedSource,
        sections: tuple[StructuredSection, ...],
    ) -> tuple[ChunkDraft, ...]:
        del source
        groups = self._article_groups(sections)
        chunks: list[ChunkDraft] = []
        for group in groups:
            current: list[StructuredSection] = []
            current_tokens = 0
            for section in group:
                token_count = self._counter.count(section.text)
                if token_count > self.max_tokens:
                    if current:
                        chunks.append(self._build_chunk(len(chunks), current))
                        current = []
                        current_tokens = 0
                    chunks.extend(self._split_long_section(section, len(chunks)))
                    continue

                if current and current_tokens + token_count > self.max_tokens:
                    chunks.append(self._build_chunk(len(chunks), current))
                    current = []
                    current_tokens = 0

                current.append(section)
                current_tokens += token_count

            if current:
                chunks.append(self._build_chunk(len(chunks), current))

        return tuple(chunks)

    @staticmethod
    def _article_groups(
        sections: tuple[StructuredSection, ...],
    ) -> tuple[tuple[StructuredSection, ...], ...]:
        groups: list[list[StructuredSection]] = []
        current: list[StructuredSection] = []
        for section in sections:
            if section.kind is SectionKind.ARTICLE and current:
                groups.append(current)
                current = []
            current.append(section)
        if current:
            groups.append(current)
        return tuple(tuple(group) for group in groups)

    def _build_chunk(
        self,
        ordinal: int,
        sections: list[StructuredSection],
    ) -> ChunkDraft:
        text = "\n\n".join(section.text for section in sections)
        return ChunkDraft(
            ordinal=ordinal,
            text=text,
            content_hash=_content_hash(text),
            token_count=self._counter.count(text),
            hierarchy_path=sections[-1].hierarchy_path,
            source_section_ids=_ordered_unique(section.id for section in sections),
            strategy=ChunkingStrategy.LEGAL_STRUCTURE,
            config=self.config,
        )

    def _split_long_section(
        self,
        section: StructuredSection,
        first_ordinal: int,
    ) -> list[ChunkDraft]:
        tokens = self._counter.tokens(section.text)
        step = self.max_tokens - self.overlap_tokens
        chunks: list[ChunkDraft] = []
        for start in range(0, len(tokens), step):
            window = tokens[start : start + self.max_tokens]
            text = " ".join(window)
            chunks.append(
                ChunkDraft(
                    ordinal=first_ordinal + len(chunks),
                    text=text,
                    content_hash=_content_hash(text),
                    token_count=len(window),
                    hierarchy_path=section.hierarchy_path,
                    source_section_ids=(section.id,),
                    strategy=ChunkingStrategy.LEGAL_STRUCTURE,
                    config=self.config,
                )
            )
            if start + self.max_tokens >= len(tokens):
                break
        return chunks

    @property
    def config(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "overlap_tokens": self.overlap_tokens,
            "token_counter": "whitespace_v1",
            "boundary_policy": "never_cross_article_v1",
        }

