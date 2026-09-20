"""Rule-based recognition of Russian and Kazakh legal document structure."""

import re
from dataclasses import dataclass
from typing import ClassVar
from uuid import NAMESPACE_URL, uuid5

from regagent.application.ingestion.models import (
    NormalizedSource,
    SectionKind,
    StructuredSection,
)


@dataclass(frozen=True)
class _DetectedStructure:
    kind: SectionKind
    level: int
    label: str | None
    heading: str | None


class LegalStructureParser:
    """Build a stable hierarchy from common RU/KK legal headings and numbering."""

    VERSION = "ru_kk_legal_structure_v1"

    _patterns: ClassVar[tuple[tuple[SectionKind, int, re.Pattern[str]], ...]] = (
        (
            SectionKind.PART,
            1,
            re.compile(
                r"^(?:раздел|часть|бөлім|бөлiм|бөлік)\s+"
                r"(?P<label>[IVXLCDM\d]+)\s*[.\-–—:]?\s*(?P<title>.*)$",
                re.IGNORECASE,
            ),
        ),
        (
            SectionKind.PART,
            1,
            re.compile(
                r"^(?P<label>[IVXLCDM\d]+)\s*[-–—]\s*"
                r"(?:бөлім|бөлiм|бөлік)\s*[.:]?\s*(?P<title>.*)$",
                re.IGNORECASE,
            ),
        ),
        (
            SectionKind.CHAPTER,
            2,
            re.compile(
                r"^(?:глава|тарау)\s+(?P<label>[IVXLCDM\d]+)"
                r"\s*[.\-–—:]?\s*(?P<title>.*)$",
                re.IGNORECASE,
            ),
        ),
        (
            SectionKind.CHAPTER,
            2,
            re.compile(
                r"^(?P<label>[IVXLCDM\d]+)\s*[-–—]\s*тарау"
                r"\s*[.:]?\s*(?P<title>.*)$",
                re.IGNORECASE,
            ),
        ),
        (
            SectionKind.ARTICLE,
            3,
            re.compile(
                r"^(?:статья|бап)\s+(?P<label>\d+(?:-\d+)*)"
                r"\s*[.]?\s*(?P<title>.*)$",
                re.IGNORECASE,
            ),
        ),
        (
            SectionKind.ARTICLE,
            3,
            re.compile(
                r"^(?P<label>\d+(?:-\d+)*)\s*[-–—]\s*бап"
                r"\s*[.]?\s*(?P<title>.*)$",
                re.IGNORECASE,
            ),
        ),
        (
            SectionKind.SUBPARAGRAPH,
            5,
            re.compile(
                r"^(?P<label>(?:\d+(?:-\d+)*|[a-zа-яәіңғүұқөһ]))\)\s+"
                r"(?P<title>.+)$",
                re.IGNORECASE,
            ),
        ),
        (
            SectionKind.PARAGRAPH,
            4,
            re.compile(r"^(?P<label>\d+(?:-\d+)*)[.]\s+(?P<title>.+)$"),
        ),
    )

    def parse(
        self,
        source: NormalizedSource,
        pipeline_signature: str = "default",
    ) -> tuple[StructuredSection, ...]:
        sections: list[StructuredSection] = []
        stack: dict[int, StructuredSection] = {}

        for block in source.blocks:
            detected = self._detect(block.text, block.style)
            if detected.kind is SectionKind.BODY:
                parent = stack[max(stack)] if stack else None
                hierarchy_path = parent.hierarchy_path if parent else ()
                article = stack[3].label if 3 in stack else None
                paragraph = stack[4].label if 4 in stack else None
            else:
                for level in tuple(stack):
                    if level >= detected.level:
                        del stack[level]
                lower_levels = [level for level in stack if level < detected.level]
                parent = stack[max(lower_levels)] if lower_levels else None
                descriptor = self._descriptor(detected)
                hierarchy_path = (*parent.hierarchy_path, descriptor) if parent else (descriptor,)
                article = detected.label if detected.kind is SectionKind.ARTICLE else None
                if article is None and 3 in stack:
                    article = stack[3].label
                paragraph = detected.label if detected.kind is SectionKind.PARAGRAPH else None
                if paragraph is None and 4 in stack:
                    paragraph = stack[4].label

            section = StructuredSection(
                id=uuid5(
                    NAMESPACE_URL,
                    f"{source.content_hash}:{pipeline_signature}:section:{block.ordinal}",
                ),
                parent_id=parent.id if parent else None,
                ordinal=block.ordinal,
                kind=detected.kind,
                label=detected.label,
                heading=detected.heading,
                article=article,
                paragraph=paragraph,
                hierarchy_path=hierarchy_path,
                text=block.text,
                page=block.page,
            )
            sections.append(section)
            if detected.kind is not SectionKind.BODY:
                stack[detected.level] = section

        return tuple(sections)

    def _detect(self, text: str, style: str | None) -> _DetectedStructure:
        single_line = " ".join(text.splitlines())
        for kind, level, pattern in self._patterns:
            match = pattern.match(single_line)
            if match:
                title = match.groupdict().get("title", "").strip(" .:–—-")
                return _DetectedStructure(
                    kind=kind,
                    level=level,
                    label=match.group("label"),
                    heading=title or None,
                )

        if style and (style.casefold().startswith("heading") or style.startswith("h")):
            return _DetectedStructure(
                kind=SectionKind.HEADING,
                level=1,
                label=None,
                heading=single_line,
            )

        return _DetectedStructure(SectionKind.BODY, 99, None, None)

    @staticmethod
    def _descriptor(detected: _DetectedStructure) -> str:
        label = f":{detected.label}" if detected.label else ""
        title = f":{detected.heading}" if detected.heading else ""
        return f"{detected.kind.value}{label}{title}"
