from pathlib import Path

from regagent.application.ingestion.models import SectionKind
from regagent.application.ingestion.normalizer import TextNormalizer
from regagent.application.ingestion.structure import LegalStructureParser
from regagent.infrastructure.parsers import ParserRegistry


def _sections(fixture: str):  # type: ignore[no-untyped-def]
    path = Path(__file__).parents[1] / "fixtures" / fixture
    parsed = ParserRegistry().parse(path)
    normalized = TextNormalizer().normalize(parsed)
    return LegalStructureParser().parse(normalized)


def test_recognises_russian_legal_hierarchy() -> None:
    sections = _sections("sample_regulation_ru.html")

    articles = [section for section in sections if section.kind is SectionKind.ARTICLE]
    paragraphs = [section for section in sections if section.kind is SectionKind.PARAGRAPH]
    subparagraphs = [
        section for section in sections if section.kind is SectionKind.SUBPARAGRAPH
    ]

    assert [section.label for section in articles] == ["1", "2"]
    assert paragraphs[0].article == "1"
    assert subparagraphs[0].parent_id == paragraphs[0].id
    assert subparagraphs[0].article == "1"


def test_recognises_kazakh_suffix_headings() -> None:
    sections = _sections("sample_regulation_kk.html")

    assert any(
        section.kind is SectionKind.PART and section.label == "1" for section in sections
    )
    assert any(
        section.kind is SectionKind.CHAPTER and section.label == "1" for section in sections
    )
    assert [
        section.label for section in sections if section.kind is SectionKind.ARTICLE
    ] == ["1", "2"]


def test_section_identifiers_are_stable_for_same_content() -> None:
    first = _sections("sample_regulation_ru.html")
    second = _sections("sample_regulation_ru.html")

    assert [section.id for section in first] == [section.id for section in second]

