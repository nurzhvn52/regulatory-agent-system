from pathlib import Path

from regagent.application.ingestion.chunking import FixedWindowChunker, LegalStructureChunker
from regagent.application.ingestion.models import ChunkingStrategy
from regagent.application.ingestion.normalizer import TextNormalizer
from regagent.application.ingestion.structure import LegalStructureParser
from regagent.infrastructure.parsers import ParserRegistry


def _source_and_sections():  # type: ignore[no-untyped-def]
    path = Path(__file__).parents[1] / "fixtures" / "sample_regulation_ru.html"
    source = TextNormalizer().normalize(ParserRegistry().parse(path))
    return source, LegalStructureParser().parse(source)


def test_fixed_window_chunker_respects_size_and_records_provenance() -> None:
    source, sections = _source_and_sections()

    chunks = FixedWindowChunker(max_tokens=12, overlap_tokens=3).chunk(source, sections)

    assert all(chunk.token_count <= 12 for chunk in chunks)
    assert all(chunk.strategy is ChunkingStrategy.FIXED_WINDOW for chunk in chunks)
    assert any(len(chunk.source_section_ids) > 1 for chunk in chunks)


def test_legal_chunker_never_crosses_article_boundaries() -> None:
    source, sections = _source_and_sections()
    article_by_section = {section.id: section.article for section in sections}

    chunks = LegalStructureChunker(max_tokens=24, overlap_tokens=4).chunk(source, sections)

    for chunk in chunks:
        article_labels = {
            article_by_section[section_id]
            for section_id in chunk.source_section_ids
            if article_by_section[section_id] is not None
        }
        assert len(article_labels) <= 1
    assert all(chunk.token_count <= 24 for chunk in chunks)

