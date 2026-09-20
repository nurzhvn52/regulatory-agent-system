"""UTF-8 plain-text parser used for fixtures and already-normalised sources."""

from pathlib import Path

from regagent.application.ingestion.models import ParsedSource, RawBlock


class PlainTextParser:
    extensions = frozenset({".txt"})
    name = "plain_text"
    version = "1"

    def parse(self, path: Path) -> ParsedSource:
        text = path.read_text(encoding="utf-8-sig")
        paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
        blocks = tuple(
            RawBlock(ordinal=index, text=value)
            for index, value in enumerate(paragraphs)
        )
        return ParsedSource(
            path=path,
            title=paragraphs[0] if paragraphs else path.stem,
            media_type="text/plain",
            blocks=blocks,
            parser_name=self.name,
            parser_version=self.version,
        )
