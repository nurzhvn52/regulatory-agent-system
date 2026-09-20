"""Select a deterministic local parser from a file extension."""

from pathlib import Path
from typing import Protocol

from regagent.application.ingestion.models import ParsedSource
from regagent.infrastructure.parsers.docx import DocxParser
from regagent.infrastructure.parsers.errors import UnsupportedDocumentTypeError
from regagent.infrastructure.parsers.html import HtmlParser
from regagent.infrastructure.parsers.pdf import PdfParser
from regagent.infrastructure.parsers.text import PlainTextParser


class _Parser(Protocol):
    extensions: frozenset[str]
    name: str
    version: str

    def parse(self, path: Path) -> ParsedSource: ...


class ParserRegistry:
    def __init__(self, parsers: tuple[_Parser, ...] | None = None) -> None:
        configured = parsers or (
            HtmlParser(),
            DocxParser(),
            PdfParser(),
            PlainTextParser(),
        )
        self._parsers = {
            extension: parser for parser in configured for extension in parser.extensions
        }

    def parse(self, path: Path) -> ParsedSource:
        parser = self._parsers.get(path.suffix.casefold())
        if parser is None:
            supported = ", ".join(sorted(self._parsers))
            raise UnsupportedDocumentTypeError(
                f"Unsupported document type {path.suffix!r}; supported: {supported}"
            )
        return parser.parse(path)
