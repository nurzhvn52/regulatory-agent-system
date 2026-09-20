"""Text-PDF parser using positional blocks in reading order."""

from pathlib import Path
from typing import Any

import pymupdf

from regagent.application.ingestion.models import ParsedSource, RawBlock
from regagent.infrastructure.parsers.errors import ScannedDocumentError


class PdfParser:
    extensions = frozenset({".pdf"})
    name = "pymupdf_blocks"
    version = "1"

    def parse(self, path: Path) -> ParsedSource:
        blocks: list[RawBlock] = []
        title: str | None = None
        with pymupdf.open(str(path)) as document:  # type: ignore[no-untyped-call]
            metadata: dict[str, Any] = document.metadata or {}
            metadata_title = metadata.get("title")
            if isinstance(metadata_title, str) and metadata_title.strip():
                title = metadata_title.strip()

            for page_number, page in enumerate(document, start=1):
                page_blocks = page.get_text("blocks", sort=True)
                for block in page_blocks:
                    if len(block) >= 7 and block[6] != 0:
                        continue
                    text = str(block[4]).strip()
                    if text:
                        blocks.append(
                            RawBlock(
                                ordinal=len(blocks),
                                text=text,
                                style="pdf-text-block",
                                page=page_number,
                            )
                        )

        if not blocks:
            raise ScannedDocumentError(
                f"{path} contains no extractable text; OCR adapter is required"
            )
        return ParsedSource(
            path=path,
            title=title or blocks[0].text or path.stem,
            media_type="application/pdf",
            blocks=tuple(blocks),
            parser_name=self.name,
            parser_version=self.version,
        )
