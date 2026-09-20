"""DOCX parser for paragraphs and table rows in document order."""

from pathlib import Path

from docx import Document as open_document
from docx.table import Table
from docx.text.paragraph import Paragraph

from regagent.application.ingestion.models import ParsedSource, RawBlock


class DocxParser:
    extensions = frozenset({".docx"})
    name = "docx"
    version = "1"

    def parse(self, path: Path) -> ParsedSource:
        document = open_document(str(path))
        blocks: list[RawBlock] = []
        title: str | None = None

        for item in document.iter_inner_content():
            if isinstance(item, Paragraph):
                text = item.text.strip()
                if not text:
                    continue
                style = item.style.name if item.style is not None else None
                if title is None and style and style.casefold().startswith(("title", "heading 1")):
                    title = text
                blocks.append(
                    RawBlock(
                        ordinal=len(blocks),
                        text=text,
                        style=style,
                    )
                )
            elif isinstance(item, Table):
                for row in item.rows:
                    text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if text:
                        blocks.append(
                            RawBlock(
                                ordinal=len(blocks),
                                text=text,
                                style="table-row",
                            )
                        )

        if title is None:
            title = blocks[0].text if blocks else path.stem
        return ParsedSource(
            path=path,
            title=title,
            media_type=(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ),
            blocks=tuple(blocks),
            parser_name=self.name,
            parser_version=self.version,
        )
