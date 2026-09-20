"""HTML parser that preserves visible block order and heading styles."""

from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import Tag

from regagent.application.ingestion.models import ParsedSource, RawBlock


class HtmlParser:
    extensions = frozenset({".html", ".htm"})
    name = "html"
    version = "2"
    _block_tags = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr")

    def parse(self, path: Path) -> ParsedSource:
        soup = BeautifulSoup(path.read_bytes(), "lxml")
        for unwanted in soup.find_all(("script", "style", "template", "nav", "footer")):
            unwanted.decompose()

        candidate = soup.find("main") or soup.find("article") or soup.body
        root = candidate if isinstance(candidate, Tag) else soup
        blocks: list[RawBlock] = []

        for tag in root.find_all(self._block_tags):
            if not isinstance(tag, Tag) or self._is_nested_duplicate(tag):
                continue
            raw_classes = tag.get("class")
            classes = (
                " ".join(str(item) for item in raw_classes)
                if isinstance(raw_classes, list)
                else ""
            )
            style = f"{tag.name} {classes}".strip()
            for text in self._tag_texts(tag):
                blocks.append(RawBlock(ordinal=len(blocks), text=text, style=style))

        title_tag = root.find("h1") or soup.find("title")
        title = title_tag.get_text(" ", strip=True) if isinstance(title_tag, Tag) else path.stem
        return ParsedSource(
            path=path,
            title=title,
            media_type="text/html",
            blocks=tuple(blocks),
            parser_name=self.name,
            parser_version=self.version,
        )

    @staticmethod
    def _tag_texts(tag: Tag) -> tuple[str, ...]:
        if tag.name == "tr":
            cells = [cell.get_text(" ", strip=True) for cell in tag.find_all(("th", "td"))]
            text = " | ".join(cell for cell in cells if cell)
            return (text,) if text else ()

        # Preserve explicit legal heading boundaries without splitting normal inline tags.
        marker = "__REGAGENT_EXPLICIT_LINE_BREAK__"
        for line_break in tag.find_all("br"):
            line_break.replace_with(marker)
        text = tag.get_text(" ", strip=True)
        return tuple(
            normalized
            for line in text.split(marker)
            if (normalized := " ".join(line.split()))
        )

    @staticmethod
    def _is_nested_duplicate(tag: Tag) -> bool:
        if tag.name in {"p", "li"} and tag.find_parent("tr") is not None:
            return True
        if tag.name == "li" and tag.find(("p", "li")) is not None:
            return True
        return False
