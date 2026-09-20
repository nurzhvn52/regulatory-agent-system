"""Conservative text normalisation that preserves legal boundaries."""

import hashlib
import re
import unicodedata

from regagent.application.ingestion.models import NormalizedSource, ParsedSource, RawBlock

_HORIZONTAL_WHITESPACE = re.compile(r"[\t\v\f \u00a0]+")
_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")


class TextNormalizer:
    """Normalise encoding and whitespace without rewriting legal language."""

    VERSION = "conservative_normalizer_v1"

    def normalize(self, source: ParsedSource) -> NormalizedSource:
        blocks: list[RawBlock] = []
        for block in source.blocks:
            text = self.normalize_text(block.text)
            if not text:
                continue
            blocks.append(
                RawBlock(
                    ordinal=len(blocks),
                    text=text,
                    style=block.style,
                    page=block.page,
                )
            )

        if not blocks:
            raise ValueError(f"No readable text found in {source.path}")

        # Version identity depends on the legal text sequence, not on parser block borders.
        canonical = " ".join(" ".join(block.text.split()) for block in blocks)
        content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return NormalizedSource(
            path=source.path,
            title=self.normalize_text(source.title) or source.path.stem,
            media_type=source.media_type,
            blocks=tuple(blocks),
            parser_name=source.parser_name,
            parser_version=source.parser_version,
            content_hash=content_hash,
        )

    @staticmethod
    def normalize_text(value: str) -> str:
        text = unicodedata.normalize("NFC", value)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = _ZERO_WIDTH.sub("", text)
        lines = (_HORIZONTAL_WHITESPACE.sub(" ", line).strip() for line in text.split("\n"))
        text = "\n".join(line for line in lines if line)
        return _MULTIPLE_NEWLINES.sub("\n\n", text).strip()
