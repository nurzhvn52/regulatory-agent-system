"""Parser-specific failures translated into actionable ingestion errors."""


class UnsupportedDocumentTypeError(ValueError):
    pass


class ScannedDocumentError(ValueError):
    """Raised when a PDF contains no extractable text and requires OCR."""

