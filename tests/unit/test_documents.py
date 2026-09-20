from datetime import date
from uuid import uuid4

import pytest
from pydantic import ValidationError

from regagent.domain.documents import DocumentVersion, SourceReference


def test_rejects_invalid_effective_interval() -> None:
    with pytest.raises(ValidationError):
        DocumentVersion(
            document_id=uuid4(),
            effective_from=date(2026, 2, 1),
            effective_to=date(2026, 1, 1),
            content_hash="a" * 64,
            source=SourceReference(
                url="https://example.gov.kz/document/1",
                retrieved_at=date(2026, 9, 20),
                publisher="Example publisher",
            ),
        )

