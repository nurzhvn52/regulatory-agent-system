"""Command-line entry point for reproducible data operations."""

import argparse
import asyncio
import sys
from datetime import date
from pathlib import Path
from typing import Any

from regagent.application.ingestion.chunking import FixedWindowChunker, LegalStructureChunker
from regagent.application.ingestion.models import IngestionRequest
from regagent.application.ingestion.service import IngestionService
from regagent.domain.documents import ActType, Language
from regagent.infrastructure.database.repositories import SqlAlchemyDocumentRepository
from regagent.infrastructure.database.session import Database
from regagent.infrastructure.parsers import ParserRegistry
from regagent.settings import get_settings


def _iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected ISO date in YYYY-MM-DD format") from error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="regagent")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest = subparsers.add_parser("ingest", help="Parse, structure, chunk, and store a document")
    ingest.add_argument("file_path", type=Path)
    ingest.add_argument("--source-url", required=True)
    ingest.add_argument("--publisher", required=True)
    ingest.add_argument("--issuer", required=True)
    ingest.add_argument("--language", type=Language, choices=list(Language), required=True)
    ingest.add_argument("--act-type", type=ActType, choices=list(ActType), required=True)
    ingest.add_argument("--title")
    ingest.add_argument("--official-number")
    ingest.add_argument("--source-identifier")
    ingest.add_argument("--jurisdiction", default="KZ")
    ingest.add_argument("--retrieved-at", type=_iso_date, default=date.today())
    ingest.add_argument("--adopted-at", type=_iso_date)
    ingest.add_argument("--effective-from", type=_iso_date)
    ingest.add_argument("--effective-to", type=_iso_date)
    ingest.add_argument("--version-label")
    ingest.add_argument("--max-tokens", type=int, default=256)
    ingest.add_argument("--overlap-tokens", type=int, default=32)
    return parser


async def _ingest(arguments: argparse.Namespace) -> None:
    database = Database(get_settings())
    service = IngestionService(
        source_loader=ParserRegistry(),
        repository=SqlAlchemyDocumentRepository(database),
        chunkers=(
            FixedWindowChunker(arguments.max_tokens, arguments.overlap_tokens),
            LegalStructureChunker(arguments.max_tokens, arguments.overlap_tokens),
        ),
    )
    try:
        request_data: dict[str, Any] = {
            "file_path": arguments.file_path,
            "language": arguments.language,
            "act_type": arguments.act_type,
            "issuer": arguments.issuer,
            "source_url": arguments.source_url,
            "source_publisher": arguments.publisher,
            "retrieved_at": arguments.retrieved_at,
            "title": arguments.title,
            "jurisdiction": arguments.jurisdiction,
            "official_number": arguments.official_number,
            "source_identifier": arguments.source_identifier,
            "adopted_at": arguments.adopted_at,
            "effective_from": arguments.effective_from,
            "effective_to": arguments.effective_to,
            "version_label": arguments.version_label,
        }
        result = await service.ingest(IngestionRequest.model_validate(request_data))
        print(result.model_dump_json(indent=2))
    finally:
        await database.dispose()


def main() -> None:
    arguments = _build_parser().parse_args()
    if arguments.command == "ingest":
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(
                asyncio.WindowsSelectorEventLoopPolicy()
            )
        asyncio.run(_ingest(arguments))


if __name__ == "__main__":
    main()
