"""Command-line entry point for reproducible data operations."""

import argparse
import asyncio
import sys
from datetime import date
from pathlib import Path
from typing import Any
from uuid import UUID

from regagent.application.ingestion.chunking import FixedWindowChunker, LegalStructureChunker
from regagent.application.ingestion.models import IngestionRequest
from regagent.application.ingestion.service import IngestionService
from regagent.application.ports import Retriever
from regagent.application.retrieval.bm25 import BM25Retriever
from regagent.application.retrieval.services import (
    DenseRetriever,
    EmbeddingIndexService,
    ReciprocalRankFusionRetriever,
)
from regagent.domain.documents import ActType, Language
from regagent.domain.retrieval import RetrievalFilters, RetrievalQuery, RetrievalStrategy
from regagent.infrastructure.database.repositories import SqlAlchemyDocumentRepository
from regagent.infrastructure.database.retrieval_repository import (
    SqlAlchemyRetrievalRepository,
)
from regagent.infrastructure.database.session import Database
from regagent.infrastructure.embeddings import SentenceTransformerEmbeddingProvider
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

    index = subparsers.add_parser("index", help="Create missing dense embeddings")
    _add_corpus_arguments(index)

    search = subparsers.add_parser("search", help="Search one reproducible corpus snapshot")
    search.add_argument("query")
    search.add_argument(
        "--strategy",
        type=RetrievalStrategy,
        choices=(
            RetrievalStrategy.BM25,
            RetrievalStrategy.DENSE,
            RetrievalStrategy.HYBRID,
        ),
        default=RetrievalStrategy.HYBRID,
    )
    search.add_argument("--top-k", type=int, default=10)
    _add_corpus_arguments(search)
    return parser


def _add_corpus_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pipeline-signature", required=True)
    parser.add_argument("--chunking-strategy", default="legal_structure")
    parser.add_argument("--language", type=Language, action="append")
    parser.add_argument("--act-type", type=ActType, action="append")
    parser.add_argument("--document-id", type=UUID, action="append")
    parser.add_argument("--effective-on", type=_iso_date)


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


def _retrieval_filters(arguments: argparse.Namespace) -> RetrievalFilters:
    return RetrievalFilters(
        pipeline_signature=arguments.pipeline_signature,
        chunking_strategy=arguments.chunking_strategy,
        languages=tuple(arguments.language or ()),
        act_types=tuple(arguments.act_type or ()),
        document_ids=tuple(arguments.document_id or ()),
        effective_on=arguments.effective_on,
    )


def _embedding_provider() -> SentenceTransformerEmbeddingProvider:
    settings = get_settings()
    return SentenceTransformerEmbeddingProvider(
        settings.embedding_model,
        settings.embedding_revision,
        settings.embedding_dimensions,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )


async def _index(arguments: argparse.Namespace) -> None:
    settings = get_settings()
    database = Database(settings)
    repository = SqlAlchemyRetrievalRepository(database)
    service = EmbeddingIndexService(
        repository,
        _embedding_provider(),
        batch_size=settings.embedding_batch_size,
    )
    try:
        result = await service.index(_retrieval_filters(arguments))
        print(result.model_dump_json(indent=2))
    finally:
        await database.dispose()


async def _search(arguments: argparse.Namespace) -> None:
    database = Database(get_settings())
    repository = SqlAlchemyRetrievalRepository(database)
    lexical = BM25Retriever(repository)
    try:
        retriever: Retriever
        if arguments.strategy is RetrievalStrategy.BM25:
            retriever = lexical
        else:
            dense = DenseRetriever(repository, _embedding_provider())
            retriever = (
                dense
                if arguments.strategy is RetrievalStrategy.DENSE
                else ReciprocalRankFusionRetriever((lexical, dense))
            )
        hits = await retriever.retrieve(
            RetrievalQuery(
                text=arguments.query,
                top_k=arguments.top_k,
                filters=_retrieval_filters(arguments),
            )
        )
        print(
            "[\n"
            + ",\n".join(hit.model_dump_json(indent=2) for hit in hits)
            + "\n]"
        )
    finally:
        await database.dispose()


def main() -> None:
    arguments = _build_parser().parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    if arguments.command == "ingest":
        asyncio.run(_ingest(arguments))
    elif arguments.command == "index":
        asyncio.run(_index(arguments))
    elif arguments.command == "search":
        asyncio.run(_search(arguments))


if __name__ == "__main__":
    main()
