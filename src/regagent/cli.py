"""Command-line entry point for reproducible data operations."""

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any
from uuid import UUID

from regagent.application.agent_generation.runtime import CitedQAAgent
from regagent.application.config import load_agent_spec
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
from regagent.infrastructure.agent_evaluation_runner import run_agent_evaluation
from regagent.infrastructure.agent_review_scoring import score_agent_review
from regagent.infrastructure.benchmark_runner import run_benchmark
from regagent.infrastructure.database.agent_runs import SqlAlchemyAgentRunRepository
from regagent.infrastructure.database.repositories import SqlAlchemyDocumentRepository
from regagent.infrastructure.database.retrieval_repository import (
    SqlAlchemyRetrievalRepository,
)
from regagent.infrastructure.database.session import Database
from regagent.infrastructure.embeddings import SentenceTransformerEmbeddingProvider
from regagent.infrastructure.llm import OpenAICompatibleLLM
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
    answer = subparsers.add_parser("answer", help="Run a cited QA agent and persist its trace")
    answer.add_argument("question")
    answer.add_argument("--spec", type=Path, default=Path("configs/agents/cited_qa_mvp.yaml"))
    answer.add_argument("--experiment-id")
    _add_corpus_arguments(answer)
    evaluate = subparsers.add_parser("evaluate", help="Compare retrieval on a pinned benchmark")
    evaluate.add_argument("dataset", type=Path)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--pipeline-signature", required=True)
    evaluate.add_argument("--chunking-strategy", default="legal_structure")
    evaluate.add_argument("--allow-draft", action="store_true")
    evaluate.add_argument(
        "--strategies",
        nargs="+",
        choices=("bm25", "dense", "hybrid"),
        default=["bm25", "dense", "hybrid"],
    )
    evaluate.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    evaluate_agent = subparsers.add_parser(
        "evaluate-agent", help="Evaluate cited QA and produce a human-review packet"
    )
    evaluate_agent.add_argument("dataset", type=Path)
    evaluate_agent.add_argument(
        "--spec", type=Path, default=Path("configs/agents/cited_qa_pilot.yaml")
    )
    evaluate_agent.add_argument("--output-dir", type=Path, required=True)
    evaluate_agent.add_argument("--pipeline-signature", required=True)
    evaluate_agent.add_argument("--chunking-strategy", default="legal_structure")
    evaluate_agent.add_argument("--effective-on", type=_iso_date)
    evaluate_agent.add_argument("--allow-draft", action="store_true")
    evaluate_agent.add_argument("--max-pairs", type=int)
    score_review = subparsers.add_parser(
        "score-agent-review", help="Score independently reviewed agent outputs"
    )
    score_review.add_argument("run_dir", type=Path)
    score_review.add_argument("--output", type=Path, required=True)
    score_review.add_argument("--allow-draft", action="store_true")
    return parser


def _add_corpus_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pipeline-signature", required=True)
    parser.add_argument("--chunking-strategy", default="legal_structure")
    parser.add_argument("--language", type=Language, action="append")
    parser.add_argument("--act-type", type=ActType, action="append")
    parser.add_argument("--document-id", type=UUID, action="append")
    parser.add_argument("--version-id", type=UUID, action="append")
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
        version_ids=tuple(arguments.version_id or ()),
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


def _llm_provider() -> OpenAICompatibleLLM:
    settings = get_settings()
    if settings.llm_provider != "openai_compatible":
        raise ValueError("Set REGAGENT_LLM_PROVIDER=openai_compatible for agent execution")
    if not settings.llm_model or not settings.llm_base_url:
        raise ValueError("Set REGAGENT_LLM_MODEL and REGAGENT_LLM_BASE_URL")
    return OpenAICompatibleLLM(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else None,
    )


async def _answer(arguments: argparse.Namespace) -> None:
    settings = get_settings()
    spec = load_agent_spec(arguments.spec)
    filters = _retrieval_filters(arguments)
    if not filters.languages:
        filters = filters.model_copy(update={"languages": spec.languages})
    llm = _llm_provider()
    database = Database(settings)
    runs = SqlAlchemyAgentRunRepository(database)
    repository = SqlAlchemyRetrievalRepository(database)
    lexical = BM25Retriever(repository)
    try:
        # Validate the plan and date scope before recording an attempted run.
        if spec.reranker_enabled or spec.retrieval_strategy is RetrievalStrategy.HYBRID_RERANK:
            raise ValueError("The cited QA runtime does not yet support reranking")
        if spec.verification.require_effective_version and filters.effective_on is None:
            raise ValueError("This agent requires --effective-on YYYY-MM-DD")
        if not filters.version_ids:
            raise ValueError("This agent requires at least one --version-id UUID")
        if spec.retrieval_strategy is RetrievalStrategy.BM25:
            retriever: Retriever = lexical
        else:
            dense = DenseRetriever(repository, _embedding_provider())
            retriever = (
                dense
                if spec.retrieval_strategy is RetrievalStrategy.DENSE
                else ReciprocalRankFusionRetriever((lexical, dense))
            )
        agent = CitedQAAgent(retriever, llm)
        run_id = await runs.start(spec, arguments.question, filters, arguments.experiment_id)
        try:
            result = await agent.answer(spec, arguments.question, filters)
            await runs.finish(run_id, result)
        except Exception as error:
            await runs.fail(run_id, error)
            raise
        print(
            json.dumps(
                {"run_id": str(run_id), **result.model_dump(mode="json")},
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        await database.dispose()


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
        print("[\n" + ",\n".join(hit.model_dump_json(indent=2) for hit in hits) + "\n]")
    finally:
        await database.dispose()


async def _evaluate(arguments: argparse.Namespace) -> None:
    database = Database(get_settings())
    try:
        result = await run_benchmark(
            arguments.dataset,
            arguments.output_dir,
            RetrievalFilters(
                pipeline_signature=arguments.pipeline_signature,
                chunking_strategy=arguments.chunking_strategy,
            ),
            SqlAlchemyRetrievalRepository(database),
            _embedding_provider(),
            strategies=tuple(arguments.strategies),
            k_values=tuple(sorted(set(arguments.k))),
            allow_draft=arguments.allow_draft,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        await database.dispose()


async def _evaluate_agent(arguments: argparse.Namespace) -> None:
    spec = load_agent_spec(arguments.spec)
    llm = _llm_provider()
    database = Database(get_settings())
    try:
        result = await run_agent_evaluation(
            arguments.dataset,
            arguments.output_dir,
            RetrievalFilters(
                pipeline_signature=arguments.pipeline_signature,
                chunking_strategy=arguments.chunking_strategy,
                effective_on=arguments.effective_on,
            ),
            spec,
            SqlAlchemyRetrievalRepository(database),
            (
                _embedding_provider()
                if spec.retrieval_strategy is not RetrievalStrategy.BM25
                else None
            ),
            llm,
            SqlAlchemyAgentRunRepository(database),
            allow_draft=arguments.allow_draft,
            max_pairs=arguments.max_pairs,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        await database.dispose()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    arguments = _build_parser().parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    if arguments.command == "ingest":
        asyncio.run(_ingest(arguments))
    elif arguments.command == "index":
        asyncio.run(_index(arguments))
    elif arguments.command == "search":
        asyncio.run(_search(arguments))
    elif arguments.command == "evaluate":
        try:
            asyncio.run(_evaluate(arguments))
        except (ValueError, FileNotFoundError) as error:
            raise SystemExit(f"Evaluation failed: {error}") from error
    elif arguments.command == "evaluate-agent":
        try:
            asyncio.run(_evaluate_agent(arguments))
        except (ValueError, FileNotFoundError) as error:
            raise SystemExit(f"Agent evaluation failed: {error}") from error
    elif arguments.command == "score-agent-review":
        try:
            print(
                json.dumps(
                    score_agent_review(
                        arguments.run_dir,
                        arguments.output,
                        allow_draft=arguments.allow_draft,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        except (ValueError, FileNotFoundError, KeyError) as error:
            raise SystemExit(f"Review scoring failed: {error}") from error
    elif arguments.command == "answer":
        try:
            asyncio.run(_answer(arguments))
        except (ValueError, FileNotFoundError) as error:
            raise SystemExit(f"Agent failed: {error}") from error


if __name__ == "__main__":
    main()
