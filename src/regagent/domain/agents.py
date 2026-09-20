"""Declarative specification of agents generated for regulatory tasks."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from regagent.domain.documents import Language
from regagent.domain.retrieval import RetrievalStrategy


class TaskType(StrEnum):
    CITED_QA = "cited_qa"
    REQUIREMENT_EXTRACTION = "requirement_extraction"
    VERSION_COMPARISON = "version_comparison"


class ToolType(StrEnum):
    DOCUMENT_SEARCH = "document_search"
    READ_SECTION = "read_section"
    LOOKUP_VERSION = "lookup_version"
    COMPARE_VERSIONS = "compare_versions"
    VERIFY_CITATIONS = "verify_citations"


class VerificationPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    citations_required: bool = True
    entailment_check: bool = True
    refuse_without_evidence: bool = True
    require_effective_version: bool = True


class AgentSpec(BaseModel):
    """Research-controlled input to the agent generation method."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    task_type: TaskType
    languages: tuple[Language, ...] = (Language.RU, Language.KK)
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RERANK
    top_k: int = Field(default=15, ge=1, le=100)
    reranker_enabled: bool = True
    tools: tuple[ToolType, ...] = (
        ToolType.DOCUMENT_SEARCH,
        ToolType.READ_SECTION,
        ToolType.VERIFY_CITATIONS,
    )
    verification: VerificationPolicy = Field(default_factory=VerificationPolicy)
    output_schema: str = Field(min_length=1)

