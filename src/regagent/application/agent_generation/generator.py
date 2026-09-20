"""Core agent-generation method, independent of the workflow runtime."""

from enum import StrEnum
from itertools import pairwise
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from regagent.domain.agents import AgentSpec, TaskType, ToolType
from regagent.domain.retrieval import RetrievalStrategy


class NodeType(StrEnum):
    RETRIEVE = "retrieve"
    RERANK = "rerank"
    ANSWER_WITH_CITATIONS = "answer_with_citations"
    EXTRACT_REQUIREMENTS = "extract_requirements"
    COMPARE_VERSIONS = "compare_versions"
    VERIFY_EVIDENCE = "verify_evidence"
    FINISH = "finish"


class WorkflowEdge(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: NodeType
    target: NodeType


class GeneratedAgentPlan(BaseModel):
    """Runtime-neutral graph produced by the generation method."""

    model_config = ConfigDict(frozen=True)

    name: str
    task_type: TaskType
    nodes: tuple[NodeType, ...]
    edges: tuple[WorkflowEdge, ...]
    tools: tuple[ToolType, ...]
    retrieval_strategy: RetrievalStrategy
    top_k: int


class AgentGenerator:
    """Generate a workflow based on task, retrieval, and verification constraints."""

    _task_nodes: ClassVar[dict[TaskType, NodeType]] = {
        TaskType.CITED_QA: NodeType.ANSWER_WITH_CITATIONS,
        TaskType.REQUIREMENT_EXTRACTION: NodeType.EXTRACT_REQUIREMENTS,
        TaskType.VERSION_COMPARISON: NodeType.COMPARE_VERSIONS,
    }

    def generate(self, spec: AgentSpec) -> GeneratedAgentPlan:
        task_node = self._task_nodes[spec.task_type]
        nodes: list[NodeType] = [NodeType.RETRIEVE]

        should_rerank = (
            spec.reranker_enabled
            or spec.retrieval_strategy is RetrievalStrategy.HYBRID_RERANK
        )
        if should_rerank:
            nodes.append(NodeType.RERANK)

        nodes.append(task_node)
        if self._requires_verification(spec):
            nodes.append(NodeType.VERIFY_EVIDENCE)
        nodes.append(NodeType.FINISH)

        edges = tuple(
            WorkflowEdge(source=source, target=target)
            for source, target in pairwise(nodes)
        )

        tools = self._required_tools(spec)
        return GeneratedAgentPlan(
            name=spec.name,
            task_type=spec.task_type,
            nodes=tuple(nodes),
            edges=edges,
            tools=tools,
            retrieval_strategy=spec.retrieval_strategy,
            top_k=spec.top_k,
        )

    @staticmethod
    def _requires_verification(spec: AgentSpec) -> bool:
        policy = spec.verification
        return any(
            (
                policy.citations_required,
                policy.entailment_check,
                policy.refuse_without_evidence,
                policy.require_effective_version,
            )
        )

    @staticmethod
    def _required_tools(spec: AgentSpec) -> tuple[ToolType, ...]:
        tools = list(spec.tools)
        tools.extend((ToolType.DOCUMENT_SEARCH, ToolType.READ_SECTION))

        if spec.task_type is TaskType.VERSION_COMPARISON:
            tools.extend((ToolType.LOOKUP_VERSION, ToolType.COMPARE_VERSIONS))

        if AgentGenerator._requires_verification(spec):
            tools.append(ToolType.VERIFY_CITATIONS)

        return tuple(dict.fromkeys(tools))
