from regagent.application.agent_generation.generator import AgentGenerator, NodeType
from regagent.domain.agents import AgentSpec, TaskType, ToolType, VerificationPolicy
from regagent.domain.retrieval import RetrievalStrategy


def test_generates_cited_qa_plan_with_reranking_and_verification() -> None:
    spec = AgentSpec(
        name="qa",
        task_type=TaskType.CITED_QA,
        retrieval_strategy=RetrievalStrategy.HYBRID_RERANK,
        output_schema="cited_answer_v1",
    )

    plan = AgentGenerator().generate(spec)

    assert plan.nodes == (
        NodeType.RETRIEVE,
        NodeType.RERANK,
        NodeType.ANSWER_WITH_CITATIONS,
        NodeType.VERIFY_EVIDENCE,
        NodeType.FINISH,
    )
    assert ToolType.VERIFY_CITATIONS in plan.tools


def test_version_comparison_adds_version_tools() -> None:
    spec = AgentSpec(
        name="comparison",
        task_type=TaskType.VERSION_COMPARISON,
        output_schema="version_comparison_v1",
    )

    plan = AgentGenerator().generate(spec)

    assert ToolType.LOOKUP_VERSION in plan.tools
    assert ToolType.COMPARE_VERSIONS in plan.tools
    assert NodeType.COMPARE_VERSIONS in plan.nodes


def test_verification_node_can_be_disabled_for_ablation() -> None:
    spec = AgentSpec(
        name="ablation-no-verifier",
        task_type=TaskType.REQUIREMENT_EXTRACTION,
        retrieval_strategy=RetrievalStrategy.BM25,
        reranker_enabled=False,
        verification=VerificationPolicy(
            citations_required=False,
            entailment_check=False,
            refuse_without_evidence=False,
            require_effective_version=False,
        ),
        tools=(ToolType.DOCUMENT_SEARCH,),
        output_schema="requirements_v1",
    )

    plan = AgentGenerator().generate(spec)

    assert NodeType.RERANK not in plan.nodes
    assert NodeType.VERIFY_EVIDENCE not in plan.nodes
    assert ToolType.VERIFY_CITATIONS not in plan.tools

