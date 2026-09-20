from pathlib import Path

from regagent.application.config import load_agent_spec
from regagent.domain.agents import TaskType
from regagent.domain.retrieval import RetrievalStrategy


def test_loads_version_controlled_agent_spec() -> None:
    project_root = Path(__file__).parents[2]

    spec = load_agent_spec(project_root / "configs" / "agents" / "cited_qa.yaml")

    assert spec.task_type is TaskType.CITED_QA
    assert spec.retrieval_strategy is RetrievalStrategy.HYBRID_RERANK
    assert spec.verification.citations_required is True

