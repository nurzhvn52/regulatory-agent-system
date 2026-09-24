import asyncio
import os
import sys
from uuid import UUID

import pytest

from regagent.application.agent_generation.runtime import AnswerStatus, CitedAnswer
from regagent.domain.agents import AgentSpec, TaskType
from regagent.domain.retrieval import RetrievalFilters, RetrievalStrategy
from regagent.infrastructure.database.agent_runs import SqlAlchemyAgentRunRepository
from regagent.infrastructure.database.models import AgentRunRecord, AgentSpecRecord
from regagent.infrastructure.database.session import Database
from regagent.settings import Settings

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest.mark.asyncio
async def test_agent_run_is_persisted_with_spec_and_trace() -> None:
    database_url = os.getenv("REGAGENT_TEST_DATABASE_URL")
    if database_url is None:
        pytest.skip("Set REGAGENT_TEST_DATABASE_URL to run the PostgreSQL integration test")

    database = Database(Settings(database_url=database_url))
    repository = SqlAlchemyAgentRunRepository(database)
    spec = AgentSpec(
        name="integration-qa",
        task_type=TaskType.CITED_QA,
        retrieval_strategy=RetrievalStrategy.BM25,
        reranker_enabled=False,
        output_schema="cited_answer_v1",
    )
    filters = RetrievalFilters(pipeline_signature="a" * 64, version_ids=(UUID(int=3),))
    run_id: UUID | None = None
    spec_id: UUID | None = None
    try:
        run_id = await repository.start(spec, "Test question", filters, "integration-test")
        await repository.finish(
            run_id,
            CitedAnswer(
                status=AnswerStatus.REFUSED,
                reason="No matching evidence",
                trace=({"node": "finish", "status": "refused"},),
            ),
        )
        async with database.session() as session:
            run = await session.get(AgentRunRecord, run_id)
            assert run is not None
            spec_id = run.agent_spec_id
            assert run.status == "refused"
            assert run.output is not None
            assert run.output["trace"][0]["node"] == "finish"
            assert run.input["filters"]["version_ids"] == [str(UUID(int=3))]
            stored_spec = await session.get(AgentSpecRecord, spec_id)
            assert stored_spec is not None
            assert stored_spec.spec["task_type"] == "cited_qa"
    finally:
        if run_id is not None and spec_id is not None:
            async with database.session() as session, session.begin():
                run = await session.get(AgentRunRecord, run_id)
                if run is not None:
                    await session.delete(run)
                stored_spec = await session.get(AgentSpecRecord, spec_id)
                if stored_spec is not None:
                    await session.delete(stored_spec)
        await database.dispose()
