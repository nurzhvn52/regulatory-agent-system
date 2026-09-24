"""Persist immutable agent specifications and individual execution traces."""

from datetime import UTC, datetime
from uuid import UUID

from regagent.application.agent_generation.runtime import CitedAnswer
from regagent.domain.agents import AgentSpec
from regagent.domain.retrieval import RetrievalFilters
from regagent.infrastructure.database.models import AgentRunRecord, AgentSpecRecord
from regagent.infrastructure.database.session import Database


class SqlAlchemyAgentRunRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def start(
        self,
        spec: AgentSpec,
        question: str,
        filters: RetrievalFilters,
        experiment_id: str | None = None,
    ) -> UUID:
        async with self._database.session() as session, session.begin():
            spec_record = AgentSpecRecord(
                name=spec.name,
                schema_version="1",
                spec=spec.model_dump(mode="json"),
            )
            session.add(spec_record)
            await session.flush()
            run = AgentRunRecord(
                agent_spec_id=spec_record.id,
                status="running",
                experiment_id=experiment_id,
                input={
                    "question": question,
                    "filters": filters.model_dump(mode="json"),
                },
            )
            session.add(run)
            await session.flush()
            return run.id

    async def finish(self, run_id: UUID, result: CitedAnswer) -> None:
        async with self._database.session() as session, session.begin():
            run = await session.get(AgentRunRecord, run_id)
            if run is None or run.status != "running":
                raise ValueError("Agent run is missing or already finished")
            run.status = result.status.value
            run.llm_model = result.model
            run.output = result.model_dump(mode="json")
            run.completed_at = datetime.now(UTC)

    async def fail(self, run_id: UUID, error: Exception) -> None:
        async with self._database.session() as session, session.begin():
            run = await session.get(AgentRunRecord, run_id)
            if run is None or run.status != "running":
                raise ValueError("Agent run is missing or already finished")
            run.status = "failed"
            run.error = f"{type(error).__name__}: {error}"
            run.completed_at = datetime.now(UTC)
