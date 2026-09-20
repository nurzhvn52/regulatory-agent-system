"""FastAPI application factory."""

from fastapi import FastAPI
from pydantic import BaseModel

from regagent import __version__
from regagent.application.agent_generation.generator import (
    AgentGenerator,
    GeneratedAgentPlan,
)
from regagent.domain.agents import AgentSpec
from regagent.settings import get_settings


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=__version__)
    generator = AgentGenerator()

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            environment=settings.environment,
        )

    @app.post("/v1/agent-plans/compile", response_model=GeneratedAgentPlan)
    async def compile_agent_plan(spec: AgentSpec) -> GeneratedAgentPlan:
        return generator.generate(spec)

    return app

