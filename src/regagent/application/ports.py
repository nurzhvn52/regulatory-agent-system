"""Interfaces implemented by external providers and storage adapters."""

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from regagent.domain.retrieval import RetrievalHit, RetrievalQuery


class LLMRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    system_prompt: str
    user_prompt: str
    response_schema: str | None = None


class LLMResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class LLMProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    async def generate(self, request: LLMRequest) -> LLMResponse: ...


class Retriever(Protocol):
    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalHit]: ...
