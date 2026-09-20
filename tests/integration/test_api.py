import httpx
import pytest

from regagent.api.app import create_app


@pytest.mark.asyncio
async def test_health_endpoint() -> None:
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_compile_agent_plan_endpoint() -> None:
    payload = {
        "name": "qa-api-test",
        "task_type": "cited_qa",
        "retrieval_strategy": "hybrid_rerank",
        "top_k": 12,
        "output_schema": "cited_answer_v1",
    }

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/agent-plans/compile", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["nodes"] == [
        "retrieve",
        "rerank",
        "answer_with_citations",
        "verify_evidence",
        "finish",
    ]
    assert body["top_k"] == 12
