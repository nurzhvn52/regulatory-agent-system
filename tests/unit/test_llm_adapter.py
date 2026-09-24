import httpx
import pytest

from regagent.application.ports import LLMRequest
from regagent.infrastructure.llm import OpenAICompatibleLLM


@pytest.mark.asyncio
async def test_openai_compatible_adapter_posts_json_messages() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.content.decode()
        return httpx.Response(
            200,
            json={
                "model": "local-test",
                "choices": [{"message": {"content": '{"claims": []}'}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleLLM(
            model="local-test", base_url="http://localhost:1234/v1", client=client
        )
        response = await provider.generate(
            LLMRequest(system_prompt="system", user_prompt="question", response_schema="json")
        )

    assert seen["url"] == "http://localhost:1234/v1/chat/completions"
    assert '"response_format":{"type":"json_object"}' in str(seen["body"])
    assert response.input_tokens == 3
    assert response.output_tokens == 4
