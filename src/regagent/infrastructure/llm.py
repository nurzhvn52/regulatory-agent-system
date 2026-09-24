"""HTTP adapter for a configured OpenAI-compatible chat-completions endpoint."""

from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ValidationError

from regagent.application.ports import LLMRequest, LLMResponse


class _Message(BaseModel):
    content: str | None


class _Choice(BaseModel):
    message: _Message


class _Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class _Completion(BaseModel):
    model: str
    choices: list[_Choice]
    usage: _Usage | None = None


class OpenAICompatibleLLM:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("LLM base URL must be an absolute HTTP(S) URL")
        if not model.strip():
            raise ValueError("LLM model must be configured")
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._client = client

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(self, request: LLMRequest) -> LLMResponse:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "temperature": 0,
            "stream": False,
        }
        if request.response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "regagent_response",
                    "strict": True,
                    "schema": request.response_schema,
                },
            }
        if self._client is not None:
            response = await self._client.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
        else:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self._timeout,
                )
        response.raise_for_status()
        try:
            completion = _Completion.model_validate(response.json())
        except (ValueError, ValidationError) as error:
            raise ValueError("LLM returned an invalid chat completion") from error
        if not completion.choices or not completion.choices[0].message.content:
            raise ValueError("LLM returned no text content")
        usage = completion.usage or _Usage()
        return LLMResponse(
            text=completion.choices[0].message.content,
            model=completion.model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
        )
