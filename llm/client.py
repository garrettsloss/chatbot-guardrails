from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from core.types import ChatResponse

from openai import AsyncAzureOpenAI
from core.config import get_config

class LLMClientError(Exception):
    pass


class BaseLLMClient(ABC):
    @abstractmethod
    async def generate(self, prompt_payload: dict[str, Any], timeout: int | None = None) -> ChatResponse:
        pass

    @abstractmethod
    async def stream(self, prompt_payload: dict[str, Any]) -> Any:
        pass

    @abstractmethod
    def estimate_cost(self, prompt_payload: dict[str, Any]) -> float:
        pass


class OpenAIClientAdapter(BaseLLMClient):
    def __init__(self, api_key: str, model: str, logger: Any | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self.logger = logger

    async def generate(
        self,
        prompt_payload: dict[str, Any],
        timeout: int | None = 60
    ) -> ChatResponse:

        config = get_config()

        if self.logger:
            self.logger.debug("Calling Azure OpenAI with model %s", self.model)

        client = AsyncAzureOpenAI(
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version,
            azure_endpoint=config.azure_openai_endpoint,
        )

        messages = prompt_payload.get("messages", [
            {"role": "user", "content": prompt_payload.get("prompt", "")}
        ])

        response = await client.chat.completions.create(
            model=self.model,  # Azure deployment name
            messages=messages,
            temperature=0.7,
        )

        text = response.choices[0].message.content

        return ChatResponse(
            request_id=prompt_payload.get("request_id", "llm"),
            timestamp=__import__("datetime").datetime.utcnow(),
            source_module="llm.client.azure",
            response_text=text,
            safe=True,
            policy_action=None,
            reasons=[],
            tool_results=[],
            metadata={"model": self.model},
        )

    async def stream(self, prompt_payload: dict[str, Any]) -> Any:
        async def generator() -> Any:
            yield {"text": "[streaming response chunk]"}
        return generator()

    def estimate_cost(self, prompt_payload: dict[str, Any]) -> float:
        return 0.0
