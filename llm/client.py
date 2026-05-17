from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from core.types import ChatResponse


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

    async def generate(self, prompt_payload: dict[str, Any], timeout: int | None = 60) -> ChatResponse:
        if self.logger:
            self.logger.debug("Calling OpenAI-compatible API with model %s", self.model)
        await asyncio.sleep(0)
        return ChatResponse(
            request_id=prompt_payload.get("request_id", "llm"),
            timestamp=prompt_payload.get("timestamp", None) or __import__("datetime").datetime.utcnow(),
            source_module="llm.client.openai",
            response_text="[generated response placeholder]",
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
