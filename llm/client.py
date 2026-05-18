from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from core.types import ChatResponse

from openai import AsyncAzureOpenAI, BadRequestError
from core.config import get_config

logger = logging.getLogger(__name__)


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
    def __init__(self, api_key: str, model: str, logger_: Any | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self._logger = logger_ or logger

    async def generate(
        self,
        prompt_payload: dict[str, Any],
        timeout: int | None = 60,
    ) -> ChatResponse:
        config = get_config()

        client = AsyncAzureOpenAI(
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version,
            azure_endpoint=config.azure_openai_endpoint,
        )

        messages = prompt_payload.get("messages", [
            {"role": "user", "content": prompt_payload.get("prompt", "")}
        ])

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
            )
        except BadRequestError as exc:
            # Azure content management policy blocked the request
            if getattr(exc, "code", None) == "content_filter" or (
                hasattr(exc, "body")
                and isinstance(exc.body, dict)
                and exc.body.get("error", {}).get("code") == "content_filter"
            ):
                logger.warning(
                    "Azure content filter triggered | model=%s | request_id=%s",
                    self.model,
                    prompt_payload.get("request_id", "unknown"),
                )
                return ChatResponse(
                    request_id=prompt_payload.get("request_id", "llm"),
                    timestamp=__import__("datetime").datetime.utcnow(),
                    source_module="llm.client.azure",
                    response_text="",
                    safe=False,
                    policy_action=None,
                    reasons=["azure_content_filter"],
                    tool_results=[],
                    metadata={"model": self.model, "azure_error": "content_filter"},
                )
            raise

        text = response.choices[0].message.content or ""
        usage = response.usage

        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0

        logger.info(
            "LLM call complete | model=%s | request_id=%s | tokens=prompt:%d+completion:%d=%d",
            self.model,
            prompt_payload.get("request_id", "unknown"),
            prompt_tokens,
            completion_tokens,
            total_tokens,
        )

        return ChatResponse(
            request_id=prompt_payload.get("request_id", "llm"),
            timestamp=__import__("datetime").datetime.utcnow(),
            source_module="llm.client.azure",
            response_text=text,
            safe=True,
            policy_action=None,
            reasons=[],
            tool_results=[],
            metadata={
                "model": self.model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        )

    async def stream(self, prompt_payload: dict[str, Any]) -> Any:
        async def generator() -> Any:
            yield {"text": "[streaming response chunk]"}
        return generator()

    def estimate_cost(self, prompt_payload: dict[str, Any]) -> float:
        return 0.0
