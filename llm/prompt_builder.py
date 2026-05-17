from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.types import ContextDocument


@dataclass(frozen=True)
class PromptPayload:
    template_name: str
    messages: list[dict[str, str]]
    token_estimate: int
    metadata: dict[str, Any] = None


class PromptBuilder:
    def __init__(self, model_template: str, max_tokens: int = 4096) -> None:
        self.model_template = model_template
        self.max_tokens = max_tokens

    def build(self, user_prompt: str, context: list[ContextDocument], history: list[dict[str, str]] | None = None) -> PromptPayload:
        history = history or []
        context_text = "\n\n".join([f"Source: {doc.source}\n{doc.content}" for doc in context])
        system_message = self.model_template.strip()
        messages = [
            {"role": "system", "content": system_message},
            {"role": "system", "content": context_text},
        ]
        messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})
        token_estimate = self._estimate_tokens(messages)
        return PromptPayload(
            template_name="default",
            messages=messages,
            token_estimate=token_estimate,
            metadata={"context_sources": [doc.source for doc in context], "context_count": len(context)},
        )

    def _estimate_tokens(self, messages: list[dict[str, str]]) -> int:
        return sum(len(message["content"]) // 4 + 1 for message in messages)
