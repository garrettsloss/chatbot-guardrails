from __future__ import annotations

import re
from typing import Iterable

from core.types import ContextDocument


class ContextSanitizer:
    hidden_instruction_pattern = re.compile(r"<hidden_instruction>|ignore.*instructions|follow.*secret", re.IGNORECASE)
    encoded_payload_pattern = re.compile(r"(?:base64|hex)\s*[:=]", re.IGNORECASE)

    async def sanitize_document(self, document: ContextDocument) -> ContextDocument:
        content = document.content
        content = self.hidden_instruction_pattern.sub("", content)
        content = self.encoded_payload_pattern.sub("", content)
        content = self._remove_malicious_formatting(content)
        return ContextDocument(
            request_id=document.request_id,
            timestamp=document.timestamp,
            source_module="guardrails.context_sanitizer",
            document_id=document.document_id,
            content=content.strip(),
            source=document.source,
            metadata=document.metadata,
            relevance_score=document.relevance_score,
            token_count=document.token_count,
        )

    async def sanitize_documents(self, documents: Iterable[ContextDocument]) -> list[ContextDocument]:
        return [await self.sanitize_document(document) for document in documents]

    @classmethod
    def _remove_malicious_formatting(cls, text: str) -> str:
        text = text.replace("\x0b", " ").replace("\x0c", " ")
        text = re.sub(r"\s{2,}", " ", text)
        return text
