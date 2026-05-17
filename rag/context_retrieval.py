from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from core.types import ChatRequest, ContextDocument


class VectorDBClient(ABC):
    @abstractmethod
    async def upsert(self, documents: list[ContextDocument]) -> None:
        pass

    @abstractmethod
    async def search(self, embedding: list[float], top_k: int, metadata_filter: dict[str, Any] | None = None) -> list[ContextDocument]:
        pass


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        pass


class ContextRetriever:
    def __init__(self, embedding_provider: EmbeddingProvider, vector_db: VectorDBClient, max_tokens: int = 1500) -> None:
        self.embedding_provider = embedding_provider
        self.vector_db = vector_db
        self.max_tokens = max_tokens

    async def retrieve(self, request: ChatRequest, metadata_filter: dict[str, Any] | None = None) -> list[ContextDocument]:
        embedding = await self.embedding_provider.embed(request.prompt)
        results = await self.vector_db.search(embedding, top_k=8, metadata_filter=metadata_filter)
        documents = []
        token_budget = self.max_tokens
        for document in results:
            if document.token_count and document.token_count > token_budget:
                continue
            documents.append(document)
            token_budget -= document.token_count or 0
            if token_budget <= 0:
                break
        return documents


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = "text-embedding-3-large", logger: Any | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self.logger = logger

    async def embed(self, text: str) -> list[float]:
        from openai import AsyncAzureOpenAI
        from core.config import get_config

        config = get_config()

        if self.logger:
            self.logger.debug(
                "Generating embedding for text length=%d",
                len(text),
            )

        client = AsyncAzureOpenAI(
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version,
            azure_endpoint=config.azure_openai_endpoint,
        )

        response = await client.embeddings.create(
            model=self.model,  # MUST be Azure deployment name
            input=text,
        )

        return response.data[0].embedding

