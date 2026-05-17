from __future__ import annotations

from typing import Any
import chromadb

from core.types import ContextDocument
from rag.context_retrieval import VectorDBClient  # adjust import if needed


class ChromaVectorDB(VectorDBClient):
    def __init__(self, path: str = "./data/chroma") -> None:
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection("documents")

    async def upsert(self, documents: list[ContextDocument]) -> None:
        self.collection.upsert(
            ids=[d.id for d in documents],
            documents=[d.content for d in documents],
            metadatas=[d.metadata or {} for d in documents],
            embeddings=[d.embedding for d in documents],
        )

    async def search(
        self,
        embedding: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[ContextDocument]:

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=metadata_filter,
        )

        docs: list[ContextDocument] = []

        for i in range(len(results["ids"][0])):
            docs.append(
                ContextDocument(
                    id=results["ids"][0][i],
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i],
                    token_count=len(results["documents"][0][i].split()),
                )
            )

        return docs