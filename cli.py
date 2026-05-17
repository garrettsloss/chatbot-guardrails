from __future__ import annotations

import argparse
import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config import get_config
from core.events import EventBus
from core.types import ChatRequest, ContextDocument, UserSession
from guardrails.context_sanitizer import ContextSanitizer
from guardrails.input_filter import InputFilter
from guardrails.output_filter import OutputFilter
from guardrails.policy_engine import PolicyEngine
from llm.client import OpenAIClientAdapter
from llm.prompt_builder import PromptBuilder
from observability.audit import AuditLogger, ConsoleAuditProvider
from rag.context_retrieval import ContextRetriever, OpenAIEmbeddingProvider, VectorDBClient
from security.auth import AuthManager
from security.rate_limit import RateLimiter
from tools.gateway import ToolRegistry
from pipeline.orchestrator import Orchestrator


class InMemoryVectorDB(VectorDBClient):
    async def upsert(self, documents: list[ContextDocument]) -> None:
        pass

    async def search(self, embedding: list[float], top_k: int, metadata_filter: dict[str, Any] | None = None) -> list[ContextDocument]:
        return []


def build_components(config: Any) -> Orchestrator:
    event_bus = EventBus()
    audit_logger = AuditLogger([ConsoleAuditProvider()])
    auth_manager = AuthManager(config)
    rate_limiter = RateLimiter(config)
    input_filter = InputFilter()
    policy_engine = PolicyEngine(Path("policies.yml"))
    retriever = ContextRetriever(OpenAIEmbeddingProvider(config.api_key, config.embedding_model), InMemoryVectorDB())
    sanitizer = ContextSanitizer()
    prompt_builder = PromptBuilder(model_template="You are a safety-first assistant.")
    llm_client = OpenAIClientAdapter(config.api_key, config.openai_model)
    output_filter = OutputFilter()
    tool_registry = ToolRegistry()

    return Orchestrator(
        config=config,
        event_bus=event_bus,
        authenticator=auth_manager,
        rate_limiter=rate_limiter,
        input_filter=input_filter,
        policy_engine=policy_engine,
        retriever=retriever,
        sanitizer=sanitizer,
        prompt_builder=prompt_builder,
        llm_client=llm_client,
        output_filter=output_filter,
        tool_gateway=tool_registry,
        audit_logger=audit_logger,
    )


def build_request(prompt: str, user_id: str, session_id: str) -> ChatRequest:
    return ChatRequest(
        request_id=str(uuid.uuid4()),
        timestamp=datetime.utcnow(),
        source_module="cli",
        user_id=user_id,
        session_id=session_id,
        prompt=prompt,
        conversation_history=[],
        metadata={"source": "cli"},
    )


def build_session(user_id: str, session_id: str, roles: list[str]) -> UserSession:
    return UserSession(
        request_id=str(uuid.uuid4()),
        timestamp=datetime.utcnow(),
        source_module="cli",
        session_id=session_id,
        user_id=user_id,
        roles=roles,
        is_active=True,
        metadata={"source": "cli"},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the chatbot guardrails pipeline from CLI.")
    parser.add_argument("--prompt", required=True, help="User prompt text")
    parser.add_argument("--user-id", default="user-1", help="User identifier")
    parser.add_argument("--session-id", default="session-1", help="Session identifier")
    parser.add_argument("--roles", default="user", help="Comma-separated roles")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = get_config()

    orchestrator = build_components(config)
    request = build_request(args.prompt, args.user_id, args.session_id)
    session = build_session(args.user_id, args.session_id, [role.strip() for role in args.roles.split(",")])

    try:
        response = asyncio.run(orchestrator.process(request, session))
        if hasattr(response, "response_text"):
            print(response.response_text)
            return 0
    except Exception as exc:
        logging.error("Pipeline error: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
