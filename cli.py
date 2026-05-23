from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config import get_config
from core.events import EventBus
from core.types import ChatRequest, UserSession
from guardrails.context_sanitizer import ContextSanitizer
from guardrails.input_filter import (
    HarmfulContentDetector,
    InputFilter,
    JailbreakDetector,
    LLMHarmfulContentDetector,
    PIIDetector,
    PromptInjectionDetector,
    RegexRuleDetector,
    SpamDetector,
    UnicodeObfuscationDetector,
)
from guardrails.output_filter import OutputFilter
from guardrails.policy_engine import PolicyEngine
from guardrails.topic_guard import (
    TopicEmbeddingDetector,
    TopicKeywordDetector,
    get_topic_profile,
)
from llm.client import OpenAIClientAdapter
from llm.prompt_builder import PromptBuilder
from observability.audit import AuditLogger, SilentAuditProvider
from rag.context_retrieval import ContextRetriever, OpenAIEmbeddingProvider
from security.auth import AuthManager
from security.rate_limit import RateLimiter
from tools.gateway import ToolRegistry
from pipeline.orchestrator import Orchestrator
from core.chroma_db import ChromaVectorDB


def build_components(config: Any, topic_profile: dict[str, Any]) -> Orchestrator:
    event_bus = EventBus()
    audit_logger = AuditLogger([SilentAuditProvider()])
    auth_manager = AuthManager(config)
    rate_limiter = RateLimiter(config)

    vector_db = ChromaVectorDB(path="./data/chroma")
    embedding_provider = OpenAIEmbeddingProvider(config.api_key, config.embedding_model)
    retriever = ContextRetriever(embedding_provider, vector_db)
    sanitizer = ContextSanitizer()

    # Layered input guardrails — every request passes through all detectors in parallel:
    #   Layer 1 — regex rules: blocks shutdown/delete commands
    #   Layer 2 — harmful content (regex): fast pattern-match for known dangerous phrases
    #   Layer 3 — harmful content (LLM): GPT-4.1-mini classifier catches creative rephrasing
    #   Layer 4 — prompt injection & jailbreak phrase detection
    #   Layer 5 — PII detection (SSN, credit card patterns)
    #   Layer 6 — spam detection (repeated-word flooding)
    #   Layer 7 — unicode obfuscation detection
    #   Layer 8 — topic keyword pre-filter (fast, no API call)
    #   Layer 9 — topic embedding similarity via Ada-002 (semantic gate, definitive off-topic block)
    # Layers 2–3 run concurrently; the LLM classifier adds one fast parallel API call per turn.
    input_filter = InputFilter(detectors=[
        RegexRuleDetector([r"\bshutdown\b", r"\bdelete\b"]),
        HarmfulContentDetector(),
        LLMHarmfulContentDetector(),
        PromptInjectionDetector(),
        JailbreakDetector(),
        PIIDetector(),
        SpamDetector(),
        UnicodeObfuscationDetector(),
        TopicKeywordDetector(topic_profile["keywords"]),
        TopicEmbeddingDetector(
            embedding_provider,
            topic_profile["anchor_phrases"],
            threshold=config.topic_relevance_threshold,
        ),
    ])

    policy_engine = PolicyEngine(Path("policies.yml"))

    # Topic-aware output filter — flags substantive responses with no topic content
    output_filter = OutputFilter(topic_keywords=topic_profile["keywords"])

    prompt_builder = PromptBuilder(model_template=topic_profile["system_prompt"])
    llm_client = OpenAIClientAdapter(config.api_key, config.openai_model)
    tool_registry = ToolRegistry()

    rejection_messages = {
        "harmful": "I'm not able to help with that request. If you have a gardening question, I'd be happy to help!",
        "injection": topic_profile["injection_response"],
        "off_topic": topic_profile["off_topic_response"],
        "output_failure": "I'm sorry, I couldn't generate a safe response. Please try rephrasing.",
        "default": topic_profile["off_topic_response"],
    }

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
        rejection_messages=rejection_messages,
    )


def build_request(
    prompt: str,
    user_id: str,
    session_id: str,
    history: list[dict[str, str]] | None = None,
) -> ChatRequest:
    return ChatRequest(
        request_id=str(uuid.uuid4()),
        timestamp=datetime.utcnow(),
        source_module="cli",
        user_id=user_id,
        session_id=session_id,
        prompt=prompt,
        conversation_history=history or [],
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


def _setup_logging() -> None:
    os.makedirs("logs", exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console: WARNING+ only — keeps the chatbot output uncluttered.
    # Denial warnings are logged at WARNING so they surface here.
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)

    # File: INFO+ — full audit trail including token usage, every denial
    # with reasons, and pipeline events. Rotates at 5 MB.
    from logging.handlers import RotatingFileHandler
    fh = RotatingFileHandler("logs/guardrails.log", maxBytes=5_000_000, backupCount=3)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(fh)


async def _chat_loop(
    orchestrator: Orchestrator,
    session: UserSession,
    topic_profile: dict[str, Any],
    config: Any,
) -> None:
    """
    Runs the interactive chat loop inside a single persistent event loop.

    Using a single asyncio.run() call (rather than one per turn) keeps the
    httpx connection pool alive across turns, eliminating the 'Event loop is
    closed' errors that occur when asyncio.run() tears down its loop while
    httpx still has pending cleanup tasks.

    Blocking input() is dispatched to a thread executor so the event loop
    remains free to process async work while waiting for the user to type.
    """
    display_name = topic_profile["display_name"]
    conversation_history: list[dict[str, str]] = []
    loop = asyncio.get_running_loop()
    turn = 0

    print(f"\n{'=' * 60}")
    print(f"  {display_name}")
    print(f"{'=' * 60}")
    print("Type 'quit' or 'exit' to end the conversation.\n")

    while True:
        # Read user input without blocking the event loop
        try:
            user_input = await loop.run_in_executor(None, lambda: input("You: "))
            user_input = user_input.strip()
        except (EOFError, KeyboardInterrupt, asyncio.CancelledError):
            print(f"\n{display_name}: Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q", "bye"):
            print(f"\n{display_name}: Goodbye! Happy gardening!")
            break

        request = build_request(user_input, "cli-user", session.session_id, conversation_history)

        try:
            response = await orchestrator.process(request, session)
        except PermissionError as exc:
            print(f"\n{display_name}: {exc}\n")
            continue
        except Exception as exc:
            logging.error("Pipeline error: %s", exc, exc_info=True)
            print(f"\n{display_name}: Something went wrong. Please try again.\n")
            continue

        # Log token usage per turn to the audit file (not shown on console)
        turn += 1
        total_tokens = (response.metadata or {}).get("total_tokens", 0)
        if total_tokens:
            logging.getLogger(__name__).info(
                "TOKEN_USAGE | turn=%d | total=%d | prompt=%s | completion=%s",
                turn,
                total_tokens,
                (response.metadata or {}).get("prompt_tokens", "?"),
                (response.metadata or {}).get("completion_tokens", "?"),
            )

        print(f"\n{display_name}: {response.response_text}\n")

        # Only include genuine (non-blocked) exchanges in history so the LLM
        # does not see rejected turns as prior context.
        if response.safe:
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": response.response_text})

            # Trim to the configured rolling window to prevent context overflow.
            # Each turn = 2 entries (user + assistant); oldest are dropped first.
            max_messages = config.max_history_turns * 2
            if len(conversation_history) > max_messages:
                conversation_history = conversation_history[-max_messages:]


def main() -> int:
    _setup_logging()

    config = get_config()
    topic_profile = get_topic_profile(config.chatbot_topic)

    orchestrator = build_components(config, topic_profile)

    session_id = str(uuid.uuid4())
    session = build_session("cli-user", session_id, ["user"])

    # A single asyncio.run() keeps one event loop alive for the entire session.
    # This prevents the 'Event loop is closed' errors that httpx raises when
    # asyncio.run() tears down its loop while connection-pool cleanup tasks are
    # still pending.
    try:
        asyncio.run(_chat_loop(orchestrator, session, topic_profile, config))
    except KeyboardInterrupt:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
