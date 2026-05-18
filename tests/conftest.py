"""
Shared fixtures and helpers for the guardrails test suite.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Plain helpers (not fixtures) — usable in any test file with a direct import
# ---------------------------------------------------------------------------

def make_request(prompt: str, history: list[dict] | None = None):
    from core.types import ChatRequest
    return ChatRequest(
        request_id=str(uuid.uuid4()),
        timestamp=datetime.utcnow(),
        source_module="test",
        user_id="test-user",
        session_id="test-session",
        prompt=prompt,
        conversation_history=history or [],
        metadata={"source": "test"},
    )


def make_session():
    from core.types import UserSession
    return UserSession(
        request_id=str(uuid.uuid4()),
        timestamp=datetime.utcnow(),
        source_module="test",
        session_id="test-session",
        user_id="test-user",
        roles=["user"],
        is_active=True,
        metadata={},
    )


def make_moderation(request, *, risk_score: float = 0.0, allowed: bool = True):
    """Create a ModerationResult for use in PolicyEngine tests."""
    from core.types import ModerationResult
    return ModerationResult(
        request_id=request.request_id,
        timestamp=request.timestamp,
        source_module="test",
        allowed=allowed,
        risk_score=risk_score,
        reasons=["test"],
        details={},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def session():
    return make_session()


@pytest.fixture
def gardening_profile():
    from guardrails.topic_guard import get_topic_profile
    return get_topic_profile("gardening")


@pytest.fixture
def mock_embedding_provider():
    """
    Deterministic mock for the Azure Ada-002 embedding provider.

    Returns a unit vector in dimension-0 for gardening-related text and a
    unit vector in dimension-1 for everything else.  Cosine similarity
    between the two vectors is 0.0 — well below the 0.30 topic-relevance
    threshold — so off-topic queries are reliably blocked by
    TopicEmbeddingDetector without real API calls.
    """
    GARDENING_EMB = [1.0] + [0.0] * 1535   # dim-0 unit vector
    OFF_TOPIC_EMB = [0.0, 1.0] + [0.0] * 1534  # dim-1 unit vector (similarity = 0)

    _KEYWORDS = {
        "garden", "plant", "grow", "soil", "seed", "flower", "vegetable",
        "prune", "water", "fertilize", "compost", "herb", "rose", "tomato",
        "gardening", "weed", "mulch", "harvest", "irrigation", "horticulture",
        "shrub", "tree", "lawn", "grass", "cutting", "propagation", "blossom",
        "root", "leaf", "branch", "bulb", "perennial", "annual", "fruit",
        # common gardening subjects that tests ask about:
        "pepper", "peppers", "garlic", "basil", "mint", "carrot", "lettuce",
        "cucumber", "strawberr", "blueberr", "berry", "berries",
    }

    provider = AsyncMock()

    async def _embed(text: str) -> list[float]:
        return GARDENING_EMB if any(kw in text.lower() for kw in _KEYWORDS) else OFF_TOPIC_EMB

    provider.embed.side_effect = _embed
    return provider


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.moderation_thresholds = {"block": 0.8, "review": 0.5}
    cfg.topic_relevance_threshold = 0.30
    cfg.rate_limits = {"user_per_minute": 60, "ip_per_minute": 120}
    return cfg


@pytest.fixture
def mock_llm():
    """LLM that returns a safe gardening response by default."""
    from core.types import ChatResponse
    llm = AsyncMock()
    llm.generate.return_value = ChatResponse(
        request_id="test-llm",
        timestamp=datetime.utcnow(),
        source_module="test",
        response_text=(
            "Great question! The best approach is to prepare your garden soil "
            "with compost, ensure regular watering, and choose plants suited to "
            "your local climate."
        ),
        safe=True,
        policy_action=None,
        reasons=[],
        tool_results=[],
        metadata={"total_tokens": 60, "prompt_tokens": 40, "completion_tokens": 20},
    )
    return llm


@pytest.fixture
def policies_file(tmp_path):
    """Minimal policies.yml written to a temp directory."""
    p = tmp_path / "policies.yml"
    p.write_text(
        "policies:\n"
        "  - id: block_high_risk_inputs\n"
        "    description: Block requests when moderation risk >= 0.7\n"
        "    condition:\n"
        "      min_moderation_score: 0.7\n"
        "    action: deny\n"
        "    severity: high\n"
        "    enabled: true\n"
    )
    return p


@pytest.fixture
def test_orchestrator(
    mock_config, mock_embedding_provider, mock_llm, gardening_profile, policies_file
):
    """
    Full Orchestrator instance with mocked external APIs.

    Real components:  InputFilter chain, OutputFilter, PolicyEngine,
                      ContextSanitizer, PromptBuilder, EventBus, ToolRegistry.
    Mocked components: LLM client, embedding provider, vector DB, rate limiter.

    This lets integration and adversarial tests exercise every guardrail layer
    without making live API calls.
    """
    from core.events import EventBus
    from guardrails.context_sanitizer import ContextSanitizer
    from guardrails.input_filter import (
        HarmfulContentDetector, InputFilter, JailbreakDetector,
        PIIDetector, PromptInjectionDetector, RegexRuleDetector,
        SpamDetector, UnicodeObfuscationDetector,
    )
    from guardrails.output_filter import OutputFilter
    from guardrails.policy_engine import PolicyEngine
    from guardrails.topic_guard import TopicEmbeddingDetector, TopicKeywordDetector
    from llm.prompt_builder import PromptBuilder
    from observability.audit import AuditLogger, SilentAuditProvider
    from pipeline.orchestrator import Orchestrator
    from rag.context_retrieval import ContextRetriever
    from tools.gateway import ToolRegistry

    mock_vector_db = AsyncMock()
    mock_vector_db.search.return_value = []

    mock_rate_limiter = AsyncMock()
    mock_rate_limiter.check_ip.return_value = True
    mock_rate_limiter.check_user.return_value = True

    input_filter = InputFilter(detectors=[
        RegexRuleDetector([r"\bshutdown\b", r"\bdelete\b"]),
        HarmfulContentDetector(),
        PromptInjectionDetector(),
        JailbreakDetector(),
        PIIDetector(),
        SpamDetector(),
        UnicodeObfuscationDetector(),
        TopicKeywordDetector(gardening_profile["keywords"]),
        TopicEmbeddingDetector(
            mock_embedding_provider,
            gardening_profile["anchor_phrases"],
            threshold=0.30,
        ),
    ])

    return Orchestrator(
        config=mock_config,
        event_bus=EventBus(),
        authenticator=MagicMock(),
        rate_limiter=mock_rate_limiter,
        input_filter=input_filter,
        policy_engine=PolicyEngine(policies_file),
        retriever=ContextRetriever(mock_embedding_provider, mock_vector_db),
        sanitizer=ContextSanitizer(),
        prompt_builder=PromptBuilder(model_template=gardening_profile["system_prompt"]),
        llm_client=mock_llm,
        output_filter=OutputFilter(topic_keywords=gardening_profile["keywords"]),
        tool_gateway=ToolRegistry(),
        audit_logger=AuditLogger([SilentAuditProvider()]),
        rejection_messages={
            "harmful": "I'm not able to help with that. Please ask a gardening question.",
            "injection": gardening_profile["injection_response"],
            "off_topic": gardening_profile["off_topic_response"],
            "output_failure": "I couldn't generate a safe response.",
            "default": gardening_profile["off_topic_response"],
        },
    )
