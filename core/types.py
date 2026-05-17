from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PolicyAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REVIEW = "review"
    REDIRECT = "redirect"


class ToolPermission(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    CONDITIONAL = "conditional"


class EventType(str, Enum):
    REQUEST_RECEIVED = "request_received"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    POLICY_DECISION = "policy_decision"
    MODERATION = "moderation"
    TOOL_EXECUTION = "tool_execution"
    LLM_CALL = "llm_call"
    CONTEXT_RETRIEVAL = "context_retrieval"
    CONTEXT_SANITIZATION = "context_sanitization"
    PROMPT_BUILD = "prompt_build"
    RESPONSE_DELIVERED = "response_delivered"
    ERROR = "error"


class SharedBaseModel(BaseModel):
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_module: str = Field(..., description="Source module name")

    model_config = ConfigDict(extra="forbid", stricttypes=True)


class ModerationResult(SharedBaseModel):
    allowed: bool = Field(...)
    risk_score: float = Field(..., ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    policy_references: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class PolicyDecision(SharedBaseModel):
    allowed: bool = Field(...)
    action: PolicyAction = Field(...)
    risk_score: float = Field(..., ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    violated_rules: list[str] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)


class ToolCall(SharedBaseModel):
    tool_name: str = Field(...)
    arguments: dict[str, Any] = Field(default_factory=dict)
    permission: ToolPermission = Field(...)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResult(SharedBaseModel):
    tool_name: str = Field(...)
    success: bool = Field(...)
    output: str = Field(default="")
    error: str | None = Field(default=None)
    execution_time_ms: int | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextDocument(SharedBaseModel):
    document_id: str = Field(...)
    content: str = Field(...)
    source: str = Field(...)
    metadata: dict[str, Any] = Field(default_factory=dict)
    relevance_score: float | None = Field(default=None, ge=0.0, le=1.0)
    token_count: int | None = Field(default=None, ge=0)


class AuditEvent(SharedBaseModel):
    event_type: EventType = Field(...)
    trace_id: str = Field(...)
    payload: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = Field(default=None)


class UserSession(SharedBaseModel):
    session_id: str = Field(...)
    user_id: str = Field(...)
    roles: list[str] = Field(default_factory=list)
    api_key_id: str | None = Field(default=None)
    is_active: bool = Field(default=True)
    expires_at: datetime | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GuardrailDecision(SharedBaseModel):
    decision_type: str = Field(...)
    allowed: bool = Field(...)
    risk_score: float = Field(..., ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)


class PipelineContext(SharedBaseModel):
    request_id: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_module: str = Field(default="pipeline.orchestrator")
    request: ChatRequest | None = None
    session: UserSession | None = None
    input_moderation: ModerationResult | None = None
    policy_decision: PolicyDecision | None = None
    retrieved_context: list[ContextDocument] = Field(default_factory=list)
    sanitized_context: list[ContextDocument] = Field(default_factory=list)
    prompt_payload: dict[str, Any] | None = None
    llm_response: str | None = None
    output_moderation: ModerationResult | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    audit_events: list[AuditEvent] = Field(default_factory=list)


class ChatRequest(SharedBaseModel):
    user_id: str = Field(...)
    session_id: str = Field(...)
    prompt: str = Field(...)
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    locale: str | None = Field(default=None)
    channel: str | None = Field(default=None)


class ChatResponse(SharedBaseModel):
    response_text: str = Field(...)
    safe: bool = Field(default=True)
    policy_action: PolicyAction | None = Field(default=None)
    reasons: list[str] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
