from __future__ import annotations

import logging
from typing import Any

from core.chroma_db import ChromaVectorDB

logger = logging.getLogger(__name__)
from core.config import AppConfig
from core.events import EventBus
from core.types import (
    AuditEvent,
    ChatRequest,
    ChatResponse,
    ContextDocument,
    EventType,
    ModerationResult,
    PipelineContext,
    PolicyAction,
    PolicyDecision,
    ToolCall,
    ToolResult,
    UserSession,
)


class Orchestrator:
    def __init__(
        self,
        config: AppConfig,
        event_bus: EventBus,
        authenticator: Any,
        rate_limiter: Any,
        input_filter: Any,
        policy_engine: Any,
        retriever: Any,
        sanitizer: Any,
        prompt_builder: Any,
        llm_client: Any,
        output_filter: Any,
        tool_gateway: Any,
        audit_logger: Any,
        rejection_messages: dict[str, str] | None = None,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.authenticator = authenticator
        self.rate_limiter = rate_limiter
        self.input_filter = input_filter
        self.policy_engine = policy_engine
        self.retriever = retriever
        self.sanitizer = sanitizer
        self.prompt_builder = prompt_builder
        self.llm_client = llm_client
        self.output_filter = output_filter
        self.tool_gateway = tool_gateway
        self.audit_logger = audit_logger
        self.rejection_messages: dict[str, str] = rejection_messages or {
            "injection": "I detected an attempt to manipulate my instructions. Please ask a genuine question.",
            "off_topic": "I can only discuss the assigned topic. Please ask a relevant question.",
            "output_failure": "I'm sorry, I couldn't generate a safe response. Please try rephrasing.",
            "default": "I'm sorry, I can't help with that.",
        }

    async def process(
        self,
        request: ChatRequest,
        session: UserSession | None,
        client_ip: str | None = None,
    ) -> ChatResponse:
        pipeline = PipelineContext(
            request_id=request.request_id,
            timestamp=request.timestamp,
            source_module="pipeline.orchestrator",
            request=request,
            session=session,
        )
        await self._publish(EventType.REQUEST_RECEIVED, request.request_id, {"prompt": request.prompt})

        if session is None:
            raise PermissionError("Unauthenticated session")

        if client_ip and not await self.rate_limiter.check_ip(client_ip):
            raise PermissionError("IP rate limit exceeded")
        if not await self.rate_limiter.check_user(session.session_id):
            raise PermissionError("User rate limit exceeded")

        # --- Layer 1-3: input moderation (injection, jailbreak, PII, topic) ---
        input_moderation = await self.input_filter.filter(
            request, self.config.moderation_thresholds.get("review", 0.5)
        )
        pipeline.input_moderation = input_moderation
        await self._publish(EventType.MODERATION, request.request_id, input_moderation.model_dump())

        if not input_moderation.allowed:
            message = self._pick_rejection_message(input_moderation.reasons)
            return self._rejection_response(request, message, input_moderation.reasons)

        # --- Layer 3: policy engine (YAML rules) ---
        policy_decision = self.policy_engine.evaluate(request, input_moderation, session)
        pipeline.policy_decision = policy_decision
        await self._publish(EventType.POLICY_DECISION, request.request_id, policy_decision.model_dump())

        if not policy_decision.allowed:
            message = self._pick_rejection_message(policy_decision.reasons)
            return self._rejection_response(request, message, policy_decision.reasons)

        # --- RAG retrieval + context sanitization ---
        retrieved_context = await self.retriever.retrieve(request)
        pipeline.retrieved_context = retrieved_context
        await self._publish(EventType.CONTEXT_RETRIEVAL, request.request_id, {"documents": len(retrieved_context)})

        sanitized_context = await self.sanitizer.sanitize_documents(retrieved_context)
        pipeline.sanitized_context = sanitized_context
        await self._publish(EventType.CONTEXT_SANITIZATION, request.request_id, {"documents": len(sanitized_context)})

        # --- Prompt construction + LLM call ---
        prompt_payload = self.prompt_builder.build(request.prompt, sanitized_context, request.conversation_history)
        pipeline.prompt_payload = prompt_payload.model_dump() if hasattr(prompt_payload, "model_dump") else prompt_payload.__dict__
        await self._publish(EventType.PROMPT_BUILD, request.request_id, {"token_estimate": prompt_payload.token_estimate})

        llm_response = await self.llm_client.generate(
            prompt_payload.__dict__ if hasattr(prompt_payload, "__dict__") else prompt_payload,
            timeout=60,
        )
        pipeline.llm_response = llm_response.response_text

        # Handle Azure content filter or any other LLM-level block
        if not llm_response.safe:
            message = self._pick_rejection_message(llm_response.reasons)
            return self._rejection_response(request, message, llm_response.reasons)

        token_meta = llm_response.metadata or {}
        await self._publish(EventType.LLM_CALL, request.request_id, {
            "response_length": len(llm_response.response_text),
            "total_tokens": token_meta.get("total_tokens", 0),
            "prompt_tokens": token_meta.get("prompt_tokens", 0),
            "completion_tokens": token_meta.get("completion_tokens", 0),
        })

        # --- Layer 4: output moderation ---
        output_moderation = await self.output_filter.filter(
            llm_response.response_text, self.config.moderation_thresholds.get("block", 0.8)
        )
        pipeline.output_moderation = output_moderation
        await self._publish(EventType.MODERATION, request.request_id, output_moderation.model_dump())

        if not output_moderation.allowed:
            message = self.rejection_messages.get("output_failure", self.rejection_messages["default"])
            return self._rejection_response(request, message, output_moderation.reasons)

        # --- Tool execution (optional) ---
        final_response = llm_response
        if pipeline.tool_calls:
            tool_results: list[ToolResult] = []
            for tool_call in pipeline.tool_calls:
                result = await self.tool_gateway.execute(tool_call)
                tool_results.append(result)
                audit_event = self.tool_gateway.audit_event(tool_call, result)
                self.audit_logger.log(audit_event)
            final_response.tool_results = tool_results

        audit_event = AuditEvent(
            request_id=request.request_id,
            timestamp=request.timestamp,
            source_module="pipeline.orchestrator",
            event_type=EventType.RESPONSE_DELIVERED,
            trace_id=request.request_id,
            payload={
                "policy_allowed": policy_decision.allowed,
                "output_safe": output_moderation.allowed,
            },
        )
        self.audit_logger.log(audit_event)
        await self._publish(EventType.RESPONSE_DELIVERED, request.request_id, audit_event.payload)

        return final_response

    def _pick_rejection_message(self, reasons: list[str]) -> str:
        reasons_str = " ".join(reasons).lower()
        if "injection" in reasons_str or "jailbreak" in reasons_str:
            return self.rejection_messages.get("injection", self.rejection_messages["default"])
        if "off_topic" in reasons_str or "topic" in reasons_str:
            return self.rejection_messages.get("off_topic", self.rejection_messages["default"])
        return self.rejection_messages["default"]

    def _rejection_response(
        self, request: ChatRequest, message: str, reasons: list[str]
    ) -> ChatResponse:
        logger.warning(
            "DENIED | request_id=%s | user=%s | reasons=[%s] | prompt=%.80r",
            request.request_id,
            request.user_id,
            ", ".join(reasons),
            request.prompt,
        )
        return ChatResponse(
            request_id=request.request_id,
            timestamp=request.timestamp,
            source_module="pipeline.orchestrator",
            response_text=message,
            safe=False,
            policy_action=PolicyAction.DENY,
            reasons=reasons,
        )

    async def _publish(self, event_type: EventType, request_id: str, payload: dict[str, Any]) -> None:
        event = AuditEvent(
            request_id=request_id,
            timestamp=__import__("datetime").datetime.utcnow(),
            source_module="pipeline.orchestrator",
            event_type=event_type,
            trace_id=request_id,
            payload=payload,
        )
        await self.event_bus.publish(event)
