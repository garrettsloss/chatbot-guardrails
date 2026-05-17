from __future__ import annotations

import asyncio
from typing import Any

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

    async def process(self, request: ChatRequest, session: UserSession | None, client_ip: str | None = None) -> ChatResponse:
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

        input_moderation = await self.input_filter.filter(request, self.config.moderation_thresholds.get("review", 0.5))
        pipeline.input_moderation = input_moderation
        await self._publish(EventType.MODERATION, request.request_id, input_moderation.model_dump())

        policy_decision = self.policy_engine.evaluate(request, input_moderation, session)
        pipeline.policy_decision = policy_decision
        await self._publish(EventType.POLICY_DECISION, request.request_id, policy_decision.model_dump())

        if not policy_decision.allowed:
            raise PermissionError("Policy rejected request")

        retrieved_context = await self.retriever.retrieve(request)
        pipeline.retrieved_context = retrieved_context
        await self._publish(EventType.CONTEXT_RETRIEVAL, request.request_id, {"documents": len(retrieved_context)})

        sanitized_context = await self.sanitizer.sanitize_documents(retrieved_context)
        pipeline.sanitized_context = sanitized_context
        await self._publish(EventType.CONTEXT_SANITIZATION, request.request_id, {"documents": len(sanitized_context)})

        prompt_payload = self.prompt_builder.build(request.prompt, sanitized_context, request.conversation_history)
        pipeline.prompt_payload = prompt_payload.model_dump() if hasattr(prompt_payload, "model_dump") else prompt_payload.__dict__
        await self._publish(EventType.PROMPT_BUILD, request.request_id, {"token_estimate": prompt_payload.token_estimate})

        llm_response = await self.llm_client.generate(prompt_payload.__dict__ if hasattr(prompt_payload, "__dict__") else prompt_payload, timeout=60)
        pipeline.llm_response = llm_response.response_text
        await self._publish(EventType.LLM_CALL, request.request_id, {"response_length": len(llm_response.response_text)})

        output_moderation = await self.output_filter.filter(llm_response.response_text, self.config.moderation_thresholds.get("block", 0.8))
        pipeline.output_moderation = output_moderation
        await self._publish(EventType.MODERATION, request.request_id, output_moderation.model_dump())

        if not output_moderation.allowed:
            raise PermissionError("Output moderation failed")

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
