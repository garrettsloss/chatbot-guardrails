from __future__ import annotations

from core.types import AuditEvent, EventType, ModerationResult


class OutputFilter:
    async def filter(self, response_text: str, threshold: float = 0.8) -> ModerationResult:
        risk_score = 0.1 if len(response_text) < 1000 else 0.3
        allowed = risk_score < threshold
        return ModerationResult(
            request_id="output",
            timestamp=__import__("datetime").datetime.utcnow(),
            source_module="guardrails.output_filter",
            allowed=allowed,
            risk_score=risk_score,
            reasons=["safe" if allowed else "review_required"],
            details={"length": len(response_text)},
        )

    def audit_event(self, request_id: str, payload: dict[str, object]) -> AuditEvent:
        return AuditEvent(
            request_id=request_id,
            timestamp=__import__("datetime").datetime.utcnow(),
            source_module="guardrails.output_filter",
            event_type=EventType.MODERATION,
            trace_id=request_id,
            payload=payload,
        )
