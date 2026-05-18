from __future__ import annotations

import re

from core.types import AuditEvent, EventType, ModerationResult


class OutputFilter:
    """
    Layer 4 (output): post-generation topic and safety check.

    Flags responses that are longer than a trivial length but contain no
    topic-relevant keywords, which is a signal that the LLM may have been
    manipulated into an off-topic answer.
    """

    def __init__(self, topic_keywords: list[str] | None = None) -> None:
        keywords = [kw.lower() for kw in (topic_keywords or [])]
        # Pre-compile a single word-boundary regex for efficient matching.
        # Word boundaries prevent short keywords like "pot" from matching
        # unrelated words like "hypotenuse".
        if keywords:
            pattern = r'\b(?:' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
            self._topic_pattern: re.Pattern | None = re.compile(pattern)
        else:
            self._topic_pattern = None

    async def filter(self, response_text: str, threshold: float = 0.8) -> ModerationResult:
        length_risk = 0.1 if len(response_text) < 1000 else 0.3

        topic_risk = 0.0
        reason = "safe"
        if self._topic_pattern and len(response_text) > 80:
            has_topic_content = bool(self._topic_pattern.search(response_text.lower()))
            if not has_topic_content:
                # Response is substantive but contains zero topic keywords —
                # likely off-topic LLM output despite the system prompt.
                topic_risk = 0.85
                reason = "off_topic_output"

        risk_score = max(length_risk, topic_risk)
        allowed = risk_score < threshold

        return ModerationResult(
            request_id="output",
            timestamp=__import__("datetime").datetime.utcnow(),
            source_module="guardrails.output_filter",
            allowed=allowed,
            risk_score=risk_score,
            reasons=[reason if allowed else f"blocked_{reason}"],
            details={"length": len(response_text), "topic_risk": topic_risk},
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
