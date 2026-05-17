from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.types import ChatRequest, ModerationResult, PolicyAction, PolicyDecision, UserSession


@dataclass(frozen=True)
class PolicyRule:
    rule_id: str
    description: str
    condition: dict[str, Any]
    action: PolicyAction
    severity: str
    enabled: bool = True


class PolicyEngine:
    def __init__(self, policy_path: Path, logger: Any | None = None) -> None:
        self.policy_path = policy_path
        self._rules: list[PolicyRule] = []
        self._last_loaded = 0.0
        self._logger = logger
        self.reload_policies()

    def reload_policies(self) -> None:
        if not self.policy_path.exists():
            raise FileNotFoundError(f"Policy file not found: {self.policy_path}")
        stat = self.policy_path.stat()
        if stat.st_mtime <= self._last_loaded:
            return
        with self.policy_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        self._rules = [
            PolicyRule(
                rule_id=str(item.get("id", index)),
                description=item.get("description", ""),
                condition=item.get("condition", {}),
                action=PolicyAction(item.get("action", "deny")),
                severity=item.get("severity", "medium"),
                enabled=item.get("enabled", True),
            )
            for index, item in enumerate(data.get("policies", []))
            if item.get("enabled", True)
        ]
        self._last_loaded = stat.st_mtime
        if self._logger:
            self._logger.info("Reloaded %d policy rules", len(self._rules))

    def evaluate(self, request: ChatRequest, moderation: ModerationResult, session: UserSession) -> PolicyDecision:
        self.reload_policies()
        violated_rules: list[str] = []
        reasons: list[str] = []
        allowed = True
        effective_action = PolicyAction.ALLOW
        max_risk = moderation.risk_score

        for rule in self._rules:
            if not rule.enabled:
                continue
            condition = rule.condition
            if self._matches_condition(request, moderation, session, condition):
                violated_rules.append(rule.rule_id)
                reasons.append(rule.description or f"matched {rule.rule_id}")
                if rule.action == PolicyAction.DENY:
                    allowed = False
                    effective_action = PolicyAction.DENY
                elif rule.action == PolicyAction.REVIEW and effective_action != PolicyAction.DENY:
                    allowed = False
                    effective_action = PolicyAction.REVIEW
                elif rule.action == PolicyAction.REDIRECT and effective_action not in (PolicyAction.DENY, PolicyAction.REVIEW):
                    effective_action = PolicyAction.REDIRECT
                max_risk = max(max_risk, 0.75 if rule.severity == "high" else 0.5)

        return PolicyDecision(
            request_id=request.request_id,
            timestamp=request.timestamp,
            source_module="guardrails.policy_engine",
            allowed=allowed,
            action=effective_action,
            risk_score=max_risk,
            reasons=reasons,
            violated_rules=violated_rules,
            recommended_next_steps=["escalate" if effective_action == PolicyAction.REVIEW else "none"],
        )

    @staticmethod
    def _matches_condition(request: ChatRequest, moderation: ModerationResult, session: UserSession, condition: dict[str, Any]) -> bool:
        if not condition:
            return False
        if condition.get("min_moderation_score") is not None and moderation.risk_score < float(condition["min_moderation_score"]):
            return False
        if condition.get("user_role") and condition["user_role"] not in session.roles:
            return False
        if condition.get("contains") and condition["contains"].lower() not in request.prompt.lower():
            return False
        return True
