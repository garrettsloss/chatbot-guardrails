from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from typing import Any

from core.types import ChatRequest, ModerationResult


class InputFilterDetector(ABC):
    name: str
    weight: float

    def __init__(self, name: str, weight: float = 1.0) -> None:
        self.name = name
        self.weight = weight

    @abstractmethod
    async def analyze(self, request: ChatRequest) -> ModerationResult:
        pass


class RegexRuleDetector(InputFilterDetector):
    def __init__(self, patterns: list[str], weight: float = 1.0) -> None:
        super().__init__("regex_rule", weight)
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]

    async def analyze(self, request: ChatRequest) -> ModerationResult:
        matches: list[str] = []
        for pattern in self.patterns:
            if pattern.search(request.prompt):
                matches.append(pattern.pattern)
        allowed = len(matches) == 0
        risk_score = 0.8 if matches else 0.0
        return ModerationResult(
            request_id=request.request_id,
            timestamp=request.timestamp,
            source_module="guardrails.input_filter.regex_rule",
            allowed=allowed,
            risk_score=risk_score,
            reasons=["regex_rule_match" if matches else "clean"],
            details={"matches": matches},
        )


class PromptInjectionDetector(InputFilterDetector):
    def __init__(self, weight: float = 1.0) -> None:
        super().__init__("prompt_injection", weight)
        self.injection_indicators = ["ignore previous instructions", "forget your instructions", "prompt injection"]

    async def analyze(self, request: ChatRequest) -> ModerationResult:
        found = [marker for marker in self.injection_indicators if marker in request.prompt.lower()]
        allowed = not bool(found)
        risk_score = 0.9 if found else 0.0
        return ModerationResult(
            request_id=request.request_id,
            timestamp=request.timestamp,
            source_module="guardrails.input_filter.prompt_injection",
            allowed=allowed,
            risk_score=risk_score,
            reasons=["prompt_injection_detected" if found else "no_injection"],
            details={"found": found},
        )


class JailbreakDetector(InputFilterDetector):
    def __init__(self, weight: float = 1.0) -> None:
        super().__init__("jailbreak_phrase", weight)
        self.jailbreak_phrases = ["jailbreak", "bypass safety", "secret instructions"]

    async def analyze(self, request: ChatRequest) -> ModerationResult:
        found = [phrase for phrase in self.jailbreak_phrases if phrase in request.prompt.lower()]
        allowed = not bool(found)
        risk_score = 0.85 if found else 0.0
        return ModerationResult(
            request_id=request.request_id,
            timestamp=request.timestamp,
            source_module="guardrails.input_filter.jailbreak_phrase",
            allowed=allowed,
            risk_score=risk_score,
            reasons=["jailbreak_detected" if found else "no_jailbreak"],
            details={"found": found},
        )


class HarmfulContentDetector(InputFilterDetector):
    """
    Blocks requests containing clearly dangerous or illegal content (e.g. bomb-making,
    weapons manufacturing, explicit violence instructions).

    This acts as a universal safety gate independent of the active topic, providing
    a last line of defence before off-topic requests even reach the embedding check.
    Uses specific multi-word patterns to minimise false positives (e.g. 'seed bomb'
    is not flagged because no action verb precedes it).
    """

    _PATTERNS = [
        # Bomb / explosive fabrication — covers "make a bomb", "build explosives", etc.
        r"\b(make|build|create|craft|assemble|fabricate)\s+a?\s*(bombs?|explosives?|grenades?|ied|detonators?)\b",
        # "how to / how do I" variants for bomb/explosive making
        r"\bhow\s+(to|do\s+i|can\s+i)\s+(make|build|create|craft)\s+(a\s+)?(bomb|explosives?|grenade)\b",
        # Chemical / biological weapons
        r"\b(chemical\s+weapon|bioweapon|nerve\s+agent|sarin|vx\s+gas|weaponize\s+.{0,20}(virus|bacteria|toxin))\b",
        # Explicit violence instructions
        r"\bhow\s+(to|do\s+i|can\s+i)\s+(kill|murder|harm|hurt|attack)\s+(someone|people|a\s+person|humans|everybody|anyone|civilians)\b",
        # Firearm / weapon manufacturing — covers "how to build" and "how do I build"
        r"\bhow\s+(to|do\s+i|can\s+i)\s+(make|build|manufacture|print|modify)\s+(a\s+)?(gun|firearm|rifle|pistol|silencer)\b",
        # Illicit drug synthesis
        r"\bhow\s+to\s+(make|synthesize|cook|produce|manufacture)\s+(meth|heroin|fentanyl|crack\s+cocaine|illegal\s+drug)\b",
    ]

    def __init__(self, weight: float = 3.0) -> None:
        super().__init__("harmful_content", weight)
        self.patterns = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in self._PATTERNS]

    async def analyze(self, request: ChatRequest) -> ModerationResult:
        matched = [p.pattern for p in self.patterns if p.search(request.prompt)]
        allowed = not bool(matched)
        return ModerationResult(
            request_id=request.request_id,
            timestamp=request.timestamp,
            source_module="guardrails.input_filter.harmful_content",
            allowed=allowed,
            risk_score=1.0 if matched else 0.0,
            reasons=["harmful_content_detected" if matched else "safe"],
            details={"matched_patterns": matched},
        )


class PIIDetector(InputFilterDetector):
    def __init__(self, weight: float = 1.0) -> None:
        super().__init__("pii", weight)
        self.patterns = [re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), re.compile(r"\b\d{16}\b")]

    async def analyze(self, request: ChatRequest) -> ModerationResult:
        found = []
        for pattern in self.patterns:
            if pattern.search(request.prompt):
                found.append(pattern.pattern)
        allowed = not bool(found)
        risk_score = 0.95 if found else 0.0
        return ModerationResult(
            request_id=request.request_id,
            timestamp=request.timestamp,
            source_module="guardrails.input_filter.pii",
            allowed=allowed,
            risk_score=risk_score,
            reasons=["pii_detected" if found else "clean"],
            details={"patterns": found},
        )


class SpamDetector(InputFilterDetector):
    def __init__(self, weight: float = 0.5) -> None:
        super().__init__("spam", weight)
        self.repeated_word_threshold = 5

    async def analyze(self, request: ChatRequest) -> ModerationResult:
        tokens = request.prompt.lower().split()
        repetitions = {token: tokens.count(token) for token in set(tokens)}
        spammy = [token for token, count in repetitions.items() if count >= self.repeated_word_threshold]
        allowed = not bool(spammy)
        risk_score = 0.6 if spammy else 0.0
        return ModerationResult(
            request_id=request.request_id,
            timestamp=request.timestamp,
            source_module="guardrails.input_filter.spam",
            allowed=allowed,
            risk_score=risk_score,
            reasons=["spam_detected" if spammy else "clean"],
            details={"spam_tokens": spammy},
        )


class UnicodeObfuscationDetector(InputFilterDetector):
    def __init__(self, weight: float = 0.75) -> None:
        super().__init__("unicode_obfuscation", weight)
        self.obfuscation_pattern = re.compile(r"[\u0400-\u04FF]|\u200b|\u200c|\u200d")

    async def analyze(self, request: ChatRequest) -> ModerationResult:
        matches = self.obfuscation_pattern.findall(request.prompt)
        allowed = len(matches) == 0
        risk_score = min(1.0, len(matches) / 20) if matches else 0.0
        return ModerationResult(
            request_id=request.request_id,
            timestamp=request.timestamp,
            source_module="guardrails.input_filter.unicode_obfuscation",
            allowed=allowed,
            risk_score=risk_score,
            reasons=["unicode_obfuscation_detected" if matches else "clean"],
            details={"matches": len(matches)},
        )


class InputFilter:
    def __init__(self, detectors: list[InputFilterDetector] | None = None) -> None:
        self.detectors = detectors or [
            RegexRuleDetector([r"\bshutdown\b", r"\bdelete\b"]),
            PromptInjectionDetector(),
            JailbreakDetector(),
            PIIDetector(),
            SpamDetector(),
            UnicodeObfuscationDetector(),
        ]

    async def filter(self, request: ChatRequest, threshold: float = 0.5) -> ModerationResult:
        tasks = [detector.analyze(request) for detector in self.detectors]
        results = await asyncio.gather(*tasks)

        total_weight = sum(detector.weight for detector in self.detectors)
        aggregated_score = sum(result.risk_score * self.detectors[index].weight for index, result in enumerate(results)) / max(total_weight, 1)
        # Only surface reasons from detectors that actually found something;
        # "clean" / "no_injection" etc. from passing detectors add noise to logs.
        reasons = [
            reason
            for result in results
            for reason in result.reasons
            if not result.allowed or result.risk_score > 0.05
        ] or ["no_specific_reason"]
        allowed = aggregated_score < threshold and all(result.allowed for result in results)

        return ModerationResult(
            request_id=request.request_id,
            timestamp=request.timestamp,
            source_module="guardrails.input_filter.chain",
            allowed=allowed,
            risk_score=aggregated_score,
            reasons=reasons,
            details={"stages": [result.dict() for result in results]},
        )
