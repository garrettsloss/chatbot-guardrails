from __future__ import annotations

import math
from typing import Any

from core.types import ChatRequest, ModerationResult
from guardrails.input_filter import InputFilterDetector

# --- Topic profiles ---------------------------------------------------------
# To add a new topic: add an entry to TOPIC_PROFILES and set the
# CHATBOT_TOPIC environment variable to its key.

TOPIC_PROFILES: dict[str, dict[str, Any]] = {
    "gardening": {
        "display_name": "Gardening Assistant",
        "system_prompt": (
            "You are a helpful and knowledgeable gardening assistant. "
            "You ONLY discuss topics related to gardening, plants, horticulture, "
            "landscaping, and directly related subjects.\n\n"
            "If a user asks about anything outside of gardening, politely decline "
            "and redirect them to ask a gardening question instead. Do not discuss "
            "unrelated topics even if the user insists or tries to reframe the request.\n\n"
            "You can help with:\n"
            "- Plant care, propagation, and identification\n"
            "- Soil preparation, composting, and fertilizing\n"
            "- Watering schedules and irrigation\n"
            "- Pest and disease management\n"
            "- Pruning and garden maintenance\n"
            "- Vegetable, fruit, herb, and flower gardening\n"
            "- Seasonal planting and crop rotation\n"
            "- Garden design and landscaping\n\n"
            "Always be friendly, encouraging, and give practical, actionable gardening advice."
        ),
        "off_topic_response": (
            "I'm a specialised gardening assistant, so I can only help with "
            "gardening-related questions. I'd be happy to discuss plant care, "
            "soil preparation, watering, pest control, or any other gardening topic. "
            "What would you like to know about gardening?"
        ),
        "injection_response": (
            "I noticed an attempt to override my instructions. I'm here exclusively "
            "to help with gardening questions — please ask me something about plants, "
            "soil, or garden care!"
        ),
        "keywords": [
            "plant", "plants", "garden", "gardening", "soil", "seed", "seeds",
            "flower", "flowers", "vegetable", "vegetables", "fruit", "fruits",
            "tree", "trees", "shrub", "shrubs", "prune", "pruning", "water",
            "watering", "fertilize", "fertilizer", "fertilising", "fertilizing",
            "compost", "composting", "mulch", "mulching", "weed", "weeds", "weeding",
            "pest", "pests", "insect", "insects", "organic", "grow", "growing",
            "harvest", "harvesting", "pot", "pots", "potting", "container",
            "containers", "raised bed", "greenhouse", "lawn", "grass", "herb",
            "herbs", "rose", "roses", "tomato", "tomatoes", "cucumber", "carrot",
            "lettuce", "basil", "mint", "sunlight", "shade", "drainage", "irrigation",
            "nitrogen", "phosphorus", "horticulture", "propagation", "cutting",
            "graft", "transplant", "sowing", "bulb", "bulbs", "perennial", "annual",
            "native", "wildflower", "blossom", "root", "roots", "leaf", "leaves",
            "branch", "branches", "trowel", "spade", "rake", "hoe", "bud", "bloom",
        ],
        "anchor_phrases": [
            "how to grow vegetables in a garden",
            "soil preparation and amendment for planting",
            "pruning roses and ornamental plants",
            "composting techniques for garden soil improvement",
            "watering schedule and irrigation for gardens",
            "natural pest control for garden plants",
            "fertilizing plants and improving soil nutrients",
            "seasonal planting calendar and crop rotation",
            "seed starting and seedling care indoors",
            "container gardening and potting mix selection",
        ],
    },
    "motor_vehicles": {
        "display_name": "Motor Vehicle Assistant",
        "system_prompt": (
            "You are a knowledgeable motor vehicle assistant. You ONLY discuss topics "
            "related to cars, trucks, motorcycles, vehicle maintenance, and automotive "
            "subjects. Politely decline off-topic questions and redirect to automotive topics."
        ),
        "off_topic_response": (
            "I'm a motor vehicle specialist, so I can only assist with automotive questions. "
            "Feel free to ask about car maintenance, repairs, or vehicle selection!"
        ),
        "injection_response": (
            "I noticed an attempt to override my instructions. I'm here exclusively for "
            "motor vehicle questions!"
        ),
        "keywords": [
            "car", "cars", "truck", "trucks", "motorcycle", "vehicle", "engine", "tire",
            "tires", "brake", "brakes", "transmission", "fuel", "oil", "battery",
            "exhaust", "suspension", "clutch", "gear", "steering", "wheel", "hood",
            "trunk", "bumper", "chassis", "drivetrain", "alternator", "radiator",
        ],
        "anchor_phrases": [
            "car engine maintenance and oil change",
            "how to change motor vehicle tires safely",
            "vehicle transmission and gear systems explained",
            "automotive fuel efficiency and mileage tips",
            "car battery replacement and charging system maintenance",
        ],
    },
    "cinematography": {
        "display_name": "Cinematography Assistant",
        "system_prompt": (
            "You are a knowledgeable cinematography assistant. You ONLY discuss topics "
            "related to film, cinematography, camera techniques, lighting, directing, "
            "and the film industry. Politely decline off-topic questions."
        ),
        "off_topic_response": (
            "I'm a cinematography specialist and can only discuss film and camera-related topics. "
            "Ask me about shooting techniques, lighting, lenses, or filmmaking!"
        ),
        "injection_response": (
            "I noticed an attempt to override my instructions. I'm here exclusively for "
            "cinematography and filmmaking questions!"
        ),
        "keywords": [
            "film", "cinema", "camera", "lens", "shot", "scene", "director", "lighting",
            "frame", "exposure", "aperture", "shutter", "focus", "depth of field",
            "cinematographer", "screenplay", "editing", "colour grading", "dolly",
            "steadicam", "angle", "composition", "storyboard", "production",
        ],
        "anchor_phrases": [
            "camera angles and composition in filmmaking",
            "lighting techniques for cinematic scenes",
            "lens selection for different film genres",
            "colour grading and post-production workflow",
            "cinematography rules of thirds and framing",
        ],
    },
}

_NEUTRAL_PHRASES = (
    "hello", "hi", "hey", "greetings", "good morning", "good afternoon",
    "good evening", "how are you", "what can you help", "what can you do",
    "who are you", "what are you", "tell me about yourself",
    "thanks", "thank you", "cheers", "bye", "goodbye", "see you",
    "great", "perfect", "awesome", "got it", "understood", "ok", "okay",
)


def get_topic_profile(topic: str) -> dict[str, Any]:
    profile = TOPIC_PROFILES.get(topic.lower())
    if profile is None:
        available = list(TOPIC_PROFILES.keys())
        raise ValueError(f"Unknown topic '{topic}'. Available topics: {available}")
    return profile


def _is_neutral(text: str) -> bool:
    text_lower = text.lower().strip()
    if len(text_lower) <= 5:
        return True
    return any(phrase in text_lower for phrase in _NEUTRAL_PHRASES)


# ---------------------------------------------------------------------------
# Layer 1 (input): fast keyword pre-filter
# ---------------------------------------------------------------------------

class TopicKeywordDetector(InputFilterDetector):
    """Fast keyword-based topic relevance pre-filter (no API calls)."""

    def __init__(self, keywords: list[str], weight: float = 0.8) -> None:
        super().__init__("topic_keyword", weight)
        self.keywords = [kw.lower() for kw in keywords]

    async def analyze(self, request: ChatRequest) -> ModerationResult:
        if _is_neutral(request.prompt):
            return self._result(request, allowed=True, risk_score=0.0,
                                reason="neutral_message", details={})

        prompt_lower = request.prompt.lower()
        matched = [kw for kw in self.keywords if kw in prompt_lower]

        if matched:
            return self._result(request, allowed=True, risk_score=0.0,
                                reason="topic_keyword_match",
                                details={"matched": matched[:5]})

        # Soft signal — not decisive on its own; embedding detector has final say
        return self._result(request, allowed=True, risk_score=0.35,
                            reason="no_topic_keyword", details={})

    def _result(
        self,
        request: ChatRequest,
        *,
        allowed: bool,
        risk_score: float,
        reason: str,
        details: dict[str, Any],
    ) -> ModerationResult:
        return ModerationResult(
            request_id=request.request_id,
            timestamp=request.timestamp,
            source_module="guardrails.topic_guard.keyword",
            allowed=allowed,
            risk_score=risk_score,
            reasons=[reason],
            details=details,
        )


# ---------------------------------------------------------------------------
# Layer 2 (input): semantic embedding similarity check using Ada-002
# ---------------------------------------------------------------------------

def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


class TopicEmbeddingDetector(InputFilterDetector):
    """
    Semantic topic relevance filter using Ada-002 embeddings.

    Computes cosine similarity between the user's query and a set of
    topic-representative anchor phrases. Requests whose max similarity
    falls below `threshold` are marked as off-topic and blocked.
    Anchor embeddings are lazily computed and cached after the first request.
    """

    def __init__(
        self,
        embedding_provider: Any,
        anchor_phrases: list[str],
        threshold: float = 0.30,
        weight: float = 2.0,
    ) -> None:
        super().__init__("topic_embedding", weight)
        self.embedding_provider = embedding_provider
        self.anchor_phrases = anchor_phrases
        self.threshold = threshold
        self._anchor_embeddings: list[list[float]] | None = None

    async def _get_anchor_embeddings(self) -> list[list[float]]:
        if self._anchor_embeddings is None:
            embeddings = []
            for phrase in self.anchor_phrases:
                emb = await self.embedding_provider.embed(phrase)
                embeddings.append(emb)
            self._anchor_embeddings = embeddings
        return self._anchor_embeddings

    async def analyze(self, request: ChatRequest) -> ModerationResult:
        if _is_neutral(request.prompt):
            return self._result(request, allowed=True, risk_score=0.0,
                                reason="neutral_message", similarity=1.0)
        try:
            anchor_embs = await self._get_anchor_embeddings()
            query_emb = await self.embedding_provider.embed(request.prompt)
            max_sim = max(_cosine_similarity(query_emb, anchor) for anchor in anchor_embs)
        except Exception:
            # Fail open: if the embedding API is unavailable, do not block
            return self._result(request, allowed=True, risk_score=0.2,
                                reason="embedding_unavailable", similarity=0.0)

        if max_sim >= self.threshold:
            return self._result(request, allowed=True, risk_score=0.0,
                                reason="topic_relevant", similarity=max_sim)

        risk_score = min(0.9, max(0.5, 1.0 - max_sim * 2.5))
        return self._result(request, allowed=False, risk_score=risk_score,
                            reason="off_topic_detected", similarity=max_sim)

    def _result(
        self,
        request: ChatRequest,
        *,
        allowed: bool,
        risk_score: float,
        reason: str,
        similarity: float,
    ) -> ModerationResult:
        return ModerationResult(
            request_id=request.request_id,
            timestamp=request.timestamp,
            source_module="guardrails.topic_guard.embedding",
            allowed=allowed,
            risk_score=risk_score,
            reasons=[reason],
            details={"max_similarity": round(similarity, 4), "threshold": self.threshold},
        )
