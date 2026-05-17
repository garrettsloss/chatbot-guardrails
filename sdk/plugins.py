from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from core.types import ChatRequest, ModerationResult, PolicyDecision, ToolCall, ToolResult


@dataclass
class PluginMetadata:
    name: str
    version: str
    author: str
    compatible_versions: list[str]
    description: str | None = None


class PluginBase(ABC):
    metadata: PluginMetadata

    @abstractmethod
    async def initialize(self, config: Any) -> None:
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        pass


class DetectorPlugin(PluginBase):
    @abstractmethod
    async def analyze_request(self, request: ChatRequest) -> ModerationResult:
        pass


class PolicyPlugin(PluginBase):
    @abstractmethod
    async def evaluate(self, request: ChatRequest, moderation: ModerationResult) -> PolicyDecision:
        pass


class ToolPlugin(PluginBase):
    @abstractmethod
    async def to_tool_call(self, request: ChatRequest) -> ToolCall:
        pass

    @abstractmethod
    async def execute(self, call: ToolCall) -> ToolResult:
        pass


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, PluginBase] = {}

    def register(self, plugin: PluginBase) -> None:
        self._plugins[plugin.metadata.name] = plugin

    def get(self, name: str) -> PluginBase:
        return self._plugins[name]

    def list(self) -> list[str]:
        return list(self._plugins.keys())

    async def initialize_all(self, config: Any) -> None:
        for plugin in self._plugins.values():
            await plugin.initialize(config)

    async def shutdown_all(self) -> None:
        for plugin in self._plugins.values():
            await plugin.shutdown()
