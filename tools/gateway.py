from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from core.types import AuditEvent, EventType, ToolCall, ToolResult, ToolPermission


class ToolInterface(ABC):
    @abstractmethod
    async def execute(self, call: ToolCall) -> ToolResult:
        pass

    @abstractmethod
    def schema(self) -> dict[str, Any]:
        pass


class ToolRegistry:
    def __init__(self, logger: Any | None = None) -> None:
        self._tools: dict[str, ToolInterface] = {}
        self.logger = logger

    def register(self, name: str, tool: ToolInterface) -> None:
        self._tools[name] = tool
        if self.logger:
            self.logger.info("Registered tool %s", name)

    def get(self, name: str) -> ToolInterface:
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        return self._tools[name]

    async def execute(self, call: ToolCall, timeout_seconds: int = 10) -> ToolResult:
        tool = self.get(call.tool_name)
        if call.permission != ToolPermission.ALLOW:
            return ToolResult(
                request_id=call.request_id,
                timestamp=call.timestamp,
                source_module="tools.gateway",
                tool_name=call.tool_name,
                success=False,
                output="",
                error="Permission denied",
                metadata={"permission": call.permission.value},
            )
        try:
            result = await asyncio.wait_for(tool.execute(call), timeout=timeout_seconds)
            return result
        except asyncio.TimeoutError as exc:
            return ToolResult(
                request_id=call.request_id,
                timestamp=call.timestamp,
                source_module="tools.gateway",
                tool_name=call.tool_name,
                success=False,
                output="",
                error=str(exc),
                metadata={"timeout_seconds": timeout_seconds},
            )
        except Exception as exc:
            return ToolResult(
                request_id=call.request_id,
                timestamp=call.timestamp,
                source_module="tools.gateway",
                tool_name=call.tool_name,
                success=False,
                output="",
                error=str(exc),
                metadata={"exception_type": type(exc).__name__},
            )

    def audit_event(self, call: ToolCall, result: ToolResult) -> AuditEvent:
        return AuditEvent(
            request_id=call.request_id,
            timestamp=call.timestamp,
            source_module="tools.gateway",
            event_type=EventType.TOOL_EXECUTION,
            trace_id=call.request_id,
            payload={"tool_name": call.tool_name, "success": result.success, "error": result.error},
        )
