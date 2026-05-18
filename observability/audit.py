from __future__ import annotations

import json
import logging
from abc import ABC
from logging.handlers import RotatingFileHandler
from typing import Any

from core.types import AuditEvent


class AuditProvider(ABC):
    def emit(self, event: AuditEvent) -> None:
        raise NotImplementedError()


class ConsoleAuditProvider(AuditProvider):
    def emit(self, event: AuditEvent) -> None:
        print(json.dumps(self._serialize(event), default=str))

    @staticmethod
    def _serialize(event: AuditEvent) -> dict[str, Any]:
        return event.model_dump()


class FileAuditProvider(AuditProvider):
    def __init__(self, path: str, max_bytes: int = 10_000_000, backup_count: int = 5) -> None:
        self.path = path
        self.logger = logging.getLogger("observability.audit")
        handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def emit(self, event: AuditEvent) -> None:
        self.logger.info(json.dumps(self._serialize(event), default=str))

    @staticmethod
    def _serialize(event: AuditEvent) -> dict[str, Any]:
        return event.model_dump()


class SilentAuditProvider(AuditProvider):
    """Logs audit events via Python's logging system at DEBUG level (no console noise)."""

    def emit(self, event: AuditEvent) -> None:
        logging.getLogger("observability.audit").debug(
            json.dumps(self._serialize(event), default=str)
        )

    @staticmethod
    def _serialize(event: AuditEvent) -> dict[str, Any]:
        return event.model_dump()


class AuditLogger:
    def __init__(self, providers: list[AuditProvider]) -> None:
        self.providers = providers

    def log(self, event: AuditEvent) -> None:
        redacted = self._redact_sensitive(event)
        for provider in self.providers:
            provider.emit(redacted)

    @staticmethod
    def _redact_sensitive(event: AuditEvent) -> AuditEvent:
        payload = event.payload.copy()
        for key in list(payload.keys()):
            if "password" in key.lower() or "secret" in key.lower() or "token" in key.lower():
                payload[key] = "[REDACTED]"
        return AuditEvent(**{**event.model_dump(), "payload": payload})
