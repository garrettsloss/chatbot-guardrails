from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from core.types import AuditEvent, EventType

EventHandler = Callable[[AuditEvent], Awaitable[None]]
FilterFn = Callable[[AuditEvent], bool]


class EventSubscription:
    def __init__(self, handler: EventHandler, filter_fn: FilterFn | None = None) -> None:
        self.handler = handler
        self.filter_fn = filter_fn or (lambda _: True)


class EventBus:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._subscriptions: dict[EventType, list[EventSubscription]] = {}
        self._lock = asyncio.Lock()
        self._dead_letter_queue: list[AuditEvent] = []
        self._logger = logger or logging.getLogger("core.events")

    async def subscribe(self, event_type: EventType, handler: EventHandler, filter_fn: FilterFn | None = None) -> None:
        async with self._lock:
            self._subscriptions.setdefault(event_type, []).append(EventSubscription(handler, filter_fn))
            self._logger.debug("Subscribed handler to event type %s", event_type)

    async def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        async with self._lock:
            handlers = self._subscriptions.get(event_type, [])
            self._subscriptions[event_type] = [sub for sub in handlers if sub.handler != handler]
            self._logger.debug("Unsubscribed handler from event type %s", event_type)

    async def publish(self, event: AuditEvent, max_retries: int = 2) -> None:
        async with self._lock:
            subscribers = list(self._subscriptions.get(event.event_type, []))
        if not subscribers:
            self._logger.debug("No subscribers for event type %s", event.event_type)
            return
        for subscriber in subscribers:
            if not subscriber.filter_fn(event):
                continue
            retry = 0
            while retry <= max_retries:
                try:
                    await subscriber.handler(event)
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    retry += 1
                    self._logger.warning("Event handler failed for %s attempt %d: %s", event.event_type, retry, exc)
                    if retry > max_retries:
                        self._dead_letter_queue.append(event)
                        self._logger.error("Moved event to dead-letter queue: %s", event.event_type)

    async def publish_fire_and_forget(self, event: AuditEvent) -> None:
        asyncio.create_task(self.publish(event))

    def get_dead_letter_queue(self) -> list[AuditEvent]:
        return list(self._dead_letter_queue)
