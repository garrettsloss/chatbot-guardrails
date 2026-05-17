from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from aioredis import Redis, from_url
from core.config import AppConfig


class RateLimiter:
    def __init__(self, config: AppConfig, redis_client: Redis | None = None, logger: Any | None = None) -> None:
        self.config = config
        self.redis = redis_client
        self.logger = logger
        self._local_counters: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    async def connect_redis(self) -> None:
        if self.redis is None and self.config.redis_url:
            self.redis = await from_url(self.config.redis_url)

    async def check_limit(self, key: str, limit: int, window_seconds: int) -> bool:
        now = time.time()
        if self.redis is not None:
            bucket = int(now // window_seconds)
            redis_key = f"rate:{key}:{bucket}"
            value = await self.redis.incr(redis_key)
            if value == 1:
                await self.redis.expire(redis_key, window_seconds)
            allowed = value <= limit
            if self.logger:
                self.logger.debug("Redis rate limit %s=%s/%s", key, value, limit)
            return allowed

        async with self._lock:
            window = self._local_counters.setdefault(key, [])
            window[:] = [ts for ts in window if ts > now - window_seconds]
            window.append(now)
            allowed = len(window) <= limit
            if self.logger:
                self.logger.debug("Local rate limit %s=%s/%s", key, len(window), limit)
            return allowed

    async def check_user(self, session_id: str) -> bool:
        user_limit = self.config.rate_limits.get("user_per_minute", 60)
        return await self.check_limit(f"user:{session_id}", user_limit, 60)

    async def check_ip(self, client_ip: str) -> bool:
        ip_limit = self.config.rate_limits.get("ip_per_minute", 120)
        return await self.check_limit(f"ip:{client_ip}", ip_limit, 60)
