from __future__ import annotations

import bcrypt
import jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from typing import Any

from core.config import AppConfig
from core.types import AuditEvent, EventType, UserSession


class AuthManager:
    def __init__(self, config: AppConfig, logger: Any | None = None) -> None:
        self.config = config
        self.logger = logger
        self.token_scheme = HTTPBearer(auto_error=False)
        self.secret = config.api_key
        self.algorithm = "HS256"

    def hash_password(self, password: str) -> bytes:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    def verify_password(self, password: str, hashed: bytes) -> bool:
        return bcrypt.checkpw(password.encode("utf-8"), hashed)

    def create_access_token(self, user_id: str, roles: list[str], expires_delta: timedelta | None = None) -> str:
        expires_delta = expires_delta or timedelta(minutes=30)
        payload = {
            "sub": user_id,
            "roles": roles,
            "exp": datetime.utcnow() + expires_delta,
            "iat": datetime.utcnow(),
        }
        return jwt.encode(payload, self.secret, algorithm=self.algorithm)

    def decode_token(self, token: str) -> dict[str, Any]:
        try:
            return jwt.decode(token, self.secret, algorithms=[self.algorithm])
        except jwt.ExpiredSignatureError as exc:
            raise HTTPException(status_code=401, detail="Token expired") from exc
        except jwt.PyJWTError as exc:
            raise HTTPException(status_code=401, detail="Invalid token") from exc

    def login(self, user_id: str, password: str, stored_hash: bytes, roles: list[str]) -> dict[str, str]:
        if not self.verify_password(password, stored_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        access_token = self.create_access_token(user_id, roles)
        return {"access_token": access_token, "token_type": "bearer"}

    def refresh_token(self, token: str) -> dict[str, str]:
        decoded = self.decode_token(token)
        return {"access_token": self.create_access_token(decoded["sub"], decoded.get("roles", [])), "token_type": "bearer"}

    def session_from_token(self, token: str) -> UserSession:
        decoded = self.decode_token(token)
        return UserSession(
            request_id="auth",
            timestamp=datetime.utcnow(),
            source_module="security.auth",
            session_id=str(decoded.get("session_id", decoded["sub"])),
            user_id=decoded["sub"],
            roles=list(decoded.get("roles", [])),
            is_active=True,
            expires_at=datetime.fromtimestamp(decoded["exp"]),
        )

    async def get_current_session(self, request: Request) -> UserSession | None:
        credentials: HTTPAuthorizationCredentials | None = await self.token_scheme(request)
        if credentials is None or credentials.scheme.lower() != "bearer":
            return None
        return self.session_from_token(credentials.credentials)

    async def auth_middleware(self, request: Request, call_next: Any) -> Any:
        session = await self.get_current_session(request)
        request.state.session = session
        if session and self.logger:
            self.logger.info("Authenticated session %s", session.session_id)
        return await call_next(request)

    def audit_event(self, request_id: str, payload: dict[str, Any]) -> AuditEvent:
        return AuditEvent(
            request_id=request_id,
            timestamp=datetime.utcnow(),
            source_module="security.auth",
            event_type=EventType.AUTHENTICATION,
            trace_id=request_id,
            payload=payload,
        )
