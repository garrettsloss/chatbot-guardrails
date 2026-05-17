# chatbot-guardrails
A chatbot that implements multilayer guardrails using an API for the chatbot.

## Architecture

This repository now contains a strict modular architecture with the following packages:

- `core` - shared schemas, configuration, and internal event bus
- `guardrails` - input filtering, policy evaluation, and context sanitization
- `security` - authentication, RBAC, and rate limiting
- `rag` - vector context retrieval and embeddings abstraction
- `llm` - prompt building and provider client abstraction
- `tools` - safe tool execution gateway
- `observability` - structured audit logging
- `pipeline` - orchestration pipeline coordinating the request lifecycle
- `sdk` - plugin SDK for third-party extension points

## Installation

```bash
python -m pip install -e .
```

## Notes

- All shared data schemas are defined in `core/types.py`.
- Configuration is centralized in `core/config.py` and loaded through dependency injection.
- The orchestration pipeline is implemented in `pipeline/orchestrator.py`.

## CLI usage

```bash
python cli.py --prompt "Hello guardrails" --user-id alice --session-id session-1
```

## Environment variables

Required environment variables:

- `API_KEY` — shared application secret used for signing JWTs and internal auth.
- `VECTOR_DB_URL` — vector database endpoint used by the retrieval layer.

Optional but recommended:

- `OPENAI_MODEL` — model name for LLM calls (default `gpt-4.1`).
- `EMBEDDING_MODEL` — embedding model name (default `text-embedding-3-large`).
- `LOG_LEVEL` — logging level (default `INFO`).
- `AUDIT_LOG_FILE` — path for audit logs (default `logs/audit.log`).
- `REDIS_URL` — Redis endpoint for distributed rate limiting.
- `ENVIRONMENT` — runtime environment name (default `development`).

