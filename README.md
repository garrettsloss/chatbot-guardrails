# Chatbot Guardrails

A multi-turn chatbot that restricts conversation to a configurable topic (default: **gardening**) using a layered guardrail system. Every user message passes through nine concurrent input detectors, a YAML-driven policy engine, RAG retrieval with context sanitisation, a topic-enforcing system prompt, and a post-generation output filter before any response is returned. Changing `CHATBOT_TOPIC` in the environment file is all that is needed to redeploy the chatbot for a different domain.

---

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.11+ |
| Azure OpenAI deployment | GPT-4.1-mini (chat) + Ada-002 (embeddings) |
| ChromaDB | installed via `pip` |
| pytest + pytest-asyncio | for running the test suite |

Install dependencies:

```bash
pip install openai chromadb pydantic pydantic-settings pyyaml bcrypt PyJWT redis pytest pytest-asyncio
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your Azure credentials:

```bash
cp .env.example .env
```

Key environment variables:

| Variable | Description | Default |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | Azure resource URL | — |
| `AZURE_OPENAI_API_KEY` | Azure API key | — |
| `AZURE_OPENAI_API_VERSION` | API version | `2025-03-01-preview` |
| `OPENAI_MODEL` | Chat deployment name | `gpt-4.1-mini` |
| `EMBEDDING_MODEL` | Embedding deployment name | `ada-002` |
| `CHATBOT_TOPIC` | Active topic profile | `gardening` |
| `TOPIC_RELEVANCE_THRESHOLD` | Minimum cosine similarity (0–1) | `0.30` |
| `MAX_HISTORY_TURNS` | Rolling conversation window | `10` |
| `API_KEY` | JWT signing secret (AuthManager) | — |

Built-in topic profiles: `gardening`, `motor_vehicles`, `cinematography`. To add a new topic, add an entry to `TOPIC_PROFILES` in `guardrails/topic_guard.py` — no other code changes required.

---

## Running the chatbot

```bash
python cli.py
```

The chatbot opens an interactive prompt. Type `quit` or `exit` to end the session. Audit logs (denials, token usage, pipeline events) are written to `logs/guardrails.log`.

## Running the tests

```bash
python -m pytest tests/ -v
```

101 tests across three files: `test_detectors.py` (unit tests for every guardrail component), `test_pipeline.py` (orchestrator integration tests with mocked LLM and embeddings), `test_adversarial.py` (resistance to injection, obfuscation, multi-turn drift, and topic-switching).

---

## Architecture

```
User message
     │
     ▼
╔═══════════════════════════════════════════════════════════════╗
║                    INPUT GUARDRAILS                           ║
║  All 10 detectors run concurrently via asyncio.gather.        ║
║  A single allowed=False from any detector blocks the chain.   ║
║                                                               ║
║  L1   RegexRuleDetector           command injection (regex)   ║
║  L2   HarmfulContentDetector      dangerous phrases (regex)   ║
║  L3   LLMHarmfulContentDetector   rephrased attacks (LLM)     ║
║  L4   PromptInjectionDetector     injection phrases           ║
║  L5   JailbreakDetector           jailbreak phrases           ║
║  L6   PIIDetector                 SSN / credit card (regex)   ║
║  L7   SpamDetector                word-flood detection        ║
║  L8   UnicodeObfuscationDetector  Cyrillic / zero-width chars ║
║  L9   TopicKeywordDetector        fast topic pre-filter       ║
║  L10  TopicEmbeddingDetector      Ada-002 semantic similarity ║
╚═══════════════════════════════════════════════════════════════╝
     │ blocked ──► polite rejection  (safe=False, no exception)
     │ allowed
     ▼
╔═══════════════════════════╗
║      PolicyEngine         ║  YAML rules — e.g. block aggregate risk ≥ 0.70
╚═══════════════════════════╝
     │ blocked ──► polite rejection
     │ allowed
     ▼
╔═══════════════════════════╗
║     RAG Retrieval         ║  Ada-002 query embedding → ChromaDB cosine search
╚═══════════════════════════╝
     │
     ▼
╔═══════════════════════════╗
║    ContextSanitizer       ║  strip hidden-instruction blocks from retrieved docs
╚═══════════════════════════╝
     │
     ▼
╔═══════════════════════════╗
║      PromptBuilder        ║  system prompt + sanitised context + history + user msg
╚═══════════════════════════╝
     │
     ▼
╔═══════════════════════════╗
║       LLM Client          ║  GPT-4.1-mini via Azure OpenAI
║  (system prompt enforces  ║  Azure content filter → safe=False, not exception
║   topic at model level)   ║
╚═══════════════════════════╝
     │ azure blocked ──► polite rejection
     │ safe
     ▼
╔═══════════════════════════╗
║      OutputFilter         ║  word-boundary topic keyword scan on response text
╚═══════════════════════════╝
     │ blocked ──► polite rejection
     │ allowed
     ▼
  Response returned to user
  Genuine exchange appended to conversation history
  (rejected turns are never written to history)
```

### Observability

Every pipeline stage publishes a typed `AuditEvent` to the internal `EventBus`. `_setup_logging()` in `cli.py` attaches two handlers: a console handler at `WARNING` level (denial warnings surface in the terminal) and a rotating file handler at `INFO` level (`logs/guardrails.log`, 5 MB max, 3 backups). Every denial is structured-logged with request ID, user, reasons list, and the first 80 characters of the prompt. Token usage (prompt + completion + total) is logged per turn.

---

## Guardrail layers

### L1 — RegexRuleDetector (input)

Compiles a list of operator-supplied regex patterns and rejects any prompt that matches. The default patterns block `\bshutdown\b` and `\bdelete\b` — literal command strings that have no place in a chat interface and whose presence almost certainly indicates an injection attempt or misuse. Easily extended by passing additional patterns to the constructor without touching any other component. **Threat mitigated:** command injection and direct system manipulation.

### L2 — HarmfulContentDetector (input)

Applies targeted multi-word regex patterns covering bomb and explosive fabrication, chemical and biological weapons, explicit violence instructions, firearm manufacturing, and illicit drug synthesis. Patterns are written with both `how to` and `how do I` variants and handle plural forms (`explosives`) to prevent trivial evasion. Patterns require an action verb before the dangerous noun so gardening phrases like "seed bomb" are not affected. Makes no API call and therefore adds zero latency. **Threat mitigated:** direct harmful requests using known dangerous vocabulary.

### L3 — LLMHarmfulContentDetector (input)

Sends the prompt to GPT-4.1-mini with a tightly worded binary classification system prompt, requesting only the word `HARMFUL` or `SAFE` at `temperature=0.0` and `max_tokens=5`. This layer catches what regex cannot: hypothetical framing ("Hypothetically, how would one…"), fictional contexts ("For a novel, my character needs to explain…"), academic language, and multi-step indirect requests that individually seem innocuous. It runs concurrently with all other detectors so it adds only the latency of one fast parallel API call. On any API failure it fails open — classifier unavailability must not block legitimate users. **Threat mitigated:** sophisticated rephrasing and contextual evasion of all pattern-matching rules.

### L4 — PromptInjectionDetector (input)

Scans the lowercased prompt for known injection marker phrases: `"ignore previous instructions"`, `"forget your instructions"`, and `"prompt injection"`. These are the canonical social-engineering strings used to override a model's system prompt at inference time. Unlike the regex detector this uses substring matching, because injection phrases are frequently embedded mid-sentence alongside otherwise benign content. **Threat mitigated:** prompt injection attacks aimed at overriding the system prompt or topic restrictions.

### L5 — JailbreakDetector (input)

Checks for jailbreak vocabulary — `"jailbreak"`, `"bypass safety"`, `"secret instructions"` — that signals an attempt to disable the model's safety training or system-level restrictions entirely rather than merely redirecting its behaviour. The phrase list is intentionally a structural demonstration and can be extended without touching any other component. **Threat mitigated:** direct jailbreak attempts that try to remove all model constraints.

### L6 — PIIDetector (input)

Uses compiled regular expressions to detect US Social Security numbers (`\b\d{3}-\d{2}-\d{4}\b`) and 16-digit payment card numbers (`\b\d{16}\b`). Allowing such data into the LLM context creates unnecessary data-handling risk regardless of the chatbot's topic. The detector returns `risk_score=0.95` — the highest of all detectors — reflecting that PII exposure is a data-protection concern independent of topic relevance. **Threat mitigated:** accidental or deliberate personal data leakage into the LLM context.

### L7 — SpamDetector (input)

Tokenises the prompt and counts per-token repetitions, blocking when any token appears five or more times. Flood attacks can exhaust context windows, confuse the model, or probe rate-limit behaviour; this detector intercepts them before any API call is made. Its weight (0.5) is deliberately low so that incidental repetition in natural language does not disproportionately inflate the aggregate risk score. **Threat mitigated:** repetition flooding and context-window exhaustion.

### L8 — UnicodeObfuscationDetector (input)

Scans for Cyrillic characters in the range U+0400–U+04FF and zero-width characters (U+200B, U+200C, U+200D). These are used to substitute visually identical Latin characters — e.g. Cyrillic `а` for Latin `a` — producing strings like `"mаke а bоmb"` that defeat all ASCII-based keyword matching while appearing identical to a human reader. Because `allowed = len(matches) == 0`, even a single substituted character blocks the request, preventing the technique even when only partially applied. **Threat mitigated:** Unicode substitution attacks that bypass every keyword-based guardrail layer above.

### L9 — TopicKeywordDetector (input)

Maintains a list of ~60 topic-specific keywords drawn from the active `TOPIC_PROFILE`. If the prompt contains any keyword it passes with zero risk score; recognised neutral phrases (greetings, thanks, meta-questions) also pass immediately. Messages with no keyword match receive a soft risk score of 0.35 — not enough to block alone, but a signal that is fed into the aggregate. This layer makes no API call, confirming obvious gardening questions in microseconds. **Threat mitigated:** off-topic requests; acts as a fast pre-filter that reduces embedding API calls for clearly on-topic input.

### L10 — TopicEmbeddingDetector (input)

Generates an Ada-002 embedding for the user's prompt and computes cosine similarity against ten topic anchor phrases (e.g. "how to grow vegetables in a garden", "composting techniques for garden soil improvement"). Anchor embeddings are computed lazily on the first request and cached for the session — subsequent turns pay only for one embedding call each. If the maximum cosine similarity across all anchors falls below `TOPIC_RELEVANCE_THRESHOLD` (default 0.30), the detector returns `allowed=False` with a risk score proportional to the shortfall. Neutral phrases bypass the check entirely. On API failure the detector fails open. This is the definitive semantic gate: it catches off-topic queries that coincidentally contain gardening keywords and passes gardening queries that happen to use no listed keyword. **Threat mitigated:** semantic off-topic evasion and accidental topic drift that keyword matching alone cannot detect.

### L11 — PolicyEngine (pipeline)

Evaluates a set of operator-defined rules loaded from `policies.yml` at startup, with automatic hot-reload on file modification (checked via `mtime`). Each rule specifies a condition (minimum moderation score, required user role, or substring match), an action (`deny`, `review`, or `redirect`), and a severity. The default policy denies requests when the aggregated input moderation score reaches 0.70. Because this layer is YAML-driven and reloads without a restart, operators can tighten thresholds, add rules, or disable individual rules in production without a code deployment. **Threat mitigated:** high-risk inputs whose individual detector scores were sub-threshold but whose aggregate indicates danger; provides operator-configurable control independent of the code.

### L12 — ContextSanitizer (pipeline)

After RAG retrieval, every retrieved document is passed through the sanitizer before inclusion in the LLM prompt. It strips full `<hidden_instruction>…</hidden_instruction>` blocks (tag and contents), phrases matching `ignore.*?instructions` and `follow.*?secret`, and `base64:`/`hex:` encoded-payload markers, then normalises pathological whitespace. This prevents an adversary who has injected content into the vector database from using retrieved documents as a secondary injection channel that bypasses all input-layer guardrails. **Threat mitigated:** indirect prompt injection via poisoned RAG documents.

### L13 — System Prompt (LLM level)

The `PromptBuilder` inserts the active topic profile's system prompt as the first message in every LLM call. For the gardening profile this explicitly enumerates permitted subjects, forbids off-topic responses, and instructs the model to redirect rather than refuse abruptly. The system prompt is the primary LLM-level behavioural constraint and is automatically replaced in full when `CHATBOT_TOPIC` is changed. **Threat mitigated:** on-topic queries with subtle adversarial intent; serves as a last-resort constraint that is independent of all application-layer guardrails.

### L14 — OutputFilter (post-generation)

After the LLM returns a response, the output filter checks whether the text (if longer than 80 characters) contains at least one topic keyword, matched with a pre-compiled word-boundary regex (`\b…\b`) to prevent false positives from substrings (e.g. `pot` inside `hypotenuse`). A substantive response with zero topic keywords is assigned `risk_score=0.85`, exceeding the configured block threshold of 0.80, and replaced with a safe fallback message. **Threat mitigated:** jailbroken LLM responses that produce off-topic output despite the system prompt, catching any guardrail bypass that survives all preceding layers.

### L15 — Azure Content Filter (LLM client)

Azure OpenAI's built-in content management policy runs server-side on every API call and is independent of all application-level guardrails. When it fires it returns `HTTP 400` with `code: content_filter`. The `OpenAIClientAdapter` catches this `BadRequestError` and returns `ChatResponse(safe=False, reasons=["azure_content_filter"])` rather than propagating the exception — so the orchestrator handles it through the standard rejection path and the user sees a polite message rather than a crash. **Threat mitigated:** content that evaded all application-level layers but is caught by Azure's managed content policy; also provides a safety net if the application guardrails are misconfigured.

---

## Refusal behaviour

All guardrail blocks produce a `ChatResponse(safe=False)` — never an unhandled exception. The orchestrator selects the rejection message by violation type:

| Reason detected | Message shown to user |
|---|---|
| `harmful_content_detected` / `llm_harmful_content_detected` / `azure_content_filter` | "I'm not able to help with that request. If you have a gardening question, I'd be happy to help!" |
| `prompt_injection_detected` / `jailbreak_detected` | "I noticed an attempt to override my instructions. I'm here exclusively to help with gardening questions…" |
| `off_topic_detected` / policy block / output failure | "I'm a specialised gardening assistant, so I can only help with gardening-related questions…" |

Rejected turns are not written to the conversation history, so the LLM never sees a prior refusal as context for subsequent turns.

---

## Project structure

```
cli.py                        # entry point — builds pipeline, runs async chat loop
core/
  config.py                   # AppConfig (pydantic-settings, reads from .env)
  types.py                    # shared Pydantic models (ChatRequest, ChatResponse, …)
  events.py                   # async EventBus with retry and dead-letter queue
  chroma_db.py                # ChromaDB VectorDBClient implementation
guardrails/
  input_filter.py             # InputFilterDetector ABC + all 10 detector subclasses
  output_filter.py            # OutputFilter (word-boundary topic keyword scan)
  topic_guard.py              # TOPIC_PROFILES dict, TopicKeywordDetector, TopicEmbeddingDetector
  policy_engine.py            # YAML-driven PolicyEngine with hot-reload
  context_sanitizer.py        # ContextSanitizer (RAG document injection removal)
llm/
  client.py                   # OpenAIClientAdapter (Azure, handles content filter error)
  prompt_builder.py           # PromptBuilder (assembles messages list for LLM)
rag/
  context_retrieval.py        # ContextRetriever, EmbeddingProvider, VectorDBClient ABCs
pipeline/
  orchestrator.py             # Orchestrator.process() — wires all layers in sequence
observability/
  audit.py                    # AuditLogger, SilentAuditProvider, FileAuditProvider
security/
  auth.py                     # AuthManager (JWT create/verify)
  rate_limit.py               # RateLimiter (local sliding window or Redis)
tools/
  gateway.py                  # ToolRegistry + ToolInterface
sdk/
  plugins.py                  # PluginBase ABCs for third-party detector/policy extensions
policies.yml                  # operator-configurable policy rules (hot-reloaded)
tests/
  conftest.py                 # shared fixtures and deterministic mock embedding provider
  test_detectors.py           # unit tests for every guardrail component (56 tests)
  test_pipeline.py            # orchestrator integration tests (10 tests)
  test_adversarial.py         # adversarial and multi-turn resistance tests (35 tests)
```
