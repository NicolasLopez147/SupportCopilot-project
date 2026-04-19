# SupportCopilot Microservices Architecture Specification

## Purpose

This document defines the target microservices architecture for the next product-oriented phase of the project.

The goal is not to optimize for minimal complexity, but to deliberately implement a distributed architecture that demonstrates:

- service decomposition
- API design
- orchestration
- service-to-service communication
- containerization readiness
- CI/CD readiness
- observability and maintainability

This architecture is intended as a technical showcase aligned with backend, cloud-native, API integration, and software engineering expectations.

## Architectural Choice

The selected target architecture is a **synchronous microservices architecture** composed of four services:

1. `gateway-service`
2. `intent-service`
3. `summary-service`
4. `reply-service`

This is a deliberate design choice to demonstrate service-oriented decomposition.

Even though a modular monolith would be simpler for the current scope, the project will adopt microservices in order to showcase the ability to design and implement a more advanced distributed backend.

## Core Principles

- Each service owns a single clear responsibility.
- The `gateway-service` is the only orchestrator.
- Inter-service communication is synchronous over HTTP REST.
- Each model-based service loads its model only once at startup.
- Contracts must be explicit and versioned.
- Configuration must be environment-based and must avoid hardcoded service URLs, ports, and artifact paths.
- Logs must be structured and traceable across services.
- Failure handling must be explicit, predictable, and observable.
- Shared schemas should be centralized to reduce duplication and contract drift.

## Target Service Map

### 1. Gateway Service

Responsibilities:

- expose the public API
- validate incoming requests
- generate and propagate request IDs
- orchestrate the pipeline end-to-end
- call `intent-service`
- call `summary-service`
- call `reply-service`
- aggregate final output
- normalize service errors
- expose a unified response to the client

The gateway must not implement model inference itself.

### 2. Intent Service

Responsibilities:

- receive a conversation or customer text
- run intent classification
- return intent prediction
- return confidence and top classes
- expose a health endpoint

This service owns the intent classifier lifecycle.

### 3. Summary Service

Responsibilities:

- receive a conversation
- generate a summary
- run the summary critic
- apply summary fallback if needed
- optionally persist summary critic failures
- expose a health endpoint

This service owns the summary generation pipeline and summary quality control.

### 4. Reply Service

Responsibilities:

- receive a conversation, predicted intent, and summary
- generate the suggested reply
- run the reply critic
- apply reply fallback if needed
- optionally persist reply critic failures
- expose a health endpoint

This service owns reply generation and reply quality control.

## Orchestration Model

The orchestration model is strictly centralized.

### Rule

`gateway-service` is the only orchestrator.

### Implications

- `intent-service` does not call `summary-service`
- `summary-service` does not call `reply-service`
- `reply-service` does not call `intent-service`
- downstream services are isolated and stateless from an orchestration perspective

### Pipeline Flow

1. Client sends conversation to `gateway-service`
2. `gateway-service` calls `intent-service`
3. `gateway-service` calls `summary-service`
4. `gateway-service` calls `reply-service` with:
   - conversation
   - final intent
   - final summary
5. `gateway-service` aggregates and returns:
   - intent output
   - summary output
   - reply output
   - optional service metadata

## Communication Style

### Chosen Mode

The first implementation will be **synchronous** and **HTTP/REST-based**.

### Why

- easier to debug
- easier to demonstrate
- simpler than queues/events for a first distributed version
- enough for the current pipeline structure

### Consequences

- request latency is cumulative
- timeouts must be handled carefully
- gateway must tolerate service failures gracefully

Asynchronous patterns may be considered later for:

- batch processing
- offline retraining
- long-running evaluation jobs

## API Versioning

All services should expose versioned endpoints:

- `/api/v1/health`
- `/api/v1/intent`
- `/api/v1/summary`
- `/api/v1/reply`

This avoids breaking future clients when contracts evolve.

## Service Contracts

The following contracts should be treated as the initial API baseline.

### Gateway Service

#### `POST /api/v1/copilot/run`

Request:

```json
{
  "conversation_id": "demo_001",
  "scenario": "billing_vs_technical_routing",
  "messages": [
    {
      "speaker": "customer",
      "text": "My internet keeps freezing and I do not know if this is a billing or technical issue."
    },
    {
      "speaker": "agent",
      "text": "I can help you figure that out. Can you tell me more about the issue?"
    }
  ],
  "persist_feedback": false
}
```

Response:

```json
{
  "conversation_id": "demo_001",
  "scenario": "billing_vs_technical_routing",
  "conversation_text": "Customer: ...",
  "intent": {},
  "summary": "...",
  "summary_review": {},
  "suggested_reply": "...",
  "reply_review": {},
  "request_id": "req-123"
}
```

### Intent Service

#### `POST /api/v1/intent`

Request:

```json
{
  "conversation_id": "demo_001",
  "messages": [
    {
      "speaker": "customer",
      "text": "My internet keeps freezing."
    }
  ]
}
```

Response:

```json
{
  "conversation_id": "demo_001",
  "input_text": "My internet keeps freezing.",
  "predicted_intent": "slow_connection_diagnosis",
  "confidence": 0.87,
  "top_classes": [
    {
      "label": "slow_connection_diagnosis",
      "score": 0.87
    }
  ]
}
```

### Summary Service

#### `POST /api/v1/summary`

Request:

```json
{
  "conversation_id": "demo_001",
  "scenario": "billing_vs_technical_routing",
  "messages": [
    {
      "speaker": "customer",
      "text": "My internet keeps freezing."
    },
    {
      "speaker": "agent",
      "text": "Can you tell me more about the issue?"
    }
  ],
  "persist_feedback": false
}
```

Response:

```json
{
  "conversation_id": "demo_001",
  "conversation_text": "Customer: ...",
  "summary_raw": "...",
  "summary_review": {
    "passed": true,
    "score": 0.91,
    "issues": [],
    "final_summary": "..."
  },
  "summary": "..."
}
```

### Reply Service

#### `POST /api/v1/reply`

Request:

```json
{
  "conversation_id": "demo_001",
  "scenario": "billing_vs_technical_routing",
  "messages": [
    {
      "speaker": "customer",
      "text": "My internet keeps freezing."
    },
    {
      "speaker": "agent",
      "text": "Can you tell me more about the issue?"
    }
  ],
  "predicted_intent": "billing_vs_technical_routing",
  "summary_text": "Customer reports freezing internet and uncertainty about routing.",
  "persist_feedback": false
}
```

Response:

```json
{
  "conversation_id": "demo_001",
  "conversation_text": "Customer: ...",
  "suggested_reply_raw": "...",
  "reply_review": {
    "passed": false,
    "score": 0.41,
    "issues": [
      "missing_next_step"
    ],
    "used_fallback": true,
    "final_reply": "..."
  },
  "suggested_reply": "..."
}
```

## Model Loading Strategy

Each model-based service must load its model only once at startup.

### Mandatory Rule

The service must not reload:

- the classifier
- the embedding encoder
- the summary model
- the reply model

on every request.

### Recommended Implementation

- initialize models at service startup
- keep them in memory
- reuse them for all incoming requests

### Reason

Without this rule, the architecture would become too slow and would falsely suggest that the distributed design itself is the problem, while the real problem would be repeated model initialization.

## Error Handling

Each service must expose explicit and predictable error behavior.

### Common Error Categories

- `400 Bad Request`
  - malformed request
  - missing required fields
  - invalid message format

- `422 Unprocessable Entity`
  - schema validation failure

- `503 Service Unavailable`
  - model artifacts missing
  - service not ready
  - downstream dependency unavailable

- `504 Gateway Timeout`
  - downstream service timeout

- `500 Internal Server Error`
  - unexpected runtime error

### Service-Specific Error Cases

#### Gateway Service

- downstream service unavailable
- downstream timeout
- malformed downstream response
- partial orchestration failure

#### Intent Service

- no customer text extractable
- classifier artifact missing
- embedding model not available locally
- prediction runtime failure

#### Summary Service

- summary model artifact missing
- invalid conversation input
- generation failure
- critic execution failure
- fallback generation failure

#### Reply Service

- reply model artifact missing
- invalid conversation input
- missing intent or summary in request
- generation failure
- critic execution failure
- fallback generation failure

### Error Response Format

All services should aim to return a normalized error body:

```json
{
  "error": {
    "code": "SERVICE_UNAVAILABLE",
    "message": "Summary model artifact not found.",
    "service": "summary-service",
    "request_id": "req-123"
  }
}
```

## Timeouts and Resilience

The gateway must call downstream services with explicit timeouts.

### Initial Recommendation

- `intent-service`: `10s`
- `summary-service`: `45s`
- `reply-service`: `45s`

Exact values can be defined later in configuration.

### Initial Failure Policy

- if `intent-service` fails: return orchestration error
- if `summary-service` fails: return orchestration error
- if `reply-service` fails: return orchestration error

Later, degraded modes may be added, but the first version should prioritize correctness and explicit failure over silent degradation.

## Observability

Observability is mandatory in a microservices architecture.

### Logs Per Service

Each service must emit its own logs.

Recommended minimum fields:

- timestamp
- service name
- log level
- request ID
- route
- event name
- message
- duration when relevant

### Request IDs

The `gateway-service` must generate a `request_id` if none is provided by the client.

That same `request_id` must be propagated to all downstream services through request headers.

### Chosen Strategy

- request IDs travel in HTTP headers
- they do not travel in cookies
- they do not travel in the URL
- they do not need to be part of the business payload

Recommended header:

- `X-Request-ID`

The `request_id` may also be echoed back in the response body or response headers for easier debugging, but the canonical transport mechanism is the request header.

### Validated Response Behavior

The `gateway-service` must also return the same `X-Request-ID` in the response header.

### Clear Failure Messages

Logs must be explicit enough to answer:

- which service failed
- on which request
- at which stage
- with which error category

Examples:

- `intent prediction failed: no customer message found`
- `summary-service timeout after downstream call`
- `reply critic applied fallback due to missing_next_step`

## Configuration Strategy

The system must avoid hardcoding:

- ports
- service URLs
- model directories
- runtime flags
- timeout values

### Recommended Configuration Mechanism

- environment variables as the primary source
- optional `.env` support later if needed

### Expected Variables

- `GATEWAY_HOST`
- `GATEWAY_PORT`
- `INTENT_SERVICE_URL`
- `SUMMARY_SERVICE_URL`
- `REPLY_SERVICE_URL`
- `INTENT_MODEL_PATH`
- `SUMMARY_MODEL_PATH`
- `REPLY_MODEL_PATH`
- `INTENT_SERVICE_TIMEOUT_SECONDS`
- `SUMMARY_SERVICE_TIMEOUT_SECONDS`
- `REPLY_SERVICE_TIMEOUT_SECONDS`
- `LOG_LEVEL`
- `ENABLE_FEEDBACK_PERSISTENCE`

### Rule

No service URL should be hardcoded in orchestration code.

## Shared Code and Contract Management

To avoid duplication and contract drift, the project should define a shared layer for:

- request models
- response models
- shared error shapes
- request ID utilities
- common logging helpers
- common config loading helpers

This shared layer can later become a reusable internal package if needed.

### Chosen Direction

A `shared/` package will be created from the beginning.

It will host:

- shared schemas
- configuration helpers
- shared error schemas
- request ID helpers
- shared logging helpers
- common utility code needed by multiple services

### Initial Shared Structure

```text
shared/
  config/
  schemas/
  logging/
  utils/
```

## Data and Feedback Ownership

The architecture must define where failure memory is written.

### Initial Decision

- `summary-service` owns summary critic feedback persistence
- `reply-service` owns reply critic feedback persistence
- `intent-service` owns intent critic feedback persistence if an intent critic is added in service mode
- `gateway-service` does not write critic-specific feedback on behalf of downstream services

This keeps domain ownership consistent.

### Default Persistence Policy

Feedback persistence is disabled by default in service mode.

This means:

- `persist_feedback = false` by default
- feedback logging can be enabled explicitly for experimentation or controlled batch runs

## Testing Strategy

Testing must be layered.

### 1. Unit Tests

For:

- schema validation
- request-to-payload transformations
- critics
- fallback logic
- config loaders

### 2. Service Tests

For each microservice:

- health endpoint
- valid request
- invalid request
- missing artifact case

### 3. Contract Tests

To ensure:

- `gateway-service` sends the right payload shape
- downstream services return the expected contract
- no contract drift appears between services

### 4. End-to-End Tests

For the full path:

- gateway -> intent -> summary -> reply

## Security and Input Validation

Even in an internal or demonstration setting, basic API hygiene is required.

### Minimum Rules

- validate all request bodies with schemas
- reject empty message lists
- reject malformed speaker values if needed
- avoid exposing internal paths or stack traces to clients
- configure CORS explicitly when a UI is added

## Deployment and Runtime Assumptions

This specification is compatible with later:

- Docker
- Docker Compose
- CI/CD

Each service should be designed so that it can later be:

- containerized independently
- configured independently
- started independently
- health-checked independently

## Recommended Repository Evolution

The repository can later evolve toward a structure such as:

```text
services/
  gateway-service/
  intent-service/
  summary-service/
  reply-service/
shared/
  schemas/
  config/
  logging/
  utils/
```

This is not mandatory immediately, but it is the natural direction if the microservices architecture is implemented fully.

This repository evolution is validated as the target direction.

## Non-Goals for the First Implementation

The first microservices version should not try to include everything at once.

Not required in the first pass:

- async event bus
- queue-based orchestration
- distributed tracing platform
- service discovery platform
- Kubernetes
- autoscaling
- authentication between services

These can be mentioned as future improvements, but they should not block the first implementation.

## Validated Decisions

The following decisions are fixed for the first implementation:

- `gateway-service` remains the single orchestrator
- communication is synchronous over HTTP REST
- `intent-service` exposes `POST /api/v1/intent`
- `summary-service` exposes `POST /api/v1/summary`
- `reply-service` exposes `POST /api/v1/reply`
- request IDs travel through HTTP headers
- timeouts start with `10s / 45s / 45s`
- feedback persistence is disabled by default
- a `shared/` package will be created from the start
- the future UI will talk only to `gateway-service`
- if orchestration fails, the full request fails
- all services share a normalized error format
- service names remain:
  - `gateway-service`
  - `intent-service`
  - `summary-service`
  - `reply-service`
- the repository will evolve toward:
  - `services/`
  - `shared/`

### Naming Convention Note

Service names remain:

- `gateway-service`
- `intent-service`
- `summary-service`
- `reply-service`

For Python package compatibility, repository folder names may use underscores:

- `gateway_service`
- `intent_service`
- `summary_service`
- `reply_service`

## Remaining Questions

The main points still open for later implementation detail are:

- the exact shape of the structured JSON log entries
- whether each service will expose one combined health endpoint only, or separate readiness/liveness endpoints later
- whether downstream services should also echo `X-Request-ID` in their own response headers, or only the gateway should do so

## Final Recommendation

The implementation should aim for:

- a small but real distributed system
- four services only
- strict orchestration by the gateway
- explicit contracts
- structured logs
- startup model loading
- strong configuration discipline

This gives the project enough complexity to demonstrate backend and cloud-native engineering skills without making it unmanageable.
