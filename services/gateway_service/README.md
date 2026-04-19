# gateway-service

Public entrypoint and orchestrator of the SupportCopilot microservices architecture.

Current status:

- implemented
- supports:
  - embedded compatibility mode
  - HTTP orchestration mode toward `intent-service`, `summary-service`, and `reply-service`
- will later call `intent-service`, `summary-service`, and `reply-service` over HTTP
