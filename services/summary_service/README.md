# summary-service

Responsibilities:

- receive a conversation
- generate a summary
- run summary critic
- apply summary fallback
- optionally persist failures

Current status:

- implemented
- exposes:
  - `GET /api/v1/health`
  - `POST /api/v1/summary`
- loads the summary model once at startup
