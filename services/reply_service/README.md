# reply-service

Responsibilities:

- receive a conversation, intent, and summary
- generate a reply
- run reply critic
- apply reply fallback
- optionally persist failures

Current status:

- implemented
- exposes:
  - `GET /api/v1/health`
  - `POST /api/v1/reply`
- loads the reply model once at startup
