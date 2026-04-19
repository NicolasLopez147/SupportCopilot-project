# intent-service

Responsibilities:

- receive a conversation
- extract intent text
- run intent classification
- return predicted intent, confidence, and top classes
- optionally persist critic failures

Current status:

- implemented
- exposes:
  - `GET /api/v1/health`
  - `POST /api/v1/intent`
- loads the intent model once at startup
